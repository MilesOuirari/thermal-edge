// SPDX-License-Identifier: GPL-2.0
/*
 * Thermal sensor I2C kernel driver.
 *
 * Provides a sysfs interface for temperature readings.
 * Uses threaded IRQ for data-ready notification, avoiding polling.
 * Feeds the edge-thermal-ai inference pipeline via sysfs.
 *
 * Sysfs attributes:
 *   temperature_raw   — raw ADC value
 *   temperature_mc    — temperature in milli-Celsius
 *   alert_threshold   — over-temperature alert threshold (mC)
 */

#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/interrupt.h>
#include <linux/gpio/consumer.h>
#include <linux/mutex.h>
#include <linux/sysfs.h>
#include <linux/of.h>
#include <linux/delay.h>

#define DRIVER_NAME "thermal_i2c"

/* Register map */
#define REG_TEMP_MSB       0x00
#define REG_TEMP_LSB       0x01
#define REG_CONFIG         0x01
#define REG_ALERT_HIGH_MSB 0x02
#define REG_ALERT_HIGH_LSB 0x03

#define CONFIG_CONTINUOUS  0x00
#define CONFIG_RESOLUTION  0x60  /* 12-bit */
#define SCALE_MC           625   /* 0.0625 °C per LSB → 62.5 m°C per unit */

struct thermal_dev {
    struct i2c_client *client;
    struct mutex       lock;
    s32                raw_temp;     /* latest raw ADC value */
    s32                alert_mc;     /* alert threshold in milli-Celsius */
    struct gpio_desc  *alert_gpio;
    int                irq;
};

/* ─── Temperature conversion ─────────────────────────────────────────── */

static s32 raw_to_mc(s32 raw)
{
    /* 13-bit signed (1 sign + 12 data) from MSB/LSB pair, 0.0625 °C/LSB */
    s32 temp = raw >> 3;  /* right-justify 12 bits */
    if (temp & 0x1000)    /* sign extension for negative */
        temp |= ~0x1FFF;
    return temp * SCALE_MC / 10; /* convert to milli-Celsius */
}

static int read_temp_raw(struct thermal_dev *priv, s32 *out)
{
    s32 msb = i2c_smbus_read_byte_data(priv->client, REG_TEMP_MSB);
    s32 lsb = i2c_smbus_read_byte_data(priv->client, REG_TEMP_LSB);
    if (msb < 0 || lsb < 0) return -EIO;
    *out = (msb << 8) | (lsb & 0xFF);
    return 0;
}

/* ─── Threaded IRQ — alert pin asserted ──────────────────────────────── */

static irqreturn_t thermal_alert_irq(int irq, void *dev_id)
{
    struct thermal_dev *priv = dev_id;
    s32 raw;

    mutex_lock(&priv->lock);
    if (read_temp_raw(priv, &raw) == 0) {
        priv->raw_temp = raw;
        dev_warn(&priv->client->dev,
                 "Thermal alert! Temperature: %d mC\n",
                 raw_to_mc(raw));
    }
    mutex_unlock(&priv->lock);

    return IRQ_HANDLED;
}

/* ─── sysfs ──────────────────────────────────────────────────────────── */

static ssize_t temperature_raw_show(struct device *dev,
                                    struct device_attribute *attr, char *buf)
{
    struct thermal_dev *priv = i2c_get_clientdata(to_i2c_client(dev));
    s32 raw;
    int ret;

    mutex_lock(&priv->lock);
    ret = read_temp_raw(priv, &raw);
    if (ret == 0) priv->raw_temp = raw;
    mutex_unlock(&priv->lock);

    if (ret) return ret;
    return sysfs_emit(buf, "%d\n", raw);
}

static ssize_t temperature_mc_show(struct device *dev,
                                   struct device_attribute *attr, char *buf)
{
    struct thermal_dev *priv = i2c_get_clientdata(to_i2c_client(dev));
    s32 raw;
    int ret;

    mutex_lock(&priv->lock);
    ret = read_temp_raw(priv, &raw);
    if (ret == 0) priv->raw_temp = raw;
    mutex_unlock(&priv->lock);

    if (ret) return ret;
    return sysfs_emit(buf, "%d\n", raw_to_mc(raw));
}

static ssize_t alert_threshold_show(struct device *dev,
                                    struct device_attribute *attr, char *buf)
{
    struct thermal_dev *priv = i2c_get_clientdata(to_i2c_client(dev));
    return sysfs_emit(buf, "%d\n", priv->alert_mc);
}

static ssize_t alert_threshold_store(struct device *dev,
                                     struct device_attribute *attr,
                                     const char *buf, size_t count)
{
    struct thermal_dev *priv = i2c_get_clientdata(to_i2c_client(dev));
    long mc;
    int ret;

    ret = kstrtol(buf, 10, &mc);
    if (ret) return ret;

    mutex_lock(&priv->lock);
    priv->alert_mc = (s32)mc;
    /* Write threshold to hardware (2 registers) */
    s32 raw_threshold = (s32)(mc * 10 / SCALE_MC) << 3;
    i2c_smbus_write_byte_data(priv->client, REG_ALERT_HIGH_MSB,
                               (u8)((raw_threshold >> 8) & 0xFF));
    i2c_smbus_write_byte_data(priv->client, REG_ALERT_HIGH_LSB,
                               (u8)(raw_threshold & 0xFF));
    mutex_unlock(&priv->lock);

    return count;
}

static DEVICE_ATTR_RO(temperature_raw);
static DEVICE_ATTR_RO(temperature_mc);
static DEVICE_ATTR_RW(alert_threshold);

static struct attribute *thermal_attrs[] = {
    &dev_attr_temperature_raw.attr,
    &dev_attr_temperature_mc.attr,
    &dev_attr_alert_threshold.attr,
    NULL,
};
ATTRIBUTE_GROUPS(thermal);

/* ─── Probe / remove ─────────────────────────────────────────────────── */

static int thermal_probe(struct i2c_client *client,
                          const struct i2c_device_id *id)
{
    struct thermal_dev *priv;
    s32 val;
    int ret;

    priv = devm_kzalloc(&client->dev, sizeof(*priv), GFP_KERNEL);
    if (!priv) return -ENOMEM;

    priv->client   = client;
    priv->alert_mc = 85000; /* 85 °C default */
    mutex_init(&priv->lock);
    i2c_set_clientdata(client, priv);

    /* Configure sensor: continuous mode, 12-bit resolution */
    ret = i2c_smbus_write_byte_data(client, REG_CONFIG,
                                     CONFIG_CONTINUOUS | CONFIG_RESOLUTION);
    if (ret < 0) return ret;

    /* Verify we can read temperature */
    ret = read_temp_raw(priv, &val);
    if (ret < 0) {
        dev_err(&client->dev, "Failed initial temperature read\n");
        return ret;
    }

    /* Optional alert IRQ */
    priv->alert_gpio = devm_gpiod_get_optional(&client->dev, "alert", GPIOD_IN);
    if (!IS_ERR_OR_NULL(priv->alert_gpio)) {
        priv->irq = gpiod_to_irq(priv->alert_gpio);
        ret = devm_request_threaded_irq(&client->dev, priv->irq,
                                         NULL, thermal_alert_irq,
                                         IRQF_TRIGGER_FALLING | IRQF_ONESHOT,
                                         DRIVER_NAME, priv);
        if (ret)
            dev_warn(&client->dev, "Alert IRQ setup failed, disabled\n");
    }

    dev_info(&client->dev, "Thermal sensor ready, T=%d mC\n",
             raw_to_mc(val));
    return 0;
}

static void thermal_remove(struct i2c_client *client)
{
    /* Sensor enters standby when config register is cleared */
    i2c_smbus_write_byte_data(client, REG_CONFIG, 0x01); /* one-shot / standby */
}

static const struct i2c_device_id thermal_id[] = {
    { DRIVER_NAME, 0 },
    {}
};
MODULE_DEVICE_TABLE(i2c, thermal_id);

static const struct of_device_id thermal_of_match[] = {
    { .compatible = "thermal,i2c-sensor" },
    {}
};
MODULE_DEVICE_TABLE(of, thermal_of_match);

static struct i2c_driver thermal_driver = {
    .driver = {
        .name           = DRIVER_NAME,
        .of_match_table = thermal_of_match,
        .dev_groups     = thermal_groups,
    },
    .probe    = thermal_probe,
    .remove   = thermal_remove,
    .id_table = thermal_id,
};
module_i2c_driver(thermal_driver);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Selim Ouirari");
MODULE_DESCRIPTION("Thermal I2C sensor driver with alert IRQ for edge AI pipeline");
