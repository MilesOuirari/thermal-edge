#include "../inference/tflite_engine.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>

static std::atomic<bool> g_running{true};
static void sig_handler(int) { g_running.store(false); }

/* Read temperature from sysfs (kernel driver) */
static float read_temperature_sysfs(const std::string& sysfs_path)
{
    std::ifstream f(sysfs_path + "/temperature_mc");
    int mc = 0;
    f >> mc;
    return (float)mc / 1000.0f; /* milli-Celsius → Celsius */
}

/* Write PWM duty cycle 0–100 via sysfs */
static void set_pwm_duty(const std::string& pwm_path, int duty_pct)
{
    std::ofstream f(pwm_path + "/duty_cycle");
    if (f.is_open())
        f << std::clamp(duty_pct, 0, 100);
}

/* Compute fan speed from anomaly score: 30%–100% */
static int compute_fan_duty(float anomaly_score)
{
    return (int)(30.0f + anomaly_score * 70.0f);
}

int main(int argc, char *argv[])
{
    const char *model_path = argc > 1 ? argv[1] : "inference/model/thermal.tflite";
    const char *sysfs_temp = argc > 2 ? argv[2]
        : "/sys/bus/i2c/devices/1-0048";
    const char *sysfs_pwm  = argc > 3 ? argv[3]
        : "/sys/bus/platform/devices/pwm-controller";

    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    inference::TfliteEngine::Config cfg;
    cfg.model_path       = model_path;
    cfg.alert_threshold  = 0.7f;
    cfg.num_threads      = 2;
    cfg.use_xnnpack      = true;

    inference::TfliteEngine engine(cfg);
    if (!engine.load()) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }

    std::cout << "Thermal Guardian running.\n"
              << "  Model  : " << model_path << "\n"
              << "  Sensor : " << sysfs_temp << "\n"
              << "  PWM    : " << sysfs_pwm  << "\n\n";

    long cycle_count = 0;
    long alert_count = 0;

    auto loop_start = std::chrono::steady_clock::now();

    while (g_running.load()) {
        auto t0 = std::chrono::steady_clock::now();

        /* 1. Read sensor */
        inference::ThermalSample sample{};
        float t = read_temperature_sysfs(sysfs_temp);
        for (auto& v : sample.temperatures) v = t;

        /* 2. Inference */
        auto pred = engine.predict(sample);

        /* 3. Actuate */
        int duty = compute_fan_duty(pred.anomaly_score);
        set_pwm_duty(sysfs_pwm, duty);

        /* 4. Log */
        if (pred.alert) {
            alert_count++;
            std::cout << "[ALERT] T=" << pred.predicted_max_temp
                      << "°C  score=" << pred.anomaly_score
                      << "  fan=" << duty << "%"
                      << "  inference=" << pred.inference_time.count() << "µs\n";
        }

        cycle_count++;

        auto t1 = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

        /* Target: 1 ms loop */
        auto sleep_us = std::chrono::microseconds(1000) - elapsed;
        if (sleep_us.count() > 0)
            std::this_thread::sleep_for(sleep_us);
    }

    auto total_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - loop_start).count();

    std::cout << "\nStopped after " << total_s << "s"
              << "  cycles=" << cycle_count
              << "  alerts=" << alert_count << "\n";
    return 0;
}
