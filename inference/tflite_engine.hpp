#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace inference {

struct ThermalSample {
    float temperatures[16];  /* 4x4 sensor array or time-series window */
    float timestamp_s;
};

struct ThermalPrediction {
    float anomaly_score;         /* 0.0 = normal, 1.0 = critical */
    float predicted_max_temp;    /* predicted peak temperature (°C) */
    bool  alert;                 /* true if anomaly_score > threshold */
    std::chrono::microseconds inference_time;
};

/**
 * TFLite INT8 inference engine.
 *
 * Wraps TensorFlow Lite C API with XNNPACK delegate for NEON acceleration
 * on ARM Cortex-A. Target: < 8 ms inference on Cortex-A8 @ 1 GHz.
 */
class TfliteEngine {
public:
    struct Config {
        std::string model_path;
        float       alert_threshold = 0.7f;
        int         num_threads     = 2;
        bool        use_xnnpack     = true;
    };

    explicit TfliteEngine(const Config& cfg);
    ~TfliteEngine();

    TfliteEngine(const TfliteEngine&)            = delete;
    TfliteEngine& operator=(const TfliteEngine&) = delete;

    bool load();
    ThermalPrediction predict(const ThermalSample& sample);
    bool is_loaded() const;

    /* Benchmark: run N inferences and return mean/max latency in µs */
    struct BenchResult { float mean_us; float max_us; };
    BenchResult benchmark(int n_iterations = 100);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace inference
