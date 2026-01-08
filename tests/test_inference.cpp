#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../inference/tflite_engine.hpp"

TEST_CASE("Engine loads without model file (stub mode)", "[inference]")
{
    inference::TfliteEngine::Config cfg;
    cfg.model_path = "/tmp/nonexistent.tflite";
    inference::TfliteEngine engine(cfg);
    /* In stub mode (no TFLite installed), load always returns true */
    bool ok = engine.load();
    REQUIRE(ok);
    REQUIRE(engine.is_loaded());
}

TEST_CASE("Prediction on normal temperature", "[inference]")
{
    inference::TfliteEngine::Config cfg;
    cfg.model_path      = "/tmp/nonexistent.tflite";
    cfg.alert_threshold = 0.7f;
    inference::TfliteEngine engine(cfg);
    engine.load();

    inference::ThermalSample sample{};
    for (auto& t : sample.temperatures) t = 45.0f; /* 45°C — normal */

    auto pred = engine.predict(sample);
    REQUIRE(pred.anomaly_score >= 0.0f);
    REQUIRE(pred.anomaly_score <= 1.0f);
    REQUIRE_FALSE(pred.alert);  /* 45°C should not trigger alert */
    REQUIRE(pred.inference_time.count() >= 0);
}

TEST_CASE("Prediction on critical temperature", "[inference]")
{
    inference::TfliteEngine::Config cfg;
    cfg.model_path      = "/tmp/nonexistent.tflite";
    cfg.alert_threshold = 0.7f;
    inference::TfliteEngine engine(cfg);
    engine.load();

    inference::ThermalSample sample{};
    for (auto& t : sample.temperatures) t = 100.0f; /* 100°C — critical */

    auto pred = engine.predict(sample);
    REQUIRE(pred.alert);  /* must trigger alert at 100°C */
}

TEST_CASE("Benchmark returns valid stats", "[inference]")
{
    inference::TfliteEngine::Config cfg;
    cfg.model_path = "/tmp/nonexistent.tflite";
    inference::TfliteEngine engine(cfg);
    engine.load();

    auto result = engine.benchmark(20);
    REQUIRE(result.mean_us >= 0.0f);
    REQUIRE(result.max_us >= result.mean_us);
}
