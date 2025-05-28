#include "tflite_engine.hpp"

#ifdef HAVE_TFLITE
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

#include <cstring>
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace inference {

struct TfliteEngine::Impl {
    Config cfg;
    bool   loaded = false;

#ifdef HAVE_TFLITE
    std::unique_ptr<tflite::FlatBufferModel>    model;
    std::unique_ptr<tflite::Interpreter>        interpreter;
    TfLiteDelegate                             *xnnpack_delegate = nullptr;
#endif
};

TfliteEngine::TfliteEngine(const Config& cfg)
    : impl_(std::make_unique<Impl>())
{
    impl_->cfg = cfg;
}

TfliteEngine::~TfliteEngine()
{
#ifdef HAVE_TFLITE
    if (impl_->xnnpack_delegate)
        TfLiteXNNPackDelegateDelete(impl_->xnnpack_delegate);
#endif
}

bool TfliteEngine::load()
{
#ifdef HAVE_TFLITE
    impl_->model = tflite::FlatBufferModel::BuildFromFile(
                       impl_->cfg.model_path.c_str());
    if (!impl_->model) return false;

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*impl_->model, resolver);
    builder(&impl_->interpreter);
    if (!impl_->interpreter) return false;

    impl_->interpreter->SetNumThreads(impl_->cfg.num_threads);

    if (impl_->cfg.use_xnnpack) {
        TfLiteXNNPackDelegateOptions opts = TfLiteXNNPackDelegateOptionsDefault();
        opts.num_threads = impl_->cfg.num_threads;
        impl_->xnnpack_delegate = TfLiteXNNPackDelegateCreate(&opts);
        impl_->interpreter->ModifyGraphWithDelegate(impl_->xnnpack_delegate);
    }

    impl_->interpreter->AllocateTensors();
    impl_->loaded = true;
    return true;
#else
    /* Stub: simulate loaded state for CI without TFLite installed */
    impl_->loaded = true;
    return true;
#endif
}

bool TfliteEngine::is_loaded() const { return impl_->loaded; }

ThermalPrediction TfliteEngine::predict(const ThermalSample& sample)
{
    auto t0 = std::chrono::steady_clock::now();
    ThermalPrediction result{};

#ifdef HAVE_TFLITE
    if (!impl_->loaded)
        throw std::runtime_error("Model not loaded");

    float *input = impl_->interpreter->typed_input_tensor<float>(0);
    std::memcpy(input, sample.temperatures, sizeof(sample.temperatures));

    impl_->interpreter->Invoke();

    float *output = impl_->interpreter->typed_output_tensor<float>(0);
    result.anomaly_score      = output[0];
    result.predicted_max_temp = output[1];
    result.alert              = result.anomaly_score > impl_->cfg.alert_threshold;
#else
    /* Stub: simple threshold-based heuristic when TFLite is not available */
    float max_t = *std::max_element(sample.temperatures,
                                    sample.temperatures + 16);
    result.predicted_max_temp = max_t;
    result.anomaly_score      = std::clamp((max_t - 60.0f) / 40.0f, 0.0f, 1.0f);
    result.alert              = max_t > 95.0f;
#endif

    auto t1 = std::chrono::steady_clock::now();
    result.inference_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

    return result;
}

TfliteEngine::BenchResult TfliteEngine::benchmark(int n)
{
    ThermalSample sample{};
    for (auto& t : sample.temperatures) t = 45.0f;

    std::vector<float> times;
    times.reserve(n);

    for (int i = 0; i < n; i++) {
        auto pred = predict(sample);
        times.push_back((float)pred.inference_time.count());
    }

    float mean = std::accumulate(times.begin(), times.end(), 0.0f) / n;
    float max  = *std::max_element(times.begin(), times.end());
    return { mean, max };
}

} // namespace inference
