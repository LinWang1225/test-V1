#include <stdexcept>
#include <string>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cupti.h>

static void check(CUptiResult result) {
    if (result != CUPTI_SUCCESS) {
        const char* message = nullptr;
        cuptiGetResultString(result, &message);
        throw std::runtime_error(message ? message : "CUPTI failure");
    }
}
#define KV_CUPTI(call) do { try { check(call); } catch (const std::exception& e) { throw std::runtime_error(std::string(#call) + ": " + e.what()); } } while(0)
#include "nvidia_range_profiling.h"

struct Counter {
    CuptiProfilerHost host;
    std::unique_ptr<RangeProfilerTarget> target;
    std::vector<const char*> metrics{"dram__bytes_read.sum", "dram__bytes_write.sum"};
    std::vector<uint8_t> config, image;
    CUcontext context = nullptr;
    bool active = false;

    Counter() {
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context)
            throw std::runtime_error("Initialize the PyTorch CUDA context first");
        CUdevice device;
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS) throw std::runtime_error("cuCtxGetDevice failed");
        RangeProfilerConfig options;
        options.maxNumOfRanges = 1;
        options.minNestingLevel = 1;
        options.numOfNestingLevel = 1;
        target.reset(new RangeProfilerTarget(context, options));
        std::string chip;
        KV_CUPTI(RangeProfilerTarget::GetChipName(device, chip));
        std::vector<uint8_t> availability;
        KV_CUPTI(RangeProfilerTarget::GetCounterAvailabilityImage(context, availability));
        host.SetUp(chip, availability);
        KV_CUPTI(host.CreateConfigImage(metrics, config));
        CUpti_Profiler_Host_GetNumOfPasses_Params passes{CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
        passes.pConfigImage = config.data();
        passes.configImageSize = config.size();
        KV_CUPTI(cuptiProfilerHostGetNumOfPasses(&passes));
        if (passes.numOfPasses != 1) throw std::runtime_error("Counters require multiple passes; generation replay is forbidden");
        KV_CUPTI(target->EnableRangeProfiler());
    }
    void start() {
        if (active) throw std::runtime_error("Nested physical counter window");
        host.ClearResults();
        KV_CUPTI(target->CreateCounterDataImage(metrics, image));
        KV_CUPTI(target->SetConfig(CUPTI_UserRange, CUPTI_UserReplay, config, image));
        KV_CUPTI(target->StartRangeProfiler());
        KV_CUPTI(target->PushRange("live_work"));
        active = true;
    }
    void stop(uint64_t* values) {
        if (!active) throw std::runtime_error("No active counter window");
        if (cuCtxSynchronize() != CUDA_SUCCESS) throw std::runtime_error("CUDA synchronization failed");
        KV_CUPTI(target->PopRange());
        KV_CUPTI(target->StopRangeProfiler());
        if (!target->IsAllPassSubmitted()) throw std::runtime_error("Counter pass incomplete; no replay allowed");
        KV_CUPTI(target->DecodeCounterData());
        size_t ranges = 0;
        KV_CUPTI(host.GetNumOfRanges(image, ranges));
        if (ranges != 1) throw std::runtime_error("Missing or extra hardware counter range");
        KV_CUPTI(host.EvaluateCounterData(0, metrics, image));
        for (size_t i = 0; i < metrics.size(); ++i) {
            double value = host.Results().at(0).metricValues.at(metrics[i]);
            if (!std::isfinite(value) || value < 0 || std::floor(value) != value)
                throw std::runtime_error("Invalid DRAM byte counter");
            values[i] = static_cast<uint64_t>(value);
        }
        active = false;
    }
    void close() {
        if (active) throw std::runtime_error("Cannot close an active counter window");
        KV_CUPTI(target->DisableRangeProfiler());
        host.TearDown();
    }
};

static std::unique_ptr<Counter> counter;
static std::string error;
template<class F> int guarded(F operation) {
    try { operation(); return 0; }
    catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" {
const char* kv_error() { return error.c_str(); }
int kv_create() { return guarded([] { if(counter) throw std::runtime_error("Counter already initialized"); counter.reset(new Counter()); }); }
int kv_start() { return guarded([] { if(!counter) throw std::runtime_error("Counter not initialized"); counter->start(); }); }
int kv_stop(uint64_t* output) { return guarded([&] { counter->stop(output); }); }
int kv_close() { return guarded([] { counter->close(); counter.reset(); }); }
size_t kv_image_size() { return counter->image.size(); }
const uint8_t* kv_image() { return counter->image.data(); }
}
