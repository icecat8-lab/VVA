#ifndef VVA_H
#define VVA_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vva {

namespace gpu {
class Backend;
}

// Where the work runs.
//   AUTO        the GPU (Vulkan) when it passed its start-up self-test and is faster than the CPU, otherwise the CPU
//   CPU_NEON    the multi-threaded CPU executor
//   GPU_VULKAN  the GPU; processing fails when the GPU path is unavailable (see Engine::gpuStatus())
//   NPU_NNAPI   not implemented: treated like AUTO
// A GPU error in the middle of a job never fails the job: the remaining work continues on the CPU.
enum class DeviceType { CPU_NEON, GPU_VULKAN, NPU_NNAPI, AUTO };

// Called after every finished tile with the overall progress in 0..1.
// Return false to cancel the job.
using ProgressCallback = std::function<bool(float fraction)>;

// Result of Engine::classify().
struct Classification {
    int index = -1;                   // most likely class
    float confidence = 0.f;           // its probability (softmax of the logits)
    std::string label;                // meta.classes[index], or "class N" when the model has no names
    std::vector<float> probabilities;  // one per class, sums to 1
    std::vector<float> logits;         // raw network outputs
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Loads a .vva (version 2) model and checks that the executor can run every
    // operator in it. On failure returns false and lastError() explains why; a
    // previously loaded model stays active.
    bool loadModel(const std::string& modelPath);
    std::string lastError() const;
    bool modelLoaded() const;
    std::string modelName() const;
    int scale() const;  // output size / input size; 0 when no model is loaded or the model is a classifier
    bool isClassifier() const;
    std::vector<std::string> classLabels() const;  // empty when the model has no class names

    // Upscales an RGBA8 image (straight, non-premultiplied alpha, tightly packed
    // rows). `outRGBA` must hold (w*scale) * (h*scale) * 4 bytes. Large images
    // are processed in tiles so memory stays within the configured budget.
    // The colour channels go through the network; alpha, when it is not fully
    // opaque, is upscaled bilinearly. Blocks until done; call it from a worker
    // thread. Returns false on error or cancellation (see lastError()).
    bool processImageTile(const uint8_t* inRGBA, int w, int h, uint8_t* outRGBA, DeviceType device);

    // Classifier models (meta.task = "classify"): resizes the image to the model's input size
    // (PIL-style antialiased bilinear), normalises it with the model's mean/std and runs the network.
    // RGBA8 with tightly packed rows; alpha is ignored. Returns false on error (see lastError()).
    bool classify(const uint8_t* inRGBA, int w, int h, Classification& result);

    // Tuning. Set these before processing; they are not synchronised with a running job.
    void setTileSize(int pixels);        // 0 = choose automatically from the memory budget
    void setTilePad(int pixels);         // overlap read around each tile (default 16)
    void setThreads(int threads);        // 0 = automatic (up to 8)
    void setMemoryBudgetMB(int mb);      // working memory for one tile (default 320)
    void setProgressCallback(ProgressCallback cb);

    // May be called from any thread; stops the running job as soon as possible.
    void cancel();

    // Device that executed the last job (CPU_NEON when any part of it had to fall back to the CPU).
    DeviceType lastDevice() const;

    // Why the GPU path is or is not active for the loaded model, in one line.
    std::string gpuStatus() const;
    bool gpuActive() const;

    // Internal: lets tests inject a backend. Takes ownership. `checkSpeed` = false skips the speed comparison.
    void adoptGpuBackendForTesting(gpu::Backend* backend, bool checkSpeed);

    // Vulkan diagnostics (probed on first use; never needed for processing).
    bool hasVulkan() const;
    std::string vulkanDeviceName() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace vva

#endif
