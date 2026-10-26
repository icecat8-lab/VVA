#ifndef VVA_H
#define VVA_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace vva {

// Where the work runs. Today every request is executed by the multi-threaded
// CPU executor: AUTO and CPU_NEON are the supported values, GPU_VULKAN and
// NPU_NNAPI are accepted but fall back to the CPU (see Engine::lastDevice()).
enum class DeviceType { CPU_NEON, GPU_VULKAN, NPU_NNAPI, AUTO };

// Called after every finished tile with the overall progress in 0..1.
// Return false to cancel the job.
using ProgressCallback = std::function<bool(float fraction)>;

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
    int scale() const;  // output size / input size, 0 when no model is loaded

    // Upscales an RGBA8 image (straight, non-premultiplied alpha, tightly packed
    // rows). `outRGBA` must hold (w*scale) * (h*scale) * 4 bytes. Large images
    // are processed in tiles so memory stays within the configured budget.
    // The colour channels go through the network; alpha, when it is not fully
    // opaque, is upscaled bilinearly. Blocks until done; call it from a worker
    // thread. Returns false on error or cancellation (see lastError()).
    bool processImageTile(const uint8_t* inRGBA, int w, int h, uint8_t* outRGBA, DeviceType device);

    // Tuning. Set these before processing; they are not synchronised with a running job.
    void setTileSize(int pixels);        // 0 = choose automatically from the memory budget
    void setTilePad(int pixels);         // overlap read around each tile (default 16)
    void setThreads(int threads);        // 0 = automatic (up to 8)
    void setMemoryBudgetMB(int mb);      // working memory for one tile (default 320)
    void setProgressCallback(ProgressCallback cb);

    // May be called from any thread; stops the running job as soon as possible.
    void cancel();

    // Device that executed the last job.
    DeviceType lastDevice() const;

    // Vulkan diagnostics (probed on first use; never needed for processing).
    bool hasVulkan() const;
    std::string vulkanDeviceName() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace vva

#endif
