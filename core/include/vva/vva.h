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

enum class DeviceType { CPU_NEON, GPU_VULKAN, NPU_NNAPI, AUTO };

using ProgressCallback = std::function<bool(float fraction)>;

struct Classification {
    int index = -1;
    float confidence = 0.f;
    std::string label;
    std::vector<float> probabilities;
    std::vector<float> logits;
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool loadModel(const std::string& modelPath);
    std::string lastError() const;
    bool modelLoaded() const;
    std::string modelName() const;
    int scale() const;
    bool isClassifier() const;
    std::vector<std::string> classLabels() const;

    bool processImageTile(const uint8_t* inRGBA, int w, int h, uint8_t* outRGBA, DeviceType device);

    bool classify(const uint8_t* inRGBA, int w, int h, Classification& result);

    void setTileSize(int pixels);
    void setTilePad(int pixels);
    void setThreads(int threads);
    void setMemoryBudgetMB(int mb);
    void setProgressCallback(ProgressCallback cb);
    void setCacheDir(const std::string& dir);

    void cancel();

    DeviceType lastDevice() const;

    std::string gpuStatus() const;
    bool gpuActive() const;
    std::string gpuProfile(int h, int w);

    void adoptGpuBackendForTesting(gpu::Backend* backend, bool checkSpeed);

    bool hasVulkan() const;
    std::string vulkanDeviceName() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

}

#endif
