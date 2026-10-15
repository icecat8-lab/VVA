#ifndef VVA_H
#define VVA_H

#include <cstdint>
#include <memory>
#include <string>

namespace vva {

enum class DeviceType { CPU_NEON, GPU_VULKAN, NPU_NNAPI, AUTO };

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool loadModel(const std::string& modelPath);

    // Not implemented yet: returns false until the graph executor lands.
    bool processImageTile(const uint8_t* inRGBA, int w, int h, uint8_t* outRGBA, DeviceType device);

    // Vulkan diagnostics (usable from the CLI and from apps).
    bool hasVulkan() const;
    std::string vulkanDeviceName() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace vva

#endif
