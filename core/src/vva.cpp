#include "vva/vva.h"

#include "model_loader.cpp"
#include "backend/vulkan_context.cpp"

namespace vva {

class Engine::Impl {
public:
    ModelLoader loader;
    VulkanContext vkContext;
    bool modelLoaded = false;

    Impl() { vkContext.init(); }

    bool load(const std::string& path) {
        modelLoaded = loader.loadVVA(path);
        return modelLoaded;
    }
};

Engine::Engine() : pimpl(new Impl()) {}
Engine::~Engine() = default;

bool Engine::loadModel(const std::string& modelPath) {
    return pimpl->load(modelPath);
}

bool Engine::processImageTile(const uint8_t*, int, int, uint8_t*, DeviceType) {
    // The graph executor is not implemented yet. Return false instead of
    // running half-built pipelines.
    return false;
}

bool Engine::hasVulkan() const {
    return pimpl->vkContext.ready();
}

std::string Engine::vulkanDeviceName() const {
    return pimpl->vkContext.deviceName;
}

}  // namespace vva
