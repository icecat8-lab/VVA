#include "vva/vva.h"

#include <utility>

#include "model.h"
#include "backend/vulkan_context.cpp"

namespace vva {

class Engine::Impl {
public:
    std::unique_ptr<Model> model;
    std::string lastError;
    VulkanContext vkContext;

    Impl() { vkContext.init(); }

    bool load(const std::string& path) {
        std::string error;
        std::unique_ptr<Model> loaded = Model::load(path, &error);
        if (!loaded) {
            lastError = error;
            return false;
        }
        model = std::move(loaded);
        lastError.clear();
        return true;
    }
};

Engine::Engine() : pimpl(new Impl()) {}
Engine::~Engine() = default;

bool Engine::loadModel(const std::string& modelPath) {
    return pimpl->load(modelPath);
}

std::string Engine::lastError() const {
    return pimpl->lastError;
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
