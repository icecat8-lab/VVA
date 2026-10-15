#include "vva/vva.h"
#include "model_loader.cpp"
#include "backend/vulkan_context.cpp"
#include "backend/vulkan_dispatcher.cpp"
#include "backend/vulkan_zero_copy.cpp"
#include "backend/nnapi_backend.cpp"
#include "scheduler.cpp"
#include "graph.h"

namespace vva {

class Engine::Impl {
public:
    ModelLoader loader;
    VulkanContext vkContext;
    VulkanZeroCopyManager vkZeroCopy;
    VulkanDispatcher vkDispatcher;
    NNAPIBackend nnapi;
    Scheduler scheduler;
    Graph graph;

    Impl() {
        vkContext.init();
        nnapi.initModel();
    }

    bool load(const std::string& path) {
        if (!loader.loadVVA(path)) return false;
        graph.buildExecutionOrder();
        return true;
    }

    bool process(const uint8_t* inRGBA, int w, int h, uint8_t* outRGBA, DeviceType device) {
        scheduler.dispatchTile(inRGBA, outRGBA, w, h);
        
        if (device == DeviceType::GPU_VULKAN || device == DeviceType::AUTO) {
            vkDispatcher.submitQueue(vkContext.computeQueue);
        }
        
        if (device == DeviceType::NPU_NNAPI || device == DeviceType::AUTO) {
            nnapi.compileAndExecute();
        }

        return true;
    }
};

Engine::Engine() : pimpl(new Impl()) {}
Engine::~Engine() {}

bool Engine::loadModel(const std::string& modelPath) {
    return pimpl->load(modelPath);
}

bool Engine::processImageTile(const uint8_t* inRGBA, int w, int h, uint8_t* outRGBA, DeviceType device) {
    return pimpl->process(inRGBA, w, h, outRGBA, device);
}

}
