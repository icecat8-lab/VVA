// Interface between the engine and the Vulkan runtime (kept free of Vulkan headers).
#ifndef VVA_GPU_BACKEND_H
#define VVA_GPU_BACKEND_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "../model.h"

namespace vva {
namespace gpu {

struct VulkanHandles {
    void* physicalDevice = nullptr;
    void* device = nullptr;
    void* queue = nullptr;
    uint32_t queueFamily = 0;
};

class Backend {
public:
    enum class Status { Ok, Cancelled, DeviceLost, Error };
    virtual ~Backend() {}

    // Whether an h x w image can run within this GPU's limits (also warms up the program cache).
    virtual bool fits(int h, int w) = 0;
    // Activation memory needed for an h x w image, in bytes (0 when the size cannot run).
    virtual size_t estimateBytes(int h, int w) = 0;
    // in: planar float [inChannels][h][w]; out: planar float [outChannels][h*scale][w*scale].
    virtual Status run(const float* in, int h, int w, float* out, const std::atomic<bool>* cancel, std::string* error) = 0;
    virtual int inChannels() const = 0;
    virtual int outChannels() const = 0;
};

// Returns nullptr and fills `status` when the model or the device cannot use the GPU path, or when the library
// was built without GPU kernels.
std::unique_ptr<Backend> createVulkanBackend(const VulkanHandles& vk, const Model& model, std::string* status);

}  // namespace gpu
}  // namespace vva

#endif
