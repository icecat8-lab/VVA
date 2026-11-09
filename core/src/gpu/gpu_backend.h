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

    virtual bool fits(int h, int w) = 0;
    virtual size_t estimateBytes(int h, int w) = 0;
    virtual Status run(const float* in, int h, int w, float* out, const std::atomic<bool>* cancel, std::string* error) = 0;
    virtual std::string describe() const = 0;
    virtual std::string identity() const = 0;
    virtual std::string profile(int h, int w) = 0;
    virtual int variantCount() const = 0;
    virtual void setVariant(int variant) = 0;
    virtual int variantIndex() const = 0;
    virtual int inChannels() const = 0;
    virtual int outChannels() const = 0;
};

std::unique_ptr<Backend> createVulkanBackend(const VulkanHandles& vk, const Model& model, std::string* status);

}
}

#endif
