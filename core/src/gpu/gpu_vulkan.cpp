#include "gpu_backend.h"

#if !defined(VVA_NO_VULKAN) && defined(__has_include)
#if __has_include("spirv_data.h")
#define VVA_GPU_AVAILABLE 1
#endif
#endif

#ifndef VVA_GPU_AVAILABLE

namespace vva {
namespace gpu {

std::unique_ptr<Backend> createVulkanBackend(const VulkanHandles&, const Model&, std::string* status) {
    if (status) *status = "this build has no GPU kernels";
    return nullptr;
}

}  // namespace gpu
}  // namespace vva

#else  // VVA_GPU_AVAILABLE

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "gpu_plan.h"
#include "spirv_data.h"

namespace vva {
namespace gpu {
namespace {

constexpr uint64_t kFenceTimeoutNs = 20ull * 1000ull * 1000ull * 1000ull;
constexpr size_t kMaxArenaBytes = 1ull << 30;

struct Mem {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t size = 0;
};

struct Compiled {
    Program program;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> segments;
};

class VulkanBackend : public Backend {
public:
    VulkanBackend(const VulkanHandles& vk, std::unique_ptr<Plan> plan)
        : pd_(static_cast<VkPhysicalDevice>(vk.physicalDevice)),
          dev_(static_cast<VkDevice>(vk.device)),
          queue_(static_cast<VkQueue>(vk.queue)),
          family_(vk.queueFamily),
          plan_(std::move(plan)) {}

    ~VulkanBackend() override { teardown(); }

    bool init(std::string* status) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd_, &props);
        const VkPhysicalDeviceLimits& l = props.limits;
        if (l.maxPushConstantsSize < sizeof(Push) || l.minStorageBufferOffsetAlignment > 256 ||
            l.maxComputeWorkGroupInvocations < 64 || l.maxComputeWorkGroupCount[0] < 32768 ||
            l.maxComputeWorkGroupCount[1] < 4096 || l.maxComputeWorkGroupCount[2] < 1024) {
            *status = "the GPU's compute limits are too small";
            return false;
        }
        maxRange_ = l.maxStorageBufferRange;

        if (!createShader(kSpvConv3x3, sizeof(kSpvConv3x3), shaders_[0]) || !createShader(kSpvEltwise, sizeof(kSpvEltwise), shaders_[1]) ||
            !createShader(kSpvResize, sizeof(kSpvResize), shaders_[2])) {
            *status = "the driver rejected a shader";
            return false;
        }
        VkDescriptorSetLayoutBinding binds[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dl{};
        dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dl.bindingCount = 4;
        dl.pBindings = binds;
        if (vkCreateDescriptorSetLayout(dev_, &dl, nullptr, &setLayout_) != VK_SUCCESS) {
            *status = "cannot create the descriptor layout";
            return false;
        }
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(Push);
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &setLayout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(dev_, &pl, nullptr, &layout_) != VK_SUCCESS) {
            *status = "cannot create the pipeline layout";
            return false;
        }
        for (int k = 0; k < 3; ++k) {
            VkComputePipelineCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = shaders_[k];
            ci.stage.pName = "main";
            ci.layout = layout_;
            if (vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &ci, nullptr, &pipelines_[k]) != VK_SUCCESS) {
                *status = "the driver could not build a compute pipeline";
                return false;
            }
        }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(dev_, &fi, nullptr, &fence_) != VK_SUCCESS) {
            *status = "cannot create a fence";
            return false;
        }
        const std::vector<float>& w = plan_->weights();
        if (!createMem(w.size() * sizeof(float), weights_)) {
            *status = "cannot allocate the weights buffer";
            return false;
        }
        std::memcpy(weights_.mapped, w.data(), w.size() * sizeof(float));
        return true;
    }

    int inChannels() const override { return plan_->inChannels(); }
    int outChannels() const override { return plan_->outChannels(); }

    bool fits(int h, int w) override { return estimateBytes(h, w) > 0; }

    size_t estimateBytes(int h, int w) override {
        Program p;
        std::string err;
        if (!plan_->build(h, w, p, &err)) return 0;
        if (p.maxBindBytes > maxRange_ || p.arenaBytes > kMaxArenaBytes) return 0;
        return p.arenaBytes;
    }

    Status run(const float* in, int h, int w, float* out, const std::atomic<bool>* cancel, std::string* error) override {
        if (lost_) return fail(error, "the GPU was lost earlier", Status::DeviceLost);
        Compiled* c = compiled(h, w, error);
        if (!c) return Status::Error;
        const Program& p = c->program;

        if (lastH_ != h || lastW_ != w) {
            std::memset(arena_.mapped, 0, arena_.size);  // borders rely on never-written zeros
            lastH_ = h;
            lastW_ = w;
        }
        const size_t inRow = static_cast<size_t>(w) + 2;
        float* inBase = reinterpret_cast<float*>(static_cast<char*>(arena_.mapped) + p.inputOffset);
        for (int ch = 0; ch < p.inChannels; ++ch) {
            for (int y = 0; y < h; ++y) {
                std::memcpy(inBase + static_cast<size_t>(ch) * p.inPlane + (static_cast<size_t>(y) + 1) * inRow + 1,
                            in + (static_cast<size_t>(ch) * h + y) * w, static_cast<size_t>(w) * sizeof(float));
            }
        }

        for (VkCommandBuffer cmd : c->segments) {
            if (cancel && cancel->load(std::memory_order_relaxed)) return fail(error, "cancelled", Status::Cancelled);
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            VkResult r = vkQueueSubmit(queue_, 1, &si, fence_);
            if (r != VK_SUCCESS) return deviceFailure(r, "vkQueueSubmit", error);
            r = vkWaitForFences(dev_, 1, &fence_, VK_TRUE, kFenceTimeoutNs);
            if (r != VK_SUCCESS) return deviceFailure(r, "vkWaitForFences", error);
            vkResetFences(dev_, 1, &fence_);
        }

        const size_t outRow = static_cast<size_t>(p.outW) + 2;
        const float* outBase = reinterpret_cast<const float*>(static_cast<const char*>(arena_.mapped) + p.outputOffset);
        for (int ch = 0; ch < p.outChannels; ++ch) {
            for (int y = 0; y < p.outH; ++y) {
                std::memcpy(out + (static_cast<size_t>(ch) * p.outH + y) * p.outW,
                            outBase + static_cast<size_t>(ch) * p.outPlane + (static_cast<size_t>(y) + 1) * outRow + 1,
                            static_cast<size_t>(p.outW) * sizeof(float));
            }
        }
        return Status::Ok;
    }

private:
    Status fail(std::string* error, const std::string& m, Status s) {
        if (error) *error = m;
        return s;
    }

    Status deviceFailure(VkResult r, const char* what, std::string* error) {
        // A timeout or a lost device leaves the GPU in an unknown state: never use it again.
        lost_ = true;
        return fail(error, std::string(what) + " failed (" + std::to_string(static_cast<int>(r)) + ")",
                    Status::DeviceLost);
    }

    bool createShader(const uint32_t* code, size_t bytes, VkShaderModule& module) {
        VkShaderModuleCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        si.codeSize = bytes;
        si.pCode = code;
        return vkCreateShaderModule(dev_, &si, nullptr, &module) == VK_SUCCESS;
    }

    bool pickMemoryType(uint32_t bits, uint32_t& index) {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd_, &mp);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        bool found = false;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if (!(bits & (1u << i)) || (mp.memoryTypes[i].propertyFlags & want) != want) continue;
            if (!found || (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) index = i;
            found = true;
        }
        return found;
    }

    bool createMem(size_t size, Mem& m) {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(dev_, &bi, nullptr, &m.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, m.buffer, &req);
        uint32_t type = 0;
        if (!pickMemoryType(req.memoryTypeBits, type)) return false;
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (vkAllocateMemory(dev_, &ai, nullptr, &m.memory) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(dev_, m.buffer, m.memory, 0) != VK_SUCCESS) return false;
        if (vkMapMemory(dev_, m.memory, 0, VK_WHOLE_SIZE, 0, &m.mapped) != VK_SUCCESS) return false;
        m.size = size;
        std::memset(m.mapped, 0, size);
        return true;
    }

    void destroyMem(Mem& m) {
        if (m.mapped) vkUnmapMemory(dev_, m.memory);
        if (m.buffer != VK_NULL_HANDLE) vkDestroyBuffer(dev_, m.buffer, nullptr);
        if (m.memory != VK_NULL_HANDLE) vkFreeMemory(dev_, m.memory, nullptr);
        m = Mem();
    }

    void destroyCompiled(Compiled& c) {
        if (c.commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(dev_, c.commandPool, nullptr);
        if (c.pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev_, c.pool, nullptr);
        c = Compiled();
    }

    void dropPrograms() {
        for (auto& kv : cache_) destroyCompiled(kv.second);
        cache_.clear();
    }

    void teardown() {
        if (dev_ == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(dev_);
        dropPrograms();
        destroyMem(arena_);
        destroyMem(weights_);
        if (fence_ != VK_NULL_HANDLE) vkDestroyFence(dev_, fence_, nullptr);
        for (int k = 0; k < 3; ++k) {
            if (pipelines_[k] != VK_NULL_HANDLE) vkDestroyPipeline(dev_, pipelines_[k], nullptr);
            if (shaders_[k] != VK_NULL_HANDLE) vkDestroyShaderModule(dev_, shaders_[k], nullptr);
        }
        if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev_, layout_, nullptr);
        if (setLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev_, setLayout_, nullptr);
    }

    // Builds (once per image size) the program, descriptor sets and command buffers.
    Compiled* compiled(int h, int w, std::string* error) {
        const std::pair<int, int> key(h, w);
        auto it = cache_.find(key);
        if (it != cache_.end()) return &it->second;
        if (cache_.size() >= 6) {
            vkQueueWaitIdle(queue_);
            dropPrograms();
        }
        Compiled c;
        if (!plan_->build(h, w, c.program, error)) return nullptr;
        const Program& p = c.program;
        if (p.maxBindBytes > maxRange_ || p.arenaBytes > kMaxArenaBytes) {
            if (error) *error = "this image size needs more GPU memory than the device allows";
            return nullptr;
        }
        if (arena_.size < p.arenaBytes) {
            vkQueueWaitIdle(queue_);
            dropPrograms();
            destroyMem(arena_);
            if (!createMem(p.arenaBytes, arena_)) {
                destroyMem(arena_);
                if (error) *error = "cannot allocate GPU memory for the activations";
                return nullptr;
            }
            lastH_ = lastW_ = -1;
        }

        size_t dispatches = 0;
        for (const Command& cmd : p.commands) dispatches += cmd.type == Command::kDispatch ? 1 : 0;
        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = static_cast<uint32_t>(std::max<size_t>(dispatches, 1) * 4);
        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = static_cast<uint32_t>(std::max<size_t>(dispatches, 1));
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &ps;
        VkCommandPoolCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.queueFamilyIndex = family_;
        size_t segments = 1;
        for (const Command& cmd : p.commands) segments += cmd.type == Command::kBoundary ? 1 : 0;
        if (vkCreateDescriptorPool(dev_, &dp, nullptr, &c.pool) != VK_SUCCESS ||
            vkCreateCommandPool(dev_, &cp, nullptr, &c.commandPool) != VK_SUCCESS) {
            destroyCompiled(c);
            if (error) *error = "cannot create command resources";
            return nullptr;
        }
        c.segments.resize(segments);
        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = c.commandPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = static_cast<uint32_t>(segments);
        if (vkAllocateCommandBuffers(dev_, &cai, c.segments.data()) != VK_SUCCESS) {
            destroyCompiled(c);
            if (error) *error = "cannot allocate command buffers";
            return nullptr;
        }

        std::vector<VkDescriptorSetLayout> layouts(std::max<size_t>(dispatches, 1), setLayout_);
        std::vector<VkDescriptorSet> sets(layouts.size(), VK_NULL_HANDLE);
        VkDescriptorSetAllocateInfo sai{};
        sai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        sai.descriptorPool = c.pool;
        sai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        sai.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(dev_, &sai, sets.data()) != VK_SUCCESS) {
            destroyCompiled(c);
            if (error) *error = "cannot allocate descriptor sets";
            return nullptr;
        }

        size_t segment = 0, setIndex = 0;
        auto begin = [&](VkCommandBuffer cb) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            return vkBeginCommandBuffer(cb, &bi) == VK_SUCCESS;
        };
        auto barrier = [&](VkCommandBuffer cb, bool toHost) {
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = toHost ? VK_ACCESS_HOST_READ_BIT
                                      : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                         VK_ACCESS_TRANSFER_WRITE_BIT);
            const VkPipelineStageFlags dstStage =
                toHost ? VK_PIPELINE_STAGE_HOST_BIT : (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0, 1, &mb, 0,
                                 nullptr, 0, nullptr);
        };
        bool ok = begin(c.segments[0]);
        for (const Command& cmd : p.commands) {
            VkCommandBuffer cb = c.segments[segment];
            if (cmd.type == Command::kBoundary) {
                barrier(cb, true);
                ok = ok && vkEndCommandBuffer(cb) == VK_SUCCESS;
                ++segment;
                ok = ok && begin(c.segments[segment]);
            } else if (cmd.type == Command::kCopy) {
                VkBufferCopy region{};
                region.srcOffset = cmd.copySrc;
                region.dstOffset = cmd.copyDst;
                region.size = cmd.copyBytes;
                vkCmdCopyBuffer(cb, arena_.buffer, arena_.buffer, 1, &region);
                barrier(cb, false);
            } else {
                VkDescriptorBufferInfo infos[4]{};
                VkWriteDescriptorSet writes[4]{};
                for (uint32_t i = 0; i < 4; ++i) {
                    infos[i].buffer = cmd.bind[i].buffer == 1 ? weights_.buffer : arena_.buffer;
                    infos[i].offset = cmd.bind[i].offset;
                    infos[i].range = cmd.bind[i].size;
                    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writes[i].dstSet = sets[setIndex];
                    writes[i].dstBinding = i;
                    writes[i].descriptorCount = 1;
                    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[i].pBufferInfo = &infos[i];
                }
                vkUpdateDescriptorSets(dev_, 4, writes, 0, nullptr);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[static_cast<int>(cmd.kernel)]);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &sets[setIndex], 0, nullptr);
                vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &cmd.push);
                vkCmdDispatch(cb, cmd.gx, cmd.gy, cmd.gz);
                barrier(cb, false);
                ++setIndex;
            }
        }
        barrier(c.segments[segment], true);
        ok = ok && vkEndCommandBuffer(c.segments[segment]) == VK_SUCCESS;
        if (!ok) {
            destroyCompiled(c);
            if (error) *error = "cannot record the command buffers";
            return nullptr;
        }
        return &(cache_[key] = std::move(c));
    }

    VkPhysicalDevice pd_;
    VkDevice dev_;
    VkQueue queue_;
    uint32_t family_;
    std::unique_ptr<Plan> plan_;
    size_t maxRange_ = 0;
    VkShaderModule shaders_[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipeline pipelines_[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    Mem arena_, weights_;
    std::map<std::pair<int, int>, Compiled> cache_;
    int lastH_ = -1, lastW_ = -1;
    bool lost_ = false;
};

}  // namespace

std::unique_ptr<Backend> createVulkanBackend(const VulkanHandles& vk, const Model& model, std::string* status) {
    std::string reason;
    std::unique_ptr<Plan> plan = Plan::create(model, &reason);
    if (!plan) {
        if (status) *status = "model not supported on the GPU path: " + reason;
        return nullptr;
    }
    std::unique_ptr<VulkanBackend> backend(new VulkanBackend(vk, std::move(plan)));
    if (!backend->init(&reason)) {
        if (status) *status = reason;
        return nullptr;
    }
    return std::unique_ptr<Backend>(backend.release());
}

}  // namespace gpu
}  // namespace vva

#endif  // VVA_GPU_AVAILABLE
