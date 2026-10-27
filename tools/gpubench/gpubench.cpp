// vva-gpubench: measures a 3x3 convolution on the GPU (Vulkan compute, fp32) and checks
// it against a CPU reference. It tells whether a GPU backend is worth building.
//
//   vva-gpubench conv3x3.spv
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "backend/vulkan_context.cpp"

namespace {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

struct Config {
    int cin, cout, h, w;
};

bool check(VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    std::printf("Vulkan error %d in %s\n", static_cast<int>(r), what);
    return false;
}

bool findMemoryType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want, uint32_t& index) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            index = i;
            return true;
        }
    }
    return false;
}

bool createBuffer(const vva::VulkanContext& c, size_t size, Buffer& b) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(vkCreateBuffer(c.device, &bi, nullptr, &b.buffer), "vkCreateBuffer")) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(c.device, b.buffer, &req);
    uint32_t type = 0;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!findMemoryType(c.physicalDevice, req.memoryTypeBits, want, type)) {
        std::printf("no host-visible coherent memory type\n");
        return false;
    }
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (!check(vkAllocateMemory(c.device, &ai, nullptr, &b.memory), "vkAllocateMemory")) return false;
    if (!check(vkBindBufferMemory(c.device, b.buffer, b.memory, 0), "vkBindBufferMemory")) return false;
    return check(vkMapMemory(c.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
}

void destroyBuffer(const vva::VulkanContext& c, Buffer& b) {
    if (b.mapped) vkUnmapMemory(c.device, b.memory);
    if (b.buffer != VK_NULL_HANDLE) vkDestroyBuffer(c.device, b.buffer, nullptr);
    if (b.memory != VK_NULL_HANDLE) vkFreeMemory(c.device, b.memory, nullptr);
    b = Buffer();
}

bool loadSpirv(const char* path, std::vector<uint32_t>& code) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize size = f.tellg();
    if (size <= 0 || size % 4 != 0) return false;
    code.resize(static_cast<size_t>(size) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), size);
    return static_cast<bool>(f);
}

struct Pipeline {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
};

bool createPipeline(const vva::VulkanContext& c, const std::vector<uint32_t>& code, int maxSets, Pipeline& p) {
    VkShaderModuleCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    si.codeSize = code.size() * sizeof(uint32_t);
    si.pCode = code.data();
    if (!check(vkCreateShaderModule(c.device, &si, nullptr, &p.shader), "vkCreateShaderModule")) return false;

    VkDescriptorSetLayoutBinding binds[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 3;
    dl.pBindings = binds;
    if (!check(vkCreateDescriptorSetLayout(c.device, &dl, nullptr, &p.setLayout), "vkCreateDescriptorSetLayout")) return false;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 4 * sizeof(int32_t);
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &p.setLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    if (!check(vkCreatePipelineLayout(c.device, &pl, nullptr, &p.layout), "vkCreatePipelineLayout")) return false;

    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = p.shader;
    ci.stage.pName = "main";
    ci.layout = p.layout;
    if (!check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline), "vkCreateComputePipelines")) return false;

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = static_cast<uint32_t>(3 * maxSets);
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = static_cast<uint32_t>(maxSets);
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    if (!check(vkCreateDescriptorPool(c.device, &dp, nullptr, &p.pool), "vkCreateDescriptorPool")) return false;

    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.queueFamilyIndex = c.queueFamilyIndex;
    if (!check(vkCreateCommandPool(c.device, &cp, nullptr, &p.cmdPool), "vkCreateCommandPool")) return false;

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    return check(vkCreateFence(c.device, &fi, nullptr, &p.fence), "vkCreateFence");
}

void destroyPipeline(const vva::VulkanContext& c, Pipeline& p) {
    if (p.fence != VK_NULL_HANDLE) vkDestroyFence(c.device, p.fence, nullptr);
    if (p.cmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(c.device, p.cmdPool, nullptr);
    if (p.pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(c.device, p.pool, nullptr);
    if (p.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(c.device, p.pipeline, nullptr);
    if (p.layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(c.device, p.layout, nullptr);
    if (p.setLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(c.device, p.setLayout, nullptr);
    if (p.shader != VK_NULL_HANDLE) vkDestroyShaderModule(c.device, p.shader, nullptr);
    p = Pipeline();
}

// Returns false on a Vulkan error. `gflops` is the measured rate; `correct` tells whether the output matched.
bool runConfig(const vva::VulkanContext& c, Pipeline& p, const Config& cfg, double& gflops, double& msPerDispatch,
               bool& correct) {
    const size_t plane = static_cast<size_t>(cfg.h) * cfg.w;
    const size_t inCount = static_cast<size_t>(cfg.cin) * plane;
    const size_t outCount = static_cast<size_t>(cfg.cout) * plane;
    const size_t wCount = static_cast<size_t>(cfg.cout) * cfg.cin * 9;

    // Random data; weights are kept in logical order for the CPU check and packed for the shader.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> input(inCount), wlog(wCount);
    for (float& v : input) v = dist(rng) + 0.5f;
    for (float& v : wlog) v = dist(rng) * 0.1f;

    Buffer bin, bw, bout;
    bool ok = createBuffer(c, inCount * sizeof(float), bin) && createBuffer(c, wCount * sizeof(float), bw) &&
              createBuffer(c, outCount * sizeof(float), bout);
    if (ok) {
        std::memcpy(bin.mapped, input.data(), inCount * sizeof(float));
        float* packed = static_cast<float*>(bw.mapped);
        for (int ocb = 0; ocb < cfg.cout / 8; ++ocb) {
            for (int ic = 0; ic < cfg.cin; ++ic) {
                for (int tap = 0; tap < 9; ++tap) {
                    for (int j = 0; j < 8; ++j) {
                        const size_t dstIndex = ((static_cast<size_t>(ocb) * cfg.cin + ic) * 9 + tap) * 8 + j;
                        packed[dstIndex] = wlog[((static_cast<size_t>(ocb) * 8 + j) * cfg.cin + ic) * 9 + tap];
                    }
                }
            }
        }
        std::memset(bout.mapped, 0, outCount * sizeof(float));
    }

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const int kPerSubmit = 3;
    const int kSubmits = 6;
    double seconds = 0.0;

    if (ok) {
        vkResetDescriptorPool(c.device, p.pool, 0);
        vkResetCommandPool(c.device, p.cmdPool, 0);

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = p.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &p.setLayout;
        ok = check(vkAllocateDescriptorSets(c.device, &ai, &set), "vkAllocateDescriptorSets");
    }
    if (ok) {
        VkDescriptorBufferInfo infos[3]{};
        infos[0].buffer = bin.buffer;
        infos[0].range = VK_WHOLE_SIZE;
        infos[1].buffer = bw.buffer;
        infos[1].range = VK_WHOLE_SIZE;
        infos[2].buffer = bout.buffer;
        infos[2].range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(c.device, 3, writes, 0, nullptr);

        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = p.cmdPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        ok = check(vkAllocateCommandBuffers(c.device, &cai, &cmd), "vkAllocateCommandBuffers");
    }
    if (ok) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        ok = check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    }
    if (ok) {
        const int32_t params[4] = {cfg.cin, cfg.cout, cfg.h, cfg.w};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), params);
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        for (int i = 0; i < kPerSubmit; ++i) {
            vkCmdDispatch(cmd, static_cast<uint32_t>((cfg.w + 7) / 8), static_cast<uint32_t>((cfg.h + 7) / 8),
                          static_cast<uint32_t>(cfg.cout / 8));
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &barrier, 0, nullptr, 0, nullptr);
        }
        ok = check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    }
    if (ok) {
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        for (int s = 0; s < kSubmits + 1 && ok; ++s) {  // the first submit is a warm-up
            const auto t0 = std::chrono::steady_clock::now();
            ok = check(vkQueueSubmit(c.computeQueue, 1, &si, p.fence), "vkQueueSubmit") &&
                 check(vkWaitForFences(c.device, 1, &p.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            const auto t1 = std::chrono::steady_clock::now();
            vkResetFences(c.device, 1, &p.fence);
            if (s > 0) seconds += std::chrono::duration<double>(t1 - t0).count();
        }
    }

    correct = false;
    if (ok) {
        // CPU reference at random positions.
        const float* out = static_cast<const float*>(bout.mapped);
        std::mt19937 pick(99);
        int bad = 0;
        for (int n = 0; n < 200; ++n) {
            const int oc = static_cast<int>(pick() % cfg.cout);
            const int y = static_cast<int>(pick() % cfg.h);
            const int x = static_cast<int>(pick() % cfg.w);
            double ref = 0.0;
            for (int ic = 0; ic < cfg.cin; ++ic) {
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        const int yy = y + ky - 1, xx = x + kx - 1;
                        if (yy < 0 || yy >= cfg.h || xx < 0 || xx >= cfg.w) continue;
                        ref += static_cast<double>(input[ic * plane + static_cast<size_t>(yy) * cfg.w + xx]) *
                               wlog[(static_cast<size_t>(oc) * cfg.cin + ic) * 9 + ky * 3 + kx];
                    }
                }
            }
            const double got = out[static_cast<size_t>(oc) * plane + static_cast<size_t>(y) * cfg.w + x];
            if (std::fabs(got - ref) > 1e-3 * std::fmax(1.0, std::fabs(ref))) ++bad;
        }
        correct = bad == 0;
        const double dispatches = static_cast<double>(kSubmits) * kPerSubmit;
        msPerDispatch = seconds * 1000.0 / dispatches;
        const double flops = 2.0 * cfg.cin * cfg.cout * 9.0 * plane;
        gflops = flops * dispatches / seconds / 1e9;
    }

    destroyBuffer(c, bin);
    destroyBuffer(c, bw);
    destroyBuffer(c, bout);
    return ok;
}

int runBench(const vva::VulkanContext& c, const char* spvPath) {
    std::vector<uint32_t> code;
    if (!loadSpirv(spvPath, code)) {
        std::printf("cannot read the shader %s\n", spvPath);
        return 1;
    }
    Pipeline pipe;
    if (!createPipeline(c, code, 1, pipe)) {
        destroyPipeline(c, pipe);
        return 3;
    }
    const Config configs[] = {{64, 32, 128, 128}, {64, 64, 128, 128}, {64, 64, 256, 256}};
    int status = 0;
    double reference = 0.0;
    for (const Config& cfg : configs) {
        double gflops = 0.0, ms = 0.0;
        bool correct = false;
        if (!runConfig(c, pipe, cfg, gflops, ms, correct)) {
            status = 3;
            break;
        }
        std::printf("conv3x3 %3d -> %3d channels at %dx%d: %7.2f ms, %6.1f GFLOP/s, output %s\n", cfg.cin, cfg.cout,
                    cfg.w, cfg.h, ms, gflops, correct ? "correct" : "WRONG");
        if (!correct) status = 4;
        if (cfg.cout == 64 && cfg.h == 128) reference = gflops;
    }
    if (status == 0 && reference > 0.0) {
        // maru-x4 (RRDBNet nf=64 gc=32 nb=6) on a 128x192 image: 2 * 5,706,432 MAC/px * 24,576 px
        const double modelFlops = 2.0 * 5706432.0 * 24576.0;
        std::printf("\nRough estimate for maru-x4 on a 128x192 image with this kernel: %.1f s\n",
                    modelFlops / (reference * 1e9));
        std::printf("(measured on this phone's CPU with 8 threads: 15.0 s)\n");
    }
    destroyPipeline(c, pipe);
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: vva-gpubench conv3x3.spv\n");
        return 1;
    }
    vva::VulkanContext ctx;
    if (!ctx.init()) {
        std::printf("Vulkan is not available\n");
        return 2;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
    std::printf("Device: %s\n", props.deviceName);
    std::printf("Max workgroup invocations: %u, shared memory: %u bytes\n",
                props.limits.maxComputeWorkGroupInvocations, props.limits.maxComputeSharedMemorySize);
    return runBench(ctx, argv[1]);
}
