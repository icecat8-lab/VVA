// vva-gpubench: compares several 3x3 convolution kernels on the GPU (Vulkan compute) and checks
// each one against a CPU reference before reporting its speed.
//
//   vva-gpubench <directory containing the .spv files>
//
// Kernels: baseline fp32, register-tiled fp32 (padded input, 4 pixels x 8 channels per thread) and
// the same tiled kernel in fp16 (only when the device supports it).
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

struct Dev {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

struct Config {
    int cin, cout, h, w;
};

struct Variant {
    const char* label;
    const char* file;
    bool padded;  // input planes carry a one-pixel zero border
    bool half;    // fp16 storage and arithmetic
    int px;       // output pixels per invocation along x
    int wgx, wgy; // local workgroup size
};

struct Result {
    bool ran = false;
    bool correct = false;
    double gflops = 0.0;
    double ms = 0.0;
    double err = 0.0;
};

enum Status { kOk = 0, kVulkanError = 1, kFatal = 2 };

const Variant kVariants[] = {
    {"baseline fp32", "conv3x3.spv", false, false, 1, 8, 8},
    {"tiled fp32", "conv3x3_t4.spv", true, false, 4, 16, 4},
    {"tiled fp16", "conv3x3_t4_f16.spv", true, true, 4, 16, 4},
};

bool check(VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    std::printf("  Vulkan error %d in %s\n", static_cast<int>(r), what);
    return false;
}

uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t expBits = (x >> 23) & 0xFFu;
    const uint32_t man = x & 0x7FFFFFu;
    if (expBits == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
    const int32_t exp = static_cast<int32_t>(expBits) - 127 + 15;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = man | 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
}

float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            do {
                ++e;
                man <<= 1;
            } while ((man & 0x400u) == 0);
            man &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
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

bool createBuffer(const Dev& c, size_t size, Buffer& b) {
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
        std::printf("  no host-visible coherent memory type\n");
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

void destroyBuffer(const Dev& c, Buffer& b) {
    if (b.mapped) vkUnmapMemory(c.device, b.memory);
    if (b.buffer != VK_NULL_HANDLE) vkDestroyBuffer(c.device, b.buffer, nullptr);
    if (b.memory != VK_NULL_HANDLE) vkFreeMemory(c.device, b.memory, nullptr);
    b = Buffer();
}

bool loadSpirv(const std::string& path, std::vector<uint32_t>& code) {
    std::ifstream f(path.c_str(), std::ios::binary | std::ios::ate);
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

bool createPipeline(const Dev& c, const std::vector<uint32_t>& code, Pipeline& p) {
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
    ps.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = 1;
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

void destroyPipeline(const Dev& c, Pipeline& p) {
    if (p.fence != VK_NULL_HANDLE) vkDestroyFence(c.device, p.fence, nullptr);
    if (p.cmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(c.device, p.cmdPool, nullptr);
    if (p.pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(c.device, p.pool, nullptr);
    if (p.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(c.device, p.pipeline, nullptr);
    if (p.layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(c.device, p.layout, nullptr);
    if (p.setLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(c.device, p.setLayout, nullptr);
    if (p.shader != VK_NULL_HANDLE) vkDestroyShaderModule(c.device, p.shader, nullptr);
    p = Pipeline();
}

// A second logical device with fp16 arithmetic and 16-bit storage enabled. Returns false when the
// driver does not offer them (creation fails cleanly in that case).
bool createFp16Device(const vva::VulkanContext& base, Dev& out) {
    static const char* const kExtensions[] = {"VK_KHR_shader_float16_int8", "VK_KHR_16bit_storage"};
    for (int attempt = 0; attempt < 2; ++attempt) {
        VkPhysicalDevice16BitStorageFeatures storage{};
        storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        storage.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceShaderFloat16Int8FeaturesKHR fp16{};
        fp16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
        fp16.shaderFloat16 = VK_TRUE;
        fp16.pNext = &storage;

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = base.queueFamilyIndex;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;

        VkDeviceCreateInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        di.pNext = &fp16;
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = attempt == 0 ? 2u : 1u;  // second try: without the 16-bit storage extension name
        di.ppEnabledExtensionNames = kExtensions;

        VkDevice device = VK_NULL_HANDLE;
        if (vkCreateDevice(base.physicalDevice, &di, nullptr, &device) == VK_SUCCESS) {
            out.physicalDevice = base.physicalDevice;
            out.device = device;
            out.queueFamilyIndex = base.queueFamilyIndex;
            vkGetDeviceQueue(device, base.queueFamilyIndex, 0, &out.computeQueue);
            return true;
        }
    }
    return false;
}

// Runs one kernel on one problem size and fills `res`.
Status runConfig(const Dev& c, Pipeline& p, const Variant& v, const Config& cfg, Result& res) {
    res = Result();
    if (cfg.w % v.px != 0 || cfg.cout % 8 != 0) return kVulkanError;

    const size_t elem = v.half ? sizeof(uint16_t) : sizeof(float);
    const size_t plane = static_cast<size_t>(cfg.h) * cfg.w;
    const size_t pw = static_cast<size_t>(cfg.w) + 2;
    const size_t inPlane = v.padded ? static_cast<size_t>(cfg.h + 2) * pw : plane;
    const size_t inCount = static_cast<size_t>(cfg.cin) * inPlane;
    const size_t outCount = static_cast<size_t>(cfg.cout) * plane;
    const size_t wCount = static_cast<size_t>(cfg.cout) * cfg.cin * 9;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> input(static_cast<size_t>(cfg.cin) * plane), wlog(wCount);
    for (float& x : input) x = dist(rng) + 0.5f;
    for (float& x : wlog) x = dist(rng) * 0.1f;

    Buffer bin, bw, bout;
    Status status = kOk;
    bool ok = createBuffer(c, inCount * elem, bin) && createBuffer(c, wCount * elem, bw) &&
              createBuffer(c, outCount * elem, bout);

    auto store = [&](void* base, size_t index, float value) {
        if (v.half) static_cast<uint16_t*>(base)[index] = floatToHalf(value);
        else static_cast<float*>(base)[index] = value;
    };

    if (ok) {
        std::memset(bin.mapped, 0, inCount * elem);
        for (int ic = 0; ic < cfg.cin; ++ic) {
            for (int y = 0; y < cfg.h; ++y) {
                for (int x = 0; x < cfg.w; ++x) {
                    const size_t dstIndex = v.padded ? static_cast<size_t>(ic) * inPlane + (static_cast<size_t>(y) + 1) * pw + x + 1
                                                     : static_cast<size_t>(ic) * plane + static_cast<size_t>(y) * cfg.w + x;
                    store(bin.mapped, dstIndex, input[static_cast<size_t>(ic) * plane + static_cast<size_t>(y) * cfg.w + x]);
                }
            }
        }
        for (int ocb = 0; ocb < cfg.cout / 8; ++ocb) {
            for (int ic = 0; ic < cfg.cin; ++ic) {
                for (int tap = 0; tap < 9; ++tap) {
                    for (int j = 0; j < 8; ++j) {
                        const size_t dstIndex = ((static_cast<size_t>(ocb) * cfg.cin + ic) * 9 + tap) * 8 + j;
                        store(bw.mapped, dstIndex, wlog[((static_cast<size_t>(ocb) * 8 + j) * cfg.cin + ic) * 9 + tap]);
                    }
                }
            }
        }
        std::memset(bout.mapped, 0, outCount * elem);
    }

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const int kPerSubmit = 2;
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
        const uint32_t gx = static_cast<uint32_t>(((cfg.w / v.px) + v.wgx - 1) / v.wgx);
        const uint32_t gy = static_cast<uint32_t>((cfg.h + v.wgy - 1) / v.wgy);
        const uint32_t gz = static_cast<uint32_t>(cfg.cout / 8);
        for (int i = 0; i < kPerSubmit; ++i) {
            vkCmdDispatch(cmd, gx, gy, gz);
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
            ok = check(vkQueueSubmit(c.computeQueue, 1, &si, p.fence), "vkQueueSubmit");
            if (!ok) break;
            const VkResult wr = vkWaitForFences(c.device, 1, &p.fence, VK_TRUE, 20ull * 1000ull * 1000ull * 1000ull);
            if (wr == VK_TIMEOUT) {
                std::printf("  the GPU did not finish within 20 s - stopping\n");
                return kFatal;  // resources are left alone on purpose: the GPU may still be using them
            }
            ok = check(wr, "vkWaitForFences");
            const auto t1 = std::chrono::steady_clock::now();
            vkResetFences(c.device, 1, &p.fence);
            if (s > 0) seconds += std::chrono::duration<double>(t1 - t0).count();
        }
    }

    if (ok) {
        const void* out = bout.mapped;
        auto fetch = [&](size_t index) -> double {
            return v.half ? halfToFloat(static_cast<const uint16_t*>(out)[index]) : static_cast<const float*>(out)[index];
        };
        std::mt19937 pick(99);
        double worst = 0.0;
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
                        ref += static_cast<double>(input[static_cast<size_t>(ic) * plane + static_cast<size_t>(yy) * cfg.w + xx]) *
                               wlog[(static_cast<size_t>(oc) * cfg.cin + ic) * 9 + ky * 3 + kx];
                    }
                }
            }
            const double got = fetch(static_cast<size_t>(oc) * plane + static_cast<size_t>(y) * cfg.w + x);
            const double e = std::fabs(got - ref) / std::fmax(1.0, std::fabs(ref));
            if (!(e == e)) worst = HUGE_VAL;  // NaN counts as a failure and stays one
            else if (e > worst) worst = e;
        }
        res.err = worst;
        res.correct = worst <= (v.half ? 3e-2 : 1e-3);
        const double dispatches = static_cast<double>(kSubmits) * kPerSubmit;
        res.ms = seconds * 1000.0 / dispatches;
        res.gflops = 2.0 * cfg.cin * cfg.cout * 9.0 * static_cast<double>(plane) * dispatches / seconds / 1e9;
        res.ran = true;
    } else {
        status = kVulkanError;
    }

    destroyBuffer(c, bin);
    destroyBuffer(c, bw);
    destroyBuffer(c, bout);
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: vva-gpubench <directory with the .spv files>\n");
        return 1;
    }
    const std::string dir = argv[1];

    vva::VulkanContext ctx;
    if (!ctx.init()) {
        std::printf("Vulkan is not available\n");
        return 2;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
    std::printf("Device: %s\n", props.deviceName);
    std::printf("Max workgroup invocations: %u, shared memory: %u bytes\n\n",
                props.limits.maxComputeWorkGroupInvocations, props.limits.maxComputeSharedMemorySize);

    Dev dev32;
    dev32.physicalDevice = ctx.physicalDevice;
    dev32.device = ctx.device;
    dev32.computeQueue = ctx.computeQueue;
    dev32.queueFamilyIndex = ctx.queueFamilyIndex;
    Dev dev16;
    bool tried16 = false, have16 = false;

    const Config smoke = {64, 32, 32, 32};  // quick correctness gate before the heavy sizes
    const Config configs[] = {{64, 32, 128, 128}, {64, 64, 128, 128}, {64, 64, 256, 256}};
    const size_t kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);
    std::vector<Result> finalResult(kVariantCount);
    int rc = 0;

    for (size_t vi = 0; vi < kVariantCount; ++vi) {
        const Variant& v = kVariants[vi];
        std::printf("== %s ==\n", v.label);

        std::vector<uint32_t> code;
        if (!loadSpirv(dir + "/" + v.file, code)) {
            std::printf("  skipped: shader %s not found\n\n", v.file);
            continue;
        }
        Dev* dev = &dev32;
        if (v.half) {
            if (!tried16) {
                tried16 = true;
                have16 = createFp16Device(ctx, dev16);
            }
            if (!have16) {
                std::printf("  skipped: this device does not offer fp16 compute\n\n");
                continue;
            }
            dev = &dev16;
        }
        Pipeline pipe;
        if (!createPipeline(*dev, code, pipe)) {
            std::printf("  skipped: the driver could not build this pipeline\n\n");
            destroyPipeline(*dev, pipe);
            continue;
        }

        Result res;
        Status st = runConfig(*dev, pipe, v, smoke, res);
        if (st == kFatal) return 5;
        if (st != kOk || !res.correct) {
            std::printf("  skipped: failed the correctness check (error %.1e)\n\n", res.err);
            if (vi == 0) rc = 4;
            destroyPipeline(*dev, pipe);
            continue;
        }
        for (const Config& cfg : configs) {
            st = runConfig(*dev, pipe, v, cfg, res);
            if (st == kFatal) return 5;
            if (st != kOk) break;
            std::printf("  %3d -> %3d channels at %3dx%-3d: %8.2f ms, %7.1f GFLOP/s, max error %.1e %s\n", cfg.cin, cfg.cout,
                        cfg.w, cfg.h, res.ms, res.gflops, res.err, res.correct ? "(ok)" : "(WRONG)");
            if (cfg.cout == 64 && cfg.h == 256) finalResult[vi] = res;
        }
        std::printf("\n");
        destroyPipeline(*dev, pipe);
    }

    std::printf("== Summary (64 -> 64 channels at 256x256, the shape of the large layers) ==\n");
    double best = 0.0;
    const char* bestName = "";
    for (size_t vi = 0; vi < kVariantCount; ++vi) {
        const Result& r = finalResult[vi];
        if (!r.ran) {
            std::printf("  %-14s not measured\n", kVariants[vi].label);
            continue;
        }
        std::printf("  %-14s %7.1f GFLOP/s   max error %.1e %s\n", kVariants[vi].label, r.gflops, r.err,
                    r.correct ? "" : "(WRONG)");
        if (r.correct && r.gflops > best) {
            best = r.gflops;
            bestName = kVariants[vi].label;
        }
    }
    if (best > 0.0) {
        const double modelFlops = 2.0 * 5706432.0 * 24576.0;  // maru-x4 on a 128x192 image
        std::printf("\nBest: %s. Rough estimate for maru-x4 on a 128x192 image: %.1f s (CPU, 8 threads: 15.0 s)\n", bestName,
                    modelFlops / (best * 1e9));
    } else {
        rc = rc ? rc : 3;
    }

    if (have16) vkDestroyDevice(dev16.device, nullptr);
    return rc;
}
