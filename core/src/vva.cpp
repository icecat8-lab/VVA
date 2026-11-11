#include "vva/vva.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#include "exec.h"
#include "gpu/gpu_backend.h"
#include "gpu/gpu_types.h"
#include "model.h"

#ifndef VVA_NO_VULKAN
#include "backend/vulkan_context.cpp"
#endif

namespace vva {

namespace {

constexpr size_t kMiB = 1024u * 1024u;
constexpr int kMaxImageSide = 1 << 16;

struct AxisTaps {
    std::vector<int> first;
    std::vector<int> count;
    std::vector<float> w;
    int stride = 0;
};

AxisTaps bilinearTaps(int inSize, int outSize) {
    AxisTaps t;
    const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
    const double filterScale = scale > 1.0 ? scale : 1.0;
    const double support = filterScale;
    t.stride = static_cast<int>(std::ceil(support)) * 2 + 1;
    t.first.assign(static_cast<size_t>(outSize), 0);
    t.count.assign(static_cast<size_t>(outSize), 0);
    t.w.assign(static_cast<size_t>(outSize) * t.stride, 0.f);
    for (int x = 0; x < outSize; ++x) {
        const double center = (x + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        int n = xmax - xmin;
        if (n > t.stride) n = t.stride;
        double total = 0.0;
        float* wk = &t.w[static_cast<size_t>(x) * t.stride];
        for (int k = 0; k < n; ++k) {
            double d = (k + xmin - center + 0.5) / filterScale;
            if (d < 0) d = -d;
            const double wv = d < 1.0 ? 1.0 - d : 0.0;
            wk[k] = static_cast<float>(wv);
            total += wv;
        }
        if (total != 0.0) {
            for (int k = 0; k < n; ++k) wk[k] = static_cast<float>(wk[k] / total);
        }
        t.first[static_cast<size_t>(x)] = xmin;
        t.count[static_cast<size_t>(x)] = n;
    }
    return t;
}

inline uint8_t roundToByte(float v) {
    v += 0.5f;
    if (!(v > 0.f)) return 0;
    if (v >= 255.f) return 255;
    return static_cast<uint8_t>(v);
}

void resizeRgb8(const uint8_t* rgba, int w, int h, int size, std::vector<uint8_t>& out) {
    const AxisTaps tx = bilinearTaps(w, size);
    const AxisTaps ty = bilinearTaps(h, size);
    std::vector<uint8_t> mid(static_cast<size_t>(h) * size * 3);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = rgba + static_cast<size_t>(y) * w * 4;
        uint8_t* dst = &mid[static_cast<size_t>(y) * size * 3];
        for (int x = 0; x < size; ++x) {
            const float* wk = &tx.w[static_cast<size_t>(x) * tx.stride];
            const int first = tx.first[static_cast<size_t>(x)], n = tx.count[static_cast<size_t>(x)];
            float r = 0.f, g = 0.f, b = 0.f;
            for (int k = 0; k < n; ++k) {
                const uint8_t* px = row + static_cast<size_t>(first + k) * 4;
                r += wk[k] * px[0];
                g += wk[k] * px[1];
                b += wk[k] * px[2];
            }
            dst[3 * x + 0] = roundToByte(r);
            dst[3 * x + 1] = roundToByte(g);
            dst[3 * x + 2] = roundToByte(b);
        }
    }
    out.assign(static_cast<size_t>(size) * size * 3, 0);
    for (int y = 0; y < size; ++y) {
        const float* wk = &ty.w[static_cast<size_t>(y) * ty.stride];
        const int first = ty.first[static_cast<size_t>(y)], n = ty.count[static_cast<size_t>(y)];
        uint8_t* dst = &out[static_cast<size_t>(y) * size * 3];
        for (int x = 0; x < size * 3; ++x) {
            float acc = 0.f;
            for (int k = 0; k < n; ++k) acc += wk[k] * mid[static_cast<size_t>(first + k) * size * 3 + x];
            dst[x] = roundToByte(acc);
        }
    }
}

long parseMemAvailableKb(const std::string& text) {
    const std::string key = "MemAvailable:";
    const size_t pos = text.find(key);
    if (pos == std::string::npos) return -1;
    const long v = std::strtol(text.c_str() + pos + key.size(), nullptr, 10);
    return v > 0 ? v : -1;
}

size_t autoBudgetBytes(long availableKb) {
    const size_t lo = 320 * kMiB, hi = 768 * kMiB;
    if (availableKb < 0) return lo;
    const size_t quarter = static_cast<size_t>(availableKb) * 1024 / 4;
    return std::min(hi, std::max(lo, quarter));
}

size_t currentAutoBudget() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return autoBudgetBytes(-1);
    char buf[2048];
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = 0;
    return autoBudgetBytes(parseMemAvailableKb(buf));
}

struct GpuConfig {
    std::string identity;
    bool usable = false;
    int variant = 0;
    std::string reason;
    long long when = 0;
};

std::string formatGpuConfig(const GpuConfig& c) {
    std::string s = "vva-gpu-config 2\n";
    s += "identity=" + c.identity + "\n";
    s += std::string("usable=") + (c.usable ? "1" : "0") + "\n";
    s += "variant=" + std::to_string(c.variant) + "\n";
    s += "reason=" + c.reason + "\n";
    s += "time=" + std::to_string(c.when) + "\n";
    return s;
}

bool parseGpuConfig(const std::string& text, GpuConfig& c) {
    c = GpuConfig();
    if (text.compare(0, 17, "vva-gpu-config 2\n") != 0) return false;
    size_t pos = 17;
    bool haveIdentity = false, haveUsable = false;
    while (pos < text.size()) {
        size_t end = text.find('\n', pos);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(pos, end - pos);
        pos = end + 1;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq), value = line.substr(eq + 1);
        if (key == "identity") {
            c.identity = value;
            haveIdentity = true;
        } else if (key == "usable") {
            c.usable = value == "1";
            haveUsable = value == "1" || value == "0";
        } else if (key == "variant") {
            c.variant = static_cast<int>(std::strtol(value.c_str(), nullptr, 10));
        } else if (key == "reason") {
            c.reason = value.substr(0, 200);
        } else if (key == "time") {
            c.when = std::strtoll(value.c_str(), nullptr, 10);
        }
    }
    return haveIdentity && haveUsable;
}

bool gpuConfigFresh(const GpuConfig& c, const std::string& identity, int variants, long long nowSeconds) {
    if (c.identity != identity || c.variant < 0 || c.variant >= variants) return false;
    const long long age = nowSeconds - c.when;
    return age >= 0 && age <= 30LL * 24 * 3600;
}

bool readTextFile(const std::string& path, std::string& text) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[4096];
    const size_t n = std::fread(buf, 1, sizeof(buf), f);
    std::fclose(f);
    text.assign(buf, n);
    return true;
}

void writeTextFile(const std::string& path, const std::string& text) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
}

inline float clamp01(float v) {
    if (!(v > 0.f)) return 0.f;
    if (v > 1.f) return 1.f;
    return v;
}

void upscaleAlpha(const uint8_t* in, int w, int h, uint8_t* out, int s) {
    const int ow = w * s, oh = h * s;
    const float inv = 1.0f / static_cast<float>(s);
    for (int oy = 0; oy < oh; ++oy) {
        float fy = (static_cast<float>(oy) + 0.5f) * inv - 0.5f;
        if (fy < 0.f) fy = 0.f;
        int y0 = static_cast<int>(fy);
        if (y0 > h - 1) y0 = h - 1;
        const int y1 = std::min(y0 + 1, h - 1);
        const float wy = std::min(fy - static_cast<float>(y0), 1.f);
        for (int ox = 0; ox < ow; ++ox) {
            float fx = (static_cast<float>(ox) + 0.5f) * inv - 0.5f;
            if (fx < 0.f) fx = 0.f;
            int x0 = static_cast<int>(fx);
            if (x0 > w - 1) x0 = w - 1;
            const int x1 = std::min(x0 + 1, w - 1);
            const float wx = std::min(fx - static_cast<float>(x0), 1.f);
            const float a00 = in[(static_cast<size_t>(y0) * w + x0) * 4 + 3];
            const float a01 = in[(static_cast<size_t>(y0) * w + x1) * 4 + 3];
            const float a10 = in[(static_cast<size_t>(y1) * w + x0) * 4 + 3];
            const float a11 = in[(static_cast<size_t>(y1) * w + x1) * 4 + 3];
            const float top = a00 + (a01 - a00) * wx;
            const float bot = a10 + (a11 - a10) * wx;
            const float v = top + (bot - top) * wy + 0.5f;
            out[(static_cast<size_t>(oy) * ow + ox) * 4 + 3] = static_cast<uint8_t>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
        }
    }
}

}

class Engine::Impl {
public:
    ~Impl() {
        gpuBackend.reset();
    }

    std::unique_ptr<Model> model;
    std::unique_ptr<Executor> exec;

    mutable std::mutex errorMutex;
    std::string error;

    std::mutex runMutex;
    std::atomic<bool> cancelFlag{false};

    std::string modelPath;
    int classifierSize = 0;

    int tileSize = 0;
    int tilePad = 16;
    int threads = 0;
    int budgetMB = 0;
    ProgressCallback progress;
    DeviceType usedDevice = DeviceType::CPU_NEON;

    std::unique_ptr<gpu::Backend> gpuBackend;
    std::string gpuState = "no model loaded";
    bool gpuNeedsSpeedCheck = true;

#ifndef VVA_NO_VULKAN
    VulkanContext vkContext;
    std::once_flag vkOnce;
#endif

    void setError(const std::string& message) {
        std::lock_guard<std::mutex> lock(errorMutex);
        error = message;
    }
    std::string getError() const {
        std::lock_guard<std::mutex> lock(errorMutex);
        return error;
    }

    int effectiveThreads() const {
        if (threads > 0) return std::min(threads, 64);
        const unsigned hw = std::thread::hardware_concurrency();
        return static_cast<int>(std::min(8u, std::max(1u, hw)));
    }

    bool load(const std::string& path) {
        std::unique_lock<std::mutex> lock(runMutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            setError("the engine is busy");
            return false;
        }
        std::string err;
        std::unique_ptr<Model> loaded = Model::load(path, &err);
        if (!loaded) {
            setError(err);
            return false;
        }
        const bool classifier = loaded->task() == "classify";
        std::unique_ptr<Executor> prepared;
        try {
            prepared = Executor::create(*loaded, &err, classifier);
        } catch (const std::bad_alloc&) {
            setError("out of memory while preparing the model");
            return false;
        }
        if (!prepared) {
            setError(err);
            return false;
        }
        int size = 0;
        if (classifier) {
            if (prepared->inChannels() != 3) {
                setError("a classifier must take 3 input channels (RGB)");
                return false;
            }
            if (prepared->fixedInputHeight() != prepared->fixedInputWidth()) {
                setError("classifiers with a non-square fixed input are not supported");
                return false;
            }
            size = prepared->fixedInputHeight() > 0 ? prepared->fixedInputHeight() : loaded->inputSize();
            Shape out;
            if (!prepared->outputShape(size, size, &out) || out.h != 1 || out.w != 1) {
                setError("a classifier must end in one value per class (GlobalAveragePool or a 1x1 output)");
                return false;
            }
            if (!loaded->classes().empty() && static_cast<int>(loaded->classes().size()) != out.c) {
                setError("meta.classes does not match the number of outputs");
                return false;
            }
        } else if (prepared->inChannels() != 3 || prepared->outChannels() != 3) {
            setError("the model must take 3 input channels and produce 3 output channels (RGB)");
            return false;
        }
        gpuBackend.reset();
        modelPath = path;
        model = std::move(loaded);
        exec = std::move(prepared);
        classifierSize = size;
        if (classifier) {
            gpuState = "classifiers run on the CPU";
        } else {
            initGpu(path);
        }
        setError("");
        return true;
    }

    static double now() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static void fillProbe(std::vector<float>& v, unsigned seed) {
        for (float& x : v) {
            seed = seed * 1664525u + 1013904223u;
            x = static_cast<float>(seed >> 8) * (1.0f / 16777216.0f);
        }
    }

    std::string verifyBackend(bool checkSpeed, int onlyVariant, std::string* detail, int* chosen, std::string* shortReason) {
        const int ch = exec->inChannels();
        const int h1 = 20, w1 = 24;
        std::vector<float> in(static_cast<size_t>(ch) * h1 * w1);
        fillProbe(in, 7);
        Shape os;
        if (!exec->outputShape(h1, w1, &os)) return "the model cannot run the test image";
        const size_t outN = numel(os);
        std::vector<float> cpu(outN), gpuOut(outN);
        std::string err;
        const double t0 = now();
        if (exec->run(in.data(), h1, w1, cpu.data(), effectiveThreads(), nullptr, &err) != Executor::Status::Ok) {
            return "the CPU reference run failed: " + err;
        }
        const double cpuSeconds = now() - t0;
        double range = 1.0;
        for (size_t i = 0; i < outN; ++i) range = std::max(range, static_cast<double>(std::fabs(cpu[i])));

        const bool measure = checkSpeed && onlyVariant < 0;
        const int h2 = 48, w2 = 48;
        std::vector<float> in2, out2;
        if (measure) {
            in2.assign(static_cast<size_t>(ch) * h2 * w2, 0.f);
            fillProbe(in2, 11);
            Shape os2;
            if (!exec->outputShape(h2, w2, &os2)) return "the model cannot run the speed test image";
            out2.assign(numel(os2), 0.f);
        }

        const int variants = std::max(1, gpuBackend->variantCount());
        const int lo = onlyVariant >= 0 ? onlyVariant : 0;
        const int hi = onlyVariant >= 0 ? onlyVariant + 1 : variants;
        int best = -1;
        double bestRate = 0.0, bestScore = 0.0;
        std::string problem, rates;
        bool wrongOnly = true;
        for (int v = lo; v < hi; ++v) {
            gpuBackend->setVariant(v);
            err.clear();
            const gpu::Backend::Status st = gpuBackend->run(in.data(), h1, w1, gpuOut.data(), nullptr, &err);
            if (st != gpu::Backend::Status::Ok) {
                problem = "the GPU test run failed: " + err;
                wrongOnly = false;
                if (st == gpu::Backend::Status::DeviceLost) break;
                continue;
            }
            double worst = 0.0;
            for (size_t i = 0; i < outN; ++i) {
                const double d = std::fabs(static_cast<double>(cpu[i]) - static_cast<double>(gpuOut[i]));
                if (!(d <= worst)) worst = d;
            }
            const double tolerance = gpu::variantIsHalf(v) ? 6e-3 : 2e-3;
            if (!(worst / range <= tolerance)) {
                problem = std::string("the GPU test gave wrong results (error ") + std::to_string(worst / range) + ")";
                continue;
            }
            double rate = 0.0;
            if (measure) {
                if (gpuBackend->run(in2.data(), h2, w2, out2.data(), nullptr, &err) != gpu::Backend::Status::Ok) {
                    problem = "the GPU speed test failed: " + err;
                    wrongOnly = false;
                    continue;
                }
                const double t1 = now();
                if (gpuBackend->run(in2.data(), h2, w2, out2.data(), nullptr, &err) != gpu::Backend::Status::Ok) {
                    problem = "the GPU speed test failed: " + err;
                    wrongOnly = false;
                    continue;
                }
                rate = (static_cast<double>(h2) * w2) / std::max(now() - t1, 1e-6);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%s%s %.0f", rates.empty() ? "" : ", ", gpu::convVariantName(v), rate);
                rates += buf;
            }
            const double score = gpu::variantIsHalf(v) ? rate / 1.15 : rate;
            if (best < 0 || score > bestScore) {
                best = v;
                bestRate = rate;
                bestScore = score;
            }
        }
        if (best < 0) {
            if (wrongOnly && !problem.empty()) *shortReason = "gave wrong results";
            return problem.empty() ? "no GPU kernel passed the self-test" : problem;
        }
        gpuBackend->setVariant(best);
        *chosen = best;
        if (!measure) {
            *detail = onlyVariant >= 0 ? std::string("kernel ") + gpu::convVariantName(best) + ", remembered for this device"
                                       : std::string("self-test passed");
            return std::string();
        }
        const double cpuRate = (static_cast<double>(h1) * w1) / std::max(cpuSeconds, 1e-6);
        char buf[200];
        std::snprintf(buf, sizeof(buf), "kernel %s [px/s: %s]; CPU %.0f px/s", gpu::convVariantName(best), rates.c_str(), cpuRate);
        *detail = buf;
        if (bestRate < 1.3 * cpuRate) {
            *shortReason = std::string("not faster than the CPU (") + buf + ")";
            return std::string("the GPU is not faster than the CPU on this device (") + buf + ")";
        }
        return std::string();
    }

    void finishGpuSetup(bool checkSpeed) {
        const std::string identity = gpuBackend->identity();
        const std::string cachePath = (checkSpeed && !modelPath.empty()) ? modelPath + ".gpucfg" : std::string();
        GpuConfig cfg;
        bool cached = false;
        if (!cachePath.empty()) {
            std::string text;
            cached = readTextFile(cachePath, text) && parseGpuConfig(text, cfg) &&
                     gpuConfigFresh(cfg, identity, std::max(1, gpuBackend->variantCount()), static_cast<long long>(std::time(nullptr)));
        }
        if (cached && !cfg.usable) {
            gpuBackend.reset();
            gpuState = "not used (an earlier test on this device found it " + cfg.reason + "; delete " + cachePath + " to test again)";
            return;
        }
        std::string detail, shortReason, problem;
        int chosen = 0;
        if (cached) {
            problem = verifyBackend(checkSpeed, cfg.variant, &detail, &chosen, &shortReason);
            if (!problem.empty()) cached = false;
        }
        if (!cached) {
            shortReason.clear();
            problem = verifyBackend(checkSpeed, -1, &detail, &chosen, &shortReason);
        }
        if (!problem.empty()) {
            gpuBackend.reset();
            gpuState = problem;
            if (!cachePath.empty() && !shortReason.empty()) {
                GpuConfig neg;
                neg.identity = identity;
                neg.usable = false;
                neg.reason = shortReason;
                neg.when = static_cast<long long>(std::time(nullptr));
                writeTextFile(cachePath, formatGpuConfig(neg));
            }
            return;
        }
        gpuState = "active (" + detail + "; " + gpuBackend->describe() + ")";
        if (!cachePath.empty() && !cached) {
            GpuConfig pos;
            pos.identity = identity;
            pos.usable = true;
            pos.variant = chosen;
            pos.when = static_cast<long long>(std::time(nullptr));
            writeTextFile(cachePath, formatGpuConfig(pos));
        }
    }

    void initGpu(const std::string& modelPath) {
        gpuState = "not available";
        if (std::getenv("VVA_DISABLE_GPU")) {
            gpuState = "disabled by the VVA_DISABLE_GPU environment variable";
            return;
        }
#ifdef VVA_NO_VULKAN
        gpuState = "this build has no Vulkan support";
        (void)modelPath;
#else
        const std::string marker = modelPath + ".gpuprobe";
        struct stat st;
        if (::stat(marker.c_str(), &st) == 0) {
            const double age = std::difftime(std::time(nullptr), st.st_mtime);
            if (age >= 0 && age < 24.0 * 3600.0) {
                gpuState = "skipped: an earlier GPU test did not finish (delete " + marker + " to try again)";
                return;
            }
        }
        std::FILE* f = std::fopen(marker.c_str(), "wb");
        if (!f) {
            gpuState = "skipped: cannot write " + marker;
            return;
        }
        std::fputs("vva gpu probe\n", f);
        std::fclose(f);

        std::call_once(vkOnce, [this]() { vkContext.init(); });
        if (!vkContext.ready()) {
            gpuState = "Vulkan is not available on this device";
        } else {
            gpu::VulkanHandles handles;
            handles.physicalDevice = vkContext.physicalDevice;
            handles.device = vkContext.device;
            handles.queue = vkContext.computeQueue;
            handles.queueFamily = vkContext.queueFamilyIndex;
            handles.fp16 = vkContext.fp16;
            std::string why;
            gpuBackend = gpu::createVulkanBackend(handles, *model, &why);
            if (!gpuBackend) {
                gpuState = why;
            } else {
                finishGpuSetup(true);
            }
        }
        ::unlink(marker.c_str());
#endif
    }

    size_t budgetBytes() const {
        return budgetMB > 0 ? static_cast<size_t>(budgetMB) * kMiB : currentAutoBudget();
    }

    int chooseTile(int w, int h, int pad, bool useGpu) const {
        const size_t budget = budgetBytes();
        const int longest = std::max(w, h);
        const int mult = std::max(1, exec->inputMultiple());
        auto fitsBudget = [&](int ph, int pw) {
            ph = (ph + mult - 1) / mult * mult;
            pw = (pw + mult - 1) / mult * mult;
            if (useGpu) {
                const size_t bytes = gpuBackend->estimateBytes(ph, pw);
                return bytes > 0 && bytes <= budget;
            }
            return exec->estimatePeakBytes(ph, pw) <= budget;
        };
        if (fitsBudget(h, w)) return longest;
        static const int kCandidates[] = {512, 384, 256, 192, 160, 128, 96, 80, 64, 48, 32};
        for (int t : kCandidates) {
            if (t >= longest) continue;
            const int rw = std::min(w, t + 2 * pad);
            const int rh = std::min(h, t + 2 * pad);
            if (fitsBudget(rh, rw)) return t;
        }
        return 32;
    }

    bool process(const uint8_t* in, int w, int h, uint8_t* out, DeviceType device) {
        (void)device;
        std::unique_lock<std::mutex> lock(runMutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            setError("the engine is busy");
            return false;
        }
        if (!exec) {
            setError("no model loaded");
            return false;
        }
        if (classifierSize > 0) {
            setError("this model is a classifier; use classify()");
            return false;
        }
        if (!in || !out || w < 1 || h < 1 || w > kMaxImageSide || h > kMaxImageSide) {
            setError("invalid image (size must be 1.." + std::to_string(kMaxImageSide) + " pixels per side)");
            return false;
        }
        const int s = exec->scale();
        if (static_cast<int64_t>(w) * s > (1 << 20) * 4 || static_cast<int64_t>(h) * s > (1 << 20) * 4) {
            setError("the output image would be too large");
            return false;
        }
        cancelFlag.store(false);
        usedDevice = DeviceType::CPU_NEON;
        setError("");

        const int pad = std::max(0, std::min(tilePad, 128));
        bool useGpu = gpuBackend != nullptr && device != DeviceType::CPU_NEON;
        if (device == DeviceType::GPU_VULKAN && !useGpu) {
            setError("the GPU is not available: " + gpuState);
            return false;
        }
        bool gpuFellBack = false;
        int tile = tileSize > 0 ? std::max(tileSize, 16) : chooseTile(w, h, pad, useGpu);
        if (tile >= std::max(w, h)) tile = std::max(w, h);
        if (tileSize > 0 && useGpu) {
            const int gm = std::max(1, exec->inputMultiple());
            while (tile > 32) {
                const int rw = std::min(w, tile + 2 * pad), rh = std::min(h, tile + 2 * pad);
                if (gpuBackend->fits((rh + gm - 1) / gm * gm, (rw + gm - 1) / gm * gm)) break;
                tile = std::max(32, tile * 7 / 8 / 8 * 8);
            }
        }
        const int nx = (w + tile - 1) / tile;
        const int ny = (h + tile - 1) / tile;
        const int total = nx * ny;
        const int regionW = std::min(w, tile + 2 * pad);
        const int regionH = std::min(h, tile + 2 * pad);
        const int nthreads = effectiveThreads();
        const int outW = w * s;

        const int mult = std::max(1, exec->inputMultiple());
        auto roundUp = [mult](int v) { return (v + mult - 1) / mult * mult; };
        const size_t inTileFloats = static_cast<size_t>(3) * roundUp(regionW) * roundUp(regionH);
        const size_t outTileFloats = inTileFloats * static_cast<size_t>(s) * s;
        std::unique_ptr<float[]> tileIn(new (std::nothrow) float[inTileFloats]);
        std::unique_ptr<float[]> tileOut(new (std::nothrow) float[outTileFloats]);
        if (!tileIn || !tileOut) {
            setError("out of memory; use a smaller tile size or memory budget");
            return false;
        }

        int done = 0;
        for (int ty = 0; ty < ny; ++ty) {
            for (int tx = 0; tx < nx; ++tx) {
                const int x0 = tx * tile, x1 = std::min(w, x0 + tile);
                const int y0 = ty * tile, y1 = std::min(h, y0 + tile);
                const int rx0 = std::max(0, x0 - pad), rx1 = std::min(w, x1 + pad);
                const int ry0 = std::max(0, y0 - pad), ry1 = std::min(h, y1 + pad);
                const int rw = rx1 - rx0, rh = ry1 - ry0;

                const int prw = roundUp(rw), prh = roundUp(rh);
                const size_t plane = static_cast<size_t>(prw) * prh;
                for (int y = 0; y < prh; ++y) {
                    float* r = tileIn.get() + static_cast<size_t>(y) * prw;
                    float* g = r + plane;
                    float* b = g + plane;
                    if (y < rh) {
                        const uint8_t* src = in + (static_cast<size_t>(ry0 + y) * w + rx0) * 4;
                        for (int x = 0; x < rw; ++x) {
                            r[x] = src[4 * x + 0] * (1.0f / 255.0f);
                            g[x] = src[4 * x + 1] * (1.0f / 255.0f);
                            b[x] = src[4 * x + 2] * (1.0f / 255.0f);
                        }
                        for (int x = rw; x < prw; ++x) {
                            r[x] = r[rw - 1];
                            g[x] = g[rw - 1];
                            b[x] = b[rw - 1];
                        }
                    } else {
                        const size_t last = static_cast<size_t>(rh - 1) * prw;
                        const float* r0 = tileIn.get() + last;
                        std::memcpy(r, r0, static_cast<size_t>(prw) * sizeof(float));
                        std::memcpy(g, r0 + plane, static_cast<size_t>(prw) * sizeof(float));
                        std::memcpy(b, r0 + 2 * plane, static_cast<size_t>(prw) * sizeof(float));
                    }
                }

                std::string err;
                Executor::Status st = Executor::Status::Ok;
                if (useGpu) {
                    const gpu::Backend::Status gs = gpuBackend->run(tileIn.get(), prh, prw, tileOut.get(), &cancelFlag, &err);
                    if (gs == gpu::Backend::Status::Cancelled) {
                        setError("cancelled");
                        return false;
                    }
                    if (gs != gpu::Backend::Status::Ok) {
                        gpuState = "stopped after an error: " + err;
                        gpuBackend.reset();
                        useGpu = false;
                        gpuFellBack = true;
                        err.clear();
                    }
                }
                if (!useGpu) st = exec->run(tileIn.get(), prh, prw, tileOut.get(), nthreads, &cancelFlag, &err);
                if (st != Executor::Status::Ok) {
                    setError(err.empty() ? "processing failed" : err);
                    return false;
                }

                const int orw = prw * s, orh = prh * s;
                const size_t oplane = static_cast<size_t>(orw) * orh;
                const int cx = (x0 - rx0) * s, cy = (y0 - ry0) * s;
                const int cw = (x1 - x0) * s, ch = (y1 - y0) * s;
                for (int y = 0; y < ch; ++y) {
                    const float* r = tileOut.get() + static_cast<size_t>(cy + y) * orw + cx;
                    const float* g = r + oplane;
                    const float* b = g + oplane;
                    uint8_t* dst = out + (static_cast<size_t>(y0 * s + y) * outW + static_cast<size_t>(x0) * s) * 4;
                    for (int x = 0; x < cw; ++x) {
                        dst[4 * x + 0] = static_cast<uint8_t>(clamp01(r[x]) * 255.0f + 0.5f);
                        dst[4 * x + 1] = static_cast<uint8_t>(clamp01(g[x]) * 255.0f + 0.5f);
                        dst[4 * x + 2] = static_cast<uint8_t>(clamp01(b[x]) * 255.0f + 0.5f);
                        dst[4 * x + 3] = 255;
                    }
                }

                ++done;
                if (progress && !progress(static_cast<float>(done) / static_cast<float>(total))) {
                    cancelFlag.store(true);
                    setError("cancelled");
                    return false;
                }
            }
        }

        bool opaque = true;
        const size_t pixels = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < pixels; ++i) {
            if (in[i * 4 + 3] != 255) {
                opaque = false;
                break;
            }
        }
        if (!opaque) upscaleAlpha(in, w, h, out, s);
        usedDevice = (useGpu && !gpuFellBack) ? DeviceType::GPU_VULKAN : DeviceType::CPU_NEON;
        return true;
    }

    bool classify(const uint8_t* in, int w, int h, Classification& result) {
        std::unique_lock<std::mutex> lock(runMutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            setError("the engine is busy");
            return false;
        }
        if (!exec) {
            setError("no model loaded");
            return false;
        }
        if (classifierSize <= 0) {
            setError("this model is not a classifier; use processImageTile()");
            return false;
        }
        if (!in || w < 1 || h < 1 || w > kMaxImageSide || h > kMaxImageSide) {
            setError("invalid image (size must be 1.." + std::to_string(kMaxImageSide) + " pixels per side)");
            return false;
        }
        cancelFlag.store(false);
        usedDevice = DeviceType::CPU_NEON;
        setError("");

        const int size = classifierSize;
        std::vector<uint8_t> rgb;
        resizeRgb8(in, w, h, size, rgb);
        const size_t plane = static_cast<size_t>(size) * size;
        std::vector<float> input(plane * 3);
        const float* mean = model->mean();
        const float* sd = model->stdDev();
        for (size_t i = 0; i < plane; ++i) {
            for (int c = 0; c < 3; ++c) {
                input[static_cast<size_t>(c) * plane + i] = (rgb[i * 3 + static_cast<size_t>(c)] * (1.0f / 255.0f) - mean[c]) / sd[c];
            }
        }

        const int classes = exec->outChannels();
        std::vector<float> logits(static_cast<size_t>(classes));
        std::string err;
        const Executor::Status st = exec->run(input.data(), size, size, logits.data(), effectiveThreads(), &cancelFlag, &err);
        if (st != Executor::Status::Ok) {
            setError(err.empty() ? "processing failed" : err);
            return false;
        }

        float maxLogit = logits[0];
        for (float v : logits) maxLogit = v > maxLogit ? v : maxLogit;
        double total = 0.0;
        std::vector<double> e(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            e[i] = std::exp(static_cast<double>(logits[i] - maxLogit));
            total += e[i];
        }
        result = Classification();
        result.logits = logits;
        result.probabilities.resize(logits.size());
        int best = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            result.probabilities[i] = static_cast<float>(e[i] / total);
            if (result.probabilities[i] > result.probabilities[static_cast<size_t>(best)]) best = static_cast<int>(i);
        }
        result.index = best;
        result.confidence = result.probabilities[static_cast<size_t>(best)];
        const std::vector<std::string>& names = model->classes();
        result.label = names.empty() ? "class " + std::to_string(best) : names[static_cast<size_t>(best)];
        return true;
    }
};

Engine::Engine() : pimpl(new Impl()) {}
Engine::~Engine() = default;

bool Engine::loadModel(const std::string& modelPath) {
    try {
        return pimpl->load(modelPath);
    } catch (const std::exception& e) {
        pimpl->setError(std::string("unexpected error: ") + e.what());
        return false;
    }
}

std::string Engine::lastError() const { return pimpl->getError(); }
bool Engine::modelLoaded() const { return pimpl->exec != nullptr; }
std::string Engine::modelName() const { return pimpl->model ? pimpl->model->name() : std::string(); }
int Engine::scale() const { return pimpl->exec ? pimpl->exec->scale() : 0; }
bool Engine::isClassifier() const { return pimpl->exec != nullptr && pimpl->classifierSize > 0; }
std::vector<std::string> Engine::classLabels() const {
    return pimpl->model ? pimpl->model->classes() : std::vector<std::string>();
}

bool Engine::classify(const uint8_t* inRGBA, int w, int h, Classification& result) {
    try {
        return pimpl->classify(inRGBA, w, h, result);
    } catch (const std::bad_alloc&) {
        pimpl->setError("out of memory");
        return false;
    } catch (const std::exception& e) {
        pimpl->setError(std::string("unexpected error: ") + e.what());
        return false;
    }
}

bool Engine::processImageTile(const uint8_t* inRGBA, int w, int h, uint8_t* outRGBA, DeviceType device) {
    try {
        return pimpl->process(inRGBA, w, h, outRGBA, device);
    } catch (const std::bad_alloc&) {
        pimpl->setError("out of memory; use a smaller tile size or memory budget");
        return false;
    } catch (const std::exception& e) {
        pimpl->setError(std::string("unexpected error: ") + e.what());
        return false;
    }
}

void Engine::setTileSize(int pixels) { pimpl->tileSize = std::max(0, pixels); }
void Engine::setTilePad(int pixels) { pimpl->tilePad = std::max(0, pixels); }
void Engine::setThreads(int threads) { pimpl->threads = std::max(0, threads); }
void Engine::setMemoryBudgetMB(int mb) { pimpl->budgetMB = std::max(16, mb); }
void Engine::setProgressCallback(ProgressCallback cb) { pimpl->progress = std::move(cb); }
void Engine::cancel() { pimpl->cancelFlag.store(true); }
DeviceType Engine::lastDevice() const { return pimpl->usedDevice; }
std::string Engine::gpuStatus() const { return pimpl->gpuState; }
bool Engine::gpuActive() const { return pimpl->gpuBackend != nullptr; }

std::string Engine::gpuProfile(int h, int w) {
    std::unique_lock<std::mutex> lock(pimpl->runMutex, std::try_to_lock);
    if (!lock.owns_lock()) return "the engine is busy";
    if (!pimpl->gpuBackend) return "the GPU is not active: " + pimpl->gpuState;
    return pimpl->gpuBackend->profile(h, w);
}

void Engine::adoptGpuBackendForTesting(gpu::Backend* backend, bool checkSpeed) {
    std::lock_guard<std::mutex> lock(pimpl->runMutex);
    pimpl->gpuBackend.reset(backend);
    if (!pimpl->exec || !backend) {
        pimpl->gpuState = "no model or backend";
        return;
    }
    pimpl->finishGpuSetup(checkSpeed);
}

bool Engine::hasVulkan() const {
#ifdef VVA_NO_VULKAN
    return false;
#else
    std::call_once(pimpl->vkOnce, [this]() { pimpl->vkContext.init(); });
    return pimpl->vkContext.ready();
#endif
}

std::string Engine::vulkanDeviceName() const {
#ifdef VVA_NO_VULKAN
    return std::string();
#else
    hasVulkan();
    return pimpl->vkContext.deviceName;
#endif
}

}
