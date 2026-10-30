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
#include "model.h"

#ifndef VVA_NO_VULKAN
#include "backend/vulkan_context.cpp"
#endif

namespace vva {

namespace {

constexpr size_t kMiB = 1024u * 1024u;
constexpr int kMaxImageSide = 1 << 16;

// ---- PIL-style antialiased bilinear resize (what torchvision's Resize uses on PIL images) ----

struct AxisTaps {
    std::vector<int> first;   // first source index per output index
    std::vector<int> count;   // number of taps per output index
    std::vector<float> w;     // weights, `stride` entries per output index
    int stride = 0;
};

AxisTaps bilinearTaps(int inSize, int outSize) {
    AxisTaps t;
    const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
    const double filterScale = scale > 1.0 ? scale : 1.0;
    const double support = filterScale;  // bilinear support is 1
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

// RGBA8 (w x h) -> RGB8 (size x size); the intermediate image is 8 bit, like PIL's two-pass resample.
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

inline float clamp01(float v) {
    if (!(v > 0.f)) return 0.f;  // also maps NaN to 0
    if (v > 1.f) return 1.f;
    return v;
}

// Bilinear upscaling of the alpha channel (pixel centres aligned).
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

}  // namespace

class Engine::Impl {
public:
    std::unique_ptr<Model> model;
    std::unique_ptr<Executor> exec;

    mutable std::mutex errorMutex;
    std::string error;

    std::mutex runMutex;  // one job (or model load) at a time
    std::atomic<bool> cancelFlag{false};

    int classifierSize = 0;  // square input size of a classifier model (0 for image models)

    int tileSize = 0;
    int tilePad = 16;
    int threads = 0;
    int budgetMB = 320;
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
        gpuBackend.reset();  // it points into the previous model
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

    // ---- GPU --------------------------------------------------------------------------------------

    static double now() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static void fillProbe(std::vector<float>& v, unsigned seed) {
        for (float& x : v) {
            seed = seed * 1664525u + 1013904223u;
            x = static_cast<float>(seed >> 8) * (1.0f / 16777216.0f);
        }
    }

    // Runs the backend against the CPU executor. Returns an empty string on success, otherwise the reason.
    std::string verifyBackend(bool checkSpeed, std::string* detail) {
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
        if (gpuBackend->run(in.data(), h1, w1, gpuOut.data(), nullptr, &err) != gpu::Backend::Status::Ok) {
            return "the GPU test run failed: " + err;
        }
        double worst = 0.0, range = 1.0;
        for (size_t i = 0; i < outN; ++i) {
            range = std::max(range, static_cast<double>(std::fabs(cpu[i])));
            const double d = std::fabs(static_cast<double>(cpu[i]) - static_cast<double>(gpuOut[i]));
            if (!(d <= worst)) worst = d;  // NaN counts as the worst case
        }
        const double error = worst / range;
        if (!(error <= 2e-3)) return "the GPU test gave wrong results (error " + std::to_string(error) + ")";
        if (!checkSpeed) {
            *detail = "self-test passed";
            return std::string();
        }
        // Speed: pixels per second on a larger probe for the GPU against the CPU's rate from the run above.
        const int h2 = 40, w2 = 40;
        std::vector<float> in2(static_cast<size_t>(ch) * h2 * w2);
        fillProbe(in2, 11);
        Shape os2;
        if (!exec->outputShape(h2, w2, &os2)) return "the model cannot run the speed test image";
        std::vector<float> out2(numel(os2));
        if (gpuBackend->run(in2.data(), h2, w2, out2.data(), nullptr, &err) != gpu::Backend::Status::Ok) {  // warm-up
            return "the GPU speed test failed: " + err;
        }
        const double t1 = now();
        if (gpuBackend->run(in2.data(), h2, w2, out2.data(), nullptr, &err) != gpu::Backend::Status::Ok) {
            return "the GPU speed test failed: " + err;
        }
        const double gpuSeconds = now() - t1;
        const double cpuRate = (static_cast<double>(h1) * w1) / std::max(cpuSeconds, 1e-6);
        const double gpuRate = (static_cast<double>(h2) * w2) / std::max(gpuSeconds, 1e-6);
        char buf[160];
        std::snprintf(buf, sizeof(buf), "GPU %.0f px/s, CPU %.0f px/s", gpuRate, cpuRate);
        *detail = buf;
        if (gpuRate < 1.3 * cpuRate) return std::string("the GPU is not faster than the CPU on this device (") + buf + ")";
        return std::string();
    }

    void finishGpuSetup(bool checkSpeed) {
        std::string detail;
        const std::string problem = verifyBackend(checkSpeed, &detail);
        if (!problem.empty()) {
            gpuBackend.reset();
            gpuState = problem;
        } else {
            gpuState = "active (" + detail + ")";
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
        // A marker file guards against drivers that crash during the start-up test: if the previous test never
        // finished, the GPU stays off (the marker expires after a day, or can be deleted).
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

    // Largest tile (in input pixels) whose working set fits in the budget.
    int chooseTile(int w, int h, int pad, bool useGpu) const {
        const size_t budget = static_cast<size_t>(std::max(budgetMB, 16)) * kMiB;
        const int longest = std::max(w, h);
        auto fitsBudget = [&](int ph, int pw) {
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
        (void)device;  // every device request is served by the CPU executor
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
        const int nx = (w + tile - 1) / tile;
        const int ny = (h + tile - 1) / tile;
        const int total = nx * ny;
        const int regionW = std::min(w, tile + 2 * pad);
        const int regionH = std::min(h, tile + 2 * pad);
        const int nthreads = effectiveThreads();
        const int outW = w * s;

        const size_t inTileFloats = static_cast<size_t>(3) * regionW * regionH;
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

                // RGBA8 -> planar float RGB in [0, 1]
                const size_t plane = static_cast<size_t>(rw) * rh;
                for (int y = 0; y < rh; ++y) {
                    const uint8_t* src = in + (static_cast<size_t>(ry0 + y) * w + rx0) * 4;
                    float* r = tileIn.get() + static_cast<size_t>(y) * rw;
                    float* g = r + plane;
                    float* b = g + plane;
                    for (int x = 0; x < rw; ++x) {
                        r[x] = src[4 * x + 0] * (1.0f / 255.0f);
                        g[x] = src[4 * x + 1] * (1.0f / 255.0f);
                        b[x] = src[4 * x + 2] * (1.0f / 255.0f);
                    }
                }

                std::string err;
                Executor::Status st = Executor::Status::Ok;
                if (useGpu) {
                    const gpu::Backend::Status gs = gpuBackend->run(tileIn.get(), rh, rw, tileOut.get(), &cancelFlag, &err);
                    if (gs == gpu::Backend::Status::Cancelled) {
                        setError("cancelled");
                        return false;
                    }
                    if (gs != gpu::Backend::Status::Ok) {
                        // Never fail the job because of the GPU: switch to the CPU for this and every later tile.
                        gpuState = "stopped after an error: " + err;
                        gpuBackend.reset();
                        useGpu = false;
                        gpuFellBack = true;
                        err.clear();
                    }
                }
                if (!useGpu) st = exec->run(tileIn.get(), rh, rw, tileOut.get(), nthreads, &cancelFlag, &err);
                if (st != Executor::Status::Ok) {
                    setError(err.empty() ? "processing failed" : err);
                    return false;
                }

                // Crop away the overlap and convert back to RGBA8.
                const int orw = rw * s, orh = rh * s;
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

        // Alpha: copy through when opaque, otherwise upscale it separately.
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

}  // namespace vva
