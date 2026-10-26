#include "vva/vva.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <utility>

#include "exec.h"
#include "model.h"

#ifndef VVA_NO_VULKAN
#include "backend/vulkan_context.cpp"
#endif

namespace vva {

namespace {

constexpr size_t kMiB = 1024u * 1024u;
constexpr int kMaxImageSide = 1 << 16;

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

    int tileSize = 0;
    int tilePad = 16;
    int threads = 0;
    int budgetMB = 320;
    ProgressCallback progress;
    DeviceType usedDevice = DeviceType::CPU_NEON;

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
        std::unique_ptr<Executor> prepared;
        try {
            prepared = Executor::create(*loaded, &err);
        } catch (const std::bad_alloc&) {
            setError("out of memory while preparing the model");
            return false;
        }
        if (!prepared) {
            setError(err);
            return false;
        }
        if (prepared->inChannels() != 3 || prepared->outChannels() != 3) {
            setError("the model must take 3 input channels and produce 3 output channels (RGB)");
            return false;
        }
        model = std::move(loaded);
        exec = std::move(prepared);
        setError("");
        return true;
    }

    // Largest tile (in input pixels) whose working set fits in the budget.
    int chooseTile(int w, int h, int pad) const {
        const size_t budget = static_cast<size_t>(std::max(budgetMB, 16)) * kMiB;
        const int longest = std::max(w, h);
        if (exec->estimatePeakBytes(h, w) <= budget) return longest;
        static const int kCandidates[] = {512, 384, 256, 192, 160, 128, 96, 80, 64, 48, 32};
        for (int t : kCandidates) {
            if (t >= longest) continue;
            const int rw = std::min(w, t + 2 * pad);
            const int rh = std::min(h, t + 2 * pad);
            if (exec->estimatePeakBytes(rh, rw) <= budget) return t;
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
        int tile = tileSize > 0 ? std::max(tileSize, 16) : chooseTile(w, h, pad);
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
                const Executor::Status st = exec->run(tileIn.get(), rh, rw, tileOut.get(), nthreads, &cancelFlag, &err);
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
