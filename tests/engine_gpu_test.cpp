// Engine behaviour with a GPU backend: device selection, self-test, tiling, and falling back to the CPU when the
// GPU fails or lies. The backend here replays the real GPU program on the host (tests/gpu_emu.cpp).
//   engine_gpu_test <rrdbnet.vva>   (channels must be multiples of 8)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gpu/gpu_backend.h"
#include "gpu/gpu_plan.h"
#include "gpu_emu.h"
#include "vva/vva.h"

static int g_checks = 0, g_failed = 0;
#define CHECK(cond, msg)                                          \
    do {                                                          \
        ++g_checks;                                               \
        if (!(cond)) {                                            \
            ++g_failed;                                           \
            std::printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        }                                                         \
    } while (0)

using vva::gpu::Backend;

struct Mode {
    int failOnCall = -1;      // 1-based call index at which run() fails (-1: never)
    bool deviceLost = false;  // report DeviceLost instead of Error
    float corrupt = 0.f;      // added to every output value
};

class EmuBackend : public Backend {
public:
    EmuBackend(std::unique_ptr<vva::gpu::Plan> plan, Mode mode)
        : plan_(std::move(plan)), emu_(plan_->weights()), mode_(mode) {}

    bool fits(int h, int w) override { return estimateBytes(h, w) > 0; }
    size_t estimateBytes(int h, int w) override {
        vva::gpu::Program p;
        std::string err;
        return plan_->build(h, w, p, &err) ? p.arenaBytes : 0;
    }
    Status run(const float* in, int h, int w, float* out, const std::atomic<bool>* cancel, std::string* error) override {
        ++calls;
        if (cancel && cancel->load()) {
            if (error) *error = "cancelled";
            return Status::Cancelled;
        }
        if (mode_.failOnCall > 0 && calls == mode_.failOnCall) {
            if (error) *error = "simulated failure";
            return mode_.deviceLost ? Status::DeviceLost : Status::Error;
        }
        vva::gpu::Program prog;
        std::string err;
        if (!plan_->build(h, w, prog, &err) || !emu_.run(prog, in, out, &err)) {
            if (error) *error = err;
            return Status::Error;
        }
        if (mode_.corrupt != 0.f) {
            const size_t n = static_cast<size_t>(prog.outChannels) * prog.outH * prog.outW;
            for (size_t i = 0; i < n; ++i) out[i] += mode_.corrupt;
        }
        return Status::Ok;
    }
    int inChannels() const override { return plan_->inChannels(); }
    int outChannels() const override { return plan_->outChannels(); }

    int calls = 0;

private:
    std::unique_ptr<vva::gpu::Plan> plan_;
    vva::gpu::Emulator emu_;
    Mode mode_;
};

static std::vector<uint8_t> makeImage(int w, int h) {
    std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
    unsigned s = 4242u;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            s = s * 1664525u + 1013904223u;
            uint8_t* p = &img[(static_cast<size_t>(y) * w + x) * 4];
            p[0] = static_cast<uint8_t>(x * 255 / (w - 1));
            p[1] = static_cast<uint8_t>(y * 255 / (h - 1));
            p[2] = static_cast<uint8_t>(s >> 24);
            p[3] = 255;
        }
    }
    return img;
}

static int maxDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    int m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
    return m;
}

static EmuBackend* attach(vva::Engine& e, const vva::Model& model, Mode mode, bool speed = false) {
    std::string why;
    std::unique_ptr<vva::gpu::Plan> plan = vva::gpu::Plan::create(model, &why);
    if (!plan) {
        std::printf("plan failed: %s\n", why.c_str());
        std::exit(2);
    }
    EmuBackend* b = new EmuBackend(std::move(plan), mode);
    e.adoptGpuBackendForTesting(b, speed);
    return b;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::string err;
    std::unique_ptr<vva::Model> model = vva::Model::load(argv[1], &err);
    if (!model) {
        std::printf("cannot load %s\n", err.c_str());
        return 2;
    }
    const int W = 40, H = 30, S = 4;
    const size_t outBytes = static_cast<size_t>(W) * S * H * S * 4;
    const std::vector<uint8_t> in = makeImage(W, H);
    const vva::DeviceType kAuto = vva::DeviceType::AUTO;

    // CPU reference result.
    vva::Engine ref;
    CHECK(ref.loadModel(argv[1]), "model loads");
    CHECK(!ref.gpuActive() && !ref.gpuStatus().empty(), "without Vulkan the status explains why");
    ref.setTileSize(16);
    ref.setTilePad(8);
    std::vector<uint8_t> want(outBytes);
    CHECK(ref.processImageTile(in.data(), W, H, want.data(), kAuto), "CPU reference run");
    CHECK(ref.lastDevice() == vva::DeviceType::CPU_NEON, "reference ran on the CPU");

    {  // A good GPU backend is verified, adopted and used for every tile.
        vva::Engine e;
        e.loadModel(argv[1]);
        e.setTileSize(16);
        e.setTilePad(8);
        EmuBackend* b = attach(e, *model, Mode());
        CHECK(e.gpuActive() && e.gpuStatus().compare(0, 6, "active") == 0, ("good backend adopted: " + e.gpuStatus()).c_str());
        const int probeCalls = b->calls;
        std::vector<uint8_t> got(outBytes);
        CHECK(e.processImageTile(in.data(), W, H, got.data(), kAuto), "GPU run");
        CHECK(e.lastDevice() == vva::DeviceType::GPU_VULKAN, "reported as a GPU run");
        CHECK(b->calls - probeCalls == 6, "one GPU run per tile (3x2 tiles)");
        CHECK(maxDiff(got, want) <= 1, "GPU result equals the CPU result");

        // The caller can force the CPU.
        const int before = b->calls;
        CHECK(e.processImageTile(in.data(), W, H, got.data(), vva::DeviceType::CPU_NEON), "forced CPU run");
        CHECK(b->calls == before && e.lastDevice() == vva::DeviceType::CPU_NEON, "CPU_NEON does not touch the GPU");
        CHECK(maxDiff(got, want) == 0, "CPU runs are identical");

        // A small memory budget makes the engine choose smaller tiles with the GPU's own estimate.
        e.setTileSize(0);
        e.setMemoryBudgetMB(16);
        int tiles = 0;
        e.setProgressCallback([&](float) {
            ++tiles;
            return true;
        });
        CHECK(e.processImageTile(in.data(), W, H, got.data(), kAuto), "GPU run with a tiny budget");
        CHECK(tiles >= 1 && maxDiff(got, want) <= 1, "budget-sized tiles still give the same picture");
        e.setProgressCallback(nullptr);
    }

    for (int lost = 0; lost < 2; ++lost) {  // The GPU fails in the middle of a job: the job still succeeds on the CPU.
        vva::Engine e;
        e.loadModel(argv[1]);
        e.setTileSize(16);
        e.setTilePad(8);
        Mode m;
        m.failOnCall = 3 + 3;  // the self-test uses calls 1..1 (no speed check); fail on the 5th tile
        m.deviceLost = lost == 1;
        EmuBackend* b = attach(e, *model, m);
        CHECK(e.gpuActive(), "backend adopted before the failure");
        std::vector<uint8_t> got(outBytes);
        CHECK(e.processImageTile(in.data(), W, H, got.data(), kAuto), "the job survives a GPU failure");
        CHECK(!e.gpuActive() && e.gpuStatus().find("stopped after an error") != std::string::npos,
              ("status after the failure: " + e.gpuStatus()).c_str());
        CHECK(e.lastDevice() == vva::DeviceType::CPU_NEON, "a job that fell back reports the CPU");
        CHECK(maxDiff(got, want) <= 1, "result is still correct after the fallback");
        (void)b;
        std::vector<uint8_t> again(outBytes);
        CHECK(e.processImageTile(in.data(), W, H, again.data(), kAuto) && maxDiff(again, want) <= 1, "later jobs run on the CPU");
    }

    {  // A GPU that returns wrong numbers is rejected by the self-test.
        vva::Engine e;
        e.loadModel(argv[1]);
        Mode m;
        m.corrupt = 0.25f;
        attach(e, *model, m);
        CHECK(!e.gpuActive() && e.gpuStatus().find("wrong results") != std::string::npos, ("wrong results are caught: " + e.gpuStatus()).c_str());
        std::vector<uint8_t> got(outBytes);
        e.setTileSize(16);
        e.setTilePad(8);
        CHECK(e.processImageTile(in.data(), W, H, got.data(), kAuto) && maxDiff(got, want) <= 1, "processing continues on the CPU");
    }

    {  // Explicit GPU request without a usable GPU fails clearly.
        vva::Engine e;
        e.loadModel(argv[1]);
        std::vector<uint8_t> got(outBytes);
        CHECK(!e.processImageTile(in.data(), W, H, got.data(), vva::DeviceType::GPU_VULKAN) &&
                  e.lastError().find("GPU is not available") != std::string::npos,
              ("explicit GPU request: " + e.lastError()).c_str());
        CHECK(e.processImageTile(in.data(), W, H, got.data(), kAuto), "AUTO still works afterwards");
    }

    {  // Cancelling while the GPU works.
        vva::Engine e;
        e.loadModel(argv[1]);
        e.setTileSize(16);
        e.setTilePad(8);
        attach(e, *model, Mode());
        int calls = 0;
        e.setProgressCallback([&](float) { return ++calls < 2; });
        std::vector<uint8_t> got(outBytes);
        CHECK(!e.processImageTile(in.data(), W, H, got.data(), kAuto) && e.lastError() == "cancelled", "cancel with the GPU");
        e.setProgressCallback(nullptr);
        CHECK(e.processImageTile(in.data(), W, H, got.data(), kAuto) && maxDiff(got, want) <= 1, "usable after a cancel");
        CHECK(e.gpuActive(), "a cancel does not disable the GPU");
    }

    std::printf("engine_gpu_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
