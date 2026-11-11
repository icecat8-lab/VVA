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
    int failOnCall = -1;
    bool deviceLost = false;
    float corrupt = 0.f;
    int badMask = 0;
    std::string identity = "emulated";
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
        ++totalRuns;
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
        if (!plan_->build(h, w, prog, &err, variant_) || !emu_.run(prog, in, out, &err)) {
            if (error) *error = err;
            return Status::Error;
        }
        if (mode_.corrupt != 0.f || (mode_.badMask & (1 << variant_))) {
            const size_t n = static_cast<size_t>(prog.outChannels) * prog.outH * prog.outW;
            for (size_t i = 0; i < n; ++i) out[i] += mode_.corrupt != 0.f ? mode_.corrupt : 0.25f;
        }
        return Status::Ok;
    }
    int variantCount() const override { return vva::gpu::kConvVariants; }
    void setVariant(int v) override { variant_ = v; }
    int variantIndex() const override { return variant_; }
    std::string describe() const override { return "emulated"; }
    std::string identity() const override { return mode_.identity; }
    std::string profile(int, int) override { return "emulated profile\n"; }
    int inChannels() const override { return plan_->inChannels(); }
    int outChannels() const override { return plan_->outChannels(); }

    int calls = 0;
    static int totalRuns;
    int variant_ = 0;
    ~EmuBackend() override { ++destroyed; }
    static int destroyed;

private:
    std::unique_ptr<vva::gpu::Plan> plan_;
    vva::gpu::Emulator emu_;
    Mode mode_;
};

int EmuBackend::destroyed = 0;
int EmuBackend::totalRuns = 0;

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

    vva::Engine ref;
    CHECK(ref.loadModel(argv[1]), "model loads");
    CHECK(!ref.gpuActive() && !ref.gpuStatus().empty(), "without Vulkan the status explains why");
    ref.setTileSize(16);
    ref.setTilePad(8);
    std::vector<uint8_t> want(outBytes);
    CHECK(ref.processImageTile(in.data(), W, H, want.data(), kAuto), "CPU reference run");
    CHECK(ref.lastDevice() == vva::DeviceType::CPU_NEON, "reference ran on the CPU");

    {
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

        const int before = b->calls;
        CHECK(e.processImageTile(in.data(), W, H, got.data(), vva::DeviceType::CPU_NEON), "forced CPU run");
        CHECK(b->calls == before && e.lastDevice() == vva::DeviceType::CPU_NEON, "CPU_NEON does not touch the GPU");
        CHECK(maxDiff(got, want) == 0, "CPU runs are identical");

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

    for (int lost = 0; lost < 2; ++lost) {
        vva::Engine e;
        e.loadModel(argv[1]);
        e.setTileSize(16);
        e.setTilePad(8);
        Mode m;
        m.failOnCall = 3 + 3;
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

    {
        vva::Engine e;
        e.loadModel(argv[1]);
        e.setTileSize(16);
        e.setTilePad(8);
        Mode m;
        m.badMask = 1 << 0;
        EmuBackend* b = attach(e, *model, m);
        CHECK(e.gpuActive() && b->variantIndex() == 1, "a wrong variant is skipped and the next correct one is used");
        std::vector<uint8_t> got(outBytes);
        CHECK(e.processImageTile(in.data(), W, H, got.data(), kAuto) && maxDiff(got, want) <= 1, "the chosen variant produces the right picture");
        vva::Engine e2;
        e2.loadModel(argv[1]);
        Mode all;
        all.badMask = (1 << vva::gpu::kConvVariants) - 1;
        attach(e2, *model, all);
        CHECK(!e2.gpuActive() && e2.gpuStatus().find("wrong results") != std::string::npos, "every variant wrong -> GPU rejected");
        for (int v = 0; v < vva::gpu::kConvVariants; ++v) {
            vva::Engine e3;
            e3.loadModel(argv[1]);
            e3.setTileSize(16);
            e3.setTilePad(8);
            Mode only;
            only.badMask = ((1 << vva::gpu::kConvVariants) - 1) & ~(1 << v);
            EmuBackend* bv = attach(e3, *model, only);
            CHECK(e3.gpuActive() && bv->variantIndex() == v, "only variant v is correct and gets chosen");
            std::vector<uint8_t> g(outBytes);
            CHECK(e3.processImageTile(in.data(), W, H, g.data(), kAuto) && maxDiff(g, want) <= (vva::gpu::variantIsHalf(v) ? 2 : 1),
                  "variant output equals the CPU picture");
        }
    }

    {
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

    {
        vva::Engine e;
        e.loadModel(argv[1]);
        std::vector<uint8_t> got(outBytes);
        CHECK(!e.processImageTile(in.data(), W, H, got.data(), vva::DeviceType::GPU_VULKAN) &&
                  e.lastError().find("GPU is not available") != std::string::npos,
              ("explicit GPU request: " + e.lastError()).c_str());
        CHECK(e.processImageTile(in.data(), W, H, got.data(), kAuto), "AUTO still works afterwards");
    }

    {
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

    {
        const std::string cfg = std::string(argv[1]) + ".gpucfg";
        std::remove(cfg.c_str());
        vva::Engine e;
        e.loadModel(argv[1]);
        const int r0 = EmuBackend::totalRuns;
        attach(e, *model, Mode(), true);
        CHECK(!e.gpuActive() && e.gpuStatus().find("not faster") != std::string::npos, ("a slow GPU is rejected: " + e.gpuStatus()).c_str());
        CHECK(EmuBackend::totalRuns > r0, "the first test really ran the GPU");
        std::FILE* f = std::fopen(cfg.c_str(), "rb");
        char buf[512] = {0};
        if (f) {
            std::fread(buf, 1, sizeof(buf) - 1, f);
            std::fclose(f);
        }
        CHECK(f != nullptr && std::strstr(buf, "usable=0") != nullptr && std::strstr(buf, "identity=emulated") != nullptr,
              "the verdict is written next to the model");

        vva::Engine e2;
        e2.loadModel(argv[1]);
        const int r1 = EmuBackend::totalRuns;
        attach(e2, *model, Mode(), true);
        CHECK(EmuBackend::totalRuns == r1, "a remembered verdict skips the GPU test entirely");
        CHECK(!e2.gpuActive() && e2.gpuStatus().find("earlier test") != std::string::npos, ("status explains: " + e2.gpuStatus()).c_str());

        vva::Engine e3;
        e3.loadModel(argv[1]);
        Mode other;
        other.identity = "another gpu|driver 0x2";
        const int r2 = EmuBackend::totalRuns;
        attach(e3, *model, other, true);
        CHECK(EmuBackend::totalRuns > r2, "a different device is tested again");
        std::remove(cfg.c_str());
    }

    {
        vva::Engine e;
        e.loadModel(argv[1]);
        attach(e, *model, Mode());
        CHECK(e.gpuStatus().find("emulated") != std::string::npos, "the status includes the device description");
        CHECK(e.gpuProfile(8, 8) == "emulated profile\n", "gpuProfile asks the backend");
        vva::Engine none;
        none.loadModel(argv[1]);
        CHECK(none.gpuProfile(8, 8).find("not active") != std::string::npos, "gpuProfile explains when the GPU is off");
    }

    {
        const int before = EmuBackend::destroyed;
        {
            vva::Engine e;
            e.loadModel(argv[1]);
            attach(e, *model, Mode());
            CHECK(e.gpuActive(), "backend adopted");
        }
        CHECK(EmuBackend::destroyed == before + 1, "the backend is destroyed together with the engine");
        {
            vva::Engine e;
            e.loadModel(argv[1]);
            Mode m;
            m.failOnCall = 2;
            attach(e, *model, m);
        }
    }

    std::printf("engine_gpu_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
