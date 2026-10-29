// API-level tests of vva::Engine (tiling, alpha, progress, cancel, errors).
//   engine_test <rrdbnet.vva> <bad_model.vva>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vva/vva.h"

static int g_checks = 0, g_failed = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        ++g_checks;                                                   \
        if (!(cond)) {                                                \
            ++g_failed;                                               \
            std::printf("  FAIL: %s (line %d)\n", msg, __LINE__);    \
        }                                                             \
    } while (0)

static std::vector<uint8_t> makeImage(int w, int h, int alphaMode) {
    std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
    unsigned s = 12345u;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            s = s * 1664525u + 1013904223u;
            uint8_t* p = &img[(static_cast<size_t>(y) * w + x) * 4];
            p[0] = static_cast<uint8_t>((x * 255) / (w > 1 ? w - 1 : 1));
            p[1] = static_cast<uint8_t>((y * 255) / (h > 1 ? h - 1 : 1));
            p[2] = static_cast<uint8_t>(s >> 24);
            p[3] = alphaMode == 0 ? 255 : (alphaMode == 1 ? 100 : static_cast<uint8_t>(x * 255 / (w > 1 ? w - 1 : 1)));
        }
    }
    return img;
}

static int maxRgbDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    int m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (i % 4 == 3) continue;
        const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: engine_test model.vva bad_model.vva\n");
        return 2;
    }
    const vva::DeviceType dev = vva::DeviceType::AUTO;

    {  // no model loaded
        vva::Engine e;
        std::vector<uint8_t> in = makeImage(4, 4, 0), out(4 * 4 * 4);
        CHECK(!e.modelLoaded() && e.scale() == 0, "no model: scale is 0");
        CHECK(!e.processImageTile(in.data(), 4, 4, out.data(), dev), "processing without a model fails");
        CHECK(e.lastError() == "no model loaded", "error says no model loaded");
        CHECK(!e.loadModel("/nonexistent/model.vva") && !e.lastError().empty(), "missing file is reported");
        CHECK(!e.loadModel(argv[2]) && !e.lastError().empty(), "bad model is rejected with a message");
    }

    vva::Engine e;
    CHECK(e.loadModel(argv[1]), "model loads");
    CHECK(e.modelLoaded() && e.scale() == 4 && e.modelName() == "test-model", "scale and name");
    CHECK(!e.loadModel(argv[2]) && e.modelLoaded() && e.scale() == 4, "a failed load keeps the old model");
    CHECK(!e.isClassifier() && e.classLabels().empty(), "an upscaler is not a classifier");
    {
        vva::Classification c;
        std::vector<uint8_t> px(4 * 4 * 4, 128);
        CHECK(!e.classify(px.data(), 4, 4, c) && e.lastError() == "this model is not a classifier; use processImageTile()",
              "classify() on an upscaler fails with an explanation");
    }

    // Wider than the network's receptive field (~34 px) so that tiling really happens.
    const int W = 120, H = 20, S = 4;
    const size_t outBytes = static_cast<size_t>(W) * S * H * S * 4;

    // Whole image vs tiles with enough overlap -> same picture.
    std::vector<uint8_t> in = makeImage(W, H, 0);
    std::vector<uint8_t> whole(outBytes), tiled(outBytes), onThread(outBytes);
    e.setThreads(1);
    CHECK(e.processImageTile(in.data(), W, H, whole.data(), dev), "whole image");
    CHECK(e.lastDevice() == vva::DeviceType::CPU_NEON, "ran on CPU");
    bool anyNonOpaque = false;
    for (size_t i = 3; i < whole.size(); i += 4) anyNonOpaque |= whole[i] != 255;
    CHECK(!anyNonOpaque, "opaque input gives opaque output");

    e.setTileSize(16);
    e.setTilePad(40);
    CHECK(e.processImageTile(in.data(), W, H, tiled.data(), dev), "tiled image");
    CHECK(maxRgbDiff(whole, tiled) <= 1, "tiling with a wide overlap matches the whole image");

    e.setTileSize(0);
    e.setThreads(4);
    CHECK(e.processImageTile(in.data(), W, H, onThread.data(), dev), "4 threads");
    CHECK(std::memcmp(whole.data(), onThread.data(), outBytes) == 0, "result does not depend on thread count");

    // Progress: one call per tile, increasing, ending at 1.
    {
        e.setTileSize(16);
        e.setTilePad(16);
        std::vector<float> seen;
        e.setProgressCallback([&](float p) {
            seen.push_back(p);
            return true;
        });
        std::vector<uint8_t> out(outBytes);
        CHECK(e.processImageTile(in.data(), W, H, out.data(), dev), "progress run");
        const size_t tiles = static_cast<size_t>((W + 15) / 16) * ((H + 15) / 16);
        CHECK(seen.size() == tiles, "one progress call per tile");
        bool increasing = true;
        for (size_t i = 1; i < seen.size(); ++i) increasing &= seen[i] > seen[i - 1];
        CHECK(increasing && !seen.empty() && seen.back() == 1.0f, "progress increases to 1.0");

        // Cancel through the callback.
        int calls = 0;
        e.setProgressCallback([&](float) { return ++calls < 2; });
        CHECK(!e.processImageTile(in.data(), W, H, out.data(), dev), "callback can cancel");
        CHECK(e.lastError() == "cancelled" && calls == 2, "cancel reported after the second tile");

        // Cancel through Engine::cancel() while running (executor notices at the next operator).
        calls = 0;
        e.setProgressCallback([&](float) {
            if (++calls == 1) e.cancel();
            return true;
        });
        CHECK(!e.processImageTile(in.data(), W, H, out.data(), dev), "cancel() stops the job");
        CHECK(e.lastError() == "cancelled", "cancel() reported");

        // The engine is usable again afterwards.
        e.setProgressCallback(nullptr);
        CHECK(e.processImageTile(in.data(), W, H, out.data(), dev), "works again after a cancel");
    }

    // Memory budget forces tiling on large inputs (tiny model, 16 MB minimum budget).
    {
        e.setTileSize(0);
        e.setTilePad(16);
        e.setThreads(2);
        int calls = 0;
        e.setProgressCallback([&](float) {
            ++calls;
            return true;
        });
        const int bw = 150, bh = 140;
        std::vector<uint8_t> big = makeImage(bw, bh, 0), out(static_cast<size_t>(bw) * S * bh * S * 4);
        e.setMemoryBudgetMB(16);
        CHECK(e.processImageTile(big.data(), bw, bh, out.data(), dev), "large image with a small budget");
        CHECK(calls > 1, "a small budget splits the image into tiles");
        const int tilesSmall = calls;
        calls = 0;
        e.setMemoryBudgetMB(4096);
        CHECK(e.processImageTile(big.data(), bw, bh, out.data(), dev), "large image with a big budget");
        CHECK(calls == 1 && tilesSmall > 1, "a big budget keeps one tile");
        e.setProgressCallback(nullptr);
        e.setMemoryBudgetMB(320);
        e.setThreads(1);
    }

    // Alpha: constant stays constant, a gradient stays monotonic and in range.
    {
        e.setTileSize(0);
        std::vector<uint8_t> half = makeImage(W, H, 1), out(outBytes);
        CHECK(e.processImageTile(half.data(), W, H, out.data(), dev), "constant alpha");
        bool constant = true;
        for (size_t i = 3; i < out.size(); i += 4) constant &= out[i] == 100;
        CHECK(constant, "constant alpha stays constant");

        std::vector<uint8_t> grad = makeImage(W, H, 2);
        CHECK(e.processImageTile(grad.data(), W, H, out.data(), dev), "gradient alpha");
        bool mono = true;
        const int ow = W * S;
        for (int y = 0; y < H * S; ++y) {
            for (int x = 1; x < ow; ++x) mono &= out[(static_cast<size_t>(y) * ow + x) * 4 + 3] >= out[(static_cast<size_t>(y) * ow + x - 1) * 4 + 3];
        }
        CHECK(mono, "alpha gradient stays monotonic");
        CHECK(out[3] == 0 && out[(ow - 1) * 4 + 3] == 255, "alpha ends keep their values");
    }

    // Tiny images and argument checks.
    {
        std::vector<uint8_t> one(4, 200), out(S * S * 4);
        e.setTileSize(0);
        CHECK(e.processImageTile(one.data(), 1, 1, out.data(), dev), "1x1 image");
        CHECK(!e.processImageTile(one.data(), 0, 1, out.data(), dev), "zero width rejected");
        CHECK(!e.processImageTile(nullptr, 1, 1, out.data(), dev), "null input rejected");
        CHECK(!e.processImageTile(one.data(), 1, 1, nullptr, dev), "null output rejected");
        CHECK(!e.processImageTile(one.data(), 100000, 1, out.data(), dev), "absurd width rejected");
    }

    std::printf("engine_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
