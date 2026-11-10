#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

static std::vector<uint8_t> makeImage(int w, int h) {
    std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
    unsigned s = 777u + static_cast<unsigned>(w * 31 + h);
    for (size_t i = 0; i < img.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        img[i] = (i % 4 == 3) ? 255 : static_cast<uint8_t>(s >> 24);
    }
    return img;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: engine_multiple_test model.vva scale\n");
        return 2;
    }
    const int scale = std::atoi(argv[2]);
    vva::Engine e;
    CHECK(e.loadModel(argv[1]), "model loads");
    CHECK(e.scale() == scale, "scale is read from the file");
    const int sizes[][2] = {{1, 1}, {2, 3}, {5, 7}, {37, 23}, {64, 64}, {61, 45}};
    for (const auto& sz : sizes) {
        const int w = sz[0], h = sz[1];
        std::vector<uint8_t> in = makeImage(w, h);
        std::vector<uint8_t> whole(static_cast<size_t>(w) * scale * h * scale * 4, 7);
        e.setTileSize(0);
        e.setMemoryBudgetMB(0);
        const bool ok = e.processImageTile(in.data(), w, h, whole.data(), vva::DeviceType::CPU_NEON);
        CHECK(ok, "an image whose size is not a multiple of the model's input block is processed");
        if (!ok) {
            std::printf("    %dx%d: %s\n", w, h, e.lastError().c_str());
            continue;
        }
        bool touched = false;
        for (size_t i = 3; i < whole.size(); i += 4) touched = touched || whole[i] == 255;
        CHECK(touched, "the output was written");
        if (std::max(w, h) >= 32) {
            std::vector<uint8_t> tiled(whole.size(), 9);
            e.setTileSize(16);
            e.setTilePad(16);
            CHECK(e.processImageTile(in.data(), w, h, tiled.data(), vva::DeviceType::CPU_NEON), "tiled run succeeds");
            int diff = 0;
            for (size_t i = 0; i < whole.size(); ++i) {
                if (i % 4 == 3) continue;
                const int d = std::abs(static_cast<int>(whole[i]) - static_cast<int>(tiled[i]));
                if (d > diff) diff = d;
            }
            CHECK(diff <= 1, "tiled output equals the untiled output");
            if (diff > 1) std::printf("    %dx%d max diff %d\n", w, h, diff);
            e.setTileSize(0);
        }
    }
    std::printf("engine_multiple_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
