// vva-cli: run .vva models from a terminal (Termux, adb shell, desktop Linux).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vva/vva.h"

#ifndef VVA_VERSION
#define VVA_VERSION "dev"
#endif

namespace {

void printUsage() {
    std::printf(
        "vva-cli %s\n"
        "Usage:\n"
        "  vva-cli --version                       show the version\n"
        "  vva-cli --info                          show Vulkan device information\n"
        "  vva-cli info <model.vva>                check a model and show its details\n"
        "  vva-cli run <model.vva> <in.ppm> <out.ppm> [options]   upscale with an image model\n"
        "  vva-cli classify <model.vva> <in.ppm>                  classify an image with a classifier model\n"
        "\n"
        "Options for run:\n"
        "  --tile N      tile size in input pixels (default: automatic)\n"
        "  --pad N       overlap around each tile (default 16)\n"
        "  --threads N   worker threads (default: automatic, up to 8)\n"
        "  --budget MB   working memory per tile (default 320)\n"
        "\n"
        "Images are binary PPM (P6, 8 bit). Convert from/to PNG or JPEG with any\n"
        "tool, e.g.  ffmpeg -i in.png in.ppm   /   ffmpeg -i out.ppm out.png\n",
        VVA_VERSION);
}

bool readToken(std::FILE* f, std::string& tok) {
    tok.clear();
    int c;
    for (;;) {
        c = std::fgetc(f);
        if (c == EOF) return false;
        if (c == '#') {
            while (c != '\n' && c != EOF) c = std::fgetc(f);
            continue;
        }
        if (c > ' ') break;
    }
    while (c != EOF && c > ' ') {
        tok.push_back(static_cast<char>(c));
        if (tok.size() > 32) return false;
        c = std::fgetc(f);
    }
    return true;  // the single whitespace after the token has been consumed
}

bool readPPM(const char* path, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        err = std::string("cannot open ") + path;
        return false;
    }
    std::string magic, sw, sh, smax;
    if (!readToken(f, magic) || !readToken(f, sw) || !readToken(f, sh) || !readToken(f, smax) || magic != "P6") {
        std::fclose(f);
        err = "not a binary PPM (P6) file";
        return false;
    }
    w = std::atoi(sw.c_str());
    h = std::atoi(sh.c_str());
    const int maxval = std::atoi(smax.c_str());
    if (w < 1 || h < 1 || w > 32768 || h > 32768 || maxval != 255) {
        std::fclose(f);
        err = "unsupported PPM (need 1..32768 pixels per side and maxval 255)";
        return false;
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    const size_t got = std::fread(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
    if (got != rgb.size()) {
        err = "the PPM file is truncated";
        return false;
    }
    rgba.resize(static_cast<size_t>(w) * h * 4);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        rgba[4 * i + 0] = rgb[3 * i + 0];
        rgba[4 * i + 1] = rgb[3 * i + 1];
        rgba[4 * i + 2] = rgb[3 * i + 2];
        rgba[4 * i + 3] = 255;
    }
    return true;
}

bool writePPM(const char* path, const std::vector<uint8_t>& rgba, int w, int h, std::string& err) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) {
        err = std::string("cannot write ") + path;
        return false;
    }
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::vector<uint8_t> row(static_cast<size_t>(w) * 3);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgba.data() + static_cast<size_t>(y) * w * 4;
        for (int x = 0; x < w; ++x) {
            row[3 * x + 0] = src[4 * x + 0];
            row[3 * x + 1] = src[4 * x + 1];
            row[3 * x + 2] = src[4 * x + 2];
        }
        if (std::fwrite(row.data(), 1, row.size(), f) != row.size()) {
            std::fclose(f);
            err = "write error";
            return false;
        }
    }
    if (std::fclose(f) != 0) {
        err = "write error";
        return false;
    }
    return true;
}

bool parseIntOption(int argc, char** argv, int& i, const char* name, int& out) {
    if (std::strcmp(argv[i], name) != 0) return false;
    if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(1);
    }
    out = std::atoi(argv[++i]);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    if (std::strcmp(argv[1], "--version") == 0) {
        std::printf("vva-cli %s\n", VVA_VERSION);
        return 0;
    }

    if (std::strcmp(argv[1], "--info") == 0) {
        vva::Engine engine;
        if (engine.hasVulkan()) {
            std::printf("Vulkan: available\nDevice: %s\n", engine.vulkanDeviceName().c_str());
            return 0;
        }
        std::printf("Vulkan: NOT available in this environment\n");
        return 2;
    }

    if (std::strcmp(argv[1], "info") == 0 && argc == 3) {
        vva::Engine engine;
        if (!engine.loadModel(argv[2])) {
            std::fprintf(stderr, "error: %s\n", engine.lastError().c_str());
            return 1;
        }
        if (engine.isClassifier()) {
            std::printf("model: %s\nkind: classifier (%zu named classes)\n", engine.modelName().c_str(),
                        engine.classLabels().size());
            for (const std::string& label : engine.classLabels()) std::printf("  class: %s\n", label.c_str());
        } else {
            std::printf("model: %s\nscale: x%d\n", engine.modelName().c_str(), engine.scale());
        }
        std::printf("status: OK (all operators supported)\n");
        return 0;
    }

    if (std::strcmp(argv[1], "classify") == 0 && argc == 4) {
        vva::Engine engine;
        if (!engine.loadModel(argv[2])) {
            std::fprintf(stderr, "error: %s\n", engine.lastError().c_str());
            return 1;
        }
        std::vector<uint8_t> in;
        int w = 0, h = 0;
        std::string err;
        if (!readPPM(argv[3], in, w, h, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        vva::Classification result;
        if (!engine.classify(in.data(), w, h, result)) {
            std::fprintf(stderr, "error: %s\n", engine.lastError().c_str());
            return 1;
        }
        std::printf("class: %d (%s)\nconfidence: %.4f\n", result.index, result.label.c_str(), result.confidence);
        for (size_t i = 0; i < result.probabilities.size(); ++i) {
            std::printf("  [%zu] probability %.6f  logit %.6f\n", i, result.probabilities[i], result.logits[i]);
        }
        return 0;
    }

    if (std::strcmp(argv[1], "run") == 0 && argc >= 5) {
        int tile = 0, pad = 16, threads = 0, budget = 320;
        for (int i = 5; i < argc; ++i) {
            if (parseIntOption(argc, argv, i, "--tile", tile) || parseIntOption(argc, argv, i, "--pad", pad) ||
                parseIntOption(argc, argv, i, "--threads", threads) || parseIntOption(argc, argv, i, "--budget", budget)) {
                continue;
            }
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 1;
        }

        vva::Engine engine;
        if (!engine.loadModel(argv[2])) {
            std::fprintf(stderr, "error: %s\n", engine.lastError().c_str());
            return 1;
        }
        std::vector<uint8_t> in;
        int w = 0, h = 0;
        std::string err;
        if (!readPPM(argv[3], in, w, h, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        const int s = engine.scale();
        std::vector<uint8_t> out;
        try {
            out.resize(static_cast<size_t>(w) * s * static_cast<size_t>(h) * s * 4);
        } catch (...) {
            std::fprintf(stderr, "error: not enough memory for the output image\n");
            return 1;
        }

        engine.setTileSize(tile);
        engine.setTilePad(pad);
        engine.setThreads(threads);
        engine.setMemoryBudgetMB(budget);
        engine.setProgressCallback([](float p) {
            std::fprintf(stderr, "\r%3d%%", static_cast<int>(p * 100.f + 0.5f));
            std::fflush(stderr);
            return true;
        });

        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = engine.processImageTile(in.data(), w, h, out.data(), vva::DeviceType::AUTO);
        std::fprintf(stderr, "\n");
        if (!ok) {
            std::fprintf(stderr, "error: %s\n", engine.lastError().c_str());
            return 1;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!writePPM(argv[4], out, w * s, h * s, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        std::printf("%dx%d -> %dx%d in %.2f s\n", w, h, w * s, h * s, secs);
        return 0;
    }

    printUsage();
    return 1;
}
