// Test harness: plans a model for the GPU path and replays the program on the host.
//   gpu_emu_run <model.vva> <in.f32> <C> <H> <W> <out.f32> [second_H second_W]
// With the optional second size the same emulator instance is reused for a second image of that size
// (this exercises the "zero the memory when the layout changes" rule), and only that result is written.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu/gpu_plan.h"
#include "gpu_emu.h"

int main(int argc, char** argv) {
    if (argc != 7 && argc != 9) {
        std::fprintf(stderr, "usage\n");
        return 2;
    }
    if (const char* e = std::getenv("VVA_TEST_DISPATCH_FLOPS")) vva::gpu::Plan::setWorkTargets(std::atof(e), std::atof(e) * 3.0);
    std::string err;
    std::unique_ptr<vva::Model> model = vva::Model::load(argv[1], &err);
    if (!model) {
        std::printf("LOAD_ERROR: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<vva::gpu::Plan> plan = vva::gpu::Plan::create(*model, &err);
    if (!plan) {
        std::printf("UNSUPPORTED: %s\n", err.c_str());
        return 3;
    }
    const int c = std::atoi(argv[3]);
    int h = std::atoi(argv[4]), w = std::atoi(argv[5]);
    std::vector<float> in(static_cast<size_t>(c) * h * w);
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f || std::fread(in.data(), 4, in.size(), f) != in.size()) return 1;
    std::fclose(f);

    vva::gpu::Emulator emu(plan->weights());
    std::vector<float> out;
    auto runOnce = [&](const std::vector<float>& input, int hh, int ww) {
        vva::gpu::Program prog;
        if (!plan->build(hh, ww, prog, &err)) {
            std::printf("BUILD_ERROR: %s\n", err.c_str());
            return false;
        }
        out.assign(static_cast<size_t>(prog.outChannels) * prog.outH * prog.outW, -12345.f);
        if (!emu.run(prog, input.data(), out.data(), &err)) {
            std::printf("EMU_ERROR: %s\n", err.c_str());
            return false;
        }
        std::printf("OK out=%dx%d commands=%zu arena=%zu flops=%.3g\n", prog.outH, prog.outW, prog.commands.size(),
                    prog.arenaBytes, prog.flops);
        return true;
    };
    if (argc == 9) {  // first a different-sized image with unrelated data, then the one that is compared
        const int h2 = std::atoi(argv[7]), w2 = std::atoi(argv[8]);
        std::vector<float> noise(static_cast<size_t>(c) * h2 * w2);
        for (size_t i = 0; i < noise.size(); ++i) noise[i] = static_cast<float>((i * 7919) % 1000) / 1000.f + 0.5f;
        if (!runOnce(noise, h2, w2)) return 1;
    }
    if (!runOnce(in, h, w)) return 1;
    f = std::fopen(argv[6], "wb");
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    return 0;
}
