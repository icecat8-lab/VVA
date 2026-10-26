// Test harness: runs a .vva model on raw float32 data with the production executor.
//   exec_run <model.vva> <in.f32> <C> <H> <W> <out.f32> [threads]
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "exec.h"
#include "model.h"

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: exec_run model in.f32 C H W out.f32 [threads]\n");
        return 2;
    }
    std::string err;
    std::unique_ptr<vva::Model> model = vva::Model::load(argv[1], &err);
    if (!model) {
        std::printf("LOAD_ERROR: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<vva::Executor> exec = vva::Executor::create(*model, &err);
    if (!exec) {
        std::printf("PREPARE_ERROR: %s\n", err.c_str());
        return 1;
    }
    const int c = std::atoi(argv[3]), h = std::atoi(argv[4]), w = std::atoi(argv[5]);
    const int threads = argc > 7 ? std::atoi(argv[7]) : 1;
    std::vector<float> in(static_cast<size_t>(c) * h * w);
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f || std::fread(in.data(), 4, in.size(), f) != in.size()) {
        std::printf("INPUT_ERROR\n");
        return 1;
    }
    std::fclose(f);
    if (c != exec->inChannels()) {
        std::printf("CHANNEL_ERROR: model expects %d channels\n", exec->inChannels());
        return 1;
    }
    const size_t outN = static_cast<size_t>(exec->outChannels()) * h * exec->scale() * w * exec->scale();
    std::vector<float> out(outN);
    if (exec->run(in.data(), h, w, out.data(), threads, nullptr, &err) != vva::Executor::Status::Ok) {
        std::printf("RUN_ERROR: %s\n", err.c_str());
        return 1;
    }
    f = std::fopen(argv[6], "wb");
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    std::printf("OK scale=%d out_channels=%d peak_bytes=%zu\n", exec->scale(), exec->outChannels(),
                exec->estimatePeakBytes(h, w));
    return 0;
}
