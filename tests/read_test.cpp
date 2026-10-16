// Host-side test program: loads a .vva file with the production reader and
// prints a machine-readable summary that tests/test_format.py checks.
#include <cstdio>
#include <string>

#include "model.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: read_test <file.vva>\n");
        return 2;
    }
    std::string error;
    std::unique_ptr<vva::Model> model = vva::Model::load(argv[1], &error);
    if (!model) {
        std::printf("ERROR: %s\n", error.c_str());
        return 1;
    }
    std::printf("scale %d\n", model->scale());
    std::printf("name %s\n", model->name().c_str());
    std::printf("inputs %zu\n", model->inputs().size());
    std::printf("outputs %zu\n", model->outputs().size());
    std::printf("nodes %zu\n", model->nodes().size());
    std::printf("tensors %zu\n", model->tensors().size());
    for (const vva::Node& n : model->nodes()) {
        std::printf("node %s %zu %zu\n", n.op.c_str(), n.inputs.size(), n.outputs.size());
    }
    for (const vva::TensorInfo& t : model->tensors()) {
        const uint8_t* p = model->tensorData(t);
        std::printf("tensor %s %s %llu %llu %08x aligned=%d\n", t.name.c_str(), vva::dtypeName(t.dtype),
                    static_cast<unsigned long long>(t.offset), static_cast<unsigned long long>(t.size),
                    vva::crc32(p, static_cast<size_t>(t.size)),
                    static_cast<int>(reinterpret_cast<uintptr_t>(p) % 64 == 0));
    }
    return 0;
}
