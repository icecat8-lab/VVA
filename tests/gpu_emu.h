// Host replay of a GPU Program. The kernels are the very same source files that become shaders
// (core/src/gpu/kernels/*.inc); here they are compiled as C++ with a tiny shim for the GLSL types.
// Buffer accesses are bounds-checked against the bound ranges, like a robust GPU would.
#ifndef VVA_GPU_EMU_H
#define VVA_GPU_EMU_H

#include <string>
#include <vector>

#include "gpu/gpu_types.h"

namespace vva {
namespace gpu {

class Emulator {
public:
    explicit Emulator(const std::vector<float>& weights) : weights_(weights) {}
    // in: planar [inChannels][h][w]; out: planar [outChannels][outH][outW]. Returns false with a message on any
    // out-of-range access or malformed program.
    bool run(const Program& program, const float* in, float* out, std::string* error);

private:
    const std::vector<float>& weights_;
    std::vector<float> arena_;
    int lastH_ = -1, lastW_ = -1;
    size_t lastBytes_ = 0;
};

}  // namespace gpu
}  // namespace vva

#endif
