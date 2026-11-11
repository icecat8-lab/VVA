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
    bool run(const Program& program, const float* in, float* out, std::string* error);

private:
    const std::vector<float>& weights_;
    std::vector<float> arena_;
    std::vector<float> weightsHalf_;
    int lastH_ = -1, lastW_ = -1;
    size_t lastBytes_ = 0;
};

}
}

#endif
