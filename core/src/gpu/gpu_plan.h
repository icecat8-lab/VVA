// GPU execution planning (no Vulkan here, so it can be tested on the host).
//
// Plan::create() converts a model into a list of GPU operators when every node is supported:
//   Conv2D 3x3 (stride 1, padding 1, no groups), LeakyReLU, ReLU, Add, Mul by a scalar, Concat (channels),
//   ResizeNearest. Activations after a convolution are fused into it (LeakyReLU, scalar scale, residual add).
// Plan::build() then lays the operators out for one image size and emits a Program.
#ifndef VVA_GPU_PLAN_H
#define VVA_GPU_PLAN_H

#include <memory>
#include <string>
#include <vector>

#include "../model.h"
#include "gpu_types.h"

namespace vva {
namespace gpu {

class Plan {
public:
    // Returns nullptr and fills `reason` when the model cannot run on the GPU path.
    static std::unique_ptr<Plan> create(const Model& model, std::string* reason);
    ~Plan();
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    int inChannels() const;
    int outChannels() const;
    int scale() const;
    const std::vector<float>& weights() const;  // packed constants shared by all programs

    // Work per dispatch and per submit, in FLOPs. Only tests change the defaults (1e9 and 4e9).
    static void setWorkTargets(double dispatchFlops, double segmentFlops);

    // Lays the model out for an h x w input.
    bool build(int h, int w, Program& program, std::string* error) const;

private:
    Plan();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gpu
}  // namespace vva

#endif
