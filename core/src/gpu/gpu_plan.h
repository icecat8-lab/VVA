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
    static std::unique_ptr<Plan> create(const Model& model, std::string* reason);
    ~Plan();
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    int inChannels() const;
    int outChannels() const;
    int scale() const;
    const std::vector<float>& weights() const;

    static void setWorkTargets(double dispatchFlops, double segmentFlops);

    bool build(int h, int w, Program& program, std::string* error, int convVariant = 0) const;

private:
    Plan();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}

#endif
