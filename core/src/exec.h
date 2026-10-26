// CPU executor for .vva graphs.
//
// Runs the v2 core operator set (see docs/FORMAT.md) in float32, NCHW, batch 1.
// Weights are converted from fp16 once at load time. Work is split across a
// few threads; the result does not depend on the thread count.
#ifndef VVA_EXEC_H
#define VVA_EXEC_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace vva {

struct Shape {
    int c = 0, h = 0, w = 0;  // batch is always 1
};

inline size_t numel(const Shape& s) {
    return static_cast<size_t>(s.c) * static_cast<size_t>(s.h) * static_cast<size_t>(s.w);
}

// IEEE 754 half -> float (exposed for tests).
float halfToFloat(uint16_t h);

class Executor {
public:
    enum class Status { Ok, Cancelled, OutOfMemory, Error };

    // Checks that this executor can run every node of the model and prepares
    // the weights. Returns nullptr and fills `error` when it cannot.
    static std::unique_ptr<Executor> create(const Model& model, std::string* error);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    int inChannels() const;
    int outChannels() const;
    int scale() const;  // output size / input size (integer, >= 1)

    // Upper bound of the working memory (bytes) needed to run an h x w image,
    // including the input and output tensors.
    size_t estimatePeakBytes(int h, int w) const;

    // in : planar float [inChannels][h][w]
    // out: planar float [outChannels][h*scale][w*scale] (caller allocates)
    // `cancel` may be null; it is polled between operators.
    Status run(const float* in, int h, int w, float* out, int threads, const std::atomic<bool>* cancel,
               std::string* error) const;

private:
    Executor();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vva

#endif  // VVA_EXEC_H
