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
    int c = 0, h = 0, w = 0;
};

inline size_t numel(const Shape& s) {
    return static_cast<size_t>(s.c) * static_cast<size_t>(s.h) * static_cast<size_t>(s.w);
}

float halfToFloat(uint16_t h);

class Executor {
public:
    enum class Status { Ok, Cancelled, OutOfMemory, Error };

    static std::unique_ptr<Executor> create(const Model& model, std::string* error, bool allowAnyOutput = false);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    int inChannels() const;
    int outChannels() const;
    int scale() const;

    int fixedInputHeight() const;
    int fixedInputWidth() const;
    int inputMultiple() const;

    bool outputShape(int h, int w, Shape* out) const;

    size_t estimatePeakBytes(int h, int w) const;

    Status run(const float* in, int h, int w, float* out, int threads, const std::atomic<bool>* cancel,
               std::string* error) const;

private:
    Executor();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}

#endif
