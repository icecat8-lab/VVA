// Data shared by the GPU planner, the Vulkan runtime and the host emulator used in tests.
//
// A Program is a flat list of commands (dispatches and buffer copies) for one image size. It refers to
// memory only through byte offsets into one "arena" buffer (activations) and one weights buffer, so it
// can be executed by Vulkan or replayed on the host.
#ifndef VVA_GPU_TYPES_H
#define VVA_GPU_TYPES_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vva {
namespace gpu {

enum class Kernel : int { Conv3x3 = 0, Eltwise = 1, Resize = 2 };

// Push constants, identical for every kernel (the kernels give the fields their own names).
struct Push {
    int32_t i[8];
    float f[2];
};

struct Bind {
    int buffer = 0;     // 0 = activation arena, 1 = weights
    size_t offset = 0;  // bytes, multiple of 256
    size_t size = 0;    // bytes
};

struct Command {
    enum Type { kDispatch, kCopy, kBoundary } type = kDispatch;
    Kernel kernel = Kernel::Conv3x3;
    Push push{};
    Bind bind[4];
    uint32_t gx = 0, gy = 0, gz = 0;
    size_t copySrc = 0, copyDst = 0, copyBytes = 0;  // kCopy: arena to arena
    double flops = 0;
};

struct Program {
    int h = 0, w = 0, outH = 0, outW = 0;
    int inChannels = 0, outChannels = 0;
    std::vector<Command> commands;
    size_t arenaBytes = 0;
    size_t inputOffset = 0, outputOffset = 0;  // bytes
    size_t inPlane = 0, outPlane = 0;          // floats between channel planes
    size_t maxBindBytes = 0;
    double flops = 0;
};

// Every activation plane is stored with a one-pixel zero border; planes are padded to a multiple of
// 64 floats so that tensor offsets stay 256-byte aligned and borders line up when memory is reused.
inline size_t planeFloats(int h, int w) {
    const size_t p = static_cast<size_t>(h + 2) * static_cast<size_t>(w + 2);
    return (p + 63) / 64 * 64;
}

}  // namespace gpu
}  // namespace vva

#endif
