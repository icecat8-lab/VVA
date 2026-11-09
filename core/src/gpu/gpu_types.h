#ifndef VVA_GPU_TYPES_H
#define VVA_GPU_TYPES_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vva {
namespace gpu {

enum class Kernel : int { Conv3x3 = 0, Eltwise = 1, Resize = 2, Conv3x3_4oc = 3, Conv3x3_8px = 4 };

constexpr int kConvVariants = 3;
inline Kernel convKernel(int variant) {
    return variant == 1 ? Kernel::Conv3x3_4oc : (variant == 2 ? Kernel::Conv3x3_8px : Kernel::Conv3x3);
}
inline const char* convVariantName(int variant) {
    return variant == 1 ? "4px x 4ch" : (variant == 2 ? "8px x 8ch" : "4px x 8ch");
}

struct Push {
    int32_t i[8];
    float f[2];
};

struct Bind {
    int buffer = 0;
    size_t offset = 0;
    size_t size = 0;
};

struct Command {
    enum Type { kDispatch, kCopy, kBoundary } type = kDispatch;
    Kernel kernel = Kernel::Conv3x3;
    Push push{};
    Bind bind[4];
    uint32_t gx = 0, gy = 0, gz = 0;
    size_t copySrc = 0, copyDst = 0, copyBytes = 0;
    double flops = 0;
};

struct Program {
    int h = 0, w = 0, outH = 0, outW = 0;
    int inChannels = 0, outChannels = 0;
    std::vector<Command> commands;
    size_t arenaBytes = 0;
    size_t inputOffset = 0, outputOffset = 0;
    size_t inPlane = 0, outPlane = 0;
    size_t maxBindBytes = 0;
    double flops = 0;
};

inline size_t planeFloats(int h, int w) {
    const size_t p = static_cast<size_t>(h + 2) * static_cast<size_t>(w + 2);
    return (p + 63) / 64 * 64;
}

}
}

#endif
