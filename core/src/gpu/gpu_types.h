#ifndef VVA_GPU_TYPES_H
#define VVA_GPU_TYPES_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vva {
namespace gpu {

enum class Kernel : int {
    Conv3x3 = 0, Eltwise = 1, Resize = 2, Conv3x3_4oc = 3, Conv3x3_8px = 4,
    Conv3x3_F16 = 5, Conv3x3_F16Math = 6, Eltwise_F16 = 7, Resize_F16 = 8
};

constexpr int kKernelCount = 9;
constexpr int kConvVariants = 5;

inline bool variantIsHalf(int variant) { return variant >= 3; }
inline int variantElemBytes(int variant) { return variantIsHalf(variant) ? 2 : 4; }

inline Kernel convKernel(int variant) {
    switch (variant) {
        case 1: return Kernel::Conv3x3_4oc;
        case 2: return Kernel::Conv3x3_8px;
        case 3: return Kernel::Conv3x3_F16;
        case 4: return Kernel::Conv3x3_F16Math;
        default: return Kernel::Conv3x3;
    }
}
inline bool isConvKernel(Kernel k) {
    return k == Kernel::Conv3x3 || k == Kernel::Conv3x3_4oc || k == Kernel::Conv3x3_8px || k == Kernel::Conv3x3_F16 ||
           k == Kernel::Conv3x3_F16Math;
}
inline const char* convVariantName(int variant) {
    switch (variant) {
        case 1: return "4px x 4ch";
        case 2: return "8px x 8ch";
        case 3: return "fp16 storage, 4px x 8ch";
        case 4: return "fp16 math, 4px x 8ch";
        default: return "4px x 8ch";
    }
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
    int elemBytes = 4;
    size_t inputOffset = 0, outputOffset = 0;
    size_t inPlane = 0, outPlane = 0;
    size_t maxBindBytes = 0;
    double flops = 0;
};

inline size_t planeFloats(int h, int w, int elemBytes = 4) {
    const size_t p = static_cast<size_t>(h + 2) * static_cast<size_t>(w + 2);
    const size_t unit = static_cast<size_t>(256 / elemBytes);
    return (p + unit - 1) / unit * unit;
}

inline uint16_t floatToHalf(float f) {
    uint32_t x;
    static_assert(sizeof(x) == sizeof(f), "float must be 32 bits");
    __builtin_memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t abs = x & 0x7fffffffu;
    if (abs >= 0x7f800000u) return static_cast<uint16_t>(sign | (abs > 0x7f800000u ? 0x7e00u : 0x7bffu));
    if (abs >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7bffu);
    if (abs < 0x33000001u) return static_cast<uint16_t>(sign);
    if (abs < 0x38800000u) {
        const uint32_t mant = (abs & 0x7fffffu) | 0x800000u;
        const int shift = 126 - static_cast<int>(abs >> 23);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = ((abs >> 13) - 0x1c000u);
    const uint32_t rem = abs & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
}

inline float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 113;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

inline float roundToHalf(float f) { return halfToFloat(floatToHalf(f)); }

}
}

#endif
