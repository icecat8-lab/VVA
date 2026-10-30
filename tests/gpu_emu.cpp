#include "gpu_emu.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vva {
namespace gpu {
namespace {

// ---- GLSL shim ---------------------------------------------------------------------------------------

struct vec4 {
    float x, y, z, w;
    vec4() : x(0), y(0), z(0), w(0) {}
    vec4(float s) : x(s), y(s), z(s), w(s) {}
};
inline vec4 operator+(const vec4& a, const vec4& b) {
    vec4 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    r.w = a.w + b.w;
    return r;
}
inline vec4 operator*(float s, const vec4& a) {
    vec4 r;
    r.x = s * a.x;
    r.y = s * a.y;
    r.z = s * a.z;
    r.w = s * a.w;
    return r;
}
inline vec4& operator+=(vec4& a, const vec4& b) {
    a = a + b;
    return a;
}

struct uvec3 {
    unsigned x, y, z;
};

struct PushView {
    int i0, i1, i2, i3, i4, i5, i6, i7;
    float f0, f1;
};

template <class T>
struct Buf {
    T* ptr = nullptr;
    size_t count = 0;
    T& operator[](int i) const {
        if (i < 0 || static_cast<size_t>(i) >= count) {
            throw std::out_of_range("buffer access outside the bound range: index " + std::to_string(i) + " of " + std::to_string(count));
        }
        return ptr[i];
    }
};

Buf<const float> src, res, srcA, srcB;
Buf<const vec4> wgt;
Buf<float> dst;
PushView p;
uvec3 gl_GlobalInvocationID;

#include "kernels/kernel_common.inc"

// ---- convolution ---------------------------------------------------------------------------------------
#define P_CIN p.i0
#define P_H p.i1
#define P_W p.i2
#define P_BIAS p.i3
#define P_YBASE p.i4
#define P_HASRES p.i5
#define P_PLANE p.i6
#define P_ALPHA p.f0
#define P_SCALE p.f1
#include "kernels/kernel_conv_out.inc"
void emuConv() {
#include "kernels/kernel_conv3x3.inc"
}
#undef P_CIN
#undef P_H
#undef P_W
#undef P_BIAS
#undef P_YBASE
#undef P_HASRES
#undef P_PLANE
#undef P_ALPHA
#undef P_SCALE

// ---- element-wise ---------------------------------------------------------------------------------------
#define P_COUNT p.i0
#define P_HASB p.i1
#define P_GRIDW p.i2
#define P_ALPHA p.f0
#define P_SCALE p.f1
void emuElt() {
#include "kernels/kernel_eltwise.inc"
}
#undef P_COUNT
#undef P_HASB
#undef P_GRIDW
#undef P_ALPHA
#undef P_SCALE

// ---- resize ---------------------------------------------------------------------------------------------
#define P_H p.i0
#define P_W p.i1
#define P_SH p.i2
#define P_SW p.i3
#define P_PLANEIN p.i4
#define P_PLANEOUT p.i5
void emuResize() {
#include "kernels/kernel_resize.inc"
}
#undef P_H
#undef P_W
#undef P_SH
#undef P_SW
#undef P_PLANEIN
#undef P_PLANEOUT

}  // namespace

bool Emulator::run(const Program& prog, const float* in, float* out, std::string* error) {
    auto fail = [&](const std::string& m) {
        if (error) *error = m;
        return false;
    };
    try {
        // Memory is zeroed whenever the layout changes, because borders rely on never-written zeros.
        if (prog.h != lastH_ || prog.w != lastW_ || prog.arenaBytes != lastBytes_) {
            arena_.assign(prog.arenaBytes / sizeof(float), 0.f);
            lastH_ = prog.h;
            lastW_ = prog.w;
            lastBytes_ = prog.arenaBytes;
        }
        const size_t inW = static_cast<size_t>(prog.w) + 2;
        float* inBase = arena_.data() + prog.inputOffset / sizeof(float);
        for (int c = 0; c < prog.inChannels; ++c) {
            for (int y = 0; y < prog.h; ++y) {
                std::memcpy(inBase + static_cast<size_t>(c) * prog.inPlane + (static_cast<size_t>(y) + 1) * inW + 1,
                            in + (static_cast<size_t>(c) * prog.h + y) * prog.w, static_cast<size_t>(prog.w) * sizeof(float));
            }
        }

        for (const Command& cmd : prog.commands) {
            if (cmd.type == Command::kBoundary) continue;
            if (cmd.type == Command::kCopy) {
                if (cmd.copySrc + cmd.copyBytes > prog.arenaBytes || cmd.copyDst + cmd.copyBytes > prog.arenaBytes) {
                    return fail("copy outside the arena");
                }
                std::memmove(reinterpret_cast<char*>(arena_.data()) + cmd.copyDst,
                             reinterpret_cast<const char*>(arena_.data()) + cmd.copySrc, cmd.copyBytes);
                continue;
            }
            // Bind buffers.
            Buf<const float>* floatReads[4] = {&src, nullptr, nullptr, &res};
            (void)floatReads;
            auto arenaRange = [&](const Bind& b, bool& ok) {
                ok = b.buffer == 0 && b.offset % 256 == 0 && b.offset + b.size <= prog.arenaBytes;
                return arena_.data() + b.offset / sizeof(float);
            };
            bool ok0 = true, ok1 = true, ok2 = true, ok3 = true;
            float* a0 = arenaRange(cmd.bind[0], ok0);
            float* a2 = arenaRange(cmd.bind[2], ok2);
            float* a3 = arenaRange(cmd.bind[3], ok3);
            if (!ok0 || !ok2 || !ok3) return fail("bad arena binding");
            src = {a0, cmd.bind[0].size / sizeof(float)};
            srcA = src;
            res = {a3, cmd.bind[3].size / sizeof(float)};
            dst = {a2, cmd.bind[2].size / sizeof(float)};
            if (cmd.kernel == Kernel::Conv3x3) {
                const Bind& wb = cmd.bind[1];
                ok1 = wb.buffer == 1 && wb.offset % 256 == 0 && wb.offset + wb.size <= weights_.size() * sizeof(float);
                if (!ok1) return fail("bad weights binding");
                wgt = {reinterpret_cast<const vec4*>(weights_.data() + wb.offset / sizeof(float)), wb.size / sizeof(vec4)};
            } else {
                float* a1 = arenaRange(cmd.bind[1], ok1);
                if (!ok1) return fail("bad arena binding");
                srcB = {a1, cmd.bind[1].size / sizeof(float)};
            }
            std::memcpy(&p, &cmd.push, sizeof(p));
            if (sizeof(p) != sizeof(cmd.push)) return fail("push constant layout mismatch");
            const unsigned lx = cmd.kernel == Kernel::Eltwise ? 64u : 16u;
            const unsigned ly = cmd.kernel == Kernel::Eltwise ? 1u : 4u;
            for (unsigned gz = 0; gz < cmd.gz; ++gz) {
                for (unsigned gy = 0; gy < cmd.gy; ++gy) {
                    for (unsigned gx = 0; gx < cmd.gx; ++gx) {
                        for (unsigned iy = 0; iy < ly; ++iy) {
                            for (unsigned ix = 0; ix < lx; ++ix) {
                                gl_GlobalInvocationID = {gx * lx + ix, gy * ly + iy, gz};
                                if (cmd.kernel == Kernel::Conv3x3) emuConv();
                                else if (cmd.kernel == Kernel::Eltwise) emuElt();
                                else emuResize();
                            }
                        }
                    }
                }
            }
        }

        const size_t outW = static_cast<size_t>(prog.outW) + 2;
        const float* outBase = arena_.data() + prog.outputOffset / sizeof(float);
        for (int c = 0; c < prog.outChannels; ++c) {
            for (int y = 0; y < prog.outH; ++y) {
                std::memcpy(out + (static_cast<size_t>(c) * prog.outH + y) * prog.outW,
                            outBase + static_cast<size_t>(c) * prog.outPlane + (static_cast<size_t>(y) + 1) * outW + 1,
                            static_cast<size_t>(prog.outW) * sizeof(float));
            }
        }
        return true;
    } catch (const std::exception& e) {
        return fail(e.what());
    }
}

}  // namespace gpu
}  // namespace vva
