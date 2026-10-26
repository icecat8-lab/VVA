#include "exec.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define VVA_RESTRICT __restrict__
#else
#define VVA_RESTRICT
#endif

namespace vva {

float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal half: normalise into a float
            int e = -1;
            do {
                ++e;
                man <<= 1;
            } while ((man & 0x400u) == 0);
            man &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

namespace {

enum class Op { Conv, Add, Mul, MulScalar, LeakyReLU, ReLU, Sigmoid, Clip, Concat, Resize, PixelShuffle };

struct Step {
    Op op = Op::Add;
    std::string label;  // "node #3 (Conv2D)" for messages
    std::vector<int> in;
    int out = -1;

    // Conv2D
    int outC = 0, inC = 0, kh = 0, kw = 0, sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1, groups = 1;
    bool fast = false;
    std::vector<float> weights;  // fast: packed per output group, generic: OIHW
    std::vector<float> bias;
    std::vector<int> groupStart, groupSize;
    std::vector<size_t> groupOffset;

    // Other operators
    float f0 = 0.f, f1 = 0.f;  // alpha / scalar / clip min,max
    int i0 = 1, i1 = 1;        // resize scale or pixel-shuffle factor

    bool inplace = false;  // output reuses the buffer of input `inplaceArg`
    int inplaceArg = 0;
    std::vector<int> release;  // values that can be freed after this step
};

struct Buf {
    std::unique_ptr<float[]> own;
    float* mut = nullptr;          // writable view (null for the borrowed input)
    const float* data = nullptr;   // readable view
    Shape shape;
};

bool failMsg(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

template <class F>
void parallelFor(int threads, int n, F&& f) {
    if (n <= 0) return;
    threads = std::min(threads, n);
    if (threads <= 1) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<int> next(0);
    auto worker = [&]() {
        for (;;) {
            const int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            f(i);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads - 1));
    for (int t = 0; t < threads - 1; ++t) {
        try {
            pool.emplace_back(worker);
        } catch (...) {
            break;  // could not start a thread: the remaining workers still finish all items
        }
    }
    worker();
    for (std::thread& t : pool) t.join();
}

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------

constexpr int kMaxGroupOut = 4;  // output channels computed together

// Computes NO output channels x NX consecutive output pixels.
//   p0       : padded input, positioned at (channel 0, row y, column x0)
//   wp       : weights packed as [ic][ky][kx][NO]
//   dst      : output positioned at (first channel, y, x0); channel stride oStride
template <int NO, int NX>
inline void convBlock(const float* VVA_RESTRICT p0, size_t planeStride, size_t rowStride, int inC, int kh, int kw,
                      const float* VVA_RESTRICT wp, const float* VVA_RESTRICT bias, float* VVA_RESTRICT dst,
                      size_t oStride) {
    float acc[NO][NX];
    for (int o = 0; o < NO; ++o) {
        for (int x = 0; x < NX; ++x) acc[o][x] = bias[o];
    }
    const float* w = wp;
    for (int ic = 0; ic < inC; ++ic) {
        const float* pl = p0 + static_cast<size_t>(ic) * planeStride;
        for (int ky = 0; ky < kh; ++ky) {
            const float* pr = pl + static_cast<size_t>(ky) * rowStride;
            for (int kx = 0; kx < kw; ++kx) {
                const float* src = pr + kx;
                for (int o = 0; o < NO; ++o) {
                    const float wv = w[o];
                    for (int x = 0; x < NX; ++x) acc[o][x] += wv * src[x];
                }
                w += NO;
            }
        }
    }
    for (int o = 0; o < NO; ++o) {
        float* d = dst + static_cast<size_t>(o) * oStride;
        for (int x = 0; x < NX; ++x) d[x] = acc[o][x];
    }
}

template <int NO>
void convRow(const float* pRow, size_t planeStride, size_t rowStride, int inC, int kh, int kw, const float* wp,
             const float* bias, float* dstRow, size_t oStride, int wo) {
    int x = 0;
    for (; x + 16 <= wo; x += 16) {
        convBlock<NO, 16>(pRow + x, planeStride, rowStride, inC, kh, kw, wp, bias, dstRow + x, oStride);
    }
    for (; x + 4 <= wo; x += 4) {
        convBlock<NO, 4>(pRow + x, planeStride, rowStride, inC, kh, kw, wp, bias, dstRow + x, oStride);
    }
    for (; x < wo; ++x) {
        convBlock<NO, 1>(pRow + x, planeStride, rowStride, inC, kh, kw, wp, bias, dstRow + x, oStride);
    }
}

void packFastWeights(Step& st, const std::vector<float>& oihw) {
    st.weights.assign(oihw.size(), 0.f);
    size_t offset = 0;
    int o0 = 0;
    while (o0 < st.outC) {
        const int left = st.outC - o0;
        const int n = left >= 4 ? 4 : (left >= 2 ? 2 : 1);
        st.groupStart.push_back(o0);
        st.groupSize.push_back(n);
        st.groupOffset.push_back(offset);
        for (int ic = 0; ic < st.inC; ++ic) {
            for (int ky = 0; ky < st.kh; ++ky) {
                for (int kx = 0; kx < st.kw; ++kx) {
                    for (int o = 0; o < n; ++o) {
                        const size_t src = (static_cast<size_t>(o0 + o) * st.inC + ic) * st.kh * st.kw +
                                           static_cast<size_t>(ky) * st.kw + kx;
                        st.weights[offset++] = oihw[src];
                    }
                }
            }
        }
        o0 += n;
    }
}

void runConvFast(const Step& st, const Buf& in, Buf& out, int threads) {
    const int H = in.shape.h, W = in.shape.w;
    const int Ho = out.shape.h, Wo = out.shape.w;
    const int Hp = H + 2 * st.ph, Wp = W + 2 * st.pw;

    std::unique_ptr<float[]> padded;
    const float* P = in.data;
    size_t planeStride = static_cast<size_t>(H) * W;
    size_t rowStride = static_cast<size_t>(W);
    if (st.ph != 0 || st.pw != 0) {
        planeStride = static_cast<size_t>(Hp) * Wp;
        rowStride = static_cast<size_t>(Wp);
        padded.reset(new float[planeStride * st.inC]);
        float* pbase = padded.get();
        parallelFor(threads, st.inC, [&](int c) {
            float* dst = pbase + static_cast<size_t>(c) * planeStride;
            const float* src = in.data + static_cast<size_t>(c) * H * W;
            std::memset(dst, 0, planeStride * sizeof(float));
            for (int y = 0; y < H; ++y) {
                std::memcpy(dst + static_cast<size_t>(y + st.ph) * Wp + st.pw, src + static_cast<size_t>(y) * W,
                            static_cast<size_t>(W) * sizeof(float));
            }
        });
        P = pbase;
    }

    const size_t oStride = static_cast<size_t>(Ho) * Wo;
    const int numGroups = static_cast<int>(st.groupStart.size());
    const int tasks = numGroups * Ho;
    parallelFor(threads, tasks, [&](int t) {
        const int g = t / Ho;
        const int y = t % Ho;
        const int o0 = st.groupStart[g];
        const float* wp = st.weights.data() + st.groupOffset[g];
        const float* bias = st.bias.data() + o0;
        const float* pRow = P + static_cast<size_t>(y) * rowStride;
        float* dstRow = out.mut + static_cast<size_t>(o0) * oStride + static_cast<size_t>(y) * Wo;
        switch (st.groupSize[g]) {
            case 4:
                convRow<4>(pRow, planeStride, rowStride, st.inC, st.kh, st.kw, wp, bias, dstRow, oStride, Wo);
                break;
            case 2:
                convRow<2>(pRow, planeStride, rowStride, st.inC, st.kh, st.kw, wp, bias, dstRow, oStride, Wo);
                break;
            default:
                convRow<1>(pRow, planeStride, rowStride, st.inC, st.kh, st.kw, wp, bias, dstRow, oStride, Wo);
                break;
        }
    });
}

void runConvGeneric(const Step& st, const Buf& in, Buf& out, int threads) {
    const int H = in.shape.h, W = in.shape.w;
    const int Ho = out.shape.h, Wo = out.shape.w;
    const int icg = st.inC / st.groups;
    const int og = st.outC / st.groups;
    parallelFor(threads, st.outC * Ho, [&](int t) {
        const int oc = t / Ho;
        const int oy = t % Ho;
        const int g = oc / og;
        float* dst = out.mut + (static_cast<size_t>(oc) * Ho + oy) * Wo;
        for (int ox = 0; ox < Wo; ++ox) {
            float sum = st.bias[oc];
            for (int icl = 0; icl < icg; ++icl) {
                const int ic = g * icg + icl;
                const float* plane = in.data + static_cast<size_t>(ic) * H * W;
                const float* wk = st.weights.data() + (static_cast<size_t>(oc) * icg + icl) * st.kh * st.kw;
                for (int ky = 0; ky < st.kh; ++ky) {
                    const int iy = oy * st.sh - st.ph + ky * st.dh;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < st.kw; ++kx) {
                        const int ix = ox * st.sw - st.pw + kx * st.dw;
                        if (ix < 0 || ix >= W) continue;
                        sum += wk[ky * st.kw + kx] * plane[static_cast<size_t>(iy) * W + ix];
                    }
                }
            }
            dst[ox] = sum;
        }
    });
}

// ---------------------------------------------------------------------------
// Element-wise and layout operators
// ---------------------------------------------------------------------------

constexpr size_t kChunk = 1u << 15;

template <class F>
void forEachChunk(int threads, size_t n, F&& f) {
    const int chunks = static_cast<int>((n + kChunk - 1) / kChunk);
    parallelFor(threads, chunks, [&](int c) {
        const size_t b = static_cast<size_t>(c) * kChunk;
        const size_t e = std::min(n, b + kChunk);
        f(b, e);
    });
}

void runUnary(const Step& st, const float* a, float* d, size_t n, int threads) {
    switch (st.op) {
        case Op::LeakyReLU: {
            const float alpha = st.f0;
            forEachChunk(threads, n, [&](size_t b, size_t e) {
                for (size_t i = b; i < e; ++i) d[i] = a[i] > 0.f ? a[i] : a[i] * alpha;
            });
            break;
        }
        case Op::ReLU:
            forEachChunk(threads, n, [&](size_t b, size_t e) {
                for (size_t i = b; i < e; ++i) d[i] = a[i] > 0.f ? a[i] : 0.f;
            });
            break;
        case Op::Sigmoid:
            forEachChunk(threads, n, [&](size_t b, size_t e) {
                for (size_t i = b; i < e; ++i) d[i] = 1.f / (1.f + std::exp(-a[i]));
            });
            break;
        case Op::Clip: {
            const float lo = st.f0, hi = st.f1;
            forEachChunk(threads, n, [&](size_t b, size_t e) {
                for (size_t i = b; i < e; ++i) {
                    float v = a[i];
                    if (v < lo) v = lo;
                    if (v > hi) v = hi;
                    d[i] = v;
                }
            });
            break;
        }
        case Op::MulScalar: {
            const float s = st.f0;
            forEachChunk(threads, n, [&](size_t b, size_t e) {
                for (size_t i = b; i < e; ++i) d[i] = a[i] * s;
            });
            break;
        }
        default:
            break;
    }
}

void runBinary(Op op, const float* a, const float* b, float* d, size_t n, int threads) {
    if (op == Op::Add) {
        forEachChunk(threads, n, [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i) d[i] = a[i] + b[i];
        });
    } else {
        forEachChunk(threads, n, [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i) d[i] = a[i] * b[i];
        });
    }
}

void runResize(const Step& st, const Buf& in, Buf& out, int threads) {
    const int C = in.shape.c, H = in.shape.h, W = in.shape.w;
    const int Ho = out.shape.h, Wo = out.shape.w;
    parallelFor(threads, C * Ho, [&](int t) {
        const int c = t / Ho;
        const int oy = t % Ho;
        const float* src = in.data + (static_cast<size_t>(c) * H + oy / st.i0) * W;
        float* dst = out.mut + (static_cast<size_t>(c) * Ho + oy) * Wo;
        for (int ox = 0; ox < Wo; ++ox) dst[ox] = src[ox / st.i1];
    });
}

void runPixelShuffle(const Step& st, const Buf& in, Buf& out, int threads) {
    const int r = st.i0;
    const int H = in.shape.h, W = in.shape.w;
    const int Co = out.shape.c, Ho = out.shape.h, Wo = out.shape.w;
    parallelFor(threads, Co * Ho, [&](int t) {
        const int c = t / Ho;
        const int oy = t % Ho;
        const int y = oy / r, i = oy % r;
        float* dst = out.mut + (static_cast<size_t>(c) * Ho + oy) * Wo;
        for (int ox = 0; ox < Wo; ++ox) {
            const int x = ox / r, j = ox % r;
            const size_t ic = static_cast<size_t>(c) * r * r + static_cast<size_t>(i) * r + j;
            dst[ox] = in.data[(ic * H + y) * W + x];
        }
    });
}

// ---------------------------------------------------------------------------
// Attribute parsing
// ---------------------------------------------------------------------------

bool attrInt(const json::Value& attrs, const char* key, int def, int minV, int maxV, int& out, std::string* err,
             const std::string& where) {
    out = def;
    const json::Value* v = attrs.find(key);
    if (!v) return true;
    if (!v->isNumber() || std::floor(v->number) != v->number || v->number < minV || v->number > maxV) {
        return failMsg(err, where + ": attribute '" + key + "' must be an integer in " + std::to_string(minV) + ".." +
                                std::to_string(maxV));
    }
    out = static_cast<int>(v->number);
    return true;
}

bool attrPair(const json::Value& attrs, const char* key, int d0, int d1, int minV, int maxV, int& a, int& b,
              std::string* err, const std::string& where) {
    a = d0;
    b = d1;
    const json::Value* v = attrs.find(key);
    if (!v) return true;
    if (!v->isArray() || v->array.size() != 2) {
        return failMsg(err, where + ": attribute '" + key + "' must be an array of two integers");
    }
    int vals[2];
    for (int i = 0; i < 2; ++i) {
        const json::Value& e = v->array[i];
        if (!e.isNumber() || std::floor(e.number) != e.number || e.number < minV || e.number > maxV) {
            return failMsg(err, where + ": attribute '" + key + "' values must be integers in " +
                                    std::to_string(minV) + ".." + std::to_string(maxV));
        }
        vals[i] = static_cast<int>(e.number);
    }
    a = vals[0];
    b = vals[1];
    return true;
}

bool attrFloat(const json::Value& attrs, const char* key, bool& present, float& out, std::string* err,
               const std::string& where) {
    present = false;
    const json::Value* v = attrs.find(key);
    if (!v) return true;
    if (!v->isNumber() || !std::isfinite(v->number)) {
        return failMsg(err, where + ": attribute '" + key + "' must be a finite number");
    }
    present = true;
    out = static_cast<float>(v->number);
    return true;
}

bool loadConstant(const Model& model, const TensorInfo& t, std::vector<float>& out) {
    const size_t n = static_cast<size_t>(t.size / dtypeSize(t.dtype));
    out.resize(n);
    const uint8_t* p = model.tensorData(t);
    if (t.dtype == DType::F32) {
        std::memcpy(out.data(), p, n * sizeof(float));
    } else {
        for (size_t i = 0; i < n; ++i) {
            uint16_t h;
            std::memcpy(&h, p + 2 * i, 2);
            out[i] = halfToFloat(h);
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Executor::Impl
// ---------------------------------------------------------------------------

struct Executor::Impl {
    std::vector<Step> steps;
    int numValues = 1;  // value 0 is the graph input
    int outputId = 0;
    int inC = 0, outC = 0, scale = 1;
    std::vector<int> lastUse;  // per value: last step index that reads it (steps.size() for the output)
    std::vector<int> defStep;  // per value: step that writes it (-1 for the input)

    bool inferShapes(const Shape& in, std::vector<Shape>& s, std::string* err) const {
        if (in.c < 1 || in.h < 1 || in.w < 1 || in.h > (1 << 20) || in.w > (1 << 20)) {
            return failMsg(err, "invalid input size");
        }
        s.assign(static_cast<size_t>(numValues), Shape());
        s[0] = in;
        for (const Step& st : steps) {
            Shape o;
            const Shape& a = s[st.in[0]];
            switch (st.op) {
                case Op::Conv: {
                    if (a.c != st.inC) {
                        return failMsg(err, st.label + ": expects " + std::to_string(st.inC) +
                                                " input channels but receives " + std::to_string(a.c));
                    }
                    const int64_t hh = static_cast<int64_t>(a.h) + 2LL * st.ph - static_cast<int64_t>(st.dh) * (st.kh - 1) - 1;
                    const int64_t ww = static_cast<int64_t>(a.w) + 2LL * st.pw - static_cast<int64_t>(st.dw) * (st.kw - 1) - 1;
                    if (hh < 0 || ww < 0) return failMsg(err, st.label + ": input is smaller than the kernel");
                    o.c = st.outC;
                    o.h = static_cast<int>(hh / st.sh + 1);  // inputs are <= 2^20, so this fits
                    o.w = static_cast<int>(ww / st.sw + 1);
                    break;
                }
                case Op::Add:
                case Op::Mul: {
                    const Shape& b = s[st.in[1]];
                    if (a.c != b.c || a.h != b.h || a.w != b.w) {
                        return failMsg(err, st.label + ": operands have different shapes");
                    }
                    o = a;
                    break;
                }
                case Op::MulScalar:
                case Op::LeakyReLU:
                case Op::ReLU:
                case Op::Sigmoid:
                case Op::Clip:
                    o = a;
                    break;
                case Op::Concat: {
                    o = a;
                    o.c = 0;
                    for (int id : st.in) {
                        const Shape& b = s[id];
                        if (b.h != a.h || b.w != a.w) return failMsg(err, st.label + ": inputs differ in height or width");
                        o.c += b.c;
                    }
                    break;
                }
                case Op::Resize: {
                    const int64_t hh = static_cast<int64_t>(a.h) * st.i0;
                    const int64_t ww = static_cast<int64_t>(a.w) * st.i1;
                    if (hh > (1 << 28) || ww > (1 << 28)) return failMsg(err, st.label + ": output tensor is too large");
                    o.c = a.c;
                    o.h = static_cast<int>(hh);
                    o.w = static_cast<int>(ww);
                    break;
                }
                case Op::PixelShuffle: {
                    const int rr = st.i0 * st.i0;
                    if (a.c % rr != 0) {
                        return failMsg(err, st.label + ": channel count " + std::to_string(a.c) +
                                                " is not divisible by upscale^2");
                    }
                    const int64_t hh = static_cast<int64_t>(a.h) * st.i0;
                    const int64_t ww = static_cast<int64_t>(a.w) * st.i0;
                    if (hh > (1 << 28) || ww > (1 << 28)) return failMsg(err, st.label + ": output tensor is too large");
                    o.c = a.c / rr;
                    o.h = static_cast<int>(hh);
                    o.w = static_cast<int>(ww);
                    break;
                }
            }
            if (o.c < 1 || o.h < 1 || o.w < 1 || static_cast<int64_t>(o.c) * o.h * o.w > INT_MAX) {
                return failMsg(err, st.label + ": output tensor is empty or too large");
            }
            s[st.out] = o;
        }
        return true;
    }
};

Executor::Executor() : impl_(new Impl()) {}
Executor::~Executor() = default;

int Executor::inChannels() const { return impl_->inC; }
int Executor::outChannels() const { return impl_->outC; }
int Executor::scale() const { return impl_->scale; }

std::unique_ptr<Executor> Executor::create(const Model& model, std::string* error) {
    if (model.inputs().size() != 1 || model.outputs().size() != 1) {
        failMsg(error, "only models with exactly one input and one output can be executed");
        return nullptr;
    }
    const IoInfo& input = model.inputs()[0];
    const IoInfo& output = model.outputs()[0];
    if (input.shape.size() != 4 || input.shape[0] != 1 || input.shape[1] < 1) {
        failMsg(error, "model input must be NCHW with batch 1 and a fixed channel count, e.g. [1, 3, -1, -1]");
        return nullptr;
    }
    if (input.shape[2] > 0 || input.shape[3] > 0) {
        failMsg(error, "model input has a fixed size; export it with dynamic height and width ([1, C, -1, -1])");
        return nullptr;
    }

    std::unique_ptr<Executor> exec(new Executor());
    Impl& im = *exec->impl_;
    im.inC = static_cast<int>(input.shape[1]);

    std::unordered_map<std::string, int> ids;
    ids[input.name] = 0;
    std::unordered_map<std::string, const TensorInfo*> constants;
    for (const TensorInfo& t : model.tensors()) constants[t.name] = &t;

    int next = 1;
    size_t index = 0;
    for (const Node& n : model.nodes()) {
        const std::string where = "node #" + std::to_string(index) + " (" + n.op + ")";
        ++index;
        if (n.outputs.size() != 1) {
            failMsg(error, where + ": operators with several outputs are not supported");
            return nullptr;
        }
        Step st;
        st.label = where;

        auto valueOf = [&](const std::string& name, int& id) {
            auto it = ids.find(name);
            if (it == ids.end()) return false;
            id = it->second;
            return true;
        };

        if (n.op == "Conv2D") {
            if (n.inputs.size() < 2 || n.inputs.size() > 3) {
                failMsg(error, where + ": expects inputs x, weight[, bias]");
                return nullptr;
            }
            int x = 0;
            if (!valueOf(n.inputs[0], x)) {
                failMsg(error, where + ": input '" + n.inputs[0] + "' is not an activation");
                return nullptr;
            }
            st.in = {x};
            auto wIt = constants.find(n.inputs[1]);
            if (wIt == constants.end() || wIt->second->shape.size() != 4) {
                failMsg(error, where + ": weight '" + n.inputs[1] + "' must be a constant tensor [O, I/groups, kh, kw]");
                return nullptr;
            }
            const TensorInfo& wt = *wIt->second;
            int g = 1;
            if (!attrInt(n.attrs, "groups", 1, 1, 1 << 20, g, error, where)) return nullptr;
            if (!attrPair(n.attrs, "stride", 1, 1, 1, 1 << 12, st.sh, st.sw, error, where) ||
                !attrPair(n.attrs, "padding", 0, 0, 0, 1 << 12, st.ph, st.pw, error, where) ||
                !attrPair(n.attrs, "dilation", 1, 1, 1, 1 << 12, st.dh, st.dw, error, where)) {
                return nullptr;
            }
            st.groups = g;
            st.outC = static_cast<int>(wt.shape[0]);
            st.kh = static_cast<int>(wt.shape[2]);
            st.kw = static_cast<int>(wt.shape[3]);
            const int64_t icg = wt.shape[1];
            if (st.outC < 1 || icg < 1 || st.kh < 1 || st.kw < 1 || st.kh > 64 || st.kw > 64 || icg > (1 << 20) ||
                wt.shape[0] > (1 << 20)) {
                failMsg(error, where + ": unsupported weight shape");
                return nullptr;
            }
            st.inC = static_cast<int>(icg) * g;
            if (st.outC % g != 0) {
                failMsg(error, where + ": output channels are not divisible by groups");
                return nullptr;
            }
            std::vector<float> oihw;
            loadConstant(model, wt, oihw);
            if (n.inputs.size() == 3) {
                auto bIt = constants.find(n.inputs[2]);
                if (bIt == constants.end() || bIt->second->shape.size() != 1 || bIt->second->shape[0] != st.outC) {
                    failMsg(error, where + ": bias '" + n.inputs[2] + "' must be a constant tensor [O]");
                    return nullptr;
                }
                loadConstant(model, *bIt->second, st.bias);
            } else {
                st.bias.assign(static_cast<size_t>(st.outC), 0.f);
            }
            st.op = Op::Conv;
            st.fast = st.sh == 1 && st.sw == 1 && st.dh == 1 && st.dw == 1 && g == 1;
            if (st.fast) {
                packFastWeights(st, oihw);
            } else {
                st.weights = std::move(oihw);
            }
        } else if (n.op == "Add" || n.op == "Mul") {
            bool hasScalar = false;
            float scalar = 0.f;
            if (!attrFloat(n.attrs, "scalar", hasScalar, scalar, error, where)) return nullptr;
            if (n.inputs.size() == 2) {
                int a = 0, b = 0;
                if (!valueOf(n.inputs[0], a) || !valueOf(n.inputs[1], b)) {
                    failMsg(error, where + ": both operands must be activations (constants are not supported here)");
                    return nullptr;
                }
                st.op = n.op == "Add" ? Op::Add : Op::Mul;
                st.in = {a, b};
            } else if (n.op == "Mul" && n.inputs.size() == 1 && hasScalar) {
                int a = 0;
                if (!valueOf(n.inputs[0], a)) {
                    failMsg(error, where + ": operand is not an activation");
                    return nullptr;
                }
                st.op = Op::MulScalar;
                st.in = {a};
                st.f0 = scalar;
            } else {
                failMsg(error, where + ": expects two inputs" + std::string(n.op == "Mul" ? " (or one input and a 'scalar' attribute)" : ""));
                return nullptr;
            }
        } else if (n.op == "LeakyReLU" || n.op == "ReLU" || n.op == "Sigmoid" || n.op == "Clip") {
            int a = 0;
            if (n.inputs.size() != 1 || !valueOf(n.inputs[0], a)) {
                failMsg(error, where + ": expects exactly one activation input");
                return nullptr;
            }
            st.in = {a};
            bool present = false;
            if (n.op == "LeakyReLU") {
                st.op = Op::LeakyReLU;
                st.f0 = 0.01f;
                float v = 0.f;
                if (!attrFloat(n.attrs, "alpha", present, v, error, where)) return nullptr;
                if (present) st.f0 = v;
            } else if (n.op == "ReLU") {
                st.op = Op::ReLU;
            } else if (n.op == "Sigmoid") {
                st.op = Op::Sigmoid;
            } else {
                st.op = Op::Clip;
                st.f0 = -std::numeric_limits<float>::infinity();
                st.f1 = std::numeric_limits<float>::infinity();
                float v = 0.f;
                if (!attrFloat(n.attrs, "min", present, v, error, where)) return nullptr;
                if (present) st.f0 = v;
                if (!attrFloat(n.attrs, "max", present, v, error, where)) return nullptr;
                if (present) st.f1 = v;
                if (st.f0 > st.f1) {
                    failMsg(error, where + ": min is larger than max");
                    return nullptr;
                }
            }
        } else if (n.op == "Concat") {
            if (n.inputs.size() < 2) {
                failMsg(error, where + ": expects at least two inputs");
                return nullptr;
            }
            int axis = 1;
            if (!attrInt(n.attrs, "axis", 1, -4, 3, axis, error, where)) return nullptr;
            if (axis != 1 && axis != -3) {
                failMsg(error, where + ": only concatenation along the channel axis (1) is supported");
                return nullptr;
            }
            for (const std::string& name : n.inputs) {
                int id = 0;
                if (!valueOf(name, id)) {
                    failMsg(error, where + ": input '" + name + "' is not an activation");
                    return nullptr;
                }
                st.in.push_back(id);
            }
            st.op = Op::Concat;
        } else if (n.op == "ResizeNearest") {
            int a = 0;
            if (n.inputs.size() != 1 || !valueOf(n.inputs[0], a)) {
                failMsg(error, where + ": expects exactly one activation input");
                return nullptr;
            }
            st.in = {a};
            st.op = Op::Resize;
            if (!n.attrs.find("scale")) {
                failMsg(error, where + ": missing attribute 'scale'");
                return nullptr;
            }
            if (!attrPair(n.attrs, "scale", 1, 1, 1, 64, st.i0, st.i1, error, where)) return nullptr;
        } else if (n.op == "PixelShuffle") {
            int a = 0;
            if (n.inputs.size() != 1 || !valueOf(n.inputs[0], a)) {
                failMsg(error, where + ": expects exactly one activation input");
                return nullptr;
            }
            st.in = {a};
            st.op = Op::PixelShuffle;
            if (!n.attrs.find("upscale")) {
                failMsg(error, where + ": missing attribute 'upscale'");
                return nullptr;
            }
            if (!attrInt(n.attrs, "upscale", 1, 1, 64, st.i0, error, where)) return nullptr;
        } else {
            failMsg(error, "unsupported operator '" + n.op + "' (" + where + ")");
            return nullptr;
        }

        st.out = next;
        ids[n.outputs[0]] = next;
        ++next;
        im.steps.push_back(std::move(st));
    }
    im.numValues = next;

    auto outIt = ids.find(output.name);
    if (outIt == ids.end()) {
        failMsg(error, "graph output '" + output.name + "' is not produced by any node");
        return nullptr;
    }
    im.outputId = outIt->second;

    // Liveness: when can each value be released, and which steps can work in place.
    const int numSteps = static_cast<int>(im.steps.size());
    im.lastUse.assign(static_cast<size_t>(im.numValues), -1);
    im.defStep.assign(static_cast<size_t>(im.numValues), -1);
    for (int i = 0; i < numSteps; ++i) {
        im.defStep[im.steps[i].out] = i;
        for (int v : im.steps[i].in) im.lastUse[v] = i;
    }
    im.lastUse[im.outputId] = numSteps;
    for (int v = 1; v < im.numValues; ++v) {
        if (im.lastUse[v] < im.defStep[v]) im.lastUse[v] = im.defStep[v];
    }
    for (int i = 0; i < numSteps; ++i) {
        Step& st = im.steps[i];
        const bool elementwise = st.op == Op::Add || st.op == Op::Mul || st.op == Op::MulScalar ||
                                 st.op == Op::LeakyReLU || st.op == Op::ReLU || st.op == Op::Sigmoid || st.op == Op::Clip;
        if (elementwise) {
            for (size_t k = 0; k < st.in.size(); ++k) {
                const int v = st.in[k];
                if (v != 0 && im.lastUse[v] == i) {
                    st.inplace = true;
                    st.inplaceArg = static_cast<int>(k);
                    break;
                }
            }
        }
        std::unordered_set<int> seen;
        for (size_t k = 0; k < st.in.size(); ++k) {
            const int v = st.in[k];
            if (v == 0 || v == im.outputId || im.lastUse[v] != i || !seen.insert(v).second) continue;
            if (st.inplace && static_cast<int>(k) == st.inplaceArg) continue;  // ownership moves to the output
            st.release.push_back(v);
        }
        if (st.out != im.outputId && im.lastUse[st.out] == i) st.release.push_back(st.out);
    }

    // Shape checks on two different sizes: channels must line up and the output
    // must be an exact integer multiple of the input (needed for tiling).
    std::vector<Shape> s1, s2;
    Shape a;
    a.c = im.inC;
    a.h = 16;
    a.w = 24;
    Shape b = a;
    b.h = 20;
    b.w = 12;
    if (!im.inferShapes(a, s1, error) || !im.inferShapes(b, s2, error)) return nullptr;
    const Shape& o1 = s1[im.outputId];
    const Shape& o2 = s2[im.outputId];
    if (o1.h % a.h != 0 || o1.w % a.w != 0 || o1.h / a.h != o1.w / a.w || o2.h % b.h != 0 || o2.w % b.w != 0 ||
        o2.h / b.h != o1.h / a.h || o2.w / b.w != o1.h / a.h) {
        failMsg(error, "the model output size is not an exact integer multiple of the input size");
        return nullptr;
    }
    im.scale = o1.h / a.h;
    im.outC = o1.c;
    if (output.shape.size() != 4 || (output.shape[1] > 0 && output.shape[1] != im.outC)) {
        failMsg(error, "declared output channels (" + std::to_string(output.shape.size() == 4 ? output.shape[1] : -1) +
                           ") do not match the graph (" + std::to_string(im.outC) + ")");
        return nullptr;
    }
    if (model.scale() > 1 && model.scale() != im.scale) {
        failMsg(error, "model declares scale " + std::to_string(model.scale()) + " but its graph produces scale " +
                           std::to_string(im.scale));
        return nullptr;
    }
    return exec;
}

size_t Executor::estimatePeakBytes(int h, int w) const {
    const Impl& im = *impl_;
    Shape in;
    in.c = im.inC;
    in.h = h;
    in.w = w;
    std::vector<Shape> shapes;
    if (!im.inferShapes(in, shapes, nullptr)) return std::numeric_limits<size_t>::max();
    size_t peak = 0;
    const int numSteps = static_cast<int>(im.steps.size());
    for (int i = 0; i < numSteps; ++i) {
        size_t live = 0;
        for (int v = 0; v < im.numValues; ++v) {
            if (im.defStep[v] <= i && im.lastUse[v] >= i) live += numel(shapes[v]) * sizeof(float);
        }
        const Step& st = im.steps[i];
        if (st.op == Op::Conv && st.fast && (st.ph != 0 || st.pw != 0)) {
            const Shape& a = shapes[st.in[0]];
            live += static_cast<size_t>(st.inC) * (a.h + 2 * st.ph) * (a.w + 2 * st.pw) * sizeof(float);
        }
        peak = std::max(peak, live);
    }
    return peak;
}

Executor::Status Executor::run(const float* in, int h, int w, float* out, int threads,
                               const std::atomic<bool>* cancel, std::string* error) const {
    const Impl& im = *impl_;
    if (!in || !out || h < 1 || w < 1) {
        failMsg(error, "invalid arguments");
        return Status::Error;
    }
    if (threads < 1) threads = 1;
    try {
        Shape inShape;
        inShape.c = im.inC;
        inShape.h = h;
        inShape.w = w;
        std::vector<Shape> shapes;
        if (!im.inferShapes(inShape, shapes, error)) return Status::Error;

        std::vector<Buf> vals(static_cast<size_t>(im.numValues));
        vals[0].data = in;
        vals[0].shape = shapes[0];

        for (const Step& st : im.steps) {
            if (cancel && cancel->load(std::memory_order_relaxed)) {
                failMsg(error, "cancelled");
                return Status::Cancelled;
            }
            const Shape& os = shapes[st.out];
            const size_t n = numel(os);

            Buf dst;
            dst.shape = os;
            if (st.inplace) {
                const int src = st.in[st.inplaceArg];
                dst = std::move(vals[src]);
                vals[src] = Buf();
                dst.shape = os;
            } else if (st.out == im.outputId) {
                dst.mut = out;
                dst.data = out;
            } else {
                dst.own.reset(new float[n]);
                dst.mut = dst.own.get();
                dst.data = dst.mut;
            }

            const int movedId = st.inplace ? st.in[st.inplaceArg] : -1;
            auto inData = [&](int k) -> const float* {
                // After an in-place move the source buffer now lives in `dst`.
                if (st.in[k] == movedId) return dst.data;
                return vals[st.in[k]].data;
            };

            switch (st.op) {
                case Op::Conv: {
                    const Buf& x = vals[st.in[0]];
                    if (st.fast) runConvFast(st, x, dst, threads);
                    else runConvGeneric(st, x, dst, threads);
                    break;
                }
                case Op::Add:
                case Op::Mul:
                    runBinary(st.op, inData(0), inData(1), dst.mut, n, threads);
                    break;
                case Op::MulScalar:
                case Op::LeakyReLU:
                case Op::ReLU:
                case Op::Sigmoid:
                case Op::Clip:
                    runUnary(st, inData(0), dst.mut, n, threads);
                    break;
                case Op::Concat: {
                    size_t offset = 0;
                    for (int id : st.in) {
                        const Buf& b = vals[id];
                        const size_t bn = numel(b.shape);
                        std::memcpy(dst.mut + offset, b.data, bn * sizeof(float));
                        offset += bn;
                    }
                    break;
                }
                case Op::Resize:
                    runResize(st, vals[st.in[0]], dst, threads);
                    break;
                case Op::PixelShuffle:
                    runPixelShuffle(st, vals[st.in[0]], dst, threads);
                    break;
            }

            vals[st.out] = std::move(dst);
            for (int v : st.release) vals[v] = Buf();
        }

        const Buf& result = vals[im.outputId];
        if (result.data != out) std::memcpy(out, result.data, numel(result.shape) * sizeof(float));
        return Status::Ok;
    } catch (const std::bad_alloc&) {
        failMsg(error, "out of memory while running the model; use a smaller tile size or memory budget");
        return Status::OutOfMemory;
    } catch (const std::exception& e) {
        failMsg(error, std::string("unexpected error: ") + e.what());
        return Status::Error;
    }
}

}  // namespace vva
