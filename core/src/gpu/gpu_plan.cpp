#include "gpu_plan.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace vva {
namespace gpu {

namespace {

double gDispatchFlops = 1.0e9;  // target work per dispatch (rows of a convolution are split to fit)
double gSegmentFlops = 4.0e9;   // submit boundary after this much work

float halfBitsToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
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

bool fail(std::string* s, const std::string& m) {
    if (s) *s = m;
    return false;
}

size_t roundUp(size_t v, size_t m) { return (v + m - 1) / m * m; }

enum class PKind { Conv, Leaky, MulS, Add, Concat, Resize };

struct Prim {
    PKind kind = PKind::Add;
    std::vector<int> in;
    int out = -1;
    float value = 0.f;  // Leaky: alpha, MulS: scalar
    int sh = 1, sw = 1;
    int cin = 0, cout = 0;
    int weightIndex = -1;  // conv: index into convs_
    bool dead = false;
};

struct ConvData {
    int cin = 0, cout = 0;
    size_t offset = 0;  // floats into the weights blob (64-float aligned)
    size_t floats = 0;
    int biasVec4 = 0;   // bias position in vec4 units relative to `offset`
};

enum class OKind { Conv, Elt, Concat, Resize };

struct Op {
    OKind kind = OKind::Elt;
    std::vector<int> in;  // conv: [x]; elt: [a] or [a, b]; concat: inputs; resize: [x]
    int out = -1;
    int conv = -1;                  // index into convs
    float alpha = 1.f, scale = 1.f;  // conv epilogue / elt
    int res = -1;                    // conv: residual tensor
    int sh = 1, sw = 1;
};

struct Tensor {
    int c = 0;
    int def = -1;       // op index producing it, -1 for the graph input
    int lastUse = -1;
    int h = 0, w = 0;   // set per build
};

struct ClassAlloc {
    size_t unit = 0;  // bytes of one plane
    std::vector<std::pair<size_t, size_t>> freeList;  // (offset, size), sorted by offset
    size_t top = 0;

    size_t alloc(size_t bytes) {
        for (size_t i = 0; i < freeList.size(); ++i) {
            if (freeList[i].second >= bytes) {
                const size_t off = freeList[i].first;
                if (freeList[i].second == bytes) {
                    freeList.erase(freeList.begin() + static_cast<long>(i));
                } else {
                    freeList[i].first += bytes;
                    freeList[i].second -= bytes;
                }
                return off;
            }
        }
        const size_t off = top;
        top += bytes;
        return off;
    }

    void release(size_t off, size_t bytes) {
        size_t pos = 0;
        while (pos < freeList.size() && freeList[pos].first < off) ++pos;
        freeList.insert(freeList.begin() + static_cast<long>(pos), std::make_pair(off, bytes));
        if (pos + 1 < freeList.size() && freeList[pos].first + freeList[pos].second == freeList[pos + 1].first) {
            freeList[pos].second += freeList[pos + 1].second;
            freeList.erase(freeList.begin() + static_cast<long>(pos) + 1);
        }
        if (pos > 0 && freeList[pos - 1].first + freeList[pos - 1].second == freeList[pos].first) {
            freeList[pos - 1].second += freeList[pos].second;
            freeList.erase(freeList.begin() + static_cast<long>(pos));
        }
    }
};

}  // namespace

struct Plan::Impl {
    std::vector<float> weights;
    std::vector<ConvData> convs;
    std::vector<Op> ops;
    std::vector<Tensor> tensors;  // c, def, lastUse filled at create()
    int inputTensor = 0, outputTensor = 0;
    int inC = 0, outC = 0, scale = 1;
};

Plan::Plan() : impl_(new Impl()) {}
Plan::~Plan() = default;

int Plan::inChannels() const { return impl_->inC; }
int Plan::outChannels() const { return impl_->outC; }
int Plan::scale() const { return impl_->scale; }
const std::vector<float>& Plan::weights() const { return impl_->weights; }

std::unique_ptr<Plan> Plan::create(const Model& model, std::string* reason) {
    if (model.inputs().size() != 1 || model.outputs().size() != 1) {
        fail(reason, "the GPU path needs exactly one input and one output");
        return nullptr;
    }
    std::unique_ptr<Plan> plan(new Plan());
    Impl& im = *plan->impl_;

    std::unordered_map<std::string, const TensorInfo*> constants;
    for (const TensorInfo& t : model.tensors()) constants[t.name] = &t;

    const IoInfo& input = model.inputs()[0];
    if (input.shape.size() != 4 || input.shape[1] < 1) {
        fail(reason, "unsupported input layout");
        return nullptr;
    }
    std::unordered_map<std::string, int> ids;
    std::vector<int> channels;  // per tensor id
    ids[input.name] = 0;
    channels.push_back(static_cast<int>(input.shape[1]));
    im.inC = channels[0];

    std::vector<Prim> prims;
    for (const Node& n : model.nodes()) {
        if (n.outputs.size() != 1) {
            fail(reason, "node with several outputs");
            return nullptr;
        }
        Prim p;
        auto valueId = [&](const std::string& name) {
            auto it = ids.find(name);
            return it == ids.end() ? -1 : it->second;
        };
        auto attrInt = [&](const char* key, int def) {
            const json::Value* v = n.attrs.find(key);
            return (v && v->isNumber()) ? static_cast<int>(v->number) : def;
        };
        auto pairIs = [&](const char* key, int defA, int defB, int a, int b) {
            const json::Value* v = n.attrs.find(key);
            int x = defA, y = defB;
            if (v) {
                if (!v->isArray() || v->array.size() != 2 || !v->array[0].isNumber() || !v->array[1].isNumber()) return false;
                x = static_cast<int>(v->array[0].number);
                y = static_cast<int>(v->array[1].number);
            }
            return x == a && y == b;
        };

        if (n.op == "Conv2D") {
            if (n.inputs.size() < 2 || n.inputs.size() > 3) {
                fail(reason, "Conv2D with an unexpected number of inputs");
                return nullptr;
            }
            const int x = valueId(n.inputs[0]);
            auto wIt = constants.find(n.inputs[1]);
            if (x < 0 || wIt == constants.end() || wIt->second->shape.size() != 4) {
                fail(reason, "Conv2D operands must be an activation and a constant weight");
                return nullptr;
            }
            const TensorInfo& wt = *wIt->second;
            if (wt.shape[2] != 3 || wt.shape[3] != 3 || attrInt("groups", 1) != 1 || !pairIs("stride", 1, 1, 1, 1) ||
                !pairIs("padding", 0, 0, 1, 1) || !pairIs("dilation", 1, 1, 1, 1)) {
                fail(reason, "only 3x3 convolutions with stride 1, padding 1 and no groups run on the GPU");
                return nullptr;
            }
            ConvData cd;
            cd.cout = static_cast<int>(wt.shape[0]);
            cd.cin = static_cast<int>(wt.shape[1]);
            if (cd.cin != channels[static_cast<size_t>(x)]) {
                fail(reason, "convolution input channels do not match");
                return nullptr;
            }
            std::vector<float> w(static_cast<size_t>(cd.cout) * cd.cin * 9);
            const uint8_t* data = model.tensorData(wt);
            for (size_t i = 0; i < w.size(); ++i) {
                if (wt.dtype == DType::F32) {
                    std::memcpy(&w[i], data + 4 * i, 4);
                } else {
                    uint16_t h;
                    std::memcpy(&h, data + 2 * i, 2);
                    w[i] = halfBitsToFloat(h);
                }
            }
            std::vector<float> b(static_cast<size_t>(cd.cout), 0.f);
            if (n.inputs.size() == 3) {
                auto bIt = constants.find(n.inputs[2]);
                if (bIt == constants.end() || bIt->second->shape.size() != 1 || bIt->second->shape[0] != cd.cout) {
                    fail(reason, "bad convolution bias");
                    return nullptr;
                }
                const uint8_t* bd = model.tensorData(*bIt->second);
                for (int i = 0; i < cd.cout; ++i) {
                    if (bIt->second->dtype == DType::F32) {
                        std::memcpy(&b[static_cast<size_t>(i)], bd + 4 * i, 4);
                    } else {
                        uint16_t h;
                        std::memcpy(&h, bd + 2 * i, 2);
                        b[static_cast<size_t>(i)] = halfBitsToFloat(h);
                    }
                }
            }
            // Pack: [output block of 8][input channel][tap][8], then the bias; start 256-byte aligned.
            const int ob = (cd.cout + 7) / 8;
            const size_t packed = static_cast<size_t>(ob) * cd.cin * 9 * 8;
            cd.offset = roundUp(im.weights.size(), 64);
            cd.biasVec4 = static_cast<int>(packed / 4);
            cd.floats = packed + static_cast<size_t>(ob) * 8;
            im.weights.resize(cd.offset + roundUp(cd.floats, 64), 0.f);
            for (int o = 0; o < ob; ++o) {
                for (int ic = 0; ic < cd.cin; ++ic) {
                    for (int tap = 0; tap < 9; ++tap) {
                        for (int j = 0; j < 8; ++j) {
                            const int oc = o * 8 + j;
                            const size_t dst = cd.offset + ((static_cast<size_t>(o) * cd.cin + ic) * 9 + tap) * 8 + j;
                            im.weights[dst] = oc < cd.cout ? w[(static_cast<size_t>(oc) * cd.cin + ic) * 9 + tap] : 0.f;
                        }
                    }
                }
                for (int j = 0; j < 8; ++j) {
                    const int oc = o * 8 + j;
                    im.weights[cd.offset + packed + static_cast<size_t>(o) * 8 + j] = oc < cd.cout ? b[static_cast<size_t>(oc)] : 0.f;
                }
            }
            p.kind = PKind::Conv;
            p.in = {x};
            p.cin = cd.cin;
            p.cout = cd.cout;
            p.weightIndex = static_cast<int>(im.convs.size());
            im.convs.push_back(cd);
        } else if (n.op == "LeakyReLU" || n.op == "ReLU") {
            const int x = n.inputs.size() == 1 ? valueId(n.inputs[0]) : -1;
            if (x < 0) {
                fail(reason, "activation with an unsupported operand");
                return nullptr;
            }
            p.kind = PKind::Leaky;
            p.in = {x};
            p.value = 0.f;
            if (n.op == "LeakyReLU") {
                const json::Value* a = n.attrs.find("alpha");
                p.value = (a && a->isNumber()) ? static_cast<float>(a->number) : 0.01f;
            }
        } else if (n.op == "Mul") {
            const json::Value* s = n.attrs.find("scalar");
            const int x = n.inputs.size() == 1 ? valueId(n.inputs[0]) : -1;
            if (x < 0 || !s || !s->isNumber()) {
                fail(reason, "only Mul by a scalar runs on the GPU");
                return nullptr;
            }
            p.kind = PKind::MulS;
            p.in = {x};
            p.value = static_cast<float>(s->number);
        } else if (n.op == "Add") {
            const int a = n.inputs.size() == 2 ? valueId(n.inputs[0]) : -1;
            const int b = n.inputs.size() == 2 ? valueId(n.inputs[1]) : -1;
            if (a < 0 || b < 0) {
                fail(reason, "Add needs two activations");
                return nullptr;
            }
            p.kind = PKind::Add;
            p.in = {a, b};
        } else if (n.op == "Concat") {
            const int axis = attrInt("axis", 1);
            if (axis != 1 && axis != -3) {
                fail(reason, "Concat is only supported along channels");
                return nullptr;
            }
            for (const std::string& name : n.inputs) {
                const int id = valueId(name);
                if (id < 0) {
                    fail(reason, "Concat of a constant");
                    return nullptr;
                }
                p.in.push_back(id);
            }
            if (p.in.size() < 2) {
                fail(reason, "Concat needs two inputs");
                return nullptr;
            }
            p.kind = PKind::Concat;
        } else if (n.op == "ResizeNearest") {
            const json::Value* sc = n.attrs.find("scale");
            const int x = n.inputs.size() == 1 ? valueId(n.inputs[0]) : -1;
            if (x < 0 || !sc || !sc->isArray() || sc->array.size() != 2 || !sc->array[0].isNumber() || !sc->array[1].isNumber()) {
                fail(reason, "bad ResizeNearest");
                return nullptr;
            }
            p.kind = PKind::Resize;
            p.in = {x};
            p.sh = static_cast<int>(sc->array[0].number);
            p.sw = static_cast<int>(sc->array[1].number);
            if (p.sh < 1 || p.sw < 1 || p.sh > 16 || p.sw > 16) {
                fail(reason, "unsupported resize factor");
                return nullptr;
            }
        } else {
            fail(reason, "operator " + n.op + " is not available on the GPU path");
            return nullptr;
        }

        // Output channels.
        int oc = 0;
        switch (p.kind) {
            case PKind::Conv: oc = p.cout; break;
            case PKind::Concat:
                for (int id : p.in) oc += channels[static_cast<size_t>(id)];
                break;
            default: oc = channels[static_cast<size_t>(p.in[0])]; break;
        }
        if (p.kind == PKind::Add && channels[static_cast<size_t>(p.in[0])] != channels[static_cast<size_t>(p.in[1])]) {
            fail(reason, "Add of tensors with different channel counts");
            return nullptr;
        }
        p.out = static_cast<int>(channels.size());
        ids[n.outputs[0]] = p.out;
        channels.push_back(oc);
        prims.push_back(p);
    }

    auto outIt = ids.find(model.outputs()[0].name);
    if (outIt == ids.end() || outIt->second == 0) {
        fail(reason, "unsupported graph output");
        return nullptr;
    }
    int outputTensor = outIt->second;

    // ---- fusion -------------------------------------------------------------------------------
    const int numTensors = static_cast<int>(channels.size());
    std::vector<int> uses(static_cast<size_t>(numTensors), 0);
    std::vector<int> defPrim(static_cast<size_t>(numTensors), -1);
    for (size_t i = 0; i < prims.size(); ++i) {
        defPrim[static_cast<size_t>(prims[i].out)] = static_cast<int>(i);
        for (int t : prims[i].in) ++uses[static_cast<size_t>(t)];
    }
    ++uses[static_cast<size_t>(outputTensor)];

    auto consumerOf = [&](int tensor, size_t after) -> int {
        for (size_t j = after + 1; j < prims.size(); ++j) {
            if (prims[j].dead) continue;
            for (int t : prims[j].in) {
                if (t == tensor) return static_cast<int>(j);
            }
        }
        return -1;
    };

    std::vector<Op> ops;
    for (size_t i = 0; i < prims.size(); ++i) {
        Prim& p = prims[i];
        if (p.dead) continue;
        Op op;
        if (p.kind == PKind::Conv) {
            op.kind = OKind::Conv;
            op.in = p.in;
            op.out = p.out;
            op.conv = p.weightIndex;
            bool act = false, scaled = false, hasRes = false;
            for (;;) {
                if (uses[static_cast<size_t>(op.out)] != 1) break;
                const int j = consumerOf(op.out, i);
                if (j < 0) break;
                Prim& q = prims[static_cast<size_t>(j)];
                if ((q.kind == PKind::Leaky) && !act && !scaled && !hasRes) {
                    op.alpha = q.value;
                    act = true;
                } else if (q.kind == PKind::MulS && !scaled && !hasRes) {
                    op.scale = q.value;
                    scaled = true;
                } else if (q.kind == PKind::Add && !hasRes) {
                    const int other = q.in[0] == op.out ? q.in[1] : q.in[0];
                    if (q.in[0] == q.in[1] || defPrim[static_cast<size_t>(other)] >= static_cast<int>(i)) break;
                    op.res = other;
                    hasRes = true;
                } else {
                    break;
                }
                op.out = q.out;
                q.dead = true;
                if (hasRes) break;
            }
        } else if (p.kind == PKind::MulS) {
            op.kind = OKind::Elt;
            op.in = p.in;
            op.out = p.out;
            op.scale = p.value;
            if (uses[static_cast<size_t>(op.out)] == 1) {
                const int j = consumerOf(op.out, i);
                if (j >= 0 && prims[static_cast<size_t>(j)].kind == PKind::Add) {
                    Prim& q = prims[static_cast<size_t>(j)];
                    const int other = q.in[0] == op.out ? q.in[1] : q.in[0];
                    if (q.in[0] != q.in[1] && defPrim[static_cast<size_t>(other)] < static_cast<int>(i)) {
                        op.in = {p.in[0], other};
                        op.out = q.out;
                        q.dead = true;
                    }
                }
            }
        } else if (p.kind == PKind::Leaky) {
            op.kind = OKind::Elt;
            op.in = p.in;
            op.out = p.out;
            op.alpha = p.value;
        } else if (p.kind == PKind::Add) {
            op.kind = OKind::Elt;
            op.in = p.in;
            op.out = p.out;
        } else if (p.kind == PKind::Concat) {
            op.kind = OKind::Concat;
            op.in = p.in;
            op.out = p.out;
        } else {
            op.kind = OKind::Resize;
            op.in = p.in;
            op.out = p.out;
            op.sh = p.sh;
            op.sw = p.sw;
        }
        ops.push_back(op);
    }

    // A fused chain may have replaced the graph output tensor id by the id of its last member.
    if (!ops.empty() && std::none_of(ops.begin(), ops.end(), [&](const Op& o) { return o.out == outputTensor; })) {
        fail(reason, "internal error: lost the graph output while fusing");
        return nullptr;
    }

    // ---- tensors, liveness and channel rules -------------------------------------------------------
    im.tensors.assign(static_cast<size_t>(numTensors), Tensor());
    for (int t = 0; t < numTensors; ++t) im.tensors[static_cast<size_t>(t)].c = channels[static_cast<size_t>(t)];
    for (size_t k = 0; k < ops.size(); ++k) {
        im.tensors[static_cast<size_t>(ops[k].out)].def = static_cast<int>(k);
        auto use = [&](int t) { im.tensors[static_cast<size_t>(t)].lastUse = static_cast<int>(k); };
        for (int t : ops[k].in) use(t);
        if (ops[k].res >= 0) use(ops[k].res);
    }
    im.tensors[static_cast<size_t>(outputTensor)].lastUse = static_cast<int>(ops.size());

    for (size_t k = 0; k < ops.size(); ++k) {
        const Op& op = ops[k];
        auto need8 = [&](int t) {
            return im.tensors[static_cast<size_t>(t)].c % 8 == 0;
        };
        if (op.kind != OKind::Conv) {
            for (int t : op.in) {
                if (!need8(t)) {
                    fail(reason, "channel counts must be multiples of 8 around element-wise, concat and resize operators");
                    return nullptr;
                }
            }
            if (!need8(op.out)) {
                fail(reason, "channel counts must be multiples of 8 around element-wise, concat and resize operators");
                return nullptr;
            }
        } else if (op.res >= 0 && !need8(op.res)) {
            fail(reason, "a fused residual needs a multiple of 8 channels");
            return nullptr;
        }
    }
    // Tensors produced by elt/concat/resize must have 8k channels (checked above); the graph input may be anything.
    im.ops = ops;
    im.inputTensor = 0;
    im.outputTensor = outputTensor;
    im.outC = im.tensors[static_cast<size_t>(outputTensor)].c;
    int scaleNum = 1;
    for (const Op& op : ops) {
        if (op.kind == OKind::Resize) scaleNum *= op.sh;  // checked against the real output size in build()
    }
    im.scale = scaleNum;
    if (ops.empty()) {
        fail(reason, "empty graph");
        return nullptr;
    }
    return plan;
}

void Plan::setWorkTargets(double dispatchFlops, double segmentFlops) {
    gDispatchFlops = dispatchFlops;
    gSegmentFlops = segmentFlops;
}

bool Plan::build(int h, int w, Program& prog, std::string* error) const {
    const Impl& im = *impl_;
    prog = Program();
    if (h < 1 || w < 1 || h > 16384 || w > 16384) return fail(error, "invalid size");

    std::vector<Tensor> tensors = im.tensors;
    tensors[static_cast<size_t>(im.inputTensor)].h = h;
    tensors[static_cast<size_t>(im.inputTensor)].w = w;
    for (const Op& op : im.ops) {
        Tensor& out = tensors[static_cast<size_t>(op.out)];
        const Tensor& a = tensors[static_cast<size_t>(op.in[0])];
        if (op.kind == OKind::Resize) {
            out.h = a.h * op.sh;
            out.w = a.w * op.sw;
        } else {
            out.h = a.h;
            out.w = a.w;
            for (int t : op.in) {
                if (tensors[static_cast<size_t>(t)].h != a.h || tensors[static_cast<size_t>(t)].w != a.w) {
                    return fail(error, "operands with different sizes");
                }
            }
            if (op.res >= 0 && (tensors[static_cast<size_t>(op.res)].h != a.h || tensors[static_cast<size_t>(op.res)].w != a.w)) {
                return fail(error, "residual with a different size");
            }
        }
        if (out.h > 16384 || out.w > 16384) return fail(error, "tensor too large");
    }

    auto c8 = [&](int t) { return static_cast<size_t>((tensors[static_cast<size_t>(t)].c + 7) / 8 * 8); };
    auto plane = [&](int t) { return planeFloats(tensors[static_cast<size_t>(t)].h, tensors[static_cast<size_t>(t)].w); };
    auto bytes = [&](int t) { return c8(t) * plane(t) * sizeof(float); };

    // ---- memory planning: first-fit per plane size, live ranges from `tensors` ------------------------
    // Memory is shared only between tensors with the same geometry: the zero borders must stay at the same
    // positions, and two sizes can have equal plane sizes but different row widths.
    auto geometry = [&](int t) {
        return (static_cast<uint64_t>(tensors[static_cast<size_t>(t)].h) << 32) | static_cast<uint64_t>(tensors[static_cast<size_t>(t)].w);
    };
    std::unordered_map<uint64_t, ClassAlloc> classes;
    std::vector<size_t> rel(tensors.size(), 0);
    std::vector<bool> allocated(tensors.size(), false);
    auto allocate = [&](int t) {
        ClassAlloc& ca = classes[geometry(t)];
        ca.unit = plane(t) * sizeof(float);
        rel[static_cast<size_t>(t)] = ca.alloc(bytes(t));
        allocated[static_cast<size_t>(t)] = true;
    };
    auto release = [&](int t) { classes[geometry(t)].release(rel[static_cast<size_t>(t)], bytes(t)); };

    allocate(im.inputTensor);
    for (size_t k = 0; k < im.ops.size(); ++k) {
        const Op& op = im.ops[k];
        allocate(op.out);
        std::vector<int> touched = op.in;
        if (op.res >= 0) touched.push_back(op.res);
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (int t : touched) {
            if (tensors[static_cast<size_t>(t)].lastUse == static_cast<int>(k) && t != im.outputTensor) release(t);
        }
        if (tensors[static_cast<size_t>(op.out)].lastUse < static_cast<int>(k)) release(op.out);  // never used
    }
    std::unordered_map<uint64_t, size_t> base;
    size_t cursor = 0;
    std::vector<uint64_t> keys;
    for (const auto& kv : classes) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (uint64_t key : keys) {
        const ClassAlloc& ca = classes[key];
        cursor = roundUp(cursor, ca.unit);
        base[key] = cursor;
        cursor += ca.top;
    }
    auto offset = [&](int t) { return base[geometry(t)] + rel[static_cast<size_t>(t)]; };
    prog.arenaBytes = roundUp(cursor, 256) + 256;  // slack: the convolution may read a few floats past a plane

    // ---- commands --------------------------------------------------------------------------------------
    double sinceBoundary = 0.0;
    auto bindArena = [&](int t) {
        Bind b;
        b.buffer = 0;
        b.offset = offset(t);
        b.size = bytes(t);
        prog.maxBindBytes = std::max(prog.maxBindBytes, b.size);
        return b;
    };
    auto bindArenaSlack = [&](int t) {
        Bind b = bindArena(t);
        b.size = std::min(b.size + 256, prog.arenaBytes - b.offset);
        return b;
    };
    auto push = [&](Command cmd) {
        sinceBoundary += cmd.flops;
        prog.flops += cmd.flops;
        prog.commands.push_back(cmd);
        if (sinceBoundary >= gSegmentFlops) {
            Command boundary;
            boundary.type = Command::kBoundary;
            prog.commands.push_back(boundary);
            sinceBoundary = 0.0;
        }
    };

    for (const Op& op : im.ops) {
        if (op.kind == OKind::Conv) {
            const ConvData& cd = im.convs[static_cast<size_t>(op.conv)];
            const Tensor& tin = tensors[static_cast<size_t>(op.in[0])];
            const int cout8 = (cd.cout + 7) / 8 * 8;
            const double rowFlops = 2.0 * cd.cin * cout8 * 9.0 * tin.w;
            int rows = static_cast<int>(gDispatchFlops / std::max(rowFlops, 1.0));
            rows = std::max(4, rows / 4 * 4);
            for (int y0 = 0; y0 < tin.h; y0 += rows) {
                const int n = std::min(rows, tin.h - y0);
                Command cmd;
                cmd.type = Command::kDispatch;
                cmd.kernel = Kernel::Conv3x3;
                cmd.push.i[0] = cd.cin;
                cmd.push.i[1] = tin.h;
                cmd.push.i[2] = tin.w;
                cmd.push.i[3] = cd.biasVec4;
                cmd.push.i[4] = y0;
                cmd.push.i[5] = op.res >= 0 ? 1 : 0;
                cmd.push.i[6] = static_cast<int32_t>(plane(op.in[0]));
                cmd.push.f[0] = op.alpha;
                cmd.push.f[1] = op.scale;
                cmd.bind[0] = bindArenaSlack(op.in[0]);
                cmd.bind[1].buffer = 1;
                cmd.bind[1].offset = cd.offset * sizeof(float);
                cmd.bind[1].size = cd.floats * sizeof(float);
                cmd.bind[2] = bindArena(op.out);
                cmd.bind[3] = op.res >= 0 ? bindArena(op.res) : bindArena(op.in[0]);
                cmd.gx = static_cast<uint32_t>((((tin.w + 3) / 4) + 15) / 16);
                cmd.gy = static_cast<uint32_t>((n + 3) / 4);
                cmd.gz = static_cast<uint32_t>(cout8 / 8);
                cmd.flops = rowFlops * n;
                prog.maxBindBytes = std::max(prog.maxBindBytes, cmd.bind[1].size);
                push(cmd);
            }
        } else if (op.kind == OKind::Elt) {
            const size_t count = c8(op.out) * plane(op.out);
            Command cmd;
            cmd.type = Command::kDispatch;
            cmd.kernel = Kernel::Eltwise;
            const size_t groups = (count + 63) / 64;
            const size_t gx = std::min<size_t>(groups, 32768);
            cmd.gx = static_cast<uint32_t>(gx);
            cmd.gy = static_cast<uint32_t>((groups + gx - 1) / gx);
            cmd.gz = 1;
            cmd.push.i[0] = static_cast<int32_t>(count);
            cmd.push.i[1] = op.in.size() == 2 ? 1 : 0;
            cmd.push.i[2] = static_cast<int32_t>(gx * 64);
            cmd.push.f[0] = op.alpha;
            cmd.push.f[1] = op.scale;
            cmd.bind[0] = bindArena(op.in[0]);
            cmd.bind[1] = bindArena(op.in.size() == 2 ? op.in[1] : op.in[0]);
            cmd.bind[2] = bindArena(op.out);
            cmd.bind[3] = cmd.bind[2];
            cmd.flops = static_cast<double>(count);
            push(cmd);
        } else if (op.kind == OKind::Concat) {
            size_t channelOffset = 0;
            for (int t : op.in) {
                Command cmd;
                cmd.type = Command::kCopy;
                cmd.copySrc = offset(t);
                cmd.copyDst = offset(op.out) + channelOffset * plane(op.out) * sizeof(float);
                cmd.copyBytes = bytes(t);
                cmd.flops = static_cast<double>(cmd.copyBytes) / 4.0;
                channelOffset += c8(t);
                push(cmd);
            }
        } else {
            const Tensor& tin = tensors[static_cast<size_t>(op.in[0])];
            const Tensor& tout = tensors[static_cast<size_t>(op.out)];
            Command cmd;
            cmd.type = Command::kDispatch;
            cmd.kernel = Kernel::Resize;
            cmd.push.i[0] = tin.h;
            cmd.push.i[1] = tin.w;
            cmd.push.i[2] = op.sh;
            cmd.push.i[3] = op.sw;
            cmd.push.i[4] = static_cast<int32_t>(plane(op.in[0]));
            cmd.push.i[5] = static_cast<int32_t>(plane(op.out));
            cmd.bind[0] = bindArena(op.in[0]);
            cmd.bind[1] = cmd.bind[0];
            cmd.bind[2] = bindArena(op.out);
            cmd.bind[3] = cmd.bind[2];
            cmd.gx = static_cast<uint32_t>((tout.w + 15) / 16);
            cmd.gy = static_cast<uint32_t>((tout.h + 3) / 4);
            cmd.gz = static_cast<uint32_t>(c8(op.out));
            cmd.flops = static_cast<double>(c8(op.out) * static_cast<size_t>(tout.h) * tout.w);
            push(cmd);
        }
    }

    const Tensor& tin = tensors[static_cast<size_t>(im.inputTensor)];
    const Tensor& tout = tensors[static_cast<size_t>(im.outputTensor)];
    prog.h = h;
    prog.w = w;
    prog.outH = tout.h;
    prog.outW = tout.w;
    prog.inChannels = tin.c;
    prog.outChannels = tout.c;
    prog.inputOffset = offset(im.inputTensor);
    prog.outputOffset = offset(im.outputTensor);
    prog.inPlane = plane(im.inputTensor);
    prog.outPlane = plane(im.outputTensor);
    if (prog.maxBindBytes == 0) return fail(error, "empty program");
    return true;
}

}  // namespace gpu
}  // namespace vva
