#include "model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <unordered_set>
#include <utility>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The .vva reader requires a little-endian host"
#endif

namespace vva {

namespace {

constexpr size_t kHeaderSize = 64;
constexpr uint64_t kAlign = 64;
constexpr uint64_t kMaxMetaSize = 64ull * 1024 * 1024;
constexpr size_t kMaxEntries = 200000;

struct CrcTable {
    uint32_t t[256];
    CrcTable() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
    }
};

uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

uint64_t rd64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

bool inRange(uint64_t offset, uint64_t length, uint64_t total) {
    return offset <= total && length <= total - offset;
}

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool asInt(const json::Value* v, int64_t& out) {
    if (!v || !v->isNumber()) return false;
    const double d = v->number;
    if (std::floor(d) != d || std::fabs(d) > 9.0e15) return false;
    out = static_cast<int64_t>(d);
    return true;
}

bool asString(const json::Value* v, std::string& out) {
    if (!v || !v->isString()) return false;
    out = v->string;
    return true;
}

bool asIntArray(const json::Value* v, std::vector<int64_t>& out) {
    if (!v || !v->isArray()) return false;
    out.clear();
    for (const json::Value& e : v->array) {
        int64_t i = 0;
        if (!asInt(&e, i)) return false;
        out.push_back(i);
    }
    return true;
}

bool asStringArray(const json::Value* v, std::vector<std::string>& out) {
    if (!v || !v->isArray()) return false;
    out.clear();
    for (const json::Value& e : v->array) {
        if (!e.isString()) return false;
        out.push_back(e.string);
    }
    return true;
}

bool parseIoList(const json::Value* list, const char* what, std::vector<IoInfo>& out, std::string* error) {
    if (!list || !list->isArray() || list->array.empty()) {
        return fail(error, std::string("'") + what + "' must be a non-empty array");
    }
    if (list->array.size() > kMaxEntries) return fail(error, std::string("too many entries in '") + what + "'");
    std::unordered_set<std::string> seen;
    for (const json::Value& e : list->array) {
        IoInfo io;
        if (!e.isObject() || !asString(e.find("name"), io.name) || io.name.empty() ||
            !asIntArray(e.find("shape"), io.shape)) {
            return fail(error, std::string("invalid entry in '") + what + "' (needs 'name' and integer 'shape')");
        }
        for (int64_t d : io.shape) {
            if (d < -1) return fail(error, std::string("invalid dimension in '") + what + "' entry '" + io.name + "'");
        }
        if (!seen.insert(io.name).second) {
            return fail(error, std::string("duplicate name '") + io.name + "' in '" + what + "'");
        }
        out.push_back(std::move(io));
    }
    return true;
}

}  // namespace

uint32_t crc32(const uint8_t* data, size_t size) {
    static const CrcTable table;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) c = table.t[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

Model::~Model() {
    if (map_ != nullptr) munmap(map_, mapSize_);
}

const TensorInfo* Model::findTensor(const std::string& name) const {
    for (const TensorInfo& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

std::unique_ptr<Model> Model::load(const std::string& path, std::string* error) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fail(error, "cannot open '" + path + "': " + std::strerror(errno));
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        fail(error, "cannot read '" + path + "' (empty or unreadable file)");
        return nullptr;
    }
    const size_t size = static_cast<size_t>(st.st_size);
    void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) {
        fail(error, "cannot map '" + path + "': " + std::strerror(errno));
        return nullptr;
    }

    std::unique_ptr<Model> model(new Model());
    model->map_ = map;
    model->mapSize_ = size;
    if (!model->parse(static_cast<const uint8_t*>(map), size, error)) return nullptr;
    return model;
}

bool Model::parse(const uint8_t* base, size_t size, std::string* error) {
    if (size < kHeaderSize) return fail(error, "file is too small to be a .vva file");

    if (std::memcmp(base, "VVA1", 4) == 0) {
        return fail(error, "this is a legacy v1 .vva file; convert the model again with the current converter");
    }
    if (std::memcmp(base, "VVA2", 4) != 0) return fail(error, "not a .vva file (bad magic)");

    const uint32_t headerSize = rd32(base + 4);
    const uint32_t flags = rd32(base + 8);
    const uint64_t metaOffset = rd64(base + 16);
    const uint64_t metaSize = rd64(base + 24);
    const uint64_t dataOffset = rd64(base + 32);
    const uint64_t dataSize = rd64(base + 40);
    const uint64_t fileSize = rd64(base + 48);
    const uint32_t metaCrc = rd32(base + 56);

    if (headerSize != kHeaderSize) return fail(error, "unsupported header size");
    if (flags != 0) return fail(error, "unsupported flags in header");
    if (fileSize != size) {
        return fail(error, "file size mismatch (header says " + std::to_string(fileSize) + " bytes, file has " +
                               std::to_string(size) + "): the file is truncated or corrupted");
    }
    if (metaOffset < kHeaderSize || metaSize == 0 || metaSize > kMaxMetaSize || !inRange(metaOffset, metaSize, size)) {
        return fail(error, "metadata block is out of range");
    }
    if (dataOffset % kAlign != 0 || !inRange(dataOffset, dataSize, size) || dataOffset + dataSize != size) {
        return fail(error, "data block is out of range or misaligned");
    }
    if (metaOffset + metaSize > dataOffset) return fail(error, "metadata and data blocks overlap");
    if (crc32(base + metaOffset, static_cast<size_t>(metaSize)) != metaCrc) {
        return fail(error, "metadata checksum mismatch: the file is corrupted");
    }

    json::Value root;
    std::string jsonError;
    if (!json::parse(reinterpret_cast<const char*>(base + metaOffset), static_cast<size_t>(metaSize), root, jsonError)) {
        return fail(error, "invalid metadata: " + jsonError);
    }
    if (!root.isObject()) return fail(error, "metadata must be a JSON object");

    int64_t version = 0;
    if (!asInt(root.find("version"), version) || version != 2) {
        return fail(error, "unsupported metadata version (this reader supports version 2)");
    }

    // Optional descriptive block.
    if (const json::Value* meta = root.find("meta")) {
        if (!meta->isObject()) return fail(error, "'meta' must be an object");
        if (const json::Value* n = meta->find("name")) {
            if (!asString(n, name_)) return fail(error, "'meta.name' must be a string");
        }
        if (const json::Value* s = meta->find("scale")) {
            int64_t scale = 0;
            if (!asInt(s, scale) || scale < 1 || scale > 64) return fail(error, "'meta.scale' must be an integer in 1..64");
            scale_ = static_cast<int>(scale);
        }
        if (const json::Value* t = meta->find("task")) {
            if (!asString(t, task_) || (task_ != "image" && task_ != "classify")) {
                return fail(error, "'meta.task' must be \"image\" or \"classify\"");
            }
        }
        if (const json::Value* c = meta->find("classes")) {
            if (!asStringArray(c, classes_) || classes_.empty() || classes_.size() > 4096) {
                return fail(error, "'meta.classes' must be a non-empty array of strings");
            }
            for (const std::string& label : classes_) {
                if (label.size() > 256) return fail(error, "'meta.classes' contains a label longer than 256 bytes");
            }
        }
        if (const json::Value* sz = meta->find("input_size")) {
            int64_t v = 0;
            if (!asInt(sz, v) || v < 8 || v > 4096) return fail(error, "'meta.input_size' must be an integer in 8..4096");
            inputSize_ = static_cast<int>(v);
        }
        for (int which = 0; which < 2; ++which) {
            const char* key = which == 0 ? "mean" : "std";
            const json::Value* arr = meta->find(key);
            if (!arr) continue;
            if (!arr->isArray() || arr->array.size() != 3) {
                return fail(error, std::string("'meta.") + key + "' must be an array of three numbers");
            }
            for (int i = 0; i < 3; ++i) {
                const json::Value& e = arr->array[static_cast<size_t>(i)];
                if (!e.isNumber() || !std::isfinite(e.number) || (which == 1 && e.number <= 0.0)) {
                    return fail(error, std::string("'meta.") + key + "' must hold finite numbers (std > 0)");
                }
                (which == 0 ? mean_ : std_)[i] = static_cast<float>(e.number);
            }
        }
    }

    if (!parseIoList(root.find("inputs"), "inputs", inputs_, error)) return false;
    if (!parseIoList(root.find("outputs"), "outputs", outputs_, error)) return false;

    // Tensors (constants stored in the data block).
    const json::Value* tensorList = root.find("tensors");
    if (!tensorList || !tensorList->isArray()) return fail(error, "'tensors' must be an array");
    if (tensorList->array.size() > kMaxEntries) return fail(error, "too many tensors");
    std::unordered_set<std::string> available;
    for (const IoInfo& in : inputs_) available.insert(in.name);

    for (const json::Value& e : tensorList->array) {
        TensorInfo t;
        std::string dtype;
        int64_t offset = 0, bytes = 0;
        if (!e.isObject() || !asString(e.find("name"), t.name) || t.name.empty() ||
            !asString(e.find("dtype"), dtype) || !asIntArray(e.find("shape"), t.shape) ||
            !asInt(e.find("offset"), offset) || !asInt(e.find("size"), bytes) || offset < 0 || bytes < 0) {
            return fail(error, "invalid tensor entry (needs name, dtype, shape, offset, size)");
        }
        if (dtype == "f32") t.dtype = DType::F32;
        else if (dtype == "f16") t.dtype = DType::F16;
        else return fail(error, "tensor '" + t.name + "' has unsupported dtype '" + dtype + "'");

        uint64_t numel = 1;
        for (int64_t d : t.shape) {
            if (d < 0) return fail(error, "tensor '" + t.name + "' has a negative dimension");
            if (d != 0 && numel > (1ull << 40) / static_cast<uint64_t>(d)) {
                return fail(error, "tensor '" + t.name + "' is too large");
            }
            numel *= static_cast<uint64_t>(d);
        }
        t.offset = static_cast<uint64_t>(offset);
        t.size = static_cast<uint64_t>(bytes);
        if (t.size != numel * dtypeSize(t.dtype)) {
            return fail(error, "tensor '" + t.name + "' size does not match its shape and dtype");
        }
        if (t.offset % kAlign != 0 || !inRange(t.offset, t.size, dataSize)) {
            return fail(error, "tensor '" + t.name + "' lies outside the data block or is misaligned");
        }
        if (!available.insert(t.name).second) {
            return fail(error, "duplicate tensor or input name '" + t.name + "'");
        }
        tensors_.push_back(std::move(t));
    }

    // Graph nodes: single static assignment, topologically ordered.
    const json::Value* nodeList = root.find("nodes");
    if (!nodeList || !nodeList->isArray() || nodeList->array.empty()) {
        return fail(error, "'nodes' must be a non-empty array");
    }
    if (nodeList->array.size() > kMaxEntries) return fail(error, "too many nodes");

    size_t index = 0;
    for (const json::Value& e : nodeList->array) {
        const std::string where = "node #" + std::to_string(index);
        Node n;
        if (!e.isObject() || !asString(e.find("op"), n.op) || n.op.empty() ||
            !asStringArray(e.find("inputs"), n.inputs) || !asStringArray(e.find("outputs"), n.outputs) ||
            n.outputs.empty()) {
            return fail(error, where + " is invalid (needs 'op', 'inputs', non-empty 'outputs')");
        }
        if (const json::Value* nm = e.find("name")) {
            if (!asString(nm, n.name)) return fail(error, where + " has a non-string 'name'");
        }
        if (const json::Value* a = e.find("attrs")) {
            if (!a->isObject()) return fail(error, where + " ('" + n.op + "') has non-object 'attrs'");
            n.attrs = *a;
        } else {
            n.attrs.type = json::Value::Type::Object;
        }
        for (const std::string& in : n.inputs) {
            if (available.find(in) == available.end()) {
                return fail(error, where + " ('" + n.op + "') uses '" + in +
                                       "' which is not defined (graph must be topologically ordered)");
            }
        }
        for (const std::string& out : n.outputs) {
            if (out.empty() || !available.insert(out).second) {
                return fail(error, where + " ('" + n.op + "') defines '" + out + "' more than once");
            }
        }
        nodes_.push_back(std::move(n));
        ++index;
    }

    for (const IoInfo& out : outputs_) {
        if (available.find(out.name) == available.end()) {
            return fail(error, "graph output '" + out.name + "' is not produced by any node");
        }
    }

    data_ = base + dataOffset;
    dataSize_ = dataSize;
    return true;
}

}  // namespace vva
