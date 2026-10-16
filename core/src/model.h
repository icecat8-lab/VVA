#ifndef VVA_MODEL_H
#define VVA_MODEL_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "json.h"

namespace vva {

enum class DType { F32, F16 };

inline size_t dtypeSize(DType t) { return t == DType::F32 ? 4 : 2; }
inline const char* dtypeName(DType t) { return t == DType::F32 ? "f32" : "f16"; }

struct TensorInfo {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    uint64_t offset = 0;  // bytes, relative to the start of the data blob
    uint64_t size = 0;    // bytes
};

struct IoInfo {
    std::string name;
    std::vector<int64_t> shape;  // -1 marks a dynamic dimension
};

struct Node {
    std::string op;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    json::Value attrs;  // always a JSON object
};

// CRC-32 (ISO-HDLC, the same as zlib.crc32 / Python binascii.crc32).
uint32_t crc32(const uint8_t* data, size_t size);

class Model {
public:
    // Memory-maps and validates a .vva v2 file. On failure returns nullptr and
    // fills `error` (if not null) with a human-readable message.
    static std::unique_ptr<Model> load(const std::string& path, std::string* error);

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const std::string& name() const { return name_; }
    int scale() const { return scale_; }  // 1 when the model does not declare one
    const std::vector<IoInfo>& inputs() const { return inputs_; }
    const std::vector<IoInfo>& outputs() const { return outputs_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::vector<Node>& nodes() const { return nodes_; }

    const TensorInfo* findTensor(const std::string& name) const;

    // Pointer into the memory map; 64-byte aligned; valid while the Model lives.
    const uint8_t* tensorData(const TensorInfo& tensor) const { return data_ + tensor.offset; }

    uint64_t dataSize() const { return dataSize_; }
    uint64_t fileSize() const { return mapSize_; }

private:
    Model() = default;
    bool parse(const uint8_t* base, size_t size, std::string* error);

    void* map_ = nullptr;
    size_t mapSize_ = 0;
    const uint8_t* data_ = nullptr;
    uint64_t dataSize_ = 0;

    std::string name_;
    int scale_ = 1;
    std::vector<IoInfo> inputs_;
    std::vector<IoInfo> outputs_;
    std::vector<TensorInfo> tensors_;
    std::vector<Node> nodes_;
};

}  // namespace vva

#endif  // VVA_MODEL_H
