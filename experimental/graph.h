#ifndef VVA_GRAPH_H
#define VVA_GRAPH_H

#include <vector>
#include <string>
#include <memory>

namespace vva {

struct Tensor {
    std::vector<int> shape;
    std::string precision;
    void* data;
};

class Graph {
public:
    std::vector<Tensor> tensors;
    
    void addTensor(const std::vector<int>& shape, const std::string& precision, void* data_ptr) {
        tensors.push_back({shape, precision, data_ptr});
    }

    void buildExecutionOrder() {
    }
};

}
#endif
