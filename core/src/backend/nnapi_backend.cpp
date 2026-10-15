#include <NeuralNetworks.h>
#include <vector>

namespace vva {

class NNAPIBackend {
private:
    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksCompilation* compilation = nullptr;
    ANeuralNetworksExecution* execution = nullptr;

public:
    bool initModel() {
        if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR) return false;
        return true;
    }

    void addFP16Tensor(uint32_t index, const std::vector<uint32_t>& dims) {
        ANeuralNetworksOperandType type;
        type.type = ANEURALNETWORKS_TENSOR_FLOAT16;
        type.dimensionCount = dims.size();
        type.dimensions = dims.data();
        type.scale = 0.0f;
        type.zeroPoint = 0;

        ANeuralNetworksModel_addOperand(model, &type);
    }

    bool compileAndExecute() {
        ANeuralNetworksModel_finish(model);
        ANeuralNetworksCompilation_create(model, &compilation);
        
        ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
        ANeuralNetworksCompilation_finish(compilation);

        ANeuralNetworksExecution_create(compilation, &execution);
        
        int result = ANeuralNetworksExecution_compute(execution);
        return result == ANEURALNETWORKS_NO_ERROR;
    }

    ~NNAPIBackend() {
        if (execution) ANeuralNetworksExecution_free(execution);
        if (compilation) ANeuralNetworksCompilation_free(compilation);
        if (model) ANeuralNetworksModel_free(model);
    }
};

}
