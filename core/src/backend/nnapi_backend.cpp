#include <NeuralNetworks.h>
#include <vector>
#include <array>

namespace vva {

class NNAPIBackend {
private:
    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksCompilation* compilation = nullptr;
    ANeuralNetworksExecution* execution = nullptr;
    uint32_t operandIndex = 0;

public:
    bool initModel() {
        if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR) return false;
        return true;
    }

    uint32_t addFP16Tensor(const std::vector<uint32_t>& dims) {
        ANeuralNetworksOperandType type;
        type.type = ANEURALNETWORKS_TENSOR_FLOAT16;
        type.dimensionCount = dims.size();
        type.dimensions = dims.data();
        type.scale = 0.0f;
        type.zeroPoint = 0;

        ANeuralNetworksModel_addOperand(model, &type);
        return operandIndex++;
    }

    uint32_t addInt32Scalar() {
        ANeuralNetworksOperandType type;
        type.type = ANEURALNETWORKS_INT32;
        type.dimensionCount = 0;
        type.dimensions = nullptr;
        type.scale = 0.0f;
        type.zeroPoint = 0;

        ANeuralNetworksModel_addOperand(model, &type);
        return operandIndex++;
    }

    void buildConv2D(uint32_t inputIdx, uint32_t weightIdx, uint32_t biasIdx, uint32_t outIdx, 
                     int32_t pad, int32_t strideX, int32_t strideY, int32_t activation) {
        
        uint32_t padIdx = addInt32Scalar();
        uint32_t strideXIdx = addInt32Scalar();
        uint32_t strideYIdx = addInt32Scalar();
        uint32_t actIdx = addInt32Scalar();

        ANeuralNetworksModel_setOperandValue(model, padIdx, &pad, sizeof(int32_t));
        ANeuralNetworksModel_setOperandValue(model, strideXIdx, &strideX, sizeof(int32_t));
        ANeuralNetworksModel_setOperandValue(model, strideYIdx, &strideY, sizeof(int32_t));
        ANeuralNetworksModel_setOperandValue(model, actIdx, &activation, sizeof(int32_t));

        std::array<uint32_t, 7> inputs = {inputIdx, weightIdx, biasIdx, padIdx, padIdx, padIdx, padIdx};
        std::array<uint32_t, 1> outputs = {outIdx};

        ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 
                                          inputs.size(), inputs.data(), 
                                          outputs.size(), outputs.data());
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
