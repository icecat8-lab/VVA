#include <NeuralNetworks.h>
#include <vector>
#include <array>

namespace vva {

class NNAPIBackendExt {
private:
    ANeuralNetworksModel* model = nullptr;
    uint32_t operandIndex = 0;

public:
    void buildAdd(uint32_t input1Idx, uint32_t input2Idx, uint32_t outIdx, int32_t activation) {
        ANeuralNetworksOperandType actType = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
        ANeuralNetworksModel_addOperand(model, &actType);
        uint32_t actIdx = operandIndex++;
        ANeuralNetworksModel_setOperandValue(model, actIdx, &activation, sizeof(int32_t));

        std::array<uint32_t, 3> inputs = {input1Idx, input2Idx, actIdx};
        std::array<uint32_t, 1> outputs = {outIdx};

        ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 
                                          inputs.size(), inputs.data(), 
                                          outputs.size(), outputs.data());
    }
};

}
