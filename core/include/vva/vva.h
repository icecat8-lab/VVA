#ifndef VVA_H
#define VVA_H
#include <string>

namespace vva {
enum class DeviceType { CPU_NEON, GPU_VULKAN, NPU_NNAPI, AUTO };
class Engine {
public:
    Engine();
    ~Engine();
    bool loadModel(const std::string& modelPath);
    bool processImageTile(const uint8_t* inRGBA, int w, int h, uint8_t* outRGBA, DeviceType device);
};
}
#endif
