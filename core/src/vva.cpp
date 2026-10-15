#include "vva/vva.h"
namespace vva {
Engine::Engine() {}
Engine::~Engine() {}
bool Engine::loadModel(const std::string& path) { return true; }
bool Engine::processImageTile(const uint8_t*, int, int, uint8_t*, DeviceType) { return true; }
}
