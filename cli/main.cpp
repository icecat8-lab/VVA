#include <cstdio>
#include <cstring>

#include "vva/vva.h"

#ifndef VVA_VERSION
#define VVA_VERSION "dev"
#endif

static void printUsage() {
    std::printf(
        "vva-cli %s\n"
        "Usage:\n"
        "  vva-cli --version   show the version\n"
        "  vva-cli --info      show Vulkan device information\n",
        VVA_VERSION);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    if (std::strcmp(argv[1], "--version") == 0) {
        std::printf("vva-cli %s\n", VVA_VERSION);
        return 0;
    }

    if (std::strcmp(argv[1], "--info") == 0) {
        vva::Engine engine;
        if (engine.hasVulkan()) {
            std::printf("Vulkan: available\nDevice: %s\n", engine.vulkanDeviceName().c_str());
            return 0;
        }
        std::printf("Vulkan: NOT available in this environment\n");
        return 2;
    }

    printUsage();
    return 1;
}
