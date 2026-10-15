# Android only: links against the NDK's Vulkan stub library.
add_library(vva_core STATIC src/vva.cpp)
target_include_directories(vva_core PUBLIC include PRIVATE src)
target_link_libraries(vva_core PUBLIC vulkan android log)
