cmake_minimum_required(VERSION 3.22.1)
project("vva_engine")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core" vva_core)
add_library(vva_engine SHARED vva_jni.cpp)

target_link_libraries(vva_engine vva_core android log vulkan)
