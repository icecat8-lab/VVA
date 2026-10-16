cmake_minimum_required(VERSION 3.22.1)
project(VVA VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VVA_BUILD_JNI "Build the Android JNI library (used by Gradle)" OFF)
option(VVA_BUILD_SHARED "Build libvva.so" OFF)
option(VVA_BUILD_CLI "Build the vva-cli executable" OFF)

add_subdirectory(core)

if(VVA_BUILD_JNI)
    add_library(vva_engine SHARED android/vva-android/src/main/cpp/vva_jni.cpp)
    target_link_libraries(vva_engine PRIVATE vva_core android log)
endif()

if(VVA_BUILD_SHARED)
    add_library(vva SHARED ${VVA_CORE_SOURCES})
    target_include_directories(vva PUBLIC core/include PRIVATE core/src)
    target_link_libraries(vva PUBLIC vulkan android log)
endif()

if(VVA_BUILD_CLI)
    add_executable(vva-cli cli/main.cpp)
    target_link_libraries(vva-cli PRIVATE vva_core)
    target_compile_definitions(vva-cli PRIVATE VVA_VERSION="${PROJECT_VERSION}")
endif()
