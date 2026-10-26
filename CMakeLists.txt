cmake_minimum_required(VERSION 3.22.1)
project(VVA VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VVA_BUILD_JNI "Build the Android JNI library (used by Gradle)" OFF)
option(VVA_BUILD_SHARED "Build libvva.so" OFF)
option(VVA_BUILD_CLI "Build the vva-cli executable" OFF)


    set(_vva_vulkan_default ON)
else()
    set(_vva_vulkan_default OFF)
endif()
option(VVA_ENABLE_VULKAN "Link Vulkan for the diagnostic probe" ${_vva_vulkan_default})


find_package(Threads REQUIRED)
add_subdirectory(core)


set(VVA_PAGE_ALIGN_FLAGS "-Wl,-z,max-page-size=16384")

if(VVA_BUILD_JNI)
    add_library(vva_engine SHARED android/vva-android/src/main/cpp/vva_jni.cpp)
    target_link_libraries(vva_engine PRIVATE vva_core jnigraphics android log)
    target_link_options(vva_engine PRIVATE ${VVA_PAGE_ALIGN_FLAGS})
endif()

if(VVA_BUILD_SHARED)
    add_library(vva SHARED ${VVA_CORE_SOURCES})
    vva_configure(vva)
    target_link_options(vva PRIVATE ${VVA_PAGE_ALIGN_FLAGS})
endif()

if(VVA_BUILD_CLI)
    add_executable(vva-cli cli/main.cpp)
    target_link_libraries(vva-cli PRIVATE vva_core)
    target_compile_definitions(vva-cli PRIVATE VVA_VERSION="${PROJECT_VERSION}")
    if(ANDROID)
        target_link_options(vva-cli PRIVATE ${VVA_PAGE_ALIGN_FLAGS})
    endif()
endif()
