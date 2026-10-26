# VVA core: model reader, CPU executor and engine.
find_package(Threads REQUIRED)

set(VVA_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VVA_CORE_SOURCES
    ${VVA_CORE_DIR}/src/vva.cpp
    ${VVA_CORE_DIR}/src/model.cpp
    ${VVA_CORE_DIR}/src/exec.cpp)
set(VVA_CORE_SOURCES ${VVA_CORE_SOURCES} PARENT_SCOPE)

# Shared settings for every target that compiles the core sources.
function(vva_configure target)
    # Resolved relative to this file, so the function also works when called from the top level.
    set(core_dir ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
    target_include_directories(${target} PUBLIC ${core_dir}/include PRIVATE ${core_dir}/src)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_options(${target} PRIVATE -O3)
    target_link_libraries(${target} PUBLIC Threads::Threads)
    if(VVA_ENABLE_VULKAN)
        target_link_libraries(${target} PUBLIC vulkan)
    else()
        target_compile_definitions(${target} PRIVATE VVA_NO_VULKAN)
    endif()
    if(ANDROID)
        target_link_libraries(${target} PUBLIC android log)
    endif()
endfunction()

add_library(vva_core STATIC ${VVA_CORE_SOURCES})
vva_configure(vva_core)
