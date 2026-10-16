# Android only: links against the NDK's Vulkan stub library.
set(VVA_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vva.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model.cpp)
set(VVA_CORE_SOURCES ${VVA_CORE_SOURCES} PARENT_SCOPE)

add_library(vva_core STATIC ${VVA_CORE_SOURCES})
target_include_directories(vva_core PUBLIC include PRIVATE src)
target_link_libraries(vva_core PUBLIC vulkan android log)
