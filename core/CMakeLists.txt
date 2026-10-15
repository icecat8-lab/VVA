find_program(GLSLC_COMPILER glslc 
    PATHS ${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}
)

set(SHADER_SRC 
    src/shaders/add_fp16.comp
    src/shaders/conv2d_fp16.comp
    src/shaders/instancenorm_fp32.comp
    src/shaders/leaky_relu_fp16.comp
    src/shaders/pixel_shuffle_fp16.comp
    src/shaders/prelu_fp16.comp
)

foreach(SHADER ${SHADER_SRC})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(SPIRV_OUT ${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.spv)
    
    add_custom_command(
        OUTPUT ${SPIRV_OUT}
        COMMAND ${GLSLC_COMPILER} -O -o ${SPIRV_OUT} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
    )
    list(APPEND SPIRV_FILES ${SPIRV_OUT})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_FILES})
add_dependencies(vva_core compile_shaders)
