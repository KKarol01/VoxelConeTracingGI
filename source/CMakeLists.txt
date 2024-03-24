project(vxgi)

add_custom_target(copy_data ALL 
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/data
)

add_executable(vxgi
    main.cpp
    renderer.cpp
    context.cpp
    input.cpp
    render_graph.cpp
    pipelines.cpp
)

target_compile_options(vxgi
    PRIVATE
        -fopenmp
	$<$<CONFIG:Relase>: -O2>
)

target_compile_features(vxgi
    PRIVATE
        cxx_std_20
)

target_compile_definitions(vxgi
    PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        VC_EXTRALEAN
        NOMINMAX
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)

target_precompile_headers(vxgi
    PRIVATE
        <vulkan/vulkan.hpp>
        <spdlog/spdlog.h>
        <shaderc/shaderc.hpp>
        <spirv_cross/spirv_cross.hpp>
)

target_include_directories(vxgi
    PUBLIC
        $ENV{VULKAN_SDK}/include
        ${CMAKE_SOURCE_DIR}/third_party
)

target_link_directories(vxgi
    PRIVATE
        $ENV{VULKAN_SDK}/lib
)

target_link_libraries(vxgi
    PRIVATE
        glm
        vk-bootstrap
        glfw
        spdlog
        vma
        fastgltf
        stb
        imgui
        $<$<CONFIG:Debug>: shaderc_sharedd>
        $<$<CONFIG:Release>: shaderc_shared>
        $<$<CONFIG:Debug>: spirv-cross-cored>
        $<$<CONFIG:Release>: spirv-cross-core>
        libomp
)
