project(vxgi)

add_custom_target(copy_data ALL 
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/data
)

add_executable(vxgi
    main.cpp
)

target_compile_options(vxgi
    PRIVATE
)

target_compile_features(vxgi
    PRIVATE
        cxx_std_20
)

target_compile_definitions(vxgi
    PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
)

target_include_directories(vxgi
    PUBLIC
        $ENV{VULKAN_SDK}/include
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
        vulkan-1
)