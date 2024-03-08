cmake_minimum_required(VERSION 3.6)
project(setup)
find_package(Vulkan REQUIRED)

add_subdirectory(third_party/glm SYSTEM)
target_compile_definitions(glm INTERFACE 
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
    GLM_FORCE_AVX2
)

add_subdirectory(third_party/vk-bootstrap)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_VULKAN_STATIC OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)

set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/spdlog)
target_compile_definitions(spdlog PRIVATE FMT_EXCEPTIONS=0)

set(VMA_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)
set(VMA_STATIC_VULKAN_FUNCTIONS OFF CACHE BOOL "" FORCE)
set(VMA_DYNAMIC_VULKAN_FUNCTIONS ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/VulkanMemoryAllocator)
add_library(vma STATIC source/build_vma.cpp)
target_include_directories(vma PUBLIC third_party/VulkanMemoryAllocator/include ${Vulkan_INCLUDE_DIRS})

add_subdirectory(third_party/fastgltf)

add_library(stb STATIC source/build_stb.cpp)

add_library(imgui STATIC
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_vulkan.cpp
    third_party/imgui/backends/imgui_impl_glfw.cpp
)
target_include_directories(imgui PRIVATE third_party/imgui ${Vulkan_INCLUDE_DIRS} third_party/glfw/include)

add_subdirectory(source)