#pragma once
#include "types.hpp"
#include "renderer_types.hpp"
#include "pipelines.hpp"
#include "descriptor.hpp"
#include "context.hpp"
#include "scene.hpp"
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <array>

class RenderGraph;
class RendererAllocator;

struct GpuMesh {
    const Mesh* mesh;
    u32 vertex_offset, vertex_count;
    u32 index_offset, index_count;
    u32 instance_offset, instance_count;
};

struct GpuInstancedMesh {
    u32 diffuse_texture_idx{0};
    u32 normal_texture_idx{0};
};

struct GpuModel { 
    Handle<Model> model;
    u64 offset_to_gpu_meshes{0};
    u64 offset_to_instanced_meshes{0};
};

struct GpuScene {
    void render(vk::CommandBuffer cmd);
    
    std::vector<GpuModel> models;
    std::vector<GpuMesh> meshes;
    Buffer vertex_buffer;
    Buffer index_buffer;
    Buffer instance_buffer;
    Buffer indirect_commands_buffer;
    u32 draw_count; 
};

inline vk::ImageViewType to_vk_view_type(vk::ImageType type) {
    switch (type) {
        case vk::ImageType::e1D: { return vk::ImageViewType::e1D; }
        case vk::ImageType::e2D: { return vk::ImageViewType::e2D; }
        case vk::ImageType::e3D: { return vk::ImageViewType::e3D; }
        default: {
            spdlog::error("Unrecognized ImageType: {}", (u32)type);
            std::terminate();
        }
    }
}

inline vk::ImageAspectFlags deduce_vk_image_aspect(vk::Format format) {
    switch(format) {
        case vk::Format::eD16Unorm:
        case vk::Format::eD32Sfloat: {
            return vk::ImageAspectFlagBits::eDepth;
        }
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint: {
            return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        }
        default: {
            return vk::ImageAspectFlagBits::eColor;
        }
    }
}

class Renderer {
public:
    bool initialize();
    void setup_scene();
    void render();

private:
    bool initialize_vulkan();
    bool initialize_swapchain();
    bool initialize_frame_resources();
    bool initialize_imgui();
    bool initialize_render_passes();

    void load_waiting_textures(vk::CommandBuffer cmd);
    void draw_ui(vk::CommandBuffer cmd, vk::ImageView swapchain_view);

    FrameResources& get_frame_res() { return frames.at(frame_number % FRAMES_IN_FLIGHT); }

public:
    static constexpr inline u32 FRAMES_IN_FLIGHT = 2;

    u32 frame_number{0};
    u32 window_width{1024}, window_height{768};
    GLFWwindow *window{nullptr};
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    u32 graphics_queue_idx, presentation_queue_idx;
    vk::Queue graphics_queue, presentation_queue;
    VmaAllocator vma;
    vk::SwapchainKHR swapchain;
    std::vector<vk::Image> swapchain_images;
    std::vector<vk::ImageView> swapchain_views;
    vk::Format swapchain_format;
    std::array<FrameResources, FRAMES_IN_FLIGHT> frames{};
    struct {
        PFN_vkGetInstanceProcAddr get_instance_proc_addr;
        PFN_vkGetDeviceProcAddr get_device_proc_addr;
    } vulkan_function_pointers;
    
    DescriptorSet global_set, material_set;
    Texture3D voxel_albedo, voxel_normal, voxel_radiance;
    Texture2D depth_texture;
    Buffer global_buffer;

    Pipeline pp_default_lit;
    Pipeline pp_voxelize;
    Pipeline pp_merge_voxels;
    Pipeline pp_imgui;
    bool recompile_pipelines = false;

    DescriptorAllocator* descriptor_allocator;
    RenderGraph* render_graph;
    RendererAllocator* allocator;
    GpuScene render_scene;

    std::vector<std::function<void()>> deletion_queue;
};