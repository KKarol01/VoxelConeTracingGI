#pragma once
#include "types.hpp"
#include "renderer_types.hpp"
#include "context.hpp"
#include "input.hpp"
#include "scene.hpp"
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <array>
#include <unordered_map>
#include <memory>
#include <optional>
#include <variant>

class RenderGraph;

struct GpuMesh {
    const Mesh* mesh;
    u64 vertex_offset, vertex_count;
    u64 index_offset, index_count;
    u64 instance_offset, instance_count;
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
    std::vector<GpuModel> models;
    std::vector<GpuMesh> meshes;
    Handle<GpuBuffer> vertex_buffer;
    Handle<GpuBuffer> index_buffer;
    Handle<GpuBuffer> instance_buffer;
};

template<typename VkObject> struct VulkanObjectType;
template<> struct VulkanObjectType<vk::SwapchainKHR> { static inline constexpr vk::ObjectType type = vk::ObjectType::eSwapchainKHR; };
template<> struct VulkanObjectType<vk::CommandPool> { static inline constexpr vk::ObjectType type = vk::ObjectType::eCommandPool; };
template<> struct VulkanObjectType<vk::CommandBuffer> { static inline constexpr vk::ObjectType type = vk::ObjectType::eCommandBuffer; };
template<> struct VulkanObjectType<vk::Fence> { static inline constexpr vk::ObjectType type = vk::ObjectType::eFence; };
template<> struct VulkanObjectType<vk::Semaphore> { static inline constexpr vk::ObjectType type = vk::ObjectType::eSemaphore; };
template<> struct VulkanObjectType<vk::Buffer> { static inline constexpr vk::ObjectType type = vk::ObjectType::eBuffer; };
template<> struct VulkanObjectType<vk::Pipeline> { static inline constexpr vk::ObjectType type = vk::ObjectType::ePipeline; };
template<> struct VulkanObjectType<vk::PipelineLayout> { static inline constexpr vk::ObjectType type = vk::ObjectType::ePipelineLayout; };
template<> struct VulkanObjectType<vk::ShaderModule> { static inline constexpr vk::ObjectType type = vk::ObjectType::eShaderModule; };
template<> struct VulkanObjectType<vk::DescriptorSetLayout> { static inline constexpr vk::ObjectType type = vk::ObjectType::eDescriptorSetLayout; };
template<> struct VulkanObjectType<vk::DescriptorPool> { static inline constexpr vk::ObjectType type = vk::ObjectType::eDescriptorPool; };
template<> struct VulkanObjectType<vk::Image> { static inline constexpr vk::ObjectType type = vk::ObjectType::eImage; };
template<> struct VulkanObjectType<vk::ImageView> { static inline constexpr vk::ObjectType type = vk::ObjectType::eImageView; };

template<typename T> void set_debug_name(vk::Device device, T object, std::string_view name);

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

class RendererAllocator {
    struct UploadJob {
        std::variant<Handle<TextureStorage>, Handle<GpuBuffer>> storage;
        std::vector<std::byte> data;
    };
    
public:
    explicit RendererAllocator(vk::Device device, VmaAllocator vma): device(device), vma(vma) {}

    Handle<TextureStorage> create_texture_storage(std::string_view label, const vk::ImageCreateInfo& info, u64 size_bytes = 0ull, const void* optional_data = nullptr);
    Handle<GpuBuffer> create_buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, u64 size_bytes, const void* optional_data = nullptr);

    TextureStorage& get_texture(Handle<TextureStorage> handle) { return find_with_handle(handle, textures); }
    GpuBuffer& get_buffer(Handle<GpuBuffer> handle) { return find_with_handle(handle, buffers); }

private:
    template<typename T> T& find_with_handle(Handle<T> handle, std::vector<T>& storage) {
        return *std::lower_bound(storage.begin(), storage.end(), handle);
    }
    GpuBuffer* create_buffer_ptr(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, u64 size_bytes);

    vk::Device device;
    VmaAllocator vma;
    std::vector<TextureStorage> textures;
    std::vector<GpuBuffer> buffers;
    std::vector<UploadJob> jobs;
};

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
    std::vector<GpuBuffer*> buffers;
    std::vector<TextureStorage*> texture_storages;
    struct {
        PFN_vkGetInstanceProcAddr get_instance_proc_addr;
        PFN_vkGetDeviceProcAddr get_device_proc_addr;
    } vulkan_function_pointers;
    
    vk::DescriptorSetLayout global_set_layout;
    vk::DescriptorSetLayout default_lit_set_layout;
    vk::DescriptorSetLayout voxelize_set_layout;
    vk::DescriptorSetLayout merge_voxels_set_layout;
    vk::DescriptorSetLayout material_set_layout;

    vk::DescriptorPool global_desc_pool;
    DescriptorSet global_set;
    DescriptorSet material_set;

    Texture3D voxel_albedo, voxel_normal, voxel_radiance;
    Texture2D depth_texture;
    vk::ImageView depth_texture_view;
    GpuBuffer* global_buffer;

    Pipeline pp_default_lit;
    Pipeline pp_voxelize;
    Pipeline pp_merge_voxels;
    Pipeline pp_imgui;
    bool recompile_pipelines = false;

    RenderGraph* render_graph;
    RendererAllocator* allocator;
    GpuScene render_scene;

    std::vector<std::function<void()>> deletion_queue;
};

template<typename T> void set_debug_name(vk::Device device, T object, std::string_view name) {
    device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        VulkanObjectType<T>::type, (u64)static_cast<T::NativeType>(object), name.data()
    }, vk::DispatchLoaderDynamic{
            get_context().renderer->instance,
            get_context().renderer->vulkan_function_pointers.get_instance_proc_addr,
            get_context().renderer->device,
            get_context().renderer->vulkan_function_pointers.get_device_proc_addr
        }
    );
}