#pragma once
#include "types.hpp"
#include "context.hpp"
#include "input.hpp"
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <array>
#include <unordered_map>

struct DescriptorInfo {
    enum Resource : u8 { None=0, Buffer=1, Image=2, Sampler=3 };
    
    constexpr DescriptorInfo() {}
    constexpr DescriptorInfo(vk::DescriptorType type, vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size): 
        buffer_info(buffer, offset, size),
        type(type),
        resource(Buffer) {}
    constexpr DescriptorInfo(vk::DescriptorType type, vk::ImageView image, vk::ImageLayout layout): 
        image_info({}, image, layout),
        type(type),
        resource(Image) {}
    constexpr DescriptorInfo(vk::DescriptorType type, vk::Sampler sampler): 
        image_info(sampler, {}, {}),
        type(type),
        resource(Sampler) {}

    union {
        vk::DescriptorBufferInfo buffer_info;
        vk::DescriptorImageInfo image_info;
        vk::DescriptorImageInfo sampler_info;
    };
    vk::DescriptorType type;
    Resource resource{None};
};

struct DescriptorSet {
    DescriptorSet() = default;
    DescriptorSet(vk::Device device, vk::DescriptorPool pool, vk::DescriptorSetLayout layout);

    void update_bindings(vk::Device device, u32 dst_binding, u32 dst_arr_element, std::span<DescriptorInfo> infos);
    
    vk::DescriptorSet set;
};

class Renderer;

struct Pipeline {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
};

class PipelineBuilder {
public:
    PipelineBuilder(const Renderer* renderer): renderer(renderer) {}

    PipelineBuilder& with_shaders(const std::vector<std::pair<vk::ShaderStageFlagBits, vk::ShaderModule>>& shaders) {
        this->shaders = shaders;        
        return *this;
    }

    PipelineBuilder &with_vertex_input(
        const std::vector<vk::VertexInputBindingDescription> &bindings,
        const std::vector<vk::VertexInputAttributeDescription> &attributes) {
        this->bindings = bindings;
        this->attributes = attributes;
        return *this;
    }

    PipelineBuilder& with_culling(vk::CullModeFlagBits culling, vk::FrontFace front_face) {
        cull_mode = culling;
        this->front_face = front_face;
        return *this;
    } 

    PipelineBuilder& with_depth_testing(bool depth_test, bool depth_write, vk::CompareOp depth_compare) {
        this->depth_test = depth_test;
        this->depth_write = depth_write;
        this->depth_compare = depth_compare;
        return *this;
    }

    PipelineBuilder& with_layout(vk::DescriptorSetLayout set) {
        set_layouts.push_back(set);
        return *this;
    }

    PipelineBuilder& with_color_attachments(const std::vector<vk::Format>& formats) {
        color_attachment_formats = formats;
        return *this;
    }

    PipelineBuilder& with_depth_attachment(vk::Format format) {
        depth_attachment_format = format;
        return *this;
    }

    Pipeline build_graphics(std::string_view label);

    Pipeline build_compute(std::string_view label);

private:
    const Renderer* renderer;
    std::vector<std::pair<vk::ShaderStageFlagBits, vk::ShaderModule>> shaders;
    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;
    vk::CullModeFlagBits cull_mode{vk::CullModeFlagBits::eBack};
    vk::FrontFace front_face{vk::FrontFace::eCounterClockwise};
    bool depth_test{true};
    bool depth_write{true};
    vk::CompareOp depth_compare{vk::CompareOp::eLess};
    std::vector<vk::DescriptorSetLayout> set_layouts;
    std::vector<vk::Format> color_attachment_formats;
    vk::Format depth_attachment_format{vk::Format::eUndefined};
};

struct GpuBuffer {
    vk::Buffer buffer;
    void* data;
    u64 size;
    VmaAllocation alloc;
};

struct Window {
    u32 width{1024}, height{768};
    GLFWwindow* window{nullptr};
};

struct FrameResources {
    vk::CommandPool pool;
    vk::CommandBuffer cmd;
    vk::Semaphore swapchain_semaphore, rendering_semaphore;
    vk::Fence in_flight_fence;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

struct alignas(16) Material {
    glm::vec4 ambient_color{0.0f};
    glm::vec4 diffuse_color{1.0f};
    glm::vec4 specular_color{1.0f};
};

struct Mesh {
    std::string name;
    Material material;
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
};

struct Model {
    std::vector<Mesh> meshes;
};

struct GpuMesh {
    Mesh* mesh;
    u32 vertex_offset, vertex_count;
    u32 index_offset, index_count;
};

struct Scene {
    std::vector<GpuMesh> models;
    GpuBuffer* vertex_buffer;
    GpuBuffer* index_buffer;
    GpuBuffer* material_buffer;
};

struct TextureStorage {
    vk::ImageType type;
    u32 width, height, depth;
    u32 mips, layers;
    vk::Format format;
    vk::ImageLayout current_layout;
    vk::Image image;
    VmaAllocation alloc;
};

struct Texture2D {
    Texture2D() = default;
    Texture2D(u32 width, u32 height, vk::Format format, u32 mips, vk::ImageUsageFlags usage); 

    TextureStorage* storage{};
};

struct Texture3D {
    Texture3D() = default;
    Texture3D(u32 width, u32 height, u32 depth, vk::Format format, u32 mips, vk::ImageUsageFlags usage); 

    TextureStorage* storage{};
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

class Renderer {
public:
    bool initialize();

    bool load_model_from_file(std::string_view name, const std::filesystem::path& path);

    void setup_scene();

    void draw();

    TextureStorage* create_texture_storage(const vk::ImageCreateInfo& image_info) {
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        
        VkImage image;
        VmaAllocation alloc;
        VmaAllocationInfo alloc_i;
        vmaCreateImage(vma, (VkImageCreateInfo*)&image_info, &alloc_info, &image, &alloc, &alloc_i);

        texture_storages.push_back(new TextureStorage{});
        auto& ts = *texture_storages.back();
        ts.type = image_info.imageType;
        ts.width = image_info.extent.width;
        ts.height = image_info.extent.height;
        ts.depth = image_info.extent.depth;
        ts.mips = image_info.mipLevels;
        ts.layers = image_info.arrayLayers;
        ts.format = image_info.format;
        ts.current_layout = vk::ImageLayout::eUndefined;
        ts.image = image;
        ts.alloc = alloc;

        return &ts;
    }

private:
    bool initialize_vulkan();
    bool initialize_swapchain();
    bool initialize_frame_resources();
    bool initialize_imgui();
    bool initialize_render_passes();

    void draw_ui();

    template<typename T> GpuBuffer* create_buffer(std::string_view label, vk::BufferUsageFlags usage, std::span<T> data) {
        vk::BufferCreateInfo buffer_info{
            {},
            (VkDeviceSize)data.size_bytes(),
            usage
        };
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer buffer;
        VmaAllocation alloc;
        VmaAllocationInfo alloc_i;
        vmaCreateBuffer(vma, (VkBufferCreateInfo*)&buffer_info, &alloc_info, &buffer, &alloc, &alloc_i);

        memcpy(alloc_i.pMappedData, data.data(), data.size_bytes());

        buffers.push_back(new GpuBuffer{
            .buffer = buffer,
            .data = alloc_i.pMappedData,
            .size = data.size_bytes(),
            .alloc = alloc
        });
        set_debug_name(device, buffers.back()->buffer, label);

        return buffers.back();
    }

    FrameResources& get_frame_res() { return frames.at(frame_number % FRAMES_IN_FLIGHT); }
    std::vector<u32> compile_shader(std::string_view filename, std::string_view file);

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
    DescriptorSet default_lit_set;
    DescriptorSet voxelize_set;
    DescriptorSet merge_voxels_set;
    DescriptorSet material_set;

    Texture3D voxel_albedo, voxel_normal, voxel_radiance;
    Texture2D depth_texture;
    vk::ImageView depth_texture_view;
    GpuBuffer* global_buffer;

    Pipeline pp_default_lit;
    Pipeline pp_voxelize;
    Pipeline pp_merge_voxels;
    bool recompile_pipelines = false;

    vk::QueryPool query_pool;
    float tick_length;
    
    std::unordered_map<std::string, Model> models;
    Scene scene;
    Camera camera;
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