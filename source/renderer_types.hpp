#pragma once
#include "types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vk_mem_alloc.h>
#include <vector>
#include <array>

enum class DescriptorType {
    SampledImage, StorageImage, Sampler, UniformBuffer, StorageBuffer, CombinedImageSampler
};

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

struct DescriptorBinding {
    std::string name;
    DescriptorType type;
    u32 binding;
};

struct DescriptorLayout {
    vk::DescriptorSetLayout layout;
    std::vector<DescriptorBinding> bindings;
};

struct ShaderResource {
    u32 descriptor_set;
    DescriptorBinding resource;
};

struct Shader {
    std::string path;
    vk::ShaderModule module;
    std::vector<ShaderResource> resources;
};

struct PipelineLayout {
    static inline constexpr u32 MAX_DESCRIPTOR_SET_COUNT = 4u;

    const DescriptorBinding* find_binding(std::string_view name) const {
        for(auto& s : descriptor_sets) {
            for(auto& b : s.bindings) {
                if(b.name == name) { return &b; }
            }
        }
        return nullptr;
    }

    vk::PipelineLayout layout;
    std::array<DescriptorLayout, MAX_DESCRIPTOR_SET_COUNT> descriptor_sets;
};

enum class PipelineType { None, Graphics, Compute };

struct Pipeline {
    vk::Pipeline pipeline;
    PipelineLayout layout;
    PipelineType type{PipelineType::None};
};

struct GpuBuffer {
    vk::Buffer buffer;
    void* data;
    u64 size;
    VmaAllocation alloc;
};

struct FrameResources {
    vk::CommandPool pool;
    vk::CommandBuffer cmd;
    vk::Semaphore swapchain_semaphore, rendering_semaphore;
    vk::Fence in_flight_fence;
};

struct TextureStorage {
    vk::ImageType type;
    u32 width, height, depth;
    u32 mips, layers;
    vk::Format format;
    vk::ImageLayout current_layout;
    vk::Image image;
    vk::ImageAspectFlags aspect;
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

struct GLFWwindow;
struct Window {
    u32 width{1024}, height{768};
    GLFWwindow* window{nullptr};
};
