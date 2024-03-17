#pragma once
#include "types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vk_mem_alloc.h>
#include <vector>
#include <array>

enum class DescriptorType {
    None, SampledImage, StorageImage, Sampler, UniformBuffer, StorageBuffer
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

struct Pipeline {
    vk::Pipeline pipeline;
    PipelineLayout layout;
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
