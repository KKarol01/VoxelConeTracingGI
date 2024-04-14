#pragma once

#include "types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vk_mem_alloc.h>
#include <vector>
#include <array>
#include <variant>

enum class DescriptorType {
    SampledImage, StorageImage, Sampler, UniformBuffer, StorageBuffer, CombinedImageSampler
};


inline constexpr vk::DescriptorType to_vk_desc_type(DescriptorType type) {
    switch (type) {
        case DescriptorType::SampledImage: { return vk::DescriptorType::eSampledImage; }
        case DescriptorType::StorageImage: { return vk::DescriptorType::eStorageImage; }
        case DescriptorType::StorageBuffer: { return vk::DescriptorType::eStorageBuffer; }
        case DescriptorType::UniformBuffer: { return vk::DescriptorType::eUniformBuffer; }
        case DescriptorType::Sampler: { return vk::DescriptorType::eSampler; }
        case DescriptorType::CombinedImageSampler: { return vk::DescriptorType::eCombinedImageSampler; }
        default: {
            spdlog::error("Unrecognized descriptor type {}", (u32)type);
            std::abort();
        }
    }
}

// struct DescriptorInfo {
//     enum Resource : u8 { None=0, Buffer=1, Image=2, Sampler=3 };
    
//     constexpr DescriptorInfo() {}
//     constexpr DescriptorInfo(vk::DescriptorType type, vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size): 
//         buffer_info(buffer, offset, size),
//         type(type),
//         resource(Buffer) {}
//     constexpr DescriptorInfo(vk::DescriptorType type, vk::ImageView image, vk::ImageLayout layout): 
//         image_info({}, image, layout),
//         type(type),
//         resource(Image) {}
//     constexpr DescriptorInfo(vk::DescriptorType type, vk::Sampler sampler): 
//         image_info(sampler, {}, {}),
//         type(type),
//         resource(Sampler) {}

//     union {
//         vk::DescriptorBufferInfo buffer_info;
//         vk::DescriptorImageInfo image_info;
//         vk::DescriptorImageInfo sampler_info;
//     };
//     vk::DescriptorType type;
//     Resource resource{None};
// };


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

struct GpuBuffer : public Handle<GpuBuffer> {
    constexpr GpuBuffer() = default;
    GpuBuffer(vk::Buffer buffer, void* data, u64 size, VmaAllocation alloc)
        : Handle(HandleGenerate), buffer(buffer), data(data), size(size) {}

    vk::Buffer buffer;
    void* data;
    u64 size;
    VmaAllocation alloc;
};

struct Buffer {
    constexpr Buffer() = default;
    Buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, u64 size);
    Buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, std::span<const std::byte> optional_data = {});

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    constexpr Buffer(Buffer&& other) noexcept { *this = std::move(other); }
    constexpr Buffer& operator=(Buffer&& other) noexcept {
        storage = std::move(other.storage);
        buffer = std::exchange(other.buffer, nullptr);
        return *this;
    }

    constexpr operator bool() const noexcept { return static_cast<bool>(storage); }
    GpuBuffer* operator->();
    const GpuBuffer* operator->() const;

    Handle<GpuBuffer> storage;
    vk::Buffer buffer;
};

struct FrameResources {
    vk::CommandPool pool;
    vk::CommandBuffer cmd;
    vk::Semaphore swapchain_semaphore, rendering_semaphore;
    vk::Fence in_flight_fence;
};

struct TextureStorage : public Handle<TextureStorage> {
    constexpr TextureStorage() = default;
    TextureStorage(
        vk::ImageType type,
        u32 width, 
        u32 height, 
        u32 depth,
        u32 mips, 
        u32 layers,
        vk::Format format,
        vk::ImageLayout current_layout,
        vk::Image image,
        VmaAllocation alloc,
        vk::ImageView default_view)
        : Handle(HandleGenerate), type(type), width(width), height(height), depth(depth),
            mips(mips), layers(layers), format(format), current_layout(current_layout), image(image),
            alloc(alloc), default_view(default_view) {}
    
    vk::ImageType type;
    u32 width, height, depth;
    u32 mips, layers;
    vk::Format format;
    vk::ImageLayout current_layout;
    vk::Image image;
    VmaAllocation alloc;
    vk::ImageView default_view;
};

struct Texture {
    constexpr Texture() = default;
    Texture(Handle<TextureStorage> storage): storage(storage) {
        image = Texture::operator->()->image;
    }

    constexpr operator bool() const noexcept { return static_cast<bool>(storage); }
    TextureStorage* operator->();
    const TextureStorage* operator->() const;

    Handle<TextureStorage> storage;
    vk::Image image;
};

struct Texture2D : public Texture {
    constexpr Texture2D() = default;
    Texture2D(std::string_view label, u32 width, u32 height, vk::Format format, u32 mips, vk::ImageUsageFlags usage, std::span<const std::byte> optional_data = {});
};

struct Texture3D : public Texture {
    constexpr Texture3D() = default;
    Texture3D(std::string_view label, u32 width, u32 height, u32 depth, vk::Format format, u32 mips, vk::ImageUsageFlags usage); 
};

struct GLFWwindow;
struct Window {
    u32 width{1024}, height{768};
    GLFWwindow* window{nullptr};
};
