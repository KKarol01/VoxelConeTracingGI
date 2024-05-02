#pragma once

#include "types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vk_mem_alloc.h>
#include <vector>
#include <array>

struct BufferAllocation : public Handle<BufferAllocation> {
    constexpr BufferAllocation() = default;
    BufferAllocation(vk::Buffer buffer, vk::BufferUsageFlags usage, void* data, u64 size, VmaAllocation alloc)
        : Handle(HandleGenerate), buffer(buffer), usage(usage), data(data), size(size) {}

    vk::Buffer buffer;
    vk::BufferUsageFlags usage;
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
        allocation = std::move(other.allocation);
        buffer = std::exchange(other.buffer, nullptr);
        return *this;
    }

    constexpr operator bool() const noexcept { return static_cast<bool>(allocation); }
    BufferAllocation* operator->();
    const BufferAllocation* operator->() const;

    Handle<BufferAllocation> allocation;
    vk::Buffer buffer;
};

struct FrameResources {
    vk::CommandPool pool;
    vk::CommandBuffer cmd;
    vk::Semaphore swapchain_semaphore, rendering_semaphore;
    vk::Fence in_flight_fence;
};

struct TextureAllocation : public Handle<TextureAllocation> {
    constexpr TextureAllocation() = default;
    TextureAllocation(
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
        vk::ImageView default_view,
        vk::Filter min_filter = vk::Filter::eLinear,
        vk::Filter mag_filter = vk::Filter::eLinear,
        vk::SamplerMipmapMode mip_filter = vk::SamplerMipmapMode::eLinear)
        : Handle(HandleGenerate), type(type), width(width), height(height), depth(depth),
            mips(mips), layers(layers), format(format), current_layout(current_layout), image(image),
            alloc(alloc), default_view(default_view), min_filter(min_filter), mag_filter(mag_filter),
            mip_filter(mip_filter)
        {}
    
    vk::ImageType type;
    u32 width, height, depth;
    u32 mips, layers;
    vk::Format format;
    vk::ImageLayout current_layout;
    vk::Image image;
    VmaAllocation alloc;
    vk::ImageView default_view;
    vk::Filter min_filter{vk::Filter::eLinear}, mag_filter{vk::Filter::eLinear};
    vk::SamplerMipmapMode mip_filter{vk::SamplerMipmapMode::eLinear};
};

struct Texture {
    constexpr Texture() = default;
    Texture(Handle<TextureAllocation> storage): storage(storage) {
        image = Texture::operator->()->image;
    }

    constexpr operator bool() const noexcept { return static_cast<bool>(storage); }
    TextureAllocation* operator->();
    const TextureAllocation* operator->() const;

    Handle<TextureAllocation> storage;
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
