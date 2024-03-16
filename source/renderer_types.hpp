#pragma once
#include "types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vk_mem_alloc.h>

struct GLFWwindow;

/* Rendering resources */
struct Pipeline {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
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

/* === */

struct Window {
    u32 width{1024}, height{768};
    GLFWwindow* window{nullptr};
};
