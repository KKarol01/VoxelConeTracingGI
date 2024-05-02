#pragma once

#include "types.hpp"
#include "context.hpp"
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
template<> struct VulkanObjectType<vk::DescriptorSet> { static inline constexpr vk::ObjectType type = vk::ObjectType::eDescriptorSet; };
template<> struct VulkanObjectType<vk::Image> { static inline constexpr vk::ObjectType type = vk::ObjectType::eImage; };
template<> struct VulkanObjectType<vk::ImageView> { static inline constexpr vk::ObjectType type = vk::ObjectType::eImageView; };

struct SetDebugNameDetails {
    static vk::Device device();
    static vk::Instance instance();
    static PFN_vkGetInstanceProcAddr get_instance_proc_addr();
    static PFN_vkGetDeviceProcAddr get_device_proc_addr();
};

template<typename T> inline void set_debug_name(T object, std::string_view name) {
    SetDebugNameDetails::device().setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        VulkanObjectType<T>::type, (u64)static_cast<T::NativeType>(object), name.data()
    }, vk::DispatchLoaderDynamic{
            SetDebugNameDetails::instance(),
            SetDebugNameDetails::get_instance_proc_addr(),
            SetDebugNameDetails::device(),
            SetDebugNameDetails::get_device_proc_addr(),
        }
    );
}