#pragma once

#include "renderer_types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vk_mem_alloc.h>
#include <variant>
#include <span>
#include <string_view>
#include <vector>

class RendererAllocator {
    struct UploadJob {
        UploadJob(std::variant<Handle<TextureAllocation>, Handle<BufferAllocation>> storage, std::span<const std::byte> data): storage(storage) {
            this->data.resize(data.size_bytes());
            std::memcpy(this->data.data(), data.data(), data.size_bytes());
        }
        std::variant<Handle<TextureAllocation>, Handle<BufferAllocation>> storage;
        std::vector<std::byte> data;
    };
    
public:
    explicit RendererAllocator(vk::Device device, VmaAllocator vma): device(device), vma(vma) {}

    Handle<TextureAllocation> create_texture_storage(std::string_view label, const vk::ImageCreateInfo& info, std::span<const std::byte> optional_data = {});
    Handle<BufferAllocation> create_buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, u64 size_bytes);
    Handle<BufferAllocation> create_buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, std::span<const std::byte> optional_data = {});

    void destroy_buffer(Handle<BufferAllocation> handle) {
        if(!handle) { return; }
        auto* ptr = find_with_handle(handle, buffers);
        if(!ptr) { return; }
        vmaDestroyBuffer(vma, ptr->buffer, ptr->alloc);
    }

    TextureAllocation& get_texture(Handle<TextureAllocation> handle) { return *find_with_handle(handle, textures); }
    BufferAllocation& get_buffer(Handle<BufferAllocation> handle) { return *find_with_handle(handle, buffers); }
    const BufferAllocation& get_buffer(Handle<BufferAllocation> handle) const { return *find_with_handle(handle, buffers); }

    bool has_jobs() const { return !jobs.empty(); }
    void complete_jobs(vk::CommandBuffer cmd);

private:
    template<typename T> T* find_with_handle(Handle<T> handle, std::vector<T>& storage) {
        auto it = std::lower_bound(storage.begin(), storage.end(), handle);
        if(it == storage.end() || *it != handle) { return nullptr; }
        return &*it;
    }
    template<typename T> const T* find_with_handle(Handle<T> handle, const std::vector<T>& storage) const {
        auto it = std::lower_bound(storage.begin(), storage.end(), handle);
        if(it == storage.end() || *it != handle) { return nullptr; }
        return &*it;
    }
    BufferAllocation* create_buffer_ptr(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, u64 size_bytes);
    TextureAllocation* create_texture_ptr(std::string_view label, const vk::ImageCreateInfo& info);

    vk::Device device;
    VmaAllocator vma;
    std::vector<TextureAllocation> textures;
    std::vector<BufferAllocation> buffers;
    std::vector<UploadJob> jobs;
};