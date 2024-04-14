#pragma once

#include "renderer_types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vector>
#include <unordered_map>
#include <variant>

struct DescriptorBufferLayoutBinding {
    DescriptorBufferLayoutBinding(DescriptorType type, u32 count, bool is_runtime_sized = false)
        : type(type), count(count), is_runtime_sized(is_runtime_sized) {}

    DescriptorType type;
    u32 count;
    bool is_runtime_sized;
};

struct DescriptorBufferLayout {
    DescriptorBufferLayout(std::string_view name, const std::vector<DescriptorBufferLayoutBinding> &bindings)
        : name(name), bindings(bindings) {}

    std::string name;
    std::vector<DescriptorBufferLayoutBinding> bindings;
};

struct DescriptorBufferSizes {
    constexpr DescriptorBufferSizes() = default;
    DescriptorBufferSizes(vk::PhysicalDevice pdev);
    u64 get_offset_alignment() const { return pdev_descbuff_props.descriptorBufferOffsetAlignment; }
    u64 get_descriptor_size(DescriptorType type) const;
    vk::PhysicalDeviceDescriptorBufferPropertiesEXT pdev_descbuff_props;
};

struct DescriptorBufferRuntimeLayoutMetadata {
    u32 max, count; 
};

struct DescriptorBufferAllocation : public Handle<DescriptorBufferAllocation> {
    constexpr DescriptorBufferAllocation() = default;
    constexpr DescriptorBufferAllocation(u64 where, u64 size, vk::DescriptorSetLayout layout)
        : Handle(HandleGenerate), where(where), size(size), layout(layout) { }
    
    u64 where, size;
    vk::DescriptorSetLayout layout;
};

struct DescriptorBufferFreeListItem {
    u64 where, size;
};

struct DescriptorBufferDescriptor {
    DescriptorType type;
    std::variant<std::tuple<vk::ImageView, vk::ImageLayout>, 
                 std::tuple<Handle<GpuBuffer>, u64>,
                 vk::Sampler> payload;
};

class DescriptorBuffer {
public: 
    DescriptorBuffer() = default;
    DescriptorBuffer(vk::PhysicalDevice pdev, vk::Device device);

    Handle<DescriptorBufferAllocation> push_layout(const DescriptorBufferLayout& layout);
    std::vector<Handle<DescriptorBufferAllocation>> push_layouts(const std::vector<DescriptorBufferLayout>& layouts);
    bool allocate_descriptor(Handle<DescriptorBufferAllocation> layout, u32 binding, const DescriptorBufferDescriptor& descriptor);
    bool allocate_descriptor(Handle<DescriptorBufferAllocation> layout, u32 binding, u32 array_index, const DescriptorBufferDescriptor& descriptor);
    vk::DeviceAddress get_buffer_address() const;

private:
    Handle<DescriptorBufferAllocation> push_layout(vk::DescriptorSetLayout vklayout);
    std::pair<DescriptorBufferFreeListItem*, u64> find_free_item(u64 size);
    u64 calculate_free_space() const;
    void defragment();
    bool resize(u64 new_size);

    Buffer buffer;
    vk::PhysicalDevice pdev;
    vk::Device device;
    DescriptorBufferSizes sizes;
    std::vector<DescriptorBufferFreeListItem> free_list;
    std::vector<DescriptorBufferAllocation> set_layouts;
    std::unordered_map<Handle<DescriptorBufferAllocation>, DescriptorBufferRuntimeLayoutMetadata> runtime_layout_metadatas;
};