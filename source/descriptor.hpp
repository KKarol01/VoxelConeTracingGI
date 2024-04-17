#pragma once

#include "renderer_types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vector>
#include <map>
#include <variant>

struct DescriptorBufferLayoutBinding {
    DescriptorBufferLayoutBinding(DescriptorType type, u32 count, bool is_runtime_sized = false)
        : type(type), count(count), is_runtime_sized(is_runtime_sized) {}

    DescriptorType type;
    u32 count;
    bool is_runtime_sized;
};

struct DescriptorBufferLayout : public Handle<DescriptorBufferLayout> {
    constexpr DescriptorBufferLayout() = default;
    DescriptorBufferLayout(std::string_view name, const std::vector<DescriptorBufferLayoutBinding> &bindings)
        : Handle(HandleGenerate), name(name), bindings(bindings) {}

    std::string name;
    std::vector<DescriptorBufferLayoutBinding> bindings;
    vk::DescriptorSetLayout layout;
};

struct DescriptorBufferAllocation : public Handle<DescriptorBufferAllocation> {
    constexpr DescriptorBufferAllocation() = default;
    constexpr DescriptorBufferAllocation(vk::DescriptorSet set, vk::DescriptorSetLayout layout, u32 variable_binding, u32 max_variable_size)
        : Handle(HandleGenerate), set(set), layout(layout), variable_binding(variable_binding), max_variable_size(max_variable_size), current_variable_size(0u) {}
    vk::DescriptorSet set;
    vk::DescriptorSetLayout layout;
    u32 variable_binding;
    u32 max_variable_size;
    u32 current_variable_size;
};

struct DescriptorBufferDescriptor {
    DescriptorType type;
    std::variant<std::tuple<vk::ImageView, vk::ImageLayout>, 
                 std::tuple<Handle<GpuBuffer>, u64>,
                 vk::Sampler> payload;
};

class DescriptorSet {
public: 
    DescriptorSet() = default;
    DescriptorSet(vk::Device device);

    Handle<DescriptorBufferAllocation> push_layout(const DescriptorBufferLayout& layout);
    std::vector<Handle<DescriptorBufferAllocation>> push_layouts(const std::vector<DescriptorBufferLayout>& layouts);
    bool write_descriptor(Handle<DescriptorBufferAllocation> handle, u32 binding, const DescriptorBufferDescriptor& descriptor);
    bool write_descriptor(Handle<DescriptorBufferAllocation> handle, u32 binding, u32 array_index, const DescriptorBufferDescriptor& descriptor);
    vk::DescriptorSet get_set(Handle<DescriptorBufferAllocation> handle);
    vk::DescriptorSetLayout get_layout(Handle<DescriptorBufferAllocation> handle);

private:
    DescriptorBufferLayout* find_matching_layout(const DescriptorBufferLayout& layout);
    std::vector<DescriptorType> get_layout_types(const DescriptorBufferLayout& layout);
    DescriptorBufferAllocation& get_allocation(Handle<DescriptorBufferAllocation> handle);
    bool is_pool_compatible_with_layout(const std::vector<DescriptorType>& pool, const DescriptorBufferLayout& layout);
    void insert_compatible_pools_to_layout(const DescriptorBufferLayout& layout);
    void propagate_pool_to_compatible_layouts(vk::DescriptorPool pool, const std::vector<DescriptorType>& types);

    vk::Device device;
    std::vector<DescriptorBufferAllocation> sets;
    std::vector<std::pair<vk::DescriptorPool, std::vector<DescriptorType>>> pools;
    std::map<vk::DescriptorSetLayout, std::vector<vk::DescriptorPool>> layout_compatible_pools;
    std::vector<DescriptorBufferLayout> layouts;
};