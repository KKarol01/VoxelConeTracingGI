#pragma once

#include "renderer_types.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vector>
#include <map>
#include <variant>

struct DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding(DescriptorType type, u32 count, bool is_runtime_sized = false)
        : type(type), count(count), is_runtime_sized(is_runtime_sized) {}

    DescriptorType type;
    u32 count;
    bool is_runtime_sized;
};

struct DescriptorSetLayout : public Handle<DescriptorSetLayout> {
    constexpr DescriptorSetLayout() = default;
    DescriptorSetLayout(std::string_view name, const std::vector<DescriptorSetLayoutBinding> &bindings)
        : Handle(HandleGenerate), name(name), bindings(bindings) {}

    std::string name;
    std::vector<DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout layout;
};

struct DescriptorSetAllocation : public Handle<DescriptorSetAllocation> {
    constexpr DescriptorSetAllocation() = default;
    constexpr DescriptorSetAllocation(vk::DescriptorSet set, vk::DescriptorSetLayout layout, u32 variable_binding, u32 max_variable_size)
        : Handle(HandleGenerate), set(set), layout(layout), variable_binding(variable_binding), max_variable_size(max_variable_size), current_variable_size(0u) {}
    vk::DescriptorSet set;
    vk::DescriptorSetLayout layout;
    u32 variable_binding;
    u32 max_variable_size;
    u32 current_variable_size;
};

struct DescriptorSetUpdate {
    DescriptorType type;
    std::variant<std::tuple<vk::ImageView, vk::ImageLayout>, 
                 std::tuple<Handle<GpuBuffer>, u64>,
                 vk::Sampler> payload;
};

class DescriptorSet {
public: 
    DescriptorSet() = default;
    DescriptorSet(vk::Device device);

    Handle<DescriptorSetAllocation> push_layout(const DescriptorSetLayout& layout);
    std::vector<Handle<DescriptorSetAllocation>> push_layouts(const std::vector<DescriptorSetLayout>& layouts);
    bool write_descriptor(Handle<DescriptorSetAllocation> handle, u32 binding, const DescriptorSetUpdate& descriptor);
    bool write_descriptor(Handle<DescriptorSetAllocation> handle, u32 binding, u32 array_index, const DescriptorSetUpdate& descriptor);
    vk::DescriptorSet get_set(Handle<DescriptorSetAllocation> handle);
    vk::DescriptorSetLayout get_layout(Handle<DescriptorSetAllocation> handle);

private:
    DescriptorSetLayout* find_matching_layout(const DescriptorSetLayout& layout);
    std::vector<DescriptorType> get_layout_types(const DescriptorSetLayout& layout);
    DescriptorSetAllocation& get_allocation(Handle<DescriptorSetAllocation> handle);
    bool is_pool_compatible_with_layout(const std::vector<DescriptorType>& pool, const DescriptorSetLayout& layout);
    void insert_compatible_pools_to_layout(const DescriptorSetLayout& layout);
    void propagate_pool_to_compatible_layouts(vk::DescriptorPool pool, const std::vector<DescriptorType>& types);

    vk::Device device;
    std::vector<DescriptorSetAllocation> sets;
    std::vector<std::pair<vk::DescriptorPool, std::vector<DescriptorType>>> pools;
    std::map<vk::DescriptorSetLayout, std::vector<vk::DescriptorPool>> layout_compatible_pools;
    std::vector<DescriptorSetLayout> layouts;
};