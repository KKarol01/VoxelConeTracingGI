#pragma once

#include "types.hpp"
#include "pipelines.hpp"
#include <vulkan/vulkan_structs.hpp>
#include <vector>
#include <variant>

struct DescriptorUpdate {
    std::variant<
        std::tuple<vk::ImageView, vk::ImageLayout, vk::Sampler>,
        const Buffer* // buffer, offset = 0, size = read
    > data;
};

struct DescriptorPool {
    vk::DescriptorPool pool; 
    u32 max_allocations;
    u32 allocations;
};

struct DescriptorAllocation : public Handle<DescriptorAllocation> {
    constexpr DescriptorAllocation(u32 layout_idx, vk::DescriptorPool pool, u32 variable_size): Handle(HandleGenerate), layout_idx(layout_idx), pool(pool), variable_size(variable_size) {}
    u32 layout_idx;
    vk::DescriptorPool pool;
    u32 variable_size;
};

class DescriptorAllocator;

class DescriptorSet {
public:
    constexpr DescriptorSet() noexcept = default;
    constexpr DescriptorSet(DescriptorAllocator* allocator, vk::DescriptorSet set, Handle<DescriptorAllocation> allocation): allocator(allocator), set(set), allocation(allocation) {}

    void update(u32 binding, u32 array_element, const std::vector<DescriptorUpdate>& updates);

    vk::DescriptorSet set;

private:
    DescriptorAllocator* allocator;
    Handle<DescriptorAllocation> allocation; 
};

class DescriptorAllocator {
public: 
    DescriptorAllocator() noexcept = default;
    DescriptorAllocator(vk::Device device) noexcept: device(device) {};

    DescriptorSet allocate(std::string_view label, const DescriptorLayout& layout, u32 max_sets = 8, u32 variable_size = 0);
    DescriptorAllocation* find_allocation(Handle<DescriptorAllocation> allocation);

private:
    struct Pools {
        std::vector<DescriptorPool> pools;
    };

    std::pair<DescriptorAllocator::Pools*, u32> find_matching_pools(const DescriptorLayout& layout);
    DescriptorPool* create_pool(const DescriptorLayout& layout, u32 max_sets, Pools& pools);

    vk::Device device;
    std::vector<DescriptorLayout> layouts;
    std::vector<Pools> pools;
    std::vector<DescriptorAllocation> allocations;

    friend class DescriptorSet;
};