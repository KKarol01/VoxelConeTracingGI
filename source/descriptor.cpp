#include "descriptor.hpp"
#include "renderer.hpp"
#include <vulkan/vulkan.hpp>

void DescriptorSet::update(u32 binding, u32 array_element, const std::vector<DescriptorUpdate>& updates) {
    std::vector<vk::WriteDescriptorSet> write_sets;
    std::vector<vk::DescriptorBufferInfo> buffer_infos;
    std::vector<vk::DescriptorImageInfo> image_infos;

    write_sets.reserve(updates.size());
    buffer_infos.reserve(updates.size());
    image_infos.reserve(updates.size());

    const auto* alloc = allocator->find_allocation(allocation);
    if(!alloc) { 
        spdlog::error("Cannot find allocation in descriptorallocator that was supposed to be valid. This is serious error");
        return;
    }

    const auto& layout = allocator->layouts.at(alloc->layout_idx);
    if(layout.count == 0u) { 
        spdlog::warn("Trying to update bindings in a layout that has 0 bindings.");
        return;
    }

    for(const auto& e : updates) {
        const auto& b = layout.bindings.at(binding);
        const auto is_last = layout.count == binding + 1u;
        const auto count = is_last && layout.variable_sized ? alloc->variable_size : b.count;

        vk::WriteDescriptorSet ws{
            set, binding, array_element, 1, b.type
        };

        if(auto* img = std::get_if<0>(&e.data)) {
            const auto& [view, layout, sampler] = *img;
            image_infos.push_back(vk::DescriptorImageInfo{sampler, view, layout});
            ws.setImageInfo(image_infos.back());
        } else if(auto* buff = std::get_if<1>(&e.data)) {
            buffer_infos.push_back(vk::DescriptorBufferInfo{
                (*buff)->buffer, 0, vk::WholeSize
            });

            ws.setBufferInfo(buffer_infos.back());
        } else { spdlog::error("Unhandled descriptor udpate type: {}", (u32)e.data.index()); return; }

        write_sets.push_back(ws);

        if(count <= array_element + 1u) {
            binding++;
            array_element = 0;
        } else if(count > array_element + 1u) {
            array_element++;
        }
    }

    allocator->device.updateDescriptorSets(write_sets, {});
}

DescriptorSet DescriptorAllocator::allocate(std::string_view label, const DescriptorLayout& layout, u32 max_sets, u32 variable_size) {
    Pools* pools{nullptr};
    u32 idx = 0;
    std::tie(pools, idx) = find_matching_pools(layout);
    DescriptorPool* pool = nullptr;

    if(!pools) {
        layouts.push_back(layout);
        layouts.back().layout = create_layout(layout);
        set_debug_name(device, layouts.back().layout, std::format("{}_layout", label));
        idx = this->pools.size();
        pools = &this->pools.emplace_back();
        pool = create_pool(layout, max_sets, *pools);
        set_debug_name(device, pool->pool, std::format("{}_pool_{}", label, pools->pools.size()));
    } else { pool = &pools->pools.back(); }

    if(!pool) { 
        spdlog::error("Corrupted descriptor pools vector");
        std::terminate();
    }

    int tries = 0;
    do {
        ++tries;
        try {
            vk::DescriptorSetVariableDescriptorCountAllocateInfo variable_info;
            if(layout.variable_sized) {
                variable_info.setDescriptorCounts(variable_size);
            }

            auto set = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
                pool->pool,
                layouts.at(idx).layout,
                &variable_info
            })[0];

            set_debug_name(device, set, label);
            
            pool->allocations++;

            const auto layout_idx = [&pools, this] {
                for(u32 i = 0; i < this->pools.size(); ++i) { if(&this->pools.at(i) == pools) { return i; } } return 0u;
            }();
            const auto& allocation = allocations.emplace_back(layout_idx, pool->pool, variable_size);

            return DescriptorSet{this, set, allocation};
        } catch(const std::exception& err) {
            spdlog::error("Could not allocate descriptor set: {}. Retrying...", err.what());
            pool = create_pool(layout, max_sets, *pools);
        }
    } while(tries < 2);

    spdlog::error("Retrying failed. Returning invalid descriptor set");
    return DescriptorSet{};
}

DescriptorAllocation* DescriptorAllocator::find_allocation(Handle<DescriptorAllocation> allocation) {
    auto it = std::find(begin(allocations), end(allocations), allocation);
    if(it == end(allocations)) { return nullptr; }
    return &*it;
}

std::pair<DescriptorAllocator::Pools*, u32> DescriptorAllocator::find_matching_pools(const DescriptorLayout& layout) {
    assert("Layouts and pools must be same in length" && layouts.size() == pools.size());

    for(u32 i = 0; i < layouts.size(); ++i) {
        const auto& l = layouts.at(i);
        auto& ps = pools.at(i);

        if(l.count != layout.count) { continue; }
        if(l.variable_sized != layout.variable_sized) { continue; }

        bool invalid_bindings = false;
        for(u32 j = 0; j < l.count; ++j) {
            if(l.bindings.at(j).type != layout.bindings.at(j).type) { invalid_bindings = true; break; }
            if(l.bindings.at(j).count != layout.bindings.at(j).count) { invalid_bindings = true; break; }
        }

        if(invalid_bindings) { continue; }

        return {&ps, i};
    }

    return {nullptr, 0u};
}

DescriptorPool* DescriptorAllocator::create_pool(const DescriptorLayout& layout, u32 max_sets, DescriptorAllocator::Pools& pools) {
    std::unordered_map<vk::DescriptorType, u32> counts;
    for(u32 i = 0; i < layout.count; ++i) {
        counts[layout.bindings[i].type]++;
    }

    std::vector<vk::DescriptorPoolSize> sizes;
    sizes.reserve(counts.size());

    for(const auto& [type, count] : counts) {
        sizes.push_back(vk::DescriptorPoolSize{type, count * max_sets});
    }
    
    try {
        return &pools.pools.emplace_back(
            device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
                max_sets,
                sizes
            }),
            max_sets,
            0
        );
    } catch(const std::exception& err) {
        spdlog::error("Could not create descriptor pool: {}", err.what());
        return nullptr;
    }
}

vk::DescriptorSetLayout DescriptorAllocator::create_layout(const DescriptorLayout& layout) {
    static constexpr vk::ShaderStageFlags ALL_STAGE_FLAGS = 
        vk::ShaderStageFlagBits::eFragment | 
        vk::ShaderStageFlagBits::eVertex |
        vk::ShaderStageFlagBits::eCompute | 
        vk::ShaderStageFlagBits::eGeometry;

    auto& l = layout;
    std::array<vk::DescriptorSetLayoutBinding, DescriptorLayout::MAX_BINDINGS> bindings{};
    std::array<vk::DescriptorBindingFlags, DescriptorLayout::MAX_BINDINGS> flags{};

    for(u32 b = 0; b < l.count; ++b) {
        bindings.at(b) = {b, l.bindings.at(b).type, l.bindings.at(b).count, ALL_STAGE_FLAGS};
        flags.at(b) = vk::DescriptorBindingFlagBits::eUpdateAfterBind | vk::DescriptorBindingFlagBits::ePartiallyBound;
        if(l.variable_sized && b + 1u == l.count) { flags.at(b) |= vk::DescriptorBindingFlagBits::eVariableDescriptorCount; }
    }

    vk::DescriptorSetLayoutBindingFlagsCreateInfo flag_info{l.count, flags.data()};

    auto vklayout = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
        vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        l.count,
        bindings.data(),
        &flag_info
    });
    
    return vklayout;
}