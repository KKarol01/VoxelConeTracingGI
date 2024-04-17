#include "descriptor.hpp"
#include "renderer.hpp"
#include <vulkan/vulkan.hpp>

static vk::DescriptorSetLayout create_set_layout(vk::Device device, const DescriptorSetLayout& layout) {
    static constexpr vk::ShaderStageFlags all_stages = 
        vk::ShaderStageFlagBits::eVertex | 
        vk::ShaderStageFlagBits::eGeometry | 
        vk::ShaderStageFlagBits::eFragment | 
        vk::ShaderStageFlagBits::eCompute;
    
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    std::vector<vk::DescriptorBindingFlags> binding_flags;
    bindings.reserve(layout.bindings.size());
    binding_flags.reserve(layout.bindings.size());
    for(u32 i=0; i<layout.bindings.size(); ++i) {
        auto& binding = layout.bindings.at(i);
        bindings.push_back(vk::DescriptorSetLayoutBinding{
            i,
            to_vk_desc_type(binding.type),
            binding.count,
            all_stages
        });

        vk::DescriptorBindingFlags descriptor_flags =
            vk::DescriptorBindingFlagBits::eUpdateAfterBind | vk::DescriptorBindingFlagBits::ePartiallyBound;
        if(binding.is_runtime_sized) { descriptor_flags |= vk::DescriptorBindingFlagBits::eVariableDescriptorCount; }
        
        binding_flags.push_back(descriptor_flags);
    }

    vk::DescriptorSetLayoutBindingFlagsCreateInfo info_flags{binding_flags};

    auto vklayout = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
        vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        bindings,
        &info_flags
    });

    set_debug_name(device, vklayout, layout.name);

    return vklayout;
} 

static vk::DescriptorPool create_set_pool(vk::Device device, const DescriptorSetLayout& layout) {
    std::unordered_map<DescriptorType, u32> counts;
    u32 max_sets = 0;

    for(auto& e : layout.bindings) {
        counts[e.type] += e.count;
        max_sets = std::max(max_sets, e.count);
    }
    
    std::vector<vk::DescriptorPoolSize> pool_sizes;
    pool_sizes.reserve(counts.size());

    for(auto& [type, count] : counts) {
        pool_sizes.push_back(vk::DescriptorPoolSize{to_vk_desc_type(type), count});
    }

    vk::DescriptorPoolCreateInfo info{
        vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        max_sets,
        pool_sizes
    };

    auto vkpool = device.createDescriptorPool(info);

    set_debug_name(device, vkpool, std::format("{}_pool", layout.name));

    return vkpool;
}

DescriptorSet::DescriptorSet(vk::Device device): device(device) { }

Handle<DescriptorSetAllocation> DescriptorSet::push_layout(const DescriptorSetLayout& layout) {
    if(layout.bindings.empty()) { return {}; }

    auto* matching_layout = find_matching_layout(layout);

    if(!matching_layout) {
        auto vklayout = create_set_layout(device, layout);
        auto& inserted_layout = layouts.emplace_back(layout);
        inserted_layout.layout = vklayout;
        matching_layout = &inserted_layout;
        insert_compatible_pools_to_layout(*matching_layout); 
    }

    u32 variable_sizes[] {layout.bindings.back().is_runtime_sized ? layout.bindings.back().count : 0};
    vk::DescriptorSetVariableDescriptorCountAllocateInfo variable_desc{variable_sizes};
    vk::DescriptorSetAllocateInfo info{nullptr, matching_layout->layout, &variable_desc};
    vk::DescriptorSet desc_set;

    for(auto& compatible_pool : layout_compatible_pools[matching_layout->layout]) {
        try {
            info.setDescriptorPool(compatible_pool);
            desc_set = device.allocateDescriptorSets(info)[0];
            break;
        } catch(const std::exception& err) { desc_set = nullptr; }
    }
    if(!desc_set) {
        auto matching_pool = create_set_pool(device, layout);
        const auto layout_types = get_layout_types(layout);
        pools.emplace_back(matching_pool, layout_types);
        propagate_pool_to_compatible_layouts(matching_pool, layout_types);
        info.setDescriptorPool(matching_pool);
        desc_set = device.allocateDescriptorSets(info)[0];
    }

    set_debug_name(device, desc_set, std::format("{}_descriptor_set", layout.name));
    auto& set = sets.emplace_back(desc_set, matching_layout->layout, layout.bindings.back().is_runtime_sized ? layout.bindings.size()-1 : -1ul, layout.bindings.back().count);

    spdlog::debug("Descset: layout {}, handle {}", layout.name, matching_layout->handle);

    return set;
}

std::vector<Handle<DescriptorSetAllocation>> DescriptorSet::push_layouts(const std::vector<DescriptorSetLayout>& layouts) {
    std::vector<Handle<DescriptorSetAllocation>> handles;
    handles.reserve(layouts.size());
    for(auto& e : layouts) { handles.push_back(push_layout(e)); }
    return handles;
}

bool DescriptorSet::write_descriptor(Handle<DescriptorSetAllocation> handle, u32 binding, const DescriptorSetUpdate& descriptor) {
    auto& alloc = get_allocation(handle);

    if(alloc.max_variable_size <= alloc.current_variable_size) { 
        spdlog::warn("DescriptorSet::write_descriptor: Descriptor binding array overflow. use method with array_index parameter.");
        return false; 
    }

    return write_descriptor(handle, binding, alloc.variable_binding == binding ? alloc.current_variable_size++ : 0, descriptor);
}

bool DescriptorSet::write_descriptor(Handle<DescriptorSetAllocation> handle, u32 binding, u32 array_index, const DescriptorSetUpdate& descriptor) {
    vk::WriteDescriptorSet write_set{
        get_allocation(handle).set,
        binding,
        array_index,
        1,
        to_vk_desc_type(descriptor.type)
    };

    vk::DescriptorImageInfo image_info;
    vk::DescriptorBufferInfo buffer_info;

    if(auto* payload = std::get_if<std::tuple<vk::ImageView, vk::ImageLayout>>(&descriptor.payload)) {
        image_info = vk::DescriptorImageInfo{{}, std::get<0>(*payload), std::get<1>(*payload)};
        write_set.setImageInfo(image_info);
    } else if(auto* payload = std::get_if<std::tuple<Handle<GpuBuffer>, u64>>(&descriptor.payload)) {
        auto& buffer = get_context().renderer->allocator->get_buffer(std::get<0>(*payload));
        buffer_info = vk::DescriptorBufferInfo{buffer.buffer, 0ull, std::get<1>(*payload)};
        write_set.setBufferInfo(buffer_info);
    } else if(auto* payload = std::get_if<vk::Sampler>(&descriptor.payload)) {
        image_info = vk::DescriptorImageInfo{*payload};
        write_set.setImageInfo(image_info);
    }

    device.updateDescriptorSets(write_set, {});
    return true;
}

vk::DescriptorSet DescriptorSet::get_set(Handle<DescriptorSetAllocation> handle) {
    return get_allocation(handle).set;
}

vk::DescriptorSetLayout DescriptorSet::get_layout(Handle<DescriptorSetAllocation> handle) {
    return get_allocation(handle).layout;
}

std::vector<DescriptorType> DescriptorSet::get_layout_types(const DescriptorSetLayout& layout) {
    std::unordered_set<DescriptorType> types;
    for(auto& e : layout.bindings) { types.insert(e.type); }
    return {begin(types), end(types)};
}

DescriptorSetAllocation& DescriptorSet::get_allocation(Handle<DescriptorSetAllocation> handle) {
    return *std::find(begin(sets), end(sets), handle);
}

DescriptorSetLayout* DescriptorSet::find_matching_layout(const DescriptorSetLayout& layout) {
    for(auto& dslayout : layouts) {
        if(dslayout.bindings.size() != layout.bindings.size()) { continue; }
        for(u32 i=0; i<dslayout.bindings.size(); ++i) {
            auto& dsb = dslayout.bindings.at(i);
            auto& lb = layout.bindings.at(i);
            if(dsb.type != lb.type || dsb.count != lb.count || dsb.is_runtime_sized != lb.is_runtime_sized) { continue; }
            return &dslayout;
        }
    }

    return nullptr;
}

bool DescriptorSet::is_pool_compatible_with_layout(const std::vector<DescriptorType>& pool, const DescriptorSetLayout& layout) {
    const auto layout_types = get_layout_types(layout);
    std::unordered_set<DescriptorType> layout_types_set{begin(layout_types), end(layout_types)};
    for(auto pool_type : pool) { layout_types_set.erase(pool_type); }
    return layout_types_set.empty();
}

void DescriptorSet::insert_compatible_pools_to_layout(const DescriptorSetLayout& layout) {
    for(auto& pool : pools) {
        if(is_pool_compatible_with_layout(pool.second, layout)) {
            layout_compatible_pools[layout.layout].push_back(pool.first);
        }
    }
}

void DescriptorSet::propagate_pool_to_compatible_layouts(vk::DescriptorPool pool, const std::vector<DescriptorType>& types) {
    for(const auto& layout : layouts) {
        if(is_pool_compatible_with_layout(types, layout)) {
            layout_compatible_pools[layout.layout].push_back(pool);
        }
    }
}