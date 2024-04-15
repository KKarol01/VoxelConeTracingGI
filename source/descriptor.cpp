#include "descriptor.hpp"
#include "renderer.hpp"
#include <vulkan/vulkan.hpp>

DescriptorBufferSizes::DescriptorBufferSizes(vk::PhysicalDevice pdev) {
    vk::PhysicalDeviceProperties2 pdev_props;
    pdev_props.pNext = &pdev_descbuff_props;
    pdev.getProperties2(&pdev_props);
}

u64 DescriptorBufferSizes::get_descriptor_size(DescriptorType type) const {
    switch(type) {
        case DescriptorType::SampledImage:      { return pdev_descbuff_props.sampledImageDescriptorSize;}
        case DescriptorType::StorageImage:      { return pdev_descbuff_props.storageImageDescriptorSize;}
        case DescriptorType::Sampler:           { return pdev_descbuff_props.samplerDescriptorSize;}
        case DescriptorType::UniformBuffer:     { return pdev_descbuff_props.uniformBufferDescriptorSize;}
        case DescriptorType::StorageBuffer:     { return pdev_descbuff_props.storageBufferDescriptorSize;}
        case DescriptorType::CombinedImageSampler: { return pdev_descbuff_props.combinedImageSamplerDescriptorSize;}
        default: {
            spdlog::error("get_descriptor_size() doesn't support type: {}", (u32)type);
            std::terminate();
            return 0ull;
        }
    }
}

constexpr static u64 align_up(u64 size, u64 alignment) {
    return (size + alignment - 1ull) & -alignment;
}

static vk::DescriptorSetLayout create_set_layout(vk::Device device, const DescriptorBufferLayout& layout) {
    static constexpr vk::ShaderStageFlags all_stages = 
        vk::ShaderStageFlagBits::eVertex | 
        vk::ShaderStageFlagBits::eGeometry | 
        vk::ShaderStageFlagBits::eFragment | 
        vk::ShaderStageFlagBits::eCompute;
    
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    std::vector<vk::DescriptorBindingFlags> binding_flags;
    bindings.reserve(layout.bindings.size());
    binding_flags.reserve(layout.bindings.size());
    for(u32 i=0; auto& e : layout.bindings) {
        bindings.push_back(vk::DescriptorSetLayoutBinding{
            i,
            to_vk_desc_type(e.type),
            e.count,
            all_stages
        });

        if(e.is_runtime_sized) { binding_flags.push_back(vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::ePartiallyBound); }
        else { binding_flags.push_back({}); }
    }

    vk::DescriptorSetLayoutBindingFlagsCreateInfo info_flags{binding_flags};

    auto vklayout = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
        vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT,
        bindings,
        &info_flags
    });

    set_debug_name(device, vklayout, layout.name);

    return vklayout;
} 

DescriptorBuffer::DescriptorBuffer(vk::PhysicalDevice pdev, vk::Device device, u64 initial_size): pdev(pdev), device(device), sizes(pdev) { resize(initial_size); }

Handle<DescriptorBufferAllocation> DescriptorBuffer::push_layout(const DescriptorBufferLayout& layout) {
    const auto handle = push_layout(create_set_layout(device, layout));

    if(!handle) { return {}; }

    if(layout.bindings.back().is_runtime_sized) {
        runtime_layout_metadatas.emplace(handle, DescriptorBufferRuntimeLayoutMetadata{layout.bindings.back().count, 0u});
    }

    return handle;
}

std::vector<Handle<DescriptorBufferAllocation>> DescriptorBuffer::push_layouts(const std::vector<DescriptorBufferLayout>& layouts) {
    std::vector<Handle<DescriptorBufferAllocation>> handles;
    handles.reserve(layouts.size());
    for(auto& e : layouts) { handles.push_back(push_layout(e)); }
    return handles;
}

bool DescriptorBuffer::allocate_descriptor(Handle<DescriptorBufferAllocation> layout, u32 binding, const DescriptorBufferDescriptor& descriptor) {
    if(!runtime_layout_metadatas.contains(layout)) {
        return allocate_descriptor(layout, binding, 0, descriptor);
    }
    auto& runtime = runtime_layout_metadatas.at(layout);
    if(runtime.count >= runtime.max) { return false; }
    const auto idx = runtime.count++;
    return allocate_descriptor(layout, binding, idx, descriptor);
}

bool DescriptorBuffer::allocate_descriptor(Handle<DescriptorBufferAllocation> layout, u32 binding, u32 array_index, const DescriptorBufferDescriptor& descriptor) {
    auto& set = *std::find(begin(set_layouts), end(set_layouts), layout);
    const auto binding_offset = device.getDescriptorSetLayoutBindingOffsetEXT(set.layout, binding);
    const auto desc_size = sizes.get_descriptor_size(descriptor.type);
    auto* memory_location = static_cast<std::byte*>(buffer->data) + set.where + binding_offset + desc_size * array_index;

    vk::DescriptorDataEXT desc_data{};
    vk::DescriptorImageInfo image_info;
    vk::DescriptorAddressInfoEXT buffer_info;

    std::visit(visitor{
            [&descriptor](const auto&) { spdlog::error("Unsupported DescriptorbufferDescriptor (idx: {})", descriptor.payload.index()); std::terminate(); },
            [&descriptor, &desc_data, &image_info](const std::tuple<vk::ImageView, vk::ImageLayout>& image) { 
                image_info.setImageView(std::get<0>(image)).setImageLayout(std::get<1>(image));
                switch(descriptor.type) {
                    case DescriptorType::SampledImage: { desc_data.setPSampledImage(&image_info); } break;
                    case DescriptorType::StorageImage: { desc_data.setPStorageImage(&image_info); } break;
                    default: { std::terminate(); }
                }
            },
            [&descriptor, &desc_data, &buffer_info, this](const std::tuple<Handle<GpuBuffer>, u64>& buffer) { 
                const auto& alloc_buffer = get_context().renderer->allocator->get_buffer(std::get<0>(buffer));
                vk::BufferDeviceAddressInfo address_info{alloc_buffer.buffer};
                const auto address = device.getBufferAddress(&address_info);
                buffer_info.setAddress(address).setRange(std::get<1>(buffer));
                switch(descriptor.type) {
                    case DescriptorType::UniformBuffer: { desc_data.setPUniformBuffer(&buffer_info); } break;
                    case DescriptorType::StorageBuffer: { desc_data.setPStorageBuffer(&buffer_info); } break;
                    default: { std::terminate(); }
                }
            },
            [&descriptor, &desc_data](vk::Sampler sampler) { desc_data.setPSampler(&sampler); }
        },
        descriptor.payload
    );

    vk::DescriptorGetInfoEXT get_info{to_vk_desc_type(descriptor.type), desc_data};
    device.getDescriptorEXT(get_info, sizes.get_descriptor_size(descriptor.type), memory_location);
    return true;
}

vk::DeviceAddress DescriptorBuffer::get_buffer_address() const {
    return device.getBufferAddress(vk::BufferDeviceAddressInfo{buffer.buffer});
}

Handle<DescriptorBufferAllocation> DescriptorBuffer::push_layout(vk::DescriptorSetLayout vklayout) {
    const auto size = align_up(device.getDescriptorSetLayoutSizeEXT(vklayout), sizes.get_offset_alignment());
    if(!buffer) { resize(size); }
    const auto capacity = buffer->size; 
    
    auto [free_spot, free_spot_idx] = find_free_item(size);
    if(!free_spot) {
        defragment();
        if(calculate_free_space() < size) {
            if(!resize(capacity + size)) { return {}; }
        }

        std::tie(free_spot, free_spot_idx) = find_free_item(size);
        if(!free_spot) {
            spdlog::error("Critical error in DescriptorBuffer's defragment/resize/find_free_spot algorithm");
            return {};
        }
    }

    const auto& alloc = set_layouts.emplace_back(free_spot->where, size, vklayout);
    free_spot->where += size;
    free_spot->size -= size;
    if(free_spot->size == 0ull) { free_list.erase(free_list.begin() + free_spot_idx); }
    return alloc;
}

std::pair<DescriptorBufferFreeListItem*, u64> DescriptorBuffer::find_free_item(u64 size) {
    for(auto i=0ull; i<free_list.size(); ++i) {
        if(free_list.at(i).size >= size) {
            return {&free_list.at(i), i};
        }
    }
    return {nullptr, -1ull};
}

u64 DescriptorBuffer::calculate_free_space() const {
    u64 size = 0u;
    for(const auto& e : free_list) { size += e.size; } 
    return size;
}

void DescriptorBuffer::defragment() {
    if(free_list.empty()) { return; }

    std::vector<u64> new_offsets;
    new_offsets.reserve(set_layouts.size());

    const auto buffer_data = static_cast<std::byte*>(buffer->data);
    const auto buffer_size = buffer->size;

    std::vector<std::byte> temp(buffer_size);
    memcpy(temp.data(), buffer_data, buffer_size);

    u64 offset=0ull;
    for(auto i=0ull; i<set_layouts.size(); ++i) {
        memcpy(buffer_data + offset, temp.data() + set_layouts.at(i).where, set_layouts.at(i).size);
        set_layouts.at(i).where = offset;
        offset += set_layouts.at(i).size;
    } 

    free_list.clear();
    free_list.emplace_back(offset, buffer_size - offset);
}

bool DescriptorBuffer::resize(u64 new_size) {
    u64 old_size = buffer ? buffer->size : 0ull;
    u64 size = old_size + old_size / 2;
    if(size < new_size) { size = new_size; }
    if(new_size == 0ull) { return true; }

    Buffer new_buffer{
        "DescriptorBuffer_Storage", 
        vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT | vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        true,
        size
    };

    if(!new_buffer) {
        spdlog::error("Could not resize DescriptorBuffer");
        return false;
    }

    if(old_size > 0ull) {
        memcpy(new_buffer->data, buffer->data, buffer->size);
        get_context().renderer->deletion_queue.push_back([buffer=this->buffer.storage] {
            get_context().renderer->allocator->destroy_buffer(buffer);
        });
    }
    buffer = std::move(new_buffer);
    free_list.emplace_back(old_size, size - old_size);
    return true;
}
