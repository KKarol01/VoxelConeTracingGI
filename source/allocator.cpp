#include "allocator.hpp"
#include "renderer.hpp"
#include <spdlog/spdlog.h>

Handle<TextureAllocation> RendererAllocator::create_texture_storage(std::string_view label, const vk::ImageCreateInfo& info, std::span<const std::byte> optional_data) {
    auto* texture = create_texture_ptr(label, info);
    if(!texture) { return {}; }

    if(optional_data.size_bytes() == 0ull) { return *texture; }

    jobs.emplace_back(*texture, optional_data);
    return *texture;
}

Handle<BufferAllocation> RendererAllocator::create_buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, u64 size_bytes) {
    if(size_bytes == 0ull) { 
        spdlog::warn("Requested buffer ({}) size is 0. This is probably a bug.", label);
        return {};
    }

    auto* buffer = create_buffer_ptr(label, usage, map_memory, size_bytes);
    if(!buffer) { return {}; }
    return *buffer;
}

Handle<BufferAllocation> RendererAllocator::create_buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, std::span<const std::byte> optional_data) {
    if(optional_data.size_bytes() == 0) { 
        spdlog::warn("Requested buffer ({}) size is 0. This is probably a bug.", label);
        return {};
    }
    
    auto* buffer = create_buffer_ptr(label, usage, map_memory, optional_data.size_bytes());
    if(!buffer) { return {}; }

    if(optional_data.size_bytes() == 0ull) { return *buffer; }

    if(map_memory && buffer->data) {
        memcpy(buffer->data, optional_data.data(), optional_data.size_bytes());
    } else {
        jobs.emplace_back(*buffer, optional_data);
    }

    return *buffer;
}

void RendererAllocator::complete_jobs(vk::CommandBuffer cmd) {
    const auto total_upload_size = [&jobs = this->jobs] {
        u64 sum = 0;
        for(const auto& e : jobs) {
            sum += e.data.size();
        }
        return sum;
    }();

    std::vector<std::byte> upload_data;
    upload_data.reserve(total_upload_size);

    for(auto& e : jobs) {
        upload_data.insert(upload_data.end(), e.data.begin(), e.data.end());
    }

    auto staging_buffer = Buffer{"allocator_staging_buffer", vk::BufferUsageFlagBits::eTransferSrc, true, std::span{upload_data}};

    u64 offset = 0ull;
    for(auto& job : jobs) {
        if(auto* handle = std::get_if<Handle<TextureAllocation>>(&job.storage)) {
            auto& texture = get_texture(*handle);

            vk::ImageMemoryBarrier2 barrier{
                vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                {}, {}, texture.image, {deduce_vk_image_aspect(texture.format), 0, 1, 0, texture.layers}
            };

            cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});

            const auto copy_region = vk::BufferImageCopy2{offset, texture.width, texture.height, 
                vk::ImageSubresourceLayers{deduce_vk_image_aspect(texture.format), 0, 0, texture.layers},
                vk::Offset3D{0, 0, 0},
                vk::Extent3D{texture.width, texture.height, texture.depth}
            };

            cmd.copyBufferToImage2(vk::CopyBufferToImageInfo2{
                staging_buffer.buffer, texture.image, vk::ImageLayout::eTransferDstOptimal, copy_region
            });

            if(texture.mips > 1) {
                barrier = {
                    vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlagBits2::eNone,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                    {}, {}, texture.image, {deduce_vk_image_aspect(texture.format), 1, texture.mips - 1, 0, texture.layers}
                };

                cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});

                for(u32 mip = 1; mip < texture.mips; ++mip) {
                    const auto pw = std::max(texture.width >> (mip - 1), 1u);
                    const auto cw = std::max(texture.width >> (mip - 0), 1u);
                    const auto ph = std::max(texture.height >> (mip - 1), 1u);
                    const auto ch = std::max(texture.height >> (mip - 0), 1u);
                    const auto pd = std::max(texture.depth >> (mip - 1), 1u);
                    const auto cd = std::max(texture.depth >> (mip - 0), 1u);
                    
                    barrier = {
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
                        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                        {}, {}, texture.image, {deduce_vk_image_aspect(texture.format), mip-1, 1, 0, texture.layers}
                    };

                    cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});

                    const auto blit = vk::ImageBlit2{
                        vk::ImageSubresourceLayers{deduce_vk_image_aspect(texture.format), mip-1, 0, texture.layers},
                        {
                            vk::Offset3D{0, 0, 0},
                            vk::Offset3D{(i32)pw, (i32)ph, (i32)pd},
                        },
                        vk::ImageSubresourceLayers{deduce_vk_image_aspect(texture.format), mip, 0, texture.layers},
                        {
                            vk::Offset3D{0, 0, 0},
                            vk::Offset3D{(i32)cw, (i32)ch, (i32)cd},
                        },
                    };

                    cmd.blitImage2(vk::BlitImageInfo2{
                        texture.image, vk::ImageLayout::eTransferSrcOptimal,
                        texture.image, vk::ImageLayout::eTransferDstOptimal,
                        blit,
                        vk::Filter::eLinear
                    });

                    barrier = {
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
                        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
                        vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                        {}, {}, texture.image, {deduce_vk_image_aspect(texture.format), mip-1, 1, 0, texture.layers}
                    };

                    cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});

                    if(mip == texture.mips - 1) {
                        barrier = {
                            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
                            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                            {}, {}, texture.image, {deduce_vk_image_aspect(texture.format), mip, 1, 0, texture.layers}
                        };

                        cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});
                    }
                }
            } else {
                barrier = {
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
                    vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                    {}, {}, texture.image, {deduce_vk_image_aspect(texture.format), 0, 1, 0, 1}
                };

                cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});
            }

            texture.current_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        } else if(auto* handle = std::get_if<Handle<BufferAllocation>>(&job.storage)) {
            const auto copy_region = vk::BufferCopy2{offset, 0, job.data.size()};
            cmd.copyBuffer2(vk::CopyBufferInfo2{
                staging_buffer.buffer,
                get_buffer(*handle).buffer,
                copy_region
            });
        } else { std::terminate(); }

        offset += job.data.size();
    }

    get_context().renderer->deletion_queue.push_back([this, vma = this->vma, buffer = staging_buffer.allocation.handle] {
        auto it = std::lower_bound(buffers.begin(), buffers.end(), buffer);
        if(it == buffers.end()) { return; }
        vmaDestroyBuffer(vma, it->buffer, it->alloc);
        buffers.erase(it);
    });
    jobs.clear();
}

BufferAllocation* RendererAllocator::create_buffer_ptr(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, u64 size_bytes) {
    if(!map_memory) {
        usage |= vk::BufferUsageFlagBits::eTransferDst;
    }

    vk::BufferCreateInfo buffer_info{
        {},
        (VkDeviceSize)size_bytes,
        usage
    };

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    if(map_memory) { alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT; }    

    VkBuffer buffer;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_i;
    if(VK_SUCCESS != vmaCreateBuffer(vma, (VkBufferCreateInfo*)&buffer_info, &alloc_info, &buffer, &alloc, &alloc_i)) {
        return nullptr;
    }

    auto& b = buffers.emplace_back(
        buffer,
        usage,
        alloc_i.pMappedData,
        size_bytes,
        alloc
    );

    set_debug_name(b.buffer, label);

    return &b;
}

TextureAllocation* RendererAllocator::create_texture_ptr(std::string_view label, const vk::ImageCreateInfo& info) {
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    
    VkImage image;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_i;
    if(VK_SUCCESS != vmaCreateImage(vma, (VkImageCreateInfo*)&info, &alloc_info, &image, &alloc, &alloc_i)) {
        return nullptr;
    }

    auto& texture = textures.emplace_back(
        info.imageType,
        info.extent.width,
        info.extent.height,
        info.extent.depth,
        info.mipLevels,
        info.arrayLayers,
        info.format,
        vk::ImageLayout::eUndefined,
        image,
        alloc,
        get_context().renderer->device.createImageView(vk::ImageViewCreateInfo{
            {}, 
            image,
            to_vk_view_type(info.imageType),
            info.format,
            {},
            vk::ImageSubresourceRange{deduce_vk_image_aspect(info.format), 0, info.mipLevels, 0, info.arrayLayers}
        })
    );

    set_debug_name(texture.image, label);
    set_debug_name(texture.default_view, std::format("{}_default_view", label));

    return &texture;
}