#include "render_graph.hpp"
#include "renderer_types.hpp"
#include "context.hpp"
#include "renderer.hpp"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_to_string.hpp>
#include <chrono>

static constexpr vk::PipelineStageFlags2 to_vk_pipeline_stage(RGSyncStage stage) {
    switch (stage) {
        case RGSyncStage::None:          { return vk::PipelineStageFlagBits2::eNone; }
        case RGSyncStage::Transfer:      { return vk::PipelineStageFlagBits2::eTransfer; }
        case RGSyncStage::Fragment:      { return vk::PipelineStageFlagBits2::eFragmentShader; }
        case RGSyncStage::EarlyFragment: { return vk::PipelineStageFlagBits2::eEarlyFragmentTests; }
        case RGSyncStage::LateFragment:  { return vk::PipelineStageFlagBits2::eLateFragmentTests; }
        case RGSyncStage::Compute:       { return vk::PipelineStageFlagBits2::eComputeShader; }
        case RGSyncStage::ColorAttachmentOutput: { return vk::PipelineStageFlagBits2::eColorAttachmentOutput; }
        case RGSyncStage::AllGraphics: { return vk::PipelineStageFlagBits2::eAllGraphics; }
        default: {
            spdlog::error("Unrecognized RGSyncStage {}", (u32)stage);
            std::abort();
        }
    }
}

static constexpr vk::ImageAspectFlags to_vk_aspect(RGImageAspect aspect) {
    switch (aspect) {
        case RGImageAspect::Color: { return vk::ImageAspectFlagBits::eColor; }
        case RGImageAspect::Depth: { return vk::ImageAspectFlagBits::eDepth; }
        default: {
            spdlog::error("Unrecognized RgImageAspect: {}", (u32)aspect);
            std::abort();
        }
    }
}

static constexpr vk::ImageSubresourceRange to_vk_subresource_range(TextureRange range, RGImageAspect aspect) {
    return vk::ImageSubresourceRange{
        to_vk_aspect(aspect),
        range.base_mip, range.mips,
        range.base_layer, range.layers
    };
}

static constexpr vk::ImageLayout to_vk_layout(RGImageLayout layout) {
    switch(layout) {
        case RGImageLayout::Attachment: { return vk::ImageLayout::eAttachmentOptimal; }
        case RGImageLayout::General: { return vk::ImageLayout::eGeneral; }
        case RGImageLayout::ReadOnly: { return vk::ImageLayout::eShaderReadOnlyOptimal; }
        case RGImageLayout::TransferSrc: { return vk::ImageLayout::eTransferSrcOptimal; }
        case RGImageLayout::TransferDst: { return vk::ImageLayout::eTransferDstOptimal; }
        case RGImageLayout::PresentSrc: { return vk::ImageLayout::ePresentSrcKHR; }
        case RGImageLayout::Undefined: { return vk::ImageLayout::eUndefined; }
        default: {
            spdlog::error("Unrecognized RGImageLayout: {}", (u32)layout);
            std::abort();
        }
    }
}

static constexpr vk::AttachmentLoadOp to_vk_load_op(RGAttachmentLoadStoreOp op) {
    switch(op) {
        case RGAttachmentLoadStoreOp::DontCare: { return vk::AttachmentLoadOp::eDontCare; }
        case RGAttachmentLoadStoreOp::Clear: { return vk::AttachmentLoadOp::eClear; }
        case RGAttachmentLoadStoreOp::Load: { return vk::AttachmentLoadOp::eLoad; }
        default: {
            spdlog::error("Unrecognized RGAttachmentLoadOp {}", (u32)op);
            std::abort();
        }
    }
}

static constexpr vk::AttachmentStoreOp to_vk_store_op(RGAttachmentLoadStoreOp op) {
    switch(op) {
        case RGAttachmentLoadStoreOp::DontCare: { return vk::AttachmentStoreOp::eDontCare; }
        case RGAttachmentLoadStoreOp::Store: { return vk::AttachmentStoreOp::eStore; }
        case RGAttachmentLoadStoreOp::None: { return vk::AttachmentStoreOp::eNone; }
        default: {
            spdlog::error("Unrecognized RGAttachmentStoreOp {}", (u32)op);
            std::abort();
        }
    }
}

static constexpr vk::PipelineBindPoint to_vk_bind_point(PipelineType type) { 
    switch (type) {
        case PipelineType::Compute: { return vk::PipelineBindPoint::eCompute; }
        case PipelineType::Graphics: { return vk::PipelineBindPoint::eGraphics; }
        default: {
            spdlog::error("Unrecognized PipelineType {}", (u32)type);
            std::abort();
        } 
    }
}

static constexpr vk::ImageViewType vk_img_type_to_vk_img_view_type(vk::ImageType type) {
    switch(type) {
        case vk::ImageType::e2D: { return vk::ImageViewType::e2D; }
        case vk::ImageType::e3D: { return vk::ImageViewType::e3D; }
        default: {
            spdlog::error("Unsupported vk::ImageType {}", (u32)type);
            std::abort();
        }
    }
}

static constexpr vk::DescriptorType to_vk_desc_type(DescriptorType type) {
    switch (type) {
        case DescriptorType::SampledImage: { return vk::DescriptorType::eSampledImage; }
        case DescriptorType::StorageImage: { return vk::DescriptorType::eStorageImage; }
        case DescriptorType::StorageBuffer: { return vk::DescriptorType::eStorageBuffer; }
        case DescriptorType::UniformBuffer: { return vk::DescriptorType::eUniformBuffer; }
        case DescriptorType::Sampler: { return vk::DescriptorType::eSampler; }
        default: {
            spdlog::error("Unrecognized descriptor type {}", (u32)type);
            std::abort();
        }
    }
}

static constexpr vk::Format to_vk_format(RGImageFormat format) {
    switch(format) {
        case RGImageFormat::R32UI: { return vk::Format::eR32Uint; }
        case RGImageFormat::RGBA8Unorm: { return vk::Format::eR8G8B8A8Unorm; }
        default: {
            spdlog::error("Unrecognized RGImageFormat {}", (u32)format);
            std::abort();
        }
    }
}

void RenderGraph::create_rendering_resources() {
    auto* renderer = get_context().renderer;

    for(auto& pass : passes) {
        std::vector<DescriptorInfo> descriptor_infos;
        for(const auto& rp_resource : pass.resources) {
            const auto& rg_resource = resources.at(rp_resource.resource);

            if(!rg_resource.texture) { continue; /*Swapchain image*/ }

            auto view = renderer->device.createImageView(vk::ImageViewCreateInfo{
                {},
                (*rg_resource.texture)->image,
                vk_img_type_to_vk_img_view_type((*rg_resource.texture)->type),
                rp_resource.texture_info.mutable_format == RGImageFormat::DeduceFromVkImage ? (*rg_resource.texture)->format : to_vk_format(rp_resource.texture_info.mutable_format),
                {},
                to_vk_subresource_range(rp_resource.texture_info.range, rp_resource.usage == RGResourceUsage::DepthAttachment ? RGImageAspect::Depth : RGImageAspect::Color)
            });

            set_debug_name(renderer->device, view, std::format("{}_rgview", rg_resource.name));

            image_views.emplace(std::make_pair(&pass, rp_resource.resource), view);

            if(rp_resource.usage != RGResourceUsage::Image) { continue; }
            if(pass.pipeline) {
                auto binding = pass.pipeline->layout.find_binding(rg_resource.name);
                if(!binding) {
                    continue;
                }
                descriptor_infos.push_back(DescriptorInfo{
                    to_vk_desc_type(binding->type),
                    view,
                    to_vk_layout(rp_resource.texture_info.required_layout)
                });
            }
        }

        if(pass.pipeline) {
            if(pass.make_sampler) {
                auto sampler = renderer->device.createSampler(vk::SamplerCreateInfo{
                    {},
                    vk::Filter::eLinear,
                    vk::Filter::eLinear,
                    vk::SamplerMipmapMode::eLinear,
                    {}, {}, {},
                    0.0f, false, 0.0f, false,
                    {},
                    0.0f, 9.0f
                });
                descriptor_infos.push_back(DescriptorInfo{vk::DescriptorType::eSampler, sampler});
            }
            
            pass.set = new DescriptorSet{renderer->device, renderer->global_desc_pool, pass.pipeline->layout.descriptor_sets.at(1).layout};
            pass.set->update_bindings(renderer->device, 0, 0, descriptor_infos);
        }
    }
}

RenderGraph::BarrierStages RenderGraph::deduce_stages_and_accesses(const RenderPass* src_pass, const RenderPass* dst_pass, const RPResource& src_resource, const RPResource& dst_resource, bool src_read, bool dst_read) const {
    const auto get_access = [&](RGSyncStage stage, const RPResource& resource, bool read) {
        switch (stage) {
            case RGSyncStage::Transfer: {
                if(read) { return vk::AccessFlagBits2::eTransferRead; }
                else     { return vk::AccessFlagBits2::eTransferWrite; }
            }
            case RGSyncStage::EarlyFragment:
            case RGSyncStage::LateFragment: {
                if(read) { return vk::AccessFlagBits2::eDepthStencilAttachmentRead; }
                else     { return vk::AccessFlagBits2::eDepthStencilAttachmentWrite; }
            }
            case RGSyncStage::ColorAttachmentOutput: {
                if(read) { return vk::AccessFlagBits2::eColorAttachmentRead; }   
                else     { return vk::AccessFlagBits2::eColorAttachmentWrite; }   
            }
            case RGSyncStage::AllGraphics:
            case RGSyncStage::Compute:
            case RGSyncStage::Fragment: {
                switch(resource.usage) {
                    case RGResourceUsage::Image: {
                        auto& rg_resource = resources.at(resource.resource);
                        const DescriptorBinding* binding = nullptr;
                        if(src_pass && src_pass->pipeline) {
                            binding = src_pass->pipeline->layout.find_binding(rg_resource.name);
                        }
                        if(!binding && dst_pass && dst_pass->pipeline) {
                            binding = dst_pass->pipeline->layout.find_binding(rg_resource.name);
                        }
                        if(!binding) {
                            spdlog::error("Trying to synchronize image resource {} that is not found in both passes' layouts {} {}", rg_resource.name, src_pass->name, dst_pass->name);
                            std::abort();
                        }

                        if(binding->type == DescriptorType::StorageImage) {
                            if(read)    { return vk::AccessFlagBits2::eShaderStorageRead; }
                            else        { return vk::AccessFlagBits2::eShaderStorageWrite; }
                        } else if(binding->type == DescriptorType::SampledImage) {
                            if(read)    { return vk::AccessFlagBits2::eShaderSampledRead; }
                            else {
                                spdlog::error("You cannot write to a sampled image.");
                                std::abort();
                            }
                        } else {
                            spdlog::error("Unrecognized binding type {} for Image resource {} in stages {} {}", (u32)binding->type, rg_resource.name, src_pass->name, dst_pass->name);
                            std::abort();
                        }
                    }
                    case RGResourceUsage::ColorAttachment: { 
                        return read ? vk::AccessFlagBits2::eColorAttachmentRead : vk::AccessFlagBits2::eColorAttachmentWrite;
                    }
                    default: {
                        spdlog::error("Unsupported resource usage");
                        std::abort();
                    }
                }   
            }
            default: {
                spdlog::error("Unrecognized RGSyncStage {}", (u32)stage);
                std::abort();
            }
        }
    };

    return BarrierStages{
        .src_stage = src_pass ? to_vk_pipeline_stage(src_resource.stage) : vk::PipelineStageFlagBits2::eNone,
        .dst_stage = dst_pass ? to_vk_pipeline_stage(dst_resource.stage) : vk::PipelineStageFlagBits2::eNone,
        .src_access = src_pass ? get_access(src_resource.stage, src_resource, src_read) : vk::AccessFlagBits2::eNone,
        .dst_access = dst_pass ? get_access(dst_resource.stage, dst_resource, dst_read) : vk::AccessFlagBits2::eNone,
    };
}

void RenderGraph::bake_graph() {
    using ResourceIndex = u32;

    const auto t1 = std::chrono::steady_clock::now();

    std::vector<std::vector<ResourceIndex>> stages;
    std::vector<PassDependencies> stage_dependencies;
    std::unordered_map<RenderPass*, uint32_t> pass_stage;

    const auto get_stage = [&](u64 idx) -> auto& { 
        stages.resize(std::max(stages.size(), idx+1));
        return stages.at(idx);
    };
    const auto get_stage_dependencies = [&](u64 idx) -> auto& {
        stage_dependencies.resize(std::max(stage_dependencies.size(), idx+1));
        return stage_dependencies.at(idx);
    };
    const auto insert_barrier = [&](u32 stage, const RenderPass* src_pass, const RenderPass* dst_pass, const RPResource& src_resource, const RPResource& dst_resource, vk::ImageLayout old_layout, vk::ImageLayout new_layout, bool src_read, bool dst_read, TextureRange range) {
        auto& deps = get_stage_dependencies(stage);
        const auto stages = deduce_stages_and_accesses(src_pass, dst_pass, src_resource, dst_resource, src_read, dst_read);
        auto& graph_resource = resources.at(dst_resource.resource);

        vk::ImageMemoryBarrier2 barrier{
            stages.src_stage,
            stages.src_access,
            stages.dst_stage,
            stages.dst_access,
            old_layout,
            new_layout,
            vk::QueueFamilyIgnored,
            vk::QueueFamilyIgnored,
            graph_resource.texture ? (*graph_resource.texture)->image : vk::Image{},
            to_vk_subresource_range(range, dst_resource.usage == RGResourceUsage::DepthAttachment ? RGImageAspect::Depth : RGImageAspect::Color)
        };

        if(graph_resource.texture) {
            deps.image_barriers.push_back(barrier);
        } else {
            deps.swapchain_image_barrier = barrier;
        }
    };
    
    for(u32 pass_idx = 0; auto& pass : passes) {
        u32 stage = 0u;

        for(auto& pass_resource : pass.resources) {
            auto& graph_resource = resources.at(pass_resource.resource); 

            switch (pass_resource.usage) {
                case RGResourceUsage::ColorAttachment:
                case RGResourceUsage::DepthAttachment:
                case RGResourceUsage::Image: {
                    auto& prti = pass_resource.texture_info;
                    const auto is_read = pass_resource.is_read;

                    const auto query = graph_resource.texture_accesses.query_layout_changes(prti.range, is_read);
                    for(const auto& [layout_access, overlap] : query.accesses) {
                        const auto& texture_access = graph_resource.texture_accesses.get_layout_texture_access(layout_access);
                        const auto barrier_stage = pass_stage.at(texture_access.pass) + 1u;    
                        stage = std::max(stage, barrier_stage);
                        const auto& src_resource = *texture_access.pass->get_resource(pass_resource.resource);
                        insert_barrier(barrier_stage, texture_access.pass, &pass, src_resource, pass_resource, to_vk_layout(texture_access.layout), to_vk_layout(pass_resource.texture_info.required_layout), layout_access.is_read, is_read, overlap);
                    }
                    for(const auto& r : query.previously_unaccessed.ranges) {
                        vk::ImageLayout old_layout{vk::ImageLayout::eUndefined};
                        if(graph_resource.texture) { old_layout = (*graph_resource.texture)->current_layout; }
                        insert_barrier(0, nullptr, &pass, pass_resource, pass_resource, old_layout, to_vk_layout(pass_resource.texture_info.required_layout), false, is_read, r);
                    }
                } break;
                default: {
                    spdlog::error("Unrecognized pass resource usage {}", (u32)pass_resource.usage);
                    std::abort();
                }
            }
        }

        for(auto& pass_resource : pass.resources) {
            auto& graph_resource = resources.at(pass_resource.resource);
            if(pass_resource.is_read) {
                graph_resource.texture_accesses.insert_read(RGTextureAccess{&pass, pass_resource.texture_info.required_layout, pass_resource.texture_info.range});
            } else {
                graph_resource.texture_accesses.insert_write(RGTextureAccess{&pass, pass_resource.texture_info.required_layout, pass_resource.texture_info.range});
            }
        }

        pass_stage[&pass] = stage;
        get_stage(stage).push_back(pass_idx);
        ++pass_idx;
    }

    std::vector<RenderPass> flat_resources;
    flat_resources.reserve(passes.size());
    stage_deps = std::move(stage_dependencies);
    stage_pass_counts.resize(stages.size());
    for(u32 stage_idx = 0; auto &s : stages) {
        for(auto &r : s) {
            flat_resources.push_back(std::move(passes.at(r)));
        }
        stage_pass_counts.at(stage_idx) = s.size();
        ++stage_idx;
    }
    passes = std::move(flat_resources);

    const auto t2 = std::chrono::steady_clock::now();
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    spdlog::info("Baked graph in: {}ns", dt);

    #if 0
    for(auto stage=0u,offset=0u; auto c : stage_pass_counts) {
        spdlog::debug("stage: {}", stage);

        auto& sd = stage_deps.at(stage);
        spdlog::debug("barriers: {}", sd.image_barriers.size() + (sd.swapchain_image_barrier.has_value() ? 1u : 0u));
        auto bs = sd.image_barriers;
        if(sd.swapchain_image_barrier) { bs.push_back(sd.swapchain_image_barrier.value()); }

        for(auto& ib : bs) {
            spdlog::debug("[{} {}] -> [{} {}] {}->{} [{}:{} {}:{}]",
                vk::to_string(ib.srcStageMask), 
                vk::to_string(ib.srcAccessMask), 
                vk::to_string(ib.dstStageMask), 
                vk::to_string(ib.dstAccessMask), 
                vk::to_string(ib.oldLayout), 
                vk::to_string(ib.newLayout), 
                ib.subresourceRange.baseMipLevel, 
                ib.subresourceRange.levelCount, 
                ib.subresourceRange.baseArrayLayer, 
                ib.subresourceRange.layerCount);
        }

        for(auto i=offset; i<offset + c; ++i) {
            auto& p = passes.at(i);
            spdlog::debug("pass: {}", p.name);
        }

        ++stage;
        offset += c;
    }
    #endif

    create_rendering_resources();
}

void RenderGraph::render(vk::CommandBuffer cmd, vk::Image swapchain_image, vk::ImageView swapchain_view) {
    for(u32 offset = 0, stage = 0; auto pass_count : stage_pass_counts) {

        auto& barriers = stage_deps.at(stage);
        // spdlog::debug("Stage {}. Barriers: {}", stage, barriers.image_barriers.size() + (barriers.swapchain_image_barrier.has_value()));

        std::vector<vk::ImageMemoryBarrier2> image_barriers;
        image_barriers.reserve(barriers.image_barriers.size() + 1u);
        image_barriers.insert(image_barriers.end(), barriers.image_barriers.begin(), barriers.image_barriers.end());
        if(barriers.swapchain_image_barrier) {
            barriers.swapchain_image_barrier.value().setImage(swapchain_image);
            image_barriers.push_back(barriers.swapchain_image_barrier.value());
        }

        vk::DependencyInfo dependency_info{};
        dependency_info.setImageMemoryBarriers(image_barriers);
        cmd.pipelineBarrier2(dependency_info);

        for(u32 i=0; i<dependency_info.imageMemoryBarrierCount; ++i) {
            auto& barrier = dependency_info.pImageMemoryBarriers[i];
            // spdlog::debug("({}) {}:{} -> {}:{} {} -> {} RNG: {} - {} {} - {}",
            //     (u64)(VkImage)barrier.image,
            //     vk::to_string(barrier.srcStageMask),
            //     vk::to_string(barrier.srcAccessMask),
            //     vk::to_string(barrier.dstStageMask),
            //     vk::to_string(barrier.dstAccessMask),
            //     vk::to_string(barrier.oldLayout),
            //     vk::to_string(barrier.newLayout),
            //     barrier.subresourceRange.baseMipLevel,
            //     barrier.subresourceRange.levelCount,
            //     barrier.subresourceRange.baseArrayLayer,
            //     barrier.subresourceRange.layerCount);
        }

        for(u32 i = offset; i < offset + pass_count; ++i) {
            const auto& pass = passes.at(i);
            // spdlog::debug("{}", pass.name);

            if(pass.pipeline) {
                auto* renderer = get_context().renderer;
                cmd.bindPipeline(to_vk_bind_point(pass.pipeline->type), pass.pipeline->pipeline);
                vk::DescriptorSet sets_to_bind[] {
                    renderer->global_set.set,
                    pass.set->set,
                };
                cmd.bindDescriptorSets(to_vk_bind_point(pass.pipeline->type), pass.pipeline->layout.layout, 0, sets_to_bind, {});

                if(pass.pipeline->type == PipelineType::Graphics) {
                    std::vector<vk::RenderingAttachmentInfo> color_attachments;
                    color_attachments.reserve(pass.color_attachments.size());
                    vk::RenderingAttachmentInfo depth_attachment;

                    for(auto color : pass.color_attachments) {
                        const auto& rp_resource = pass.resources.at(color);
                        const auto& rg_resource = resources.at(rp_resource.resource);

                        vk::RenderingAttachmentInfo attachment{
                            {},
                            vk::ImageLayout::eUndefined,
                            {},
                            {},
                            {},
                            to_vk_load_op(rp_resource.load_op),
                            to_vk_store_op(rp_resource.store_op),
                            vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}
                        };

                        if(rg_resource.texture) {
                            attachment.imageView = image_views.at(std::make_pair(&pass, rp_resource.resource));
                            attachment.imageLayout = to_vk_layout(rp_resource.texture_info.required_layout);
                        } else {
                            attachment.imageView = swapchain_view;
                            attachment.imageLayout = to_vk_layout(rp_resource.texture_info.required_layout);
                        }

                        color_attachments.push_back(attachment);
                    }

                    if(pass.depth_attachment) {
                        const auto& rp_resource = pass.resources.at(pass.depth_attachment.value());
                        const auto& rg_resource = resources.at(rp_resource.resource);
                        depth_attachment = vk::RenderingAttachmentInfo{
                            image_views.at(std::make_pair(&pass, rp_resource.resource)),
                            to_vk_layout(rp_resource.texture_info.required_layout),
                            {},
                            {},
                            {},
                            to_vk_load_op(rp_resource.load_op),
                            to_vk_store_op(rp_resource.store_op),
                            vk::ClearDepthStencilValue{1.0f, 0}
                        };
                    }

                    cmd.beginRendering(vk::RenderingInfo{
                        {},
                        {{(i32)pass.extent.scissor.scissor_x, (i32)pass.extent.scissor.scissor_y}, {pass.extent.scissor.scissor_width, pass.extent.scissor.scissor_height}},
                        pass.extent.layers,
                        pass.extent.viewmask,
                        color_attachments,
                        pass.depth_attachment ? &depth_attachment : nullptr,
                        nullptr
                    });
                    cmd.setViewportWithCount(vk::Viewport{
                        pass.extent.viewport.offset_x,
                        pass.extent.viewport.offset_y,
                        pass.extent.viewport.width,
                        pass.extent.viewport.height,
                        pass.extent.viewport.min_depth,
                        pass.extent.viewport.max_depth
                    });
                    cmd.setScissorWithCount(vk::Rect2D{
                        {
                            (i32)pass.extent.scissor.scissor_x,
                            (i32)pass.extent.scissor.scissor_y,
                        }, 
                        {
                            pass.extent.scissor.scissor_width,
                            pass.extent.scissor.scissor_height,
                        }
                    });
                }
            }

            if(pass.func) { pass.func(cmd); }

            if(pass.pipeline && pass.pipeline->type == PipelineType::Graphics) {
                cmd.endRendering();
            }
        }

        ++stage;
        offset += pass_count;
    }
}