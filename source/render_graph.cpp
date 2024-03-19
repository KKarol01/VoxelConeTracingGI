#include "render_graph.hpp"
#include "renderer_types.hpp"
#include "context.hpp"
#include "renderer.hpp"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_to_string.hpp>

static vk::PipelineStageFlags2 to_vk_pipeline_stage(RGSyncStage stage) {
    switch (stage) {
        using enum RGSyncStage;
        case None:          { return vk::PipelineStageFlagBits2::eNone; }
        case Transfer:      { return vk::PipelineStageFlagBits2::eTransfer; }
        case Fragment:      { return vk::PipelineStageFlagBits2::eFragmentShader; }
        case EarlyFragment: { return vk::PipelineStageFlagBits2::eEarlyFragmentTests; }
        case LateFragment:  { return vk::PipelineStageFlagBits2::eLateFragmentTests; }
        case Compute:       { return vk::PipelineStageFlagBits2::eComputeShader; }
        case ColorAttachmentOutput: { return vk::PipelineStageFlagBits2::eColorAttachmentOutput; }
        default: {
            spdlog::error("Unrecognized RGSyncStage {}", (u32)stage);
            std::abort();
        }
    }
}

static vk::ImageAspectFlags to_vk_aspect(RGImageAspect aspect) {
    switch (aspect) {
        case RGImageAspect::Color: {
            return vk::ImageAspectFlagBits::eColor;
        }
        case RGImageAspect::Depth: {
            return vk::ImageAspectFlagBits::eDepth;
        }
        default: {
            spdlog::error("Unrecognized RgImageAspect: {}", (u32)aspect);
            std::abort();
        }
    }
}

static vk::ImageSubresourceRange to_vk_subresource_range(TextureRange range, RGImageAspect aspect) {
    return vk::ImageSubresourceRange{
        to_vk_aspect(aspect),
        range.base_mip, range.mips,
        range.base_layer, range.layers
    };
}

static vk::ImageLayout to_vk_layout(RGImageLayout layout) {
    switch(layout) {
        case RGImageLayout::Attachment: { return vk::ImageLayout::eAttachmentOptimal; }
        case RGImageLayout::General: { return vk::ImageLayout::eGeneral; }
        case RGImageLayout::ReadOnly: { return vk::ImageLayout::eShaderReadOnlyOptimal; }
        case RGImageLayout::TransferSrc: { return vk::ImageLayout::eTransferSrcOptimal; }
        case RGImageLayout::TransferDst: { return vk::ImageLayout::eTransferDstOptimal; }
        case RGImageLayout::Undefined: { return vk::ImageLayout::eUndefined; }
        default: {
            spdlog::error("Unrecognized RGImageLayout: {}", (u32)layout);
            std::abort();
        }
    }
}

static vk::AttachmentLoadOp to_vk_load_op(RGAttachmentLoadStoreOp op) {
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

static vk::AttachmentStoreOp to_vk_store_op(RGAttachmentLoadStoreOp op) {
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

void RenderGraph::create_rendering_resources() {
    auto* renderer = get_context().renderer;

    for(const auto& pass : passes) {
        std::vector<u32> attachments;
        attachments.insert(attachments.end(), pass.color_attachments.begin(), pass.color_attachments.end());
        if(pass.depth_attachment) { attachments.push_back(pass.depth_attachment.value()); }

        for(const auto handle : attachments) {
            const auto& rp_resource = pass.write_resources.at(handle);
            const auto& rg_resource = resources.at(rp_resource.resource);

            if(!rg_resource.texture) { continue; /*Swapchain image*/ }

            auto view = renderer->device.createImageView(vk::ImageViewCreateInfo{
                {},
                rg_resource.texture->image,
                vk::ImageViewType::e2D,
                rg_resource.texture->format,
                {},
                to_vk_subresource_range(rp_resource.texture_info.range, rp_resource.usage == RGResourceUsage::DepthAttachment ? RGImageAspect::Depth : RGImageAspect::Color)
            });

            image_views.emplace(std::make_pair(&pass, rp_resource.resource), view);   
        }
    }
}

RenderGraph::BarrierStages RenderGraph::deduce_stages_and_accesses(const RenderPass* src_pass, const RenderPass* dst_pass, const RPResource& src_resource, const RPResource& dst_resource, bool src_read, bool dst_read) const {
    if(src_read && dst_read) {
        spdlog::info("Detected trying to insert barrier with read-after-read - this is not neccessary.");
        return BarrierStages{
            .src_stage = vk::PipelineStageFlagBits2::eNone,
            .dst_stage = vk::PipelineStageFlagBits2::eNone,
            .src_access = vk::AccessFlagBits2::eNone,
            .dst_access = vk::AccessFlagBits2::eNone,
        };
    }

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
            case RGSyncStage::Compute:
            case RGSyncStage::Fragment: {
                if(resource.usage == RGResourceUsage::Image) {
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
                } else {
                    spdlog::error("Compute/Fragment stage NonImage barrier access deduction not implemented");
                    std::abort();
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
    
    for(u32 pass_idx = 0; auto& pass : passes) {
        u32 stage = 0u;

        for(auto& pass_resource : pass.resources) {
            auto& graph_resource = resources.at(pass_resource.resource); 
            u32 barrier_stage = 0u;

            switch (pass_resource.usage) {
                case RGResourceUsage::ColorAttachment:
                case RGResourceUsage::DepthAttachment:
                case RGResourceUsage::Image: {
                    auto& txt_info = pass_resource.texture_info;
                    const auto is_read = pass_resource.is_read;
                    
                    const RGTextureAccess* access{};
                    bool is_access_read;
                    if(is_read) {
                        access = graph_resource.texture_accesses.find_intersection_in_writes(txt_info.range);
                        is_access_read = false;
                    } else {
                        const auto* write_access = graph_resource.texture_accesses.find_intersection_in_writes(txt_info.range);
                        const auto* read_access = graph_resource.texture_accesses.find_intersection_in_reads(txt_info.range);
                        const auto write_stage = write_access ? pass_stage.at(write_access->pass) : 0;
                        const auto read_stage = read_access ? pass_stage.at(read_access->pass) : 0;
                        if(write_stage > 0 && write_stage == read_stage) {
                            spdlog::error("write and read happening in the same stage. this is an error!");
                            std::abort();
                        }
                        if(write_access && !read_access) {
                            access = write_access;
                            is_access_read = false;
                        } else if(!write_access && read_access) {
                            access = read_access;
                            is_access_read = true;
                        } else if(write_access && read_access) {
                            if(write_stage > read_stage) { access = write_access; is_access_read = false; }
                            else                         { access = read_access; is_access_read = true; }
                        }
                    }

                    vk::ImageLayout old_layout, new_layout;
                    new_layout = to_vk_layout(pass_resource.texture_info.required_layout);

                    if(!access) {
                        old_layout = graph_resource.texture->current_layout;
                        if(old_layout == new_layout) {
                            continue;
                        }
                    } else {
                        const auto access_stage = pass_stage.at(access->pass);
                        if(is_access_read && is_read) {
                            barrier_stage = access_stage;
                        } else if(!is_read) {
                            barrier_stage = access_stage + 1u;
                        }

                        old_layout = to_vk_layout(access->layout);

                        //deduce stages
                        // get image
                        // get range
                        // update ranges
                        access->
                    }

                } break;
                default: {
                    spdlog::error("Unrecognized pass resource usage {}", (u32)pass_resource.usage);
                    std::abort();
                }
            }
            
            auto& pass_rg_resource = resources.at(pass_resource.resource);
            auto last_read_in = find_intersection(last_read, pass_resource.resource, pass_resource.texture_info.range);
            auto last_written_in = find_intersection(last_modified, pass_resource.resource, pass_resource.texture_info.range);
            const auto was_being_read = last_read_in != nullptr;
            const auto was_being_written = last_written_in != nullptr;

            stage = std::max({
                stage, 
                                // 1u to move to the next stage, so the appropriate barrier can be inserted before this read/write.
                was_being_read    ? 1u + pass_stage.at(last_read_in->first) : stage,
                was_being_written ? 1u + pass_stage.at(last_written_in->first) : stage
            });
            barrier_stage = std::max({
                barrier_stage, 
                was_being_read    ? 1u + pass_stage.at(last_read_in->first) : barrier_stage,
                was_being_written ? 1u + pass_stage.at(last_written_in->first) : barrier_stage
            });

            bool no_layout_transition = true;
            BarrierStages deduced_stages;
            RGImageLayout src_layout{RGImageLayout::Undefined}, dst_layout = pass_resource.texture_info.required_layout;
            RPResource* read_resource{nullptr};
            RPResource* written_resource{nullptr};
            if(was_being_read)      { read_resource = find_in_read_resources(pass_resource.resource, last_read_in->first); }
            if(was_being_written)   { written_resource = find_in_written_resources(pass_resource.resource, last_written_in->first); }
            if(pass_resource.usage == RGResourceUsage::Image &&
                !was_being_read && 
                !was_being_written && 
                pass_rg_resource.texture->current_layout != to_vk_layout(pass_resource.texture_info.required_layout)) {
                deduced_stages = deduce_stages_and_accesses(nullptr, &pass, pass_resource, pass_resource, false, is_being_read);    
                no_layout_transition = false;
            } else if(pass_resource.usage == RGResourceUsage::ColorAttachment || 
                        pass_resource.usage == RGResourceUsage::DepthAttachment) {

                if(!is_being_read) {
                    pass_resource.load_op = RGAttachmentLoadStoreOp::Clear;
                    pass_resource.store_op = RGAttachmentLoadStoreOp::Store;
                } else {
                    pass_resource.load_op = RGAttachmentLoadStoreOp::Load;
                    pass_resource.store_op = RGAttachmentLoadStoreOp::None;
                }

                if(!was_being_read && !was_being_written) {
                    deduced_stages.src_stage = vk::PipelineStageFlagBits2::eNone;
                    deduced_stages.src_access = vk::AccessFlagBits2::eNone;
                    deduced_stages.dst_stage = to_vk_pipeline_stage(pass_resource.stage);
                    if(pass_resource.usage == RGResourceUsage::ColorAttachment) {
                        deduced_stages.dst_access = is_being_read ? vk::AccessFlagBits2::eColorAttachmentRead : vk::AccessFlagBits2::eColorAttachmentWrite;
                    } else if(pass_resource.usage == RGResourceUsage::DepthAttachment) {
                        deduced_stages.dst_access = is_being_read ? vk::AccessFlagBits2::eDepthStencilAttachmentRead : vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
                    }
                    no_layout_transition = false;
                }
            }
            
            if(was_being_written && is_being_read) { 
                // read-after-write
                deduced_stages = deduce_stages_and_accesses(last_written_in->first, &pass, *written_resource, pass_resource, false, true);
                src_layout = written_resource->texture_info.required_layout;
            } else if(was_being_written && !is_being_read) { 
                // write-after-write
                deduced_stages = deduce_stages_and_accesses(last_written_in->first, &pass, *written_resource, pass_resource, false, false);
                src_layout = written_resource->texture_info.required_layout;
            } else if(was_being_read && !is_being_read) { 
                // write-after-read
                deduced_stages = deduce_stages_and_accesses(last_read_in->first, &pass, *read_resource, pass_resource, true, false);
                src_layout = read_resource->texture_info.required_layout;
            } else if(no_layout_transition) {
                // read-after-read and no layout transition needed, or no initial layout transition needed
                continue;
            }

            spdlog::debug("Resource {} is getting a barrier between passes {} -> {} in stage {}. Barrier: [{} {}] [{} {}] layout {} -> {} loadop: {} storeop: {}, range: [{} {} {} {}]", 
                pass_rg_resource.name,
                was_being_read ? last_read_in->first->name : was_being_written ? last_written_in->first->name : "None",
                pass.name, 
                barrier_stage,
                vk::to_string(deduced_stages.src_stage),
                vk::to_string(deduced_stages.src_access),
                vk::to_string(deduced_stages.dst_stage),
                vk::to_string(deduced_stages.dst_access),
                vk::to_string(to_vk_layout(src_layout)),
                vk::to_string(to_vk_layout(dst_layout)),
                vk::to_string(to_vk_load_op(pass_resource.load_op)),
                vk::to_string(to_vk_store_op(pass_resource.store_op)),
                pass_resource.texture_info.range.base_mip,
                pass_resource.texture_info.range.mips,
                pass_resource.texture_info.range.base_layer,
                pass_resource.texture_info.range.layers);

            std::vector<vk::ImageMemoryBarrier2>* barrier_storage{nullptr};
            vk::ImageMemoryBarrier2 barrier{
                deduced_stages.src_stage,
                deduced_stages.src_access,
                deduced_stages.dst_stage,
                deduced_stages.dst_access,
                to_vk_layout(src_layout),
                to_vk_layout(dst_layout),
                vk::QueueFamilyIgnored,
                vk::QueueFamilyIgnored,
                vk::Image{},
                to_vk_subresource_range(pass_resource.texture_info.range, pass_resource.usage == RGResourceUsage::DepthAttachment ? RGImageAspect::Depth : RGImageAspect::Color)
            };
            
            switch(pass_resource.usage) {
                case RGResourceUsage::Image:
                case RGResourceUsage::ColorAttachment:
                case RGResourceUsage::DepthAttachment: {
                    if(pass_rg_resource.texture) {
                        barrier_storage = &get_stage_dependencies(barrier_stage).image_barriers;
                        barrier.setImage(pass_rg_resource.texture->image);
                    } else {
                        // swapchain image case
                        get_stage_dependencies(barrier_stage).swapchain_image_barrier = barrier;                
                    }
                } break;
                default: {
                    spdlog::error("Unrecognized pass resource usage {}. Cannot select appropriate barrier storage.", (u32)pass_resource.usage);
                    std::abort();
                }
            }

            if(barrier_storage) {
                barrier_storage->push_back(barrier);
            } 
        }
        }

        for(auto &r : pass.read_resources) { 
            while(auto it = find_intersection(last_read, r.resource, r.texture_info.range)) {
                for(auto i = 0; i < last_read.at(r.resource).size(); ++i) {
                    if(last_read.at(r.resource).at(i).first == it->first) {
                        last_read.at(r.resource).erase(last_read.at(r.resource).begin() + i);
                    }
                }
            }
            last_read[r.resource].push_back(std::make_pair(&pass, r.texture_info.range));
        }
        for(auto &r : pass.write_resources) { 
            while(auto it = find_intersection(last_modified, r.resource, r.texture_info.range)) {
                for(auto i = 0; i < last_modified.at(r.resource).size(); ++i) {
                    if(last_modified.at(r.resource).at(i).first == it->first) {
                        last_modified.at(r.resource).erase(last_modified.at(r.resource).begin() + i);
                    }
                }
            }
            last_modified[r.resource].push_back(std::make_pair(&pass, r.texture_info.range));
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

    create_rendering_resources();
}

void RenderGraph::render(vk::CommandBuffer cmd, vk::Image swapchain_image, vk::ImageView swapchain_view) {
    for(u32 offset = 0, stage = 0; auto pass_count : stage_pass_counts) {

        auto& barriers = stage_deps.at(stage);
        spdlog::debug("Stage {}. Barriers: {}", stage, barriers.image_barriers.size() + (barriers.swapchain_image_barrier.has_value()));

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
            spdlog::debug("({}) {}:{} -> {}:{} {} -> {} RNG: {} - {} {} - {}",
                (u64)(VkImage)barrier.image,
                vk::to_string(barrier.srcStageMask),
                vk::to_string(barrier.srcAccessMask),
                vk::to_string(barrier.dstStageMask),
                vk::to_string(barrier.dstAccessMask),
                vk::to_string(barrier.oldLayout),
                vk::to_string(barrier.newLayout),
                barrier.subresourceRange.baseMipLevel,
                barrier.subresourceRange.levelCount,
                barrier.subresourceRange.baseArrayLayer,
                barrier.subresourceRange.layerCount);
        }

        for(u32 i = offset; i < offset + pass_count; ++i) {
            const auto& pass = passes.at(i);
            spdlog::debug("{}", pass.name);

            if(pass.pipeline) {
                if(pass.pipeline->type == PipelineType::Graphics) {
                    std::vector<vk::RenderingAttachmentInfo> color_attachments;
                    color_attachments.reserve(pass.color_attachments.size());
                    vk::RenderingAttachmentInfo depth_attachment;

                    for(auto color : pass.color_attachments) {
                        const auto& rp_resource = pass.write_resources.at(color);
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
                        const auto& rp_resource = pass.write_resources.at(pass.depth_attachment.value());
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

                if(pass.func) { pass.func(); }

                if(pass.pipeline->type == PipelineType::Graphics) {
                    cmd.endRendering();
                }
            }
        }

        ++stage;
        offset += pass_count;
    }
}