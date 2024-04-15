#include "render_graph.hpp"
#include "renderer_types.hpp"
#include "context.hpp"
#include "renderer.hpp"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_to_string.hpp>
#include <chrono>

#if 1

static constexpr vk::PipelineStageFlags2 to_vk_pipeline_stage(RGSyncStage stage);
static constexpr vk::ImageAspectFlags to_vk_aspect(RGImageAspect aspect);
static constexpr vk::ImageSubresourceRange to_vk_subresource_range(TextureRange range, RGImageAspect aspect);
static constexpr vk::ImageLayout to_vk_layout(RGImageLayout layout);
static constexpr vk::AttachmentLoadOp to_vk_load_op(RGAttachmentLoadStoreOp op);
static constexpr vk::AttachmentStoreOp to_vk_store_op(RGAttachmentLoadStoreOp op);
static constexpr vk::PipelineBindPoint to_vk_bind_point(PipelineType type);
static constexpr vk::ImageViewType vk_img_type_to_vk_img_view_type(vk::ImageType type);
static constexpr vk::Format to_vk_format(RGImageFormat format);

bool TextureRange::intersects(TextureRange r) const {
    return (
        base_layer < r.base_layer + r.layers &&
        base_layer + layers > r.base_layer &&
        base_mip < r.base_mip + r.mips &&
        base_mip + mips > r.base_mip
    );
}

bool TextureRange::fully_contains(TextureRange r) const {
    return (
        base_mip <= r.base_mip &&
        base_mip + mips >= r.base_mip + r.mips &&
        base_layer <= r.base_layer &&
        base_layer + layers >= r.base_layer + r.layers
    );
}

TextureRange TextureRange::get_overlap(TextureRange r) const {
    const auto a = std::max(base_mip, r.base_mip);
    const auto b = std::max(base_layer, r.base_layer);
    const auto x = std::min(base_mip+mips, r.base_mip+r.mips);
    const auto y = std::min(base_layer+layers, r.base_layer+r.layers);
    return TextureRange{
        .base_mip = a,
        .mips = x - std::min(x, a),
        .base_layer = b,
        .layers = y - std::min(y, b)
    };
}

void RGLayoutRanges::subtract(TextureRange range) {
    std::vector<TextureRange> new_ranges;
    new_ranges.reserve(ranges.size());

    for(auto& r : ranges) {
        if(!range.intersects(r)) { new_ranges.push_back(r); continue; }
        if(range.fully_contains(r)) { continue; }
        handle_subdivision(range, r, new_ranges);
    }   

    new_ranges.shrink_to_fit();
    ranges = std::move(new_ranges);
}

void RGLayoutRanges::handle_subdivision(TextureRange range, TextureRange r, std::vector<TextureRange>& new_ranges) {
    /*
        During range subtraction, there are multiple cases. 
        Let [] denote a range which the intersection with another range, (), 
        will be subtracted from.

        Each range is a continous range of layers, that each have equal number of mips.
        Case 1: ( [ ) ] - range () overlaps with only the left part of the range [] -> divide into ()[]
        Case 2: [ ( ] ) - range () ovelaps with only the right part of the range [] -> divide into []()
        Case 3: [ ( ) ] - range () is contained within range [] -> divide into ()[]()
        Case 4: [(   )] - range () spans exactly the same range of layers as the range [] -> divide by mip regions.

        Additionaly, range () can overlap different part of mips in the range []. In all cases,
        a small region of only mips can be either above or below the range ().
    */

    const auto overlap = range.get_overlap(r);

    if(range.base_layer <= r.base_layer && range.base_layer + range.layers < r.base_layer + r.layers) {
        // Case 1
        handle_mips(overlap, r, new_ranges);
        r.layers = r.base_layer + r.layers - (overlap.base_layer + overlap.layers);
        r.base_layer = overlap.base_layer + overlap.layers;
    } else if(range.base_layer > r.base_layer && range.base_layer + range.layers <= r.base_layer + r.layers) {
        // Case 2
        handle_mips(overlap, r, new_ranges);
        r.layers = overlap.base_layer;
    } else if(range.base_layer > r.base_layer && range.base_layer + range.layers < r.base_layer + r.layers) {
        // Case 3
        handle_mips(overlap, r, new_ranges);
        auto right = r;
        right.layers = right.base_layer + right.layers - (overlap.base_layer + overlap.layers);
        right.base_layer = overlap.base_layer + overlap.layers;
        new_ranges.push_back(right);
        r.layers = overlap.base_layer;
    } else if(range.base_layer == r.base_layer && range.layers == r.layers) {
        // Case 4
        handle_mips(overlap, r, new_ranges);
        return;
    } else {
        spdlog::error("No more cases to cover. This is an error.");
        std::abort();
    }

    new_ranges.push_back(r);
}

void RGLayoutRanges::handle_mips(TextureRange overlap, TextureRange r, std::vector<TextureRange>& new_ranges) {
    r.base_layer = overlap.base_layer;
    r.layers = overlap.layers;
    if(r.base_mip < overlap.base_mip) {
        r.mips = overlap.base_mip;
    } else if(r.base_mip > overlap.base_mip) {
        r.mips = r.base_mip + r.mips - (overlap.base_mip + overlap.mips);
        r.base_mip = overlap.base_mip + overlap.mips;
    } else {
        return;
    }
    new_ranges.push_back(r);
}
    
void RGTextureAccesses::insert_read(RGTextureAccess access) {
    layouts.emplace_back(last_read.size(), true);
    last_read.push_back(access);
}

void RGTextureAccesses::insert_write(RGTextureAccess access) {
    layouts.emplace_back(last_written.size(), false);
    last_written.push_back(access);
}

RGTextureAccess& RGTextureAccesses::get_layout_texture_access(RGLayoutAccess access) {
    auto* vec = &last_read;
    if(!access.is_read) { vec = &last_written; }
    return vec->at(access.access_idx);
}

const RGTextureAccess& RGTextureAccesses::get_layout_texture_access(RGLayoutAccess access) const {
    auto* vec = &last_read;
    if(!access.is_read) { vec = &last_written; }
    return vec->at(access.access_idx);
}

RGLayoutChangeQuery RGTextureAccesses::query_layout_changes(TextureRange range, bool is_read) const {
    RGLayoutChangeQuery query;
    query.previously_unaccessed.ranges.push_back(range);
    for(auto it = layouts.rbegin(); it != layouts.rend(); ++it) {
        auto& access = *it;
        auto& txt_access = access.is_read ? last_read.at(access.access_idx) : last_written.at(access.access_idx);
        if(auto overlap = txt_access.range.get_overlap(range); overlap.mips * overlap.layers != 0u) {
            if(std::find_if(begin(query.accesses), end(query.accesses), [&overlap](auto& e) { return e.second.fully_contains(overlap); }) == end(query.accesses)) {
                query.accesses.emplace_back(access, overlap);
                query.previously_unaccessed.subtract(overlap);
            }
        }

        if(query.previously_unaccessed.empty()) { break; }
    }

    return query;
}

RenderPass& RenderPass::set_name(const std::string& name) {
    this->name = name;
    return *this;
}

RenderPass& RenderPass::set_draw_func(std::function<void(vk::CommandBuffer)>&& f) {
    func = std::move(f);
    return *this;
}

RenderPass& RenderPass::write_to_image(RPResource info) {
    info.usage = RGResourceUsage::Image;
    info.is_read = false;
    resources.push_back(info);
    return *this;
}

RenderPass& RenderPass::read_from_image(RPResource info) {
    info.usage = RGResourceUsage::Image;
    info.is_read = true;
    resources.push_back(info);
    return *this;
}

RenderPass& RenderPass::write_color_attachment(RPResource info) {
    info.usage = RGResourceUsage::ColorAttachment;
    info.is_read = false;
    resources.push_back(info);
    color_attachments.push_back(resources.size() - 1);
    return *this;
}

RenderPass& RenderPass::read_color_attachment(RPResource info) {
    info.usage = RGResourceUsage::ColorAttachment;
    info.is_read = true;
    resources.push_back(info);
    color_attachments.push_back(resources.size() - 1);
    return *this;
}

RenderPass& RenderPass::write_depth_attachment(RPResource info) {
    info.usage = RGResourceUsage::DepthAttachment;
    info.is_read = false;
    resources.push_back(info);
    depth_attachment = resources.size() - 1;
    return *this;
}

RenderPass& RenderPass::set_rendering_extent(const RenderPassRenderingExtent& extent) {
    this->extent = extent;
    return *this;
}

RenderPass& RenderPass::set_pipeline(Pipeline* pipeline) {
    this->pipeline = pipeline;
    return *this;
}

RenderPass& RenderPass::set_make_sampler(bool make) {
    make_sampler = make;
    return *this;
}

const RPResource* RenderPass::get_resource(RgResourceHandle handle) const {
    for(auto& r : resources) {
        if(r.resource == handle) { return &r; }
    }
    return nullptr;
}

void RenderGraph::create_rendering_resources() {
    auto* renderer = get_context().renderer;

    for(auto& pass : passes) {
        pass.pipeline->layout.descriptor_sets[0].bindings[0].
        std::vector<DescriptorInfo> descriptor_infos;
        for(const auto& rp_resource : pass.resources) {
            const auto& rg_resource = resources.at(rp_resource.resource);

            std::visit(visitor{
                    [&](const std::pair<Texture, RGTextureAccesses>& resource) {
                        const auto& [texture, accesses] = resource;
                        if(!texture) { return; /*Swapchain image*/ }

                        auto view = renderer->device.createImageView(vk::ImageViewCreateInfo{
                            {},
                            texture.image,
                            vk_img_type_to_vk_img_view_type(texture->type),
                            rp_resource.texture_info.mutable_format == RGImageFormat::DeduceFromVkImage ? texture->format : to_vk_format(rp_resource.texture_info.mutable_format),
                            {},
                            to_vk_subresource_range(rp_resource.texture_info.range, rp_resource.usage == RGResourceUsage::DepthAttachment ? RGImageAspect::Depth : RGImageAspect::Color)
                        });

                        set_debug_name(renderer->device, view, std::format("{}_rgview", rg_resource.name));

                        image_views.emplace(std::make_pair(&pass, rp_resource.resource), view);

                        if(rp_resource.usage != RGResourceUsage::Image) { return; }

                        if(pass.pipeline) {
                            auto binding = pass.pipeline->layout.find_binding(rg_resource.name);
                            if(!binding) { return; }
                            descriptor_infos.push_back(DescriptorInfo{
                                to_vk_desc_type(binding->type),
                                view,
                                to_vk_layout(rp_resource.texture_info.required_layout)
                            });
                        }
                    },
                    [](const auto&) { std::terminate(); }
                },
                rg_resource.resource
            );
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

    #ifdef RG_DEBUG_PRINT
    const auto t1 = std::chrono::steady_clock::now();
    #endif

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
            std::get<1>(graph_resource.resource).first.image,
            to_vk_subresource_range(range, dst_resource.usage == RGResourceUsage::DepthAttachment ? RGImageAspect::Depth : RGImageAspect::Color)
        };

        if(barrier.image) {
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

                    const auto query = std::get<1>(graph_resource.resource).second.query_layout_changes(prti.range, is_read);
                    for(const auto& [layout_access, overlap] : query.accesses) {
                        const auto& texture_access = std::get<1>(graph_resource.resource).second.get_layout_texture_access(layout_access);
                        const auto barrier_stage = pass_stage.at(texture_access.pass) + 1u;    
                        stage = std::max(stage, barrier_stage);
                        const auto& src_resource = *texture_access.pass->get_resource(pass_resource.resource);
                        insert_barrier(barrier_stage, texture_access.pass, &pass, src_resource, pass_resource, to_vk_layout(texture_access.layout), to_vk_layout(pass_resource.texture_info.required_layout), layout_access.is_read, is_read, overlap);
                    }
                    for(const auto& r : query.previously_unaccessed.ranges) {
                        vk::ImageLayout old_layout{vk::ImageLayout::eUndefined};
                        if(std::get<1>(graph_resource.resource).first) { old_layout = std::get<1>(graph_resource.resource).first->current_layout; }
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
                std::get<1>(graph_resource.resource).second.insert_read(RGTextureAccess{&pass, pass_resource.texture_info.required_layout, pass_resource.texture_info.range});
            } else {
                std::get<1>(graph_resource.resource).second.insert_write(RGTextureAccess{&pass, pass_resource.texture_info.required_layout, pass_resource.texture_info.range});
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

    #ifdef RG_DEBUG_PRINT
    const auto t2 = std::chrono::steady_clock::now();
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    spdlog::info("Baked graph in: {}ns", dt);

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
#ifdef RG_DEBUG_PRINT
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
#endif
        }

        for(u32 i = offset; i < offset + pass_count; ++i) {
            const auto& pass = passes.at(i);
#ifdef RG_DEBUG_PRINT
            spdlog::debug("{}", pass.name);
#endif

            if(pass.pipeline) {
                auto* renderer = get_context().renderer;
                cmd.bindPipeline(to_vk_bind_point(pass.pipeline->type), pass.pipeline->pipeline);
                // vk::DescriptorSet sets_to_bind[] {
                //     renderer->global_set.set,
                //     pass.set->set,
                // };
                // cmd.bindDescriptorSets(to_vk_bind_point(pass.pipeline->type), pass.pipeline->layout.layout, 0, sets_to_bind, {});

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

                        if(std::get<1>(rg_resource.resource).first) {
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

#endif