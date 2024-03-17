#include "render_graph.hpp"
#include "renderer_types.hpp"
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

RenderGraph::BarrierStages RenderGraph::deduce_stages_and_accesses(const RenderPass* src_pass, const RenderPass* dst_pass, RPResource& src_resource, RPResource& dst_resource, bool src_read, bool dst_read) {
    if(src_read && dst_read) {
        spdlog::info("Detected trying to insert barrier with read-after-read - this is not neccessary.");
        return BarrierStages{
            .src_stage = vk::PipelineStageFlagBits2::eNone,
            .dst_stage = vk::PipelineStageFlagBits2::eNone,
            .src_access = vk::AccessFlagBits2::eNone,
            .dst_access = vk::AccessFlagBits2::eNone,
        };
    }

    const auto get_access = [&](RGSyncStage stage, RPResource& resource, bool read) {
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
        .dst_stage = to_vk_pipeline_stage(dst_resource.stage),
        .src_access = src_pass ? get_access(src_resource.stage, src_resource, src_read) : vk::AccessFlagBits2::eNone,
        .dst_access = dst_pass ? get_access(dst_resource.stage, dst_resource, dst_read) : vk::AccessFlagBits2::eNone,
    };
}

vk::ImageAspectFlags to_vk_aspect(RGImageAspect aspect) {
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

vk::ImageSubresourceRange to_vk_subresource_range(TextureRange range, RGImageAspect aspect) {
    return vk::ImageSubresourceRange{
        to_vk_aspect(aspect),
        range.base_mip, range.mips,
        range.base_layer, range.layers
    };
}

vk::ImageLayout to_vk_layout(RGImageLayout layout) {
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

void RenderGraph::bake_graph() {
    using ResourceIndex = u32;

    std::vector<std::vector<ResourceIndex>> stages;
    std::vector<PassDependencies> stage_dependencies;
    std::unordered_map<RgResourceHandle, RenderPass*> last_modified, last_read;
    std::unordered_map<RenderPass*, uint32_t> pass_stage;

    const auto get_stage = [&](u64 idx) -> auto& { 
        stages.resize(std::max(stages.size(), idx+1));
        return stages.at(idx);
    };
    const auto get_stage_dependencies = [&](u64 idx) -> auto& {
        stage_dependencies.resize(std::max(stage_dependencies.size(), idx+1));
        return stage_dependencies.at(idx);
    };
    const auto find_in_written_resources = [&](RgResourceHandle idx, RenderPass* pass) -> auto* {
        auto it = std::find_if(begin(pass->write_resources), end(pass->write_resources), [idx](auto&& e) { return e.resource == idx; });
        return it != end(pass->write_resources) ? &*it : nullptr;
    };
    const auto find_in_read_resources = [&](RgResourceHandle idx, RenderPass* pass) -> auto* {
        auto it = std::find_if(begin(pass->read_resources), end(pass->read_resources), [idx](auto&& e) { return e.resource == idx; });
        return it != end(pass->read_resources) ? &*it : nullptr;
    };
    
    for(u32 pass_idx = 0; auto& pass : passes) {
        u32 stage = 0u;
        std::vector<RPResource>* pass_rw_resources[] {
            &pass.read_resources,
            &pass.write_resources
        };
        bool pass_rw_is_read[] { true, false };

        for(auto i=0u; i<sizeof(pass_rw_resources) / sizeof(pass_rw_resources[0]); ++i) {
            auto* pass_resources = pass_rw_resources[i];
            auto is_being_read = pass_rw_is_read[i];

            for(auto& pass_resource : *pass_resources) {
                u32 barrier_stage = 0u;
                auto& pass_rg_resource = resources.at(pass_resource.resource);
                auto last_read_in = last_read.find(pass_resource.resource);
                auto last_written_in = last_modified.find(pass_resource.resource);
                const auto was_being_read = last_read_in != last_read.end();
                const auto was_being_written = last_written_in != last_modified.end();

                stage = std::max({
                    stage, 
                                    // 1u to move to the next stage, so the appropriate barrier can be inserted before this read/write.
                    was_being_read    ? 1u + pass_stage.at(last_read_in->second) : stage,
                    was_being_written ? 1u + pass_stage.at(last_written_in->second) : stage
                });
                barrier_stage = std::max({
                    barrier_stage, 
                    was_being_read    ? 1u + pass_stage.at(last_read_in->second) : barrier_stage,
                    was_being_written ? 1u + pass_stage.at(last_written_in->second) : barrier_stage
                });

                bool no_layout_transition = true;
                BarrierStages deduced_stages;
                RGImageLayout src_layout{RGImageLayout::Undefined}, dst_layout = pass_resource.texture_info.required_layout;
                RPResource* read_resource{nullptr};
                RPResource* written_resource{nullptr};
                if(was_being_read)      { read_resource = find_in_read_resources(last_read_in->first, last_read_in->second); }
                if(was_being_written)   { written_resource = find_in_written_resources(last_written_in->first, last_written_in->second); }
                if(pass_resource.usage == RGResourceUsage::Image &&
                   !was_being_read && 
                   !was_being_written && 
                   pass_rg_resource.texture->current_layout != to_vk_layout(pass_resource.texture_info.required_layout)) {
                    deduced_stages = deduce_stages_and_accesses(nullptr, &pass, pass_resource, pass_resource, false, is_being_read);    
                    no_layout_transition = false;
                } else if((pass_resource.usage == RGResourceUsage::ColorAttachment || 
                           pass_resource.usage == RGResourceUsage::DepthAttachment) &&
                    !was_being_read && 
                    !was_being_written) {

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
                
                if(was_being_written && is_being_read) { // read-after-write
                    deduced_stages = deduce_stages_and_accesses(last_written_in->second, &pass, *written_resource, pass_resource, false, true);
                    src_layout = written_resource->texture_info.required_layout;
                } else if(was_being_written && !is_being_read) { // write-after-write
                    deduced_stages = deduce_stages_and_accesses(last_written_in->second, &pass, *written_resource, pass_resource, false, false);
                    src_layout = written_resource->texture_info.required_layout;
                } else if(was_being_read && !is_being_read) { // write-after-write
                    deduced_stages = deduce_stages_and_accesses(last_written_in->second, &pass, *read_resource, pass_resource, true, false);
                    src_layout = read_resource->texture_info.required_layout;
                } else if(no_layout_transition) {
                    // read-after-read and no layout transition needed, or no initial layout transition needed
                    continue;
                }

                spdlog::debug("Resource {} is getting a barrier between passes {} -> {} in stage {}. Barrier: [{} {}] [{} {}] layout {} -> {}", 
                    pass_rg_resource.name,
                    was_being_read ? last_read_in->second->name : was_being_written ? last_written_in->second->name : "None",
                    pass.name, 
                    barrier_stage,
                    vk::to_string(deduced_stages.src_stage),
                    vk::to_string(deduced_stages.src_access),
                    vk::to_string(deduced_stages.dst_stage),
                    vk::to_string(deduced_stages.dst_access),
                    vk::to_string(to_vk_layout(src_layout)),
                    vk::to_string(to_vk_layout(dst_layout)));
                
                std::vector<vk::ImageMemoryBarrier2>* barrier_storage;
                vk::Image resource_image{};
                if(pass_resource.usage == RGResourceUsage::Image) {
                    barrier_storage = &get_stage_dependencies(barrier_stage).mem_barriers;
                    resource_image = pass_rg_resource.texture->image;
                } else if(pass_resource.usage == RGResourceUsage::ColorAttachment ||
                          pass_resource.usage == RGResourceUsage::DepthAttachment) {
                    barrier_storage = &get_stage_dependencies(barrier_stage).attachment_barriers;
                } else {
                    spdlog::error("Unrecognized pass resource usage {}. Cannot select appropriate barrier storage.", (u32)pass_resource.usage);
                    std::abort();
                }
                
                barrier_storage->push_back(vk::ImageMemoryBarrier2{
                    deduced_stages.src_stage,
                    deduced_stages.src_access,
                    deduced_stages.dst_stage,
                    deduced_stages.dst_access,
                    to_vk_layout(src_layout),
                    to_vk_layout(dst_layout),
                    vk::QueueFamilyIgnored,
                    vk::QueueFamilyIgnored,
                    resource_image,
                    to_vk_subresource_range(pass_resource.texture_info.range, pass_resource.usage == RGResourceUsage::DepthAttachment ? RGImageAspect::Depth : RGImageAspect::Color)
                });
            }
        }

        for(auto &r : pass.read_resources) { last_read[r.resource] = &pass; }
        for(auto &r : pass.write_resources) { last_modified[r.resource] = &pass; }
        pass_stage[&pass] = stage;
        get_stage(stage).push_back(pass_idx);
        ++pass_idx;
    }

    std::vector<RenderPass> flat_resources;
    flat_resources.reserve(passes.size());
    stage_deps = std::move(stage_dependencies);
    stage_pass_count.resize(stages.size());
    for(u32 stage_idx = 0; auto &s : stages) {
        for(auto &r : s) {
            flat_resources.push_back(std::move(passes.at(r)));
        }
        stage_pass_count.at(stage_idx) = s.size();
        ++stage_idx;
    }
    passes = std::move(flat_resources);

    for(u32 stage_num=0; auto c : stage_pass_count) {
        spdlog::debug("In stage {} there are {} passes with {} barriers total", stage_num, c, stage_deps.at(stage_num).attachment_barriers.size() + stage_deps.at(stage_num).mem_barriers.size());
        ++stage_num;
    }
}