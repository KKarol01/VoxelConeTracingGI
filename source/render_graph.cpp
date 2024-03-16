#include "render_graph.hpp"
#include <spdlog/spdlog.h>

static vk::PipelineStageFlags2 RGStageToVk(RGSyncStage stage) {
    switch (stage) {
        using enum RGSyncStage;
        case None:          { return vk::PipelineStageFlagBits2::eNone; }
        case Transfer:      { return vk::PipelineStageFlagBits2::eTransfer; }
        case Fragment:      { return vk::PipelineStageFlagBits2::eFragmentShader; }
        case EarlyFragment: { return vk::PipelineStageFlagBits2::eEarlyFragmentTests; }
        case LateFragment:  { return vk::PipelineStageFlagBits2::eLateFragmentTests; }
        case Compute:       { return vk::PipelineStageFlagBits2::eComputeShader; }
        default: {
            spdlog::error("RGStageToVk failed at stage: {}. Substituting with all commands stage mask", (u32)stage);
            return vk::PipelineStageFlagBits2::eAllCommands;
        }
    }
}

enum class RGAccessType { None, Read, Write };
enum class RGImageAspect { None, Color, Depth };

static vk::AccessFlags2 RGStageDeduceAccess(RGSyncStage stage, RGAccessType access, RGImageAspect aspect) {
    //color, shader, transfer, depthstencil
    vk::AccessFlagBits2 read_access, write_access;

    switch (stage) {
        case RGSyncStage::Transfer: {
            read_access = vk::AccessFlagBits2::eTransferRead;
            write_access = vk::AccessFlagBits2::eTransferWrite;
        } break;
        case RGSyncStage::Fragment:
        case RGSyncStage::Compute: {
            read_access = vk::AccessFlagBits2::eShaderRead;
            write_access = vk::AccessFlagBits2::eShaderWrite;
        } break;
    }
}

void RenderGraph::bake_graph() {
    using ResourceIndex = u32;
    struct PassDependencies {
        std::vector<vk::ImageMemoryBarrier2> mem_barriers;
    };

    std::vector<std::vector<ResourceIndex>> stages;
    std::unordered_map<RgResourceHandle, RenderPass*> last_modified, last_read;
    std::unordered_map<RenderPass*, uint32_t> pass_stage;
    std::unordered_map<RenderPass*, PassDependencies> deps;

    const auto get_stage = [&](u64 idx) -> auto& { 
        stages.resize(std::max(stages.size(), idx+1));
        return stages.at(idx);
    };
    const auto get_pass_written_resource = [&](RgResourceHandle idx, RenderPass* pass) -> auto& {
        return *std::find_if(begin(pass->write_resources), end(pass->write_resources), [idx](auto&& e) { return e.resource == idx; });
    };
    
    for(u32 pass_idx = 0; auto& pass : passes) {
        uint32_t stage = 0;

        for(auto& read : pass.read_resources) {
            auto& read_resource = resources.at(read.resource);

            if(auto written_resource_pass = last_modified.find(read.resource); written_resource_pass != last_modified.end()) {
                auto written_resource_idx = written_resource_pass->first;
                auto* writing_pass = written_resource_pass->second;
                auto& written_resource = get_pass_written_resource(written_resource_idx, writing_pass);

                stage = std::max(stage, pass_stage.at(writing_pass) + 1);

                // deps[&pass].mem_barriers.push_back(vk::ImageMemoryBarrier2{
                //     written_resource.stage,                     
                //     RGStageToVk(written_resource.stage),
                //     srcaccess,
                //     dststage,
                //     dstaccess,
                //     oldlayout,
                //     newlayout,
                //     srq,
                //     dstq,
                //     img,
                //     range
                // })
            }
        }

        // for(auto &r : pass.write_resources) {
        //     auto &res = resources.at(r.first);
        //     if(last_read.contains(r.first)) {
        //         stage = std::max(stage, pass_stage.at(last_read.at(r.first)) + 1);
        //         INSERT BARRIERS HERE
        //     }
        // }

        // for(auto &r : pass.read_resources) { last_read[r.first] = &pass; }
        // for(auto &r : pass.write_resources) { last_modified[r.first] = &pass; }
        // pass_stage[&pass] = stage;
        // get_stage(stage).push_back(pass_idx);
        // ++pass_idx;
    }

    std::vector<RenderPass> flat_resources;
    flat_resources.reserve(passes.size());
    for(auto &s : stages) {
        for(auto &r : s) {
            flat_resources.push_back(std::move(passes.at(r)));
        }
    }
    passes = std::move(flat_resources);

    for(auto &p : passes) {
        std::cout << std::format("{} ", p.name);
    }
}