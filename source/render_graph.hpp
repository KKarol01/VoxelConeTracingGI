#pragma once
#include "types.hpp"
#include <vector>
#include <string>
#include <optional>
#include <map>
#include <iostream>
#include <format>
#include <functional>
#include <vulkan/vulkan_enums.hpp>

struct Pipeline;
struct TextureStorage;

enum class RGResourceType {
    None, Buffer, Texture
};

enum class RGResourceUsage {
    None, Image, ColorAttachment, DepthAttachment
};

enum class RGAccessType { 
    None, Read, Write 
};

enum class RGImageAspect { 
    None, Color, Depth 
};

enum class RGAttachmentLoadStoreOp {
    DontCare, Clear, Load, Store, None
};

using RgResourceHandle = uint64_t;

enum class RGSyncStage {
    None, Transfer, Fragment, EarlyFragment, LateFragment, Compute, 
    ColorAttachmentOutput
};

struct BufferInfo {
};

enum class RGImageLayout {
    Undefined, General, Attachment, ReadOnly,
    TransferDst, TransferSrc
};

struct TextureRange {
    uint32_t base_mip{0}, mips{1}, base_layer{0}, layers{1};
};

struct TextureInfo {
    RGImageLayout required_layout;
    TextureRange range;
};

struct RGResource {
    // RGResource(const std::string& name, Buffer* buffer): name(name), type(RGResourceType::Buffer) {}
    RGResource(const std::string& name, TextureStorage* texture): name(name), type(RGResourceType::Texture), texture(texture) {}
    
    std::string name;
    RGResourceType type{RGResourceType::None};
    union {
        // Buffer* buffer;
        TextureStorage* texture;
    };
};

struct RPResource {
    RPResource(RgResourceHandle resource, RGSyncStage stage, const BufferInfo& info)
        : resource(resource), stage(stage), buffer_info(info) {}
    RPResource(RgResourceHandle resource, RGSyncStage stage, const TextureInfo& info)
        : resource(resource), stage(stage), texture_info(info) {}
    
    RgResourceHandle resource;
    RGSyncStage stage;
    RGResourceUsage usage{RGResourceUsage::None};
    RGAttachmentLoadStoreOp load_op{RGAttachmentLoadStoreOp::Clear}, store_op{RGAttachmentLoadStoreOp::Store};
    union {
        BufferInfo buffer_info;
        TextureInfo texture_info;
    };
};

struct RenderPassRenderingViewport {
    float offset_x, offset_y, width, height, min_depth, max_depth;
};

struct RenderPassRenderingScissor {
    uint32_t scissor_x{0}, scissor_y{0}, scissor_width{0}, scissor_height{0}; 
};

struct RenderPassRenderingExtent {
    RenderPassRenderingViewport viewport;
    RenderPassRenderingScissor scissor;
    uint32_t layers{1}, viewmask{0};
};

struct RenderPass {
    RenderPass& set_name(const std::string& name) {
        this->name = name;
        return *this;
    }
    
    RenderPass& set_draw_func(std::function<void()>&& f) {
        func = std::move(f);
        return *this;
    }

    RenderPass& write_to_image(RPResource info) {
        info.usage = RGResourceUsage::Image;
        write_resources.push_back(info);
        return *this;
    }

    RenderPass& read_from_image(RPResource info) {
        info.usage = RGResourceUsage::Image;
        read_resources.push_back(info);
        return *this;
    }

    RenderPass& write_color_attachment(RPResource info) {
        info.usage = RGResourceUsage::ColorAttachment;
        write_resources.push_back(info);
        color_attachments.push_back(write_resources.size() - 1);
        return *this;
    }

    RenderPass& write_depth_attachment(RPResource info) {
        info.usage = RGResourceUsage::DepthAttachment;
        write_resources.push_back(info);
        depth_attachment = write_resources.size() - 1;
        return *this;
    }

    RenderPass& set_rendering_extent(const RenderPassRenderingExtent& extent) {
        this->extent = extent;
        return *this;
    }

    RenderPass& set_pipeline(Pipeline* pipeline) {
        this->pipeline = pipeline;
        return *this;
    }
    
    std::string name;
    Pipeline* pipeline{};
    std::function<void()> func;
    RenderPassRenderingExtent extent;
    std::vector<RPResource> write_resources;
    std::vector<RPResource> read_resources;
    std::vector<u32> color_attachments; 
    std::optional<u32> depth_attachment;
};

class RenderGraph {
public:
    RgResourceHandle add_resource(const RGResource& resource) {
        resources.push_back(resource);
        return resources.size() - 1;
    }

    RenderGraph& add_render_pass(const RenderPass& pass) {
        passes.push_back(std::move(pass));
        return *this;
    }

    const RGResource& get_resource(RgResourceHandle resource) const {
        return resources.at(resource);
    }

    void clear_resources() { 
        resources.clear(); 
        passes.clear();
        stage_deps.clear();
        stage_pass_counts.clear();
        //todo: add them to deletion queue
        image_views.clear();
    }

    void bake_graph();
    void render(vk::CommandBuffer cmd, vk::Image swapchain_image, vk::ImageView swapchain_view);
    
private:
    struct PassDependencies {
        std::vector<vk::ImageMemoryBarrier2> image_barriers;
        std::optional<vk::ImageMemoryBarrier2> swapchain_image_barrier; // need special handling like vk::Image patching during rendering
    };
    struct BarrierStages {
        vk::PipelineStageFlags2 src_stage, dst_stage;
        vk::AccessFlags2 src_access, dst_access;
    };

    void create_rendering_resources();
    BarrierStages deduce_stages_and_accesses(const RenderPass* src_pass, const RenderPass* dst_pass, const RPResource& src_resource, const RPResource& dst_resource, bool src_read, bool dst_read) const;

    std::vector<RGResource> resources;
    std::vector<RenderPass> passes;
    std::vector<PassDependencies> stage_deps;
    std::vector<u32> stage_pass_counts;
    std::map<std::pair<const RenderPass*, RgResourceHandle>, vk::ImageView> image_views;
};