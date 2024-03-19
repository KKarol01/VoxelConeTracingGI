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

struct RenderPass;
struct RGTextureAccess {
    bool intersects(TextureRange r) const {
        return (
            range.base_mip < r.base_mip + r.mips &&
            range.base_mip + range.mips > r.base_mip &&
            range.base_layer < r.base_layer + r.layers &&
            range.base_layer + range.layers > r.base_layer
        );
    }

    RenderPass* pass{};
    RGImageLayout layout{RGImageLayout::Undefined};
    TextureRange range;
};

struct RGTextureAccesses {

    const RGTextureAccess* find_intersection_in_reads(TextureRange range) const {
        for(const auto& e : last_read) {
            if(e.intersects(range)) { return &e; }
        }
        return nullptr;
    }

    const RGTextureAccess* find_intersection_in_writes(TextureRange range) const {
        for(const auto& e : last_written) {
            if(e.intersects(range)) { return &e; }
        }
        return nullptr;
    }

    std::vector<RGTextureAccess> last_read, last_written;
};

struct RGResource {
    // RGResource(const std::string& name, Buffer* buffer): name(name), type(RGResourceType::Buffer) {}
    RGResource(const std::string& name, TextureStorage* texture): name(name), type(RGResourceType::Texture), texture(texture) {}

    RGResource(const RGResource& o) noexcept { *this = o; }
    RGResource& operator=(const RGResource& o) noexcept {
        assert(type == RGResourceType::Texture);
        name = o.name;
        type = o.type;
        texture = o.texture;
        texture_accesses = o.texture_accesses;
        return *this;
    }
    RGResource(RGResource&& o) noexcept { *this = std::move(o); }
    RGResource& operator=(RGResource&& o) noexcept {
        assert(type == RGResourceType::Texture);
        name = std::move(o.name);
        type = o.type;
        texture = o.texture;
        texture_accesses = std::move(o.texture_accesses);
        return *this;
    }
    ~RGResource() noexcept {}
    
    std::string name;
    RGResourceType type{RGResourceType::None};
    union {
        // Buffer* buffer;
        TextureStorage* texture;
    };

    union {
        RGTextureAccesses texture_accesses;
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
    bool is_read;
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
        info.is_read = false;
        resources.push_back(info);
        return *this;
    }

    RenderPass& read_from_image(RPResource info) {
        info.usage = RGResourceUsage::Image;
        resources.push_back(info);
        info.is_read = true;
        return *this;
    }

    RenderPass& write_color_attachment(RPResource info) {
        info.usage = RGResourceUsage::ColorAttachment;
        resources.push_back(info);
        color_attachments.push_back(resources.size() - 1);
        info.is_read = false;
        return *this;
    }

    RenderPass& write_depth_attachment(RPResource info) {
        info.usage = RGResourceUsage::DepthAttachment;
        resources.push_back(info);
        depth_attachment = resources.size() - 1;
        info.is_read = false;
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
    std::vector<RPResource> resources;
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

    void handle_image_pass_resource();
    void create_rendering_resources();
    BarrierStages deduce_stages_and_accesses(const RenderPass* src_pass, const RenderPass* dst_pass, const RPResource& src_resource, const RPResource& dst_resource, bool src_read, bool dst_read) const;

    std::vector<RGResource> resources;
    std::vector<RenderPass> passes;
    std::vector<PassDependencies> stage_deps;
    std::vector<u32> stage_pass_counts;
    std::map<std::pair<const RenderPass*, RgResourceHandle>, vk::ImageView> image_views;
};