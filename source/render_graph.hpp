#pragma once
#include "types.hpp"
#include "renderer_types.hpp"
#include "descriptor.hpp"
#include <vector>
#include <string>
#include <optional>
#include <map>
#include <functional>
#include <variant>

#if 1
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
    ColorAttachmentOutput, AllGraphics
};

struct BufferInfo {
};

enum class RGImageLayout {
    Undefined, General, Attachment, ReadOnly,
    TransferDst, TransferSrc, PresentSrc
};

enum class RGImageFormat {
    DeduceFromVkImage, RGBA8Unorm, R32UI
};

struct TextureRange {
    bool intersects(TextureRange r) const;
    bool fully_contains(TextureRange r) const;
    TextureRange get_overlap(TextureRange r) const;
    
    u32 base_mip{0}, mips{1}, base_layer{0}, layers{1};
};

struct TextureInfo {
    RGImageLayout required_layout;
    RGImageFormat mutable_format{RGImageFormat::DeduceFromVkImage};
    TextureRange range;
};

struct RenderPass;
struct RGTextureAccess {
    RGTextureAccess(RenderPass* pass, RGImageLayout layout, TextureRange range)
        : pass(pass), layout(layout), range(range) {}
    
    RenderPass* pass{};
    RGImageLayout layout{RGImageLayout::Undefined};
    TextureRange range;
};

/*
    Represents a single access to a texture
*/
struct RGLayoutAccess {
    u32 access_idx;
    bool is_read;
};

/*
    Represents a range of texture MIPs and LAYERs, but with
    the ability to subtract other ranges from it.
    It's purpose is to accumulate-by-subtracting subranges in which a texture
    has not been accessed.
*/
struct RGLayoutRanges {
    bool empty() const { return ranges.empty(); }
    void subtract(TextureRange range);
    void handle_subdivision(TextureRange range, TextureRange r, std::vector<TextureRange>& new_ranges);
    void handle_mips(TextureRange overlap, TextureRange r, std::vector<TextureRange>& new_ranges);
    
    std::vector<TextureRange> ranges;
};

struct RGLayoutChangeQuery {
    std::vector<std::pair<RGLayoutAccess, TextureRange>> accesses;
    RGLayoutRanges previously_unaccessed;
};

/*
    Represents the history of accesses to a texture resource.
    Latest ones are at the end of the vectors.
*/
class RGTextureAccesses {
public:
    void insert_read(RGTextureAccess access);
    void insert_write(RGTextureAccess access);
    RGTextureAccess& get_layout_texture_access(RGLayoutAccess access);
    const RGTextureAccess& get_layout_texture_access(RGLayoutAccess access) const;
    RGLayoutChangeQuery query_layout_changes(TextureRange range, bool is_read) const;

private:
    std::vector<RGTextureAccess> last_read, last_written;
    std::vector<RGLayoutAccess> layouts;
};

struct RGResource {
    RGResource(const std::string& name, const Texture& texture): name(name), resource(std::make_pair(texture, RGTextureAccesses{})) {}

    RGResource(const RGResource& o) noexcept = delete;
    RGResource& operator=(const RGResource& o) noexcept = delete;
    RGResource(RGResource&& o) noexcept { *this = std::move(o); }
    RGResource& operator=(RGResource&& o) noexcept {
        name = o.name;
        resource = std::move(o.resource);
        return *this;
    }
    
    std::string name;
    std::variant<Buffer, std::pair<Texture, RGTextureAccesses>> resource;
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
    f32 offset_x, offset_y, width, height, min_depth, max_depth;
};

struct RenderPassRenderingScissor {
    u32 scissor_x{0}, scissor_y{0}, scissor_width{0}, scissor_height{0}; 
};

struct RenderPassRenderingExtent {
    RenderPassRenderingViewport viewport;
    RenderPassRenderingScissor scissor;
    u32 layers{1}, viewmask{0};
};

struct RenderPass {
    RenderPass& set_name(const std::string& name);
    RenderPass& set_draw_func(std::function<void(vk::CommandBuffer)>&& f);
    RenderPass& write_to_image(RPResource info);
    RenderPass& read_from_image(RPResource info);
    RenderPass& write_color_attachment(RPResource info);
    RenderPass& read_color_attachment(RPResource info);
    RenderPass& write_depth_attachment(RPResource info);
    RenderPass& set_rendering_extent(const RenderPassRenderingExtent& extent);
    RenderPass& set_pipeline(Pipeline* pipeline);
    RenderPass& set_make_sampler(bool make);
    const RPResource* get_resource(RgResourceHandle handle) const;

    std::string name;
    Pipeline* pipeline{};
    Handle<DescriptorSetAllocation> descriptor;
    std::function<void(vk::CommandBuffer)> func;
    RenderPassRenderingExtent extent;
    std::vector<RPResource> resources;
    std::vector<u32> color_attachments; 
    std::optional<u32> depth_attachment;
    bool make_sampler{true};
};

class RenderGraph {
public:
    RenderGraph() = default;
    explicit RenderGraph(DescriptorSet* descriptor_buffer): descriptor_buffer(descriptor_buffer) {}

    RgResourceHandle add_resource(RGResource resource) {
        resources.push_back(std::move(resource));
        return resources.size() - 1;
    }

    RenderGraph& add_render_pass(const RenderPass& pass) {
        renderpasses.push_back(std::move(pass));
        return *this;
    }

    const RGResource& get_resource(RgResourceHandle resource) const {
        return resources.at(resource);
    }

    void clear_resources();
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
    struct PassBarriers {
        PassDependencies deps;
        std::vector<RenderPass*> passes;
    };

    void handle_image_pass_resource();
    void create_rendering_resources();
    BarrierStages deduce_stages_and_accesses(const RenderPass* src_pass, const RenderPass* dst_pass, const RPResource& src_resource, const RPResource& dst_resource, bool src_read, bool dst_read) const;

    DescriptorSet* descriptor_buffer;
    std::vector<RGResource> resources;
    std::vector<RenderPass> renderpasses;
    std::vector<PassDependencies> stage_deps;
    std::vector<u32> stage_deps_counts;
    std::map<std::pair<const RenderPass*, RgResourceHandle>, vk::ImageView> image_views;
};
#endif