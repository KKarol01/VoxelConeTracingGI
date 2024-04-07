#pragma once
#include "types.hpp"
#include <vector>
#include <string>
#include <optional>
#include <map>
#include <functional>

struct DescriptorSet;
struct Pipeline;
struct Texture;

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
    bool intersects(TextureRange r) const {
        return (
            base_layer < r.base_layer + r.layers &&
            base_layer + layers > r.base_layer &&
            base_mip < r.base_mip + r.mips &&
            base_mip + mips > r.base_mip
        );
    }

    bool fully_contains(TextureRange r) const {
        return (
            base_mip <= r.base_mip &&
            base_mip + mips >= r.base_mip + r.mips &&
            base_layer <= r.base_layer &&
            base_layer + layers >= r.base_layer + r.layers
        );
    }
    
    TextureRange get_overlap(TextureRange r) const {
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
    It's purpose is to provide every range in which a texture
    has not been accessed.
*/
struct RGLayoutRanges {
    bool empty() const { return ranges.empty(); }

    void subtract(TextureRange range) {
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



    void handle_subdivision(TextureRange range, TextureRange r, std::vector<TextureRange>& new_ranges) {
        /*
            During range subtraction, there are multiple cases. 
            Let [] denote a range which the intersection with another range, (), 
            will be subtracted from.

            Each range is a continous range of layers, that each have equal number of mips.
            Case 1: ( [ ) ] - range () overlaps with only the left part of the range [] -> divide into ()[]
            Case 2: [ ( ] ) - range () ovelaps with only the right part of the range [] -> divide into []()
            Case 3: [ ( ) ] - range () is contained within range [] -> divide into ()[]()
            Case 4: [( )] -> range () spans exactly the same range of layers as the range [] -> divide by mip regions.

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

    void handle_mips(TextureRange overlap, TextureRange r, std::vector<TextureRange>& new_ranges) {
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
    void insert_read(RGTextureAccess access) {
        layouts.emplace_back(last_read.size(), true);
        last_read.push_back(access);
    }

    void insert_write(RGTextureAccess access) {
        layouts.emplace_back(last_written.size(), false);
        last_written.push_back(access);
    }

    RGTextureAccess& get_layout_texture_access(RGLayoutAccess access) {
        auto* vec = &last_read;
        if(!access.is_read) { vec = &last_written; }
        return vec->at(access.access_idx);
    }
    const RGTextureAccess& get_layout_texture_access(RGLayoutAccess access) const {
        auto* vec = &last_read;
        if(!access.is_read) { vec = &last_written; }
        return vec->at(access.access_idx);
    }

    RGLayoutChangeQuery query_layout_changes(TextureRange range, bool is_read) const {
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

private:
    std::vector<RGTextureAccess> last_read, last_written;
    std::vector<RGLayoutAccess> layouts;
};

struct RGResource {
    RGResource(const std::string& name, Texture* texture): name(name), type(RGResourceType::Texture), texture(texture), texture_accesses() {}

    RGResource(const RGResource& o) noexcept { *this = o; }
    RGResource& operator=(const RGResource& o) noexcept {
        assert(o.type == RGResourceType::Texture);
        name = o.name;
        type = o.type;
        texture = o.texture;
        new(&texture_accesses) RGTextureAccesses{};
        texture_accesses = o.texture_accesses;
        return *this;
    }
    RGResource(RGResource&& o) noexcept { *this = std::move(o); }
    RGResource& operator=(RGResource&& o) noexcept {
        assert(o.type == RGResourceType::Texture);
        name = std::move(o.name);
        type = o.type;
        texture = o.texture;
        new(&texture_accesses) RGTextureAccesses{};
        texture_accesses = std::move(o.texture_accesses);
        return *this;
    }
    ~RGResource() noexcept {
        switch (type) {
            case RGResourceType::Texture: { texture_accesses.~RGTextureAccesses(); } break;
            default: {
                spdlog::error("Unhandled destructor in RGResource");
                std::abort();
            }
        }
    }
    
    std::string name;
    RGResourceType type{RGResourceType::None};
    union {
        // Buffer* buffer;
        Texture* texture;
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
    
    RenderPass& set_draw_func(std::function<void(vk::CommandBuffer)>&& f) {
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
        info.is_read = true;
        resources.push_back(info);
        return *this;
    }

    RenderPass& write_color_attachment(RPResource info) {
        info.usage = RGResourceUsage::ColorAttachment;
        info.is_read = false;
        resources.push_back(info);
        color_attachments.push_back(resources.size() - 1);
        return *this;
    }

    RenderPass& read_color_attachment(RPResource info) {
        info.usage = RGResourceUsage::ColorAttachment;
        info.is_read = true;
        resources.push_back(info);
        color_attachments.push_back(resources.size() - 1);
        return *this;
    }

    RenderPass& write_depth_attachment(RPResource info) {
        info.usage = RGResourceUsage::DepthAttachment;
        info.is_read = false;
        resources.push_back(info);
        depth_attachment = resources.size() - 1;
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

    RenderPass& set_make_sampler(bool make) {
        make_sampler = make;
        return *this;
    }

    const RPResource* get_resource(RgResourceHandle handle) const {
        for(auto& r : resources) {
            if(r.resource == handle) { return &r; }
        }
        return nullptr;
    }

    
    std::string name;
    Pipeline* pipeline{};
    DescriptorSet* set;
    std::function<void(vk::CommandBuffer)> func;
    RenderPassRenderingExtent extent;
    std::vector<RPResource> resources;
    std::vector<u32> color_attachments; 
    std::optional<u32> depth_attachment;
    bool make_sampler{true};
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