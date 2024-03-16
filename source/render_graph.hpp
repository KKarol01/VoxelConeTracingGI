#pragma once
#include "types.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <format>
#include <functional>
#include <vulkan/vulkan.hpp>

struct TextureStorage;

enum class RGResourceType {
    None, Buffer, Texture
};

using RgResourceHandle = uint64_t;

enum class RGSyncStage {
    None, Transfer, Fragment, EarlyFragment, LateFragment, Compute
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
    RGResource(const std::string& name, TextureStorage* texture): name(name), type(RGResourceType::Texture) {}
    
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

    RenderPass& write_to_color_image(const RPResource& info) {
        write_resources.push_back(info);
        return *this;
    }

    RenderPass& add_read_resource(const RPResource& info) {
        read_resources.push_back(info);
        return *this;
    }

    RenderPass& set_rendering_extent(const RenderPassRenderingExtent& extent) {
        this->extent = extent;
        return *this;
    }
    
    std::string name;
    std::function<void()> func;
    RenderPassRenderingExtent extent;
    std::vector<RPResource> write_resources;
    std::vector<RPResource> read_resources;
};

class RenderGraph {
public:
    RgResourceHandle add_resource(const RGResource& resource) {
        resources.push_back(resource);
        return resources.size() - 1;
    }

    RenderGraph& add_render_pass(RenderPass&& pass) {
        passes.push_back(std::move(pass));
        return *this;
    }

    void bake_graph();
    
    std::vector<RGResource> resources;
    std::vector<vk::DependencyInfo> pass_deps;
    std::vector<RenderPass> passes;
};