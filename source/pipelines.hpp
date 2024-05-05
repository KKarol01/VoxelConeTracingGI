#pragma once

#include "renderer_types.hpp"
#include <vector>
#include <string_view>
#include <filesystem>

class Renderer;

struct DescriptorBinding {
    u32 count;
    vk::DescriptorType type;
};

struct DescriptorLayout {
    static inline constexpr u32 MAX_BINDINGS = 32;

    vk::DescriptorSetLayout layout;
    std::array<DescriptorBinding, MAX_BINDINGS> bindings;
    std::vector<std::string> names;
    u32 count : 31{};
    u32 variable_sized : 1{};
};

struct PipelineLayout {
    inline static constexpr u32 MAX_SETS = 4u;

    DescriptorBinding* find_binding(std::string_view name) {
        for(auto& e : sets) { 
            u32 i = 0;
            for(u32 i=0; i<e.names.size(); ++i) { 
                if(e.names.at(i) == name) { return &e.bindings.at(i); }
            }
        }
        return nullptr;
    }

    vk::PipelineLayout layout;
    std::array<DescriptorLayout, MAX_SETS> sets{};
    vk::PushConstantRange range{};
};

struct ShaderResources {
    struct NamedBinding {
        std::string name;
        u32 index;
        DescriptorBinding binding;
    };

    std::array<std::vector<NamedBinding>, PipelineLayout::MAX_SETS> bindings{};
    vk::PushConstantRange range;
};

struct Shader {
    Shader(vk::Device device, const std::filesystem::path& path);
    
    vk::ShaderModule module;
    ShaderResources resources;
};

struct Pipeline {
    vk::PipelineBindPoint type;
    vk::Pipeline pipeline;
    PipelineLayout layout;
};

void build_layout(std::string_view label, vk::Device device, DescriptorLayout& layout);
void build_layout(std::string_view label, vk::Device device, PipelineLayout& layout);

class PipelineBuilder {
public:
    PipelineBuilder(const Renderer* renderer): renderer(renderer) {}

    PipelineBuilder& with_shaders(const std::vector<std::filesystem::path>& shaders) {
        this->shaders = shaders;        
        return *this;
    }

    PipelineBuilder &with_vertex_input(const std::vector<vk::VertexInputBindingDescription> &bindings,
                                       const std::vector<vk::VertexInputAttributeDescription> &attributes) {
        this->bindings = bindings;
        this->attributes = attributes;
        return *this;
    }

    PipelineBuilder& with_culling(vk::CullModeFlagBits culling, vk::FrontFace front_face) {
        cull_mode = culling;
        this->front_face = front_face;
        return *this;
    } 

    PipelineBuilder& with_depth_testing(bool depth_test, bool depth_write, vk::CompareOp depth_compare) {
        this->depth_test = depth_test;
        this->depth_write = depth_write;
        this->depth_compare = depth_compare;
        return *this;
    }

    PipelineBuilder& with_color_attachments(const std::vector<vk::Format>& formats) {
        color_attachment_formats = formats;
        return *this;
    }

    PipelineBuilder& with_depth_attachment(vk::Format format) {
        depth_attachment_format = format;
        return *this;
    }

    PipelineBuilder& with_variable_upper_limits(std::array<u32, PipelineLayout::MAX_SETS> limits) {
        variable_limits = limits;
        return *this;
    }

    Pipeline build_graphics(std::string_view label);
    Pipeline build_compute(std::string_view label);

private:
    PipelineLayout coalesce_shader_resources_into_layout(const std::vector<Shader>& shaders);

    const Renderer* renderer;
    std::vector<std::filesystem::path> shaders;
    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;
    std::array<u32, PipelineLayout::MAX_SETS> variable_limits;
    vk::CullModeFlagBits cull_mode{vk::CullModeFlagBits::eBack};
    vk::FrontFace front_face{vk::FrontFace::eCounterClockwise};
    bool depth_test{true};
    bool depth_write{true};
    vk::CompareOp depth_compare{vk::CompareOp::eLess};
    std::vector<vk::Format> color_attachment_formats;
    vk::Format depth_attachment_format{vk::Format::eUndefined};
};