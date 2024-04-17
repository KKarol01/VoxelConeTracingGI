#pragma once
#include "renderer_types.hpp"
#include <vector>
#include <string_view>
#include <filesystem>

class Renderer;

std::vector<u32> compile_glsl_to_spv(const std::filesystem::path& path);
std::vector<ShaderResource> get_shader_resources(const std::vector<u32>& ir);
Shader build_shader_from_spv(const std::vector<u32>& ir);

class PipelineBuilder {
public:
    PipelineBuilder(const Renderer* renderer): renderer(renderer) {}

    PipelineBuilder& with_shaders(const std::vector<std::pair<vk::ShaderStageFlagBits, Shader*>>& shaders) {
        this->shaders = shaders;        
        return *this;
    }

    PipelineBuilder &with_vertex_input(
        const std::vector<vk::VertexInputBindingDescription> &bindings,
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

    PipelineBuilder& with_push_constant(size_t offset, size_t size) {
        push_constants
            .setOffset(offset)
            .setSize(size);
        return *this;
    }

    PipelineBuilder& with_descriptor_set_layouts(const std::vector<vk::DescriptorSetLayout>& layouts) {
        this->layouts = layouts;
        return *this;
    }

    Pipeline build_graphics(std::string_view label);
    Pipeline build_compute(std::string_view label);

private:
    PipelineLayout coalesce_shader_resources_into_layout();

    const Renderer* renderer;
    std::vector<std::pair<vk::ShaderStageFlagBits, Shader*>> shaders;
    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;
    std::vector<vk::DescriptorSetLayout> layouts;
    vk::CullModeFlagBits cull_mode{vk::CullModeFlagBits::eBack};
    vk::FrontFace front_face{vk::FrontFace::eCounterClockwise};
    bool depth_test{true};
    bool depth_write{true};
    vk::CompareOp depth_compare{vk::CompareOp::eLess};
    std::vector<vk::Format> color_attachment_formats;
    vk::Format depth_attachment_format{vk::Format::eUndefined};
    vk::PushConstantRange push_constants{};
};