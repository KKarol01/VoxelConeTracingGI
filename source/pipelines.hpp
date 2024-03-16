#pragma once
#include "renderer_types.hpp"
#include <vector>
#include <string_view>

class Renderer;

class PipelineBuilder {
public:
    PipelineBuilder(const Renderer* renderer): renderer(renderer) {}

    PipelineBuilder& with_shaders(const std::vector<std::pair<vk::ShaderStageFlagBits, vk::ShaderModule>>& shaders) {
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

    PipelineBuilder& with_layout(vk::DescriptorSetLayout set) {
        set_layouts.push_back(set);
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

    Pipeline build_graphics(std::string_view label);

    Pipeline build_compute(std::string_view label);

private:
    const Renderer* renderer;
    std::vector<std::pair<vk::ShaderStageFlagBits, vk::ShaderModule>> shaders;
    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;
    vk::CullModeFlagBits cull_mode{vk::CullModeFlagBits::eBack};
    vk::FrontFace front_face{vk::FrontFace::eCounterClockwise};
    bool depth_test{true};
    bool depth_write{true};
    vk::CompareOp depth_compare{vk::CompareOp::eLess};
    std::vector<vk::DescriptorSetLayout> set_layouts;
    std::vector<vk::Format> color_attachment_formats;
    vk::Format depth_attachment_format{vk::Format::eUndefined};
};