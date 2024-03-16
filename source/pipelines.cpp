#include "pipelines.hpp"
#include "renderer.hpp"

Pipeline PipelineBuilder::build_graphics(std::string_view label) {
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    for(const auto& shader : shaders) { 
        stages.push_back(vk::PipelineShaderStageCreateInfo{{}, shader.first, shader.second, "main"}); 
    }

    vk::PipelineVertexInputStateCreateInfo   VertexInputState_   = {
        {}, bindings, attributes
    };

    vk::PipelineInputAssemblyStateCreateInfo InputAssemblyState_ = {
        {}, vk::PrimitiveTopology::eTriangleList
    };

    vk::PipelineTessellationStateCreateInfo  TessellationState_  = {};

    vk::PipelineViewportStateCreateInfo      ViewportState_      = {};

    vk::PipelineRasterizationStateCreateInfo RasterizationState_ = {
        {}, 
        false,
        false,
        vk::PolygonMode::eFill,
        cull_mode,
        front_face,
        false,
        0.0f,
        false,
        0.0f,
        1.0f
    };

    vk::PipelineMultisampleStateCreateInfo   MultisampleState_   = {};

    vk::PipelineDepthStencilStateCreateInfo  DepthStencilState_  = {
        {},
        depth_test,
        depth_write,
        depth_compare,
        false, 
        false,
        {},
        {},
        0.0f,
        1.0f,
    };

    vk::PipelineColorBlendAttachmentState    ColorBlendAtt1_     = {
        false,
        vk::BlendFactor::eOne,
        vk::BlendFactor::eZero,
        vk::BlendOp::eAdd,
        vk::BlendFactor::eOne,
        vk::BlendFactor::eZero,
        vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | 
        vk::ColorComponentFlagBits::eG | 
        vk::ColorComponentFlagBits::eB | 
        vk::ColorComponentFlagBits::eA
    };

    vk::PipelineColorBlendStateCreateInfo    ColorBlendState_    = {
        {},
        false,
        vk::LogicOp::eClear,
        ColorBlendAtt1_
    };

    vk::DynamicState                         DynamicStates[] = {
        vk::DynamicState::eScissorWithCount,
        vk::DynamicState::eViewportWithCount
    };

    vk::PipelineDynamicStateCreateInfo       DynamicState_       = {
        {}, DynamicStates            
    };

    vk::PipelineLayoutCreateInfo layout_info = {
        {},
        set_layouts,
        {}
    };

    vk::PipelineLayout layout_ = renderer->device.createPipelineLayout(layout_info);
    set_debug_name(renderer->device, layout_, std::format("{}_layout", label));

    vk::PipelineRenderingCreateInfo dynamic_rendering = {
        {}, color_attachment_formats, depth_attachment_format
    };

    // vk::PipelineRasterizationConservativeStateCreateInfoEXT conservative_rasterization = {
    //     {}, vk::ConservativeRasterizationModeEXT::eOverestimate
    // };
    // RasterizationState_.pNext = &conservative_rasterization;

    vk::GraphicsPipelineCreateInfo info{
        {},
        stages,
        &VertexInputState_,
        &InputAssemblyState_,
        &TessellationState_,
        &ViewportState_,
        &RasterizationState_,
        &MultisampleState_,
        &DepthStencilState_,
        &ColorBlendState_,
        &DynamicState_,
        layout_,
        {},
        {},
        {},
        {},
        &dynamic_rendering
    };

    auto pipeline = renderer->device.createGraphicsPipelines({}, info).value[0];
    set_debug_name(renderer->device, pipeline, label);

    return Pipeline{
        .pipeline = pipeline,
        .layout = layout_
    };
} 

Pipeline PipelineBuilder::build_compute(std::string_view label) {
    vk::PipelineLayoutCreateInfo layout_info = {
        {},
        set_layouts,
        {}
    };

    vk::PipelineLayout layout_ = renderer->device.createPipelineLayout(layout_info);
    set_debug_name(renderer->device, layout_, std::format("{}_layout", label));
    
    vk::ComputePipelineCreateInfo info{
        {},
        vk::PipelineShaderStageCreateInfo{
            {}, shaders.at(0).first, shaders.at(0).second, "main"
        },
        layout_
    };

    auto pipeline = renderer->device.createComputePipeline({}, info).value;
    set_debug_name(renderer->device, pipeline, label);

    return Pipeline{
        .pipeline = pipeline,
        .layout = layout_
    };
}