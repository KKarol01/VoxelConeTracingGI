#include "pipelines.hpp"
#include "renderer.hpp"
#include <spdlog/spdlog.h>
#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_cross.hpp>
#include <stb/stb_include.h>

static char* read_shader_file(const std::filesystem::path& path) {
    static std::filesystem::path shader_path = "data/shaders";
    char error[256] = {};
    auto full_path = shader_path / path;
    auto path_str = full_path.string();
    auto parent_path_str = shader_path.string();
    auto file = stb_include_file((char*) path_str.c_str(), 0, (char*)parent_path_str.c_str(), error);

    if(error[0] != 0) {
        spdlog::error("stb_include: Error {}", error);
    }

    return file;
}

std::vector<u32> compile_glsl_to_spv(const std::filesystem::path& path) {
    auto file = read_shader_file(path);
    shaderc::Compiler compiler;
    auto result = compiler.CompileGlslToSpv(file, shaderc_glsl_infer_from_source, path.filename().string().c_str());
    if(result.GetCompilationStatus() != shaderc_compilation_status_success) {
        spdlog::error("Shader {} compilation error: {}", path.filename().string(), result.GetErrorMessage().c_str());
        return {};
    }
    free(file);

    return std::vector<u32>{result.begin(), result.end()};
}

std::vector<ShaderResource> get_shader_resources(const std::vector<u32>& ir) {
    spirv_cross::Compiler compiler{ir}; 
    const auto& resources = compiler.get_shader_resources();

    const spirv_cross::SmallVector<spirv_cross::Resource>* resources_list[] {
        &resources.separate_images,
        &resources.storage_images,
        &resources.separate_samplers,
        &resources.uniform_buffers,
        &resources.storage_buffers,
    };
    DescriptorType resource_types[] {
        DescriptorType::SampledImage,
        DescriptorType::StorageImage,
        DescriptorType::Sampler,
        DescriptorType::UniformBuffer,
        DescriptorType::StorageBuffer
    };

    std::vector<ShaderResource> shader_resources;
    for(u32 i=0; i<sizeof(resources_list)/sizeof(resources_list[0]); ++i) {
        auto* res = resources_list[i];
        auto res_type = resource_types[i];

        for(const auto& r : *res) {
            // const auto& type = compiler.get_type(r.base_type_id); // for later use
            const auto set = compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet);
            const auto binding = compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
            shader_resources.push_back(ShaderResource{
                .descriptor_set = set,
                .resource = {
                    .name = r.name,
                    .type = res_type,
                    .binding = binding
                }
            });
        }
    }

    std::sort(begin(shader_resources), end(shader_resources), [](auto&& a, auto&& b) {
        if(a.descriptor_set != b.descriptor_set) { return a.descriptor_set < b.descriptor_set; }
        return a.resource.binding < b.resource.binding;
    });

    return shader_resources;
}

Pipeline PipelineBuilder::build_graphics(std::string_view label) {
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    for(const auto& shader : shaders) { 
        stages.push_back(vk::PipelineShaderStageCreateInfo{{}, shader.first, shader.second->module, "main"}); 
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

    PipelineLayout layout = coalesce_shader_resources_into_layout();
    set_debug_name(renderer->device, layout.layout, std::format("{}_layout", label));

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
        layout.layout,
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
        .layout = layout,
        .type = PipelineType::Graphics
    };
} 

Pipeline PipelineBuilder::build_compute(std::string_view label) {
    PipelineLayout layout = coalesce_shader_resources_into_layout();
    set_debug_name(renderer->device, layout.layout, std::format("{}_layout", label));
    
    vk::ComputePipelineCreateInfo info{
        {},
        vk::PipelineShaderStageCreateInfo{
            {}, shaders.at(0).first, shaders.at(0).second->module, "main"
        },
        layout.layout
    };

    auto pipeline = renderer->device.createComputePipeline({}, info).value;
    set_debug_name(renderer->device, pipeline, label);

    return Pipeline{
        .pipeline = pipeline,
        .layout = layout,
        .type = PipelineType::Compute
    };
}

static vk::DescriptorType to_vk_desc_type(DescriptorType type) {
    switch (type) {
        using enum DescriptorType;
        case SampledImage:   { return vk::DescriptorType::eSampledImage; }
        case StorageImage:   { return vk::DescriptorType::eStorageImage; }
        case Sampler:        { return vk::DescriptorType::eSampler; }
        case UniformBuffer:  { return vk::DescriptorType::eUniformBuffer; }
        case StorageBuffer:  { return vk::DescriptorType::eStorageBuffer; }
        default: {
            spdlog::error("Unrecognized descriptor type: {}", (u32)type);
            std::abort(); 
            return vk::DescriptorType::eSampler;
        }
    }
}

PipelineLayout PipelineBuilder::coalesce_shader_resources_into_layout() {
    PipelineLayout layout;
    std::vector<std::vector<vk::DescriptorSetLayoutBinding>> layout_bindings(PipelineLayout::MAX_DESCRIPTOR_SET_COUNT);
    static constexpr vk::ShaderStageFlags ALL_STAGE_FLAGS = 
        vk::ShaderStageFlagBits::eFragment | 
        vk::ShaderStageFlagBits::eVertex |
        vk::ShaderStageFlagBits::eCompute | 
        vk::ShaderStageFlagBits::eGeometry;

    const auto get_set_bindings = [&layout](u32 set) -> auto& {
        return layout.descriptor_sets.at(set).bindings;
    };

    const auto find_binding_in_set = [&](u32 set, u32 binding) -> auto {
        auto& bindings = get_set_bindings(set);
        return std::find_if(begin(bindings), end(bindings), [binding](auto&& a) {
            return a.binding == binding;
        });
    };
        
    for(const auto& sh : shaders) {
        for(const auto &r : sh.second->resources) {
            if(find_binding_in_set(r.descriptor_set, r.resource.binding) != end(get_set_bindings(r.descriptor_set))) {
                spdlog::info("Descriptor set={} already contains binding={}. Skipping", r.descriptor_set, r.resource.binding); 
                continue;
            }
            layout.descriptor_sets.at(r.descriptor_set).bindings.push_back(r.resource);
            layout_bindings.at(r.descriptor_set).push_back(vk::DescriptorSetLayoutBinding{
                r.resource.binding,
                to_vk_desc_type(r.resource.type),
                1,
                ALL_STAGE_FLAGS
            });
        }
    }

    std::vector<vk::DescriptorSetLayout> set_layouts(PipelineLayout::MAX_DESCRIPTOR_SET_COUNT);
    for(u32 i=0; auto& lb : layout_bindings) {
        layout.descriptor_sets.at(i).layout = get_context().renderer->device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            {},
            lb
        });
        set_layouts.at(i) = layout.descriptor_sets.at(i).layout;
        ++i;
    }

    layout.layout = get_context().renderer->device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        {},
        set_layouts
    });

    return layout;
}