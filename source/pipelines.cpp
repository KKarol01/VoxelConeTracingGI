#include "pipelines.hpp"
#include "renderer.hpp"
#include <spdlog/spdlog.h>
#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_cross.hpp>
#include <stb/stb_include.h>

static char* read_shader_file(const std::filesystem::path& path);
static vk::ShaderStageFlagBits deduce_shader_type(const std::filesystem::path& path);
static shaderc_shader_kind to_shaderc_type(vk::ShaderStageFlagBits stage);
static std::vector<u32> compile_glsl_to_spv(const std::filesystem::path& path);
static ShaderResources get_shader_resources(const std::vector<u32>& ir, vk::ShaderStageFlagBits type);

Shader::Shader(vk::Device device, const std::filesystem::path& path) {
    const auto code = compile_glsl_to_spv(path);

    module = device.createShaderModule(vk::ShaderModuleCreateInfo{
        {}, code.size() * sizeof(code[0]), code.data()
    });

    set_debug_name(module, path.string());

    resources = get_shader_resources(code, deduce_shader_type(path));
}

void build_layout(std::string_view label, vk::Device device, DescriptorLayout& layout) {
    static constexpr vk::ShaderStageFlags ALL_STAGE_FLAGS = 
        vk::ShaderStageFlagBits::eFragment | 
        vk::ShaderStageFlagBits::eVertex |
        vk::ShaderStageFlagBits::eCompute | 
        vk::ShaderStageFlagBits::eGeometry;

    std::array<vk::DescriptorSetLayoutBinding, DescriptorLayout::MAX_BINDINGS> bindings{};
    std::array<vk::DescriptorBindingFlags, DescriptorLayout::MAX_BINDINGS> flags{};

    for(u32 b = 0; b < layout.count; ++b) {
        bindings.at(b) = {b, layout.bindings.at(b).type, layout.bindings.at(b).count, ALL_STAGE_FLAGS};
        flags.at(b) = vk::DescriptorBindingFlagBits::eUpdateAfterBind | vk::DescriptorBindingFlagBits::ePartiallyBound;
        if(layout.variable_sized && b + 1u == layout.count) { flags.at(b) |= vk::DescriptorBindingFlagBits::eVariableDescriptorCount; }
    }

    vk::DescriptorSetLayoutBindingFlagsCreateInfo flag_info{layout.count, flags.data()};

    layout.layout = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
        vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        layout.count,
        bindings.data(),
        &flag_info
    });

    if(!label.empty()) { set_debug_name(layout.layout, label); }
}

void build_layout(std::string_view label, vk::Device device, PipelineLayout& layout) {
    std::array<vk::DescriptorSetLayout, PipelineLayout::MAX_SETS> sets{};
    for(u32 set = 0; set < layout.sets.size(); ++set) {
        build_layout(std::format("{}_set_{}", label, set), device, layout.sets.at(set));
        sets.at(set) = layout.sets.at(set).layout;
    }
    
    layout.layout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        {},
        sets.size(),
        sets.data(),
        layout.range.size > 0u ? 1u : 0u,
        &layout.range
    });

    if(!label.empty()) { set_debug_name(layout.layout, label); }
}

Pipeline PipelineBuilder::build_graphics(std::string_view label) {
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    std::vector<Shader> compiled_shaders;
    for(const auto& sh_path : shaders) { 
        auto& shader = compiled_shaders.emplace_back(renderer->device, sh_path);
        stages.push_back(vk::PipelineShaderStageCreateInfo{{}, deduce_shader_type(sh_path), shader.module, "main"}); 
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

    PipelineLayout layout = coalesce_shader_resources_into_layout(compiled_shaders);
    build_layout(std::format("{}_layout", label), renderer->device, layout);

    vk::PipelineRenderingCreateInfo dynamic_rendering = {
        {}, color_attachment_formats, depth_attachment_format
    };

    vk::PipelineRasterizationConservativeStateCreateInfoEXT conservative_rasterization = {
        {}, vk::ConservativeRasterizationModeEXT::eOverestimate
    };
    RasterizationState_.pNext = &conservative_rasterization;

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
    set_debug_name(pipeline, label);

    return Pipeline{
        .type = vk::PipelineBindPoint::eGraphics,
        .pipeline = pipeline,
        .layout = layout,
    };
} 

Pipeline PipelineBuilder::build_compute(std::string_view label) {
    Shader compute_shader{renderer->device, shaders.at(0)};

    PipelineLayout layout = coalesce_shader_resources_into_layout({compute_shader});
    build_layout(std::format("{}_layout", label), renderer->device, layout);

    const auto type = deduce_shader_type(shaders.at(0));
    assert("Shader type must be compute" && type == vk::ShaderStageFlagBits::eCompute);

    vk::ComputePipelineCreateInfo info{
        {},
        vk::PipelineShaderStageCreateInfo{
            {}, type, compute_shader.module, "main"
        },
        layout.layout
    };

    auto pipeline = renderer->device.createComputePipeline({}, info).value;
    set_debug_name(pipeline, label);

    return Pipeline{
        .type = vk::PipelineBindPoint::eCompute,
        .pipeline = pipeline,
        .layout = layout,
    };
}

PipelineLayout PipelineBuilder::coalesce_shader_resources_into_layout(const std::vector<Shader>& shaders) {
    PipelineLayout pipeline_layout{};

    // Merge shaders's descriptor layouts
    for(const auto& s : shaders) {
        u32 variable_binding_counter = 0;
        for(u32 set = 0; set<s.resources.bindings.size(); ++set) {
            const auto& rs = s.resources.bindings.at(set);
            auto& layout = pipeline_layout.sets.at(set);
            layout.names.resize(rs.size());

            for(u32 binding = 0; binding < rs.size(); ++binding) {
                const auto& r = rs.at(binding);
                layout.bindings.at(binding) = r.binding;
                layout.names.at(binding) = r.name;

                if(binding == rs.size() - 1u && r.binding.count == 0) {
                    layout.variable_sized = true;
                    layout.bindings.at(binding).count = variable_limits.at(variable_binding_counter);
                    variable_binding_counter++;
                }
            } 

            layout.count = std::max(layout.count, (u32)rs.size());
        }

        pipeline_layout.range.stageFlags |= s.resources.range.stageFlags;
        pipeline_layout.range.offset = std::max(pipeline_layout.range.offset, s.resources.range.offset);
        pipeline_layout.range.size = std::max(pipeline_layout.range.size, s.resources.range.size);
    }

    return pipeline_layout;
}

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

static vk::ShaderStageFlagBits deduce_shader_type(const std::filesystem::path& path) {
    const auto ext = path.extension();
    if(ext.string() == ".vert") { return vk::ShaderStageFlagBits::eVertex; }
    if(ext.string() == ".geom") { return vk::ShaderStageFlagBits::eGeometry; }
    if(ext.string() == ".frag") { return vk::ShaderStageFlagBits::eFragment; }
    if(ext.string() == ".comp") { return vk::ShaderStageFlagBits::eCompute; }
    spdlog::error("Unrecognized shader extension: {}", ext.string());
    std::abort();
    return vk::ShaderStageFlagBits::eAll;
}

static shaderc_shader_kind to_shaderc_type(vk::ShaderStageFlagBits stage) {
    switch (stage) {
        case vk::ShaderStageFlagBits::eVertex: { return shaderc_vertex_shader; }
        case vk::ShaderStageFlagBits::eGeometry: { return shaderc_geometry_shader; }
        case vk::ShaderStageFlagBits::eFragment: { return shaderc_fragment_shader; }
        case vk::ShaderStageFlagBits::eCompute: { return shaderc_compute_shader; }
        default: { assert(std::format("Unrecognized shader type: {}", vk::to_string(stage)).c_str() && false); return {}; }
    }
}

static std::vector<u32> compile_glsl_to_spv(const std::filesystem::path& path) {
    auto file = read_shader_file(path);
    shaderc::Compiler compiler;
    auto result = compiler.CompileGlslToSpv(file, to_shaderc_type(deduce_shader_type(path)), path.filename().string().c_str());
    if(result.GetCompilationStatus() != shaderc_compilation_status_success) {
        spdlog::error("Shader {} compilation error: {}", path.filename().string(), result.GetErrorMessage().c_str());
        return {};
    }
    free(file);

    return std::vector<u32>{result.begin(), result.end()};
}

static ShaderResources get_shader_resources(const std::vector<u32>& ir, vk::ShaderStageFlagBits type) {
    spirv_cross::Compiler compiler{ir}; 
    const auto& resources = compiler.get_shader_resources();

    const spirv_cross::SmallVector<spirv_cross::Resource>* resources_list[] {
        &resources.separate_images,
        &resources.storage_images,
        &resources.separate_samplers,
        &resources.uniform_buffers,
        &resources.storage_buffers,
        &resources.sampled_images,
    };

    vk::DescriptorType resource_types[] {
        vk::DescriptorType::eSampledImage,
        vk::DescriptorType::eStorageImage,
        vk::DescriptorType::eSampler,
        vk::DescriptorType::eUniformBuffer,
        vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eCombinedImageSampler,
    };

    ShaderResources shader_resources;

    if(!resources.push_constant_buffers.empty()) {
        shader_resources.range = vk::PushConstantRange{type, 0, 0};
        for(const auto range : compiler.get_active_buffer_ranges(resources.push_constant_buffers.front().id)) {
            shader_resources.range.size += range.range;
        }
    }

    for(u32 i=0; i<sizeof(resources_list)/sizeof(resources_list[0]); ++i) {
        auto* rs = resources_list[i];
        auto rt = resource_types[i];

        for(const auto& r : *rs) {
            const auto& type = compiler.get_type(r.base_type_id);
            const auto set = compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet);
            const auto binding = compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
            const auto is_struct = type.basetype == spirv_cross::SPIRType::Struct;
            const auto is_runtime_sized = [is_struct, &r, &compiler, &type, rt] {
                if(is_struct && type.member_types.empty()) { return false; }
                if(rt == vk::DescriptorType::eStorageBuffer) { return false; }

                const auto& last_type = is_struct ? compiler.get_type(type.member_types.back()) : compiler.get_type(r.type_id);
                
                if(last_type.array.empty()) { return false; }
                if(last_type.array.size() > 1u) { return false; }
                if(last_type.array_size_literal[0] == false) {
                    spdlog::error("SPIRV_CROSS array size literal at index 0 is false.");
                    std::terminate();
                }
                return last_type.array[0] == 0u;
            }();
            
            const auto count = [&r, &compiler] {
                const auto type = compiler.get_type(r.type_id);
                u32 count = 1;
                for(const auto& e : type.array) { count *= e; } // if runtime-sized, count will be zero
                return count;
            }();

            shader_resources.bindings.at(set).push_back({
                .name = r.name,
                .index = binding,
                .binding = {
                    .count = count,
                    .type = rt
                }
            });
        }
    }

    for(auto& e : shader_resources.bindings) {
        std::sort(begin(e), end(e), [](auto&& a, auto&& b) { return a.index < b.index; });
    }

    return shader_resources;
}