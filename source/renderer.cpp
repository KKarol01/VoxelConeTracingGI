#include "renderer.hpp"

#include <spdlog/spdlog.h>
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <VkBootstrap.h>
#include <shaderc/shaderc.hpp>
#include <stb/stb_include.h>
#include <format>
#include <stack>

DescriptorSet::DescriptorSet(vk::Device device, vk::DescriptorPool pool, vk::DescriptorSetLayout layout) {
    set = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{pool, layout})[0];
}

void DescriptorSet::update_bindings(vk::Device device, u32 dst_binding, u32 dst_arr_element, std::span<DescriptorInfo> infos) {
    std::vector<vk::WriteDescriptorSet> writes(infos.size());
    for(u32 i=0; i<infos.size(); ++i) {
        auto &write = writes.at(i);
        write.dstSet = set;
        write.dstBinding = dst_binding + i;
        write.dstArrayElement = dst_arr_element;

        const auto &info = infos.data()[i];
        write.descriptorCount = 1;
        write.descriptorType = info.type;
        switch(info.resource) {
            case DescriptorInfo::None: {
                write.descriptorCount = 0;
                break;
            }
            case DescriptorInfo::Buffer: {
                write.pBufferInfo = &info.buffer_info;
                break;
            }
            case DescriptorInfo::Image:
            case DescriptorInfo::Sampler: {
                write.pImageInfo = &info.image_info;
                break;
            }
        } 
    }

    device.updateDescriptorSets(writes, {});
}

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
        nullptr
    };

    auto pipeline = renderer->device.createGraphicsPipelines({}, info).value[0];
    set_debug_name(renderer->device, pipeline, label);

    return Pipeline{
        .pipeline = pipeline,
        .layout = layout_
    };
} 

bool Renderer::initialize() {
    if(!glfwInit()) {
        spdlog::error("GLFW: unable to initialize");
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(window_width, window_height, "vxgi", nullptr, nullptr);
    if(!window) {
        spdlog::error("GLFW: unable to create window");
        glfwTerminate();
        return false;
    }

    if(!initialize_vulkan()) {
        return false;
    }

    if(!initialize_swapchain()) {
        return false;
    }

    if(!initialize_frame_resources()) {
        return false;
    }

    if(!initialize_render_passes()) {
        return false;
    }

    return true;
}

bool Renderer::load_model_from_file(std::string_view name, const std::filesystem::path& path) {
    fastgltf::Parser parser;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(path);

    static constexpr auto options = 
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::LoadExternalBuffers |
        fastgltf::Options::LoadExternalImages;
    auto gltf = parser.loadGltf(&data, path.parent_path(), options);
    if(auto error = gltf.error(); error != fastgltf::Error::None) {
        spdlog::error("fastgltf: Unable to load file: {}", fastgltf::getErrorMessage(error));
        return false;
    }

    std::stack<const fastgltf::Node*> node_stack;
    for(auto node : gltf->scenes[0].nodeIndices) {
        node_stack.emplace(&gltf->nodes[node]);
    }

    Model model;

    while(!node_stack.empty()) {
        auto node = node_stack.top();
        node_stack.pop();
        spdlog::debug("{}", node->name);

        for(auto node : node->children) {
            node_stack.emplace(&gltf->nodes[node]);
        }

        if(node->meshIndex.has_value()) {
            const auto& fgmesh = gltf->meshes[node->meshIndex.value()];
            Mesh mesh;
            mesh.name = fgmesh.name;

            for(const auto& primitive : fgmesh.primitives) {
                auto& positions = gltf->accessors[primitive.findAttribute("POSITION")->second];
                auto& normals = gltf->accessors[primitive.findAttribute("NORMAL")->second];
                auto color_idx = primitive.findAttribute("COLOR_0");
                auto initial_index = mesh.vertices.size();
                mesh.vertices.resize(mesh.vertices.size() + positions.count);
                auto& _gltf = gltf.get();
                fastgltf::iterateAccessorWithIndex<glm::vec3>(_gltf, positions, [&](glm::vec3 vec, size_t idx) {
                    mesh.vertices[initial_index + idx].position = vec; 
                });
                fastgltf::iterateAccessorWithIndex<glm::vec3>(_gltf, normals, [&](glm::vec3 vec, size_t idx) {
                    mesh.vertices[initial_index + idx].normal = vec; 
                });
                if(color_idx->second < gltf->accessors.size()) {
                    auto& colors = gltf->accessors[color_idx->second];
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(_gltf, colors, [&](glm::vec4 vec, size_t idx) {
                        mesh.vertices[initial_index + idx].color = glm::vec3{vec}; 
                    });
                } else {
                    fastgltf::iterateAccessorWithIndex<glm::vec3>(_gltf, positions, [&](glm::vec3 vec, size_t idx) {
                        mesh.vertices[initial_index + idx].color = glm::vec3{1.0};
                    });
                }

                if(primitive.indicesAccessor.has_value()) {
                    u64 start_index = mesh.indices.size();
                    mesh.indices.resize(mesh.indices.size() + gltf->accessors[primitive.indicesAccessor.value()].count);
                    fastgltf::iterateAccessorWithIndex<u32>(_gltf, gltf->accessors[primitive.indicesAccessor.value()], [&](u32 index, u32 idx) {
                        mesh.indices[start_index + idx] = index;
                    });
                }
            }

            model.meshes.push_back(std::move(mesh));
        }
    }

    models[name.data()] = std::move(model);

    return true;
}

void Renderer::setup_scene() {
    std::vector<float> vertices;
    std::vector<u32> indices;

    for(const auto& [name, model] : models) {
        for(const auto& mesh : model.meshes) {
            GpuMesh gpu{
                .mesh = &mesh,
                .vertex_offset = (u32)vertices.size() / (u32)(sizeof(Vertex) / sizeof(f32)),
                .vertex_count = (u32)mesh.vertices.size(),
                .index_offset = (u32)indices.size(),
                .index_count = (u32)mesh.indices.size()
            };

            scene.models.push_back(gpu);

            for(auto& v : mesh.vertices) {
                vertices.push_back(v.position.x);
                vertices.push_back(v.position.y);
                vertices.push_back(v.position.z);
                vertices.push_back(v.normal.x);
                vertices.push_back(v.normal.y);
                vertices.push_back(v.normal.z);
                vertices.push_back(v.color.x);
                vertices.push_back(v.color.y);
                vertices.push_back(v.color.z);
            }

            indices.insert(indices.end(), mesh.indices.begin(), mesh.indices.end());
        }
    }

    scene.vertex_buffer = create_buffer("scene_vertex_buffer", vk::BufferUsageFlagBits::eVertexBuffer, std::span{vertices});
    scene.index_buffer = create_buffer("scene_index_buffer", vk::BufferUsageFlagBits::eIndexBuffer, std::span{indices});
}

bool Renderer::initialize_vulkan() {
    if(!glfwVulkanSupported()) {
        spdlog::error("Vulkan is not supported");
        return false;
    }

    vkb::InstanceBuilder instance_builder;
    auto instance_result = instance_builder
        .set_app_name("vxgi")
        .require_api_version(1, 3)
        .request_validation_layers()
        .enable_validation_layers()
        .use_default_debug_messenger()
        .enable_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
        .build();
    if(!instance_result) {
        spdlog::error("Failed to create vulkan instance: {}", instance_result.error().message());
        return false;
    }
    instance = instance_result->instance;

    VkSurfaceKHR _surface;
    glfwCreateWindowSurface(instance, window, 0, &_surface);
    if(!_surface) {
        spdlog::error("Failed to create window surface");
        return false;
    }
    surface = _surface;

    vkb::PhysicalDeviceSelector pdev_sel{instance_result.value(), _surface};
    auto pdev_sel_result = pdev_sel
        .set_minimum_version(1, 3)
        .select();
    if(!pdev_sel_result) {
        spdlog::error("Vulkan: failed to find suitable physical device: {}", pdev_sel_result.error().message());
        return false;
    }
    physical_device = pdev_sel_result->physical_device;

    vk::PhysicalDeviceFeatures2 features;
    vk::PhysicalDeviceDescriptorIndexingFeatures desc_idx_features;
    vk::PhysicalDeviceDynamicRenderingFeatures dyn_rend_features;
    dyn_rend_features.dynamicRendering = true;
    desc_idx_features.descriptorBindingVariableDescriptorCount = true;
    desc_idx_features.descriptorBindingPartiallyBound = true;
    desc_idx_features.shaderSampledImageArrayNonUniformIndexing = true;
    desc_idx_features.descriptorBindingSampledImageUpdateAfterBind = true;
    features.features.fragmentStoresAndAtomics = true;
    features.features.geometryShader = true;

    vkb::DeviceBuilder device_builder{pdev_sel_result.value()};
    auto device_builder_result = device_builder
        .add_pNext(&features)
        .add_pNext(&desc_idx_features)
        .add_pNext(&dyn_rend_features)
        .build();
    if(!device_builder_result) {
        spdlog::error("Vulkan: failed to create device: {}", device_builder_result.error().message());
        return false;
    }
    device = device_builder_result->device;

    graphics_queue_idx = device_builder_result->get_queue_index(vkb::QueueType::graphics).value();
    presentation_queue_idx = device_builder_result->get_queue_index(vkb::QueueType::present).value();
    graphics_queue = device_builder_result->get_queue(vkb::QueueType::graphics).value();
    presentation_queue = device_builder_result->get_queue(vkb::QueueType::present).value();

    VmaVulkanFunctions vma_vk_funcs{
        .vkGetInstanceProcAddr = instance_result->fp_vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = instance_result->fp_vkGetDeviceProcAddr
    };
    VmaAllocatorCreateInfo vma_info{
        .physicalDevice = physical_device,
        .device = device,
        .pVulkanFunctions = &vma_vk_funcs,
        .instance = instance,
        .vulkanApiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0),
    };
    auto vma_result = vmaCreateAllocator(&vma_info, &vma);
    if(vma_result != VK_SUCCESS) {
        spdlog::error("VMA: could not create allocator");
    }

    dynamic_loader = vk::DispatchLoaderDynamic{instance, instance_result->fp_vkGetInstanceProcAddr, device, device_builder_result->fp_vkGetDeviceProcAddr};

    return true;
}

bool Renderer::initialize_swapchain() {
    vkb::SwapchainBuilder swapchain_builder{physical_device, device, surface, graphics_queue_idx, presentation_queue_idx};
    auto swapchain_result = swapchain_builder
        .set_desired_extent(window_width, window_height)
        .set_desired_format(VkSurfaceFormatKHR{
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .build();
    if(!swapchain_result) {
        spdlog::error("Vulkan: could not create swapchain");
        return false;
    }

    auto _swapchain = swapchain_result.value();
    swapchain = _swapchain;
    auto images = _swapchain.get_images().value();
    auto views = _swapchain.get_image_views().value();
    for(auto &img : images) { swapchain_images.push_back(img); }
    for(auto &view : views) { swapchain_views.push_back(view); }
    swapchain_format = vk::Format{_swapchain.image_format};

    set_debug_name(device, swapchain, "swapchain");

    return true;
}

bool Renderer::initialize_frame_resources() {
    for(u32 i=0; i<FRAMES_IN_FLIGHT; ++i) {
        frames.at(i).pool = device.createCommandPool(vk::CommandPoolCreateInfo{
            {}, graphics_queue_idx
        });
        frames.at(i).cmd = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            frames.at(i).pool, vk::CommandBufferLevel::ePrimary, 1
        })[0];
        frames.at(i).swapchain_semaphore = device.createSemaphore({});
        frames.at(i).rendering_semaphore = device.createSemaphore({});
        frames.at(i).in_flight_fence = device.createFence({vk::FenceCreateFlagBits::eSignaled});

        set_debug_name(device, frames.at(i).pool, std::format("frame_pool_{}", i));
        set_debug_name(device, frames.at(i).cmd, std::format("frame_cmd_{}", i));
        set_debug_name(device, frames.at(i).swapchain_semaphore, std::format("frame_swapchain_semaphore_{}", i));
        set_debug_name(device, frames.at(i).rendering_semaphore, std::format("frame_rendering_semaphore_{}", i));
        set_debug_name(device, frames.at(i).in_flight_fence, std::format("frame_in_flight_fence_{}", i));
    }

    return true;
}

bool Renderer::initialize_render_passes() {
    static std::filesystem::path shader_path = "data/shaders";
    std::vector<std::filesystem::path> shader_paths{
        "default_lit.vert",
        "default_lit.frag",
        "voxelize.vert",
        "voxelize.geom",
        "voxelize.frag",
        "merge_voxels.comp",
    };
    std::vector<std::vector<u32>> irs(shader_paths.size()); 
    #pragma omp parallel for
    for(u32 i=0; i<irs.size(); ++i) {
        char error[256] = {};
        auto full_path = shader_path / shader_paths.at(i);
        auto path_str = full_path.string();
        auto parent_path_str = shader_path.string();
        auto file = stb_include_file((char*) path_str.c_str(), 0, (char*)parent_path_str.c_str(), error);

        if(error[0] != 0) {
            spdlog::error("stb_include: Error {}", error);
        }

        irs.at(i) = compile_shader(shader_paths.at(i).string(), file);
        free(file);
    }
    
    std::vector<vk::ShaderModule> modules(irs.size());
    for(u32 i=0; const auto& ir : irs) {
        modules.at(i) = device.createShaderModule(vk::ShaderModuleCreateInfo{{}, ir.size() * sizeof(u32), ir.data()});
        set_debug_name(device, modules.at(i), shader_paths.at(i).string());
    }

    static constexpr vk::ShaderStageFlags all_stages = 
        vk::ShaderStageFlagBits::eVertex | 
        vk::ShaderStageFlagBits::eGeometry | 
        vk::ShaderStageFlagBits::eFragment | 
        vk::ShaderStageFlagBits::eCompute;

    vk::DescriptorSetLayoutBinding global_set_bindings[] {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eUniformBuffer, 1, all_stages},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, all_stages},
    };

    vk::DescriptorSetLayoutBinding default_lit_set_bindings[] {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eSampledImage, 1, all_stages},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eSampler, 1, all_stages},
    };

    vk::DescriptorSetLayoutBinding voxelize_set_bindings[] {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageImage, 1, all_stages},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageImage, 1, all_stages},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eSampler, 1, all_stages},
    };

    vk::DescriptorSetLayoutBinding merge_voxels_set_bindings[] {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eSampledImage, 1, all_stages},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageImage, 1, all_stages},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageImage, 1, all_stages},
        vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eSampler, 1, all_stages},
    };

    vk::DescriptorSetLayoutCreateInfo global_set_info{{}, global_set_bindings};
    vk::DescriptorSetLayoutCreateInfo default_lit_info{{}, global_set_bindings};
    vk::DescriptorSetLayoutCreateInfo voxelize_info{{}, global_set_bindings};
    vk::DescriptorSetLayoutCreateInfo merge_voxels_info{{}, global_set_bindings};

    vk::DescriptorPoolSize global_sizes[] {
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1024},
    };
    global_desc_pool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
        {}, 1024, global_sizes
    });
    set_debug_name(device, global_desc_pool, "global_pool");

    global_set_layout = device.createDescriptorSetLayout(global_set_info);
    set_debug_name(device, global_set_layout, "global_set_layout");

    default_lit_set_layout = device.createDescriptorSetLayout(default_lit_info);
    set_debug_name(device, default_lit_set_layout, "default_lit_set_layout");

    voxelize_set_layout = device.createDescriptorSetLayout(voxelize_info);
    set_debug_name(device, voxelize_set_layout, "voxelize_set_layout");

    merge_voxels_set_layout = device.createDescriptorSetLayout(merge_voxels_info);
    set_debug_name(device, merge_voxels_set_layout, "merge_voxels_set_layout");

    global_set = DescriptorSet{device, global_desc_pool, global_set_layout};
    default_lit_set = DescriptorSet{device, global_desc_pool, default_lit_set_layout};
    voxelize_set = DescriptorSet{device, global_desc_pool, voxelize_set_layout};
    merge_voxels_set = DescriptorSet{device, global_desc_pool, merge_voxels_set_layout};

    return true;
}

std::vector<u32> Renderer::compile_shader(std::string_view filename, std::string_view file) {
    shaderc::Compiler compiler;
    auto result = compiler.CompileGlslToSpv(file.data(), shaderc_glsl_infer_from_source, filename.data());
    if(result.GetCompilationStatus() != shaderc_compilation_status_success) {
        spdlog::error("Shader compilation error: {}", result.GetErrorMessage().c_str());
        return {};
    }

    return std::vector<u32>{result.begin(), result.end()};
}