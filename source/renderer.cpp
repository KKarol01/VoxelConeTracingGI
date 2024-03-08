#include "renderer.hpp"
#include "input.hpp"
#include <spdlog/spdlog.h>
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <VkBootstrap.h>
#include <shaderc/shaderc.hpp>
#include <stb/stb_include.h>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
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

Texture2D::Texture2D(u32 width, u32 height, vk::Format format, u32 mips, vk::ImageUsageFlags usage) {
    storage = get_context().renderer->create_texture_storage(vk::ImageCreateInfo{
        {},
        vk::ImageType::e2D,
        format,
        {width, height, 1},
        mips,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        usage
    });
}

Texture3D::Texture3D(u32 width, u32 height, u32 depth, vk::Format format, u32 mips, vk::ImageUsageFlags usage) {
    storage = get_context().renderer->create_texture_storage(vk::ImageCreateInfo{
        vk::ImageCreateFlagBits::eMutableFormat,
        vk::ImageType::e3D,
        format,
        {width, height, depth},
        mips,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        usage
    });
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

    if(!initialize_imgui()) {
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

            for(const auto& primitive : fgmesh.primitives) {
                Mesh mesh;
                mesh.name = fgmesh.name;
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
                if(primitive.findAttribute("COLOR_0") != primitive.attributes.end()) {
                    auto& colors = gltf->accessors[color_idx->second];
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(_gltf, colors, [&](glm::vec4 vec, size_t idx) {
                        mesh.vertices[initial_index + idx].color = glm::vec3{vec.x, vec.y, vec.z}; 
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

                model.meshes.push_back(std::move(mesh));
            }

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

void Renderer::draw() {
    while(!glfwWindowShouldClose(window)) {
        camera.update();
        get_context().input->update();
        glfwPollEvents();

        auto P = glm::perspectiveFov(glm::radians(75.0f), 1024.0f, 768.0f, 0.01f, 25.0f);
        auto V = camera.view;
        spdlog::debug("{} {} {}", camera.pos.x, camera.pos.y, camera.pos.z);
        glm::mat4 global_buffer_data[] {
            P, V
        };
        memcpy(global_buffer->data, global_buffer_data, sizeof(global_buffer_data));

        auto& fr = get_frame_res();
        auto& cmd = fr.cmd;

        auto img = device.acquireNextImageKHR(swapchain, -1ull, fr.swapchain_semaphore).value;
        
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.bindVertexBuffers(0, scene.vertex_buffer->buffer, 0ull);
        cmd.bindIndexBuffer(scene.index_buffer->buffer, 0, vk::IndexType::eUint32);

        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pp_voxelize.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pp_default_lit.layout, 0, global_set.set, {});
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pp_voxelize.layout, 1, voxelize_set.set, {});

        cmd.clearColorImage(voxel_albedo.storage->image, vk::ImageLayout::eGeneral, vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, voxel_albedo.storage->mips, 0, 1});
        cmd.clearColorImage(voxel_normal.storage->image, vk::ImageLayout::eGeneral, vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, voxel_normal.storage->mips, 0, 1});
        cmd.clearColorImage(voxel_radiance.storage->image, vk::ImageLayout::eGeneral, vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, voxel_radiance.storage->mips, 0, 1});
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                {}, {}, voxel_albedo.storage->image, {vk::ImageAspectFlagBits::eColor, 0, voxel_albedo.storage->mips, 0, 1}});
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                {}, {}, voxel_normal.storage->image, {vk::ImageAspectFlagBits::eColor, 0, voxel_normal.storage->mips, 0, 1}});
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                {}, {}, voxel_radiance.storage->image, {vk::ImageAspectFlagBits::eColor, 0, voxel_radiance.storage->mips, 0, 1}});
        
        cmd.beginRendering(vk::RenderingInfo{
            {},
            {{}, {256, 256}},
            1,
            0,
        });

        cmd.setViewportWithCount(vk::Viewport{0.0f, 0.0, 256.0, 256.0, 0.0f, 1.0f});
        cmd.setScissorWithCount(vk::Rect2D{{}, {256, 256}});
        for(auto& gpum : scene.models) {
            cmd.drawIndexed(gpum.index_count, 1, gpum.index_offset, gpum.vertex_offset, 0);
        }
                
        cmd.endRendering();

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                {}, {}, voxel_albedo.storage->image, {vk::ImageAspectFlagBits::eColor, 0, voxel_albedo.storage->mips, 0, 1}});
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                {}, {}, voxel_normal.storage->image, {vk::ImageAspectFlagBits::eColor, 0, voxel_normal.storage->mips, 0, 1}});
 
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pp_merge_voxels.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pp_merge_voxels.layout, 0, global_set.set, {});
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pp_merge_voxels.layout, 1, merge_voxels_set.set, {});
        cmd.dispatch(256/8, 256/8, 256/8);

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eTransferRead,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                {}, {}, voxel_radiance.storage->image, {vk::ImageAspectFlagBits::eColor, 0, voxel_radiance.storage->mips, 0, 1}});

        i32 size = 256;
        for(u32 i=1; i<voxel_radiance.storage->mips; ++i) {
            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer,
                {}, {}, {},
                vk::ImageMemoryBarrier{
                    vk::AccessFlagBits::eTransferWrite,
                    vk::AccessFlagBits::eTransferRead,
                    vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                    {}, {}, voxel_radiance.storage->image, {vk::ImageAspectFlagBits::eColor, i-1, 2, 0, 1}});
            i32 mip_size = size >> i;
            cmd.blitImage(voxel_radiance.storage->image, vk::ImageLayout::eGeneral,
                voxel_radiance.storage->image, vk::ImageLayout::eGeneral,
                vk::ImageBlit{
                    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i-1, 0, 1},
                    { vk::Offset3D{}, vk::Offset3D{mip_size<<1, mip_size<<1, mip_size<<1} },
                    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i, 0, 1},
                    { vk::Offset3D{}, vk::Offset3D{mip_size, mip_size, mip_size} },
                },
                vk::Filter::eLinear);
        }
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                {}, {}, voxel_radiance.storage->image, {vk::ImageAspectFlagBits::eColor, 0, voxel_radiance.storage->mips, 0, 1}});

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eNone,
                vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentRead,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eDepthAttachmentOptimal,
                {},
                {},
                depth_texture.storage->image,
                {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
            }
        );
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eNone,
                vk::AccessFlagBits::eColorAttachmentWrite,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},
                {},
                swapchain_images.at(img),
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
            }
        );
        
        vk::RenderingAttachmentInfo color1{
            swapchain_views.at(img),
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ResolveModeFlagBits::eNone,
            {},
            {},
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore,
            vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f} 
        };
        vk::RenderingAttachmentInfo depth{
            depth_texture_view,
            vk::ImageLayout::eDepthAttachmentOptimal,
            vk::ResolveModeFlagBits::eNone,
            {},
            {},
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore,
            vk::ClearDepthStencilValue{1.0f, 0}
        };

        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pp_default_lit.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pp_default_lit.layout, 1, default_lit_set.set, {});
        cmd.beginRendering(vk::RenderingInfo{
            {},
            {{}, {1024, 768}},
            1,
            0,
            color1,
            &depth
        });

        cmd.setViewportWithCount(vk::Viewport{0.0f, 768.0f, 1024.0f, -768.0f, 0.0f, 1.0f});
        cmd.setScissorWithCount(vk::Rect2D{{}, {1024, 768}});
        for(auto& gpum : scene.models) {
            cmd.drawIndexed(gpum.index_count, 1, gpum.index_offset, gpum.vertex_offset, 0);
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("asdf");
        ImGui::End();
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        
        cmd.endRendering();

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eColorAttachmentWrite,
                vk::AccessFlagBits::eNone,
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                {},
                {},
                swapchain_images.at(img),
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
            }
        );

        cmd.end();
        vk::PipelineStageFlags wait_masks[] {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        graphics_queue.submit(vk::SubmitInfo{fr.swapchain_semaphore, wait_masks, cmd, fr.rendering_semaphore});
        u32 image_indices[] {img};
        presentation_queue.presentKHR(vk::PresentInfoKHR{
            fr.rendering_semaphore, swapchain, image_indices
        });
        device.waitIdle();
    }
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

    vulkan_function_pointers = {
        .get_instance_proc_addr = instance_result->fp_vkGetInstanceProcAddr,
        .get_device_proc_addr = instance_result->fp_vkGetDeviceProcAddr,
    };

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
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphics_queue_idx
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
bool Renderer::initialize_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    if(!ImGui_ImplGlfw_InitForVulkan(window, true)) {
        spdlog::error("IMGUI: could not init glfw for vulkan");
        return false;
    }
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physical_device;
    init_info.Device = device;
    init_info.QueueFamily = graphics_queue_idx;
    init_info.Queue = graphics_queue;
    init_info.DescriptorPool = global_desc_pool;
    init_info.MinImageCount = swapchain_images.size();
    init_info.ImageCount = swapchain_images.size();
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.UseDynamicRendering = true;
    init_info.PipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{
        0, swapchain_format, depth_texture.storage->format
    };

    if(!ImGui_ImplVulkan_Init(&init_info)) {
        spdlog::error("IMGUI: could not init vulkan impl");
        return false;
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
        ++i;
    }

    static constexpr vk::ShaderStageFlags all_stages = 
        vk::ShaderStageFlagBits::eVertex | 
        vk::ShaderStageFlagBits::eGeometry | 
        vk::ShaderStageFlagBits::eFragment | 
        vk::ShaderStageFlagBits::eCompute;

    vk::DescriptorSetLayoutBinding global_set_bindings[] {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eUniformBuffer, 1, all_stages},
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
    vk::DescriptorSetLayoutCreateInfo default_lit_info{{}, default_lit_set_bindings};
    vk::DescriptorSetLayoutCreateInfo voxelize_info{{}, voxelize_set_bindings};
    vk::DescriptorSetLayoutCreateInfo merge_voxels_info{{}, merge_voxels_set_bindings};

    vk::DescriptorPoolSize global_sizes[] {
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 10},
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

    voxel_albedo = Texture3D{256, 256, 256, vk::Format::eR32Uint, 1, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled};
    set_debug_name(device, voxel_albedo.storage->image, "voxel_albedo");

    auto voxel_albedo_view = device.createImageView(vk::ImageViewCreateInfo{{}, voxel_albedo.storage->image, vk::ImageViewType::e3D, voxel_albedo.storage->format, {}, {vk::ImageAspectFlagBits::eColor, 0, voxel_albedo.storage->mips, 0, 1}});
    set_debug_name(device, voxel_albedo_view, "voxel_albedo_view");

    voxel_normal = Texture3D{256, 256, 256, vk::Format::eR32Uint, 1, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled};
    set_debug_name(device, voxel_normal.storage->image, "voxel_normal");

    auto voxel_normal_view = device.createImageView(vk::ImageViewCreateInfo{{}, voxel_normal.storage->image, vk::ImageViewType::e3D, voxel_normal.storage->format, {}, {vk::ImageAspectFlagBits::eColor, 0, voxel_normal.storage->mips, 0, 1}});
    set_debug_name(device, voxel_normal_view, "voxel_normal_view");

    voxel_radiance = Texture3D{256, 256, 256, vk::Format::eR8G8B8A8Unorm, (u32)std::log2f(256.0f)+1, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled};
    set_debug_name(device, voxel_radiance.storage->image, "voxel_radiance");

    auto voxel_radiance_view = device.createImageView(vk::ImageViewCreateInfo{{}, voxel_radiance.storage->image, vk::ImageViewType::e3D, voxel_radiance.storage->format, {}, {vk::ImageAspectFlagBits::eColor, 0, voxel_radiance.storage->mips, 0, 1}});
    set_debug_name(device, voxel_radiance_view, "voxel_radiance_view");

    auto voxel_sampler = device.createSampler(vk::SamplerCreateInfo{
        {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        0.0f, false, 0.0f, false, vk::CompareOp::eLess, 0.0f, (f32)voxel_radiance.storage->mips
    });

    auto merge_albedo_view = device.createImageView(vk::ImageViewCreateInfo{{}, voxel_albedo.storage->image, vk::ImageViewType::e3D, vk::Format::eR8G8B8A8Unorm, {}, {vk::ImageAspectFlagBits::eColor, 0, voxel_albedo.storage->mips, 0, 1}});
    set_debug_name(device, voxel_radiance_view, "merge_albedo_view");

    auto merge_normal_view = device.createImageView(vk::ImageViewCreateInfo{{}, voxel_normal.storage->image, vk::ImageViewType::e3D, vk::Format::eR8G8B8A8Unorm, {}, {vk::ImageAspectFlagBits::eColor, 0, voxel_normal.storage->mips, 0, 1}});
    set_debug_name(device, voxel_radiance_view, "merge_normal_view");

    depth_texture = Texture2D{window_width, window_height, vk::Format::eD32Sfloat, 1, vk::ImageUsageFlagBits::eDepthStencilAttachment};
    set_debug_name(device, depth_texture.storage->image, "depth_texture");

    depth_texture_view = device.createImageView(vk::ImageViewCreateInfo{
        {}, depth_texture.storage->image, vk::ImageViewType::e2D, depth_texture.storage->format, {}, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
    });
    set_debug_name(device, depth_texture_view, "depth_texture_view");

    PipelineBuilder default_lit_builder{this};
    pp_default_lit = default_lit_builder
        .with_vertex_input(
            {
                vk::VertexInputBindingDescription{0, sizeof(Vertex), vk::VertexInputRate::eVertex}
            },
            {
                vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat, 0},
                vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat, 12},
                vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32B32Sfloat, 24},
            })
        .with_depth_testing(true, true, vk::CompareOp::eLess)
        .with_culling(vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise)
        .with_shaders({
            {vk::ShaderStageFlagBits::eVertex, modules.at(0)},
            {vk::ShaderStageFlagBits::eFragment, modules.at(1)},
        })
        .with_color_attachments({swapchain_format})
        .with_depth_attachment(vk::Format::eD32Sfloat)
        .with_layout(global_set_layout)
        .with_layout(default_lit_set_layout)
        .build_graphics("default_lit_pipeline");

    PipelineBuilder voxelize_builder{this};
    pp_voxelize = voxelize_builder
        .with_vertex_input(
            {
                vk::VertexInputBindingDescription{0, sizeof(Vertex), vk::VertexInputRate::eVertex}
            },
            {
                vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat, 0},
                vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat, 12},
                vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32B32Sfloat, 24},
            })
        .with_depth_testing(false, false, vk::CompareOp::eAlways)
        .with_culling(vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise)
        .with_shaders({
            {vk::ShaderStageFlagBits::eVertex, modules.at(2)},
            {vk::ShaderStageFlagBits::eGeometry, modules.at(3)},
            {vk::ShaderStageFlagBits::eFragment, modules.at(4)},
        })
        .with_layout(global_set_layout)
        .with_layout(voxelize_set_layout)
        .build_graphics("voxelize_pipeline");

    PipelineBuilder merge_voxels_builder{this};
    pp_merge_voxels = merge_voxels_builder
        .with_shaders({
            {vk::ShaderStageFlagBits::eCompute, modules.at(5)},
        })
        .with_layout(global_set_layout)
        .with_layout(merge_voxels_set_layout)
        .build_compute("merge_voxels_pipeline");

    glm::mat4 global_buffer_size[2];
    global_buffer = create_buffer("global_ubo", vk::BufferUsageFlagBits::eUniformBuffer, std::span{global_buffer_size, 2});

    DescriptorInfo global_set_infos[] {
        DescriptorInfo{vk::DescriptorType::eUniformBuffer, global_buffer->buffer, 0, vk::WholeSize},
    };
    global_set.update_bindings(device, 0, 0, global_set_infos);

    DescriptorInfo default_lit_set_infos[] {
        DescriptorInfo{vk::DescriptorType::eSampledImage, voxel_radiance_view, vk::ImageLayout::eGeneral},
        DescriptorInfo{vk::DescriptorType::eSampler, voxel_sampler},
    };
    default_lit_set.update_bindings(device, 0, 0, default_lit_set_infos);

    DescriptorInfo voxelize_set_infos[] {
        DescriptorInfo{vk::DescriptorType::eStorageImage, voxel_albedo_view, vk::ImageLayout::eGeneral},
        DescriptorInfo{vk::DescriptorType::eStorageImage, voxel_normal_view, vk::ImageLayout::eGeneral},
        DescriptorInfo{vk::DescriptorType::eSampler, voxel_sampler},
    };
    voxelize_set.update_bindings(device, 0, 0, voxelize_set_infos);
 
    DescriptorInfo merge_voxels_infos[] {
        DescriptorInfo{vk::DescriptorType::eSampledImage, merge_albedo_view, vk::ImageLayout::eGeneral},
        DescriptorInfo{vk::DescriptorType::eStorageImage, merge_normal_view, vk::ImageLayout::eGeneral},
        DescriptorInfo{vk::DescriptorType::eStorageImage, voxel_radiance_view, vk::ImageLayout::eGeneral},
        DescriptorInfo{vk::DescriptorType::eSampler, voxel_sampler},
    };
    merge_voxels_set.update_bindings(device, 0, 0, merge_voxels_infos);

    get_frame_res().cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    get_frame_res().cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            vk::AccessFlagBits::eNone,
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eGeneral,
            {},
            {},
            voxel_albedo.storage->image,
            {vk::ImageAspectFlagBits::eColor, 0, voxel_albedo.storage->mips, 0, 1}
        }
    );
    get_frame_res().cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            vk::AccessFlagBits::eNone,
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eGeneral,
            {},
            {},
            voxel_normal.storage->image,
            {vk::ImageAspectFlagBits::eColor, 0, voxel_normal.storage->mips, 0, 1}
        }
    );
    get_frame_res().cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            vk::AccessFlagBits::eNone,
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eGeneral,
            {},
            {},
            voxel_radiance.storage->image,
            {vk::ImageAspectFlagBits::eColor, 0, voxel_radiance.storage->mips, 0, 1}
        }
    );
        
    get_frame_res().cmd.end();
    graphics_queue.submit(vk::SubmitInfo{{}, {}, get_frame_res().cmd});
    graphics_queue.waitIdle();

    return true;
}

std::vector<u32> Renderer::compile_shader(std::string_view filename, std::string_view file) {
    shaderc::Compiler compiler;
    auto result = compiler.CompileGlslToSpv(file.data(), shaderc_glsl_infer_from_source, filename.data());
    if(result.GetCompilationStatus() != shaderc_compilation_status_success) {
        spdlog::error("Shader {} compilation error: {}", filename, result.GetErrorMessage().c_str());
        return {};
    }

    return std::vector<u32>{result.begin(), result.end()};
}