#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#include "renderer.hpp"
#include "input.hpp"
#include "pipelines.hpp"
#include "render_graph.hpp"
#include "allocator.hpp"
#include <spdlog/spdlog.h>
#include <glm/gtc/matrix_transform.hpp>
#include <VkBootstrap.h>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#include <format>
#include <tracy/Tracy.hpp>

vk::Device SetDebugNameDetails::device() {
    return get_context().renderer->device;
}

vk::Instance SetDebugNameDetails::instance() {
    return get_context().renderer->instance;
}

PFN_vkGetInstanceProcAddr SetDebugNameDetails::get_instance_proc_addr() {
    return get_context().renderer->vulkan_function_pointers.get_instance_proc_addr;
}

PFN_vkGetDeviceProcAddr SetDebugNameDetails::get_device_proc_addr() {
    return get_context().renderer->vulkan_function_pointers.get_device_proc_addr;
}

void GpuScene::render(vk::CommandBuffer cmd) {
    cmd.drawIndexedIndirect(indirect_commands_buffer.buffer, 0, draw_count, sizeof(vk::DrawIndexedIndirectCommand));
}

Buffer::Buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, u64 size)
    : allocation(get_context().renderer->allocator->create_buffer(label, usage, map_memory, size)) { 
    buffer = Buffer::operator->()->buffer;
}

Buffer::Buffer(std::string_view label, vk::BufferUsageFlags usage, bool map_memory, std::span<const std::byte> optional_data)
    : allocation(get_context().renderer->allocator->create_buffer(label, usage, map_memory, optional_data)) { 
    buffer = Buffer::operator->()->buffer;
}

BufferAllocation* Buffer::operator->() { return &get_context().renderer->allocator->get_buffer(allocation); }

const BufferAllocation* Buffer::operator->() const { return &get_context().renderer->allocator->get_buffer(allocation); }

TextureAllocation* Texture::operator->() { return &get_context().renderer->allocator->get_texture(storage); }

const TextureAllocation* Texture::operator->() const { return &get_context().renderer->allocator->get_texture(storage); }

Texture2D::Texture2D(std::string_view label, u32 width, u32 height, vk::Format format, u32 mips, vk::ImageUsageFlags usage, std::span<const std::byte> optional_data) 
    : Texture(get_context().renderer->allocator->create_texture_storage(
        label,
        vk::ImageCreateInfo{
            {},
            vk::ImageType::e2D,
            format,
            {width, height, 1},
            mips,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            usage
        }, 
        optional_data)) { }

Texture3D::Texture3D(std::string_view label, u32 width, u32 height, u32 depth, vk::Format format, u32 mips, vk::ImageUsageFlags usage) 
    : Texture(get_context().renderer->allocator->create_texture_storage(
        label,
        vk::ImageCreateInfo{
            vk::ImageCreateFlagBits::eMutableFormat,
            vk::ImageType::e3D,
            format,
            {width, height, depth},
            mips,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            usage
        })) { }

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

void Renderer::setup_scene() {
    auto& context = get_context();
    auto& scene = *context.scene;

    std::unordered_map<const TextureAllocation*, u64> texture_indices;
    std::vector<const TextureAllocation*> material_textures{}; // index 0 reserved for lack of texture
    material_textures.reserve(scene.material_textures.size());
    std::vector<GpuInstancedMesh> instanced_meshes;

    const auto add_or_get_material_texture = [&](const Texture2D& tex) {
        if(!tex) { return 0ull; }

        auto& tex_storage = allocator->get_texture(tex.storage);
        if(texture_indices.contains(&tex_storage)) { return texture_indices.at(&tex_storage); } 
        texture_indices[&tex_storage] = material_textures.size();
        material_textures.push_back(&tex_storage);
        return material_textures.size() - 1ull;
    };

    const auto model_instance_counts = [&] {
        std::unordered_map<Handle<Model>, u32> instances;
        for(const auto& e : scene.scene_models) {
            ++instances[e.model];
        }
        return instances;
    }();
    const auto total_instanced_meshes = [&] {
        u32 count = 0;
        for(const auto& [handle, instance_count] : model_instance_counts) {
            const auto& model = scene.get_model(handle);
            count += instance_count * model.meshes.size();
        }
        return count;
    }();
    const auto total_meshes = [&scene] { 
        u32 count = 0;
        for(const auto& e : scene.models) { count += e.meshes.size(); }
        return count;
    }();
    const auto instanced_models_offsets = [&] {
        std::unordered_map<Handle<Model>, u64> offsets;
        for(u64 i=0; i<scene.models.size(); ++i) {
            const auto& model = scene.models.at(i);
            if(i==0) { offsets[model] = 0; continue; }
            offsets[model] = offsets[scene.models.at(i-1)] + model_instance_counts.at(scene.models.at(i-1)) * scene.models.at(i-1).meshes.size();
        }
        return offsets;
    }();
    const auto [total_vertex_count, total_index_count] = [&] {
        u64 total_vertex_count = 0, total_index_count = 0;
        for(const auto& e : scene.models) {
            for(const auto& f : e.meshes) { 
                total_vertex_count += f.vertices.size();
                total_index_count += f.indices.size();
            }
        }
        return std::make_pair(total_vertex_count, total_index_count);
    }();

    std::vector<Vertex> vertices;
    std::vector<u32> indices;
    vertices.reserve(total_vertex_count);
    indices.reserve(total_index_count);
    
    render_scene.models.reserve(model_instance_counts.size());
    render_scene.meshes.reserve(total_meshes);
    for(u64 offset = 0; const auto& e : scene.models) {
        render_scene.models.emplace_back(e, offset);
        offset += e.meshes.size();

        for(const auto& f : e.meshes) {
            GpuMesh last_mesh{};
            if(!render_scene.meshes.empty()) { last_mesh = render_scene.meshes.back(); }

            render_scene.meshes.push_back(GpuMesh{
                .mesh = &f,
                .vertex_offset = last_mesh.vertex_offset + last_mesh.vertex_count,
                .vertex_count = (u32)f.vertices.size(),
                .index_offset = last_mesh.index_offset + last_mesh.index_count,
                .index_count = (u32)f.indices.size(),
                .instance_offset = last_mesh.instance_offset + last_mesh.instance_count,
                .instance_count = model_instance_counts.at(e)
            });

            vertices.insert(vertices.end(), f.vertices.begin(), f.vertices.end());
            indices.insert(indices.end(), f.indices.begin(), f.indices.end());
        }
    }

    std::unordered_map<Handle<Model>, u32> parsed_gpu_model_instances;
    instanced_meshes.resize(total_instanced_meshes);
    for(const auto& e : scene.scene_models) {
        GpuModel previous_gpu_model;
        if(!render_scene.models.empty()) {
            previous_gpu_model = render_scene.models.back();
        }

        for(u64 idx = 0u; const auto& f : scene.get_model(e.model).meshes) {
            const auto instanced_mesh_idx = instanced_models_offsets.at(e.model) + parsed_gpu_model_instances[e.model] + idx;
            auto& instanced_mesh = instanced_meshes.at(instanced_mesh_idx); 
            instanced_mesh.diffuse_texture_idx = add_or_get_material_texture(f.material.diffuse_texture);
            instanced_mesh.normal_texture_idx = add_or_get_material_texture(f.material.normal_texture);
            ++idx;
        }
        ++parsed_gpu_model_instances[e.model];
    }

    std::vector<vk::DrawIndexedIndirectCommand> indirect_commands;
    indirect_commands.reserve(render_scene.meshes.size());
    for(const auto& e : render_scene.meshes) {
        indirect_commands.push_back(vk::DrawIndexedIndirectCommand{
            e.index_count,
            e.instance_count,
            e.index_offset,
            (i32)e.vertex_offset,
            e.instance_offset
        });
    }

    render_scene.vertex_buffer = Buffer{"scene_vertex_buffer", vk::BufferUsageFlagBits::eVertexBuffer, false, std::as_bytes(std::span{vertices})};
    render_scene.index_buffer = Buffer{"scene_index_buffer", vk::BufferUsageFlagBits::eIndexBuffer, false, std::as_bytes(std::span{indices})};
    render_scene.instance_buffer = Buffer{"scene_instance_buffer", vk::BufferUsageFlagBits::eStorageBuffer, true, std::as_bytes(std::span{instanced_meshes})};
    render_scene.indirect_commands_buffer = Buffer{"scene_indirect_buffer", vk::BufferUsageFlagBits::eIndirectBuffer, true, std::as_bytes(std::span{indirect_commands})};
    render_scene.draw_count = indirect_commands.size();

    auto sampler = device.createSampler(vk::SamplerCreateInfo{
        {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
        {}, false, {}, false, {}, 0, 11.0f
    });

    std::vector<DescriptorUpdate> updates;
    updates.reserve(material_textures.size() + 1u);

    updates.emplace_back(&render_scene.instance_buffer);
    for(auto& e : material_textures) {
        if(!e) { continue; }
        updates.emplace_back(std::make_tuple(e->default_view, vk::ImageLayout::eShaderReadOnlyOptimal, sampler));
    }

    material_set = descriptor_allocator->allocate("material_set", DescriptorLayout{{}, {
        DescriptorBinding{1, vk::DescriptorType::eStorageBuffer},
        DescriptorBinding{128, vk::DescriptorType::eCombinedImageSampler},
    }, {}, 2, true}, 128, material_textures.size());

    material_set.update(0, 0, updates);
}

void Renderer::render() { 
    static const auto P = glm::perspectiveFov(glm::radians(75.0f), 1024.0f, 768.0f, 0.01f, 25.0f);
    auto V = get_context().camera->view;
    glm::mat4 global_buffer_data[] {
        P, V
    };
    memcpy(global_buffer->data, global_buffer_data, sizeof(global_buffer_data));

    auto& fr = get_frame_res();
    auto& cmd = fr.cmd;

    auto img = device.acquireNextImageKHR(swapchain, -1ull, fr.swapchain_semaphore).value;
    
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    if(allocator->has_jobs()) {
        allocator->complete_jobs(cmd);
        cmd.end();
        graphics_queue.submit(vk::SubmitInfo{{}, {}, cmd});
        graphics_queue.waitIdle();

        [this] {
            static bool once = true;
            if(once) setup_scene();
            once = false;
        }();
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }

    cmd.bindVertexBuffers(0, render_scene.vertex_buffer->buffer, 0ull);
    cmd.bindIndexBuffer(render_scene.index_buffer->buffer, 0, vk::IndexType::eUint32);

    vk::DescriptorSet sets[] {
        global_set.set,
        material_set.set,
    };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pp_default_lit.layout.layout, 0, sets, {});

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pp_default_lit.pipeline);

    render_graph->render(cmd, swapchain_images.at(img), swapchain_views.at(img));

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

    for(auto& e : deletion_queue) { e(); }
}

bool Renderer::initialize_vulkan() {
    if(!glfwVulkanSupported()) {
        spdlog::error("Vulkan is not supported");
        return false;
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init();
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
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_result->fp_vkGetInstanceProcAddr);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

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

    vk::PhysicalDeviceSynchronization2Features synch2_features;
    synch2_features.synchronization2 = true;

    vk::PhysicalDeviceHostQueryResetFeatures host_query_features;
    host_query_features.hostQueryReset = true;

    vk::PhysicalDeviceDynamicRenderingFeatures dyn_rend_features;
    dyn_rend_features.dynamicRendering = true;

    vk::PhysicalDeviceDescriptorIndexingFeatures desc_idx_features;
    desc_idx_features.descriptorBindingVariableDescriptorCount = true;
    desc_idx_features.descriptorBindingPartiallyBound = true;
    desc_idx_features.shaderSampledImageArrayNonUniformIndexing = true;
    desc_idx_features.shaderStorageImageArrayNonUniformIndexing = true;
    desc_idx_features.descriptorBindingSampledImageUpdateAfterBind = true;
    desc_idx_features.descriptorBindingStorageImageUpdateAfterBind = true;
    desc_idx_features.descriptorBindingUniformBufferUpdateAfterBind = true;
    desc_idx_features.descriptorBindingStorageBufferUpdateAfterBind = true;
    desc_idx_features.runtimeDescriptorArray = true;

    vk::PhysicalDeviceFeatures2 features;
    features.features.fragmentStoresAndAtomics = true;
    features.features.geometryShader = true;
    features.features.multiDrawIndirect = true;
    
    vkb::DeviceBuilder device_builder{pdev_sel_result.value()};
    auto device_builder_result = device_builder
        .add_pNext(&features)
        .add_pNext(&desc_idx_features)
        .add_pNext(&dyn_rend_features)
        .add_pNext(&host_query_features)
        .add_pNext(&synch2_features)
        .build();
    if(!device_builder_result) {
        spdlog::error("Vulkan: failed to create device: {}", device_builder_result.error().message());
        return false;
    }
    device = device_builder_result->device;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    graphics_queue_idx = device_builder_result->get_queue_index(vkb::QueueType::graphics).value();
    presentation_queue_idx = device_builder_result->get_queue_index(vkb::QueueType::present).value();
    graphics_queue = device_builder_result->get_queue(vkb::QueueType::graphics).value();
    presentation_queue = device_builder_result->get_queue(vkb::QueueType::present).value();

    VmaVulkanFunctions vma_vk_funcs{
        .vkGetInstanceProcAddr = instance_result->fp_vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = instance_result->fp_vkGetDeviceProcAddr
    };
    VmaAllocatorCreateInfo vma_info{
        .flags = {},
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

    allocator = new RendererAllocator{device, vma};

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

    set_debug_name(swapchain, "swapchain");

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

        set_debug_name(frames.at(i).pool, std::format("frame_pool_{}", i));
        set_debug_name(frames.at(i).cmd, std::format("frame_cmd_{}", i));
        set_debug_name(frames.at(i).swapchain_semaphore, std::format("frame_swapchain_semaphore_{}", i));
        set_debug_name(frames.at(i).rendering_semaphore, std::format("frame_rendering_semaphore_{}", i));
        set_debug_name(frames.at(i).in_flight_fence, std::format("frame_in_flight_fence_{}", i));
    }

    descriptor_allocator = new DescriptorAllocator{device};
    global_set = descriptor_allocator->allocate("global_set", DescriptorLayout{{}, {
        DescriptorBinding{1, vk::DescriptorType::eUniformBuffer},
        DescriptorBinding{1, vk::DescriptorType::eUniformBuffer},
        DescriptorBinding{1, vk::DescriptorType::eUniformBuffer},
    }, {}, 3, false}, 8);
    imgui_set = descriptor_allocator->allocate("imgui_set", DescriptorLayout{{}, {
        DescriptorBinding{1, vk::DescriptorType::eCombinedImageSampler}
    }, {}, 1, false}, 8);

    gi_settings_buffer = Buffer{"gi_settings_buffer", vk::BufferUsageFlagBits::eUniformBuffer, true, sizeof(GlobalIlluminationSettings)};
    new(gi_settings_buffer->data) GlobalIlluminationSettings{};
    gi_settings = static_cast<GlobalIlluminationSettings*>(gi_settings_buffer->data);

    light_settings_buffer = Buffer{"light_settings_buffer", vk::BufferUsageFlagBits::eUniformBuffer, true, sizeof(LightsUBO)};
    new(light_settings_buffer->data) LightsUBO{};
    light_settings = static_cast<LightsUBO*>(light_settings_buffer->data);

    return true;
}

bool Renderer::initialize_imgui() {
    #if 1
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    if(!ImGui_ImplGlfw_InitForVulkan(window, true)) {
        spdlog::error("IMGUI: could not init glfw for vulkan");
        return false;
    }
            
    vk::DescriptorPoolSize sizes[] {
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 4u}
    };

    auto pool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{{}, 4, sizes});
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physical_device;
    init_info.Device = device;
    init_info.QueueFamily = graphics_queue_idx;
    init_info.Queue = graphics_queue;
    init_info.DescriptorPool = pool;
    init_info.MinImageCount = swapchain_images.size();
    init_info.ImageCount = swapchain_images.size();
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.UseDynamicRendering = true;
    init_info.PipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{
        0, swapchain_format, depth_texture->format
    };

    ImGui_ImplVulkan_LoadFunctions([](const char* function_name, void* user_data) {
        return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(get_context().renderer->instance, function_name);
    });

    if(!ImGui_ImplVulkan_Init(&init_info)) {
        spdlog::error("IMGUI: could not init vulkan impl");
        return false;
    }

    #endif

    return true;
}

bool Renderer::initialize_render_passes() {
    render_graph = new RenderGraph{descriptor_allocator};
    
    voxel_albedo = Texture3D{"voxel_albedo", 256, 256, 256, vk::Format::eR32Uint, 1, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled};
    voxel_normal = Texture3D{"voxel_normal", 256, 256, 256, vk::Format::eR32Uint, 1, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled};
    voxel_radiance = Texture3D{"voxel_radiance", 256, 256, 256, vk::Format::eR8G8B8A8Unorm, (u32)std::log2f(256.0f)+1, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled};
    depth_texture = Texture2D{"depth_texture", window_width, window_height, vk::Format::eD32Sfloat, 1, vk::ImageUsageFlagBits::eDepthStencilAttachment};
    vk::Sampler radiance_sampler = device.createSampler(vk::SamplerCreateInfo{
        {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, 
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        0.0f, false, 0.0f, false, {}, 0.0f, (f32)voxel_radiance->mips
    });

    const std::vector<vk::VertexInputBindingDescription> common_input_bindings{
        vk::VertexInputBindingDescription{0, sizeof(Vertex), vk::VertexInputRate::eVertex}
    };
    const std::vector<vk::VertexInputAttributeDescription> common_input_attributes{
        vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat, 0},
        vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat, 12},
        vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32B32Sfloat, 24},
        vk::VertexInputAttributeDescription{3, 0, vk::Format::eR32G32Sfloat, 36},
        vk::VertexInputAttributeDescription{4, 0, vk::Format::eR32G32B32Sfloat, 44},
    };

    PipelineBuilder default_lit_builder{this};
    pp_default_lit = default_lit_builder
        .with_vertex_input(common_input_bindings, common_input_attributes)
        .with_depth_testing(true, true, vk::CompareOp::eLess)
        .with_culling(vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise)
        .with_shaders({"default_lit.vert", "default_lit.frag"})
        .with_color_attachments({swapchain_format})
        .with_depth_attachment(vk::Format::eD32Sfloat)
        .with_variable_upper_limits({128})
        .build_graphics("default_lit_pipeline");

    PipelineBuilder voxelize_builder{this};
    pp_voxelize = voxelize_builder
        .with_vertex_input(common_input_bindings, common_input_attributes)
        .with_depth_testing(false, false, vk::CompareOp::eAlways)
        .with_culling(vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise)
        .with_shaders({"voxelize.vert", "voxelize.geom", "voxelize.frag"})
        .with_variable_upper_limits({128})
        .build_graphics("voxelize_pipeline");

    PipelineBuilder merge_voxels_builder{this};
    pp_merge_voxels = merge_voxels_builder
        .with_shaders({"merge_voxels.comp"})
        .with_variable_upper_limits({128})
        .build_compute("merge_voxels_pipeline");

    PipelineBuilder imgui_builder{this};
    pp_imgui = imgui_builder
        .with_vertex_input(common_input_bindings, common_input_attributes)
        .with_depth_testing(false, false, vk::CompareOp::eAlways)
        .with_shaders({ "imgui.vert", "imgui.frag"})
        .build_graphics("imgui_pipeline");

    glm::mat4 global_buffer_size[2];
    global_buffer = Buffer{"global_ubo", vk::BufferUsageFlagBits::eUniformBuffer, true, std::as_bytes(std::span{global_buffer_size})};
    global_set.update(0, 0, {{&global_buffer}, {&gi_settings_buffer}, {&light_settings_buffer}});

    #if 1
    const auto res_voxel_albedo = render_graph->add_resource(RGResource{"voxel_albedo", voxel_albedo});
    const auto res_voxel_normal = render_graph->add_resource(RGResource{"voxel_normal", voxel_normal});
    const auto res_voxel_radiance = render_graph->add_resource(RGResource{"voxel_radiance", voxel_radiance});
    const auto res_color_attachment = render_graph->add_resource(RGResource{"color_attachment", Texture{}});
    const auto res_depth_attachment = render_graph->add_resource(RGResource{"depth_attachment", depth_texture});
    const auto res_radiance_sampler = render_graph->add_resource(RGResource{"radiance_sampler", radiance_sampler});

    static bool once = true;

    const auto create_clear_pass = [&](RgResourceHandle resource) {
        RenderPass pass_clear;
        pass_clear
            .set_name(std::format("clear_{}", render_graph->get_resource(resource).name))
            .write_to_image(RPResource{
                resource,
                RGSyncStage::Transfer,
                TextureInfo{
                    .required_layout = RGImageLayout::TransferDst,
                    .range = {0, vk::RemainingMipLevels, 0, vk::RemainingArrayLayers}
                }
            })
            .set_draw_func([rg = render_graph, resource](vk::CommandBuffer cmd) {
                if(!once) { return; }
                const auto& r = rg->get_resource(resource);
                cmd.clearColorImage(std::get<1>(r.resource).first.image, vk::ImageLayout::eTransferDstOptimal, vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0, vk::RemainingArrayLayers});
            });
        return pass_clear;
    };

    render_graph->add_render_pass(create_clear_pass(res_voxel_albedo));
    render_graph->add_render_pass(create_clear_pass(res_voxel_normal));
    render_graph->add_render_pass(create_clear_pass(res_voxel_radiance));

    RenderPass pass_voxelization;
    pass_voxelization
        .set_name("voxelization")
        .set_pipeline(&pp_voxelize)
        .set_rendering_extent(RenderPassRenderingExtent{
            .viewport = {0.0f, 0.0f, 256.0f, 256.0f, 0.0f, 1.0f},
            .scissor = {0, 0, 256, 256}
        })
        .write_to_image(RPResource{
            res_voxel_albedo,
            RGSyncStage::Fragment,
            TextureInfo{
                .required_layout = RGImageLayout::General,
            }
        })
        .write_to_image(RPResource{
            res_voxel_normal,
            RGSyncStage::Fragment,
            TextureInfo{
                .required_layout = RGImageLayout::General,
            }
        })
        .set_draw_func([this](vk::CommandBuffer cmd) {
            if(!once) { return; }
            render_scene.render(cmd);
        });
    render_graph->add_render_pass(pass_voxelization);

    RenderPass pass_radiance_inject;
    pass_radiance_inject
        .set_name("radiance_inject")
        .set_pipeline(&pp_merge_voxels)
        .set_rendering_extent(RenderPassRenderingExtent{
            .viewport = {0.0f, 0.0f, 256.0f, 256.0f, 0.0f, 1.0f},
            .scissor = {0, 0, 256, 256}
        })
        .read_from_image(RPResource{
            res_voxel_albedo, 
            RGSyncStage::Compute,
            TextureInfo{
                .required_layout = RGImageLayout::General,
                .mutable_format = RGImageFormat::RGBA8Unorm
            }
        })
        .read_from_image(RPResource{
            res_voxel_normal, 
            RGSyncStage::Compute,
            TextureInfo{
                .required_layout = RGImageLayout::General,
                .mutable_format = RGImageFormat::RGBA8Unorm
            }
        })
        .write_to_image(RPResource{
            res_voxel_radiance, 
            RGSyncStage::Compute,
            TextureInfo{
                .required_layout = RGImageLayout::General,
            }
        })
        .set_sampler(res_radiance_sampler) 
        .set_draw_func([this](vk::CommandBuffer cmd) {
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pp_merge_voxels.layout.layout, 0, global_set.set, {});
            cmd.dispatch(256/8, 256/8, 256/8);
        });
    render_graph->add_render_pass(pass_radiance_inject);

    for(u32 i=1; i<voxel_radiance->mips; ++i) {
        RenderPass mip_pass;
        mip_pass
            .set_name(std::format("radiance_mip_{}", i))
            .read_from_image(RPResource{
                res_voxel_radiance, 
                RGSyncStage::Transfer,
                TextureInfo{
                    .required_layout = RGImageLayout::General,
                    .range = {i-1, 1, 0, 1}
                }
            })
            .write_to_image(RPResource{
                res_voxel_radiance, 
                RGSyncStage::Transfer,
                TextureInfo{
                    .required_layout = RGImageLayout::General,
                    .range = {i, 1, 0, 1}
                }
            })
            .set_draw_func([&, i](vk::CommandBuffer cmd) {
                if(!once) { return; }
                i32 mip_size = 256u >> i;
                cmd.blitImage(voxel_radiance->image, vk::ImageLayout::eGeneral,
                    voxel_radiance->image, vk::ImageLayout::eGeneral,
                    vk::ImageBlit{
                        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i-1, 0, 1},
                        { vk::Offset3D{}, vk::Offset3D{mip_size<<1, mip_size<<1, mip_size<<1} },
                        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i, 0, 1},
                        { vk::Offset3D{}, vk::Offset3D{mip_size, mip_size, mip_size} },
                    },
                    vk::Filter::eLinear);
            });
        render_graph->add_render_pass(mip_pass);
    }
        
    RenderPass pass_default_lit;
    pass_default_lit
        .set_name("default_lit")
        .set_pipeline(&pp_default_lit)
        .set_rendering_extent(RenderPassRenderingExtent{
            .viewport = {0.0f, 768.0f, 1024.0f, -768.0f, 0.0f, 1.0f},
            .scissor = {0, 0, 1024, 768}
        })
        .read_from_image(RPResource{
            res_voxel_radiance, 
            RGSyncStage::Fragment,
            TextureInfo{
                .required_layout = RGImageLayout::General,
                .range = {0, voxel_radiance->mips, 0, 1}
            }
        })
        .write_color_attachment(RPResource{
            res_color_attachment,
            RGSyncStage::ColorAttachmentOutput,
            TextureInfo{
                .required_layout = RGImageLayout::Attachment
            }
        })
        .write_depth_attachment(RPResource{
            res_depth_attachment,
            RGSyncStage::EarlyFragment,
            TextureInfo{
                .required_layout = RGImageLayout::Attachment
            }
        })
        .set_draw_func([&](vk::CommandBuffer cmd) {
            render_scene.render(cmd);
        });
    render_graph->add_render_pass(pass_default_lit);

    RenderPass pass_imgui;
    pass_imgui
        .set_name("pass imgui")
        .set_pipeline(&pp_imgui)
        .set_rendering_extent(RenderPassRenderingExtent{
            .viewport = {0.0f, 768.0f, 1024.0f, -768.0f, 0.0f, 1.0f},
            .scissor = {0, 0, 1024, 768}
        })
        .write_color_attachment(RPResource{
            res_color_attachment,
            RGSyncStage::ColorAttachmentOutput,
            TextureInfo{
                .required_layout = RGImageLayout::Attachment
            },
            RGAttachmentLoadStoreOp::Load
        })
        .set_draw_func([&](vk::CommandBuffer cmd) {
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::Begin("asd");
            if(ImGui::CollapsingHeader("gi_settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("voxel_resolution",                 &gi_settings->voxel_resolution, 1.0f, 512.0f);
                ImGui::SliderFloat("voxel_area",                       &gi_settings->voxel_area, 0.0f, 512.0f);
                ImGui::SliderFloat("voxel_size",                       &gi_settings->voxel_size, 0.0f, 100.0f);
                ImGui::SliderFloat("trace_distance",                   &gi_settings->trace_distance, 0.0f, 512.0f);
                ImGui::SliderFloat("diffuse_cone_aperture",            &gi_settings->diffuse_cone_aperture, 0.0f, glm::pi<float>());
                ImGui::SliderFloat("specular_cone_aperture",           &gi_settings->specular_cone_aperture, 0.0f, glm::pi<float>());
                ImGui::SliderFloat("occlusion_cone_aperture",          &gi_settings->occlusion_cone_aperture, 0.001f, glm::pi<float>());
                ImGui::SliderFloat("aoalpha",          &gi_settings->aoAlpha, 0.0f, 2.0f);
                ImGui::SliderFloat("aofallof",          &gi_settings->aoFalloff, 0.0f, 1000.0f);
                ImGui::SliderFloat("traceshadowhit",          &gi_settings->traceShadowHit, 0.0f, 100.0f);
                ImGui::Checkbox("merge_voxels_calc_occlusion",         reinterpret_cast<bool*>(&gi_settings->merge_voxels_calc_occlusion));
                ImGui::Checkbox("lighting_use_merge_voxels_occlusion", reinterpret_cast<bool*>(&gi_settings->lighting_use_merge_voxels_occlusion));
                ImGui::Checkbox("lighting_calc_occlusion",             reinterpret_cast<bool*>(&gi_settings->lighting_calc_occlusion));
            }   
            if(ImGui::CollapsingHeader("light_settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                for(u32 i=0; i<light_settings->num_dirs; ++i) {
                    const auto dir = std::format("dir_light_{}", i+1);
                    const auto col = std::format("col_light_{}", i+1);
                    const auto str = std::format("str_ligh_{}", i+1);
                    ImGui::PushID(i);
                    if(ImGui::SliderFloat3(dir.c_str(), &light_settings->dirs[i].dir.x, -1.0f, 1.0f)) {
                        light_settings->dirs[i].dir = glm::normalize(light_settings->dirs[i].dir);
                    }
                    ImGui::ColorEdit3(col.c_str(), &light_settings->dirs[i].col.x);
                    ImGui::SliderFloat(str.c_str(), &light_settings->dirs[i].col.w, 0.0f, 10.0f);
                    ImGui::PopID();
                }
            }
            if(ImGui::CollapsingHeader("Other", ImGuiTreeNodeFlags_DefaultOpen)) {
                if(ImGui::Button("Recompile shaders")) {
                    initialize_render_passes();
                }
            }
            ImGui::End();

            ImGui::Render();
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), get_frame_res().cmd);
        });
    render_graph->add_render_pass(pass_imgui);

    RenderPass pass_presentation;
    pass_presentation
        .set_name("presentation")
        .read_color_attachment(RPResource{
            res_color_attachment,
            RGSyncStage::AllGraphics,
            TextureInfo{
                .required_layout = RGImageLayout::PresentSrc
            }
        });
    render_graph->add_render_pass(pass_presentation);
    render_graph->bake_graph();
    #endif

    return true;
}