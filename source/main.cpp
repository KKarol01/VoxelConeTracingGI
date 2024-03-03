#include "types.hpp"
#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp>
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <vk_mem_alloc.h>
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <stb/stb_include.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <format>
#include <array>
#include <stack>
#include <cstdio>



struct GpuBuffer {
    vk::Buffer buffer;
    void* data;
    u64 size;
    VmaAllocation alloc;
};

struct Window {
    u32 width{1024}, height{768};
    GLFWwindow* window{nullptr};
};

struct FrameResources {
    vk::CommandPool pool;
    vk::CommandBuffer cmd;
    vk::Semaphore swapchain_semaphore, rendering_semaphore;
    vk::Fence in_flight_fence;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
};

struct Model {
    std::vector<Mesh> meshes;
};

struct GpuMesh {
    const Mesh* mesh;
    u32 vertex_offset, vertex_count;
    u32 index_offset, index_count;
};

struct Scene {
    std::vector<GpuMesh> models;
    GpuBuffer* vertex_buffer;
    GpuBuffer* index_buffer;
};

template<typename VkObject> struct VulkanObjectType;
template<> struct VulkanObjectType<vk::SwapchainKHR> { static inline constexpr vk::ObjectType type = vk::ObjectType::eSwapchainKHR; };
template<> struct VulkanObjectType<vk::CommandPool> { static inline constexpr vk::ObjectType type = vk::ObjectType::eCommandPool; };
template<> struct VulkanObjectType<vk::CommandBuffer> { static inline constexpr vk::ObjectType type = vk::ObjectType::eCommandBuffer; };
template<> struct VulkanObjectType<vk::Fence> { static inline constexpr vk::ObjectType type = vk::ObjectType::eFence; };
template<> struct VulkanObjectType<vk::Semaphore> { static inline constexpr vk::ObjectType type = vk::ObjectType::eSemaphore; };
template<> struct VulkanObjectType<vk::Buffer> { static inline constexpr vk::ObjectType type = vk::ObjectType::eBuffer; };

template<typename T> void set_debug_name(vk::Device device, T object, std::string_view name);

class Renderer {
public:
    bool initialize() {
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

    bool load_model_from_file(std::string_view name, const std::filesystem::path& path) {
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

    void setup_scene() {
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

private:
    bool initialize_vulkan() {
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

    bool initialize_swapchain() {
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

    bool initialize_frame_resources() {
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

    bool initialize_render_passes() {
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
        }



        return true;
    }

    template<typename T> GpuBuffer* create_buffer(std::string_view label, vk::BufferUsageFlags usage, std::span<T> data) {
        vk::BufferCreateInfo buffer_info{
            {},
            (VkDeviceSize)data.size_bytes(),
            usage
        };
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer buffer;
        VmaAllocation alloc;
        VmaAllocationInfo alloc_i;
        vmaCreateBuffer(vma, (VkBufferCreateInfo*)&buffer_info, &alloc_info, &buffer, &alloc, &alloc_i);

        memcpy(alloc_i.pMappedData, data.data(), data.size_bytes());

        buffers.push_back(new GpuBuffer{
            .buffer = buffer,
            .data = alloc_i.pMappedData,
            .size = data.size_bytes(),
            .alloc = alloc
        });
        set_debug_name(device, buffers.back()->buffer, label);

        return buffers.back();
    }

    FrameResources& get_frame_res() { return frames.at(frame_number % FRAMES_IN_FLIGHT); }

    std::vector<u32> compile_shader(std::string_view filename, std::string_view file) {
        shaderc::Compiler compiler;
        auto result = compiler.CompileGlslToSpv(file.data(), shaderc_glsl_infer_from_source, filename.data());
        if(result.GetCompilationStatus() != shaderc_compilation_status_success) {
            spdlog::error("Shader compilation error: {}", result.GetErrorMessage().c_str());
            return {};
        }

        return std::vector<u32>{result.begin(), result.end()};
    }

public:
    static constexpr inline u32 FRAMES_IN_FLIGHT = 2;

    u32 frame_number{0};
    u32 window_width{1024}, window_height{768};
    GLFWwindow *window{nullptr};
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::DispatchLoaderDynamic dynamic_loader;
    u32 graphics_queue_idx, presentation_queue_idx;
    vk::Queue graphics_queue, presentation_queue;
    VmaAllocator vma;
    vk::SwapchainKHR swapchain;
    std::vector<vk::Image> swapchain_images;
    std::vector<vk::ImageView> swapchain_views;
    vk::Format swapchain_format;
    std::array<FrameResources, FRAMES_IN_FLIGHT> frames{};
    std::vector<GpuBuffer*> buffers;
    
    std::unordered_map<std::string, Model> models;
    Scene scene;
};

Renderer r;

template<typename T> void set_debug_name(vk::Device device, T object, std::string_view name) {
    device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        VulkanObjectType<T>::type, (u64)static_cast<T::NativeType>(object), name.data()
    }, r.dynamic_loader);
}

int main() {
    spdlog::set_level(spdlog::level::debug);

    if(!r.initialize()) {
        return -1;
    }

    r.load_model_from_file("gi_box", "data/models/gi_box.gltf");    
    r.setup_scene();
    
}