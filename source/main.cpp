#include "types.hpp"
#include <vulkan/vulkan.hpp>
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <vk_mem_alloc.h>
#include <vector>
#include <string>
#include <format>
#include <array>

struct Window {
    u32 width{1024}, height{768};
    GLFWwindow *window{nullptr};
};

struct FrameResources {
    vk::CommandPool pool;
    vk::CommandBuffer cmd;
    vk::Semaphore swapchain_semaphore, rendering_semaphore;
    vk::Fence in_flight_fence;
};

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

        return true;
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
        }

        return true;
    }

public:
    static constexpr inline u32 FRAMES_IN_FLIGHT = 2;

    u32 window_width{1024}, window_height{768};
    GLFWwindow *window{nullptr};
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    u32 graphics_queue_idx, presentation_queue_idx;
    vk::Queue graphics_queue, presentation_queue;
    VmaAllocator vma;
    vk::SwapchainKHR swapchain;
    std::vector<vk::Image> swapchain_images;
    std::vector<vk::ImageView> swapchain_views;
    vk::Format swapchain_format;
    std::array<FrameResources, FRAMES_IN_FLIGHT> frames{};
};

Renderer r;

int main() {
    if(!r.initialize()) {
        return -1;
    }

    
}