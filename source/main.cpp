#include "types.hpp"
#include "input.hpp"
#include "context.hpp"
#include "renderer.hpp"
#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp>
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <vk_mem_alloc.h>
#include <stb/stb_include.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <format>
#include <array>
#include <stack>
#include <cstdio>

int main() {
    auto& ctx = get_context();
    ctx.input = new Input{};
    ctx.renderer = new Renderer{};

    auto &r = *ctx.renderer;

    spdlog::set_level(spdlog::level::debug);

    if(!r.initialize()) {
        return -1;
    }

    r.load_model_from_file("gi_box", "data/models/gi_box.gltf");    
    r.setup_scene();

    glfwSetKeyCallback(r.window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        get_context().input->glfw_key_callback(key, action);
    });

    r.draw();
}