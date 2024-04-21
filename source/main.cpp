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

#include <tracy/Tracy.hpp>

int main() {
    
    auto& ctx = get_context();
    ctx.scene = new Scene{};
    ctx.camera = new Camera{};
    ctx.input = new Input{};
    ctx.renderer = new Renderer{};

    auto &r = *ctx.renderer;

    spdlog::set_level(spdlog::level::debug);

    if(!r.initialize()) {
        return -1;
    }

    // const auto gi_box = ctx.scene->load_model("data/models/gi_box.gltf");    
    const auto sponza = ctx.scene->load_model("data/models/Sponza.gltf");    
    // ctx.scene->add_model("gi_box", gi_box);
    ctx.scene->add_model("sponza", sponza);

    glfwSetKeyCallback(r.window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        get_context().input->glfw_key_callback(key, action);
    });

    while(!glfwWindowShouldClose(r.window)) {
        FrameMarkNamed("main");

        r.render();

        ctx.camera->update();
        get_context().input->update();
        glfwPollEvents();
    }
}