#include "types.hpp"
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
    ctx.renderer = new Renderer{};

    auto &r = *ctx.renderer;

    spdlog::set_level(spdlog::level::debug);

    if(!r.initialize()) {
        return -1;
    }

    r.load_model_from_file("gi_box", "data/models/gi_box.gltf");    
    r.setup_scene();
    
}