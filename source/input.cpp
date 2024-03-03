#include "input.hpp"
#include "renderer.hpp"
#include "context.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <GLFW/glfw3.h>

void Input::glfw_key_callback(int key, int action) {
    if(action == GLFW_PRESS && key_action[key] == GLFW_RELEASE) {
        pressed_keys.insert(key);
    }

    key_action[key] = action;
}

bool Input::key_down(int key) const { 
    if(!key_action.contains(key)) { return false; }
    return key_action.at(key) & (GLFW_PRESS | GLFW_REPEAT);
}

int Input::get_key_state(int key) const {
    if(!key_action.contains(key)) { return 0; }
    return key_action.at(key);
}

bool Input::key_just_pressed(int key) const {
    return pressed_keys.contains(key);
}

void Input::update() {
    pressed_keys.clear();
}

void Input::set_cursor(int mode) {
    glfwSetInputMode(get_context().renderer->window, GLFW_CURSOR, mode);
}

void Input::set_cursor_pos(f64 x, f64 y) {
    glfwSetCursorPos(get_context().renderer->window, x, y);
}

glm::mat4 Camera::update() {
    if(get_context().input->key_just_pressed(GLFW_KEY_TAB)) {
        unlocked = 1.0f - unlocked;
        if(unlocked > 0.0f) {
            get_context().input->set_cursor(GLFW_CURSOR_DISABLED);
        } else {
            get_context().input->set_cursor(GLFW_CURSOR_NORMAL);
            get_context().input->set_cursor_pos(get_context().renderer->window_width*0.5f, get_context().renderer->window_height*0.5f);
        }
    }
    
    double x, y;
    u32 ww = get_context().renderer->window_width;
    u32 wh = get_context().renderer->window_height;
    glfwGetCursorPos(get_context().renderer->window, &x, &y);
    x = ((f32)x/ww) * 2.0 - 1.0;
    y = ((f32)y/wh) * 2.0 - 1.0;
    f32 dx = unlocked * (-(x - px) / 1.0);
    f32 dy = unlocked * (-(y - py) / 1.0);
    px = x;
    py = y;
    
    yaw += dx * 80.0f;
    pitch = glm::clamp(pitch+dy * 80.0f, -89.0f, 89.0f);
    
    auto orientation = glm::angleAxis(glm::radians(yaw), glm::vec3{0,1,0});
    orientation *= glm::angleAxis(glm::radians(pitch), glm::vec3{1,0,0});
    
    auto forward = glm::normalize(orientation * glm::vec3{0.f, 0.f, -1.f});
    auto right   = glm::normalize(glm::cross(forward, {0,1,0}));

    if(get_context().input->key_down(GLFW_KEY_W)) { speed.z += 2.0f * acceleration; }
    if(get_context().input->key_down(GLFW_KEY_S)) { speed.z += 2.0f * -acceleration; }
    if(get_context().input->key_down(GLFW_KEY_A)) { speed.x += 2.0f * -acceleration; }
    if(get_context().input->key_down(GLFW_KEY_D)) { speed.x += 2.0f * acceleration; }
    if(get_context().input->key_down(GLFW_KEY_SPACE)) { speed.y += 2.0f * acceleration; }
    if(get_context().input->key_down(GLFW_KEY_LEFT_SHIFT)) { speed.y += 2.0f * -acceleration; }

    auto i = glm::vec3{glm::lessThan(glm::floor(glm::abs(speed) / glm::vec3{acceleration}), glm::vec3{1.0f})};
    speed -= glm::mix(glm::vec3{acceleration}, glm::abs(speed), i) * glm::sign(speed);
    speed = glm::min(speed, max_speed);

    pos += right   * speed.x * glm::vec3{1, 0, 1};
    pos += forward * speed.z * glm::vec3{1, 0, 1};
    pos.y += speed.y;
    // spdlog::info("CAMREA PS {} {} {}", pos.x, pos.y, pos.z);

    view = glm::lookAt(pos, pos + forward, {0,1,0});
    return view;
}