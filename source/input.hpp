#pragma once

#include "types.hpp"
#include <unordered_set>
#include <unordered_map>
#include <glm/glm.hpp>

struct Input {
    void glfw_key_callback(int key, int action);
    bool key_down(int key) const;
    int get_key_state(int key) const;
    bool key_just_pressed(int key) const;
    void update();
    void set_cursor(int mode);
    void set_cursor_pos(f64 x, f64 y);

    std::unordered_map<int, int> key_action;
    std::unordered_set<int> pressed_keys;
};

struct Camera {
    glm::mat4 update();

    glm::mat4 view;
    f32 unlocked = 0.0f;
    f32 px{0.0f}, py{0.0f};
    f32 yaw{}, pitch{};
    glm::vec3 pos{-0.09, 0.27, 0.04};
    glm::vec3 speed{0.0f};
    static inline float acceleration = 0.0002f;
    static inline float max_speed = 0.01f;
};
