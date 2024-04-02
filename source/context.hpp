#pragma once

class Scene;
class Camera;
class Renderer;
struct Input;

struct Context {
    Scene* scene;
    Camera* camera;
    Renderer* renderer;
    Input* input;
};

Context& get_context();