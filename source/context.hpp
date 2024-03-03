#pragma once

class Renderer;

struct Context {
    Renderer* renderer;
};

Context& get_context();