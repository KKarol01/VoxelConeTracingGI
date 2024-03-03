#pragma once

class Renderer;
struct Input;

struct Context {
    Renderer* renderer;
    Input* input;
};

Context& get_context();