#pragma once

#include "types.hpp"
#include "renderer_types.hpp"
#include <glm/glm.hpp>
#include <filesystem>
#include <string>
#include <vector>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

struct Material {
    Texture2D* diffuse_texture{};
    Texture2D* normal_texture{};
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
    Material material;
};

struct Model {
    std::string name;
    std::vector<Mesh> meshes;
};

class Scene {
public:
    bool add_model(const std::string& name, const std::filesystem::path& path);

private:
    friend class Renderer;

    DescriptorSet set;
    std::vector<Model> models;
    std::unordered_map<std::filesystem::path, Texture2D> material_textures;
};