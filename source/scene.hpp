#pragma once

#include "types.hpp"
#include "renderer_types.hpp"
#include <glm/glm.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <optional>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

struct Material {
    Texture2D diffuse_texture{};
    Texture2D normal_texture{};
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
    Material material;
};

struct Model : public Handle<Model> {
    std::vector<Mesh> meshes;
};

struct SceneModel : public Handle<SceneModel> {
    constexpr SceneModel() = default;
    SceneModel(const std::string& name, Handle<Model> model): Handle(HandleGenerate), name(name), model(model) {}
    std::string name;
    Handle<Model> model;
};

class Scene {
public:
    Handle<Model> load_model(const std::filesystem::path& path);
    Handle<SceneModel> add_model(const std::string& name, Handle<Model> model);
    Model& get_model(Handle<Model> model);
    
private:
    friend class Renderer;

    std::vector<Model> models;
    std::vector<SceneModel> scene_models;
    std::unordered_map<std::filesystem::path, Texture2D> material_textures;
};