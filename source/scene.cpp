#include "scene.hpp"

#include <spdlog/spdlog.h>
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <stb/stb_image.h>

static Texture2D* get_asset_texture(const fastgltf::Asset* asset, u64 texture_index, std::unordered_map<std::filesystem::path, Texture2D>* cache) {
    const auto& texture = asset->textures.at(texture_index);
    const auto& image = asset->images.at(texture.imageIndex.value()); // watch out for this, should further extensions be enabled.

    if(auto tex = cache->find(image.name); tex != cache->end()) {
        return &tex->second;
    } 

    return std::visit(fastgltf::visitor{
        [](auto&args) -> Texture2D* { return nullptr; },
        [&cache = *cache, &name = image.name](const fastgltf::sources::Array& vec) {
            int x, y, ch;
            auto image = reinterpret_cast<std::byte*>(stbi_load_from_memory(vec.bytes.data(), vec.bytes.size(), &x, &y, &ch, 4));
            auto shared_memory = std::make_shared<std::vector<std::byte>>(image, image + x*y*4);
            return &cache.insert({
                name,
                Texture2D{(u32)x, (u32)y, vk::Format::eR8G8B8A8Unorm, (u32)std::log2f(std::min(x, y)) + 1, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc, shared_memory}
            }).first->second;
        }
    }, image.data);
}

bool Scene::add_model(const std::string& name, const std::filesystem::path& path) {
    fastgltf::Parser parser;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(path);

    static constexpr auto options = 
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::LoadExternalBuffers |
        fastgltf::Options::LoadExternalImages;

    auto asset = parser.loadGltf(&data, path.parent_path(), options);
    if(auto error = asset.error(); error != fastgltf::Error::None) {
        spdlog::error("fastgltf: Unable to load file: {}", fastgltf::getErrorMessage(error));
        return false;
    }

    std::stack<const fastgltf::Node*> node_stack;
    for(auto node : asset->scenes[0].nodeIndices) {
        node_stack.emplace(&asset->nodes[node]);
    }

    Model model;

    while(!node_stack.empty()) {
        auto node = node_stack.top();
        node_stack.pop();

        for(auto node : node->children) {
            node_stack.emplace(&asset->nodes[node]);
        }

        if(!node->meshIndex.has_value()) { continue; }
        const auto& fgmesh = asset->meshes[node->meshIndex.value()];

        for(const auto& primitive : fgmesh.primitives) {
            if(!primitive.indicesAccessor.has_value()) {
                spdlog::error("Primitive in mesh {} has no indices. Skipping.", fgmesh.name);
                continue;
            }

            Mesh mesh;
            mesh.name = fgmesh.name;

            auto& positions = asset->accessors[primitive.findAttribute("POSITION")->second];
            auto& normals = asset->accessors[primitive.findAttribute("NORMAL")->second];
            auto& indices = asset->accessors[primitive.indicesAccessor.value()];
            auto initial_index = mesh.vertices.size();

            mesh.vertices.resize(mesh.vertices.size() + positions.count);
            fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), positions, [&](glm::vec3 vec, size_t idx) {
                mesh.vertices[initial_index + idx].position = vec; 
            });
            fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), normals, [&](glm::vec3 vec, size_t idx) {
                mesh.vertices[initial_index + idx].normal = vec; 
            });
            if(primitive.findAttribute("COLOR_0") != primitive.attributes.end()) {
                auto& colors = asset->accessors[primitive.findAttribute("COLOR_0")->second];
                fastgltf::iterateAccessorWithIndex<glm::vec4>(asset.get(), colors, [&](glm::vec4 vec, size_t idx) {
                    mesh.vertices[initial_index + idx].color = glm::vec3{vec.x, vec.y, vec.z}; 
                });
            } else {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), positions, [&](glm::vec3 vec, size_t idx) {
                    mesh.vertices[initial_index + idx].color = glm::vec3{1.0};
                });
            }

            initial_index = mesh.indices.size();
            mesh.indices.resize(mesh.indices.size() + indices.count);
            fastgltf::iterateAccessorWithIndex<u32>(asset.get(), indices, [&mesh, initial_index](u32 index, u32 idx) {
                mesh.indices[initial_index + idx] = index;
            });

            model.meshes.push_back(std::move(mesh));

            if(!primitive.materialIndex.has_value()) { continue; }

            const auto& material = asset->materials[primitive.materialIndex.value()];
            if(material.pbrData.baseColorTexture.has_value()) {
                mesh.material.diffuse_texture = get_asset_texture(&asset.get(), material.pbrData.baseColorTexture->textureIndex, &material_textures);
            }
            if(material.normalTexture.has_value()) {
                mesh.material.normal_texture = get_asset_texture(&asset.get(), material.normalTexture->textureIndex, &material_textures);
            }
        }
    }

    models.push_back(std::move(model));

    return true;
}