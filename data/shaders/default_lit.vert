#version 460 core
#pragma shader_stage(vertex)

layout(location=0) in vec3 pos;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;
// layout(location=3) in vec2 uv;
// layout(location=4) in vec3 tangent;

layout(location=0) out VS_OUT {
    // vec2 frag_uv;
    // mat3 frag_TBN;
    vec3 frag_pos;
    vec3 frag_normal;
    vec3 frag_color;
};

#include "global_set"
// layout(set=2, binding=0) readonly buffer ModelsSSBO {
//     mat4 M[];
// };

void main() {
    // frag_uv = uv;
    //gl_InstanceIndex;
    frag_pos = pos;
    frag_normal = normal;
    frag_color = color;

    // vec3 bitangent = cross(normal, tangent);
    // frag_TBN = mat3(tangent, bitangent, normal);

    gl_Position = P * V * vec4(frag_pos, 1.0);
}