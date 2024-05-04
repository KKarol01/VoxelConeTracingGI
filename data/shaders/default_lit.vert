#version 460 core

layout(location=0) in vec3 pos;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;
layout(location=3) in vec2 uv;
layout(location=4) in vec3 tangent;

layout(location=0) out VS_OUT {
    vec3 frag_pos;
    vec3 frag_normal;
    vec3 frag_color;
    vec2 frag_uv;
    vec3 frag_tan;
    vec3 frag_bitan;
    mat3 frag_tbn;
    flat int frag_instance_index;
};

#include "global_set"

void main() {
    frag_pos = pos;
    frag_normal = normal;
    frag_color = color;
    frag_uv = uv;
    frag_tan = tangent;
    frag_bitan = cross(normal, tangent);
    frag_tbn = mat3(frag_tan, frag_bitan, frag_normal);
    frag_instance_index = gl_InstanceIndex;

    gl_Position = P * V * vec4(frag_pos, 1.0);
}