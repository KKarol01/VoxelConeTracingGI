#version 460 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec3 tangent;

#include "global_set"

layout(location = 0) out VS_OUT {
	vec3 position;
	vec3 position_proj;
	vec3 normal;
	vec3 color;
	vec2 uv;
	mat3 TBN;
    flat uint instance_index;
} vert;

void main(){
	vec4 pos = vec4(position, 1.0);
	vert.position = pos.xyz;
	vert.uv = uv;
	vert.normal = normal;
	vert.color = color;
	vert.TBN = mat3(tangent, cross(normal, tangent), normal);
    vert.instance_index = gl_InstanceIndex;
	vert.position_proj = pos.xyz;
	gl_Position = pos;
}