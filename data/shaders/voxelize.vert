#version 460 core
#pragma shader_stage(vertex)

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec3 tangent;

layout(set=0, binding=0) uniform GlobalUBO {
	mat4 P;
	mat4 V;
	vec3 cam_pos;
	uint num_point_lights;
};
layout(set=2, binding=0) readonly buffer ModelsSSBO {
    mat4 Models[];
};

layout(location = 0) out VS_OUT {
	vec3 position;
	vec3 position_proj;
	vec3 normal;
	vec3 color;
	vec2 uv;
	mat3 TBN;
} vert;

void main(){
	mat4 M = Models[gl_InstanceIndex];
	vec4 pos = M * vec4(position, 1.0);
	vert.position = pos.xyz;
	vert.uv = uv;
	vert.normal = (M * vec4(normal, 0.0)).xyz;
	vert.color = color;
	vec3 _tang = vec3((M * vec4(tangent,0.0)));
	vert.TBN = mat3(_tang, cross(normal, _tang), normal);

	vert.position_proj = pos.xyz;
	gl_Position = pos;
}