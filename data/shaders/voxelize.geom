#version 460 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in GS_IN {
	vec3 position;
	vec3 position_proj;
	vec3 normal;
	vec3 color;
	vec2 uv;
	mat3 TBN;
    flat uint instance_index;
} vert[];

layout(location=0) out GS_OUT {
	vec3 position;
	vec3 position_proj;
	vec3 normal;
	vec3 color;
	vec2 uv;
	mat3 TBN;
    flat uint instance_index;
} geom;

layout(set=0, binding=0) uniform GlobalUBO {
	mat4 P;
	mat4 V;
};

void main(){
	const vec3 p1 = vert[1].position_proj - vert[0].position_proj;
	const vec3 p2 = vert[2].position_proj - vert[0].position_proj;
	const vec3 p = abs(cross(p1, p2)); 
	for(uint i = 0; i < 3; ++i){
		geom.position = vert[i].position;
		geom.position_proj = vec3(vec4(vert[i].position_proj, 1.0));
		geom.normal = vert[i].normal;
		geom.color = vert[i].color;
		geom.uv = vert[i].uv;
		geom.TBN = vert[i].TBN;
		geom.instance_index = vert[i].instance_index;
		if(p.z > p.x && p.z > p.y){
			gl_Position = vec4(vert[i].position_proj.xy, 0, 1);
		} else if (p.x > p.y && p.x > p.z){
			gl_Position = vec4(vert[i].position_proj.yz, 0, 1);
		} else {
			gl_Position = vec4(vert[i].position_proj.xz, 0, 1);
		}
		EmitVertex();
	}
    EndPrimitive();
}