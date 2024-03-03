#version 460 core
#pragma shader_stage(fragment)

// #define MAX_LIGHTS 1
// #define PI 3.14159265359

#define DIST_FACTOR 1.1 /* Distance is multiplied by this when calculating attenuation. */


layout(location=0) in FS_IN {
	vec2 uv;
	vec3 normal;
	vec3 color;
	vec3 position;
	vec3 position_proj;
	mat3 TBN;
} frag;

#include "global_set"

// struct Material {
// 	vec3 diffuseColor;
// 	vec3 specularColor;
// 	float diffuseReflectivity;
// 	float specularReflectivity;
// 	float emissivity;
// 	float transparency;
// };

// uniform Material material;
// uniform bool use_diffuse_texture;

// uniform PointLight pointLights[MAX_LIGHTS];
// PointLight point_light = pointLights[0];
// uniform int numberOfLights;
// uniform int light_intensity;
// uniform vec3 lightPos;
// uniform float far_plane;

// uniform vec3 camera_pos;
// uniform int level;
// uniform int resolution;

// layout(binding=0) uniform sampler2D tex_diffuse;
// layout(binding=1) uniform sampler2D tex_normal;
layout(set=1, binding=0, r32ui) uniform coherent volatile uimage3D voxel_albedo;
layout(set=1, binding=1, r32ui) uniform coherent volatile uimage3D voxel_normal;
layout(set=1, binding=2) uniform sampler voxel_sampler;

layout(set=2, binding=1) uniform sampler diff_samp;
layout(set=2, binding=2) uniform texture2D tex_diffuse;
layout(set=2, binding=3) uniform texture2D tex_normal;
// layout(binding=3) uniform samplerCube tex_shadow;

// float attenuate(float, in const PointLight);
// vec3 calculatePointLight(const PointLight light){
// 	const vec3 direction = normalize(light.position - geom_position);
// 	const float distanceToLight = distance(light.position, geom_position);
// 	const float attenuation = attenuate(distanceToLight, light);
// 	const float d = max(dot(normalize(geom_normal), direction), 0.0);
// 	return d * light_intensity * attenuation * light.color;
// };

vec4 convRGBA8ToVec4(uint val){
	return vec4( 
		float((val & 0x000000ff) >> 0),
		float((val & 0x0000ff00) >> 8),
		float((val & 0x00ff0000) >> 16),
		float((val & 0xff000000) >> 24)
	);
}

uint convVec4ToRGBA8(vec4 val) {
	return (
		(uint(val.w) & 0xff) << 24 |
		(uint(val.z) & 0xff) << 16 |
		(uint(val.y) & 0xff) << 8  |
		(uint(val.x) & 0xff)
	);
}

vec3 scaleAndBias(vec3 p) { return 0.5 * p + vec3(0.5); }
// vec3 scaleAndBias(vec3 p) { return p; }
// bool isInsideCube(const vec3 p, float e) { return abs(p.x) < 1 + e && abs(p.y) < 1 + e && abs(p.z) < 1 + e; }

void imageAtomicRGBA8Avg(int grid_sel, ivec3 coords, vec4 val) {
	val.rgb *= 255.0;
	uint newVal = convVec4ToRGBA8(val), prevStoredVal = 0, curStoredVal;
	int iterations = 0;

	// loop until moving average is saved for this thread
	// if this thread is the first one, then voxels[coords] == 0 == prevstoredval,
	// so newval is assigned, and the loop never runs
	// if not, then curval = voxel[coords], prev = cur
	// do moving average with count in w component
	// store in new val
	// then next iteration hopefully will be the next one amongst other threads, and
	// curstoredval will be equal to prevstored val, so that new val can be saved
	if(grid_sel == 0) {
		while((curStoredVal = imageAtomicCompSwap(voxel_albedo, coords, prevStoredVal, newVal)) != prevStoredVal && iterations < 255) {
			prevStoredVal = curStoredVal;
			vec4 rval = convRGBA8ToVec4(curStoredVal);
			rval.xyz = (rval.xyz * rval.w);
			vec4 curValF = rval + val;
			curValF.xyz /= curValF.w;
			newVal = convVec4ToRGBA8(curValF);
			++iterations;
		}
	} else if(grid_sel == 1) {
		while((curStoredVal = imageAtomicCompSwap(voxel_normal, coords, prevStoredVal, newVal)) != prevStoredVal && iterations < 255) {
			prevStoredVal = curStoredVal;
			vec4 rval = convRGBA8ToVec4(curStoredVal);
			rval.xyz = (rval.xyz * rval.w);
			vec4 curValF = rval + val;
			curValF.xyz /= curValF.w;
			newVal = convVec4ToRGBA8(curValF);
			++iterations;
		}
	}
}

// float calc_shadow() {
//     vec3 dir = geom_position - lightPos;
//     float cd = length(dir);
//     float rd = texture(tex_shadow, dir).r * far_plane;
//     return cd - 0.0001 > rd ? 1.0 : 0.0;
// }

vec4 sample_diffuse_map() {
	return texture(sampler2D(tex_diffuse, diff_samp), frag.uv).rgba;
}
vec3 sample_normal_map() {
	vec3 nrm = texture(sampler2D(tex_normal, diff_samp), frag.uv).rgb;
	nrm = nrm * 2.0 - 1.0;
	nrm = normalize(frag.TBN * nrm);
	return nrm;
}

void main() {
	vec3 p = abs(frag.position_proj);
	if(max(max(p.x, p.y), p.z) > 1.0) { discard; }
	
	// vec3 color = vec3(0.0);
	// const uint maxLights = min(numberOfLights, MAX_LIGHTS);
	// for(uint i = 0; i < maxLights; ++i) { 
	// 	color += calculatePointLight(pointLights[i]); 
	// }
	// // vec3 spec = material.specularReflectivity * material.specularColor;
	// vec3 diff = mix(geom_color, texture(diffuse, geom_uv).rgb, float(use_diffuse_texture)) * material.diffuseReflectivity;

	// color = (diff) * color + clamp(material.emissivity, 0, 1) * geom_color;
	// float alpha = pow(1 - material.transparency, 4); // For soft shadows to work better with transparent materials.
	// vec4 res = 1.0 * vec4(vec3(geom_color * color), 1);
	// res = clamp(res, 0.0, 1.0);
	
	// // Color done, now save it.

	vec3 voxel = scaleAndBias(frag.position_proj);
	ivec3 pos = ivec3(256 * voxel);
	// // pos.y += (resolution + 2) * level;
	// // imageAtomicRGBA8Avg(pos, res);
	vec4 frag_diff_rgba = sample_diffuse_map();
	vec3 frag_diff = frag_diff_rgba.rgb;
	float opacity = frag_diff_rgba.a;
	vec3 frag_nrm = sample_normal_map();

	if(opacity < 1e-6) { return; }

	frag_diff.rgb *= opacity;
	frag_nrm = frag_nrm*0.5+0.5;

	imageAtomicRGBA8Avg(0, pos, vec4(frag_diff, 1.0));
	imageAtomicRGBA8Avg(1, pos, vec4(frag_nrm, 1.0));
}