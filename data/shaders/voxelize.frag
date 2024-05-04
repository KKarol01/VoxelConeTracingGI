#version 460 core
#extension GL_EXT_nonuniform_qualifier : require

#define DIST_FACTOR 1.1 /* Distance is multiplied by this when calculating attenuation. */

layout(location=0) in FS_IN {
	vec3 position;
	vec3 position_proj;
	vec3 normal;
	vec3 color;
	vec2 uv;
	mat3 TBN;
    flat uint instance_index;
} frag;

#include "global_set"
#include "material_set"

layout(set=2, binding=0, r32ui) uniform coherent volatile uimage3D voxel_albedo;
layout(set=2, binding=1, r32ui) uniform coherent volatile uimage3D voxel_normal;

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
		(uint(val.x) & 0xff) << 0
	);
}

vec3 scaleAndBias(vec3 p) { return 0.5 * p + vec3(0.5); }

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

void main() {
	vec3 p = abs(frag.position_proj);
	if(max(max(p.x, p.y), p.z) > 1.0) { discard; }
	
	vec3 albedo = frag.color;
    vec3 normal = frag.normal;
    Material mat = materials[frag.instance_index];
    if(mat.diffuse_idx >= 0) {
        albedo = texture(material_textures[nonuniformEXT(mat.diffuse_idx)], frag.uv).rgb;
        normal = texture(material_textures[nonuniformEXT(mat.normal_idx)], frag.uv).rgb;
        normal = normalize(frag.TBN * (normal * 2.0 - 1.0));
    }

	vec3 voxel = frag.position * 0.5 + 0.5;
	ivec3 pos = ivec3(256.0 * voxel);
	vec4 frag_diff_rgba = vec4(albedo, 1.0);
	vec3 frag_diff = frag_diff_rgba.rgb;
	float opacity = frag_diff_rgba.a;
	vec3 frag_nrm = normal;

	if(opacity < 1e-6) { return; }

	frag_diff.rgb *= opacity;
	frag_nrm = frag_nrm*0.5+0.5;

	imageAtomicRGBA8Avg(0, pos, vec4(frag_diff, 1.0));
	imageAtomicRGBA8Avg(1, pos, vec4(frag_nrm, 1.0));
}