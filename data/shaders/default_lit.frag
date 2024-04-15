#version 460
#pragma shader_stage(fragment)

#define PI 3.14159265359
#define EPSILON 1e-8

layout(location=0) out vec4 outColor;

layout(location=0) in FS_IN {
    vec3 frag_pos;
    vec3 frag_normal;
    vec3 frag_color;
    flat int frag_instance_index;
};

#include "global_set"
layout(set=1, binding=0) uniform texture3D voxel_radiance;
layout(set=1, binding=1) uniform sampler voxel_sampler;
// #include "material_set"
// Material mat = materials[frag_instance_index];

// const vec3 DIFFUSE_CONE_DIRECTIONS_16[16] = {
//     vec3( 0.57735,   0.57735,   0.57735  ),
//     vec3( 0.57735,  -0.57735,  -0.57735  ),
//     vec3(-0.57735,   0.57735,  -0.57735  ),
//     vec3(-0.57735,  -0.57735,   0.57735  ),
//     vec3(-0.903007, -0.182696, -0.388844 ),
//     vec3(-0.903007,  0.182696,  0.388844 ),
//     vec3( 0.903007, -0.182696,  0.388844 ),
//     vec3( 0.903007,  0.182696, -0.388844 ),
//     vec3(-0.388844, -0.903007, -0.182696 ),
//     vec3( 0.388844, -0.903007,  0.182696 ),
//     vec3( 0.388844,  0.903007, -0.182696 ),
//     vec3(-0.388844,  0.903007,  0.182696 ),
//     vec3(-0.182696, -0.388844, -0.903007 ),
//     vec3( 0.182696,  0.388844, -0.903007 ),
//     vec3(-0.182696,  0.388844,  0.903007 ),
//     vec3( 0.182696, -0.388844,  0.903007 )
// };

// const vec3 diffuseConeDirections[] = {
//     vec3(0.0f, 1.0f, 0.0f),
//     vec3(0.0f, 0.5f, 0.866025f),
//     vec3(0.823639f, 0.5f, 0.267617f),
//     vec3(0.509037f, 0.5f, -0.7006629f),
//     vec3(-0.50937f, 0.5f, -0.7006629f),
//     vec3(-0.823639f, 0.5f, 0.267617f)
// };

// const float diffuseConeWeights[] = {
//     PI / 4.0f,
//     3.0f * PI / 20.0f,
//     3.0f * PI / 20.0f,
//     3.0f * PI / 20.0f,
//     3.0f * PI / 20.0f,
//     3.0f * PI / 20.0f,
// };

// float get_shadowing() {
//     float shadow = 0.0;
//     // vec2 texel_size = 1.0 / textureSize(samplerCube(cube_depth_map, sampler1), 0).xy;

//     // PointLight point_lights[1];
//     // point_lights[0].pos = vec3(0.0, 0.8, 0.0);
//     // point_lights[0].col = vec3(1.0);
//     // point_lights[0].att = vec3(0.3, 0.5, 0.8);
//     // PointLight pl = point_lights[0];
//     // for(int i=-1; i<=1; ++i) {
//     //     for(int j=-1; j<=1; ++j) {
//     //         vec3 coord = frag_pos - pl.pos;
//     //         coord.xy += vec2(i, j) * texel_size;
//     //         float d = texture(samplerCube(cube_depth_map, sampler1), coord).r;
//     //         shadow += d*25.0 > length(frag_pos - pl.pos)-0.01 ? 1.0 : 0.0;
//     //         return shadow;
//     //     }
//     // }
    
//     return 0.0;
// }

// vec4 SampleRadiance(vec3 pos, float mip) {
//     vec4 sample_ = textureLod(sampler3D(voxel_radiance, voxel_sampler), pos, mip).rgba;
//     if(mip < 1.0) {
//         sample_ = mix(vec4(frag_color, 1.0), sample_, clamp(mip, 0.0, 1.0));
//     }
//     return sample_;
// }

// float TraceShadowCone(vec3 position, vec3 direction, float aperture, float max_dist) {
//     const float voxel_size = 2.0 / 256.0;

//     float d = voxel_size;
//     const vec3 start_pos = position + direction * d;

//     float visibility = 0.0;
//     float k = exp2(7.0 * 0.5);

//     while(visibility < 1.0 && d < max_dist) {
//         const vec3 p = start_pos + direction * d;
//         const float diameter = 2.0 * tan(aperture / voxel_size) * d * 5.0;
//         const float mip = max(log2(diameter / voxel_size), 0.0);
//         const vec3 voxel_coord = vec3(p * 0.5 + 0.5);

//         vec4 sample_ = SampleRadiance(voxel_coord, mip);

//         visibility += (1.0 - visibility) * sample_.a * k;
//         if(visibility > 0.0) { return 1.0 - visibility; }
//         d += diameter;
//     }

//     return 1.0 - visibility;
// }

// vec3 calc_direct_light() {
//     vec3 ambient = mat.ambient_color.rgb * mat.ambient_color.a;
//     vec3 diffuse = vec3(0.0);
//     vec3 specular = vec3(0.0);
//     // vec3 frag_nrm = texture(sampler2D(tex_normal, sampler1), frag_uv).rgb;
//     vec3 frag_nrm = frag_normal;
//     // frag_nrm = normalize(frag_TBN * frag_nrm);
//     // vec3 tex_diff = texture(sampler2D(diffuseTexture, sampler1), frag_uv).rgb;
//     vec3 tex_diff = frag_color * mat.diffuse_color.rgb;

//     PointLight point_lights[1];
//     point_lights[0].pos = vec3(0.0, 0.65, 0.0);
//     point_lights[0].col = vec3(1.0);
//     point_lights[0].att = vec3(0.2, 0.4, 0.6);
//     int num_point_lights = 1;
//     vec3 cam_pos = vec3(V * vec4(0.0,0.0,0.0,1.0));
//     for(uint i=0; i<num_point_lights; ++i) {
//         PointLight pl = point_lights[i];
//         vec3 light_dir = pl.pos - frag_pos;
//         float dist = length(light_dir);
//         light_dir = normalize(light_dir);

//         vec3 v = normalize(cam_pos - frag_pos);
//         vec3 h = normalize(v + light_dir);
//         float dotNL = max(dot(frag_normal, light_dir), 0.0);
//         float dotNH = max(dot(frag_normal, h), 0.0);
//         float dotLH = max(dot(light_dir, h), 0.0);
//         float spec = pow(dotNH, mat.specular_color.a * 11.0 + 1.0);

//         float ldotp = max(dot(frag_nrm, light_dir), 0.0);
//         float att = 1.0 / (pl.att[0] + pl.att[1]*dist + pl.att[2]*dist*dist);

//         diffuse += tex_diff * att * pl.col * mat.diffuse_color.a;
//         specular += spec * dotNL * pl.col;
//     }

//     return ambient + diffuse + specular;
// }

// vec4 DiffuseCone(const vec3 origin, const vec3 dir) {
//     const float voxel_size = 2.0 / 256.0;
// 	float max_dist = 2.0;
// 	float current_dist = voxel_size;
// 	float apperture_angle = 0.25; // Angle in Radians.
// 	vec3 color = vec3(0.0);
// 	float occlusion = 0.0;

// 	while(current_dist < max_dist && occlusion < 1.0) {
// 		float current_coneDiameter = 2.0 * current_dist * tan(apperture_angle * 0.5);
// 		vec3 pos_worldspace = origin + dir * current_dist;

//         float vlevel = log2(current_coneDiameter / voxel_size); // Current mipmap level
//         vlevel = min(8.0, max(vlevel, 0.0));

//         vec3 pos_texturespace = (pos_worldspace + vec3(1.0)) * 0.5; // [-1,1] Coordinates to [0,1]
// 		vec4 voxel = textureLod(sampler3D(voxel_radiance, voxel_sampler), pos_texturespace, vlevel);	// Sample
// 		vec3 color_read = voxel.rgb;
// 		float occlusion_read = voxel.a;

//         color += (1.0 - occlusion) * color_read;
//         occlusion += (1.0 - occlusion) * occlusion_read / (1.0 + pow(current_coneDiameter, 16.0));

// 		current_dist += max(current_coneDiameter, voxel_size) * 0.8;
// 	}
// 	return vec4(color, occlusion);
// }

// vec4 specular_cone(const vec3 origin, const vec3 dir) {
//     const float voxel_size = 2.0 / 256.0;
// 	float max_dist = 2.0;
// 	float current_dist = voxel_size;
//     float apperture_angle = 0.08;
// 	vec3 color = vec3(0.0);
// 	float occlusion = 0.0;
//     PointLight point_lights[1];
//     point_lights[0].pos = vec3(0.0, 0.65, 0.0);
//     point_lights[0].col = vec3(1.0);
//     point_lights[0].att = vec3(0.2, 0.4, 0.6);

// 	while(current_dist < max_dist && occlusion < 1.0) {
// 		float current_coneDiameter = 2.0 * current_dist * tan(apperture_angle * 0.5);
// 		vec3 pos_worldspace = origin + dir * current_dist;

//         float vlevel = log2(current_coneDiameter / voxel_size); // Current mipmap level
//         vlevel = min(8.0, max(vlevel, 0.0));

//         vec3 pos_texturespace = (pos_worldspace + vec3(1.0)) * 0.5; // [-1,1] Coordinates to [0,1]
// 		vec4 voxel = textureLod(sampler3D(voxel_radiance, voxel_sampler), pos_texturespace, vlevel);	// Sample
// 		vec3 color_read = voxel.rgb;
// 		float occlusion_read = voxel.a;

//         color += (1.0 - occlusion) * color_read;
//         occlusion += (1.0 - occlusion) * occlusion_read / (1.0 + current_coneDiameter);

// 		current_dist += max(current_coneDiameter, voxel_size);
// 	}

//     const vec3 light_dir = normalize(point_lights[0].pos - frag_pos);
//     const float c = mat.specular_color.a * 0.008 * PI;
//     const float angle = max(0.0, acos(dot(light_dir, dir)) - c);
//     const float strength = pow(1.0 - (angle / PI), 4.0);
    
// 	return vec4(color * strength, occlusion);  
// }

// vec4 indirectDiffuse() {
//     const float voxel_size = 2.0 / 256.0;
// 	const vec3 origin = frag_pos + frag_normal * voxel_size;

//     vec3 guide = vec3(0.0, 1.0, 0.0);

//     if (abs(dot(frag_normal,guide)) > 0.99) {
//         guide = vec3(0.0, 0.0, 1.0);
//     }

//     // Find a tangent and a bitangent
//     vec3 right = normalize(guide - dot(frag_normal, guide) * frag_normal);
//     vec3 up = cross(right, frag_normal);
//     vec4 diffuseTrace = vec4(0.0);
//     for(int i = 0; i < 16; i++)
//     {
//         vec3 coneDirection = frag_normal;
//         coneDirection += DIFFUSE_CONE_DIRECTIONS_16[i].x * right + DIFFUSE_CONE_DIRECTIONS_16[i].z * up;
//         coneDirection = normalize(coneDirection);
//         diffuseTrace += DiffuseCone(origin, coneDirection) * max(0.0, dot(coneDirection, frag_normal));
//     }
//     diffuseTrace /= 16.0;
//     diffuseTrace.rgb *= 0.30;
//     diffuseTrace.rgb *= frag_color;
//     vec3 res = diffuseTrace.rgb;
//     return vec4(res, clamp(1.0 - diffuseTrace.a, 0.0, 1.0));
// } 


void main() {
    // vec3 position = frag_pos;
    // vec3 normal = frag_normal;
    vec3 albedo = frag_color;
    // vec3 normal = texture(sampler2D(tex_normal, sampler1), frag_uv).rgb;
    // normal = normal*2.0 - 1.0;
    // normal = normalize(frag_TBN * normal);
    // vec3 albedo = texture(sampler2D(diffuseTexture, sampler1), frag_uv).rgb;
    // vec3 view_pos = vec3(V * vec4(0.0, 0.0, 0.0, 1.0));
    // vec4 specular = specular_cone(frag_pos, normalize(reflect(normalize(frag_pos - view_pos), frag_normal)));
    // vec4 indirect = pow(indirectDiffuse(), vec4(1.0/2.2));
    // indirect.rgb *=
    // vec3 direct = calc_direct_light();
    // direct *= 0.0;
    // vec3 final = (direct.rgb + indirect.rgb + specular.rgb) * indirect.a;
    // outColor = vec4(pow(final, vec3(1.0/2.2)), 1.0);
    // final = specular.rgb * specular.a;
    // outColor = vec4(final, 1.0);
    outColor = vec4(albedo, 1.0);
}