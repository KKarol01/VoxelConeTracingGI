#version 460 core
#pragma shader_stage(fragment)

#define PI 3.14159265359
#define EPSILON 1e-8

layout(location=0) out vec4 outColor;

layout(location=0) in FS_IN {
    // vec2 frag_uv;
    // mat3 frag_TBN;
    vec3 frag_pos;
    vec3 frag_normal;
    vec3 frag_color;
};

#include "global_set"
layout(set=1, binding=0) uniform texture3D voxel_radiance;
layout(set=1, binding=1) uniform sampler voxel_sampler;
// layout(set=1, binding=2) uniform textureCube cube_depth_map;
// layout(set=2, binding=1) uniform sampler sampler1;
// layout(set=2, binding=2) uniform texture2D diffuseTexture;
// layout(set=2, binding=3) uniform texture2D tex_normal;

// const uint num_cones = 5;
// const vec3 ConeVectors[5] = vec3[5](
// 				    vec3(0.0, 1.0, 0.0),
// 				    vec3(0.0, 0.707106781, 0.707106781),
// 				    vec3(0.0, -0.707106781, 0.707106781),
// 				    vec3(0.707106781, 0.0, 0.707106781),
// 				    vec3(-0.707106781, 0.0, 0.707106781));
// const float Weights[5] = float[5]( 0.28, 0.18, 0.18, 0.18, 0.18 );
// const float Apertures[5] = float[5]( /* tan(45) */ 1.0, 1.0, 1.0, 1.0, 1.0 );

const vec3 diffuseConeDirections[] = {
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.5, 0.866025),
    vec3(0.823639, 0.5, 0.267617),
    vec3(0.509037, 0.5, -0.7006629),
    vec3(-0.50937, 0.5, -0.7006629),
    vec3(-0.823639, 0.5, 0.267617)
};

const float diffuseConeWeights[] = {
    PI / 4.0,
    3.0 * PI / 20.0,
    3.0 * PI / 20.0,
    3.0 * PI / 20.0,
    3.0 * PI / 20.0,
    3.0 * PI / 20.0,
};

// vec3 position;
// vec3 normal;
// int voxel_base_dimension = 2;
// int voxel_resolution = 256;

// vec4 trace_cone(vec3 o, vec3 d, float ca, float raystep, float rayoff, float alpha_thr) {
//     vec3 p = o + d*rayoff;
//     float t = rayoff;
//     vec4 acc = vec4(0.0);
//     const float voxel_size = 1.0 / 256.0;
//     while(acc.a < alpha_thr && t < 3.0) {
//         float cdiam = 2.0 * tan(ca) * t;
//         float sd = max(cdiam, voxel_size);
//         float slod = log2(sd / voxel_size);
//         vec3 samp = p*0.5 + 0.5;
//         vec4 ns = textureLod(sampler3D(voxels_3d, voxels_sampler), samp, slod).rgba;
//         float nsw = 1.0 - acc.a;
//         acc += nsw * ns;
//         t += raystep;
//         p = o + d*t;
//     }

//     return acc;
// }

// uint GetPCGHash(inout uint seed)
// {
//     seed = seed * 747796405u + 2891336453u;
//     uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
//     return (word >> 22u) ^ word;
// }

// float GetRandomFloat01(inout uint Random_RNGSeed)
// {
//     return float(GetPCGHash(Random_RNGSeed)) / 4294967296.0;
// }

// vec3 UniformSampleSphere(inout uint Random_RNGSeed)
// {
//     float z = GetRandomFloat01(Random_RNGSeed) * 2.0 - 1.0;
//     float a = GetRandomFloat01(Random_RNGSeed) * 2.0 * PI;
//     float r = sqrt(1.0 - z * z);
//     float x = r * cos(a);
//     float y = r * sin(a);

//     return vec3(x, y, z);
// }

// vec3 CosineSampleHemisphere(vec3 normal, inout uint Random_RNGSeed)
// {
//     return normalize(normal + UniformSampleSphere(Random_RNGSeed));
// }

// vec4 indirectDiffuse() {
//     uint Random_RNGSeed = 3454738;
//     const vec3 origin = position;

//     vec4 ret = vec4(0.0);
//     float cone_count = float(num_cones);

//     for(int i=0; i<16; ++i) {
//         ret += trace_cone(origin, CosineSampleHemisphere(normal, Random_RNGSeed), 0.35, 0.1, 0.02, 1.0);
//     }

// 	return ret * (PI / 16.0);
// } 

float get_shadowing() {
    float shadow = 0.0;
    // vec2 texel_size = 1.0 / textureSize(samplerCube(cube_depth_map, sampler1), 0).xy;

    // PointLight point_lights[1];
    // point_lights[0].pos = vec3(0.0, 0.8, 0.0);
    // point_lights[0].col = vec3(1.0);
    // point_lights[0].att = vec3(0.3, 0.5, 0.8);
    // PointLight pl = point_lights[0];
    // for(int i=-1; i<=1; ++i) {
    //     for(int j=-1; j<=1; ++j) {
    //         vec3 coord = frag_pos - pl.pos;
    //         coord.xy += vec2(i, j) * texel_size;
    //         float d = texture(samplerCube(cube_depth_map, sampler1), coord).r;
    //         shadow += d*25.0 > length(frag_pos - pl.pos)-0.01 ? 1.0 : 0.0;
    //         return shadow;
    //     }
    // }
    
    return 0.0;
}

vec3 calc_direct_light() {
    vec3 ambient = vec3(0.0), diffuse = vec3(0.0);
    // vec3 frag_nrm = texture(sampler2D(tex_normal, sampler1), frag_uv).rgb;
    vec3 frag_nrm = frag_normal;
    // frag_nrm = normalize(frag_TBN * frag_nrm);
    // vec3 tex_diff = texture(sampler2D(diffuseTexture, sampler1), frag_uv).rgb;
    vec3 tex_diff = frag_color;

    PointLight point_lights[1];
    point_lights[0].pos = vec3(0.0, 0.65, 0.0);
    point_lights[0].col = vec3(1.0);
    point_lights[0].att = vec3(0.2, 0.4, 0.4);
    int num_point_lights = 1;
    for(uint i=0; i<num_point_lights; ++i) {
        PointLight pl = point_lights[i];
        vec3 light_dir = pl.pos - frag_pos;
        float dist = length(light_dir);
        light_dir *= dist;

        float ldotp = max(dot(frag_nrm, light_dir), 0.0);
        float att = 1.0 / (pl.att[0] + pl.att[1]*dist + pl.att[2]*dist*dist);
        diffuse += tex_diff * att * pl.col;
    }

    return diffuse + ambient;
}

vec4 TraceCone(vec3 position, vec3 normal, vec3 direction, float aperture) {
    const float voxel_size = 2.0 / 256.0;
    const vec3 weight = direction*direction;

    float d = voxel_size;
    position += normal * d;
    vec3 p = position;
    vec4 result = vec4(0.0);
    float occlusion = 0.0;
    const float max_distance = 3.5;

    while(d < max_distance && result.a < 1.0) {
        const float diameter = max(voxel_size, 2.0 * tan(aperture) * d);
        const float mip = max(log2(diameter / voxel_size), 0.0);
        const vec3 voxel_coord = vec3(p * 0.5 + 0.5);

        result += (1.0 - result.a) * textureLod(sampler3D(voxel_radiance, voxel_sampler), voxel_coord, mip).rgba;

        d += diameter; 
        p = position + direction * d;
    }
    
    return vec4(result.rgb, occlusion);
}

vec4 calculate_indirect(vec3 position, vec3 normal, vec3 albedo) {
    vec4 diffuse_trace = vec4(0.0);
    if(dot(albedo, albedo) > EPSILON) {
        const float aperture = 0.57735;

        vec3 guide = vec3(0.0, 1.0, 0.0);
        if(1.0 - abs(dot(normal, guide)) < EPSILON) {
            guide = vec3(0.0, 0.0, 1.0);
        }

        vec3 right = normalize(guide - dot(normal, guide) * normal);
        vec3 up = cross(right, normal);

        vec3 cone_direction;
        for(int i=0; i<6; ++i) {
            cone_direction = normal;
            cone_direction += diffuseConeDirections[i].x * right + diffuseConeDirections[i].z * up;
            cone_direction = normalize(cone_direction);
            diffuse_trace += TraceCone(position, normal, cone_direction, aperture) * diffuseConeWeights[i];
        }
    }

    diffuse_trace.rgb *= albedo;
    return vec4(diffuse_trace.rgb, 1.0);
}

void main() {
    vec3 position = frag_pos;
    vec3 normal = frag_normal;
    vec3 albedo = frag_color;
    // vec3 normal = texture(sampler2D(tex_normal, sampler1), frag_uv).rgb;
    // normal = normal*2.0 - 1.0;
    // normal = normalize(frag_TBN * normal);
    // vec3 albedo = texture(sampler2D(diffuseTexture, sampler1), frag_uv).rgb;
    float shadow = max(get_shadowing(), 0.2);
    vec4 indirect = calculate_indirect(position, normal, albedo);
    vec3 col = calc_direct_light() * shadow;
    vec3 final = (col + indirect.rgb * 2.0) * indirect.a;

    outColor = vec4(final, 1.0);
}