#version 460 core
#extension GL_EXT_nonuniform_qualifier : require

#define PI 3.14159265359
#define EPSILON 1e-8

layout(location=0) out vec4 outColor;

layout(location=0) in FS_IN {
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
#include "material_set"
#include "point_lights"

layout(set=2, binding=0) uniform sampler3D voxel_radiance;

const vec3 DIFFUSE_CONE_DIRECTIONS_16[16] = {
    vec3( 0.57735,   0.57735,   0.57735  ),
    vec3( 0.57735,  -0.57735,  -0.57735  ),
    vec3(-0.57735,   0.57735,  -0.57735  ),
    vec3(-0.57735,  -0.57735,   0.57735  ),
    vec3(-0.903007, -0.182696, -0.388844 ),
    vec3(-0.903007,  0.182696,  0.388844 ),
    vec3( 0.903007, -0.182696,  0.388844 ),
    vec3( 0.903007,  0.182696, -0.388844 ),
    vec3(-0.388844, -0.903007, -0.182696 ),
    vec3( 0.388844, -0.903007,  0.182696 ),
    vec3( 0.388844,  0.903007, -0.182696 ),
    vec3(-0.388844,  0.903007,  0.182696 ),
    vec3(-0.182696, -0.388844, -0.903007 ),
    vec3( 0.182696,  0.388844, -0.903007 ),
    vec3(-0.182696,  0.388844,  0.903007 ),
    vec3( 0.182696, -0.388844,  0.903007 )
};

vec3 position;
vec3 albedo;
vec3 normal;

vec3 calc_direct_light() {
    vec3 ambient = vec3(0.0); 
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    vec3 frag_nrm = normal;
    vec3 tex_diff = albedo;

    int num_point_lights = 1;
    vec3 cam_pos = vec3(V * vec4(0.0,0.0,0.0,1.0));
    for(uint i=0; i<num_point_lights; ++i) {
        PointLight pl = point_lights[i];
        vec3 light_dir = pl.pos - frag_pos;
        float dist = length(light_dir);
        light_dir = normalize(light_dir);

        vec3 v = normalize(cam_pos - frag_pos);
        vec3 h = normalize(v + light_dir);
        float dotNL = max(dot(normal, light_dir), 0.0);
        float dotNH = max(dot(normal, h), 0.0);
        float dotLH = max(dot(light_dir, h), 0.0);
        float spec = pow(dotNH, 0.3 * 11.0 + 1.0);

        float ldotp = max(dot(normal, light_dir), 0.0);
        float att = 1.0 / (pl.att[0] + pl.att[1]*dist + pl.att[2]*dist*dist);

        diffuse += albedo * att * pl.col;
    }

    return ambient + diffuse + specular;
}

vec4 DiffuseCone(const vec3 origin, const vec3 dir) {
    const float voxel_size = 2.0 / 256.0;
	const float max_dist = 2.0;
	const float apperture_angle = 0.4; // Angle in Radians.
	float current_dist = voxel_size;
	vec3 color = vec3(0.0);
	float occlusion = 0.0;

	while(current_dist < max_dist && occlusion < 1.0) {
		float current_coneDiameter = 2.0 * current_dist * tan(apperture_angle * 0.5);
		vec3 pos_worldspace = origin + dir * current_dist;

        float vlevel = log2(current_coneDiameter / voxel_size); // Current mipmap level
        vlevel = min(8.0, max(vlevel, 0.0));

        vec3 pos_texturespace = pos_worldspace * 0.5 + 0.5 ; // [-1,1] Coordinates to [0,1]
		vec4 voxel = textureLod(voxel_radiance, pos_texturespace, vlevel);	// Sample
		vec3 color_read = voxel.rgb;
		float occlusion_read = voxel.a;

        color += (1.0 - occlusion) * color_read;
        occlusion += (1.0 - occlusion) * occlusion_read / (1.0 + pow(current_coneDiameter, 16.0));

		current_dist += max(current_coneDiameter, voxel_size);
	}
	return vec4(color, occlusion);
}

vec4 specular_cone(const vec3 origin, const vec3 dir) {
    const float voxel_size = 2.0 / 256.0;
	float max_dist = 2.0;
	float current_dist = voxel_size;
    float apperture_angle = 0.08;
	vec3 color = vec3(0.0);
	float occlusion = 0.0;
    
	while(current_dist < max_dist && occlusion < 1.0) {
		float current_coneDiameter = 2.0 * current_dist * tan(apperture_angle * 0.5);
		vec3 pos_worldspace = origin + dir * current_dist;

        float vlevel = log2(current_coneDiameter / voxel_size); // Current mipmap level
        vlevel = min(8.0, max(vlevel, 0.0));

        vec3 pos_texturespace = pos_worldspace * 0.5 + 0.5; // [-1,1] Coordinates to [0,1]
		vec4 voxel = textureLod(voxel_radiance, pos_texturespace, vlevel);	// Sample
		vec3 color_read = voxel.rgb;
		float occlusion_read = voxel.a;

        color += (1.0 - occlusion) * color_read;
        occlusion += (1.0 - occlusion) * occlusion_read / pow(1.0 + current_coneDiameter, 16.0);

		current_dist += max(current_coneDiameter, voxel_size);
	}

    const vec3 light_dir = normalize(point_lights[0].pos - frag_pos);
    const float c = 0.4 * 0.008 * PI;
    const float angle = max(0.0, acos(dot(light_dir, dir)) - c);
    const float strength = pow(1.0 - (angle / PI), 4.0);
    
	return vec4(color * strength, occlusion);  
}

vec4 indirectDiffuse() {
    const float voxel_size = 2.0 / 256.0;
	const vec3 origin = frag_pos + normal * voxel_size;

    vec3 guide = vec3(0.0, 1.0, 0.0);

    if (abs(dot(normal,guide)) > 0.99) {
        guide = vec3(0.0, 0.0, 1.0);
    }

    // Find a tangent and a bitangent
    vec3 right = normalize(guide - dot(normal, guide) * normal);
    vec3 up = cross(right, normal);
    vec4 diffuseTrace = vec4(0.0);

    const int dir_count = 16;
    const int dir_step = 4;

    for(int i = 0; i < dir_count; i += dir_step)
    {
        vec3 coneDirection = normal;
        coneDirection += DIFFUSE_CONE_DIRECTIONS_16[i].x * right + DIFFUSE_CONE_DIRECTIONS_16[i].z * up;
        coneDirection = normalize(coneDirection);
        // diffuseTrace += DiffuseCone(origin, coneDirection);
        diffuseTrace += DiffuseCone(origin, coneDirection) * max(0.0, dot(coneDirection, normal));
    }
    diffuseTrace *= float(dir_step) / float(dir_count);
    diffuseTrace.rgb *= albedo;
    vec3 res = diffuseTrace.rgb;
    return vec4(res, clamp(1.0 - diffuseTrace.a, 0.0, 1.0));
} 

float calc_occlusion(vec3 ro, vec3 rd, const float max_dist) {
    const float voxel_size = 2.0 / 256.0;
	float current_dist = voxel_size;
    const float apperture_angle = 0.01;
	float occlusion = 0.0;
    
	while(current_dist < max_dist && occlusion < 1.0) {
		float current_coneDiameter = 2.0 * current_dist * tan(apperture_angle * 0.5);
		vec3 pos_worldspace = ro + rd * current_dist;

        float vlevel = log2(current_coneDiameter / voxel_size); // Current mipmap level
        vlevel = min(8.0, max(vlevel, 0.0));

        vec3 pos_texturespace = pos_worldspace * 0.5 + 0.5; // [-1,1] coordinates to [0,1]
		vec4 voxel = textureLod(voxel_radiance, pos_texturespace, vlevel);	// Sample
		vec3 color_read = voxel.rgb;
		float occlusion_read = voxel.a;

        occlusion += (1.0 - occlusion) * occlusion_read / pow((1.0 + current_coneDiameter), 2.0);

		current_dist += max(current_coneDiameter, voxel_size);
	}

    return clamp(1.0 - occlusion, 0.0, 1.0);
}

void main() {
    position = frag_pos;
    albedo = frag_color;
    normal = frag_normal;

    Material mat = materials[frag_instance_index];
    if(mat.diffuse_idx >= 0) {
        albedo = texture(material_textures[nonuniformEXT(mat.diffuse_idx)], frag_uv).rgb;
        normal = texture(material_textures[nonuniformEXT(mat.normal_idx)], frag_uv).rgb;
        normal = normalize(frag_tbn * (normal * 2.0 - 1.0));
    }

    vec3 view_pos = vec3(V * vec4(0.0, 0.0, 0.0, 1.0));
    vec4 indirect = indirectDiffuse();
    indirect *= 4.0;
    vec3 direct = calc_direct_light();


    const vec3 p = position + normal * (2.0 / 256.0);
    const vec3 l = point_lights[0].pos;
    vec3 pl = l - p;
    const float pld = length(pl);
    pl /= pld;
    const float occ = calc_occlusion(p, pl, pld) * clamp(dot(pl, normal), 0.0, 1.0);

    const vec4 specular = specular_cone(p, normal);

    vec3 final = (direct.rgb * occ + indirect.rgb + specular.rgb);

    outColor = vec4(final, 1.0);
}