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

layout(set=2, binding=0) uniform sampler3D voxel_radiance;

const vec3 diffuseConeDirections[] =
{
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.5, 0.866025),
    vec3(0.823639, 0.5, 0.267617),
    vec3(0.509037, 0.5, -0.7006629),
    vec3(-0.50937, 0.5, -0.7006629),
    vec3(-0.823639, 0.5, 0.267617)
};

const float diffuseConeWeights[] =
{
    PI / 4.0,
    3.0 * PI / 20.0,
    3.0 * PI / 20.0,
    3.0 * PI / 20.0,
    3.0 * PI / 20.0,
    3.0 * PI / 20.0,
};

vec3 position;
vec3 albedo;
vec3 normal;

vec3 WorldToVoxel(vec3 position)
{
    vec3 voxelPos = position * 0.5 + 0.5; // position: [-1 - +1]*0.5+0.5 -> [0 - 1]*vxscale -> [0, 256]
    return voxelPos;
}

bool IntersectRayWithWorldAABB(vec3 ro, vec3 rd, out float enter, out float leave)
{
    vec3 worldMinPoint = vec3(-2.0);
    vec3 worldMaxPoint = vec3(2.0);

    vec3 tempMin = (worldMinPoint - ro) / rd; 
    vec3 tempMax = (worldMaxPoint - ro) / rd;
    
    vec3 v3Max = max (tempMax, tempMin);
    vec3 v3Min = min (tempMax, tempMin);
    
    leave = min (v3Max.x, min (v3Max.y, v3Max.z));
    enter = max (max (v3Min.x, 0.0), max (v3Min.y, v3Min.z));    
    
    return leave > enter;
}

vec4 TraceCone(vec3 position, vec3 normal, vec3 direction, float aperture, bool traceOcclusion)
{
    float aoAlpha = gi_settings.aoAlpha;
    uvec3 visibleFace;
    visibleFace.x = (direction.x < 0.0) ? 0 : 1;
    visibleFace.y = (direction.y < 0.0) ? 2 : 3;
    visibleFace.z = (direction.z < 0.0) ? 4 : 5;
    traceOcclusion = traceOcclusion && aoAlpha < 1.0;
    // world space grid voxel size
    float voxelWorldSize = gi_settings.voxel_size;
    // weight per axis for aniso sampling
    vec3 weight = direction * direction;
    // move further to avoid self collision
    float dst = voxelWorldSize;
    vec3 startPosition = position + normal * dst;
    // final results
    vec4 coneSample = vec4(0.0);
    float occlusion = 0.0;
    float maxDistance = gi_settings.trace_distance;
    float aoFalloff = gi_settings.aoFalloff;
    float falloff =aoFalloff;
    // out of boundaries check
    float enter = 0.0; float leave = 0.0;

    if(!IntersectRayWithWorldAABB(position, direction, enter, leave))
    {
        coneSample.a = 1.0;
    }

    while(coneSample.a < 1.0 && dst <= maxDistance)
    {
        vec3 conePosition = startPosition + direction * dst;
        // cone expansion and respective mip level based on diameter
        float diameter = 2.0 * aperture * dst;
        float mipLevel = log2(diameter / voxelWorldSize);
        // convert position to texture coord
        vec3 coord = WorldToVoxel(conePosition);
        // get directional sample from anisotropic representation
        vec4 anisoSample = textureLod(voxel_radiance, coord, mipLevel); //AnistropicSample(coord, weight, visibleFace, mipLevel);
        // front to back composition
        coneSample += (1.0 - coneSample.a) * anisoSample;
        // ambient occlusion
        if(traceOcclusion && occlusion < 1.0)
        {
            occlusion += ((1.0 - occlusion) * anisoSample.a) / (1.0 + falloff * diameter);
        }
        // move further into volume
        float samplingFactor = 1.0;
        dst += diameter * samplingFactor;
    }

    return vec4(coneSample.rgb, occlusion);
}

float TraceShadowCone(vec3 position, vec3 direction, float aperture, float maxTracingDistance) 
{
    bool hardShadows = false;

    float coneShadowTolerance = 0.51;
    if(coneShadowTolerance == 1.0) { hardShadows = true; }

    float voxelWorldSize = gi_settings.voxel_size;
    vec3 weight = direction * direction;
    float dst = voxelWorldSize;
    vec3 startPosition = position + direction * dst;
    float mipMaxLevel = max(0.0, log2(gi_settings.voxel_resolution));
    float visibility = 0.0;
    float k = exp2(7.0 * coneShadowTolerance);
    float maxDistance = maxTracingDistance;
    float enter = 0.0; float leave = 0.0;

    if(!IntersectRayWithWorldAABB(position, direction, enter, leave)) { visibility = 1.0; }
    
    while(visibility < 1.0 && dst <= maxDistance)
    {
        vec3 conePosition = startPosition + direction * dst;
        float diameter = 2.0 * aperture * dst;
        float mipLevel = log2(diameter / voxelWorldSize);
        vec3 coord = WorldToVoxel(conePosition);
        vec4 anisoSample = textureLod(voxel_radiance, coord, mipLevel);

        // hard shadows exit as soon cone hits something
        if(hardShadows && anisoSample.a > EPSILON) { return 0.0; }  
        // accumulate
        visibility += (1.0 - visibility) * anisoSample.a * k;
        // move further into volume
        float samplingFactor = 1.0;
        dst += diameter * samplingFactor;
    }

    return clamp(1.0 - visibility, 0.0, 1.0);
}

vec3 BRDF(DirectionLight light, vec3 N, vec3 X, vec3 ka)
{
    const vec3 L = light.dir;
    const vec3 V = normalize((V * vec4(0.0, 0.0, 0.0, 1.0)).xyz - X);
    const vec3 H = normalize(V + L);

    const float dotNL = max(dot(N, L), 0.0);
    const float dotNH = max(dot(N, H), 0.0);
    const float dotLH = max(dot(L, H), 0.0);
    
    const vec3 diffuse = ka.rgb * light.col.rgb * light.col.a;
   
    return diffuse * dotNL;
}

vec3 CalculateDirectional(DirectionLight light, vec3 normal, vec3 position, vec3 albedo)
{
    float visibility = 1.0;

    visibility = max(0.0, TraceShadowCone(position, light.dir, gi_settings.occlusion_cone_aperture, gi_settings.trace_distance));

    if(visibility <= 0.0) return vec3(0.0);  

    return BRDF(light, normal, position, albedo) * visibility;
}

vec4 CalculateIndirectLighting(vec3 position, vec3 normal, vec3 albedo, bool ambientOcclusion)
{
    vec4 specularTrace = vec4(0.0);
    vec4 diffuseTrace = vec4(0.0);
    vec3 coneDirection = vec3(0.0);

    // component greater than zero
    if(any(greaterThan(albedo, diffuseTrace.rgb)))
    {
        // diffuse cone setup
        const float aperture = gi_settings.diffuse_cone_aperture;
        vec3 guide = vec3(0.0, 1.0, 0.0);

        if (abs(dot(normal,guide)) == 1.0)
        {
            guide = vec3(0.0, 0.0, 1.0);
        }

        // Find a tangent and a bitangent
        vec3 right = normalize(guide - dot(normal, guide) * normal);
        vec3 up = cross(right, normal);

        for(int i = 0; i < 6; i++)
        {
            coneDirection = normal;
            coneDirection += diffuseConeDirections[i].x * right + diffuseConeDirections[i].z * up;
            coneDirection = normalize(coneDirection);
            
            diffuseTrace += TraceCone(position, normal, coneDirection, aperture, ambientOcclusion) * diffuseConeWeights[i];
        }

        diffuseTrace.rgb *= albedo;
    }

    const float bounceStrength = 2.0;
    vec3 result = bounceStrength * (diffuseTrace.rgb);

    const float aoAlpha = gi_settings.aoAlpha;
    return vec4(result, ambientOcclusion ? clamp(1.0 - diffuseTrace.a + aoAlpha, 0.0, 1.0) : 1.0);
}

vec3 CalculateDirectLighting(vec3 position, vec3 normal, vec3 albedo)
{
    vec3 directLighting = vec3(0.0);

    for(int i = 0; i < direction_light_count; ++i)
    {
        directLighting += CalculateDirectional(direction_lights[i], normal, position, albedo);
    }

    return directLighting;
}

void main() {
    position = frag_pos;
    albedo = frag_color;
    normal = frag_normal;

    Material mat = materials[frag_instance_index];

    if(mat.diffuse_idx > 0) {
        albedo = texture(material_textures[mat.diffuse_idx], frag_uv).rgb;
    }
    if(mat.normal_idx > 0) {
        normal = normalize(texture(material_textures[mat.normal_idx], frag_uv).rgb) * 2.0 - 1.0;
        normal = normalize(frag_tbn * normal);
    }

    vec3 baseColor = albedo;
    vec3 directLighting = vec3(1.0);
    vec4 indirectLighting = vec4(1.0);
    vec3 compositeLighting = vec3(1.0);
    int mode = 0;

    if(mode == 0)   // direct + indirect + ao
    {
        indirectLighting = CalculateIndirectLighting(position, normal, baseColor, true);
        directLighting = CalculateDirectLighting(position, normal, albedo);
    }
    else if(mode == 1)  // direct + indirect
    {
        indirectLighting = CalculateIndirectLighting(position, normal, baseColor, false);
        directLighting = CalculateDirectLighting(position, normal, albedo);
    }
    else if(mode == 2) // direct only
    {
        indirectLighting = vec4(0.0, 0.0, 0.0, 1.0);
        directLighting = CalculateDirectLighting(position, normal, albedo);
    }
    else if(mode == 3) // indirect only
    {
        directLighting = vec3(0.0);
        baseColor.rgb = vec3(1.0);
        indirectLighting = CalculateIndirectLighting(position, normal, baseColor, false);
    }
    else if(mode == 4) // ambient occlusion only
    {
        directLighting = vec3(0.0);
        indirectLighting = CalculateIndirectLighting(position, normal, baseColor, true);
        indirectLighting.rgb = vec3(1.0);
    }

    // convert indirect to linear space
    compositeLighting = (directLighting + indirectLighting.rgb) * indirectLighting.a;
    // -- this could be done in a post-process pass -- 

    // Reinhard tone mapping
    compositeLighting = compositeLighting / (compositeLighting + 1.0);

    outColor = vec4(compositeLighting, 1.0);
}