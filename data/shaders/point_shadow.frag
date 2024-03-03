#version 460 core
#pragma shader_stage(fragment)
layout(location=0) in vec4 FragPos;

#include "global_set"

layout(std140, set=1, binding=0) uniform LIGHT_VIEW_DIRS {
    mat4 shadowMatrices[6];
    float far_plane;
};

void main()
{
    float lightDistance = length(FragPos.xyz - point_lights[0].pos);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = lightDistance / 25.0;
    
    // write this as modified depth
    gl_FragDepth = lightDistance;
}