#version 460 core
#pragma shader_stage(geometry)
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

layout(std140, set=1, binding=0) uniform LIGHT_VIEW_DIRS {
    mat4 shadowMatrices[6];
    float far_plane;
};

layout(location=0) out vec4 FragPos; // FragPos from GS (output per emitvertex)

void main()
{
    for(int face = 0; face < 6; ++face)
    {
        gl_Layer = face; // built-in variable that specifies to which face we render.
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = shadowMatrices[face] * FragPos;
            // gl_Position = FragPos;
            EmitVertex();
        }    
        EndPrimitive();
    }
} 