#version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 TexCoords;

void main()
{
    TexCoords = aPos;
    mat4 rotView = mat4(mat3(uView));
    gl_Position = uProj * rotView * vec4(aPos, 1.0);
}