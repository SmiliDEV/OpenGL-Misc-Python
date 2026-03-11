#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

layout(std140) uniform Matrices
{
    mat4 projection;
    mat4 view;
};

uniform mat4 uModel;

out vec3 normal;
out vec3 worldPos;

void main()
{
    vec4 _worldPos = uModel * vec4(aPos, 1.0);
    worldPos = _worldPos.xyz;
    
    mat3 normalMat = mat3(transpose(inverse(uModel)));
    normal = normalize(normalMat * aNormal);

    gl_Position = projection * view * _worldPos;
}
