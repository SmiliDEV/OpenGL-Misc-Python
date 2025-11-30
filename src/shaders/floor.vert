#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

layout(std140) uniform Matrices
{
    mat4 projection;
    mat4 view;
};

uniform mat4 uModel;

out vec2 texCoord;
out vec3 worldPos;
out vec3 normal;

void main()
{
	vec4 _worldPos = uModel * vec4(aPos, 1.0);
	worldPos = _worldPos.xyz;

	texCoord = aTexCoord;
	normal = mat3(uModel) * aNormal;

	gl_Position = projection * view * _worldPos;
}