#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 normal;
out vec3 worldPos;

void main()
{
    vec4 _worldPos = uModel * vec4(aPos, 1.0);
    worldPos = _worldPos.xyz;
    
    mat3 normalMat = mat3(transpose(inverse(uModel)));
    normal = normalize(normalMat * aNormal);

    gl_Position = uProj * uView * _worldPos;
}
