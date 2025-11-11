#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main()
{
    // Debug: color by direction instead of sampling the cubemap texture.
    // TexCoords are in range roughly [-1,1]; map to [0,1] for display.
    vec3 col = normalize(TexCoords) * 0.5 + 0.5;
    FragColor = vec4(col, 1.0);
}