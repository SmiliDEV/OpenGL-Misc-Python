#version 330 core
out vec4 FragColor;

in vec3 vNormal;
in vec3 vWorldPos;

uniform vec3 uLightDir;  // direção do sol (em espaço mundo)
uniform vec3 uColor;     // cor base do material

void main()
{
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir); // vindo da direção inversa

    float diff = max(dot(N, L), 0.0);
    float ambient = 0.25;

    vec3 color = uColor * (ambient + diff * 0.85);
    FragColor = vec4(color, 1.0);
}
