#version 330 core
out vec4 FragColor;

in vec3 vNormal;
in vec3 vWorldPos;

uniform vec3 uLightDir;  // direção do sol (em espaço mundo)
uniform vec3 uColor;     // cor base do material
uniform vec3 uAmbient;   // ambient color
uniform vec3 uLightColor; // directional light color/intensity

uniform int uLightCount;
uniform vec3 uLightPos[4];
uniform vec3 uLightCol[4];
uniform float uLightInt[4];

void main()
{
    vec3 N = normalize(vNormal);

    // directional light (sun)
    vec3 Ld = normalize(-uLightDir);
    float diffd = max(dot(N, Ld), 0.0);
    vec3 dir_contrib = uLightColor * diffd;

    // point lights
    vec3 point_contrib = vec3(0.0);
    for (int i = 0; i < uLightCount; ++i) {
        vec3 L = normalize(uLightPos[i] - vWorldPos);
        float dif = max(dot(N, L), 0.0);
        point_contrib += uLightCol[i] * uLightInt[i] * dif;
    }

    vec3 base = uColor;
    vec3 color = base * (uAmbient + dir_contrib + point_contrib);
    // simple tonemapping / clamp
    color = clamp(color, 0.0, 1.0);
    FragColor = vec4(color, 1.0);
}
