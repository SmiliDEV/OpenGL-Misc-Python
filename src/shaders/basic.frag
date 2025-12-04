#version 330 core
out vec4 FragColor;

in vec3 normal;
in vec3 worldPos;

uniform vec3 uAlbedo;

uniform vec3 uLightDir; 
uniform vec3 uAmbient;   
uniform vec3 uLightDiffuse; 

uniform vec3 uSpecularColor;
uniform float uShininess;    
uniform vec3 uViewPos;
uniform float uDiffuseFactor;

uniform int uLightCount;
uniform vec3 uLightPos[4];
uniform vec3 uLightCol[4];
uniform float uLightInt[4];

uniform int uEmissive; // 0 = false, 1 = true

void main()
{
    if (uEmissive != 0) {
        FragColor = vec4(uAlbedo, 1.0);
        return;
    }

    vec3 N = normalize(normal);

    // view vector
    vec3 V = normalize(uViewPos - worldPos);

    // directional light (sun)
    vec3 Ld = normalize(-uLightDir);
    float diffd = max(dot(N, Ld), 0.0);
    vec3 dir_contrib = uLightDiffuse * diffd;
    // directional specular (Blinn)
    vec3 H_dir = normalize(Ld + V);
    float spec_dir = pow(max(dot(N, H_dir), 0.0), max(1.0, uShininess));

    // point lights: diffuse + specular
    vec3 point_contrib = vec3(0.0);
    vec3 point_spec = vec3(0.0);
    for (int i = 0; i < uLightCount; ++i) {
        vec3 L = normalize(uLightPos[i] - worldPos);
        float dif = max(dot(N, L), 0.0);
        point_contrib += uLightCol[i] * uLightInt[i] * dif;
        vec3 H = normalize(L + V);
        float s = pow(max(dot(N, H), 0.0), max(1.0, uShininess));
        point_spec += uLightCol[i] * uLightInt[i] * s;
    }

    vec3 base = uAlbedo;
    float ds = (uDiffuseFactor > 0.0) ? uDiffuseFactor : 1.0;
    vec3 diffuse = base * (uAmbient + ds * (dir_contrib + point_contrib));
    vec3 specular = uSpecularColor * (spec_dir + dot(point_spec, vec3(1.0)));

    vec3 color = diffuse + specular;
    // simple tonemapping / clamp
    color = clamp(color, 0.0, 1.0);
    FragColor = vec4(color, 1.0);
}
