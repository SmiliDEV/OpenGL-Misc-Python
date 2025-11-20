#version 330 core

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

uniform sampler2D texture1;
uniform vec3 uAmbient;
uniform vec3 uLightColor;
uniform int uLightCount;
uniform vec3 uLightPos[4];
uniform vec3 uLightCol[4];
uniform float uLightInt[4];
uniform vec3 uLightDir;

out vec4 FragColor;

void main()
{
	vec3 base = texture(texture1, TexCoord * vec2(10.0)).rgb;
	vec3 N = normalize(Normal);

	// directional
	vec3 Ld = normalize(-uLightDir);
	float diffd = max(dot(N, Ld), 0.0);
	vec3 dir = uLightColor * diffd;

	// point lights
	vec3 point = vec3(0.0);
	for (int i = 0; i < uLightCount; ++i) {
		vec3 L = normalize(uLightPos[i] - FragPos);
		float dif = max(dot(N, L), 0.0);
		point += uLightCol[i] * uLightInt[i] * dif;
	}

	vec3 color = base * (uAmbient + dir + point);
	color = clamp(color, 0.0, 1.0);
	FragColor = vec4(color, 1.0);
}