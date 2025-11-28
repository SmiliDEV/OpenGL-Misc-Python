#version 330 core
out vec4 FragColor;

in vec2 texCoord;

in vec3 normal;
in vec3 worldPos;

uniform sampler2D uAlbedoSampler;

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

void main()
{
	vec3 albedo = texture(uAlbedoSampler, texCoord * vec2(10.0)).rgb;
	vec3 N = normalize(normal);

	vec3 V = normalize(uViewPos - worldPos);

	vec3 Ld = normalize(-uLightDir);
	float diff_dir = max(dot(N, Ld), 0.0);
	vec3 diffuse_dir = uLightDiffuse * diff_dir;

	vec3 Hd = normalize(V + Ld);
	float spec_dir = pow(max(dot(N, Hd), 0.0), max(1.0, uShininess));
	vec3 specular_dir = uLightDiffuse * spec_dir * uSpecularColor;

	vec3 diffuse_point = vec3(0.0);
	vec3 spec_point = vec3(0.0);
	for (int i = 0; i < uLightCount; ++i) {
		vec3 L = normalize(uLightPos[i] - worldPos);
		float dif = max(dot(N, L), 0.0);
		diffuse_point += uLightCol[i] * uLightInt[i] * dif;

		vec3 H = normalize(V + L);
		float sp = pow(max(dot(N, H), 0.0), max(1.0, uShininess));
		spec_point += uLightCol[i] * uLightInt[i] * sp * uSpecularColor;
	}

	// apply material-specific diffuse scale to reduce overbright surfaces
	float diff_scale = (uDiffuseFactor > 0.0) ? uDiffuseFactor : 1.0;
	vec3 lighting = uAmbient + diff_scale * (diffuse_dir + diffuse_point);
	vec3 specular = specular_dir + spec_point;

	vec3 color = albedo * lighting + specular;
	color = clamp(color, 0.0, 1.0);
	FragColor = vec4(color, 1.0);
}