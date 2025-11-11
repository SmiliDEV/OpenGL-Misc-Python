#version 330 core

in vec3 ourColor;
in vec2 TexCoord;

// texture sampler
uniform sampler2D texture1;

out vec4 FragColor;

void main()
{
	vec2 t = TexCoord * vec2(10.0);
    FragColor = texture(texture1, t);
	//FragColor = texture(texture1, TexCoord);
}