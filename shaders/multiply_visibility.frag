#version 410 core

uniform sampler2D u_tex_source;
uniform sampler2D u_tex_factor;

layout(location = 0) out vec4 o_dest;

void main(void) {
	ivec2 coord_base = ivec2(gl_FragCoord.xy);
	vec4 source_sample = texelFetch(u_tex_source, coord_base, 0);
	float factor_sample = texelFetch(u_tex_factor, coord_base, 0).x;
	o_dest = source_sample * factor_sample;
}
