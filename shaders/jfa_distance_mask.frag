#version 410 core

uniform sampler2D u_tex_source;
uniform sampler2D u_tex_dist;
uniform float u_sigma;

layout(location = 0) out vec4 o_dest;

void main(void) {
	ivec2 coord = ivec2(gl_FragCoord.xy);
	float dist = texelFetch(u_tex_dist, coord, 0).x;
	float sigma = max(u_sigma, 0.0001);
	float normalized = dist / sigma;
	float gaussian = exp(-0.5 * normalized * normalized);
	float factor = gaussian;
	o_dest = texelFetch(u_tex_source, coord, 0) * factor;
}
