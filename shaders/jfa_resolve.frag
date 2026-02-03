#version 410 core

uniform sampler2D u_tex_seed_pos;
uniform sampler2D u_tex_seed_color;

layout(location = 0) out vec4 o_nearest_color;
layout(location = 1) out float o_nearest_dist;

void main(void) {
	ivec2 coord = ivec2(gl_FragCoord.xy);
	vec2 seed_pos = texelFetch(u_tex_seed_pos, coord, 0).xy;

	if (seed_pos.x < 0.0) {
		o_nearest_color = vec4(0.0);
		o_nearest_dist = 1e6;
	} else {
		o_nearest_color = texelFetch(u_tex_seed_color, coord, 0);
		o_nearest_dist = length(vec2(coord) - seed_pos);
	}
}
