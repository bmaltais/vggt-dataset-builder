#version 410 core

uniform sampler2D u_tex_seed_pos;
uniform sampler2D u_tex_seed_color;
uniform int u_step;
uniform ivec2 u_viewport_size;

layout(location = 0) out vec2 o_seed_pos;
layout(location = 1) out vec4 o_seed_color;

void main(void) {
	ivec2 base = ivec2(gl_FragCoord.xy);
	float best_dist = 1e20;
	vec2 best_pos = vec2(-1.0);
	vec4 best_color = vec4(0.0);

	for (int j = -1; j <= 1; ++j) {
		for (int i = -1; i <= 1; ++i) {
			ivec2 coord = base + ivec2(i * u_step, j * u_step);
			if (coord.x < 0 || coord.y < 0 || coord.x >= u_viewport_size.x || coord.y >= u_viewport_size.y) {
				continue;
			}

			vec2 seed_pos = texelFetch(u_tex_seed_pos, coord, 0).xy;
			if (seed_pos.x < 0.0) {
				continue;
			}

			float dist = length(vec2(base) - seed_pos);
			if (dist < best_dist) {
				best_dist = dist;
				best_pos = seed_pos;
				best_color = texelFetch(u_tex_seed_color, coord, 0);
			}
		}
	}

	o_seed_pos = best_pos;
	o_seed_color = best_color;
}
