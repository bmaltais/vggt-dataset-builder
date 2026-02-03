#version 410 core

uniform sampler2D u_tex_color;
uniform sampler2D u_tex_cam_pos;
uniform sampler2D u_tex_depth;

in vec2 v_tex_coord;

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_cam_pos;

void main(void) {
	float best_depth = 1.0;

	for (int j = 0; j < 2; ++j) {
		for (int i = 0; i < 2; ++i) {
			ivec2 coord = ivec2(int(gl_FragCoord.x) * 2 + i, int(gl_FragCoord.y) * 2 + j);

			float sample_depth = texelFetch(u_tex_depth, coord, 0).x;
			if (sample_depth < best_depth) {
				best_depth = sample_depth;

				o_color = texelFetch(u_tex_color, coord, 0);
				o_cam_pos = vec4(texelFetch(u_tex_cam_pos, coord, 0).xyz, 1.0);
			}
		}
	}

	gl_FragDepth = best_depth;
}
