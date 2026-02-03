#version 410 core

uniform sampler2D u_tex_source;

layout(location = 0) out vec4 o_dest;

void main(void) {
	vec3 x_num = vec3(0.0);
	float w_sum = 0.0;

	for (int j = 0; j < 2; ++j) {
		for (int i = 0; i < 2; ++i) {
			ivec2 coord = ivec2(int(gl_FragCoord.x) * 2 + i, int(gl_FragCoord.y) * 2 + j);

			vec4 cur_sample = texelFetch(u_tex_source, coord, 0);

			w_sum += cur_sample.w;
			x_num += cur_sample.xyz * cur_sample.w;
		}
	}

	vec3 x_next = (w_sum > 0.0) ? (x_num / w_sum) : vec3(0.0);
	float w_next = clamp(w_sum, 0.0, 1.0);

	o_dest = vec4(x_next, w_next);
}
