#version 410 core

uniform sampler2D u_tex_source;

layout(location = 0) out vec2 o_seed_pos;
layout(location = 1) out vec4 o_seed_color;

void main(void) {
	ivec2 coord = ivec2(gl_FragCoord.xy);
	vec4 sample_color = texelFetch(u_tex_source, coord, 0);
	float weight = sample_color.w;

	if (weight > 0.0) {
		vec3 color = sample_color.xyz / max(weight, 1e-6);
		o_seed_pos = vec2(coord);
		o_seed_color = vec4(color, 1.0);
	} else {
		o_seed_pos = vec2(-1.0);
		o_seed_color = vec4(0.0);
	}
}
