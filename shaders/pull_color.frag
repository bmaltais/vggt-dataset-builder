#version 410 core

uniform sampler2D u_tex_source_prev;
uniform sampler2D u_tex_source_cur;

layout(location = 0) out vec4 o_dest;

in vec2 v_tex_coord;

void main(void) {
	vec4 sample_cur = texelFetch(u_tex_source_cur, ivec2(gl_FragCoord.xy), 0);
	vec4 sample_prev = texture(u_tex_source_prev, v_tex_coord);
	float w_out = sample_cur.w + (1.0 - sample_cur.w) * sample_prev.w;
	vec3 x_out =
		(sample_cur.xyz * sample_cur.w + sample_prev.xyz * sample_prev.w * (1.0 - sample_cur.w)) / max(w_out, 1e-6);
	o_dest = vec4(x_out, w_out);
}
