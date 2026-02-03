#version 410 core

uniform sampler2D u_tex_cam_pos_levels[8];
uniform int u_num_levels;
uniform int u_coarse_level;
uniform vec2 u_viewport_size;
uniform float u_fov_y;
uniform float u_s_hpr;
uniform float u_occlusion_threshold;

layout(location = 0) out float o_visibility;

in vec2 v_tex_coord;

float occlusion_value(vec3 x, vec3 y) {
	return 1.0 - dot((y - x) / length(y - x), -y / length(y));
}

bool is_background(vec4 cam_pos) {
	return abs(cam_pos.w) < 0.0001;
}

void main(void) {
	ivec2 coord_base = ivec2(gl_FragCoord.xy);

	vec4 x_v4 = texelFetch(u_tex_cam_pos_levels[0], coord_base, 0);
	if (is_background(x_v4)) {
		o_visibility = 0.0;
		return;
	}
	vec3 x = x_v4.xyz;

	ivec2 coord_coarse = coord_base >> u_coarse_level;
	vec3 cam_coarse = texture(u_tex_cam_pos_levels[u_coarse_level], v_tex_coord).xyz;
	float z_i = max(1e-6, -cam_coarse.z);

	float inv_pixel_world_scale = (u_s_hpr * u_viewport_size.y) / (2.0 * tan(u_fov_y * 0.5) * z_i);
	float l_float = log2(inv_pixel_world_scale);
	int l = int(clamp(floor(l_float + 0.5), 0.0, float(u_num_levels - 1)));

	ivec2 coord_l = coord_base >> l;
	const ivec2 dirs[8] = ivec2[8](ivec2(1, 0), ivec2(1, 1), ivec2(0, 1), ivec2(-1, 1), ivec2(-1, 0),
		ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1));

	float sum_occ = 0.0;
	int count_occ = 0;

	for (int s = 0; s < 8; ++s) {
		float best_occ = 1.0;
		bool found = false;

		int sweep_from_level = 0, sweep_to_level = l;
		for (int dl = sweep_from_level; dl <= sweep_to_level; ++dl) {
			int L = clamp(dl, 0, u_num_levels - 1);

			ivec2 coord_L = coord_base >> L;
			ivec2 coord_sample = coord_L + dirs[s];

			vec4 y_v4 = texelFetch(u_tex_cam_pos_levels[L], coord_sample, 0);
			if (is_background(y_v4)) {
				continue;
			}
			vec3 y = y_v4.xyz;

			float occ = occlusion_value(x, y);
			if (!found || occ < best_occ) {
				best_occ = occ;
				found = true;
			}
		}

		if (found) {
			sum_occ += best_occ;
			count_occ += 1;
		}
	}

	float mean_occ = (count_occ > 0) ? (sum_occ / float(count_occ)) : 1.0;
	bool visible = !(mean_occ < u_occlusion_threshold);

	o_visibility = visible ? 1.0 : 0.0;
}
