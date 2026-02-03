#version 410 core

in vec3 v_color;
in vec3 v_world_pos;
in vec3 v_cam_pos;
in float v_confidence;

uniform float u_confidence_threshold;

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_cam_pos;
layout(location = 2) out vec4 o_world_pos;

void main(void) {
	if (v_confidence < u_confidence_threshold) {
		discard;
	}
	o_color = vec4(v_color, 1.0);
	o_cam_pos = vec4(v_cam_pos, 1.0);
	o_world_pos = vec4(v_world_pos, 1.0);
}
