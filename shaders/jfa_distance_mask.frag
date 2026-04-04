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
	float factor = exp(-0.5 * normalized * normalized);

	vec4 sample_source = texelFetch(u_tex_source, coord, 0);
	// ⚡ Bolt: Perform alpha premultiplication and distance masking on the GPU.
	// We use factor * factor to match the original CPU-side behavior where factor
	// was effectively applied twice (once in shader, once during alpha multiplication).
	// clamp(..., 0.0, 1.0) ensures correct mapping to normalized uint8 textures.
	o_dest = vec4(clamp(sample_source.xyz * sample_source.w * factor * factor, 0.0, 1.0), 1.0);
}
