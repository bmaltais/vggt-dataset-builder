#version 410 core

uniform sampler2D u_tex_source;

layout(location = 0) out vec4 o_dest;

void main(void) {
        o_dest = texelFetch(u_tex_source, ivec2(gl_FragCoord.xy), 0);
}
