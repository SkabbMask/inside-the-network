// One instance per connection (12,960). Both end points come from a node-position
// texture and weight/flow from an edge-data texture, so neurons can move without the
// CPU touching any edge. Two copies of each texture (a = previous snapshot,
// b = current) are blended so playback is smooth.
//
// Encoding: thickness = size of the weight, colour = its sign, brightness = the pulse
// of your drawing travelling through, in proportion to the signal each edge carries.
shader_type spatial;
render_mode unshaded, blend_mix, cull_disabled, depth_draw_never, shadows_disabled, skip_vertex_transform;

uniform sampler2D pos_a : filter_nearest, repeat_disable;
uniform sampler2D pos_b : filter_nearest, repeat_disable;
uniform sampler2D edge_a : filter_nearest, repeat_disable;
uniform sampler2D edge_b : filter_nearest, repeat_disable;
uniform float blend = 1.0;
uniform int tex_w = 128;
uniform int mode = 0;             // 0 = your drawing, 1 = all weights
uniform float focus_node = -1.0;  // global node id, or -1
uniform float pulse_t = 10.0;     // layer units: 0..1 crosses floor -> layer 1, and so on
uniform vec3 pos_col = vec3(1.0, 0.55, 0.18);
uniform vec3 neg_col = vec3(0.25, 0.6, 1.0);
uniform vec3 rest_alpha = vec3(0.035, 0.22, 0.3);  // per layer of edges, at rest
uniform float width = 0.0045;

varying vec3 v_col;
varying float v_alpha;
varying float v_flow;
varying float v_along;
varying float v_layer;

vec3 node_pos(int id) {
	return mix(texelFetch(pos_a, ivec2(id, 0), 0).rgb, texelFetch(pos_b, ivec2(id, 0), 0).rgb, blend);
}

void vertex() {
	int id = INSTANCE_ID;
	ivec2 tc = ivec2(id % tex_w, id / tex_w);
	vec2 d = mix(texelFetch(edge_a, tc, 0).rg, texelFetch(edge_b, tc, 0).rg, blend);
	float w = d.r;
	float f = d.g;
	int src = int(INSTANCE_CUSTOM.r + 0.5);
	int dst = int(INSTANCE_CUSTOM.g + 0.5);
	int layer = int(INSTANCE_CUSTOM.b + 0.5);
	float ra = layer == 0 ? rest_alpha.x : (layer == 1 ? rest_alpha.y : rest_alpha.z);

	float alpha = ra * (0.35 + 0.65 * abs(w));
	// For your drawing, floor edges from empty pixels carry nothing: almost hide them.
	if (mode == 0 && layer == 0 && abs(f) < 0.001) {
		alpha *= 0.12;
	}
	float flow = mode == 0 ? abs(f) : 0.0;
	if (focus_node >= 0.0) {
		bool hit = abs(INSTANCE_CUSTOM.r - focus_node) < 0.5 || abs(INSTANCE_CUSTOM.g - focus_node) < 0.5;
		if (hit) {
			alpha = max(alpha, (layer == 0 ? 0.25 : 0.6) * (0.3 + 0.7 * abs(w)));
		} else {
			alpha *= 0.15;
			flow *= 0.15;
		}
	}
	v_col = w >= 0.0 ? pos_col : neg_col;
	v_alpha = alpha;
	v_flow = flow;
	v_layer = float(layer);
	v_along = VERTEX.z + 0.5;

	vec3 a = node_pos(src);
	vec3 b = node_pos(dst);
	vec3 dir = b - a;
	vec3 fwd = dir / max(length(dir), 1e-5);
	vec3 up = abs(fwd.y) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 side = normalize(cross(fwd, up));
	vec3 up2 = cross(side, fwd);
	float t = width * (0.15 + 2.6 * abs(w));
	vec3 local = a + dir * v_along + side * VERTEX.x * t + up2 * VERTEX.y * t;
	VERTEX = (MODELVIEW_MATRIX * vec4(local, 1.0)).xyz;
}

void fragment() {
	// Where is the pulse on this edge? u runs 0..1 while it crosses this layer.
	float u = pulse_t - v_layer;
	float front = exp(-pow((v_along - u) / 0.06, 2.0));             // the travelling head
	float passed = smoothstep(0.0, 0.02, u - v_along) * 0.35;        // the trail it leaves
	float glow = v_flow * clamp(front * 1.4 + passed, 0.0, 1.4);
	vec3 col = mix(v_col * 0.75, mix(v_col, vec3(1.0, 0.97, 0.9), 0.55), clamp(glow, 0.0, 1.0));
	ALBEDO = col;
	ALPHA = clamp(v_alpha + glow * (v_layer == 0.0 ? 0.35 : 0.9), 0.0, 1.0);
}
