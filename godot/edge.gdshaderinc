// Shared by edge.gdshader (solid lines) and edge_faded.gdshader (see-through lines).
// Each draws every connection, and collapses the ones that belong to the other pass.

uniform sampler2D pos_a : filter_nearest, repeat_disable;
uniform sampler2D pos_b : filter_nearest, repeat_disable;
uniform sampler2D edge_a : filter_nearest, repeat_disable;
uniform sampler2D edge_b : filter_nearest, repeat_disable;
uniform float blend = 1.0;
uniform int tex_w = 128;
uniform int mode = 0;             // 0 = your drawing, 1 = all weights
uniform float focus_node = -1.0;  // global node id, or -1
uniform float faded_alpha = 0.06;  // lines not touching the selected neuron
uniform float pulse_t = 10.0;     // layer units: 0..1 crosses input -> layer 1, and so on
uniform vec3 pos_col = vec3(1.0, 0.55, 0.18);
uniform vec3 neg_col = vec3(0.25, 0.6, 1.0);
uniform float rest_level = 0.15;  // brightness of a line at rest, as a share of its colour (linear)
uniform float cube = 0.5;         // neuron cube size, metres
uniform float face_fill = 0.9;    // share of a cube's front face used by connection points
uniform float min_width = 0.001;  // metres, for a weight of 0
uniform float max_width = 0.04;   // metres, for the largest weights of the run

varying vec3 v_col;
varying float v_flow;
varying float v_along;
varying float v_layer;

vec3 node_pos(int id) {
	return mix(texelFetch(pos_a, ivec2(id, 0), 0).rgb, texelFetch(pos_b, ivec2(id, 0), 0).rgb, blend);
}

// Where an incoming line meets its neuron's front face: the point for the sending pixel
// (28 x 28) or the sending neuron's seat (4 x 4), laid out like the layer it comes from.
// The layers run towards -z, so the front face is at +z.
vec3 face_point(int layer, int src) {
	float n = layer == 0 ? 28.0 : 4.0;
	int i = layer == 0 ? src : (layer == 1 ? src - 784 : src - 800);
	float c = float(i % int(n));
	float r = float(i / int(n));
	float s = cube * face_fill;
	return vec3(((c + 0.5) / n - 0.5) * s, (0.5 - (r + 0.5) / n) * s, cube * 0.5);
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
	float flow = mode == 0 ? abs(f) : 0.0;
	// With a neuron selected, every line not touching it is drawn faint, in the other pass.
	bool faded = false;
	if (focus_node >= 0.0) {
		faded = !(abs(INSTANCE_CUSTOM.r - focus_node) < 0.5 || abs(INSTANCE_CUSTOM.g - focus_node) < 0.5);
	}
#ifdef FADED_PASS
	bool skip = !faded;
	flow *= 0.15;
#else
	bool skip = faded;
#endif
	v_col = w >= 0.0 ? pos_col : neg_col;
	v_flow = flow;
	v_layer = float(layer);
	v_along = VERTEX.z + 0.5;

	// Pixels send from their own place; neurons send one value from the back face.
	vec3 a = node_pos(src) + (layer == 0 ? vec3(0.0) : vec3(0.0, 0.0, -cube * 0.5));
	vec3 b = node_pos(dst) + face_point(layer, src);
	vec3 dir = b - a;
	vec3 fwd = dir / max(length(dir), 1e-5);
	vec3 up = abs(fwd.y) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 side = normalize(cross(fwd, up));
	vec3 up2 = cross(side, fwd);
	float t = mix(min_width, max_width, abs(w));
	vec3 local = a + dir * v_along + side * VERTEX.x * t + up2 * VERTEX.y * t;
	if (skip) {
		local = vec3(0.0);  // all corners in one point: nothing is drawn in this pass
	}
	VERTEX = (MODELVIEW_MATRIX * vec4(local, 1.0)).xyz;
}

void fragment() {
	// Where is the pulse on this edge? u runs 0..1 while it crosses this layer.
	float u = pulse_t - v_layer;
	float front = exp(-pow((v_along - u) / 0.06, 2.0));             // the travelling head
	// The trail it leaves fades within about one layer's crossing, to a faint residue that
	// still shows which connections carried your drawing.
	float since = u - v_along;
	float passed = smoothstep(0.0, 0.02, since) * (0.05 + 0.3 * exp(-since * 2.5));
	float glow = v_flow * clamp(front * 1.4 + passed, 0.0, 1.4);
	vec3 rest = v_col * rest_level;
	ALBEDO = mix(rest, mix(v_col, vec3(1.0, 0.97, 0.9), 0.55), clamp(glow, 0.0, 1.0));
#ifdef FADED_PASS
	ALPHA = faded_alpha;
#endif
}
