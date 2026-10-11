// Shared by edge.gdshader (solid lines) and edge_faded.gdshader (see-through lines).
// Each draws every connection, and collapses the ones that belong to the other pass.

uniform sampler2D pos_a : filter_nearest, repeat_disable;
uniform sampler2D pos_b : filter_nearest, repeat_disable;
uniform sampler2D edge_a : filter_nearest, repeat_disable;
uniform sampler2D edge_b : filter_nearest, repeat_disable;
uniform float blend = 1.0;
uniform int tex_w = 128;
uniform int id_offset = 0;        // lines come in two batches (fine image lines, cables)
uniform int mode = 0;             // 0 = your drawing, 1 = all weights
uniform float focus_node = -1.0;  // global node id, or -1
uniform float faded_alpha = 0.06;  // lines not touching the selected neuron
uniform float pulse_t = 10.0;     // layer units: 0..1 crosses input -> layer 1, and so on
uniform vec3 pos_col = vec3(1.0, 0.55, 0.18);
uniform vec3 neg_col = vec3(0.25, 0.6, 1.0);
uniform float rest_level = 0.14;  // surface colour of a line at rest, as a share of its colour
// Each tube glows a little of its own colour, like a glow stick: less for the 12,544
// pixel lines so their mass does not wash out. Tubes are opaque, so where they overlap
// you only see the front one: the glow never adds up.
uniform float glow_input = 0.25;
uniform float glow_cable = 0.45;
uniform float pulse_energy = 4.0; // how strongly the pulse glows
uniform float grid_cells = 6.0;   // wires around a tube per metre of diameter (4 to 10)
// Slack: every line hangs and sways a little, like a loose cable floating in space. This
// is decoration only: the curve carries no information. Ends stay fixed.
uniform float sway_t = 0.0;       // time, set from the scene (so grabbing can follow)
uniform float grab_sway_t = 0.0;  // the held line stays frozen at the moment it was grabbed
uniform float slack = 0.025;      // how far the middle of a line strays, per metre of length
uniform float grid_line = 0.07;   // wire thickness, as a share of a grid cell
uniform float rim_power = 3.0;    // how tightly the glow hugs a tube's silhouette
uniform float cube = 16.0;        // width of the square connection grids on a neuron, metres
uniform float ball = 12.0;        // radius of the neuron's glass ball, metres
uniform float face_fill = 0.9;    // share of a cube's face used by connection points
uniform float pixel = 12.857;     // width of one input pixel's tile, metres
uniform float min_width = 0.05;   // smallest diameter in metres, so small weights stay visible
uniform float max_width = 4.48;   // diameter in metres for the largest weights; in between
                                  // the diameter is proportional to the weight
uniform float max_width_input = 0.64;  // the same for the pixel lines (each layer has its own
                                       // ruler, as its weights have their own scale)
uniform float grab_edge = -1.0;   // the line being held, lit up

varying vec3 v_col;
varying float v_flow;
varying float v_along;
varying float v_layer;
varying float v_grab;
varying float v_len;     // length of this line, metres
varying float v_width;   // its diameter, metres

vec3 node_pos(int id) {
	return mix(texelFetch(pos_a, ivec2(id, 0), 0).rgb, texelFetch(pos_b, ivec2(id, 0), 0).rgb, blend);
}

// Lift a point of a square connection grid (x, y) onto the ball's front (+1) or back (-1)
// hemisphere, straight along z, so the grid keeps its layout seen from front or back.
// Slightly inside the surface, so tube ends never poke out.
vec3 on_ball(vec2 p, float side) {
	float r = ball * 0.97;
	return vec3(p, side * sqrt(max(r * r - dot(p, p), 0.0)));
}

// Where an incoming line meets its neuron's ball: the point for the sending pixel
// (28 x 28) or the sending neuron's seat (4 x 4), laid out like the layer it comes from.
// The layers run towards -z, so the front is at +z.
vec3 face_point(int layer, int src) {
	float n = layer == 0 ? 28.0 : 4.0;
	int i = layer == 0 ? src : (layer == 1 ? src - 784 : src - 800);
	float c = float(i % int(n));
	float r = float(i / int(n));
	float s = cube * face_fill;
	return on_ball(vec2(((c + 0.5) / n - 0.5) * s, (0.5 - (r + 0.5) / n) * s), 1.0);
}

// Where an outgoing line leaves its sender: on the back of the pixel's tile or the
// neuron's cube, at the point for the receiving neuron on a 4 x 4 grid (the next
// layer's seats, or the digits 0..9 in reading order). Every one of these lines carries
// the same value: the sender's single output.
vec3 back_point(int layer, int dst) {
	int i = layer == 0 ? dst - 784 : (layer == 1 ? dst - 800 : dst - 816);
	float s = layer == 0 ? pixel * face_fill : cube * face_fill;
	vec2 p = vec2(((float(i % 4) + 0.5) / 4.0 - 0.5) * s, (0.5 - (float(i / 4) + 0.5) / 4.0) * s);
	return layer == 0 ? vec3(p, 0.0) : on_ball(p, -1.0);
}

void vertex() {
	int id = INSTANCE_ID + id_offset;
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
	// The mesh is a unit cylinder along y: y runs the length, x and z go round.
	v_along = VERTEX.y + 0.5;

	vec3 a = node_pos(src) + back_point(layer, dst);
	vec3 b = node_pos(dst) + face_point(layer, src);
	vec3 dir = b - a;
	vec3 fwd = dir / max(length(dir), 1e-5);
	vec3 up = abs(fwd.y) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 side = normalize(cross(fwd, up));
	vec3 up2 = cross(side, fwd);
	float t = max(min_width, (layer == 0 ? max_width_input : max_width) * abs(w));
	v_len = length(dir);
	v_width = t;
	v_grab = abs(float(id) - grab_edge) < 0.5 ? 1.0 : 0.0;
	// Bend the line: the middle strays by `bend`, the ends stay put (sin 0 = sin pi = 0).
	float ph = fract(float(id) * 0.618034) * 6.2831853;
	// Pixel lines are straight (their mesh has no segments to bend, which keeps the
	// 12,544 of them cheap); the cables between neurons sag and sway.
	vec3 bend = vec3(0.0);
	float st = abs(float(id) - grab_edge) < 0.5 ? grab_sway_t : sway_t;
	if (layer > 0) {
		bend = slack * v_len * (vec3(0.0, -0.6, 0.0)
			+ side * sin(st * 0.35 + ph) * 0.8
			+ up2 * cos(st * 0.27 + ph * 1.3) * 0.8);
	}
	float sb = sin(PI * v_along);
	vec3 centre = a + dir * v_along + bend * sb;
	vec3 tang = normalize(dir + bend * PI * cos(PI * v_along));
	vec3 up_c = abs(tang.y) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	side = normalize(cross(tang, up_c));
	up2 = cross(side, tang);
	vec3 local = centre + side * VERTEX.x * t + up2 * VERTEX.z * t;
	if (skip) {
		local = vec3(0.0);  // all corners in one point: nothing is drawn in this pass
	}
	VERTEX = (MODELVIEW_MATRIX * vec4(local, 1.0)).xyz;
	NORMAL = normalize((MODELVIEW_MATRIX * vec4(side * NORMAL.x + up2 * NORMAL.z, 0.0)).xyz);
}

void fragment() {
	// Where is the pulse on this edge? u runs 0..1 while it crosses this layer.
	float u = pulse_t - v_layer;
	float front = exp(-pow((v_along - u) / 0.06, 2.0));             // the travelling head
	// The trail it leaves fades within about one layer's crossing, to a faint residue that
	// still shows which connections carried your drawing.
	float since = u - v_along;
	float passed = smoothstep(0.0, 0.02, since) * (0.05 + 0.3 * exp(-since * 2.5));
	float head = v_flow * clamp(front * 1.4, 0.0, 1.4);
	float trail = v_flow * passed;
	// Neon wireframe: a near-black body, a glowing grid of rings and wires in the tube's
	// colour, and a glowing rim along its silhouette. Grid cells are square and scale
	// with the tube's thickness. Where a cell gets smaller than a pixel the grid fades to
	// its average glow, so distant tubes do not shimmer.
	// Fewer, wider-spaced wires on thin tubes; a whole number so the grid closes round.
	float around = clamp(round(v_width * grid_cells), 4.0, 10.0);
	float cell = PI * v_width / around;
	vec2 g = vec2(UV.x * around, v_along * v_len / cell);
	vec2 fw = fwidth(g);
	vec2 d = abs(fract(g - 0.5) - 0.5);
	vec2 lines = 1.0 - smoothstep(vec2(grid_line * 0.5), vec2(grid_line * 0.5) + fw * 1.5, d);
	float wire = max(lines.x, lines.y);
	float far = clamp(max(fw.x, fw.y) * 2.0 - 0.5, 0.0, 1.0);
	wire = mix(wire, grid_line * 1.3, far);
	float rim = pow(1.0 - clamp(dot(NORMAL, normalize(VIEW)), 0.0, 1.0), rim_power);
	float own = (v_layer < 0.5 ? glow_input : glow_cable);
	ALBEDO = v_col * rest_level * 0.08;
	ROUGHNESS = 0.35;
	SPECULAR = 0.15;
	EMISSION = v_col * own * (wire * 2.4 + rim * 2.4 + 0.03)
		// The pulse head burns nearly white; the trail it leaves glows in the tube's colour.
		+ mix(v_col, vec3(1.0, 0.95, 1.0), 0.55) * head * pulse_energy
		+ v_col * trail * pulse_energy * 0.5
		+ v_col * v_grab * 0.8;
#ifdef FADED_PASS
	ALPHA = faded_alpha;
#endif
}
