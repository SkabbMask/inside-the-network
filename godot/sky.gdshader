// Deep space: a faint purple and blue nebula with two layers of stars. Decoration only.
shader_type sky;

uniform vec3 deep = vec3(0.004, 0.002, 0.012);
uniform vec3 nebula_a = vec3(0.05, 0.012, 0.08);   // purple
uniform vec3 nebula_b = vec3(0.01, 0.025, 0.07);   // blue

float hash(vec3 p) {
	p = fract(p * 0.3183099 + 0.1);
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
				mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
			mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
				mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

// One star per grid cell (where the cell's hash says so), as a soft dot.
vec3 stars(vec3 dir, float scale, float density, float size) {
	vec3 p = dir * scale;
	vec3 cell = floor(p);
	float h = hash(cell);
	if (h < 1.0 - density) {
		return vec3(0.0);
	}
	vec3 centre = cell + vec3(hash(cell + 7.1), hash(cell + 3.7), hash(cell + 1.3));
	float d = length(p - centre);
	float b = smoothstep(size, 0.0, d) * (0.4 + 0.6 * fract(h * 91.7));
	vec3 tint = mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.75, 0.95), fract(h * 37.3));
	return tint * b;
}

void sky() {
	vec3 d = EYEDIR;
	float n = noise(d * 2.5) * 0.6 + noise(d * 6.0) * 0.4;
	float m = noise(d * 1.7 + 11.0);
	vec3 col = deep + mix(nebula_b, nebula_a, m) * smoothstep(0.35, 0.85, n);
	col += stars(d, 220.0, 0.06, 0.10) * 1.6;
	col += stars(d, 520.0, 0.04, 0.12) * 0.8;
	COLOR = col;
}
