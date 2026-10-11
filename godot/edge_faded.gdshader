// The see-through pass: only lines not touching the selected neuron. See edge.gdshader.
shader_type spatial;
render_mode cull_back, blend_mix, depth_draw_never, shadows_disabled, skip_vertex_transform;

#define FADED_PASS
#include "res://edge.gdshaderinc"
