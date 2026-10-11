// One instance per connection (12,960). Both end points come from a node-position
// texture and weight/flow from an edge-data texture, so neurons can move without the
// CPU touching any edge. Two copies of each texture (a = previous snapshot,
// b = current) are blended so playback is smooth.
//
// Encoding: thickness = size of the weight (on one scale for the whole run, so a line
// only thickens when its weight grows), colour = its sign, brightness = the pulse
// of your drawing travelling through, in proportion to the signal each edge carries.
// Lines are solid: overlapping lines never add up, so nothing but thickness shows weight.
shader_type spatial;
// cull_front: the edge's basis below is mirrored, so these are the faces that point away.
render_mode unshaded, cull_front, shadows_disabled, skip_vertex_transform;

#include "res://edge.gdshaderinc"


