## The network as a space you walk through.
##
## Depth (away from the visitor) is the order of computation: the input is a 28 x 28
## mosaic on the floor, then hidden layer 1, hidden layer 2 and the answer layer.
## Within each layer, neurons are placed on a similarity map from the server: neurons
## that react to the same drawings sit close together. The map is recomputed for every
## snapshot of training, so neurons move as their behaviour changes.
extends Node3D

const SIZES := [784, 16, 16, 10]
const NODE_OFFSET := [0, 784, 800, 816]
const N_NODES := 826
const EDGE_TEX_W := 128
const EDGE_TEX_H := 102

# The whole network is built large, so you fly between neurons and the lines (which keep
# their real thickness in millimetres) have room to stay apart. Distances scale with SPREAD.
const SPREAD := 90.0
const LAYER_Z := [0.0, -3.6 * SPREAD, -6.2 * SPREAD, -8.8 * SPREAD]
const MID_Z := -4.4 * SPREAD
# Every slice (the input image and both hidden grids) is a square this wide, in metres,
# standing upright, with its lowest neurons GRID_FLOOR above the floor.
const GRID_SIZE := 4.0 * SPREAD
const GRID_FLOOR := 20.0
const GRID_Y := GRID_FLOOR + GRID_SIZE / 2
const ANSWER_WIDTH := 6.5 * SPREAD
const LABEL_PIXEL := 0.0035 * SPREAD   # text size in the network, metres per font pixel
# Every neuron is a cube. Its front face (towards the previous layer) is a small copy of
# that layer: a 28 x 28 grid for the pixels, or the 4 x 4 seats of the hidden grid. Each
# incoming line ends at its own point there; the one outgoing value leaves from the
# middle of the back face.
const CUBE := 16.0      # width of the square connection grids on each neuron
const BALL := CUBE * 0.75   # each neuron is a black glass ball; its grids sit on it
const FACE_FILL := 1.0   # the connection points' cells tile the whole face
# Line diameter is proportional to the weight: MAX_WIDTH for the largest weights of the
# run, down to MIN_WIDTH, below which lines stay at MIN_WIDTH so they remain visible.
# Each layer's weights are already on that layer's own scale (see the server), so each
# layer gets its own ruler: the 12,544 pixel lines are a fine mesh, the 416 lines between
# neurons thick cables. Both are sized by one rule: if every weight arriving at a face
# were at the maximum, the tubes would cover just over 90% of it. A round tube covers at
# most 79% of its own square cell, so it must be 1.12x the cell's width (and overlap its
# neighbours slightly) to cover 90%. Cells: 4 x 4 on a cube face, 28 x 28 for the pixels.
const MAX_WIDTH := 1.12 * CUBE / 4.0
const MAX_WIDTH_INPUT := 1.12 * CUBE / 28.0
const MIN_WIDTH := 0.05
const PICK_RADIUS := BALL

const COL_POS := Color(1.0, 0.16, 0.62)    # hot pink: pushes the next neuron on
const COL_NEG := Color(0.12, 0.55, 1.0)    # electric blue: pushes it off
const COL_IDLE := Color(0.09, 0.04, 0.15)
const COL_LIT := Color(1.0, 0.86, 1.0)
const COL_OFF := Color(0.55, 0.08, 0.12)
const COL_TEXT := Color(1.0, 0.28, 0.75)       # bright pink, in the console's retro font
const COL_TEXT_DIM := Color(0.85, 0.22, 0.62)
const ANSWER_Z := LAYER_Z[3] - 1.5 * SPREAD      # where the answer's text stands, past the row
const ANSWER_Y := GRID_Y + 40.0                 # bottom of that text
const LABEL_MAX_WIDTH := 180.0                  # metres; longer labels wrap onto more lines
const NETWORK_FONT_SCALE := 1.6                 # VT323 is small for its size; and a bit larger
const COL_FRAME := Color(0.62, 0.22, 1.0)  # the input screen's neon border

var input_pos := PackedVector3Array()
var pos_from := PackedVector3Array()   # hidden + output neurons, 42 each
var pos_to := PackedVector3Array()
var pos_now := PackedVector3Array()
var blend := 1.0
var blend_time := 0.15
var pulse_t := 10.0
var pulse_speed := 1.6   # layers per second

var node_mats := []
var node_inst := []
var acts := PackedFloat32Array()
var off := [[], [], [], []]
var selected := Vector2i(-1, -1)

var edge_mat: ShaderMaterial   # the first batch's material (for reading settings)
var _edge_mats := []           # every batch's solid and faded materials
var edge_data := PackedFloat32Array()
var _edge_tex := []   # two ImageTextures, alternating
var _pos_tex := []
var _cur := 0         # which of the two holds the newest data
var _pos_img: Image
var _input_img: Image
var _input_tex: ImageTexture
var _answer: Label3D
var _answer_caption: Label3D
var _center_note: Label3D
var _titles := []   # each layer's title label
var _notes := []    # each layer's note label above its title, or null
var _facing := []   # labels turned towards the visitor's position (not their view) each frame
var _sheet: MeshInstance3D
var _digit_labels := []
var _panel: Node3D
var _panel_img: Image
var _panel_tex: ImageTexture
var _panel_quad: MeshInstance3D
var _panel_title: Label3D
var _panel_caption: Label3D


func build() -> void:
	for i in 784:
		var r := i / 28
		var c := i % 28
		var x := (c + 0.5) / 28.0 * GRID_SIZE - GRID_SIZE / 2
		var y := GRID_Y + GRID_SIZE / 2 - (r + 0.5) / 28.0 * GRID_SIZE
		input_pos.append(Vector3(x, y, LAYER_Z[0]))
	var fixed := _fixed_positions()
	pos_from = fixed.duplicate()
	pos_to = fixed.duplicate()
	pos_now = fixed.duplicate()
	acts.resize(N_NODES)
	acts.fill(0.0)
	_pos_img = Image.create(N_NODES, 1, false, Image.FORMAT_RGBF)
	_build_input()
	_build_edges()
	_build_nodes()
	_build_labels()
	_build_panel()
	_write_positions(pos_to)
	_write_positions(pos_to)
	_bind_textures()


# ------------------------------------------------------------------ construction

## Every neuron keeps one place for the whole run, so you can follow it through training.
## Hidden layers are 4 × 4 grids filling the same square as the image; a neuron's place
## in the grid means nothing (each one is connected to every pixel). Answers: a row, 0 to 9.
func _fixed_positions() -> PackedVector3Array:
	var out := PackedVector3Array()
	for l in range(1, 3):
		for i in 16:
			var x := (i % 4) / 3.0 * GRID_SIZE - GRID_SIZE / 2
			var y := GRID_Y + GRID_SIZE / 2 - (i / 4) / 3.0 * GRID_SIZE
			out.append(Vector3(x, y, LAYER_Z[l]))
	for d in 10:
		out.append(Vector3(d / 9.0 * ANSWER_WIDTH - ANSWER_WIDTH / 2, GRID_Y, LAYER_Z[3]))
	return out


## Turn labels to face where the visitor is, around the vertical axis only. (A billboard
## follows the camera's rotation instead, so labels swing whenever you turn your head.)
func _face_labels() -> void:
	var cam := get_viewport().get_camera_3d()
	if not cam:
		return
	var eye := cam.global_position
	for l in _facing:
		var d: Vector3 = eye - (l as Node3D).global_position
		if d.x * d.x + d.z * d.z > 1e-6:
			(l as Node3D).global_rotation = Vector3(0, atan2(d.x, d.z), 0)


## The console's retro terminal font (see main.gd), for the panel that travels with it.
func _mono() -> Font:
	return load("res://assets/fonts/VT323-Regular.ttf")


func _unshaded(color: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.albedo_color = color
	return m


func _label(text: String, pos: Vector3, size := 40, color := COL_TEXT, billboard := false) -> Label3D:
	var l := Label3D.new()
	l.text = text
	l.font_size = int(size * NETWORK_FONT_SCALE)
	l.pixel_size = LABEL_PIXEL
	l.font = _mono()
	l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	l.width = LABEL_MAX_WIDTH / LABEL_PIXEL
	l.outline_size = 0
	l.modulate = color
	l.position = pos
	if billboard:
		_facing.append(l)
	add_child(l)
	return l


## The input: your drawing as 28 × 28 pixel values, standing upright as the first slice.
func _build_input() -> void:
	_input_img = Image.create(28, 28, false, Image.FORMAT_L8)
	_input_tex = ImageTexture.create_from_image(_input_img)
	var center := Vector3(0, GRID_Y, LAYER_Z[0])
	# A huge black screen with a glowing border.
	var border := MeshInstance3D.new()
	var bq := QuadMesh.new()
	var frame := GRID_SIZE / 28.0 * 0.6
	bq.size = Vector2(GRID_SIZE + 2 * frame, GRID_SIZE + 2 * frame)
	border.mesh = bq
	border.material_override = _unshaded(COL_FRAME * 1.6)
	border.position = center - Vector3(0, 0, 0.3)
	add_child(border)
	var tiles := MeshInstance3D.new()
	var q := QuadMesh.new()
	q.size = Vector2(GRID_SIZE, GRID_SIZE)
	tiles.mesh = q
	var m := _unshaded(Color.WHITE)
	m.albedo_texture = _input_tex
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	tiles.material_override = m
	tiles.position = center
	add_child(tiles)


func _build_edges() -> void:
	for k in 2:
		_edge_tex.append(ImageTexture.create_from_image(Image.create(EDGE_TEX_W, EDGE_TEX_H, false, Image.FORMAT_RGF)))
		_pos_tex.append(ImageTexture.create_from_image(_pos_img))
	# Two batches with the same shader: the 12,544 thin pixel lines need far less detail
	# than the 416 thick cables, and they are most of the GPU's work.
	_add_edge_batch(0, 12544, 6, 0)   # straight: no segments along
	_add_edge_batch(12544, 416, 16, 11)
	edge_mat = _edge_mats[0]
	set_edge_param("tex_w", EDGE_TEX_W)
	set_edge_param("cube", CUBE)
	set_edge_param("ball", BALL)
	set_edge_param("pixel", GRID_SIZE / 28.0)
	set_edge_param("max_width", MAX_WIDTH)
	set_edge_param("max_width_input", MAX_WIDTH_INPUT)
	set_edge_param("min_width", MIN_WIDTH)
	set_edge_param("face_fill", FACE_FILL)
	set_edge_param("pos_col", Vector3(COL_POS.r, COL_POS.g, COL_POS.b))
	set_edge_param("neg_col", Vector3(COL_NEG.r, COL_NEG.g, COL_NEG.b))
	set_edge_param("glow_input", 0.35)
	set_edge_param("glow_cable", 0.6)
	_bind_textures()


## One MultiMesh of round tubes for edges first .. first + count - 1 (the scene's edge
## order), with `sides` around and `rings` + 1 segments along, so they can curve.
func _add_edge_batch(first: int, count: int, sides: int, rings: int) -> void:
	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.5
	mesh.bottom_radius = 0.5
	mesh.height = 1.0
	mesh.radial_segments = sides
	mesh.rings = rings
	mesh.cap_top = false
	mesh.cap_bottom = false
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_custom_data = true
	mm.mesh = mesh
	mm.instance_count = count
	var idx := 0
	var e := 0
	for l in range(1, 4):
		for j in SIZES[l]:
			for i in SIZES[l - 1]:
				if e >= first and e < first + count:
					mm.set_instance_transform(idx, Transform3D.IDENTITY)
					mm.set_instance_custom_data(idx, Color(NODE_OFFSET[l - 1] + i, NODE_OFFSET[l] + j, l - 1, 0))
					idx += 1
				e += 1
	var inst := MultiMeshInstance3D.new()
	inst.multimesh = mm
	inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	# The shader moves every vertex, so tell the engine where the edges can be.
	inst.custom_aabb = AABB(Vector3(-ANSWER_WIDTH / 2 - CUBE, -1, LAYER_Z[3] - CUBE),
		Vector3(ANSWER_WIDTH + 2 * CUBE, GRID_Y + GRID_SIZE / 2 + CUBE + 1, -LAYER_Z[3] + 2 * CUBE))
	var mat := ShaderMaterial.new()
	mat.shader = load("res://edge.gdshader")
	# A second pass draws the lines not touching a selected neuron, see-through.
	var faded := ShaderMaterial.new()
	faded.shader = load("res://edge_faded.gdshader")
	mat.next_pass = faded
	mat.set_shader_parameter("id_offset", first)
	faded.set_shader_parameter("id_offset", first)
	_edge_mats.append(mat)
	_edge_mats.append(faded)
	inst.material_override = mat
	add_child(inst)


## Edge settings go to every batch and pass.
func set_edge_param(name: String, value) -> void:
	for m in _edge_mats:
		m.set_shader_parameter(name, value)


func _bind_textures() -> void:
	var prev := 1 - _cur
	set_edge_param("edge_a", _edge_tex[prev])
	set_edge_param("edge_b", _edge_tex[_cur])
	set_edge_param("pos_a", _pos_tex[prev])
	set_edge_param("pos_b", _pos_tex[_cur])


func _build_nodes() -> void:
	var sphere := SphereMesh.new()
	sphere.radius = BALL
	sphere.height = BALL * 2
	sphere.radial_segments = 48
	sphere.rings = 24
	# Black glass: a glossy, near-black surface that reflects the stars and the tubes
	# around it. Activation is a light inside the glass: brightest at the centre of the
	# ball as you see it, fading to the rim, where the reflections take over.
	var sh := Shader.new()
	sh.code = """
shader_type spatial;
uniform vec3 col = vec3(0.0);   // the inner light: activation
uniform float off = 0.0;        // 1 = switched off by the visitor: glows red, rim and all
void fragment() {
	float facing = clamp(dot(NORMAL, VIEW), 0.0, 1.0);
	float rim = pow(1.0 - facing, 2.0);
	ALBEDO = vec3(0.005, 0.003, 0.01);
	METALLIC = 0.0;
	ROUGHNESS = 0.03;
	SPECULAR = 1.0;
	CLEARCOAT = 1.0;
	CLEARCOAT_ROUGHNESS = 0.02;
	EMISSION = col * pow(facing, 4.0) * 1.3
		+ vec3(1.0, 0.06, 0.12) * off * (0.5 + 2.5 * rim);
}
"""
	for k in 42:
		var mi := MeshInstance3D.new()
		mi.mesh = sphere
		var m := ShaderMaterial.new()
		m.shader = sh
		mi.material_override = m
		mi.position = pos_now[k]
		add_child(mi)
		node_mats.append(m)
		node_inst.append(mi)
	for d in 10:
		var l := Label3D.new()
		l.text = str(d)
		l.font = _mono()
		l.font_size = int(64 * NETWORK_FONT_SCALE)
		l.pixel_size = LABEL_PIXEL * 0.6
		l.outline_size = 0
		l.modulate = COL_TEXT_DIM
		_facing.append(l)
		# Up and behind the ball, away from the cables arriving from the front.
		l.position = Vector3(0, BALL + CUBE * 0.8, -BALL * 1.5)
		node_inst[32 + d].add_child(l)
		_digit_labels.append(l)


func _build_labels() -> void:
	var top := GRID_Y + GRID_SIZE / 2
	var names := ["Your drawing
28 × 28 = 784 pixel values
(inputs, not neurons)",
		"Hidden layer 1 · 16 neurons", "Hidden layer 2 · 16 neurons", "Answer · one neuron per digit"]
	# Labels wrap at LABEL_MAX_WIDTH so that, seen from the side, neighbouring layers' labels
	# keep apart. Each layer's title and note are stacked upwards by _stack_labels().
	for l in 4:
		_titles.append(_label(names[l], Vector3(0, 0, LAYER_Z[l]), 44, COL_TEXT, true))
		_notes.append(null)
	_notes[1] = _label("Each neuron keeps its place, so you can follow it.
Its place in the grid means nothing: every neuron is connected to every pixel.
Line thickness compares weights within one layer, not between layers.",
		Vector3(0, 0, LAYER_Z[1]), 30, COL_TEXT_DIM, true)
	_center_note = _label("", Vector3(0, 0, LAYER_Z[0]), 30, COL_TEXT_DIM, true)
	_notes[0] = _center_note
	for l in 4:
		_stack_labels(l)
	set_center(false)
	# The answer group stands past the answer row, clear of the cables: the "% sure" caption,
	# the big digit above it, and the layer's title on top (see _stack_labels).
	_answer = _label("", Vector3(0, ANSWER_Y + 30.0, ANSWER_Z), 200, COL_LIT, true)
	_answer.vertical_alignment = VERTICAL_ALIGNMENT_BOTTOM
	_answer_caption = _label("", Vector3(0, ANSWER_Y, ANSWER_Z), 40, COL_TEXT, true)
	_answer_caption.vertical_alignment = VERTICAL_ALIGNMENT_BOTTOM


func _build_panel() -> void:
	_panel = Node3D.new()
	_panel.top_level = true
	add_child(_panel)
	place_panel(Transform3D.IDENTITY)
	_panel_img = Image.create(28, 28, false, Image.FORMAT_RGB8)
	_panel_tex = ImageTexture.create_from_image(_panel_img)
	_panel_quad = MeshInstance3D.new()
	var q := QuadMesh.new()
	q.size = Vector2(0.42, 0.42)
	_panel_quad.mesh = q
	var m := _unshaded(Color.WHITE)
	m.albedo_texture = _panel_tex
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	_panel_quad.material_override = m
	_panel.add_child(_panel_quad)
	_panel_title = Label3D.new()
	_panel_title.font_size = 48
	_panel_title.pixel_size = 0.0015
	_panel_title.outline_size = 0
	_panel_title.position = Vector3(0, 0.27, 0)
	_panel_title.modulate = Color(0.25, 1.0, 0.45)
	_panel_title.font = _mono()
	_panel.add_child(_panel_title)
	_panel_caption = Label3D.new()
	_panel_caption.font_size = 38
	_panel_caption.pixel_size = 0.0015
	_panel_caption.outline_size = 0
	_panel_caption.modulate = Color(0.12, 0.6, 0.25)
	_panel_caption.font = _mono()
	_panel_caption.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_panel_caption.width = 300
	_panel_caption.position = Vector3(0, -0.32, 0)
	_panel.add_child(_panel_caption)
	_panel.visible = false


## Your drawing as faint chalk on a sheet just in front of the input slice, shown only
## while the model is being helped, when the slice shows the centred version instead.
func add_drawing_sheet(canvas: Texture2D) -> void:
	var sh := Shader.new()
	sh.code = """
shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_never, blend_mix;
uniform sampler2D canvas : filter_linear;
void fragment() {
	float v = clamp((texture(canvas, UV).r - 0.07) / 0.93, 0.0, 1.0);
	ALBEDO = vec3(1.0, 0.45, 0.85);
	ALPHA = v * 0.45;
}
"""
	var m := ShaderMaterial.new()
	m.shader = sh
	m.set_shader_parameter("canvas", canvas)
	var sheet := MeshInstance3D.new()
	var q := QuadMesh.new()
	q.size = Vector2(GRID_SIZE, GRID_SIZE)
	sheet.mesh = q
	sheet.material_override = m
	sheet.position = Vector3(0, GRID_Y, LAYER_Z[0] + 0.05)
	sheet.visible = false
	add_child(sheet)
	_sheet = sheet


## Paint the input slice straight from the drawing, without waiting for the server.
func show_input(values: PackedFloat32Array) -> void:
	var bytes := PackedByteArray()
	bytes.resize(784)
	for i in 784:
		bytes[i] = int(clampf(values[i], 0.0, 1.0) * 255.0)
	_input_img.set_data(28, 28, false, Image.FORMAT_L8, bytes)
	_input_tex.update(_input_img)


## Where a ray hits the input slice: {t, uv} with uv in 0..1 (top left = 0, 0), or {}.
func input_hit(world_origin: Vector3, world_dir: Vector3) -> Dictionary:
	var o := to_local(world_origin)
	var d := (global_basis.inverse() * world_dir).normalized()
	if absf(d.z) < 1e-5:
		return {}
	var t := (LAYER_Z[0] - o.z) / d.z
	if t <= 0.0:
		return {}
	var p := o + d * t
	var uv := Vector2(p.x / GRID_SIZE + 0.5, 0.5 - (p.y - GRID_Y) / GRID_SIZE)
	if uv.x < 0.0 or uv.x > 1.0 or uv.y < 0.0 or uv.y > 1.0:
		return {}
	return {"t": t, "uv": uv}


## Stack a layer's title just above its top row of neurons, and its note (if any) above
## the title, measuring the wrapped text so they never overlap.
func _stack_labels(l: int) -> void:
	var y: float = GRID_Y + GRID_SIZE / 2 + BALL + 8.0 if l < 3 else ANSWER_Y + 150.0
	if l == 3:
		_titles[l].position.z = ANSWER_Z
	for lab in [_titles[l], _notes[l]]:
		if lab == null:
			continue
		var label: Label3D = lab
		label.vertical_alignment = VERTICAL_ALIGNMENT_BOTTOM
		label.position.y = y
		var size := _mono().get_multiline_string_size(label.text, HORIZONTAL_ALIGNMENT_CENTER,
			label.width, label.font_size, -1, TextServer.BREAK_MANDATORY | TextServer.BREAK_WORD_BOUND)
		y += size.y * label.pixel_size + 8.0


func set_center(on: bool) -> void:
	if _sheet:
		_sheet.visible = on
	_center_note.text = ("Helped: your drawing is resized and centred, like the digits it was trained on"
		if on else "The network gets exactly what you draw, where you draw it")
	if _titles.size() == 4:
		_stack_labels(0)


## The selected-neuron panel stands to the left of the console. `station` is where the
## visitor stands at the console, facing the network.
func place_panel(station: Transform3D) -> void:
	var local := Transform3D(Basis(), Vector3(-2.2, 1.3, -0.55)).looking_at(Vector3(0, 1.4, 0.6), Vector3.UP, true)
	_panel.global_transform = station * local


# ------------------------------------------------------------------ updates

## Send the current drawing through the network as a visible pulse.
func pulse() -> void:
	pulse_t = -0.15


func pulse_done() -> bool:
	return pulse_t > 3.4


func _process(delta: float) -> void:
	_face_labels()
	sway_t += delta
	set_edge_param("sway_t", sway_t)
	if pulse_t < 4.0:
		pulse_t += delta * pulse_speed
		set_edge_param("pulse_t", pulse_t)
		_recolor()
	if blend < 1.0:
		blend = minf(1.0, blend + delta / blend_time)
		set_edge_param("blend", blend)
		for k in 42:
			pos_now[k] = pos_from[k].lerp(pos_to[k], blend)
			node_inst[k].position = pos_now[k]


func _write_positions(hidden: PackedVector3Array) -> void:
	for i in 784:
		var p := input_pos[i]
		_pos_img.set_pixel(i, 0, Color(p.x, p.y, p.z))
	for k in 42:
		var p := hidden[k]
		_pos_img.set_pixel(784 + k, 0, Color(p.x, p.y, p.z))
	_cur = 1 - _cur
	_pos_tex[_cur].update(_pos_img)


## A snapshot is three frames (layout, edges, nodes). The layout frame starts a new
## blend; edges and nodes write into the same "current" slot. Neurons have fixed places,
## so the server's similarity-map layout itself is not used.
func apply_layout(payload: PackedByteArray) -> void:
	if payload.to_float32_array().size() != 84:
		return
	var target := _fixed_positions()
	pos_from = pos_now.duplicate()
	pos_to = target
	# The newest data moves to the other slot; the edges frame that follows fills it.
	_write_positions(target)
	_bind_textures()
	blend = 0.0
	set_edge_param("blend", 0.0)


func apply_edges(payload: PackedByteArray) -> void:
	if payload.size() != EDGE_TEX_W * EDGE_TEX_H * 8:
		return
	_edge_tex[_cur].update(Image.create_from_data(EDGE_TEX_W, EDGE_TEX_H, false, Image.FORMAT_RGF, payload))
	edge_data = payload.to_float32_array()   # (weight, flow) per edge, for grabbing


func apply_nodes(payload: PackedByteArray) -> void:
	var f := payload.to_float32_array()
	if f.size() != N_NODES:
		return
	show_input(f.slice(0, 784))
	for i in N_NODES:
		acts[i] = f[i]
	_recolor()


func apply_meta(meta: Dictionary) -> void:
	var o: Array = meta.get("off", [[], []])
	# JSON numbers arrive as floats; neuron indices are compared as ints.
	off[1] = (o[0] if o.size() > 0 else []).map(func(v): return int(v))
	off[2] = (o[1] if o.size() > 1 else []).map(func(v): return int(v))
	var pred = meta.get("prediction")
	for i in 10:
		var lit: bool = pred != null and int(pred) == i
		_digit_labels[i].modulate = COL_LIT if lit else COL_TEXT_DIM
	if pred == null:
		_answer.text = ""
		_answer_caption.text = ""
	else:
		_answer.text = str(int(pred))
		_answer_caption.text = "%d%% sure" % int(round(float(meta["probs"][int(pred)]) * 100))
	_recolor()


func _layer_of(k: int) -> Vector2i:
	if k < 16:
		return Vector2i(1, k)
	if k < 32:
		return Vector2i(2, k - 16)
	return Vector2i(3, k - 32)


func _recolor() -> void:
	for k in 42:
		var ln := _layer_of(k)
		var m: ShaderMaterial = node_mats[k]
		if off[ln.x].has(ln.y):
			m.set_shader_parameter("col", Vector3.ZERO)
			m.set_shader_parameter("off", 1.0)
		else:
			# A neuron lights up when the pulse reaches its layer.
			var arrived := smoothstep(float(ln.x) - 0.2, float(ln.x), pulse_t)
			m.set_shader_parameter("col", COL_LIT * clampf(acts[784 + k], 0.0, 1.0) * arrived)
			m.set_shader_parameter("off", 0.0)
		node_inst[k].scale = Vector3.ONE * (1.2 if selected == ln else 1.0)


func set_mode(flow: bool) -> void:
	set_edge_param("mode", 0 if flow else 1)


# ------------------------------------------------------------------ lines, for grabbing

const N_EDGES := 784 * 16 + 16 * 16 + 16 * 10


## Where a node sits: input pixels on the image, neurons at their cube's centre.
func _node_center(id: int) -> Vector3:
	return input_pos[id] if id < 784 else pos_now[id - 784]


## End points of edge `e` (same order and geometry as edge.gdshaderinc), in local space:
## from the pixel or the sender's back face, to the receiver's front-face point.
func edge_ends(e: int) -> Array:
	var layer: int
	var src: int
	var dst: int
	if e < 12544:
		layer = 0; dst = 784 + e / 784; src = e % 784
	elif e < 12800:
		layer = 1; dst = 800 + (e - 12544) / 16; src = 784 + (e - 12544) % 16
	else:
		layer = 2; dst = 816 + (e - 12800) / 16; src = 800 + (e - 12800) % 16
	var a := _node_center(src) + _back_point(layer, dst)
	var n := 28 if layer == 0 else 4
	var i := src if layer == 0 else (src - 784 if layer == 1 else src - 800)
	var f := CUBE * FACE_FILL
	var fp := _on_ball(Vector2(((i % n) + 0.5) / n - 0.5, 0.5 - ((i / n) + 0.5) / n) * f, 1.0)
	return [a, _node_center(dst) + fp]


## Same as on_ball() in edge.gdshaderinc.
func _on_ball(p: Vector2, side: float) -> Vector3:
	var r := BALL * 0.97
	return Vector3(p.x, p.y, side * sqrt(maxf(r * r - p.length_squared(), 0.0)))


## Same as back_point() in edge.gdshaderinc.
func _back_point(layer: int, dst: int) -> Vector3:
	var i := dst - 784 if layer == 0 else (dst - 800 if layer == 1 else dst - 816)
	var size := GRID_SIZE / 28.0 * FACE_FILL if layer == 0 else CUBE * FACE_FILL
	var p := Vector2((((i % 4) + 0.5) / 4.0 - 0.5) * size, (0.5 - ((i / 4) + 0.5) / 4.0) * size)
	return Vector3(p.x, p.y, 0.0) if layer == 0 else _on_ball(p, -1.0)


func edge_width(e: int) -> float:
	var w := absf(edge_data[e * 2]) if edge_data.size() > e * 2 else 0.0
	return maxf(MIN_WIDTH, (MAX_WIDTH_INPUT if e < 12544 else MAX_WIDTH) * w)


## Same slack as edge.gdshaderinc: keep the two in step.
const SLACK := 0.025
var sway_t := 0.0
var _grab_edge := -1          # the held line: frozen, so holding it never moves you
var _grab_sway_t := 0.0


## Point at `s` (0..1) along edge `e`, following its current curve, in local space.
func edge_point(e: int, s: float) -> Vector3:
	var ends := edge_ends(e)
	var a: Vector3 = ends[0]
	var dir: Vector3 = ends[1] - a
	var fwd := dir.normalized()
	var up := Vector3.RIGHT if absf(fwd.y) > 0.95 else Vector3.UP
	var side := fwd.cross(up).normalized()
	var up2 := side.cross(fwd)
	var ph := fposmod(e * 0.618034, 1.0) * TAU
	if e < 12544:   # pixel lines are straight; cables sag and sway
		return a + dir * s
	var st := _grab_sway_t if e == _grab_edge else sway_t
	var bend := SLACK * dir.length() * (Vector3(0, -0.6, 0)
		+ side * sin(st * 0.35 + ph) * 0.8 + up2 * cos(st * 0.27 + ph * 1.3) * 0.8)
	return a + dir * s + bend * sin(PI * s)


## The line a ray points at: {edge, t (along the ray), s (0..1 along the line)}, or {}.
## Lines count as a little thicker the further away they are, so distant ones can be grabbed.
func pick_edge(world_origin: Vector3, world_dir: Vector3) -> Dictionary:
	var o := to_local(world_origin)
	var d := (global_basis.inverse() * world_dir).normalized()
	var best := {}
	var best_t := INF
	for e in N_EDGES:
		var ends := edge_ends(e)
		var a: Vector3 = ends[0]
		var u: Vector3 = ends[1] - a
		var w := o - a
		var uu := u.dot(u)
		var ud := u.dot(d)
		var denom := uu - ud * ud
		if denom < 1e-9:
			continue
		var s := clampf((u.dot(w) - ud * d.dot(w)) / denom, 0.0, 1.0)
		var t := (a + u * s - o).dot(d)
		if t <= 0.0 or t >= best_t:
			continue
		var gap := (o + d * t).distance_to(a + u * s)
		var reach := edge_width(e) / 2 + t * 0.004
		if gap > reach + SLACK * sqrt(uu) * 1.3:
			continue
		# Close to the straight line: check the real curve, in 16 pieces.
		var prev := edge_point(e, 0.0)
		for k in range(1, 17):
			var sk := k / 16.0
			var cur := edge_point(e, sk)
			var su := cur - prev
			var sw := o - prev
			var suu := su.dot(su)
			var sud := su.dot(d)
			var sden := suu - sud * sud
			if sden > 1e-9:
				var ss := clampf((su.dot(sw) - sud * d.dot(sw)) / sden, 0.0, 1.0)
				var st := (prev + su * ss - o).dot(d)
				if st > 0.0 and st < best_t and (o + d * st).distance_to(prev + su * ss) < reach:
					best_t = st
					best = {"edge": e, "t": st, "s": (k - 1 + ss) / 16.0}
			prev = cur
	return best


## The cable whose surface is nearest a point, if within `reach` metres of it:
## {edge, s (0..1 along it), gap (metres from the point to its surface)}, or {}.
func nearest_edge(world_point: Vector3, reach: float) -> Dictionary:
	var p := to_local(world_point)
	var best := {}
	var best_gap := reach
	for e in N_EDGES:
		var ends := edge_ends(e)
		var a: Vector3 = ends[0]
		var u: Vector3 = ends[1] - a
		var uu := u.dot(u)
		if uu < 1e-9:
			continue
		var r := edge_width(e) / 2.0
		# Straight line first; curved cables can stray from it by their slack.
		var s := clampf((p - a).dot(u) / uu, 0.0, 1.0)
		var slack := 0.0 if e < 12544 else SLACK * sqrt(uu) * 1.3
		if p.distance_to(a + u * s) - r - slack > best_gap:
			continue
		if e >= 12544:
			# Refine on the real curve, in 32 steps, then one finer pass.
			var best_s := s
			var best_d := INF
			for k in 33:
				var sk := k / 32.0
				var d := p.distance_to(edge_point(e, sk))
				if d < best_d:
					best_d = d
					best_s = sk
			# Project exactly onto the curve pieces either side of the best sample.
			var s_best := best_s
			for side in [-1.0, 1.0]:
				var s1 := clampf(best_s + side / 32.0, 0.0, 1.0)
				var q0 := edge_point(e, best_s)
				var q1 := edge_point(e, s1)
				var seg := q1 - q0
				var ll := seg.dot(seg)
				if ll < 1e-9:
					continue
				var f := clampf((p - q0).dot(seg) / ll, 0.0, 1.0)
				var d := p.distance_to(q0 + seg * f)
				if d < best_d:
					best_d = d
					s_best = lerpf(best_s, s1, f)
			s = s_best
		var gap := p.distance_to(edge_point(e, s)) - r
		if gap < best_gap:
			best_gap = gap
			best = {"edge": e, "s": s, "gap": maxf(gap, 0.0)}
	return best


## Light up the held line (or none with -1) and freeze its sway where it is now.
func set_grabbed(e: int) -> void:
	_grab_edge = e
	_grab_sway_t = sway_t
	set_edge_param("grab_edge", float(e))
	set_edge_param("grab_sway_t", _grab_sway_t)


# ------------------------------------------------------------------ picking and selection

func pick(world_origin: Vector3, world_dir: Vector3) -> Dictionary:
	var origin := to_local(world_origin)
	var dir := (global_basis.inverse() * world_dir).normalized()
	var best := {}
	var best_t := INF
	for k in 42:
		var oc: Vector3 = origin - pos_now[k]
		var b := oc.dot(dir)
		var c := oc.dot(oc) - PICK_RADIUS * PICK_RADIUS
		var disc := b * b - c
		if disc < 0.0:
			continue
		var t := -b - sqrt(disc)
		if t > 0.0 and t < best_t:
			best_t = t
			var ln := _layer_of(k)
			best = {"t": t, "layer": ln.x, "neuron": ln.y}
	return best


func select(layer: int, neuron: int) -> void:
	if selected == Vector2i(layer, neuron):
		layer = -1
		neuron = -1
	selected = Vector2i(layer, neuron)
	if layer < 0:
		set_edge_param("focus_node", -1.0)
		_panel.visible = false
	else:
		set_edge_param("focus_node", float(NODE_OFFSET[layer] + neuron))
		_panel.visible = true
		var names := ["", "Hidden layer 1", "Hidden layer 2", "Answer"]
		_panel_title.text = "%s · neuron %d" % [names[layer], neuron + 1] if layer < 3 else "Answer neuron for %d" % neuron
		_panel_quad.visible = layer == 1
		if layer == 1:
			_panel_caption.text = "What it looks for. Ink on orange pixels switches it on, ink on blue pixels switches it off. Squeeze the grip to switch it off."
		elif layer == 2:
			_panel_caption.text = "Its connections are highlighted in the network. Squeeze the grip to switch it off."
		else:
			_panel_caption.text = "Its connections are highlighted in the network."
	_recolor()


func show_neuron_weights(values: Array) -> void:
	var bytes := PackedByteArray()
	bytes.resize(784 * 3)
	var bg := Color(0.07, 0.08, 0.1)
	for i in 784:
		var v: float = values[i]
		var c := bg.lerp(COL_POS if v >= 0.0 else COL_NEG, clampf(absf(v), 0.0, 1.0))
		bytes[i * 3] = int(c.r * 255)
		bytes[i * 3 + 1] = int(c.g * 255)
		bytes[i * 3 + 2] = int(c.b * 255)
	_panel_img.set_data(28, 28, false, Image.FORMAT_RGB8, bytes)
	_panel_tex.update(_panel_img)
