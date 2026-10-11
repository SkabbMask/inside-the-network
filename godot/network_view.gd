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

const FLOOR_CENTER := Vector3(0, 0.01, -1.35)
const FLOOR_SIZE := 1.6
const LAYER_Z := [0.0, -3.6, -6.2, -8.8]
const LAYER_Y := 1.9
const LAYER_RADIUS := 1.5   # the farthest a neuron gets from its layer's centre, in metres
const NODE_RADIUS := 0.055
const PICK_RADIUS := 0.09

const COL_POS := Color(1.0, 0.55, 0.18)
const COL_NEG := Color(0.25, 0.6, 1.0)
const COL_IDLE := Color(0.16, 0.17, 0.21)
const COL_LIT := Color(1.0, 0.96, 0.86)
const COL_OFF := Color(0.55, 0.08, 0.12)
const COL_TEXT := Color(0.75, 0.79, 0.86)

var input_pos := PackedVector3Array()
var pos_from := PackedVector3Array()   # hidden + output neurons, 42 each
var pos_to := PackedVector3Array()
var pos_now := PackedVector3Array()
var blend := 1.0
var blend_time := 0.15
var layer_scale := [0.0, 2.5, 2.5, 2.5]   # set per run from the server's extents
var pulse_t := 10.0
var pulse_speed := 1.6   # layers per second

var node_mats := []
var node_inst := []
var acts := PackedFloat32Array()
var off := [[], [], [], []]
var selected := Vector2i(-1, -1)

var edge_mat: ShaderMaterial
var _edge_tex := []   # two ImageTextures, alternating
var _pos_tex := []
var _cur := 0         # which of the two holds the newest data
var _pos_img: Image
var _input_img: Image
var _input_tex: ImageTexture
var _answer: Label3D
var _answer_caption: Label3D
var _digit_labels := []
var _layer_labels := []
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
		var x := (c + 0.5) / 28.0 * FLOOR_SIZE - FLOOR_SIZE / 2
		var z := FLOOR_CENTER.z - FLOOR_SIZE / 2 + (r + 0.5) / 28.0 * FLOOR_SIZE
		input_pos.append(Vector3(x, FLOOR_CENTER.y, z))
	# Before the first layout arrives, put each layer on a small circle.
	for l in range(1, 4):
		for i in SIZES[l]:
			var a: float = TAU * float(i) / float(SIZES[l])
			var p := Vector3(cos(a) * 0.3, LAYER_Y + sin(a) * 0.3, LAYER_Z[l])
			pos_from.append(p)
			pos_to.append(p)
			pos_now.append(p)
	acts.resize(N_NODES)
	acts.fill(0.0)
	_pos_img = Image.create(N_NODES, 1, false, Image.FORMAT_RGBF)
	_build_floor()
	_build_edges()
	_build_nodes()
	_build_labels()
	_build_panel()
	_write_positions(pos_to)
	_write_positions(pos_to)
	_bind_textures()


# ------------------------------------------------------------------ construction

func _unshaded(color: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.albedo_color = color
	return m


func _label(text: String, pos: Vector3, size := 40, color := COL_TEXT, billboard := false) -> Label3D:
	var l := Label3D.new()
	l.text = text
	l.font_size = size
	l.pixel_size = 0.0035
	l.outline_size = 0
	l.modulate = color
	l.position = pos
	if billboard:
		l.billboard = BaseMaterial3D.BILLBOARD_FIXED_Y
	add_child(l)
	return l


func _build_floor() -> void:
	_input_img = Image.create(28, 28, false, Image.FORMAT_L8)
	_input_tex = ImageTexture.create_from_image(_input_img)
	var border := MeshInstance3D.new()
	var bp := PlaneMesh.new()
	bp.size = Vector2(FLOOR_SIZE + 0.06, FLOOR_SIZE + 0.06)
	border.mesh = bp
	border.material_override = _unshaded(Color(0.22, 0.24, 0.3))
	border.position = FLOOR_CENTER - Vector3(0, 0.004, 0)
	add_child(border)
	var tiles := MeshInstance3D.new()
	var p := PlaneMesh.new()
	p.size = Vector2(FLOOR_SIZE, FLOOR_SIZE)
	tiles.mesh = p
	var m := _unshaded(Color.WHITE)
	m.albedo_texture = _input_tex
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	tiles.material_override = m
	tiles.position = FLOOR_CENTER
	add_child(tiles)
	var l := _label("What the network sees: 28 × 28 = 784 pixels", FLOOR_CENTER + Vector3(0, 0.005, FLOOR_SIZE / 2 + 0.1), 40)
	l.rotation.x = -PI / 2


func _build_edges() -> void:
	var mesh := BoxMesh.new()
	mesh.size = Vector3.ONE
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_custom_data = true
	mm.mesh = mesh
	mm.instance_count = 784 * 16 + 16 * 16 + 16 * 10
	var idx := 0
	for l in range(1, 4):
		for j in SIZES[l]:
			for i in SIZES[l - 1]:
				mm.set_instance_transform(idx, Transform3D.IDENTITY)
				mm.set_instance_custom_data(idx, Color(NODE_OFFSET[l - 1] + i, NODE_OFFSET[l] + j, l - 1, 0))
				idx += 1
	var inst := MultiMeshInstance3D.new()
	inst.multimesh = mm
	inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	# The shader moves every vertex, so tell the engine where the edges can be.
	inst.custom_aabb = AABB(Vector3(-3, -1, -11), Vector3(6, 7, 12))
	edge_mat = ShaderMaterial.new()
	edge_mat.shader = load("res://edge.gdshader")
	for k in 2:
		_edge_tex.append(ImageTexture.create_from_image(Image.create(EDGE_TEX_W, EDGE_TEX_H, false, Image.FORMAT_RGF)))
		_pos_tex.append(ImageTexture.create_from_image(_pos_img))
	edge_mat.set_shader_parameter("tex_w", EDGE_TEX_W)
	edge_mat.set_shader_parameter("pos_col", Vector3(COL_POS.r, COL_POS.g, COL_POS.b))
	edge_mat.set_shader_parameter("neg_col", Vector3(COL_NEG.r, COL_NEG.g, COL_NEG.b))
	_bind_textures()
	inst.material_override = edge_mat
	add_child(inst)


func _bind_textures() -> void:
	var prev := 1 - _cur
	edge_mat.set_shader_parameter("edge_a", _edge_tex[prev])
	edge_mat.set_shader_parameter("edge_b", _edge_tex[_cur])
	edge_mat.set_shader_parameter("pos_a", _pos_tex[prev])
	edge_mat.set_shader_parameter("pos_b", _pos_tex[_cur])


func _build_nodes() -> void:
	var sphere := SphereMesh.new()
	sphere.radius = NODE_RADIUS
	sphere.height = NODE_RADIUS * 2
	sphere.radial_segments = 16
	sphere.rings = 8
	for k in 42:
		var mi := MeshInstance3D.new()
		mi.mesh = sphere
		var m := _unshaded(COL_IDLE)
		mi.material_override = m
		mi.position = pos_now[k]
		add_child(mi)
		node_mats.append(m)
		node_inst.append(mi)
	for d in 10:
		var l := Label3D.new()
		l.text = str(d)
		l.font_size = 64
		l.pixel_size = 0.002
		l.outline_size = 0
		l.modulate = Color(0.5, 0.54, 0.6)
		l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		l.position = Vector3(0.09, 0, 0)
		node_inst[32 + d].add_child(l)
		_digit_labels.append(l)


func _build_labels() -> void:
	var names := ["", "Hidden layer 1 · 16 neurons", "Hidden layer 2 · 16 neurons", "Answer · one neuron per digit"]
	for l in range(1, 4):
		_layer_labels.append(_label(names[l], Vector3(0, LAYER_Y + LAYER_RADIUS + 0.3, LAYER_Z[l]), 44, COL_TEXT, true))
	_label("Within a layer, neurons that react to the same drawings sit close together.",
		Vector3(0, LAYER_Y + LAYER_RADIUS + 0.6, LAYER_Z[1]), 30, Color(0.6, 0.64, 0.72), true)
	_answer = _label("", Vector3(0, LAYER_Y + 0.2, LAYER_Z[3] - 1.2), 200, COL_LIT, true)
	_answer_caption = _label("", Vector3(0, LAYER_Y - 0.35, LAYER_Z[3] - 1.2), 40, COL_TEXT, true)


func _build_panel() -> void:
	_panel = Node3D.new()
	_panel.top_level = true
	add_child(_panel)
	_panel.global_position = Vector3(-1.75, 1.35, -0.75)
	_panel.look_at(Vector3(0, 1.4, 0.6), Vector3.UP, true)
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
	_panel_title.font_size = 36
	_panel_title.pixel_size = 0.0015
	_panel_title.outline_size = 0
	_panel_title.position = Vector3(0, 0.27, 0)
	_panel.add_child(_panel_title)
	_panel_caption = Label3D.new()
	_panel_caption.font_size = 28
	_panel_caption.pixel_size = 0.0015
	_panel_caption.outline_size = 0
	_panel_caption.modulate = Color(0.7, 0.74, 0.8)
	_panel_caption.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_panel_caption.width = 300
	_panel_caption.position = Vector3(0, -0.32, 0)
	_panel.add_child(_panel_caption)
	_panel.visible = false


# ------------------------------------------------------------------ updates

func set_extent(extent: Array) -> void:
	for l in 3:
		layer_scale[l + 1] = LAYER_RADIUS / maxf(float(extent[l]), 0.05)


## Send the current drawing through the network as a visible pulse.
func pulse() -> void:
	pulse_t = -0.15


func pulse_done() -> bool:
	return pulse_t > 3.4


func _process(delta: float) -> void:
	if pulse_t < 4.0:
		pulse_t += delta * pulse_speed
		edge_mat.set_shader_parameter("pulse_t", pulse_t)
		_recolor()
	if blend < 1.0:
		blend = minf(1.0, blend + delta / blend_time)
		edge_mat.set_shader_parameter("blend", blend)
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
## blend; edges and nodes write into the same "current" slot.
func apply_layout(payload: PackedByteArray) -> void:
	var f := payload.to_float32_array()
	if f.size() != 84:
		return
	var target := PackedVector3Array()
	var k := 0
	for l in range(1, 4):
		var s: float = layer_scale[l]
		for i in SIZES[l]:
			target.append(Vector3(f[k * 2] * s, LAYER_Y + f[k * 2 + 1] * s, LAYER_Z[l]))
			k += 1
	pos_from = pos_now.duplicate()
	pos_to = target
	# The newest data moves to the other slot; the edges frame that follows fills it.
	_write_positions(target)
	_bind_textures()
	blend = 0.0
	edge_mat.set_shader_parameter("blend", 0.0)


func apply_edges(payload: PackedByteArray) -> void:
	if payload.size() != EDGE_TEX_W * EDGE_TEX_H * 8:
		return
	_edge_tex[_cur].update(Image.create_from_data(EDGE_TEX_W, EDGE_TEX_H, false, Image.FORMAT_RGF, payload))


func apply_nodes(payload: PackedByteArray) -> void:
	var f := payload.to_float32_array()
	if f.size() != N_NODES:
		return
	var bytes := PackedByteArray()
	bytes.resize(784)
	for i in 784:
		bytes[i] = int(18 + clampf(f[i], 0.0, 1.0) * 237.0)
	_input_img.set_data(28, 28, false, Image.FORMAT_L8, bytes)
	_input_tex.update(_input_img)
	for i in N_NODES:
		acts[i] = f[i]
	_recolor()


func apply_meta(meta: Dictionary) -> void:
	var o: Array = meta.get("off", [[], []])
	off[1] = o[0] if o.size() > 0 else []
	off[2] = o[1] if o.size() > 1 else []
	var pred = meta.get("prediction")
	for i in 10:
		var lit: bool = pred != null and int(pred) == i
		_digit_labels[i].modulate = COL_LIT if lit else Color(0.5, 0.54, 0.6)
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
		var m: StandardMaterial3D = node_mats[k]
		if off[ln.x].has(ln.y):
			m.albedo_color = COL_OFF
		else:
			# A neuron lights up when the pulse reaches its layer.
			var arrived := smoothstep(float(ln.x) - 0.2, float(ln.x), pulse_t)
			m.albedo_color = COL_IDLE.lerp(COL_LIT, clampf(acts[784 + k], 0.0, 1.0) * arrived)
		node_inst[k].scale = Vector3.ONE * (1.7 if selected == ln else 1.0)


func set_mode(flow: bool) -> void:
	edge_mat.set_shader_parameter("mode", 0 if flow else 1)


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
		edge_mat.set_shader_parameter("focus_node", -1.0)
		_panel.visible = false
	else:
		edge_mat.set_shader_parameter("focus_node", float(NODE_OFFSET[layer] + neuron))
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
