## The canvas the visitor draws on: the network's own 28x28 input pixels, values 0..1.
## Shown on the console board and painted straight onto the input image in the scene.
extends Node3D

signal changed

const RES := 28
const BRUSH := 1.2  # radius of the solid core, in pixels
const SOFT := 1.0   # width of the soft edge, in pixels (MNIST strokes have grey edges)

var size := 0.45
var canvas := PackedFloat32Array()
var dirty := false
var _img: Image
var _tex: ImageTexture
var _last_uv := Vector2(-1, -1)


func build() -> void:
	canvas.resize(RES * RES)
	canvas.fill(0.0)
	var frame := MeshInstance3D.new()
	var fq := QuadMesh.new()
	fq.size = Vector2(size + 0.04, size + 0.04)
	frame.mesh = fq
	var fm := StandardMaterial3D.new()
	fm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	fm.albedo_color = Color(0.2, 1.0, 0.4)   # terminal green, like the console
	frame.material_override = fm
	frame.position.z = -0.003
	add_child(frame)

	_img = Image.create(RES, RES, false, Image.FORMAT_L8)
	_tex = ImageTexture.create_from_image(_img)
	var board := MeshInstance3D.new()
	var q := QuadMesh.new()
	q.size = Vector2(size, size)
	board.mesh = q
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.albedo_texture = _tex
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	board.material_override = m
	add_child(board)

	var label := Label3D.new()
	label.text = "Draw a digit here"
	label.font_size = 40
	label.pixel_size = 0.0015
	label.outline_size = 0
	label.modulate = Color(0.85, 0.88, 0.92)
	label.position = Vector3(0, size / 2 + 0.03, 0)
	label.visible = false
	add_child(label)


## The canvas as a texture (dark 0.07 .. light 1.0), so other surfaces can show it too.
func texture() -> ImageTexture:
	return _tex


## Canvas coordinates (0..1) for a world point on the board plane, or (-1,-1) if outside.
func uv_at(world_p: Vector3) -> Vector2:
	var p := to_local(world_p)
	var u := p.x / size + 0.5
	var v := 0.5 - p.y / size
	if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
		return Vector2(-1, -1)
	return Vector2(u, v)


func begin_stroke(uv: Vector2) -> void:
	_last_uv = uv
	_stamp(uv)
	_refresh()


func continue_stroke(uv: Vector2) -> void:
	if _last_uv.x < 0.0:
		begin_stroke(uv)
		return
	var a := _last_uv * RES
	var b := uv * RES
	var steps := int(ceil(a.distance_to(b) / 0.5))
	for i in range(1, steps + 1):
		_stamp(_last_uv.lerp(uv, float(i) / steps))
	_last_uv = uv
	_refresh()


func end_stroke() -> void:
	_last_uv = Vector2(-1, -1)


func clear() -> void:
	canvas.fill(0.0)
	_last_uv = Vector2(-1, -1)
	_refresh()


func values() -> Array:
	var out := []
	out.resize(canvas.size())
	for i in canvas.size():
		out[i] = snappedf(canvas[i], 0.01)
	return out


func _stamp(uv: Vector2) -> void:
	var c := uv * RES
	var r := int(ceil(BRUSH + 1))
	for y in range(int(c.y) - r, int(c.y) + r + 1):
		if y < 0 or y >= RES:
			continue
		for x in range(int(c.x) - r, int(c.x) + r + 1):
			if x < 0 or x >= RES:
				continue
			var d := Vector2(x + 0.5, y + 0.5).distance_to(c)
			var v := clampf((BRUSH + SOFT / 2 - d) / SOFT, 0.0, 1.0)
			var i := y * RES + x
			if v > canvas[i]:
				canvas[i] = v


func _refresh() -> void:
	var bytes := PackedByteArray()
	bytes.resize(RES * RES)
	for i in canvas.size():
		# Dark board, light chalk.
		bytes[i] = int(18 + canvas[i] * 237)
	_img.set_data(RES, RES, false, Image.FORMAT_L8, bytes)
	_tex.update(_img)
	dirty = true
	changed.emit()
