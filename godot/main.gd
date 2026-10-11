## Inside the Network: scene setup, input (VR or desktop), buttons and the training timeline.
##
## Command-line options (after `--`):
##   --server=ws://host:port   model server address (default ws://127.0.0.1:8765)
##   --screenshot=path.png     (testing) draw a demo digit, save a screenshot, quit
##   --at=k                    (testing) with --screenshot: show snapshot k
## VR is used when OpenXR is enabled in Project Settings and a headset is found.
extends Node3D

const NetClient := preload("res://net_client.gd")
const NetworkView := preload("res://network_view.gd")
const DrawPad := preload("res://draw_pad.gd")

const PANEL_Z := -0.4
const SEND_INTERVAL := 0.08
const SPEEDS := [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
const SNAPSHOTS_PER_SECOND := 4.5   # at speed x1
const BAR_X0 := -1.3
const BAR_X1 := -0.38
const BAR_Y := 0.9
const SPARK_H := 0.1

const COL_BTN := Color(0.17, 0.19, 0.24)
const COL_BTN_HOVER := Color(0.3, 0.33, 0.4)
const COL_TEXT := Color(0.85, 0.88, 0.92)

var server_url := "ws://127.0.0.1:8765"
var screenshot_path := ""
var screenshot_at := -1
var screenshot_pulse := -1.0
var screenshot_view := 1

var client
var view
var pad
var vr := false
var flow_mode := true
var connected := false
var recording := false
var _last_send := 0.0
var _status: Label3D
var _time_label: Label3D
var _buttons := []      # {id, rect: Rect2, label: Label3D, mat}
var _pointers := []     # {ctrl, laser, trigger, grip, scrubbing}

# timeline
var steps := []
var accuracy := []
var playhead := 0.0
var playing := false
var direction := 1
var speed_i := 3
var shown_index := -1
var _in_flight := false
var _in_flight_since := 0.0
var _jump_on_meta := false
var _pulse_on_meta := true
var _drawn := false
var _spark: MeshInstance3D
var _playhead_mark: MeshInstance3D

# desktop camera
var _cam: Camera3D
var _yaw := 0.0
var _pitch := 0.0
var _mouse_left := false
var _mouse_right := false


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--server="):
			server_url = a.trim_prefix("--server=")
		elif a.begins_with("--screenshot="):
			screenshot_path = a.trim_prefix("--screenshot=")
		elif a.begins_with("--pulse="):
			screenshot_pulse = float(a.trim_prefix("--pulse="))
		elif a.begins_with("--view="):
			screenshot_view = int(a.trim_prefix("--view="))
		elif a.begins_with("--at="):
			screenshot_at = int(a.trim_prefix("--at="))

	_build_world()
	view = NetworkView.new()
	add_child(view)
	# The network runs diagonally away to the front right, so you see both its depth
	# and the face of each layer from where you start.
	view.rotation.y = deg_to_rad(-60)
	view.position = Vector3(-0.8, 0, -1.9)
	view.build()

	pad = DrawPad.new()
	add_child(pad)
	pad.size = 0.36
	pad.position = Vector3(-1.0, 1.32, PANEL_Z)
	pad.build()

	_build_buttons()
	_build_timeline()
	_status = _make_label("", Vector3(-1.0, 1.55, PANEL_Z), 26)

	client = NetClient.new()
	add_child(client)
	client.binary_received.connect(_on_binary)
	client.json_received.connect(_on_json)
	client.connection_changed.connect(_on_connection)
	client.connect_to(server_url)

	var xr := XRServer.find_interface("OpenXR")
	if screenshot_path == "" and xr and xr.is_initialized():
		_setup_vr()
	else:
		_setup_desktop()
	_update_status()
	if screenshot_path != "":
		_run_screenshot_demo()


# ------------------------------------------------------------------ world and panels

func _make_label(text: String, pos: Vector3, size := 30, color := COL_TEXT) -> Label3D:
	var l := Label3D.new()
	l.text = text
	l.font_size = size
	l.pixel_size = 0.0015
	l.outline_size = 0
	l.modulate = color
	l.position = pos
	add_child(l)
	return l


func _flat(color: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.albedo_color = color
	return m


func _box(size: Vector3, pos: Vector3, color: Color) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var b := BoxMesh.new()
	b.size = size
	mi.mesh = b
	mi.material_override = _flat(color)
	mi.position = pos
	add_child(mi)
	return mi


func _build_world() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.03, 0.035, 0.05)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.4, 0.42, 0.5)
	env.glow_enabled = true
	env.glow_intensity = 0.6
	env.glow_bloom = 0.05
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	var floor_mi := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(16, 16)
	floor_mi.mesh = plane
	floor_mi.material_override = _flat(Color(0.055, 0.06, 0.075))
	floor_mi.position.y = -0.002
	add_child(floor_mi)


func _add_button(id: String, text: String, center: Vector2, size: Vector2) -> void:
	var mi := _box(Vector3(size.x, size.y, 0.02), Vector3(center.x, center.y, PANEL_Z - 0.01), COL_BTN)
	var l := _make_label(text, Vector3(center.x, center.y, PANEL_Z + 0.002), 22)
	_buttons.append({"id": id, "rect": Rect2(center - size / 2, size), "label": l, "mat": mi.material_override})


func _build_buttons() -> void:
	var defs := [
		["clear", "Clear board"],
		["view", "Show all weights"],
		["train", "Train from scratch"],
		["reset", "Standard run"],
	]
	var y := 1.47
	for d in defs:
		_add_button(d[0], d[1], Vector2(-0.53, y), Vector2(0.27, 0.07))
		y -= 0.09


func _build_timeline() -> void:
	var w := BAR_X1 - BAR_X0
	_box(Vector3(w, 0.012, 0.01), Vector3((BAR_X0 + BAR_X1) / 2, BAR_Y, PANEL_Z - 0.01), Color(0.25, 0.28, 0.34))
	_playhead_mark = _box(Vector3(0.006, SPARK_H + 0.04, 0.012), Vector3(BAR_X0, BAR_Y + SPARK_H / 2, PANEL_Z), Color(1.0, 0.85, 0.5))
	_spark = MeshInstance3D.new()
	_spark.material_override = _flat(Color(0.55, 0.85, 0.65))
	add_child(_spark)
	_time_label = _make_label("", Vector3((BAR_X0 + BAR_X1) / 2, BAR_Y + SPARK_H + 0.04, PANEL_Z), 24)
	_make_label("Correct answers on test digits, through training", Vector3((BAR_X0 + BAR_X1) / 2, BAR_Y - 0.035, PANEL_Z), 20, Color(0.6, 0.64, 0.7))
	var ids := [["start", "Start"], ["reverse", "Reverse"], ["play", "Play"], ["slower", "Slower"], ["faster", "Faster"], ["end", "End"]]
	var bw := w / ids.size()
	for i in ids.size():
		_add_button(ids[i][0], ids[i][1], Vector2(BAR_X0 + bw * (i + 0.5), BAR_Y - 0.09), Vector2(bw - 0.008, 0.06))


func _rebuild_spark() -> void:
	var im := ImmediateMesh.new()
	if steps.size() > 1:
		im.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
		for i in steps.size():
			var x := BAR_X0 + (BAR_X1 - BAR_X0) * i / float(steps.size() - 1)
			im.surface_add_vertex(Vector3(x, BAR_Y + 0.01 + float(accuracy[i]) * SPARK_H, PANEL_Z))
		im.surface_end()
	_spark.mesh = im


func _press_button(id: String) -> void:
	match id:
		"clear":
			pad.clear()
			pad.dirty = false
			client.send({"type": "clear"})
		"view":
			flow_mode = not flow_mode
			view.set_mode(flow_mode)
			_button("view").label.text = "Show all weights" if flow_mode else "Show my drawing"
		"train":
			client.send({"type": "retrain"})
		"reset":
			client.send({"type": "reset"})
		"play":
			if not playing and _at_end():
				playhead = 0.0 if direction > 0 else float(steps.size() - 1)
			playing = not playing
		"reverse":
			direction = -direction
			playing = true
		"slower":
			speed_i = maxi(0, speed_i - 1)
		"faster":
			speed_i = mini(SPEEDS.size() - 1, speed_i + 1)
		"start":
			playing = false
			playhead = 0.0
			_pulse_on_meta = true
		"end":
			playing = false
			playhead = float(steps.size() - 1)
			_pulse_on_meta = true
	_update_status()


func _at_end() -> bool:
	if direction > 0:
		return playhead >= steps.size() - 1
	return playhead <= 0.0


func _button(id: String) -> Dictionary:
	for b in _buttons:
		if b.id == id:
			return b
	return {}


# ------------------------------------------------------------------ VR and desktop

func _setup_vr() -> void:
	vr = true
	get_viewport().use_xr = true
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	var origin := XROrigin3D.new()
	add_child(origin)
	origin.add_child(XRCamera3D.new())
	for hand in ["left_hand", "right_hand"]:
		var ctrl := XRController3D.new()
		ctrl.tracker = hand
		ctrl.pose = "aim"
		origin.add_child(ctrl)
		var grip_box := MeshInstance3D.new()
		var b := BoxMesh.new()
		b.size = Vector3(0.03, 0.03, 0.1)
		grip_box.mesh = b
		grip_box.position.z = 0.05
		ctrl.add_child(grip_box)
		var laser := MeshInstance3D.new()
		var lb := BoxMesh.new()
		lb.size = Vector3(0.003, 0.003, 1.0)
		laser.mesh = lb
		var lm := _flat(Color(1.0, 0.85, 0.5, 0.7))
		lm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		laser.material_override = lm
		ctrl.add_child(laser)
		_pointers.append({"ctrl": ctrl, "laser": laser, "trigger": false, "grip": false, "scrubbing": false})


func _setup_desktop() -> void:
	_cam = Camera3D.new()
	_cam.fov = 75
	add_child(_cam)
	_set_camera_view(1)
	_cam.make_current()
	_pointers.append({"ctrl": null, "laser": null, "trigger": false, "grip": false, "scrubbing": false})


## Desktop camera presets: 1 = at the console (where a VR visitor starts),
## 2 = side-on to the network, to see all layers apart.
func _set_camera_view(n: int) -> void:
	if n == 1:
		_cam.position = Vector3(0.0, 2.0, 2.4)
		_cam.look_at(Vector3(2.8, 1.5, -4.6))
	else:
		var mid: Vector3 = view.to_global(Vector3(0, 1.9, -6.2))
		var side: Vector3 = view.global_basis * Vector3(1, 0, 0)
		_cam.position = mid + side * 8.0 + Vector3(0, 0.6, 0)
		_cam.look_at(mid)
	_yaw = _cam.rotation.y
	_pitch = _cam.rotation.x


func _unhandled_input(event: InputEvent) -> void:
	if vr:
		return
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			_mouse_left = event.pressed
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			_mouse_right = event.pressed
	elif event is InputEventMouseMotion and Input.is_mouse_button_pressed(MOUSE_BUTTON_MIDDLE):
		_yaw -= event.relative.x * 0.004
		_pitch = clampf(_pitch - event.relative.y * 0.004, -1.3, 1.3)
		_cam.rotation = Vector3(_pitch, _yaw, 0)
	elif event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_SPACE:
				_press_button("play")
			KEY_1:
				_set_camera_view(1)
			KEY_2:
				_set_camera_view(2)


func _process(delta: float) -> void:
	if not vr and _cam:
		var move := Vector3.ZERO
		if Input.is_key_pressed(KEY_W): move.z -= 1
		if Input.is_key_pressed(KEY_S): move.z += 1
		if Input.is_key_pressed(KEY_A): move.x -= 1
		if Input.is_key_pressed(KEY_D): move.x += 1
		if Input.is_key_pressed(KEY_Q): move.y -= 1
		if Input.is_key_pressed(KEY_E): move.y += 1
		if move != Vector3.ZERO:
			_cam.position += (Basis(Vector3.UP, _yaw) * move.normalized()) * delta * 2.0

	for p in _pointers:
		_update_pointer(p)
	_advance_timeline(delta)

	var now := Time.get_ticks_msec() / 1000.0
	if pad.dirty and now - _last_send > SEND_INTERVAL and client.is_open and not _in_flight:
		client.send({"type": "draw", "canvas": pad.values()})
		pad.dirty = false
		_last_send = now


func _advance_timeline(delta: float) -> void:
	if steps.is_empty():
		return
	var n := steps.size()
	var rate: float = SPEEDS[speed_i] * SNAPSHOTS_PER_SECOND
	if playing:
		playhead += direction * rate * delta
		if playhead <= 0.0 or playhead >= n - 1:
			playhead = clampf(playhead, 0.0, n - 1)
			playing = false
			_update_status()
	view.blend_time = clampf(1.0 / rate, 0.06, 0.5) if playing else 0.25
	_playhead_mark.position.x = BAR_X0 + (BAR_X1 - BAR_X0) * playhead / float(maxi(n - 1, 1))

	var now := Time.get_ticks_msec() / 1000.0
	if _in_flight and now - _in_flight_since > 1.0:
		_in_flight = false
	# While playing, the drawing keeps pulsing through, like a heartbeat.
	if playing and _drawn and view.pulse_done():
		view.pulse()
	var want := int(round(playhead))
	if want != shown_index and not _in_flight and client.is_open and not recording:
		client.send({"type": "seek", "index": want})
		_in_flight = true
		_in_flight_since = now


func _update_pointer(p: Dictionary) -> void:
	var origin: Vector3
	var dir: Vector3
	var trig: bool
	var grip: bool
	if p.ctrl:
		var ctrl: XRController3D = p.ctrl
		origin = ctrl.global_position
		dir = -ctrl.global_basis.z
		trig = ctrl.get_float("trigger") > (0.4 if p.trigger else 0.6)
		grip = ctrl.get_float("grip") > (0.5 if p.grip else 0.7)
	else:
		var mp := get_viewport().get_mouse_position()
		origin = _cam.project_ray_origin(mp)
		dir = _cam.project_ray_normal(mp)
		trig = _mouse_left
		grip = _mouse_right

	var h := _hit(origin, dir)
	if p.laser:
		var length: float = h.t if h.has("t") else 6.0
		p.laser.scale = Vector3(1, 1, length)
		p.laser.position = Vector3(0, 0, -length / 2)

	var kind: String = h.get("kind", "")
	var trig_down: bool = trig and not p.trigger
	if trig_down:
		match kind:
			"button":
				_press_button(h.id)
			"node":
				view.select(h.layer, h.neuron)
				if view.selected.x == 1:
					client.send({"type": "neuron_weights", "layer": 1, "neuron": h.neuron})
			"pad":
				pad.begin_stroke(h.uv)
			"scrub":
				p.scrubbing = true
				playing = false
	if trig and p.scrubbing and h.has("frac"):
		playhead = h.frac * (steps.size() - 1)
	elif trig and kind == "pad" and not trig_down:
		pad.continue_stroke(h.uv)
	if not trig:
		if p.trigger:
			pad.end_stroke()
			_pulse_on_meta = true
		p.scrubbing = false
	if grip and not p.grip and kind == "node" and h.layer < 3:
		client.send({"type": "toggle_neuron", "layer": h.layer, "neuron": h.neuron})
		_pulse_on_meta = true
	p.trigger = trig
	p.grip = grip

	for b in _buttons:
		var hover: bool = kind == "button" and h.id == b.id
		b.mat.albedo_color = COL_BTN_HOVER if hover else COL_BTN


func _hit(origin: Vector3, dir: Vector3) -> Dictionary:
	var best := {}
	if absf(dir.z) > 1e-5:
		var t := (PANEL_Z - origin.z) / dir.z
		if t > 0.0:
			var pt := origin + dir * t
			var uv: Vector2 = pad.uv_at(pt)
			if uv.x >= 0.0:
				best = {"t": t, "kind": "pad", "uv": uv}
			else:
				for b in _buttons:
					if b.rect.has_point(Vector2(pt.x, pt.y)):
						best = {"t": t, "kind": "button", "id": b.id}
				var scrub := Rect2(BAR_X0 - 0.02, BAR_Y - 0.02, BAR_X1 - BAR_X0 + 0.04, SPARK_H + 0.06)
				if best.is_empty() and scrub.has_point(Vector2(pt.x, pt.y)):
					best = {"t": t, "kind": "scrub", "frac": clampf((pt.x - BAR_X0) / (BAR_X1 - BAR_X0), 0.0, 1.0)}
			# Keep reporting a scrub position while dragging past the bar ends.
			if not best.has("frac"):
				best["frac"] = clampf((pt.x - BAR_X0) / (BAR_X1 - BAR_X0), 0.0, 1.0)
	var n: Dictionary = view.pick(origin, dir)
	if not n.is_empty() and (not best.has("kind") or n.t < best.t):
		var f = best.get("frac")
		best = {"t": n.t, "kind": "node", "layer": n.layer, "neuron": n.neuron}
		if f != null:
			best["frac"] = f
	return best


# ------------------------------------------------------------------ server messages

func _on_connection(ok: bool) -> void:
	connected = ok
	_in_flight = false
	shown_index = -1
	if ok:
		pad.dirty = true
	_update_status()


func _on_binary(kind: int, payload: PackedByteArray) -> void:
	match kind:
		3:
			view.apply_layout(payload)
		1:
			view.apply_edges(payload)
		2:
			view.apply_nodes(payload)


func _on_json(msg: Dictionary) -> void:
	match msg.get("type", ""):
		"timeline":
			steps = msg["steps"]
			accuracy = msg["accuracy"]
			view.set_extent(msg["extent"])
			recording = false
			_jump_on_meta = true
			_rebuild_spark()
		"recording":
			recording = true
			playing = false
		"meta":
			_in_flight = false
			shown_index = int(msg["index"])
			if _jump_on_meta:
				_jump_on_meta = false
				playhead = float(shown_index)
				# A fresh run starts at step 0: play it.
				if shown_index == 0:
					direction = 1
					playing = true
			view.apply_meta(msg)
			_drawn = msg.get("drawn", false)
			if _pulse_on_meta:
				_pulse_on_meta = false
				view.pulse()
			if view.selected.x == 1:
				client.send({"type": "neuron_weights", "layer": 1, "neuron": view.selected.y})
		"neuron_weights":
			if view.selected == Vector2i(int(msg.layer), int(msg.neuron)):
				view.show_neuron_weights(msg["values"])
	_update_status()


func _update_status() -> void:
	if not connected:
		_status.text = "Waiting for the model server…"
		_time_label.text = ""
		return
	if recording:
		_status.text = "Recording a new training run from random weights…"
		return
	_status.text = "Draw a digit on the board"
	_button("play").label.text = "Pause" if playing else "Play"
	if steps.is_empty() or shown_index < 0:
		return
	var i := clampi(shown_index, 0, steps.size() - 1)
	var arrow := "" if not playing else ("  ▶" if direction > 0 else "  ◀")
	_time_label.text = "Training step %d of %d   ·   %d%% correct   ·   speed ×%s%s" % [
		int(steps[i]), int(steps[-1]), int(round(float(accuracy[i]) * 100)), _speed_text(), arrow]


func _speed_text() -> String:
	var s: float = SPEEDS[speed_i]
	if s >= 1.0:
		return str(int(s))
	return "1/%d" % int(round(1.0 / s))


# ------------------------------------------------------------------ testing helper

func _run_screenshot_demo() -> void:
	var t0 := Time.get_ticks_msec()
	while steps.is_empty() and Time.get_ticks_msec() - t0 < 10000:
		await get_tree().process_frame
	if screenshot_at >= 0:
		playhead = float(screenshot_at)
	_set_camera_view(screenshot_view)
	pad.begin_stroke(Vector2(0.28, 0.25))
	pad.continue_stroke(Vector2(0.72, 0.24))
	pad.continue_stroke(Vector2(0.52, 0.55))
	pad.continue_stroke(Vector2(0.44, 0.8))
	pad.end_stroke()
	for i in 90:
		await get_tree().process_frame
	if screenshot_pulse >= 0.0:
		# Freeze the pulse part-way through.
		view.pulse_speed = 0.0
		view.pulse_t = screenshot_pulse
		view.edge_mat.set_shader_parameter("pulse_t", screenshot_pulse)
		view._recolor()
		for i in 5:
			await get_tree().process_frame
	await RenderingServer.frame_post_draw
	get_viewport().get_texture().get_image().save_png(screenshot_path)
	print("Screenshot saved: ", screenshot_path, " at snapshot ", shown_index, " answer ", view._answer.text)
	get_tree().quit()
