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
# Where visitors start, in the network's own coordinates: out to the side, level with the
# space between the hidden layers, looking across the network. The console waits in front.
const START_POS := Vector3(768.8, 368.8, -482.2)
const START_LOOK := Vector3(670.0, 372.8, -467.6)
const EYE_HEIGHT := 1.6         # metres from the VR play-space floor to the eyes, roughly
const VR_RENDER_SCALE := 0.75   # share of SteamVR's recommended resolution, per axis
const FLY_SPEED := 40.0          # metres per second; hold Shift (or click the stick) for 4x
# VR movement, tuned for comfort: no head-steered motion, snap turns, a vignette.
const SNAP_ANGLE := 30.0          # degrees per snap turn
const VR_PULL_GAIN := 25.0        # hand movement along a held cable is amplified this much
const VR_EDIT_INTERVAL := 0.12    # seconds between weight notches while a stick is held
const STICK_DEADZONE := 0.15
const HAND_SCALE := 1.5           # the hand models are drawn this much larger than life
const VR_GRAB_RANGE := 6.0        # metres: the grip catches a cable this close to the hand
# The grip only switches off a neuron no further away than the gap between two neighbouring
# neurons in a grid, measured to the ball's surface.
const VR_TOGGLE_RANGE := NetworkView.GRID_SIZE / 3.0

# The console's housing: Kenney's "computer-wide" (Space Station Kit, CC0). Its screen face
# (measured from the mesh) is centred at SCREEN_CENTER, faces SCREEN_NORMAL and is 0.8 m wide;
# it is scaled up and turned so the screen sits just behind the console's buttons and pad.
const CONSOLE_MODEL := "res://assets/kenney/computer-wide.glb"
const SCREEN_CENTER := Vector3(0.0, 0.365, 0.006)
const SCREEN_NORMAL := Vector3(0.0, 0.83, 0.55)
const CONSOLE_SCALE := 2.15
const PANEL_CENTER := Vector2(-0.84, 1.17)   # middle of the buttons, pad and timeline
# Where your head ends up relative to the console when you summon it (console space).
const SUMMON_HEAD := Vector3(-0.84, 1.5, 0.6)
# At the start it waits lower and further off, like a desk below your line of sight.
const START_HEAD := Vector3(-0.84, 1.95, 1.0)

# The console is a green-on-black terminal.
const COL_BTN := Color(0.01, 0.05, 0.02)
const COL_BTN_HOVER := Color(0.05, 0.32, 0.1)
const COL_TEXT := Color(0.25, 1.0, 0.45)
const COL_TEXT_DIM := Color(0.12, 0.6, 0.25)
const COL_GREEN := Color(0.2, 1.0, 0.4)   # frames, curve, the housing's rim
# VT323 (SIL Open Font License), bundled so it looks the same on every machine. Its
# letters are small for their size, so console text is drawn this much larger.
const TERMINAL_FONT := "res://assets/fonts/VT323-Regular.ttf"
const TERMINAL_FONT_SCALE := 1.2

var server_url := "ws://127.0.0.1:8765"
var screenshot_path := ""
var screenshot_at := -1
var screenshot_pulse := -1.0
var screenshot_view := 1
var screenshot_center := false
var screenshot_cam = null    # Vector3 in network coordinates, or null for a preset view
var screenshot_look = null
var screenshot_select := Vector2i(-1, -1)
var screenshot_grab := false
var screenshot_hands := false
var screenshot_summon := false
var screenshot_off := Vector2i(-1, -1)

var client
var view
var pad
var vr := false
var flow_mode := true
var center := false   # help the model by resizing and centring the drawing
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
var _edited := 0   # weights the visitor changed at the snapshot shown
var _console: Node3D       # parent of the board, buttons and timeline
var _station: Transform3D  # where the visitor stands at the console, facing the network
var _spark: MeshInstance3D
var _floor: MeshInstance3D
var _env: Environment
var _perf_t := 0.0
# VR
var _origin: XROrigin3D
var _xr_cam: XRCamera3D
var _hands := []            # [left, right] XRController3D
var _hand_models := []      # [left, right] XRToolsHand
var _palms := []            # [left, right] where each palm is (the hands' aim-offset nodes)
var _hand_rest: Node3D      # where the holding hand is drawn: on the cable's surface
var _vel := Vector3.ZERO    # current flying velocity, metres per second
var _snap_ready := true
var _snap_flash := 0.0      # vignette flash after a snap turn, fades out
var _edit_t := 0.0
var _vignette: ShaderMaterial
var _start_head := Vector3.ZERO   # where your eyes start
var _start_basis := Basis()        # and which way you face (level)
var _console_to := Transform3D()   # where the console is flying to after a summon
var _console_moving := false
var _summon_was_down := false
var _perf_flags := PackedStringArray()   # test switches: nomsaa, noglow, nofog, nossr
var _playhead_mark: MeshInstance3D

# desktop camera
var _cam: Camera3D
var _yaw := 0.0
var _pitch := 0.0
var _mouse_left := false
var _mouse_right := false
var _grab := {}   # the line you are holding: {edge, a, b, s, offset}


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
		elif a.begins_with("--cam="):
			screenshot_cam = _vec3_arg(a.trim_prefix("--cam="))
		elif a.begins_with("--look="):
			screenshot_look = _vec3_arg(a.trim_prefix("--look="))
		elif a.begins_with("--off="):
			var q := a.trim_prefix("--off=").split(",")
			screenshot_off = Vector2i(int(q[0]), int(q[1]))
		elif a == "--summon":
			screenshot_summon = true
		elif a == "--hands":
			screenshot_hands = true
		elif a == "--grab":
			screenshot_grab = true
		elif a.begins_with("--select="):
			var p := a.trim_prefix("--select=").split(",")
			screenshot_select = Vector2i(int(p[0]), int(p[1]))
		elif a.begins_with("--perf="):
			_perf_flags = a.trim_prefix("--perf=").split(",")
		elif a.begins_with("--center="):
			screenshot_center = a.trim_prefix("--center=") == "1"

	_build_world()
	view = NetworkView.new()
	add_child(view)
	view.build()

	# The console stands side-on to the network, so you see every layer apart and the
	# drawing (from behind) at the start of the stack.
	var head: Vector3 = view.to_global(START_POS)
	var ahead: Vector3 = view.to_global(START_LOOK) - head
	ahead.y = 0.0
	_start_basis = Basis.looking_at(ahead.normalized(), Vector3.UP)
	_start_head = head
	# The console's own frame: placed so the start position is where your head is at it.
	_station = Transform3D(_start_basis, head - _start_basis * START_HEAD)
	_console = Node3D.new()
	add_child(_console)
	_console.transform = _station
	view.place_panel(_station)
	_build_console_housing()
	_floor.position = head - Vector3(0, EYE_HEIGHT + 0.002, 0)

	pad = DrawPad.new()
	_console.add_child(pad)
	pad.size = 0.36
	pad.position = Vector3(-1.0, 1.32, PANEL_Z)
	pad.build()
	view.add_drawing_sheet(pad.texture())
	# Without help, the network gets exactly the drawing: show it on the slice at once.
	pad.changed.connect(func(): if not center: view.show_input(pad.canvas))

	_build_buttons()
	_build_timeline()
	_status = _make_label("", Vector3(-1.0, 1.535, PANEL_Z), 22)

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
	l.font_size = int(size * TERMINAL_FONT_SCALE)
	l.pixel_size = 0.0015
	l.outline_size = 0
	l.modulate = color
	l.position = pos
	l.font = _mono()
	_console.add_child(l)
	return l


## The console's retro terminal font.
func _mono() -> Font:
	return load(TERMINAL_FONT)


## Put the console's housing behind its buttons, pad and timeline, restyled to match the
## console: dark glass with a terminal-green rim.
func _build_console_housing() -> void:
	var model: Node3D = load(CONSOLE_MODEL).instantiate()
	var mat := ShaderMaterial.new()
	mat.shader = Shader.new()
	# Only the screen slab is kept: the stand and body (anything in front of the screen face
	# or more than the slab's thickness behind it) are cut away, so it floats like a panel.
	mat.shader.code = """
shader_type spatial;
render_mode cull_disabled;
uniform vec3 screen_center;
uniform vec3 screen_normal;
uniform vec3 screen_up;
varying float v_depth;
varying float v_up;
void vertex() {
	v_depth = dot(VERTEX - screen_center, screen_normal);
	v_up = dot(VERTEX - screen_center, screen_up);
}
void fragment() {
	if (v_depth < -0.105 || v_depth > 0.0 || v_up < -0.235) {
		discard;
	}
	float rim = pow(1.0 - clamp(abs(dot(NORMAL, VIEW)), 0.0, 1.0), 3.0);
	ALBEDO = vec3(0.012, 0.008, 0.02);
	ROUGHNESS = 0.15;
	SPECULAR = 0.7;
	EMISSION = vec3(0.2, 1.0, 0.4) * rim * 0.3;
}
"""
	mat.set_shader_parameter("screen_center", SCREEN_CENTER)
	mat.set_shader_parameter("screen_normal", SCREEN_NORMAL.normalized())
	# Up along the screen face; the screen's bottom edge is 0.23 below its centre.
	mat.set_shader_parameter("screen_up", Vector3(0, SCREEN_NORMAL.z, -SCREEN_NORMAL.y).normalized())
	for mi in model.find_children("*", "MeshInstance3D", true, false):
		(mi as MeshInstance3D).material_override = mat
	# Turn the screen to face the visitor (+z in console space), scale it, and move its
	# centre to just behind the middle of the panel.
	var turn := Basis(Quaternion(SCREEN_NORMAL.normalized(), Vector3.BACK))
	var basis := turn.scaled(Vector3.ONE * CONSOLE_SCALE)
	var centre := Vector3(PANEL_CENTER.x, PANEL_CENTER.y, PANEL_Z - 0.02)
	model.transform = Transform3D(basis, centre - basis * SCREEN_CENTER)
	_console.add_child(model)


## Where you are and where you look, in the network's own coordinates (for choosing the
## starting position).
func _print_camera() -> void:
	var cam: Node3D = _xr_cam if vr else _cam
	var p: Vector3 = view.to_local(cam.global_position)
	var look: Vector3 = view.to_local(cam.global_position - cam.global_basis.z * 100.0)
	print("Camera: position %s, looking at %s (network coordinates)" % [
		p.snapped(Vector3.ONE * 0.1), look.snapped(Vector3.ONE * 0.1)])


## Bring the console to just in front of you, facing you (left controller B, desktop C).
func _summon_console() -> void:
	var head := _head_pos()
	var look := -(_xr_cam.global_basis.z if vr else _cam.global_basis.z)
	look.y = 0.0
	if look.length() < 1e-3:
		look = Vector3.FORWARD
	var basis := Basis.looking_at(look.normalized(), Vector3.UP)
	_console_to = Transform3D(basis, head - basis * SUMMON_HEAD)
	_console_moving = true


## The console glides to where it was summoned.
func _move_console(delta: float) -> void:
	if not _console_moving:
		return
	var t := 1.0 - exp(-delta * 8.0)
	var now := _console.global_transform
	now.origin = now.origin.lerp(_console_to.origin, t)
	now.basis = Basis(now.basis.get_rotation_quaternion().slerp(_console_to.basis.get_rotation_quaternion(), t))
	_console.global_transform = now
	view.place_panel(now)
	if now.origin.distance_to(_console_to.origin) < 0.01:
		_console.global_transform = _console_to
		view.place_panel(_console_to)
		_console_moving = false


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
	_console.add_child(mi)
	return mi


func _build_world() -> void:
	var env := Environment.new()
	_env = env
	# Deep space with stars (sky.gdshader).
	var sky_mat := ShaderMaterial.new()
	sky_mat.shader = load("res://sky.gdshader")
	var sky := Sky.new()
	sky.sky_material = sky_mat
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.14, 0.1, 0.22)
	# Fog fades far lines towards the dark, so near ones stand out. The stars stay clear.
	env.fog_enabled = true
	env.fog_light_color = Color(0.025, 0.012, 0.045)
	env.fog_density = 0.0022
	env.fog_sky_affect = 0.0
	# A light bloom: only the brightest pixels (the tubes' own glow, the pulse) bleed a
	# little. It works on the finished image, so overlapping tubes never add up.
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR   # keeps the neon colours saturated
	# Screen-space reflections, so the glass balls mirror the neon tubes around them.
	env.ssr_enabled = true
	env.ssr_max_steps = 48
	env.glow_enabled = true
	env.glow_intensity = 0.5
	env.glow_strength = 0.9
	env.glow_bloom = 0.0
	env.glow_hdr_threshold = 0.9
	env.glow_blend_mode = Environment.GLOW_BLEND_MODE_SCREEN
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	# One light from above and to the side, to shade the round lines.
	var sun := DirectionalLight3D.new()
	sun.rotation = Vector3(deg_to_rad(-50), deg_to_rad(35), 0)
	sun.light_energy = 0.7
	sun.light_color = Color(0.85, 0.8, 1.0)
	add_child(sun)
	_floor = MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(4000, 4000)
	_floor.mesh = plane
	_floor.material_override = _flat(Color(0.055, 0.06, 0.075))
	_floor.position.y = -0.002
	_floor.visible = false   # floating in space; kept for the console's sake if wanted
	add_child(_floor)


func _add_button(id: String, text: String, center: Vector2, size: Vector2) -> void:
	# A thin green frame behind a dark face.
	_box(Vector3(size.x + 0.006, size.y + 0.006, 0.018), Vector3(center.x, center.y, PANEL_Z - 0.012), COL_GREEN)
	var mi := _box(Vector3(size.x, size.y, 0.02), Vector3(center.x, center.y, PANEL_Z - 0.01), COL_BTN)
	var l := _make_label(text, Vector3(center.x, center.y, PANEL_Z + 0.002), 22)
	_buttons.append({"id": id, "rect": Rect2(center - size / 2, size), "label": l, "mat": mi.material_override})


func _build_buttons() -> void:
	var defs := [
		["clear", "Clear board"],
		["view", "Show all weights"],
		["train", "Train from scratch"],
		["reset", "Standard run"],
		["center", "Help: centre it"],
		["undo", "Undo my changes"],
	]
	# Two columns of three, so they end above the timeline.
	for i in defs.size():
		var x := -0.625 if i < 3 else -0.325
		var y := 1.47 - (i % 3) * 0.09
		_add_button(defs[i][0], defs[i][1], Vector2(x, y), Vector2(0.29, 0.07))


func _build_timeline() -> void:
	var w := BAR_X1 - BAR_X0
	_box(Vector3(w, 0.012, 0.01), Vector3((BAR_X0 + BAR_X1) / 2, BAR_Y, PANEL_Z - 0.01), COL_TEXT_DIM)
	_playhead_mark = _box(Vector3(0.006, SPARK_H + 0.04, 0.012), Vector3(BAR_X0, BAR_Y + SPARK_H / 2, PANEL_Z), Color(0.8, 1.0, 0.85))
	_spark = MeshInstance3D.new()
	_spark.material_override = _flat(COL_GREEN)
	_console.add_child(_spark)
	_time_label = _make_label("", Vector3((BAR_X0 + BAR_X1) / 2, BAR_Y + SPARK_H + 0.04, PANEL_Z), 24)
	_make_label("Correct answers on test digits, through training", Vector3((BAR_X0 + BAR_X1) / 2, BAR_Y - 0.026, PANEL_Z), 16, COL_TEXT_DIM)
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
		"undo":
			client.send({"type": "clear_edits"})
			_pulse_on_meta = true
		"center":
			client.send({"type": "center", "on": not center})
			_pulse_on_meta = true
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
	# VR renders two large eyes at the headset's refresh rate, so trim the costliest parts:
	# a lower render resolution, and no screen-space reflections (expensive, and they
	# differ between the eyes).
	get_viewport().scaling_3d_mode = Viewport.SCALING_3D_MODE_BILINEAR
	get_viewport().scaling_3d_scale = VR_RENDER_SCALE
	_env.ssr_enabled = false
	# No MSAA in VR: with thousands of thin tube edges it is the costliest single effect.
	# (FXAA instead turned the headset view black with D3D12, so no smoothing for now.)
	get_viewport().msaa_3d = Viewport.MSAA_DISABLED
	RenderingServer.viewport_set_measure_render_time(get_viewport().get_viewport_rid(), true)
	var xr := XRServer.find_interface("OpenXR")
	print("VR: render target %s per eye (x%.2f), display %.0f Hz" % [
		xr.get_render_target_size(), VR_RENDER_SCALE, xr.get_display_refresh_rate()])
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	var origin := XROrigin3D.new()
	add_child(origin)
	origin.transform = Transform3D(_start_basis, _start_head - Vector3(0, EYE_HEIGHT, 0))
	_origin = origin
	_xr_cam = XRCamera3D.new()
	origin.add_child(_xr_cam)
	_build_vignette()
	for hand in ["left_hand", "right_hand"]:
		var ctrl := XRController3D.new()
		ctrl.tracker = hand
		ctrl.pose = "aim"
		origin.add_child(ctrl)
		_hands.append(ctrl)
		# The hand models sit in the palm; this node moves them from the aim pose (which the
		# lasers use) to where your real hand is.
		var offset := XRToolsHandAimOffset.new()
		ctrl.add_child(offset)
		var model := _make_hand(hand == "left_hand")
		offset.add_child(model)
		_hand_models.append(model)
		_palms.append(offset)
		var laser := MeshInstance3D.new()
		var lb := BoxMesh.new()
		lb.size = Vector3(0.003, 0.003, 1.0)
		laser.mesh = lb
		var lm := _flat(Color(1.0, 0.85, 0.5, 0.7))
		lm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		laser.material_override = lm
		ctrl.add_child(laser)
		_pointers.append({"ctrl": ctrl, "laser": laser, "trigger": false, "grip": false, "scrubbing": false})
	_hand_rest = Node3D.new()
	_hand_rest.top_level = true
	add_child(_hand_rest)


## A hand from Godot XR Tools (MIT; hand models CC0): its fingers follow trigger and grip,
## and it lines itself up with the controller's aim pose. Dark glass with a white rim.
func _make_hand(left: bool) -> Node3D:
	var mat := ShaderMaterial.new()
	mat.shader = Shader.new()
	mat.shader.code = """
shader_type spatial;
void fragment() {
	float rim = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), 2.5);
	ALBEDO = vec3(0.02, 0.015, 0.03);
	ROUGHNESS = 0.2;
	SPECULAR = 0.6;
	EMISSION = vec3(1.0) * rim * 1.5;   // white
}
"""
	var model: XRToolsHand = load("res://addons/godot-xr-tools/hands/scenes/lowpoly/%s_hand_low.tscn"
		% ("left" if left else "right")).instantiate()
	model.hand_material_override = mat
	# Set before the hand enters the tree: it keeps this transform relative to its target.
	model.scale = Vector3.ONE * HAND_SCALE
	return model


## Comfort vignette: dark edges that close in while you move or turn. A quad just in
## front of the eyes, drawn over everything.
func _build_vignette() -> void:
	var sh := Shader.new()
	sh.code = """
shader_type spatial;
render_mode unshaded, cull_disabled, depth_test_disabled, depth_draw_never, blend_mix, fog_disabled, shadows_disabled;
uniform float strength = 0.0;   // 0 = clear view, 1 = only a small window left
void fragment() {
	// The quad spans about 144 degrees; a headset shows roughly the middle 100 (r < 0.4).
	float r = length(UV - 0.5) * 2.0;
	float inner = mix(0.5, 0.12, strength);
	ALBEDO = vec3(0.0);
	ALPHA = smoothstep(inner, inner + 0.15, r) * step(0.001, strength);
}
"""
	_vignette = ShaderMaterial.new()
	_vignette.shader = sh
	_vignette.render_priority = 127
	var q := MeshInstance3D.new()
	var qm := QuadMesh.new()
	qm.size = Vector2(0.5, 0.5)
	q.mesh = qm
	q.material_override = _vignette
	q.position = Vector3(0, 0, -0.08)
	q.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_xr_cam.add_child(q)


## Where the visitor's eyes are, in VR or on desktop.
func _head_pos() -> Vector3:
	return _xr_cam.global_position if vr else _cam.global_position


func _stick(hand: int) -> Vector2:
	var v: Vector2 = _hands[hand].get_vector2("primary")
	return v if v.length() > STICK_DEADZONE else Vector2.ZERO


## VR movement: fly where the left controller points, rise and sink and snap-turn with the
## right stick, or (holding a cable) pull along it by hand and edit it with either stick.
func _vr_move(delta: float) -> void:
	var left := _stick(0)
	var right := _stick(1)
	# Snap turn on the right stick, around your head; it must come back to centre first.
	if absf(right.x) > 0.7 and _snap_ready:
		_snap_ready = false
		_snap_flash = 1.0
		var pivot := _xr_cam.global_position
		var turn := Transform3D(Basis(Vector3.UP, deg_to_rad(-SNAP_ANGLE * signf(right.x))), Vector3.ZERO)
		_origin.global_transform = Transform3D(Basis(), pivot) * turn * Transform3D(Basis(), -pivot) * _origin.global_transform
	elif absf(right.x) < 0.3:
		_snap_ready = true
	_snap_flash = maxf(0.0, _snap_flash - delta * 4.0)

	if not _grab.is_empty():
		_vel = Vector3.ZERO
		_vr_hold_cable(delta, left, right)
	else:
		var aim: XRController3D = _hands[0]
		var boost := 4.0 if aim.is_button_pressed("primary_click") else 1.0
		var rise := right.y if absf(right.y) > absf(right.x) else 0.0
		var want := (-aim.global_basis.z * left.y + aim.global_basis.x * left.x) * FLY_SPEED * boost
		want += Vector3.UP * rise * FLY_SPEED * 0.6
		_vel = _vel.lerp(want, 1.0 - exp(-delta * 8.0))
		_origin.global_position += _vel * delta
	var moving := clampf(_vel.length() / FLY_SPEED, 0.0, 1.0) * 0.55
	if not _grab.is_empty():
		moving = clampf(_grab.get("speed", 0.0) / FLY_SPEED, 0.0, 1.0) * 0.55
	_vignette.set_shader_parameter("strength", maxf(moving, _snap_flash * 0.7))
	_place_hands()


## XR Tools moves its hands in the physics step (60 Hz, before our movement), so while you
## fly they trail behind. Put them in place again every frame, after moving.
func _place_hands() -> void:
	for i in _hand_models.size():
		var model: XRToolsHand = _hand_models[i]
		var held: bool = not _grab.is_empty() and _grab.get("model") == model
		var target: Transform3D = _hand_rest.global_transform if held else (_palms[i] as Node3D).global_transform
		model.global_transform = target * model._transform


## Holding a cable in VR, like a rope: the hand stays where it took hold, so moving the
## hand moves you instead. Along the cable the pull is amplified, to cross the distances.
## The hand is drawn resting on the cable's surface.
func _vr_hold_cable(delta: float, left: Vector2, right: Vector2) -> void:
	var a: Vector3 = _grab.a
	var b: Vector3 = _grab.b
	var length := a.distance_to(b)
	var tangent := (b - a).normalized()
	var hand: XRController3D = _grab.hand
	var hand_pos := hand.position
	# Pull: moving the hand back along the cable moves you forward, and vice versa. Holding
	# the hand still on the cable already moves you 1:1; the extra here adds the gain.
	var moved: Vector3 = _origin.global_basis * (hand_pos - (_grab.hand_last as Vector3))
	var margin := (NetworkView.BALL + 2.0) / length
	var before: float = _grab.s
	_grab.s = clampf(_grab.s - moved.dot(tangent) * (VR_PULL_GAIN - 1.0) / length,
		0.0 if _grab.edge < 12544 else margin, 1.0 - margin)
	_grab.speed = absf(_grab.s - before) * length / maxf(delta, 1e-4)
	_grab.hand_last = hand_pos
	# Where the palm rests: on the surface, on the side it took hold, turned as it was.
	var c: Vector3 = view.to_global(view.edge_point(_grab.edge, _grab.s))
	var r: float = view.edge_width(_grab.edge) / 2.0
	var rest := Transform3D(_grab.palm_basis, c + (_grab.radial as Vector3) * (r + 0.03 * HAND_SCALE))
	_hand_rest.global_transform = rest
	# Move yourself so the controller is where that resting hand needs it.
	var want: Transform3D = rest * (_grab.ctrl_from_palm as Transform3D)
	_origin.global_position += want.origin - hand.global_position
	# Either stick up or down edits the held cable's weight, a notch at a time.
	var y := left.y if absf(left.y) > absf(right.y) else right.y
	if absf(y) > 0.5:
		_edit_t -= delta
		if _edit_t <= 0.0:
			_nudge_grabbed(1 if y > 0.0 else -1)
			_edit_t = VR_EDIT_INTERVAL
	else:
		_edit_t = 0.0


func _setup_desktop() -> void:
	_cam = Camera3D.new()
	_cam.fov = 75
	_cam.far = 6000.0
	add_child(_cam)
	_set_camera_view(1)
	_cam.make_current()
	_pointers.append({"ctrl": null, "laser": null, "trigger": false, "grip": false, "scrubbing": false})


## Desktop camera presets: 1 = at the console (where a VR visitor starts),
## 2 = further back and higher, to see the console and the whole network,
## 3 = in front of the input slice, to draw on it.
func _set_camera_view(n: int) -> void:
	var mid: Vector3 = view.to_global(Vector3(0, NetworkView.GRID_Y, NetworkView.MID_Z))
	if n == 1:
		_cam.position = _start_head
		mid = view.to_global(START_LOOK)
	elif n == 2:
		_cam.position = _station * Vector3(0.0, NetworkView.GRID_SIZE * 0.4, NetworkView.GRID_SIZE * 0.6)
	else:
		_cam.position = view.to_global(Vector3(0.0, NetworkView.GRID_Y, NetworkView.GRID_SIZE * 1.1))
		mid = view.to_global(Vector3(0, NetworkView.GRID_Y, 0))
	_cam.look_at(mid)
	_yaw = _cam.rotation.y
	_pitch = _cam.rotation.x


func _unhandled_input(event: InputEvent) -> void:
	# P works in VR too (with the game window focused): print where you are.
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_P:
		_print_camera()
		return
	if vr:
		return
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			_mouse_left = event.pressed
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			_mouse_right = event.pressed
		# While holding a line, the wheel makes its weight stronger or weaker.
		elif event.pressed and not _grab.is_empty() and event.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]:
			_nudge_grabbed(1 if event.button_index == MOUSE_BUTTON_WHEEL_UP else -1)
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
			KEY_3:
				_set_camera_view(3)
			KEY_K:
				_toggle_all_hidden()
			KEY_C:
				_summon_console()


func _process(delta: float) -> void:
	if not vr and _cam:
		# Arrows turn (left/right) and tilt (up/down) the camera in place.
		var turn := 0.0
		var tilt := 0.0
		if Input.is_key_pressed(KEY_LEFT): turn += 1
		if Input.is_key_pressed(KEY_RIGHT): turn -= 1
		if Input.is_key_pressed(KEY_UP): tilt += 1
		if Input.is_key_pressed(KEY_DOWN): tilt -= 1
		if turn != 0.0 or tilt != 0.0:
			_yaw += turn * delta * 1.5
			_pitch = clampf(_pitch + tilt * delta * 1.5, -1.3, 1.3)
			_cam.rotation = Vector3(_pitch, _yaw, 0)

		if not _grab.is_empty():
			_move_along_grab(delta)
			_after_camera(delta)
			return

		# WASD walks on the horizontal plane in the facing direction; U/J move along world Y.
		var move := Vector3.ZERO
		if Input.is_key_pressed(KEY_W): move.z -= 1
		if Input.is_key_pressed(KEY_S): move.z += 1
		if Input.is_key_pressed(KEY_A): move.x -= 1
		if Input.is_key_pressed(KEY_D): move.x += 1
		if Input.is_key_pressed(KEY_U): move.y += 1
		if Input.is_key_pressed(KEY_J): move.y -= 1
		if move != Vector3.ZERO:
			var speed := FLY_SPEED * (4.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0)
			_cam.position += (Basis(Vector3.UP, _yaw) * move.normalized()) * delta * speed
	_after_camera(delta)


func _after_camera(delta: float) -> void:
	if vr and not _hands.is_empty():
		var down: bool = (_hands[0] as XRController3D).is_button_pressed("by_button")
		if down and not _summon_was_down:
			_summon_console()
		_summon_was_down = down
	_move_console(delta)
	if vr:
		_vr_move(delta)
		_report_performance(delta)
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
	# Clicking a line (nearer than anything else under the pointer) grabs it; letting go
	# releases it. On the drawing surface or a button, the click is always for that: the
	# pixel lines start right on the screen, so they would otherwise win.
	if trig_down and not p.ctrl and kind != "button" and kind != "pad":
		var g: Dictionary = view.pick_edge(origin, dir)
		if not g.is_empty() and (kind == "" or g.t < h.t):
			_start_grab(g.edge, g.s)
			kind = "grab"
	if not trig and not _grab.is_empty() and not p.ctrl:
		_end_grab()
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
	# In VR the grip first grabs a cable right under the hand; otherwise it switches the
	# neuron the laser points at.
	var grabbed := false
	if grip and not p.grip and p.ctrl and _grab.is_empty():
		var g: Dictionary = view.nearest_edge(_palm(p.ctrl).origin, VR_GRAB_RANGE)
		if not g.is_empty():
			_start_grab(g.edge, g.s, p.ctrl)
			grabbed = true
	if grip and not p.grip and not grabbed and kind == "node" and h.layer < 3 and (not p.ctrl or h.t <= VR_TOGGLE_RANGE):
		client.send({"type": "toggle_neuron", "layer": h.layer, "neuron": h.neuron})
		_pulse_on_meta = true
	if not grip and p.grip and p.ctrl and not _grab.is_empty() and _grab.get("hand") == p.ctrl:
		_end_grab()
	p.trigger = trig
	p.grip = grip

	for b in _buttons:
		var hover: bool = kind == "button" and h.id == b.id
		b.mat.albedo_color = COL_BTN_HOVER if hover else COL_BTN


func _hit(world_origin: Vector3, world_dir: Vector3) -> Dictionary:
	var best := {}
	# Console hits are worked out in the console's own space, where its panel is z = PANEL_Z.
	var origin := _console.to_local(world_origin)
	var dir := _console.global_basis.inverse() * world_dir
	if absf(dir.z) > 1e-5:
		var t := (PANEL_Z - origin.z) / dir.z
		if t > 0.0:
			var pt := origin + dir * t
			var uv: Vector2 = pad.uv_at(_console.to_global(pt))
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
	# Drawing straight onto the input slice.
	var s: Dictionary = view.input_hit(world_origin, world_dir)
	if not s.is_empty() and (not best.has("kind") or s.t < best.t):
		var f0 = best.get("frac")
		best = {"t": s.t, "kind": "pad", "uv": s.uv}
		if f0 != null:
			best["frac"] = f0
	var n: Dictionary = view.pick(world_origin, world_dir)
	if not n.is_empty() and (not best.has("kind") or n.t < best.t):
		var f = best.get("frac")
		best = {"t": n.t, "kind": "node", "layer": n.layer, "neuron": n.neuron}
		if f != null:
			best["frac"] = f
	return best


## In VR, print frame rate and frame time every two seconds, to the Output panel.
func _report_performance(delta: float) -> void:
	_perf_t += delta
	if _perf_t < 2.0:
		return
	_perf_t = 0.0
	var xr := XRServer.find_interface("OpenXR")
	var rid := get_viewport().get_viewport_rid()
	print("VR: %d fps (headset %.0f Hz, budget %.1f ms) | GPU %.1f ms | CPU render %.1f ms | script %.1f ms" % [
		Engine.get_frames_per_second(), xr.get_display_refresh_rate(), 1000.0 / maxf(xr.get_display_refresh_rate(), 1.0),
		RenderingServer.viewport_get_measured_render_time_gpu(rid),
		RenderingServer.viewport_get_measured_render_time_cpu(rid),
		Performance.get_monitor(Performance.TIME_PROCESS) * 1000.0])


# ------------------------------------------------------------------ holding a line

const GRAB_REACH := 1.5   # how far from the line's surface you hang, metres


func _start_grab(e: int, s: float, hand: XRController3D = null) -> void:
	var ends: Array = view.edge_ends(e)
	var a: Vector3 = view.to_global(ends[0])
	var b: Vector3 = view.to_global(ends[1])
	var p: Vector3 = view.to_global(view.edge_point(e, s))
	var away := _head_pos() - p
	# Hang beside the line, on the side you came from.
	away -= (b - a).normalized() * away.dot((b - a).normalized())
	if away.length() < 1e-3:
		away = Vector3.UP
	var offset: Vector3 = away.normalized() * (view.edge_width(e) / 2 + GRAB_REACH)
	_grab = {"edge": e, "a": a, "b": b, "s": s, "offset": offset}
	if hand:
		# Hold on where the hand is: remember where it sits around the cable and how it is
		# turned, relative to the cable, so both stay put while you hang on.
		var c: Vector3 = view.to_global(view.edge_point(e, s))
		var palm := _palm(hand)
		var radial: Vector3 = palm.origin - c
		radial -= (b - a).normalized() * radial.dot((b - a).normalized())
		if radial.length() < 1e-3:
			radial = Vector3.UP
		_grab.hand = hand
		_grab.hand_last = hand.position
		_grab.radial = radial.normalized()
		_grab.palm_basis = palm.basis
		_grab.ctrl_from_palm = palm.affine_inverse() * hand.global_transform
		var model: XRToolsHand = _hand_models[_hands.find(hand)]
		model.add_target_override(_hand_rest, 10)
		_grab.model = model
	view.set_grabbed(e)


## Where a VR hand's palm is (the hand model's own frame).
func _palm(ctrl: XRController3D) -> Transform3D:
	return (_palms[_hands.find(ctrl)] as Node3D).global_transform


## Debug (desktop key K): switch every hidden neuron off, or all back on if any are off.
func _toggle_all_hidden() -> void:
	var any_off: bool = not view.off[1].is_empty() or not view.off[2].is_empty()
	for layer in [1, 2]:
		for j in 16:
			if view.off[layer].has(j) == any_off:
				client.send({"type": "toggle_neuron", "layer": layer, "neuron": j})
	_pulse_on_meta = true


## Change the held line's weight by `steps` (each 5% of its layer's largest weight).
## Positive makes it more positive: a blue (negative) line shrinks, then turns pink.
func _nudge_grabbed(steps: int) -> void:
	if not client.is_open or recording:
		return
	client.send({"type": "nudge_weight", "edge": _grab.edge, "steps": steps})
	_pulse_on_meta = _drawn


func _end_grab() -> void:
	if _grab.has("model"):
		(_grab.model as XRToolsHand).remove_target_override(_hand_rest)
	_grab = {}
	view.set_grabbed(-1)


## While holding a line, W pulls you along it towards where you look, S the other way.
## You stop just short of the neuron at either end.
func _move_along_grab(delta: float) -> void:
	var a: Vector3 = _grab.a
	var b: Vector3 = _grab.b
	var length := a.distance_to(b)
	var pull := 0.0
	if Input.is_key_pressed(KEY_W): pull += 1
	if Input.is_key_pressed(KEY_S): pull -= 1
	var look := -_cam.global_basis.z
	var toward_b := 1.0 if (b - a).dot(look) >= 0.0 else -1.0
	var speed := FLY_SPEED * (4.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0)
	var margin := (NetworkView.BALL + 2.0) / length
	_grab.s = clampf(_grab.s + pull * toward_b * speed * delta / length, 0.0 if _grab.edge < 12544 else margin, 1.0 - margin)
	# Follow the line as it sways.
	var target: Vector3 = view.to_global(view.edge_point(_grab.edge, _grab.s)) + _grab.offset
	_cam.global_position = _cam.global_position.lerp(target, 1.0 - exp(-delta * 6.0))


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
			center = msg.get("center", false)
			_edited = int(msg.get("edited", 0))
			view.set_center(center)
			_button("center").label.text = "Stop helping" if center else "Help: centre it"
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
	if _edited > 0:
		_status.text = "You changed %d weight%s at this step" % [_edited, "" if _edited == 1 else "s"]
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

func _vec3_arg(text: String) -> Vector3:
	var p := text.split(",")
	return Vector3(float(p[0]), float(p[1]), float(p[2]))


func _run_screenshot_demo() -> void:
	var t0 := Time.get_ticks_msec()
	# Wait for the first snapshot too: its arrival sets the playhead, which would undo --at.
	while (steps.is_empty() or shown_index < 0) and Time.get_ticks_msec() - t0 < 10000:
		await get_tree().process_frame
	if screenshot_at >= 0:
		playhead = float(screenshot_at)
	_set_camera_view(screenshot_view)
	if "nomsaa" in _perf_flags:
		get_viewport().msaa_3d = Viewport.MSAA_DISABLED
		get_viewport().screen_space_aa = Viewport.SCREEN_SPACE_AA_FXAA
	if "noglow" in _perf_flags: _env.glow_enabled = false
	if "nofog" in _perf_flags: _env.fog_enabled = false
	if "nossr" in _perf_flags: _env.ssr_enabled = false
	if screenshot_cam != null:
		_cam.position = view.to_global(screenshot_cam)
		_cam.look_at(view.to_global(screenshot_look if screenshot_look != null else Vector3(0, NetworkView.GRID_Y, NetworkView.MID_Z)))
	if screenshot_center:
		client.send({"type": "center", "on": true})
	pad.begin_stroke(Vector2(0.28, 0.25))
	pad.continue_stroke(Vector2(0.72, 0.24))
	pad.continue_stroke(Vector2(0.52, 0.55))
	pad.continue_stroke(Vector2(0.44, 0.8))
	pad.end_stroke()
	if screenshot_summon:
		_summon_console()
	if screenshot_hands:
		for left in [true, false]:
			# The hand keeps the transform it has when added, relative to its parent.
			var h := _make_hand(left)
			h.position = Vector3(-0.12 if left else 0.12, -0.08, -0.4)
			h.rotation = Vector3(deg_to_rad(-60.0), 0, 0)
			_cam.add_child(h)
	if screenshot_off.x > 0:
		client.send({"type": "toggle_neuron", "layer": screenshot_off.x, "neuron": screenshot_off.y})
	if screenshot_grab:
		var vp := get_viewport().get_visible_rect().size / 2
		var g: Dictionary = view.pick_edge(_cam.project_ray_origin(vp), _cam.project_ray_normal(vp))
		print("Grab: ", g, " picked in ", Time.get_ticks_msec() - t0, " ms (since start)")
		if not g.is_empty():
			_start_grab(g.edge, g.s)
	if screenshot_select.x > 0:
		view.select(screenshot_select.x, screenshot_select.y)
	for i in 90:
		await get_tree().process_frame
	if screenshot_pulse >= 0.0:
		# Freeze the pulse part-way through.
		view.pulse_speed = 0.0
		view.pulse_t = screenshot_pulse
		view.set_edge_param("pulse_t", screenshot_pulse)
		view._recolor()
		for i in 5:
			await get_tree().process_frame
	# Report how long frames take here, with vsync off, before saving.
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	var f0 := Engine.get_frames_drawn()
	var t1 := Time.get_ticks_usec()
	for i in 120:
		await get_tree().process_frame
	print("Frame time: %.2f ms" % ((Time.get_ticks_usec() - t1) / 1000.0 / maxi(Engine.get_frames_drawn() - f0, 1)))
	await RenderingServer.frame_post_draw
	get_viewport().get_texture().get_image().save_png(screenshot_path)
	print("Screenshot saved: ", screenshot_path, " at snapshot ", shown_index, " answer ", view._answer.text, " off ", view.off)
	get_tree().quit()
