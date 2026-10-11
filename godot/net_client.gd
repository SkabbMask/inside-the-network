## WebSocket link to the Python model server. Reconnects on its own.
extends Node

signal binary_received(kind: int, payload: PackedByteArray)
signal json_received(msg: Dictionary)
signal connection_changed(ok: bool)

var url := "ws://127.0.0.1:8765"
var ws := WebSocketPeer.new()
var is_open := false
var _retry_at := 0.0


func connect_to(u: String) -> void:
	url = u
	_open()


func send(msg: Dictionary) -> void:
	if is_open:
		ws.send_text(JSON.stringify(msg))


func _now() -> float:
	return Time.get_ticks_msec() / 1000.0


func _open() -> void:
	ws = WebSocketPeer.new()
	ws.inbound_buffer_size = 1 << 22   # edge frames are ~100 KB
	ws.max_queued_packets = 4096
	_retry_at = _now() + 2.0
	ws.connect_to_url(url)


func _process(_delta: float) -> void:
	ws.poll()
	var state := ws.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN:
		if not is_open:
			is_open = true
			connection_changed.emit(true)
			send({"type": "hello"})
		while ws.get_available_packet_count() > 0:
			var pkt := ws.get_packet()
			if ws.was_string_packet():
				var m = JSON.parse_string(pkt.get_string_from_utf8())
				if m is Dictionary:
					json_received.emit(m)
			elif pkt.size() >= 4:
				binary_received.emit(pkt.decode_u32(0), pkt.slice(4))
	elif state == WebSocketPeer.STATE_CLOSED:
		if is_open:
			is_open = false
			connection_changed.emit(false)
			_retry_at = _now() + 1.0
		if _now() >= _retry_at:
			_open()
