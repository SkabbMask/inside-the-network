# Inside the Network

A VR exhibit prototype: stand inside a real neural network (784 → 16 → 16 → 10,
about 13,000 weights), watch it learn, and draw digits for it to read.

- `server/` – Python (NumPy). Records a full training run as snapshots and serves any
  moment of it: weights, how your drawing flows through, and the neuron layout.
- `godot/` – Godot 4.4+ project. Runs on a monitor; VR via OpenXR (see below).

## The space

Depth is the order of computation: your drawing appears as a 28 × 28 mosaic on the
floor, then hidden layer 1, hidden layer 2 and the ten answer neurons. Within each
layer, neurons are placed on a similarity map: neurons that react to the same drawings
sit close together (distance = 1 − correlation of their reactions on 1,000 test digits).
The map is recomputed for every snapshot, so neurons move as training changes them.

Connections: thickness = size of the weight, colour = sign (orange pushes the next
neuron on, blue pushes it off). Brightness is the pulse: when you finish a stroke, your
drawing travels through layer by layer, and each connection glows in proportion to the
signal it carries. While the timeline plays, the pulse repeats like a heartbeat.

## Run it

1. `cd server`, create and activate a venv, `pip install -r requirements.txt`,
   `python server.py`. First run downloads MNIST (about 11 MB).
2. Open `godot/` in Godot and press Play.

## Controls (desktop)

| | |
|---|---|
| Draw on the board, press buttons, select a neuron, drag the timeline | Left mouse |
| Switch a hidden neuron off / on | Right mouse on it |
| Play / pause | Space |
| Camera: at the console / side-on to the network | 1 / 2 |
| Look / move | Middle-drag / WASD, Q, E |

Timeline: Start, Reverse, Play, Slower, Faster, End, or drag along the curve.
The curve is the share of test digits answered correctly. Time runs by snapshot, so the
first 200 training steps (where most of the change happens) play in more detail.

## VR

OpenXR is switched off in Project Settings for now, because enabling it crashed
Godot at start-up on our first test machine. To try VR: Project Settings → XR → OpenXR
→ Enabled. VR controls: trigger = left mouse, grip = right mouse.

Data: MNIST (LeCun, Cortes & Burges). Network size follows 3Blue1Brown's
"But what is a neural network?".
