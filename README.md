# Inside the Network

A VR exhibit prototype: stand inside a real neural network (784 → 16 → 16 → 10,
about 13,000 weights), watch it learn, and draw digits for it to read.

- `server/` – Python (NumPy). Records a full training run as snapshots and serves any
  moment of it: weights, how your drawing flows through, and the neuron layout.
- `godot/` – Godot 4.4+ project. Runs on a monitor; VR via OpenXR (see below).

## The space

Depth is the order of computation, as a stack of upright slices about 4 m across: your
drawing as 28 × 28 pixel values (inputs, not neurons), then hidden layers 1 and 2 as
4 × 4 grids, then the ten answer neurons in a row from 0 to 9. Every neuron keeps its
place for the whole run, so you can follow one through training; what changes is the
connections. A hidden neuron's place in its grid has no meaning: each one is connected
to every pixel, and its weight image (select it) covers almost the whole digit.

(The server still computes a similarity-map layout per snapshot. An earlier version
placed neurons by it; the scene no longer uses it.)

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
| Draw (on the input image or the console board), press buttons, select a neuron, drag the timeline | Left mouse |
| Switch a hidden neuron off / on | Right mouse on it |
| Play / pause | Space |
| Camera: at the console / further back, overview / in front of the input, to draw | 1 / 2 / 3 |
| Look (turn / tilt) | Arrow keys, or middle-drag |
| Fly / up, down / faster | WASD / U, J / hold Shift |

Timeline: Start, Reverse, Play, Slower, Faster, End, or drag along the curve.
The curve is the share of test digits answered correctly. Time runs by snapshot, so the
first 200 training steps (where most of the change happens) play in more detail.

## VR

OpenXR is switched off in Project Settings for now, because enabling it crashed
Godot at start-up on our first test machine. To try VR: Project Settings → XR → OpenXR
→ Enabled. VR controls: trigger = left mouse, grip = right mouse.

Data: MNIST (LeCun, Cortes & Burges). Network size follows 3Blue1Brown's
"But what is a neural network?".
