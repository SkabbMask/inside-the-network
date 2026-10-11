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
| Grab a line, pull yourself along it, let go | Hold left mouse on it, W / S, release |
| Bring the console to you | C |
| Debug: switch every hidden neuron off / back on | K |

Timeline: Start, Reverse, Play, Slower, Faster, End, or drag along the curve.
The curve is the share of test digits answered correctly. Time runs by snapshot, so the
first 200 training steps (where most of the change happens) play in more detail.

## VR

OpenXR is on, and the project renders with Direct3D 12 on Windows
(`rendering_device/driver.windows`). With Vulkan, Godot 4.7 crashed while creating the
graphics device through OpenXR (SteamVR 2.17, AMD RX 6800); D3D12 works. If VR starts
in a window instead, check that SteamVR is the active OpenXR runtime.
VR controls:

| | |
|---|---|
| Fly where the left controller points / boost | Left stick / click it in |
| Snap turn 30° / rise, sink | Right stick left, right / up, down |
| Draw, press buttons, select a neuron, drag the timeline | Trigger (laser) |
| Switch a hidden neuron off / on | Grip, pointing at it (within one neuron spacing, ~120 m) |
| Hold on to a cable, pull along it hand over hand | Grip with the hand near the cable (within ~6 m); move the hand |
| Make the held cable's weight stronger / weaker | Either stick up / down |
| Bring the console to you | B on the left controller |

The view darkens at the edges while you move or snap-turn, to reduce motion sickness.

Console housing: "computer-wide" from Kenney's Space Station Kit (CC0, www.kenney.nl),
in `godot/assets/kenney`.

Console font: VT323 by Peter Hull (SIL Open Font License 1.1), in `godot/assets/fonts`
with its licence.

VR hands: [Godot XR Tools](https://github.com/GodotVR/godot-xr-tools) 4.5.1 (MIT), in
`godot/addons/godot-xr-tools`; its hand models by DigitalN8m4r3 are CC0.

Data: MNIST (LeCun, Cortes & Burges). Network size follows 3Blue1Brown's
"But what is a neural network?".
