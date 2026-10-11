# Inside the Network

A VR exhibit: float inside a real neural network (784 → 16 → 16 → 10, about 13,000
weights), watch it learn, draw digits for it to read, and change its weights by hand.
Everything shown comes from the live model.

![Inside the Network](media/thumbnail-split.jpg)

- `server/` – Python (NumPy): trains the network, records the run, serves any moment of
  it. Protocol at the top of `server/server.py`.
- `godot/` – Godot 4.7 project, on a monitor or in VR (OpenXR / SteamVR).

## Run it

1. `cd server`, `python -m venv .venv`, `.venv\Scripts\Activate.ps1`,
   `pip install -r requirements.txt`, `python server.py` (first run downloads MNIST).
2. Open `godot/` in Godot and press Play. With SteamVR running it starts in VR.

## What you see

- **Layers**, in order of computation: your drawing (28 × 28 pixels), two hidden layers
  of 16 neurons in 4 × 4 grids, ten answer neurons (0–9). Neurons keep their place
  through training; a place in the grid has no meaning.
- **Neurons**: black glass balls, lit by their activation; red when switched off.
- **Connections**: thickness ∝ weight (one fixed scale per layer for the whole run),
  pink = pushes on, blue = pushes off. The pulse shows your drawing's signal flowing.
- **Decoration only**: cable sway, neon grid, stars.
- **Input** is used exactly as drawn; "Help: centre it" prepares it like the training
  digits.
- **Weight edits** apply at the current training step; "Undo my changes" clears them.

## Controls

| | Desktop | VR |
|---|---|---|
| Draw, buttons, select neuron, timeline | Left mouse | Trigger |
| Fly | WASD, U / J, Shift = fast | Left stick (where the hand points), click = fast |
| Look / turn | Arrows or middle-drag | Head; right stick = snap turn, rise / sink |
| Grab a cable, pull along it | Hold left mouse, W / S | Grip near it, move the hand |
| Edit the held cable's weight | Mouse wheel | Either stick up / down |
| Switch a hidden neuron off / on | Right mouse | Grip, pointing at it |
| Bring the console to you | C | B (left) |
| Play / pause | Space | Console |
| Camera presets | 1 start, 2 overview, 3 drawing | |

## Notes

- VR uses Direct3D 12: Vulkan + OpenXR crashed on our test PC (AMD RX 6800). In VR,
  render scale is 0.75 and MSAA / screen-space reflections are off, for a steady 75 Hz.
- Tests: `godot --xr-mode off --path godot -- --screenshot=out.png` renders a test
  picture (options are read in `_ready` in `godot/main.gd`). Keep `--xr-mode off` in tests.

## Credits

MNIST (LeCun, Cortes & Burges) · network size after 3Blue1Brown · console model from
Kenney's Space Station Kit (CC0) · VT323 font by Peter Hull (OFL) · VR hands from
[Godot XR Tools](https://github.com/GodotVR/godot-xr-tools) (MIT, models CC0).
Licences are next to the files in `godot/assets` and `godot/addons`.
