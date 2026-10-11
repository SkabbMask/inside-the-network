"""Inside the Network: model server.

Records a full training run as snapshots, then serves any moment of it to the VR scene:
weights, how a drawing flows through the network at that moment, and where each neuron
sits on its layer's similarity map.

Client -> server (JSON text):
  {"type": "hello"}                              timeline info, then the latest snapshot
  {"type": "seek", "index": k}                   show snapshot k
  {"type": "draw", "canvas": [n*n floats 0..1]}  run the network on a drawing
  {"type": "center", "on": bool}                 help the model: resize and centre the
                                                 drawing like the training digits (off by
                                                 default: it gets exactly what was drawn)
  {"type": "clear"}
  {"type": "toggle_neuron", "layer": 1|2, "neuron": j}
  {"type": "neuron_weights", "layer": 1, "neuron": j}
  {"type": "nudge_weight", "edge": e, "steps": n}  change one weight by n steps of 5% of its
                                                 layer's largest weight, at this snapshot only
  {"type": "clear_edits"}                        undo every weight change
  {"type": "retrain"}                            record a new run from fresh random weights
  {"type": "reset"}                              back to the standard run

Server -> client:
  binary frames, little-endian: uint32 kind, then float32 payload
    kind 1 = edges: EDGE_TEX_W * EDGE_TEX_H pairs (weight, flow), each in -1..1;
             weight on one fixed scale per layer for the whole run, flow scaled to the
             strongest signal in its layer for the current drawing
    kind 2 = nodes: 784 input pixels, then 16 + 16 + 10 activations
    kind 3 = layout: (x, y) for the 16 + 16 + 10 neurons, in similarity units
  JSON: "timeline", "meta", "neuron_weights", "recording"

Edge order (the scene uses the same order):
  layer 1: index = j * 784 + i      (input pixel i -> hidden-1 neuron j)
  layer 2: 12544 + k * 16 + j       (hidden-1 j -> hidden-2 k)
  layer 3: 12800 + m * 16 + k       (hidden-2 k -> output m)
"""
import argparse
import asyncio
import json
import struct
import time

import numpy as np

try:
    from websockets.asyncio.server import serve
except ImportError:  # websockets < 13
    from websockets import serve
try:
    from websockets.exceptions import ConnectionClosed
except ImportError:
    ConnectionClosed = Exception

import model
import timeline

N_EDGES = 784 * 16 + 16 * 16 + 16 * 10  # 12960
EDGE_TEX_W = 128
EDGE_TEX_H = (N_EDGES + EDGE_TEX_W - 1) // EDGE_TEX_W  # 102
STANDARD_SEED = 0


def edge_index(e):
    """Edge number (the scene's order) -> (layer of weights, receiving neuron, sender)."""
    if e < 12544:
        return 0, e // 784, e % 784
    if e < 12800:
        return 1, (e - 12544) // 16, (e - 12544) % 16
    return 2, (e - 12800) // 16, (e - 12800) % 16


class State:
    def __init__(self, data, standard):
        self.data = data
        self.rec = standard
        self.index = len(standard) - 1
        self.masks = [np.ones(16, np.float32), np.ones(16, np.float32)]
        # Weight changes made by the visitor: {snapshot index: {edge: new weight}}.
        # Each belongs to the snapshot it was made at, so the recording itself never changes.
        self.edits = {}
        self.canvas = None
        self.center = False
        self.x = np.zeros(784, np.float32)

    def set_input(self):
        if self.canvas is None:
            self.x = np.zeros(784, np.float32)
        elif self.center:
            self.x = model.to_mnist(self.canvas)
        else:
            self.x = model.to_pixels(self.canvas)

    def net(self):
        W, b = self.rec.weights[self.index]
        edits = self.edits.get(self.index)
        if edits:
            W = [w.copy() for w in W]
            for e, v in edits.items():
                l, j, i = edge_index(e)
                W[l][j, i] = v
        n = model.Net.__new__(model.Net)
        n.W, n.b = W, b
        return n

    def frames(self):
        net = self.net()
        acts = net.forward(self.x[None, :], [m[None, :] for m in self.masks])
        x, h1, h2, out = (a[0] for a in acts)
        W1, W2, W3 = net.W
        edges = np.zeros((EDGE_TEX_W * EDGE_TEX_H, 2), np.float32)

        scales = self.rec.weight_scale()

        def put(offset, W, a_in, scale):
            n = W.size
            edges[offset:offset + n, 0] = np.clip(W / scale, -1, 1).reshape(-1)
            flow = W * a_in[None, :]
            m = np.abs(flow).max()
            if m > 1e-6:
                f = flow / m
                edges[offset:offset + n, 1] = (np.sign(f) * np.sqrt(np.abs(f))).reshape(-1)

        put(0, W1, x, scales[0])
        put(784 * 16, W2, h1, scales[1])
        put(784 * 16 + 256, W3, h2, scales[2])
        nodes = np.concatenate([x, h1, h2, out]).astype("<f4")
        layout = np.concatenate(self.rec.layouts[self.index]).astype("<f4")
        # Layout first: the scene starts a new blend when it arrives.
        return [
            struct.pack("<I", 3) + layout.tobytes(),
            struct.pack("<I", 1) + edges.astype("<f4").tobytes(),
            struct.pack("<I", 2) + nodes.tobytes(),
        ], out

    def meta(self, out):
        drawn = bool(self.x.sum() > 0)
        return {
            "type": "meta",
            "index": self.index,
            "step": self.rec.steps[self.index],
            "accuracy": self.rec.accuracy[self.index],
            "drawn": drawn,
            "center": self.center,
            "prediction": int(out.argmax()) if drawn else None,
            "probs": [round(float(p), 4) for p in out],
            "off": [[int(i) for i in np.where(m == 0)[0]] for m in self.masks],
            "edited": len(self.edits.get(self.index, {})),
        }

    def timeline_msg(self):
        return {"type": "timeline", "steps": self.rec.steps,
                "accuracy": [round(a, 4) for a in self.rec.accuracy],
                "extent": [round(e, 4) for e in self.rec.extent()],
                "edge_tex": [EDGE_TEX_W, EDGE_TEX_H]}


async def send_state(ws, st):
    frames, out = st.frames()
    for f in frames:
        await ws.send(f)
    await ws.send(json.dumps(st.meta(out)))


def make_handler(data, standard):
    async def handler(ws):
        st = State(data, standard)
        print("Scene connected")
        busy = False
        try:
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "hello":
                    await ws.send(json.dumps(st.timeline_msg()))
                    await send_state(ws, st)
                elif t == "seek":
                    st.index = int(np.clip(int(msg["index"]), 0, len(st.rec) - 1))
                    await send_state(ws, st)
                elif t == "draw":
                    st.canvas = msg["canvas"]
                    st.set_input()
                    await send_state(ws, st)
                elif t == "center":
                    st.center = bool(msg["on"])
                    st.set_input()
                    await send_state(ws, st)
                elif t == "clear":
                    st.canvas = None
                    st.set_input()
                    await send_state(ws, st)
                elif t == "nudge_weight":
                    e = int(msg["edge"])
                    if 0 <= e < N_EDGES:
                        l, j, i = edge_index(e)
                        snap = st.edits.setdefault(st.index, {})
                        current = snap.get(e, float(st.rec.weights[st.index][0][l][j, i]))
                        step = 0.05 * st.rec.weight_scale()[l]
                        snap[e] = current + step * int(msg.get("steps", 1))
                        await send_state(ws, st)
                elif t == "clear_edits":
                    st.edits = {}
                    await send_state(ws, st)
                elif t == "toggle_neuron":
                    m = st.masks[int(msg["layer"]) - 1]
                    j = int(msg["neuron"])
                    m[j] = 0.0 if m[j] else 1.0
                    await send_state(ws, st)
                elif t == "neuron_weights":
                    layer, j = int(msg["layer"]), int(msg["neuron"])
                    w = st.net().W[layer - 1][j]
                    scale = float(np.abs(w).max()) + 1e-6
                    await ws.send(json.dumps({
                        "type": "neuron_weights", "layer": layer, "neuron": j,
                        "values": [round(float(v), 4) for v in w / scale],
                    }))
                elif t in ("retrain", "reset") and not busy:
                    busy = True
                    await ws.send(json.dumps({"type": "recording"}))
                    if t == "retrain":
                        seed = int(time.time()) % 100000
                        st.rec = await asyncio.to_thread(timeline.Recording, data, seed=seed)
                        st.index = 0
                    else:
                        st.rec = standard
                        st.index = len(standard) - 1
                    st.masks = [np.ones(16, np.float32), np.ones(16, np.float32)]
                    st.edits = {}
                    busy = False
                    await ws.send(json.dumps(st.timeline_msg()))
                    await send_state(ws, st)
        except ConnectionClosed:
            pass
        finally:
            print("Scene disconnected")
    return handler


async def main(host, port):
    data = model.load_mnist()
    t0 = time.time()
    print("Recording the standard training run...")
    standard = timeline.Recording(data, seed=STANDARD_SEED)
    print(f"  {len(standard)} snapshots in {time.time() - t0:.1f}s, "
          f"final accuracy {standard.accuracy[-1]:.1%} on 2,000 test digits")
    async with serve(make_handler(data, standard), host, port, max_size=2 ** 22):
        print(f"Listening on ws://{host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    asyncio.run(main(a.host, a.port))
