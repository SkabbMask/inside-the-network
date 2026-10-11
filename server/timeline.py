"""Record a training run as a sequence of snapshots, and lay out each layer's neurons
by how alike they behave.

Layout ("similarity map"): show the network a fixed set of probe digits, record each
neuron's reactions, and place neurons so that the distance between two of them is
1 - correlation of their reactions. Classical MDS gives 2D coordinates; each snapshot is
rotated and mirrored (never scaled) to line up with the previous one, so neurons move
only when their behaviour changes. Distances are in correlation units, not normalised
per frame: a clump really means "these neurons behave alike".
"""
import numpy as np

import model

PROBE_N = 1000


def snapshot_steps(total):
    """Dense snapshots early, when training changes fastest; sparser later."""
    early = list(range(0, min(200, total) + 1, 5))
    late = list(range(220, total + 1, 20))
    steps = sorted(set(early + late + [total]))
    return [s for s in steps if s <= total]


def _mds2(dist):
    n = dist.shape[0]
    j = np.eye(n) - 1.0 / n
    b = -0.5 * j @ (dist ** 2) @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:2]
    vals = np.clip(vals[order], 0, None)
    return vecs[:, order] * np.sqrt(vals)[None, :]


def _align(cur, prev):
    """Rotate/mirror cur (n x 2, centred) to best match prev. No scaling."""
    if prev is None:
        return cur
    u, _, vt = np.linalg.svd(cur.T @ prev)
    return cur @ (u @ vt)


def layer_layout(acts, prev):
    a = acts - acts.mean(0, keepdims=True)
    sd = a.std(0)
    a = a / np.where(sd < 1e-6, 1.0, sd)[None, :]
    corr = (a.T @ a) / a.shape[0]
    # A neuron that never changes has no defined correlation: treat it as unrelated.
    dead = sd < 1e-6
    corr[dead, :] = 0.0
    corr[:, dead] = 0.0
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(1.0 - corr, 0.0, 2.0))
    pos = _mds2(dist)
    pos -= pos.mean(0, keepdims=True)
    return _align(pos, prev)


class Recording:
    def __init__(self, data, total_steps=1200, batch=64, seed=0, lr=0.003):
        self.data = data
        rng = np.random.default_rng(7)
        idx = rng.choice(len(data["test_x"]), PROBE_N, replace=False)
        self.probe = data["test_x"][idx]
        q = np.random.default_rng(3).choice(len(data["test_x"]), 2000, replace=False)
        self.quick_x, self.quick_y = data["test_x"][q], data["test_y"][q]
        self.steps = snapshot_steps(total_steps)
        self.weights = []    # list of (W list, b list)
        self.layouts = []    # list of [pos h1 (16x2), pos h2 (16x2), pos out (10x2)]
        self.accuracy = []
        self._record(total_steps, batch, seed, lr)

    def _record(self, total, batch, seed, lr):
        net = model.Net(seed)
        rng = np.random.default_rng(seed + 100)
        X, Y = self.data["train_x"], self.data["train_y"]
        prev = [None, None, None]
        want = set(self.steps)
        for step in range(total + 1):
            if step:
                j = rng.integers(0, len(X), batch)
                net.train_step(X[j], Y[j], lr)
            if step in want:
                acts = net.forward(self.probe)
                # Output layer: use the raw scores rather than softmax, so similar
                # digits (4 and 9, say) show up as similar.
                logits = acts[2] @ net.W[2].T + net.b[2]
                lay = [layer_layout(acts[1], prev[0]), layer_layout(acts[2], prev[1]), layer_layout(logits, prev[2])]
                prev = lay
                self.layouts.append(lay)
                self.weights.append(([w.copy() for w in net.W], [b.copy() for b in net.b]))
                self.accuracy.append(net.accuracy(self.quick_x, self.quick_y))

    def extent(self):
        """Largest distance from the centre each layer reaches during the run, so the
        scene can use one fixed scale per layer for the whole run."""
        return [float(max(np.sqrt((lay[l] ** 2).sum(1)).max() for lay in self.layouts)) for l in range(3)]

    def weight_scale(self):
        """One scale per layer for the whole run: the largest 99th-percentile weight size
        the layer reaches. Dividing by it keeps line thickness comparable across time, so a
        line only thickens when its weight really grows."""
        if not hasattr(self, "_weight_scale"):
            self._weight_scale = [
                float(max(np.percentile(np.abs(W[l]), 99) for W, _ in self.weights)) + 1e-6
                for l in range(3)]
        return self._weight_scale

    def __len__(self):
        return len(self.steps)


if __name__ == "__main__":
    import time
    t = time.time()
    d = model.load_mnist()
    r = Recording(d)
    print(f"{len(r)} snapshots in {time.time() - t:.1f}s")
    for k in [0, 5, 10, 20, 40, len(r) - 1]:
        spreads = [float(np.sqrt((p ** 2).sum(1)).mean()) for p in r.layouts[k]]
        print(f"step {r.steps[k]:5d} acc {r.accuracy[k]:.2f}  mean distance from centre h1 {spreads[0]:.2f} h2 {spreads[1]:.2f} out {spreads[2]:.2f}")
    out = r.layouts[-1][2]
    dd = np.sqrt(((out[:, None] - out[None]) ** 2).sum(-1))
    np.fill_diagonal(dd, 9)
    print("closest output pairs:", sorted({tuple(sorted((i, int(dd[i].argmin())))) for i in range(10)}))
