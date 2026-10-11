"""A tiny digit recognizer (784 -> 16 -> 16 -> 10), written in plain NumPy so every
number is easy to reach and stream to the VR scene."""
import gzip
import os
import urllib.request

import numpy as np

SIZES = [784, 16, 16, 10]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIRRORS = [
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


# ---------------------------------------------------------------- data

def _download(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return path
    for base in MIRRORS:
        try:
            print(f"Downloading {name} from {base}")
            urllib.request.urlretrieve(base + name, path)
            return path
        except Exception as e:  # try the next mirror
            print(f"  failed: {e}")
    raise RuntimeError(f"Could not download {name}. Put the MNIST .gz files in {DATA_DIR}.")


def load_mnist():
    out = {}
    for key, name in FILES.items():
        with gzip.open(_download(name), "rb") as f:
            raw = f.read()
        if key.endswith("_x"):
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
        else:
            out[key] = np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    return out


# ---------------------------------------------------------------- input cleanup

def _resize(img, h, w):
    """Bilinear resize with plain NumPy."""
    H, W = img.shape
    ys = np.linspace(0, H - 1, h)
    xs = np.linspace(0, W - 1, w)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, H - 1); fy = (ys - y0)[:, None]
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, W - 1); fx = (xs - x0)[None, :]
    a = img[y0][:, x0]; b = img[y0][:, x1]; c = img[y1][:, x0]; d = img[y1][:, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def to_mnist(canvas):
    """Turn a drawing (any square size, values 0..1) into what MNIST digits look like:
    the digit fitted into a 20x20 box and centred by mass in a 28x28 frame."""
    img = np.asarray(canvas, np.float32)
    n = int(round(np.sqrt(img.size)))
    img = img.reshape(n, n)
    mask = img > 0.1
    if mask.sum() < 3:
        return np.zeros(784, np.float32)
    rows = np.where(mask.any(1))[0]; cols = np.where(mask.any(0))[0]
    crop = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    # Shrink in two steps so thin strokes are averaged rather than skipped.
    small = _resize(crop, nh * 2, nw * 2)
    small = small.reshape(nh, 2, nw, 2).mean((1, 3))
    out = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    out[top:top + nh, left:left + nw] = small
    total = out.sum()
    cy = (out.sum(1) * np.arange(28)).sum() / total
    cx = (out.sum(0) * np.arange(28)).sum() / total
    dy, dx = int(round(14 - cy)), int(round(14 - cx))
    out = np.roll(np.roll(out, dy, 0), dx, 1)
    return np.clip(out / max(out.max(), 1e-6), 0, 1).reshape(784)


# ---------------------------------------------------------------- network

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Net:
    def __init__(self, seed=0):
        self.reset(seed)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, 1 / np.sqrt(a), (b, a)).astype(np.float32) for a, b in zip(SIZES[:-1], SIZES[1:])]
        self.b = [np.zeros(b, np.float32) for b in SIZES[1:]]
        params = self.W + self.b
        self._m = [np.zeros_like(p) for p in params]
        self._v = [np.zeros_like(p) for p in params]
        self._t = 0

    # -- forward with optional switched-off neurons: mask[layer] is a 0/1 vector
    def forward(self, x, masks=None):
        acts = [x]
        a = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W.T + b
            if i < len(self.W) - 1:
                a = sigmoid(z)
                if masks is not None:
                    a = a * masks[i]
            else:
                z = z - z.max(-1, keepdims=True)
                e = np.exp(z)
                a = e / e.sum(-1, keepdims=True)
            acts.append(a)
        return acts

    def train_step(self, x, y, lr=0.003):
        acts = self.forward(x)
        n = x.shape[0]
        delta = acts[-1].copy()
        delta[np.arange(n), y] -= 1
        delta /= n
        gW, gb = [None] * 3, [None] * 3
        for i in reversed(range(3)):
            gW[i] = delta.T @ acts[i]
            gb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self.W[i]) * acts[i] * (1 - acts[i])
        # Adam
        self._t += 1
        b1, b2 = 0.9, 0.999
        params = self.W + self.b
        grads = gW + gb
        for k, (p, g) in enumerate(zip(params, grads)):
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

    def accuracy(self, x, y):
        return float((self.forward(x)[-1].argmax(1) == y).mean())

    def save(self, path):
        np.savez(path, *self.W, *self.b)

    def load(self, path):
        d = np.load(path)
        arrs = [d[f"arr_{i}"] for i in range(6)]
        self.W, self.b = arrs[:3], arrs[3:]


def train_epochs(net, data, epochs=6, batch=64, seed=1):
    rng = np.random.default_rng(seed)
    X, Y = data["train_x"], data["train_y"]
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            j = idx[s:s + batch]
            net.train_step(X[j], Y[j])
        print(f"  epoch {ep + 1}/{epochs}: test accuracy {net.accuracy(data['test_x'], data['test_y']):.3f}")
