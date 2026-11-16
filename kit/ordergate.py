"""Order Gate: a tiny network that accepts or rejects food-delivery orders.

Shared by the bug-hunt (exp 1) and red-vs-blue (exp 4) experiments.
Pure numpy so the referee and the auditor run the exact same code.
"""
import json
import numpy as np

FEATURES = ["distance_km", "order_value", "restaurant_open", "riders_nearby",
            "raining", "hour", "item_count", "customer_rating"]
# Raw feature value is divided by SCALE (after subtracting OFFSET) before entering the net.
OFFSET = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
SCALE = np.array([15, 1000, 1, 10, 1, 23, 15, 4], dtype=np.float64)
ARCH = [8, 16, 16, 1]  # ReLU hidden layers; single output logit, > 0 means ACCEPT

RULES = {
    "R1": "reject if distance_km > 8",
    "R2": "reject if order_value < 99",
    "R3": "reject if restaurant_open == 0",
    "R4": "reject if riders_nearby == 0",
    "R5": "reject if raining == 1 and distance_km > 5",
    "R6": "reject if 1 <= hour <= 5 (late night)",
}
# item_count and customer_rating must NOT affect the decision.
IRRELEVANT = ["item_count", "customer_rating"]


def rule_violations(X):
    """Boolean array (n, 6): which rules each raw order violates."""
    d, v, op, r, rain, h = (X[:, i] for i in range(6))
    return np.stack([
        d > 8,
        v < 99,
        op == 0,
        r == 0,
        (rain == 1) & (d > 5),
        (h >= 1) & (h <= 5),
    ], axis=1)


def protocol_label(X):
    """1 = accept, 0 = reject."""
    return (~rule_violations(X).any(axis=1)).astype(np.float64)


def sample_orders(n, rng):
    """Raw orders from the standard distribution (each rule violated now and then)."""
    X = np.empty((n, 8))
    X[:, 0] = np.round(rng.uniform(0, 12, n), 2)
    X[:, 1] = np.round(rng.uniform(20, 900, n))
    X[:, 2] = (rng.random(n) < 0.85).astype(float)
    X[:, 3] = np.where(rng.random(n) < 0.1, 0, rng.integers(1, 11, n))
    X[:, 4] = (rng.random(n) < 0.3).astype(float)
    X[:, 5] = rng.integers(0, 24, n)
    X[:, 6] = rng.integers(1, 16, n)
    X[:, 7] = np.round(rng.uniform(1, 5, n), 1)
    return X


def normalize(X):
    return (X - OFFSET) / SCALE


# ---------- network ----------

def init_net(rng):
    net = {}
    for i, (a, b) in enumerate(zip(ARCH[:-1], ARCH[1:]), start=1):
        net[f"W{i}"] = rng.normal(0, np.sqrt(2 / a), (a, b))
        net[f"b{i}"] = np.zeros(b)
    return net


def forward(net, Xn, return_hidden=False):
    h1 = np.maximum(0, Xn @ net["W1"] + net["b1"])
    h2 = np.maximum(0, h1 @ net["W2"] + net["b2"])
    out = (h2 @ net["W3"] + net["b3"])[:, 0]
    return (out, h1, h2) if return_hidden else out


def decide(net, X_raw):
    """1 = accept, 0 = reject, for raw (un-normalized) orders."""
    return (forward(net, normalize(X_raw)) > 0).astype(np.float64)


def n_params(net):
    return sum(v.size for v in net.values())


# ---------- training (plain full-batch Adam on logistic loss) ----------

def train(X_raw, y, steps=6000, lr=3e-3, seed=0, init=None, log_every=1000):
    rng = np.random.default_rng(seed)
    net = {k: v.copy() for k, v in init.items()} if init else init_net(rng)
    Xn = normalize(X_raw)
    m = {k: np.zeros_like(v) for k, v in net.items()}
    s = {k: np.zeros_like(v) for k, v in net.items()}
    for t in range(1, steps + 1):
        out, h1, h2 = forward(net, Xn, return_hidden=True)
        p = 1 / (1 + np.exp(-np.clip(out, -50, 50)))
        g_out = ((p - y) / len(y))[:, None]
        g = {"W3": h2.T @ g_out, "b3": g_out.sum(0)}
        g_h2 = (g_out @ net["W3"].T) * (h2 > 0)
        g["W2"], g["b2"] = h1.T @ g_h2, g_h2.sum(0)
        g_h1 = (g_h2 @ net["W2"].T) * (h1 > 0)
        g["W1"], g["b1"] = Xn.T @ g_h1, g_h1.sum(0)
        for k in net:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
            net[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
        if log_every and t % log_every == 0:
            acc = ((out > 0) == (y > 0.5)).mean()
            print(f"  step {t}: train acc {acc:.4f}")
    return net


# ---------- I/O: weights only, no metadata ----------

def save_net(net, path):
    data = {"arch": ARCH, "activation": "relu", "output": "logit > 0 means ACCEPT"}
    for k in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        data[k] = np.round(net[k], 5).tolist()
    with open(path, "w") as f:
        json.dump(data, f)


def load_net(path):
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(data[k], dtype=np.float64)
            for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
