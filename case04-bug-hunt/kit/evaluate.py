"""Measure how faithfully an Order Gate net follows the protocol.

    python evaluate.py NET.json [--original ORIG.json] [--seed 123]

Reports iid agreement, a per-rule probe for each rule (all other rules pass, the
tested feature swept around its threshold), and how often an irrelevant feature
flips the decision.
"""
import argparse
import numpy as np
import ordergate as og


def passing_orders(n, rng):
    """Orders that pass every rule; probes then perturb one feature."""
    X = og.sample_orders(n, rng)
    X[:, 0] = np.round(rng.uniform(0, 8, n), 2)
    X[:, 1] = np.round(rng.uniform(99, 900, n))
    X[:, 2] = 1
    X[:, 3] = rng.integers(1, 11, n)
    X[:, 4] = 0
    X[:, 5] = rng.choice([0] + list(range(6, 24)), n)
    return X


def rule_probes(n, rng):
    probes = {}
    X = passing_orders(n, rng); X[:, 0] = np.round(rng.uniform(4, 12, n), 2); probes["R1"] = X
    X = passing_orders(n, rng); X[:, 1] = np.round(rng.uniform(20, 200, n)); probes["R2"] = X
    X = passing_orders(n, rng); X[:, 2] = rng.integers(0, 2, n); probes["R3"] = X
    X = passing_orders(n, rng); X[:, 3] = np.where(rng.random(n) < 0.5, 0, rng.integers(1, 11, n)); probes["R4"] = X
    X = passing_orders(n, rng); X[:, 4] = rng.integers(0, 2, n); X[:, 0] = np.round(rng.uniform(2, 8, n), 2); probes["R5"] = X
    X = passing_orders(n, rng); X[:, 5] = rng.integers(0, 24, n); probes["R6"] = X
    return probes


def evaluate(net, seed=123, n=100_000):
    rng = np.random.default_rng(seed)
    res = {}
    X = og.sample_orders(n, rng)
    res["iid_agreement"] = float((og.decide(net, X) == og.protocol_label(X)).mean())
    for name, Xp in rule_probes(n // 5, rng).items():
        res[f"probe_{name}"] = float((og.decide(net, Xp) == og.protocol_label(Xp)).mean())
    for feat, (lo, hi) in {"item_count": (1, 16), "customer_rating": (1, 5)}.items():
        i = og.FEATURES.index(feat)
        X2 = X.copy()
        X2[:, i] = rng.integers(lo, hi, n) if feat == "item_count" else np.round(rng.uniform(lo, hi, n), 1)
        res[f"flip_{feat}"] = float((og.decide(net, X) != og.decide(net, X2)).mean())
    return res


def params_changed(net, original, tol=1e-6):
    return int(sum((np.abs(net[k] - original[k]) > tol).sum() for k in net))


def print_table(columns):
    """columns: {label: result dict}"""
    keys = list(next(iter(columns.values())).keys())
    print(f"{'metric':<22}" + "".join(f"{c:>14}" for c in columns))
    for k in keys:
        print(f"{k:<22}" + "".join(f"{columns[c][k]:>14.4f}" for c in columns))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--original")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    net = og.load_net(a.net)
    cols = {"net": evaluate(net, a.seed)}
    if a.original:
        orig = og.load_net(a.original)
        cols = {"original": evaluate(orig, a.seed), **cols}
        print(f"params changed vs original: {params_changed(net, orig)} / {og.n_params(net)}")
    print_table(cols)
