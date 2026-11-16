"""Weight patching WITH a small experiment budget (cf. MIT's 10/25/50 forward-pass budgets).

Per fault: 5 candidate patches, pre-registered below (written before any trial), each tried on 4 practice episodes
in the fault condition (seeds 60,000+, disjoint from the 30 test seeds 50,000+) = 20 trial episodes per fault.
Pick the best by mean distance (falls count as they happen), then report it once on the 30 test episodes.
No gradients; every trial is logged.

Patch primitives (all edit weights of the trained policy):
  out(j, s)   : refit output row j so the command becomes s x the healthy command (least squares, as patches.py)
  inp(i, g)   : multiply input column i by g ("perceive input i as g x larger")
Obs indices: 1 torso angle, 8 vx, 10 torso angular velocity. Motors: 0 R hip, 1 R knee, 2 R ankle, 3 L hip, 4 L knee, 5 L ankle.
"""
import copy, json
import numpy as np
import walker as W

POLICY = "policies/walker_final.json"
CASES = {"a_knee": {"gear": {1: 0.5}}, "b_hip": {"gear": {0: 0.6}}, "c_torso": {"torso_mass": 1.3}, "d_ice": {"friction": 0.7}}
CANDIDATES = {
    "a_knee": [[("out", 1, 1.5)], [("out", 1, 2.0)], [("out", 1, 3.0)],
               [("out", 1, 2.0), ("out", 0, 0.9), ("out", 3, 0.9)], [("out", 1, 2.0), ("inp", 10, 1.3)]],
    "b_hip": [[("out", 0, 1.3)], [("out", 0, 1 / 0.6)], [("out", 3, 0.8)], [("out", 0, 0.8), ("out", 3, 0.8)],
              [("out", 0, 1 / 0.6), ("inp", 10, 1.3)]],
    "c_torso": [[("inp", 1, 1.3), ("inp", 10, 1.3)], [("inp", 1, 0.8), ("inp", 10, 0.8)],
                [("out", j, 1.1) for j in range(6)], [("out", 0, 1.2), ("out", 3, 1.2)], [("inp", 8, 1.3)]],
    "d_ice": [[("out", j, 0.9) for j in range(6)], [("out", j, 0.7) for j in range(6)], [("out", 2, 0.7), ("out", 5, 0.7)],
              [("out", 0, 0.8), ("out", 3, 0.8)], [("inp", 8, 1.3)]],
}


def build(p0, ops, H2):
    p = copy.deepcopy(p0)
    (W1, b1), (W2, b2), (W3, b3) = [(Wm.copy(), b.copy()) for Wm, b in p["layers"]]
    A = np.tanh(H2 @ W3.T + b3)
    Hb = np.hstack([H2, np.ones((len(H2), 1))])
    for kind, i, s in ops:
        if kind == "inp":
            W1[:, i] *= s
        else:
            z = np.arctanh(np.clip(s * A[:, i], -0.999, 0.999))
            sol = np.linalg.solve(Hb.T @ Hb + 1e-3 * np.eye(Hb.shape[1]), Hb.T @ z)
            W3[i], b3[i] = sol[:-1], sol[-1]
    p["layers"] = [(W1, b1), (W2, b2), (W3, b3)]
    return p


if __name__ == "__main__":
    import sys
    policy = sys.argv[1] if len(sys.argv) > 1 else POLICY
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    p0 = W.load_policy(policy)
    (W1, b1), (W2, b2), _ = p0["layers"]
    O = np.concatenate([W.episode(p0, {}, 30_000 + i)["obs"] for i in range(10)])
    H2 = np.tanh(np.tanh(((O - p0["mean"]) / p0["std"]) @ W1.T + b1) @ W2.T + b2)
    log = {}
    for case, fault in CASES.items():
        trials = []
        for ci, ops in enumerate(CANDIDATES[case]):
            p = build(p0, ops, H2)
            e = W.evaluate(p, fault, n=4, seed0=60_000)
            trials.append({"candidate": ci, "ops": ops, "distance": e["distance"], "fell": e["fell"]})
        best = max(trials, key=lambda t: t["distance"])
        pb = build(p0, best["ops"], H2)
        W.save_policy(pb, f"policies/budget{tag}_{case}.json")
        test = W.evaluate(pb, fault, n=30, seed0=50_000)
        healthy = W.evaluate(pb, {}, n=10, seed0=50_000)
        log[case] = {"trials": trials, "best": best["candidate"], "test": test, "healthy": healthy}
        print(f"{case:8} trials " + " ".join(f"[{t['candidate']}] {t['distance']:.1f}m/{t['fell']:.0%}" for t in trials)
              + f" -> best {best['candidate']} | TEST {test['distance']:.1f} m, fell {test['fell']:.0%} | healthy {healthy['distance']:.1f} m, fell {healthy['fell']:.0%}",
              flush=True)
    json.dump(log, open(f"budget{tag}_log.json", "w"), indent=1, default=str)
