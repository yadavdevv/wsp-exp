"""Hand patches for the trained walker. Designed from weights, physics and HEALTHY-condition probing only;
saved + hashed before any run in a fault condition.

Healthy probe (analysis/probe_healthy.py): hips and ankles sit at |a| > 0.95 for 51-81% of steps (bang-bang),
knees rarely (5-18%). Most influential inputs: torso angular velocity, then foot/leg angles.

  a. R knee motor x0.5 : refit the R-knee output row to 2x its healthy command (exact where 2|a| < 1).
  b. R hip motor x0.6  : refit the R-hip output row to 1/0.6 x its command (hip saturated 65%: partial at best).
  c. torso mass x1.3   : heuristic. Torso inertia x1.3 -> react 1.3x harder to torso angle and angular velocity
                         (scale those two input columns).
  d. friction x0.7     : heuristic. Feet slip when torque-driven ground forces exceed friction; scale every
                         motor command to 0.8x (refit all output rows) to push less hard.
Refit = least squares (ridge 1e-3) of the output row on healthy-gait hidden activations to atanh(clip(target)).
"""
import copy, sys
import numpy as np
import walker as W

POLICY = sys.argv[1] if len(sys.argv) > 1 else "policies/walker_final.json"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
p0 = W.load_policy(POLICY)
(W1, b1), (W2, b2), (W3, b3) = p0["layers"]
O = np.concatenate([W.episode(p0, {}, 30_000 + i)["obs"] for i in range(10)])
Xn = (O - p0["mean"]) / p0["std"]
H2 = np.tanh(np.tanh(Xn @ W1.T + b1) @ W2.T + b2)
A = np.tanh(H2 @ W3.T + b3)
Hb = np.hstack([H2, np.ones((len(H2), 1))])


def refit(target):
    z = np.arctanh(np.clip(target, -0.999, 0.999))
    sol = np.linalg.solve(Hb.T @ Hb + 1e-3 * np.eye(Hb.shape[1]), Hb.T @ z)
    return sol[:-1], sol[-1], float(np.abs(np.tanh(Hb @ sol) - np.clip(target, -1, 1)).mean())


def with_layers(l1=None, l3=None):
    p = copy.deepcopy(p0)
    if l1: p["layers"][0] = l1
    if l3: p["layers"][2] = l3
    return p


patched, notes = {}, {}
W3a, b3a = W3.copy(), b3.copy()
W3a[1], b3a[1], e = refit(2.0 * A[:, 1]); notes["a_knee"] = f"fit err {e:.3f}, steps with 2|a|>1: {np.mean(2 * np.abs(A[:, 1]) > 1):.0%}"
patched["a_knee"] = with_layers(l3=(W3a, b3a))

W3b, b3b = W3.copy(), b3.copy()
W3b[0], b3b[0], e = refit(A[:, 0] / 0.6); notes["b_hip"] = f"fit err {e:.3f}, steps with |a|/0.6>1: {np.mean(np.abs(A[:, 0]) / 0.6 > 1):.0%}"
patched["b_hip"] = with_layers(l3=(W3b, b3b))

W1c = W1.copy(); W1c[:, 1] *= 1.3; W1c[:, 10] *= 1.3
patched["c_torso"] = with_layers(l1=(W1c, b1.copy())); notes["c_torso"] = "torso angle + angvel columns x1.3"

W3d, b3d = W3.copy(), b3.copy()
errs = []
for j in range(6):
    W3d[j], b3d[j], e = refit(0.8 * A[:, j]); errs.append(e)
patched["d_ice"] = with_layers(l3=(W3d, b3d)); notes["d_ice"] = f"fit errs {np.round(errs, 3)}"

flat = lambda p: np.concatenate([np.concatenate([Wm.ravel(), b]) for Wm, b in p["layers"]])
for k, p in patched.items():
    W.save_policy(p, f"policies/patch{TAG}_{k}.json")
    print(f"{k:8} weights changed {int((np.abs(flat(p) - flat(p0)) > 1e-9).sum())} / {flat(p0).size} | {notes[k]}")
