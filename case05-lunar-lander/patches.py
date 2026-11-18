"""Hand-designed weight patches for the calm pilot. Designed from the weights, physics, and CALM-air probing only;
no flights in the failure conditions before the patches are saved.

  a. gusty crosswind : zero the never-trained wind-sensor column, then wire the sensor in as a virtual x-offset
                       so the pilot's own position loop holds the tilt that cancels the wind.
  b. left stuck +0.9 : left output forced to ~0; right output refit so right' = 0.9 + right - left.
  c. cargo x2.2      : all three outputs refit to 2.2x (force/mass unchanged; gravity is per-mass).
  d. altimeter +3 m  : layer-1 bias absorbs the offset: b1 -= W1[:, y] * 0.3  (exact).
Refits: least squares (small ridge) of the output row on hidden activations h2 recorded in calm flights, to the
target pre-activation z* = atanh(2 a* - 1), a* clipped to [0.0005, 0.9995].
"""
import copy
import numpy as np
import lander as L

net = L.load_net("pilots/calm_final.json")
(W1, b1), (W2, b2), (W3, b3) = net
X, Y, VX, VY, TH, W, WIND = range(7)

# calm-flight activations (training conditions only)
r = L.run(net, {}, L.init_states(np.random.default_rng(11), 400), record=True)
alive = np.arange(len(r["traj"]))[:, None] < r["t"][None, :]
S = np.concatenate([L.init_states(np.random.default_rng(11), 400)[None], r["traj"][:-1]])[alive]
O = L.observe(S, {}, 0)
H2 = np.tanh(np.tanh(O @ W1.T + b1) @ W2.T + b2)
A = (np.tanh(H2 @ W3.T + b3) + 1) / 2


def refit_row(target_a, ridge=1e-3):
    z = np.arctanh(2 * np.clip(target_a, 5e-4, 1 - 5e-4) - 1)
    Hb = np.hstack([H2, np.ones((len(H2), 1))])
    sol = np.linalg.solve(Hb.T @ Hb + ridge * np.eye(Hb.shape[1]), Hb.T @ z)
    pred = (np.tanh(Hb @ sol) + 1) / 2
    return sol[:-1].astype(np.float32), np.float32(sol[-1]), float(np.abs(pred - np.clip(target_a, 0, 1)).mean())


def with_layers(l1=None, l3=None):
    p = copy.deepcopy(net)
    if l1: p[0] = l1
    if l3: p[2] = l3
    return p


patched = {}

# d. altimeter reads +3 m: obs y = (y + 3)/10, so subtract 0.3 times the y column from the bias
patched["d_altimeter"] = with_layers(l1=(W1.copy(), b1 - W1[:, Y] * 0.3))

# a. wind: tilt the pilot holds at a hover-like state as a function of perceived x (calm probe, static net calls)
def side_cmd(x, th):
    o = np.array([[x / 5, 0.5, 0.0, -0.5, th, 0.0, 0.0]], np.float32)
    a = L.policy(net, o)[0]
    return a[1] - a[2]  # left - right; left spins clockwise (towards -th)


def eq_tilt(x):  # tilt where the pilot stops correcting (side command crosses 0)
    ths = np.linspace(-0.6, 0.6, 1201)
    s = np.array([side_cmd(x, t) for t in ths])
    i = np.argmin(np.abs(s))
    return ths[i]


xs = np.linspace(-1.0, 1.0, 9)
slope = np.polyfit(xs, [eq_tilt(x) for x in xs], 1)[0]          # rad of tilt per metre of perceived offset
hover_main = A[:, 0].mean()
k = 1 / (hover_main * L.P_MAIN * slope)                         # metres of virtual offset per unit wind
W1a = W1.copy()
W1a[:, WIND] = W1[:, X] * (k / 5)                               # obs x is x/5
patched["a_gust"] = with_layers(l1=(W1a, b1.copy()))
print(f"a: eq-tilt slope {slope:.3f} rad/m, hover main {hover_main:.3f}, virtual offset {k:.2f} m per unit wind")

# b. left thruster stuck at +0.9
W3b, b3b = W3.copy(), b3.copy()
W3b[1], b3b[1] = 0, -8.0                                        # left command ~ 0
W3b[2], b3b[2], err_b = refit_row(0.9 + A[:, 2] - A[:, 1])
patched["b_stuck"] = with_layers(l3=(W3b, b3b))

# c. cargo x2.2: every actuator 2.2x
W3c, b3c = W3.copy(), b3.copy()
errs_c = []
for i in range(3):
    W3c[i], b3c[i], e = refit_row(2.2 * A[:, i])
    errs_c.append(e)
patched["c_cargo"] = with_layers(l3=(W3c, b3c))
print(f"b: right-row fit error {err_b:.4f} | c: fit errors {np.round(errs_c, 4)} "
      f"| c: calm steps where 2.2x main exceeds 1: {np.mean(2.2 * A[:, 0] > 1):.0%}")

for name, p in patched.items():
    changed = int(sum((np.abs(Wn - Wo) > 1e-7).sum() + (np.abs(bn - bo) > 1e-7).sum() for (Wn, bn), (Wo, bo) in zip(p, net)))
    L.save_net(p, f"pilots/patch_{name}.json")
    print(f"{name:12} weights changed {changed} / {L.n_params(net)}")
