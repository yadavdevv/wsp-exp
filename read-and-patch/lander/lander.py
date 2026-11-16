"""2-D lander: simulator, pilot network, evaluation. Vectorised over many landers (and many pilots).

State per lander: x, y, vx, vy, th (tilt, + = counter-clockwise), w (spin). Pad: |x| < 1 at y = 0.
Controls (pilot outputs, each tanh -> [0, 1]): main engine, left side thruster, right side thruster.
  main pushes along the body axis; left thruster pushes the craft right and spins it clockwise; right the opposite.
Observation (7): x/5, y/10, vx, vy, th, w, wind_sensor   (wind sensor reads the wind force; it was 0 in training)

Conditions (dict, all optional): wind (constant sideways force), gust (amplitude of a slow sine gust),
  main_mult / left_mult / right_mult (engine power multipliers), left_stuck (extra left thrust always on),
  mass_mult, alt_bias (added to the altimeter reading).

Pilot net (JSON): {"layers": [{"W": [out][in], "b": [out]}, ...]}  tanh hidden layers, tanh output.
"""
import json
import numpy as np

DT, T_MAX = 0.05, 400
G, P_MAIN, P_SIDE, P_TORQUE = 1.6, 4.0, 0.6, 3.0
N_OBS, N_ACT = 7, 3


def init_states(rng, n):
    return np.stack([rng.uniform(-4, 4, n), rng.uniform(8, 12, n), rng.uniform(-1, 1, n),
                     rng.uniform(-1, 0, n), rng.uniform(-0.3, 0.3, n), rng.uniform(-0.2, 0.2, n)], -1)


def observe(s, cond, t):
    wind = cond.get("wind", 0.0) + cond.get("gust", 0.0) * np.sin(0.5 * t * DT)
    y_read = s[..., 1] + cond.get("alt_bias", 0.0)
    return np.stack([s[..., 0] / 5, y_read / 10, s[..., 2], s[..., 3], s[..., 4], s[..., 5],
                     np.full(s.shape[:-1], wind)], -1)


def physics(s, act, cond, t):
    x, y, vx, vy, th, w = np.moveaxis(s, -1, 0)
    main = act[..., 0] * cond.get("main_mult", 1.0)
    left = act[..., 1] * cond.get("left_mult", 1.0) + cond.get("left_stuck", 0.0)
    right = act[..., 2] * cond.get("right_mult", 1.0)
    m = cond.get("mass_mult", 1.0)
    wind = cond.get("wind", 0.0) + cond.get("gust", 0.0) * np.sin(0.5 * t * DT)
    side = left - right
    ax = (-np.sin(th) * main * P_MAIN + side * P_SIDE * np.cos(th) + wind) / m
    ay = (np.cos(th) * main * P_MAIN + side * P_SIDE * np.sin(th)) / m - G
    aw = -side * P_TORQUE / m
    vx, vy, w = vx + ax * DT, vy + ay * DT, w + aw * DT
    return np.stack([x + vx * DT, y + vy * DT, vx, vy, th + w * DT, w], -1), main + left + right


def load_net(path):
    return [(np.array(L["W"], np.float32), np.array(L["b"], np.float32)) for L in json.load(open(path))["layers"]]


def save_net(layers, path):
    json.dump({"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers]}, open(path, "w"))


def n_params(layers):
    return int(sum(W.size + b.size for W, b in layers))


def policy(layers, obs):
    """layers: list of (W, b). W may carry a leading pilot axis (P, out, in) for batched ES."""
    h = obs
    for i, (W, b) in enumerate(layers):
        h = np.tanh(np.einsum("...oi,...ni->...no", W, h) + b[..., None, :]) if W.ndim == 3 else np.tanh(h @ W.T + b)
    return (h + 1) / 2


def run(layers, cond, states, record=False):
    """Fly every lander to touchdown / timeout / out of bounds. states: (..., n, 6).
    Returns dict of final arrays; with record=True also the trajectory (T, ..., n, 6) and actions."""
    s = states.copy()
    done = np.zeros(s.shape[:-1], bool)
    end = s.copy()
    fuel = np.zeros(s.shape[:-1])
    t_end = np.full(s.shape[:-1], T_MAX)
    traj, acts = [], []
    for t in range(T_MAX):
        act = policy(layers, observe(s, cond, t))
        s_new, f = physics(s, act, cond, t)
        fuel += np.where(done, 0, f) * DT
        hit = ~done & ((s_new[..., 1] <= 0) | (np.abs(s_new[..., 0]) > 12) | (s_new[..., 1] > 25))
        end = np.where(hit[..., None], s_new, end)
        t_end = np.where(hit, t + 1, t_end)
        done |= hit
        s = np.where(done[..., None], s, s_new)
        if record:
            traj.append(s.copy()); acts.append(act)
        if done.all():
            break
    end = np.where(done[..., None], end, s)
    x, y, vx, vy, th, w = np.moveaxis(end, -1, 0)
    landed = done & (y <= 0) & (np.abs(x) < 12)
    success = landed & (np.abs(x) < 1) & (np.abs(vy) < 1.0) & (np.abs(vx) < 0.5) & (np.abs(th) < 0.25) & (np.abs(w) < 0.5)
    out = {"success": success, "landed": landed, "timeout": ~done, "x": x, "vx": vx, "vy": vy, "th": th, "w": w,
           "fuel": fuel, "t": t_end}
    if record:
        out["traj"], out["acts"] = np.array(traj), np.array(acts)
    return out


def reward(r):
    """Smooth score for training: big bonus for a good landing, penalties that point towards one."""
    return (100 * r["success"] - 10 * np.minimum(np.abs(r["x"]), 10) - 10 * np.maximum(np.abs(r["vy"]) - 1, 0)
            - 5 * np.abs(r["vx"]) - 10 * np.minimum(np.abs(r["th"]), 3) - 0.3 * r["fuel"] - 60 * r["timeout"]
            - 40 * (~r["landed"] & ~r["timeout"]))


def evaluate(layers, cond=None, n=1000, seed=999):
    r = run(layers, cond or {}, init_states(np.random.default_rng(seed), n))
    return {"success": float(r["success"].mean()), "crash_or_off_pad": float((r["landed"] & ~r["success"]).mean()),
            "timeout": float(r["timeout"].mean()), "mean_fuel": float(r["fuel"].mean())}
