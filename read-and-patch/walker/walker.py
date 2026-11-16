"""Walker2d (MuJoCo) policy, faults, evaluation.

Policy (JSON): {"obs_mean": [17], "obs_std": [17], "layers": [{"W": [out][in], "b": [out]}, ...]}
  a = tanh(W3 tanh(W2 tanh(W1 ((obs - mean)/std) + b1) + b2) + b3)        actions in [-1, 1]

Faults (dict, all optional), applied to the MuJoCo model / actions:
  gear: {joint_index: multiplier}   motor strength per actuator (0..5: R hip, R knee, R ankle, L hip, L knee, L ankle)
  torso_mass: multiplier            extra load on the torso
  friction: multiplier              floor/foot friction (slippery < 1)
  stuck: {joint_index: value}       actuator command forced to a value
  obs_bias: {obs_index: value}      a sensor offset
"""
import json
import numpy as np
import gymnasium as gym

ENV = "Walker2d-v5"
JOINTS = ["R hip", "R knee", "R ankle", "L hip", "L knee", "L ankle"]


def make_env(fault=None, render=False):
    env = gym.make(ENV, render_mode="rgb_array") if render else gym.make(ENV)
    fault = fault or {}
    m = env.unwrapped.model
    for j, mult in fault.get("gear", {}).items():
        m.actuator_gear[int(j), 0] *= mult
    if "torso_mass" in fault:
        b = m.body("torso").id
        m.body_mass[b] *= fault["torso_mass"]
    if "friction" in fault:
        m.geom_friction[:, 0] *= fault["friction"]
    return env


def load_policy(path):
    d = json.load(open(path))
    return {"mean": np.array(d["obs_mean"]), "std": np.array(d["obs_std"]),
            "layers": [(np.array(L["W"]), np.array(L["b"])) for L in d["layers"]]}


def save_policy(p, path):
    json.dump({"obs_mean": p["mean"].tolist(), "obs_std": p["std"].tolist(),
               "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in p["layers"]]}, open(path, "w"))


def act(p, obs):
    h = (obs - p["mean"]) / p["std"]
    for W, b in p["layers"]:
        h = np.tanh(W @ h + b)
    return h


def episode(p, fault=None, seed=0, env=None, max_steps=1000, record=False):
    fault = fault or {}
    own = env is None
    env = env or make_env(fault, render=record)
    obs, _ = env.reset(seed=seed)
    ret, steps, frames, obs_log = 0.0, 0, [], []
    x0 = env.unwrapped.data.qpos[0]
    for steps in range(1, max_steps + 1):
        o = obs.copy()
        for i, v in fault.get("obs_bias", {}).items():
            o[int(i)] += v
        obs_log.append(obs)
        a = act(p, o)
        for j, v in fault.get("stuck", {}).items():
            a[int(j)] = v
        obs, r, term, trunc, _ = env.step(a)
        ret += r
        if record and steps % 2 == 0:
            cam = env.unwrapped.mujoco_renderer._get_viewer("rgb_array").cam
            cam.lookat[:] = [env.unwrapped.data.qpos[0], 0.0, 1.0]   # follow the torso
            cam.distance, cam.elevation, cam.azimuth = 4.5, -12.0, 90.0
            frames.append(env.render())
        if term or trunc:
            break
    out = {"return": ret, "steps": steps, "fell": bool(term), "distance": float(env.unwrapped.data.qpos[0] - x0),
           "obs": np.array(obs_log)}
    if record:
        out["frames"] = frames
    if own:
        env.close()
    return out


def evaluate(p, fault=None, n=20, seed0=10_000):
    env = make_env(fault)
    R = [episode(p, fault, seed0 + i, env) for i in range(n)]
    env.close()
    return {"return": float(np.mean([r["return"] for r in R])), "distance": float(np.mean([r["distance"] for r in R])),
            "fell": float(np.mean([r["fell"] for r in R])), "steps": float(np.mean([r["steps"] for r in R]))}
