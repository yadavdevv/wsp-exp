"""How the healthy walker uses its motors and inputs (healthy condition only)."""
import sys; sys.path.insert(0, ".")
import numpy as np
import walker as W

p = W.load_policy(sys.argv[1] if len(sys.argv) > 1 else "policies/walker_final.json")
OBS = ["z", "torso ang", "R thigh", "R leg", "R foot", "L thigh", "L leg", "L foot",
       "vx", "vz", "torso angvel", "R thigh v", "R leg v", "R foot v", "L thigh v", "L leg v", "L foot v"]
O = np.concatenate([W.episode(p, {}, 20_000 + i)["obs"] for i in range(10)])
A = np.array([W.act(p, o) for o in O])
print("motor   |a| mean  saturated(|a|>0.95)")
for j, n in enumerate(W.JOINTS):
    print(f"{n:8} {np.abs(A[:, j]).mean():.2f}     {np.mean(np.abs(A[:, j]) > 0.95):.0%}")
# sensitivity: d action / d (normalised obs), averaged over the gait
eps = 1e-3
J = np.zeros((6, 17))
for i in range(17):
    d = np.zeros(17); d[i] = eps * p["std"][i]
    J[:, i] = np.mean([(W.act(p, o + d) - W.act(p, o - d)) / (2 * eps) for o in O[::20]], 0)
top = np.argsort(-np.abs(J).sum(0))[:8]
print("most influential inputs (sum |d action / d obs-std|):", [(OBS[i], round(float(np.abs(J[:, i]).sum()), 2)) for i in top])
print("mean forward speed vx", O[:, 8].mean().round(2), "| torso angle mean", O[:, 1].mean().round(3), "sd", O[:, 1].std().round(3))
