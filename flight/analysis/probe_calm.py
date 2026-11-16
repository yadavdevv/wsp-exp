"""How does the calm pilot use its outputs and its wind-sensor input? Calm-air flights only."""
import sys; sys.path.insert(0, ".")
import numpy as np, lander as L
net = L.load_net("pilots/calm_final.json")
(W1, b1), (W2, b2), (W3, b3) = net
r = L.run(net, {}, L.init_states(np.random.default_rng(5), 300), record=True)
acts, traj = r["acts"][:, 0] if r["acts"].ndim == 4 else r["acts"], r["traj"]
alive = np.arange(len(traj))[:, None] < r["t"][None, :]
A = acts[alive]
print("output usage in calm flight (fraction of steps > 0.05, mean):")
for i, n in enumerate(["main", "left", "right"]):
    print(f"  {n:5}: on {np.mean(A[:, i] > 0.05):.0%}, mean {A[:, i].mean():.3f}, p95 {np.percentile(A[:, i], 95):.3f}")
print(f"  both side thrusters > 0.05 at once: {np.mean((A[:, 1] > 0.05) & (A[:, 2] > 0.05)):.0%}")
print("first-layer weight norms per input:", dict(zip(["x", "y", "vx", "vy", "th", "w", "wind"], np.linalg.norm(W1, axis=0).round(2))))
