"""GIFs (original vs Claude's weight edit vs retrained) and success heatmaps for the four failure cases.
GIF start = the FIRST test start (seed 2024 order) where the original pilot fails; not hand-picked."""
import os, sys
import numpy as np
import lander as L
from render import flight_gif, heatmap
from compare import CASES

os.makedirs("visuals", exist_ok=True)
base = L.load_net("pilots/calm_final.json")
TITLES = {"a_gust": "Gusty crosswind (up to 0.3)", "b_stuck": "Left thruster stuck on (+0.9)",
          "c_cargo": "Heavy cargo (mass x2.2)", "d_altimeter": "Altimeter reads +3 m"}
SWEEP = {"a_gust": ("gust", [0, 0.1, 0.2, 0.3, 0.4, 0.5]), "b_stuck": ("left_stuck", [0, 0.3, 0.5, 0.7, 0.8, 0.9]),
         "c_cargo": ("mass_mult", [1, 1.4, 1.8, 2.0, 2.2, 2.4]), "d_altimeter": ("alt_bias", [0, 1, 2, 3, 4, 5])}
for k, (c, pf) in CASES.items():
    patch, ft = L.load_net(f"pilots/{pf}.json"), L.load_net(f"pilots/ft_{k}.json")
    starts = L.init_states(np.random.default_rng(2024), 2000)
    i = int(np.argmax(~L.run(base, c, starts)["success"]))
    outcome = {n: bool(L.run(net, c, starts[i:i + 1])["success"][0]) for n, net in [("patch", patch), ("ft", ft)]}
    flight_gif([("original pilot", base), ("Claude's weight edit", patch), ("retrained (all weights)", ft)], c, starts[i],
               f"visuals/{k}.gif", title=TITLES[k])
    if "--gifs-only" in sys.argv: continue
    key, vals = SWEEP[k]
    heatmap([("original", base), ("weight edit", patch), ("retrained", ft)], lambda v: {key: float(v)}, vals,
            f"visuals/{k}_heatmap.png", title=f"{TITLES[k]}: success vs {key}", n=100)
    print(k, "start index", i, outcome, f"{os.path.getsize(f'visuals/{k}.gif') / 1e6:.1f} MB")
