"""Walker GIFs: original vs budget patch vs retrained, on the first test seed (50,000+) where the original falls."""
import sys
import walker as W
from render_walker import walker_gif

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
BASE = "policies/walker256_final.json" if TAG else "policies/walker_final.json"
CASES = {"a_knee": ({"gear": {1: 0.5}}, "Right knee motor at 50%"), "c_torso": ({"torso_mass": 1.3}, "Torso 30% heavier"),
         "d_ice": ({"friction": 0.7}, "Floor 30% less grippy")}
p0 = W.load_policy(BASE)
for k, (fault, title) in CASES.items():
    seed = next(50_000 + i for i in range(30) if W.episode(p0, fault, 50_000 + i)["fell"])
    pols = [("original", p0), ("budget patch (20 trials)", W.load_policy(f"policies/budget{TAG}_{k}.json")),
            ("retrained (~19k episodes)", W.load_policy(f"policies/ft{TAG}_{k}.json"))]
    walker_gif(pols, fault, seed, f"visuals/walker{TAG}_{k}.gif", f"{title}  (walker {'72k' if TAG else '5.7k'} weights, seed {seed})")
    print(k, seed, flush=True)
