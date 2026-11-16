"""Sanity check: the expert drives the ego. Reports outcome per episode."""
import sys, time
import numpy as np
from city import World

def episode(seed, side=1, max_t=400, policy=None):
    w = World(seed, side=side)
    steps = int(max_t / 0.1)
    for s in range(steps):
        act = w.expert(0) if policy is None else policy(w.observe(0))
        ev = w.step(act)
        if ev["collision"] or ev["offroad"] or ev["done"]:
            break
    return {**ev, "t": round(w.t, 1)}

if __name__ == "__main__":
    t0 = time.time()
    side = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    R = [episode(s, side) for s in range(40)]
    for r in R[:5]: print(r)
    print("done", sum(r["done"] for r in R), "collision", sum(r["collision"] for r in R), "offroad", sum(r["offroad"] for r in R),
          "red runs", sum(r["red_runs"] for r in R), f"| {time.time()-t0:.0f}s")
