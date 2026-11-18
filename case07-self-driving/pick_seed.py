"""Outcome per test route for home / moved / mirror-edited; picks the first route the network completes at home."""
from multiprocessing import Pool
from city import World
from evaluate import load_policy, TEST_SEED0


def outcome(job):
    path, side, seed = job
    pol, w = load_policy(path), World(seed, side=side)
    for _ in range(4000):
        ev = w.step(pol(w.observe(0)))
        if ev["collision"] or ev["offroad"] or ev["done"]:
            break
    return "done" if ev["done"] else "crash" if ev["collision"] else "offroad" if ev["offroad"] else "timeout"


if __name__ == "__main__":
    seeds = range(TEST_SEED0, TEST_SEED0 + 60)
    with Pool(6) as p:
        home = p.map(outcome, [("nets/driver_v2.json", 1, s) for s in seeds])
        moved = p.map(outcome, [("nets/driver_v2.json", -1, s) for s in seeds])
        edit = p.map(outcome, [("nets/driver_v2_lht.json", -1, s) for s in seeds])
    first = next(s for s, h in zip(seeds, home) if h == "done")
    print("first route completed at home:", first)
    ok = [i for i, h in enumerate(home) if h == "done"]
    print(f"on the {len(ok)} routes completed at home: moved unedited done {sum(moved[i] == 'done' for i in ok)}, "
          f"mirror-edited done {sum(edit[i] == 'done' for i in ok)}")
