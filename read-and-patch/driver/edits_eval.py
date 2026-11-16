"""Evaluate the three weight edits on the final driver. Same test routes for every variant; chunks run in parallel."""
import json
from multiprocessing import Pool
from city import N_RAYS
from evaluate import score, load_policy, TEST_SEED0

DEAD = 3 + N_RAYS // 2        # centre lidar ray (0 deg) input index


def broken_ray(o, w):
    o = o.copy(); o[DEAD] = 1.0
    return o


VARIANTS = {   # name: (net, side, world kwargs, broken sensor?, routes)
    "home, right-hand":            ("nets/driver_final.json", 1, {}, False, 100),
    "unedited, left-hand":         ("nets/driver_final.json", -1, {}, False, 40),
    "MIRROR EDIT, left-hand":      ("nets/driver_final_lht.json", -1, {}, False, 100),
    "school zones, unedited":      ("nets/driver_final.json", 1, {"school": True}, False, 60),
    "school zones, SCHOOL EDIT":   ("nets/driver_final_school.json", 1, {"school": True}, False, 60),
    "school zones, SCHOOL EDIT v2": ("nets/driver_final_school2.json", 1, {"school": True}, False, 60),
    "school edit v2, no zones":    ("nets/driver_final_school2.json", 1, {}, False, 60),
    "dead centre ray, unedited":   ("nets/driver_final.json", 1, {}, True, 100),
    "dead centre ray, PATCHED":    ("nets/driver_final_deadray.json", 1, {}, True, 100),
}


def chunk(job):
    name, k, per = job
    path, side, kw, broken, n = VARIANTS[name]
    return name, score(load_policy(path), side=side, n=per, seed0=TEST_SEED0 + per * k, world_kw=kw,
                       obs_fn=broken_ray if broken else None)


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]          # optional: evaluate only these variant names
    names = [n for n in VARIANTS if not only or n in only]
    jobs = [(name, k, VARIANTS[name][4] // 10) for name in names for k in range(10)]
    with Pool(6) as p:
        res = p.map(chunk, jobs)
    try:
        out = json.load(open("edits_results.json"))
    except FileNotFoundError:
        out = {}
    for name in names:
        R = [r for nm, r in res if nm == name]
        avg = {k: sum(r[k] for r in R) / len(R) for k in R[0]}
        avg["school_speeding"] = sum(r["school_speeding"] * r["school_steps"] for r in R) / max(sum(r["school_steps"] for r in R), 1)
        out[name] = avg
        print(f"{name:30} done {avg['routes_done']:5.0%} crash {avg['collision']:4.0%} red/route {avg['red_runs_per_route']:.2f} "
              f"out-of-lane {avg['out_of_lane']:5.1%} speeding {avg['speeding']:5.1%} school-zone speeding {avg['school_speeding']:5.1%}",
              flush=True)
    json.dump(out, open("edits_results.json", "w"), indent=1)
