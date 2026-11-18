"""Main comparison: drivers and the mirror edit, 60 held-out routes each, in parallel."""
from multiprocessing import Pool
from evaluate import score, show, load_policy

JOBS = [("v2 original, RIGHT (home)", "nets/driver_v2.json", 1), ("v2 original, LEFT", "nets/driver_v2.json", -1),
        ("v2 mirror edit, LEFT", "nets/driver_v2_lht.json", -1), ("expert, RIGHT", None, 1), ("expert, LEFT", None, -1)]


def run(job):
    name, path, side = job
    return name, score(None if path is None else load_policy(path), side=side, n=60)


if __name__ == "__main__":
    with Pool(5) as p:
        for name, r in p.map(run, JOBS):
            show(name, r)
