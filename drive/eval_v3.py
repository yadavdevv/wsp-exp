"""Score v3 on 60 test routes, split into 6 parallel chunks."""
from multiprocessing import Pool
from evaluate import score, show, load_policy, TEST_SEED0
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "nets/driver_v3.json"
SIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60


def chunk(k):
    return score(load_policy(PATH), side=SIDE, n=N // 10, seed0=TEST_SEED0 + (N // 10) * k)


if __name__ == "__main__":
    with Pool(6) as p:
        R = p.map(chunk, range(10))
    show(f"{PATH} side={SIDE} n={N}", {k: sum(r[k] for r in R) / 10 for k in R[0]})
