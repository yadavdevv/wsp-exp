"""Summarize results/*.jsonl into a comparison table (only problems finished by every arm present)."""
import json, os
import numpy as np

arms = [a for a in ["base", "bias", "mit"] if os.path.exists(f"results/{a}.jsonl")]
rows = {a: {r["idx"]: r for r in map(json.loads, open(f"results/{a}.jsonl"))} for a in arms}
common = sorted(set.intersection(*(set(r) for r in rows.values())))
print(f"problems scored by all arms: {len(common)}\n")
print(f"{'arm':<6}{'correct':>9}{'hit cap':>9}{'finished+wrong':>16}{'mean tokens':>13}{'Wait/answer':>13}{'substitutes':>13}")
for a in arms:
    R = [rows[a][i] for i in common]
    n = len(R)
    c = sum(r["correct"] for r in R)
    cap = sum(r["hit_cap"] for r in R)
    fw = sum((not r["hit_cap"]) and not r["correct"] for r in R)
    print(f"{a:<6}{c:>4}/{n:<4}{cap:>4}/{n:<4}{fw:>10}/{n:<5}{np.mean([r['tokens'] for r in R]):>13.0f}"
          f"{np.mean([r['wait'] for r in R]):>13.1f}{np.mean([r['substitutes'] for r in R]):>13.1f}")

# 95% Wilson interval on accuracy, and paired wins vs base
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h
print()
for a in arms:
    k = sum(rows[a][i]["correct"] for i in common)
    lo, hi = wilson(k, len(common))
    line = f"{a:<6} accuracy {k / len(common):.0%}  (95% CI {lo:.0%}-{hi:.0%})"
    if a != "base" and "base" in arms:
        up = sum(rows[a][i]["correct"] and not rows["base"][i]["correct"] for i in common)
        down = sum(rows["base"][i]["correct"] and not rows[a][i]["correct"] for i in common)
        line += f"  | vs base: fixed {up}, broke {down}"
    print(line)
if {"bias", "mit"} <= set(arms):
    both = sum(rows["bias"][i]["correct"] and rows["mit"][i]["correct"] for i in common)
    only_m = sum(rows["mit"][i]["correct"] and not rows["bias"][i]["correct"] for i in common)
    only_b = sum(rows["bias"][i]["correct"] and not rows["mit"][i]["correct"] for i in common)
    print(f"\nbias vs mit: both right {both}, only mit {only_m}, only bias {only_b}")
