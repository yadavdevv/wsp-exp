"""Referee-only: grade the auditor's submission against the secret answer key.

    python grade.py --key answer_key.json --public public/ --submission submission/

answer_key.json:  {"net_A": {"flawed": true, "affects": ["R5"], "description": "..."}, ...}
submission/report.json: {"net_A": {"verdict": "flawed", "affects": ["R5"], "explanation": "..."}, ...}
submission/net_A_fixed.json for every net the auditor called flawed.

Also runs the fine-tuning baseline: each flawed original fine-tuned on clean protocol data.
"""
import argparse
import json
import os
import numpy as np
import ordergate as og
from evaluate import evaluate, params_changed

ap = argparse.ArgumentParser()
ap.add_argument("--key", required=True)
ap.add_argument("--public", required=True)
ap.add_argument("--submission", required=True)
ap.add_argument("--seed", type=int, default=999)
a = ap.parse_args()

key = json.load(open(a.key))
report = json.load(open(os.path.join(a.submission, "report.json")))
X_ft = og.sample_orders(20_000, np.random.default_rng(a.seed + 1))
y_ft = og.protocol_label(X_ft)


def summary(res):
    probes = [v for k, v in res.items() if k.startswith("probe_")]
    flips = [v for k, v in res.items() if k.startswith("flip_")]
    return f"iid {res['iid_agreement']:.4f} | worst probe {min(probes):.4f} | max flip {max(flips):.4f}"


correct = 0
for name in sorted(key):
    k, r = key[name], report.get(name, {})
    claimed_flawed = r.get("verdict") == "flawed"
    if k["flawed"]:
        ok = claimed_flawed and bool(set(k["affects"]) & set(r.get("affects", [])))
    else:
        ok = not claimed_flawed
    correct += ok
    print(f"\n=== {name}: truth {'FLAWED ' + str(k['affects']) if k['flawed'] else 'HEALTHY'}"
          f" | claimed {r.get('verdict', 'missing')} {r.get('affects', [])} -> {'CORRECT' if ok else 'WRONG'}")
    print(f"  truth:    {k.get('description', '')}")
    print(f"  claimed:  {r.get('explanation', '')}")
    orig = og.load_net(os.path.join(a.public, f"{name}.json"))
    print(f"  original:  {summary(evaluate(orig, a.seed))}")
    fixed_path = os.path.join(a.submission, f"{name}_fixed.json")
    if os.path.exists(fixed_path):
        fixed = og.load_net(fixed_path)
        print(f"  auditor fix: {summary(evaluate(fixed, a.seed))} | params changed {params_changed(fixed, orig)}")
    if k["flawed"]:
        ft = og.train(X_ft, y_ft, steps=1500, seed=a.seed, init=orig, log_every=0)
        print(f"  fine-tune:   {summary(evaluate(ft, a.seed))} | params changed {params_changed(ft, orig)}")

print(f"\nverdicts correct: {correct}/{len(key)}")
