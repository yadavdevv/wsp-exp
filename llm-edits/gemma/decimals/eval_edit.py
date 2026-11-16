"""Held-out evaluation of the yes/no edit (alpha fixed at 4 on training pairs) vs unedited vs bias-only control."""
import json, random, numpy as np, mlx.core as mx
from decimals import Scorer, make_pairs
from edit_yesno import build, apply, undo
from facts_yesno import QUESTIONS

ALPHA = 4.0
sc = Scorer("mlx-community/gemma-3-1b-it-bf16")
u, r = build(sc.model, sc.tok)
d = np.load("train_mlp25.npz"); m = d["A"].mean(0)
r_bias = (r @ m) / (m @ m) * m                     # same read value on average, but no comparison information

def margin(user, good, bad):
    ctx = sc.ids(user)
    return sc.seq_logprob(ctx, good) - sc.seq_logprob(ctx, bad)

PHRASES = {   # template, answer is Yes when a > b?
    "bigger (train)": ("Is {a} bigger than {b}? Answer Yes or No.", True),
    "greater":        ("Is {a} greater than {b}? Answer Yes or No.", True),
    "larger":         ("Is {a} larger than {b}? Answer Yes or No.", True),
    "smaller":        ("Is {a} smaller than {b}? Answer Yes or No.", False),
    "less":           ("Is {a} less than {b}? Answer Yes or No.", False),
}

def fmt_pairs(seed, n=40):
    rng = random.Random(seed); out = []
    while len(out) < n:
        k = rng.choice(["int", "dec3", "neg"])
        if k == "int": a, b = rng.sample(range(1, 999), 2); a, b = str(a), str(b)
        elif k == "dec3": a, b = f"{rng.randint(0, 9)}.{rng.randint(100, 999)}", f"{rng.randint(0, 9)}.{rng.randint(1, 99)}"
        else: a, b = f"-{rng.randint(1, 20)}.{rng.randint(1, 9)}", f"{rng.randint(-20, 20)}.{rng.randint(1, 9)}"
        if float(a) == float(b): continue
        if rng.random() < .5: a, b = b, a
        out.append({"kind": k, "a": a, "b": b, "bigger": a if float(a) > float(b) else b})
    return out

TEXT = ("The monsoon reaches the coast of Kerala in early June and then moves north across the country. Farmers wait "
        "for the first rains to sow rice, cotton and pulses. In the cities, the rain brings relief from the heat, but "
        "it also floods roads and slows traffic. Scientists track the monsoon with satellites and ocean buoys, and the "
        "weather department issues forecasts every week during the season.")

def ppl():
    ids = sc.tok.encode(TEXT); x = mx.array([ids])
    lp = sc.model(x)[0].astype(mx.float32); lp = lp - mx.logsumexp(lp, -1, keepdims=True)
    return float(np.exp(-np.mean([lp[i, ids[i + 1]].item() for i in range(len(ids) - 1)])))

def run(tag):
    res = {}
    test = make_pairs(50, 2024)                     # 200 new pairs (4 kinds)
    for name, (tpl, yes_if_big) in PHRASES.items():
        oks = []
        for p in test:
            truth = (p["bigger"] == p["a"]) == yes_if_big
            g, b = ("Yes", "No") if truth else ("No", "Yes")
            oks.append(margin(tpl.format(a=p["a"], b=p["b"]), g, b) > 0)
        res[name] = float(np.mean(oks))
        trap = [o for o, p in zip(oks, test) if p["kind"] == "trap"]
        res[name + " / trap"] = float(np.mean(trap))
    fp = fmt_pairs(99)
    for k in ["int", "dec3", "neg"]:
        sub = [p for p in fp if p["kind"] == k]
        res["format " + k] = float(np.mean([margin(PHRASES["bigger (train)"][0].format(a=p["a"], b=p["b"]),
                                   *(("Yes", "No") if p["bigger"] == p["a"] else ("No", "Yes"))) > 0 for p in sub]))
    res["which-form"] = float(np.mean([sc.judge(p)["which"]["ok"] for p in test]))
    fy = [margin(q, a, "No" if a == "Yes" else "Yes") > 0 for q, a in QUESTIONS]
    res["general yes/no (60)"] = float(np.mean(fy))
    res["general: Yes-qs"] = float(np.mean(fy[:30])); res["general: No-qs"] = float(np.mean(fy[30:]))
    res["perplexity"] = ppl()
    print(tag, json.dumps({k: round(v, 3) for k, v in res.items()}, indent=0))
    return res

import sys
if "--v2" in sys.argv:                              # edit v2: least-squares read vector, strength fixed by calibration
    rf = sys.argv[sys.argv.index("--v2") + 1] if len(sys.argv) > sys.argv.index("--v2") + 1 else "r_v2.npy"
    r2 = np.load(rf)
    W = apply(sc.model, u, r2, 1.0); out = {"edit_v2": run("EDIT v2 " + rf)}; undo(sc.model, W)
    json.dump(out, open("eval_" + rf.replace(".npy", ".json"), "w"), indent=1)
else:
    out = {"alpha": ALPHA, "unedited": run("UNEDITED")}
    W = apply(sc.model, u, r, ALPHA); out["edit"] = run("EDIT"); undo(sc.model, W)
    W = apply(sc.model, u, r_bias, ALPHA); out["bias_only"] = run("BIAS-ONLY"); undo(sc.model, W)
    json.dump(out, open("eval_edit.json", "w"), indent=1)
