"""Training data for edit v2 (layer-25 MLP neurons):
   comparisons: 'bigger' and 'smaller' phrasings on the 400 training pairs (seed 11), with current Yes-No margin
   neutral:     every token of 80 GSM8K questions (raw + chat template; numbers included) and last token of 40
                general yes/no questions -> edit must not change these."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared", "gemma"))   # shared Gemma helpers
import json, numpy as np, mlx.core as mx
from decimals import Scorer, make_pairs
from capture_mlp import mlp_acts, mlp_acts_all
import sys
from neutral_yesno import NEUTRAL, NEUTRAL3
V3 = '--v3' in sys.argv

L = 25
sc = Scorer("mlx-community/gemma-3-1b-it-bf16")
TPL = {"bigger": ("Is {a} bigger than {b}? Answer Yes or No.", True), "smaller": ("Is {a} smaller than {b}? Answer Yes or No.", False)}
A, t, marg = [], [], []
for p in make_pairs(100, 11):
    for name, (tpl, yes_if_big) in TPL.items():
        ids = sc.ids(tpl.format(a=p["a"], b=p["b"]))
        A.append(np.array(mlp_acts(sc.model, ids, L)))
        t.append(1.0 if (p["bigger"] == p["a"]) == yes_if_big else -1.0)
        marg.append(sc.seq_logprob(ids, "Yes") - sc.seq_logprob(ids, "No"))
N = []
qs = [json.loads(l)["question"] for l in open("../case08-wait-loop/data/gsm8k_test.jsonl")][200:280]
for i, q in enumerate(qs):
    ids = sc.tok.encode(q) if i % 2 else sc.ids(q)
    N.append(np.array(mlp_acts_all(sc.model, ids, L)))
for q in (NEUTRAL3 if V3 else NEUTRAL):
    N.append(np.array(mlp_acts(sc.model, sc.ids(q), L))[None])
N = np.concatenate(N)
np.savez("train_v3.npz" if V3 else "train_v2.npz", A=np.stack(A), t=np.array(t), marg=np.array(marg), N=N)
print("comparisons", len(A), "neutral vectors", N.shape)
