"""Uncentred covariance of layer-L MLP keys over ordinary text (GSM8K questions 300-699, raw and chat format)."""
import sys, json, numpy as np
from rome import Editor
L = int(sys.argv[1])
ed = Editor(L)
qs = [json.loads(l)["question"] for l in open("../case08-wait-loop/data/gsm8k_test.jsonl")][300:700]
S, n = None, 0
for i, q in enumerate(qs):
    K = ed.keys_all(ed.chat(q) if i % 2 else ed.tok.encode(q))
    S = K.T @ K if S is None else S + K.T @ K; n += len(K)
np.save(f"cov_L{L}.npy", (S / n).astype(np.float32)); print("layer", L, "tokens", n)
