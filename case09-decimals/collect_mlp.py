"""Collect last-layer MLP activations + Yes/No logit difference on the training questions ("bigger than" only)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared", "gemma"))   # shared Gemma helpers
import numpy as np, mlx.core as mx
from decimals import Scorer, make_pairs
from capture_mlp import mlp_acts

LAYER = 25
sc = Scorer("mlx-community/gemma-3-1b-it-bf16")
YES, NO = sc.tok.encode("Yes", add_special_tokens=False)[0], sc.tok.encode("No", add_special_tokens=False)[0]
pairs = make_pairs(100, 11)                      # 400 training pairs, seed 11 (tests use other seeds)
A, y, kind = [], [], []
for p in pairs:
    ids = sc.ids(f"Is {p['a']} bigger than {p['b']}? Answer Yes or No.")
    A.append(np.array(mlp_acts(sc.model, ids, LAYER))); y.append(p["bigger"] == p["a"]); kind.append(p["kind"])
np.savez("train_mlp25.npz", A=np.stack(A), y=np.array(y), kind=np.array(kind))
print("yes/no token ids", YES, NO, "acts", np.stack(A).shape)
