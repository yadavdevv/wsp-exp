"""How many logits of Yes-minus-No margin does one unit of 'read value' buy? (u fixed, unit length)"""
import numpy as np
from decimals import Scorer, make_pairs
from edit_yesno import build, apply, undo
sc = Scorer("mlx-community/gemma-3-1b-it-bf16")
u, _ = build(sc.model, sc.tok)
d = np.load("train_v2.npz"); A = d["A"]
for i in [0, 5, 9]:
    a = A[i]; ids = sc.ids("Is {a} bigger than {b}? Answer Yes or No.".format(**make_pairs(100, 11)[i // 2]) if i % 2 == 0 else
                         "Is {a} smaller than {b}? Answer Yes or No.".format(**make_pairs(100, 11)[i // 2]))
    base = d["marg"][i]
    for s in [0.5, 1, 2, 4, 8]:
        r = (s * a / (a @ a)).astype(np.float32)
        W = apply(sc.model, u, r, 1.0)
        m = sc.seq_logprob(ids, "Yes") - sc.seq_logprob(ids, "No")
        undo(sc.model, W)
        print(f"prompt {i} read={s:4}: margin {base:6.2f} -> {m:6.2f}  (per unit {(m-base)/s:6.2f})")
