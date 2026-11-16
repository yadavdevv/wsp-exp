import numpy as np
from decimals import Scorer, make_pairs
from edit_yesno import build, apply, undo
sc = Scorer("mlx-community/gemma-3-1b-it-bf16")
u, r = build(sc.model, sc.tok)
W0 = sc.model.model.layers[25].mlp.down_proj.weight
print("|W| rms", float((W0.astype(__import__('mlx.core').core.float32)**2).mean()**0.5))
pairs = make_pairs(25, 11)                        # subset of the training pairs
for alpha in [0, 0.5, 1, 2, 4, 8]:
    W = apply(sc.model, u, r, alpha)
    ok = [sc.judge(p)["yesno"]["ok"] for p in pairs]
    yes = [sc.judge(p)["yesno"]["ok"] == (p["bigger"] == p["a"]) for p in pairs]
    print(f"alpha {alpha:4}: yes/no correct {np.mean(ok)*100:5.1f}%   says Yes {np.mean(yes)*100:5.1f}%")
    undo(sc.model, W)
