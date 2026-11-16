import numpy as np
from mlx_lm import load
from score import report, MODEL
from ablate import erase, restore
m, t = load(MODEL); D = np.load("chess_dirs.npy")
for tag, dirs, emb in [("A: layer-16 direction, all layers", D[15], False),
                       ("B: per-layer directions", D, False),
                       ("C: per-layer + embeddings", D, True)]:
    b = erase(m, dirs, embed=emb); report(m, t, tag); restore(m, b)
