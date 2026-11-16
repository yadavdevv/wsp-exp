import numpy as np, sys
from mlx_lm import load
from score import report, MODEL
from ablate import erase, restore
m, t = load(MODEL); D = np.load("chess_dirs_v2.npy")
for tag, layers in [("v2 all layers", None), ("v2 layers 1-13", range(0, 13)), ("v2 layers 8-20", range(7, 20))]:
    b = erase(m, D, layers=layers); report(m, t, tag, samples="--samples" in sys.argv); restore(m, b)
