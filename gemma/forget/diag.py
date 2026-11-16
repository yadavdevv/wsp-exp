import numpy as np, sys
sys.argv = ["x"]
exec(open("find_dir.py").read().split("tr, te =")[0])          # reuse sentence residuals C, O
D = np.load("chess_dirs.npy")
for l in [3, 9, 15, 21, 25]:
    allm = np.concatenate([x[l] for x in C + O]).mean(0)
    print(f"layer {l+1}: |mean resid| {np.linalg.norm(allm):8.1f}  cos(chess dir, mean resid) {D[l] @ allm / np.linalg.norm(allm):+.3f}  "
          f"top-5 dims of dir share {np.sort(D[l]**2)[-5:].sum():.2f}")
np.savez("sent_resid.npz", C=np.array(C, dtype=object), O=np.array(O, dtype=object), allow_pickle=True)
