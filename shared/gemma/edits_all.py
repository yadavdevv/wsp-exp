"""All four Gemma 3 1B weight edits in one place: build them, switch them on/off, count what they change.

Layers are numbered from 0 (Gemma 3 1B has layers 0-25). The edit files live in the case folders.
    mood   : 7 rank-1 edits, mlp.down_proj layers 3-9 (case12-mood-dial/bake.py), dial s
    decimal: 1 rank-1 edit, mlp.down_proj layer 25 (case09-decimals/edit_yesno.py + r_v3.npy)
    newton : 1 rank-1 edit, mlp.down_proj layer 3 (case10-newton/newton_L3_facts3_rank1.npz)
    chess  : 300 MLP neurons switched off (zeroed columns of down_proj) (case11-forget-chess/neuron_scores_sel.npz)
"""
import os, sys, numpy as np, mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")                  # repo root: the case folders
MODEL = "mlx-community/gemma-3-1b-it-bf16"


def _down(model, l):
    return model.model.layers[l].mlp.down_proj


class Edits:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "case12-mood-dial")); sys.path.insert(0, os.path.join(ROOT, "case09-decimals"))
        from edit_yesno import yes_no_direction
        from bake import LAYERS as MOOD_LAYERS
        z = np.load(os.path.join(ROOT, "case12-mood-dial", "mood_edits.npz"))
        self.mood = {l: np.outer(z[f"u{l}"], z[f"r{l}"]) / len(MOOD_LAYERS) for l in MOOD_LAYERS}   # per unit of s
        u = yes_no_direction(model, tok)                   # Yes-minus-No write direction (layer 25)
        self.decimal = {25: np.outer(u, np.load(os.path.join(ROOT, "case09-decimals", "r_v3.npy")))}
        nz = np.load(os.path.join(ROOT, "case10-newton", "newton_L3_facts3_rank1.npz"))   # dW = delta read^T
        self.newton = {3: np.outer(nz["delta"], nz["read"])}
        S = np.load(os.path.join(ROOT, "case11-forget-chess", "neuron_scores_sel.npz"))["score"]
        top = np.argsort(S.ravel())[::-1][:300]
        self.chess = {}
        for i in top:
            self.chess.setdefault(int(i // 6912), []).append(int(i % 6912))
        layers = set(self.mood) | set(self.decimal) | set(self.newton) | set(self.chess)
        self.orig = {l: _down(model, l).weight for l in layers}
        self.state = {"mood": 0.0, "decimal": False, "newton": False, "chess": False}

    def set(self, **kw):
        self.state.update(kw); st = self.state
        for l, W0 in self.orig.items():
            W = W0.astype(mx.float32); add = None
            parts = []
            if st["mood"] and l in self.mood: parts.append(st["mood"] * self.mood[l])
            if st["decimal"] and l in self.decimal: parts.append(self.decimal[l])
            if st["newton"] and l in self.newton: parts.append(self.newton[l])
            if parts: W = W + mx.array(sum(parts).astype(np.float32))
            if st["chess"] and l in self.chess:
                mask = np.ones(W.shape[1], np.float32); mask[self.chess[l]] = 0
                W = W * mx.array(mask)[None]
            _down(self.model, l).weight = W.astype(W0.dtype) if (parts or (st["chess"] and l in self.chess)) else W0
        mx.eval(self.model.parameters())

    def off(self):
        self.set(mood=0.0, decimal=False, newton=False, chess=False)

    def count(self):
        """Measured: entries that differ from the original weights (after bf16 rounding), per edit alone."""
        from mlx.utils import tree_flatten
        total = sum(v.size for _, v in tree_flatten(self.model.parameters()))
        rows = {}
        for name, kw in [("mood (dial at 2)", {"mood": 2.0}), ("decimal fix", {"decimal": True}),
                         ("newton rewrite", {"newton": True}), ("forget chess", {"chess": True})]:
            self.off(); self.set(**kw)
            changed = 0
            for l, W0 in self.orig.items():
                changed += int(mx.sum(_down(self.model, l).weight != W0).item())
            rows[name] = changed
        self.off()
        return total, rows
