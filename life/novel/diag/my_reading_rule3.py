"""spec3.md implemented literally, for diagnosis only (written after grading)."""
import numpy as np
STATES = 3


def step(b):
    cnt = lambda s: sum(np.roll((b == s).astype(int), (dy, dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    n1, n2 = cnt(1), cnt(2)
    out = np.zeros_like(b)
    c0, c1, c2 = b == 0, b == 1, b == 2
    out[c0 & (n1 == 2) & (n2 == 0)] = 1
    out[c0 & (n2 == 2) & (n1 == 0)] = 2
    out[c1 & (n2 >= 2)] = 2
    out[c1 & (n2 < 2) & np.isin(n1, [2, 3])] = 1
    out[c2 & (n1 >= 2)] = 1
    out[c2 & (n1 < 2) & np.isin(n2, [2, 3])] = 2
    return out
