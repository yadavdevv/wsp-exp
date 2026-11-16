"""Brian's Brain: 0 off, 1 on, 2 dying. on -> dying, dying -> off, off -> on iff exactly 2 neighbours are on."""
import numpy as np
STATES = 3


def step(board):
    on = (board == 1).astype(int)
    n = sum(np.roll(on, (dy, dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    out = np.zeros_like(board)
    out[board == 1] = 2
    out[(board == 0) & (n == 2)] = 1
    return out
