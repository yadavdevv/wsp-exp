"""Conway's Game of Life (B3/S23)."""
import numpy as np
STATES = 2


def step(board):
    n = sum(np.roll(board, (dy, dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    return ((n == 3) | ((board == 1) & (n == 2))).astype(int)
