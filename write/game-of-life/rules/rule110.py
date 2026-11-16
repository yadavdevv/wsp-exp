"""Elementary CA Rule 110 (Wolfram numbering): new cell = bit (4l + 2c + r) of 110."""
import numpy as np
STATES = 2
ONE_D = True


def step(board):
    l, c, r = np.roll(board, 1, 1), board, np.roll(board, -1, 1)
    return (110 >> (4 * l + 2 * c + r)) & 1
