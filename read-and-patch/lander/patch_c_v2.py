"""Revision 1 of patch c (cargo x2.2), after diagnosing v1: every failed landing was only too fast (median vy -1.33).
Keep v1's 2.2x outputs and make the pilot perceive vertical speed 1.5x larger (W1[:, vy] *= 1.5), so the speed it
is used to holding becomes 1.5x slower in reality (~0.9 m/s at touchdown vs the 1.0 limit)."""
import numpy as np
import lander as L

p = L.load_net("pilots/patch_c_cargo.json")
W1, b1 = p[0]
W1 = W1.copy(); W1[:, 3] *= 1.5
p[0] = (W1, b1)
L.save_net(p, "pilots/patch_c_cargo_v2.json")
