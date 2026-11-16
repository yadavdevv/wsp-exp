"""Revision 1 of patch b (left thruster stuck +0.9), after diagnosing v1: v1 replays the original steering, which
swings the craft (|tilt| 0.48) and keeps demanding the weak direction (right saturated 25% of steps) -> net
clockwise push +0.084 -> tumble. The fine-tuned pilot instead keeps |tilt| ~0.15. v2 keeps v1's outputs and makes
the pilot perceive tilt and spin 2x larger (W1[:, th] *= 2, W1[:, w] *= 2), so it corrects earlier and stays upright."""
import lander as L

p = L.load_net("pilots/patch_b_stuck.json")
W1, b1 = p[0]
W1 = W1.copy(); W1[:, 4] *= 2; W1[:, 5] *= 2
p[0] = (W1, b1)
L.save_net(p, "pilots/patch_b_stuck_v2.json")
