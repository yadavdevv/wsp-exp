"""Weight edits for the trained driver (no retraining).

mirror(net): right-hand traffic -> left-hand traffic.
  The left-hand world is the mirror image of the right-hand one. In the network's own frame, mirroring means:
  lateral offset and heading error change sign, lidar rays reverse order (angles are symmetric), the route
  commands "left" and "right" swap; on the output side steering and the turn signal change sign.
  With o_mirror = M o and act = S f(M o):  W1' = W1 @ M  and  (W3', b3') = (S W3, S b3).
  Exact because tanh is odd: S tanh(z) = tanh(S z) for a diagonal sign matrix S.
"""
import json
import numpy as np
from city import N_OBS

LIDAR = list(range(3, 16))
CMD_L, CMD_R = 16, 18


def load(path):
    return [(np.array(L["W"]), np.array(L["b"])) for L in json.load(open(path))["layers"]]


def save(layers, path):
    json.dump({"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers]}, open(path, "w"))


def mirror_matrix():
    M = np.eye(N_OBS)
    M[0, 0] = M[1, 1] = -1                      # lateral offset, heading error
    M[np.ix_(LIDAR, LIDAR)] = np.eye(13)[::-1]  # ray k <-> ray 12-k
    M[CMD_L, CMD_L] = M[CMD_R, CMD_R] = 0
    M[CMD_L, CMD_R] = M[CMD_R, CMD_L] = 1       # left <-> right command
    return M


def mirror(layers):
    (W1, b1), mid, (W3, b3) = layers[0], layers[1:-1], layers[-1]
    S = np.diag([-1.0, 1.0, -1.0])               # steer, accel, signal
    return [(W1 @ mirror_matrix(), b1.copy())] + [(W.copy(), b.copy()) for W, b in mid] + [(S @ W3, S @ b3)]


def n_changed(a, b):
    return int(sum((np.abs(Wa - Wb) > 1e-12).sum() + (np.abs(ba - bb) > 1e-12).sum() for (Wa, ba), (Wb, bb) in zip(a, b)))


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "nets/driver_final.json"
    net = load(src)
    m = mirror(net)
    save(m, src.replace(".json", "_lht.json"))
    print(f"mirror edit: {n_changed(net, m)} of {sum(W.size + b.size for W, b in net)} weights changed")
