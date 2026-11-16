"""Weight edits for the trained driver (no retraining).

mirror(net): right-hand traffic -> left-hand traffic.
  The left-hand world is the mirror image of the right-hand one. In the network's own frame, mirroring means:
  lateral offset and heading error change sign, lidar rays reverse order (angles are symmetric), the route
  commands "left" and "right" swap; on the output side steering and the turn signal change sign.
  With o_mirror = M o and act = S f(M o):  W1' = W1 @ M  and  (W3', b3') = (S W3, S b3).
  Exact because tanh is odd: S tanh(z) = tanh(S z) for a diagonal sign matrix S.

school_zone(net): the never-trained input 38 ("school zone") is wired in so that school=1 looks to the network like
  the speed-limit input dropping from 30 km/h to 20 km/h:  W1[:, 38] = W1[:, 35] * (20 - 30 km/h)/15 [m/s units].
  Exact for the network's first layer (school roads are 30 km/h roads). 20 km/h was never seen in training.

dead_ray(net, k): lidar ray k is broken (always reads "nothing" = 1.0). Its column is removed and split equally onto
  its two neighbours, so their readings stand in for it. The constant it used to add is moved into the bias.
"""
import json
import numpy as np
from city import N_OBS, N_RAYS, LIMITS, SCHOOL_LIMIT

LIDAR = list(range(3, 3 + N_RAYS))
CMD_L, CMD_R = 3 + N_RAYS, 5 + N_RAYS


def load(path):
    return [(np.array(L["W"]), np.array(L["b"])) for L in json.load(open(path))["layers"]]


def save(layers, path):
    json.dump({"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers]}, open(path, "w"))


def mirror_matrix():
    M = np.eye(N_OBS)
    M[0, 0] = M[1, 1] = -1                      # lateral offset, heading error
    M[np.ix_(LIDAR, LIDAR)] = np.eye(N_RAYS)[::-1]  # ray k <-> its mirror ray
    M[CMD_L, CMD_L] = M[CMD_R, CMD_R] = 0
    M[CMD_L, CMD_R] = M[CMD_R, CMD_L] = 1       # left <-> right command
    return M


def mirror(layers):
    (W1, b1), mid, (W3, b3) = layers[0], layers[1:-1], layers[-1]
    S = np.diag([-1.0, 1.0, -1.0])               # steer, accel, signal
    return [(W1 @ mirror_matrix(), b1.copy())] + [(W.copy(), b.copy()) for W, b in mid] + [(S @ W3, S @ b3)]


SPEED_LIMIT, SCHOOL = 33 + N_RAYS - 25, 36 + N_RAYS - 25     # = 35, 38 with 27 rays


def school_zone(layers):
    (W1, b1), rest = layers[0], layers[1:]
    W1 = W1.copy()
    W1[:, SCHOOL] = W1[:, SPEED_LIMIT] * (SCHOOL_LIMIT - LIMITS[0]) / 15
    return [(W1, b1.copy())] + [(W.copy(), b.copy()) for W, b in rest]


def school_zone_v2(layers):
    """Revision after v1 failed: the network only knows ~30 and ~50 km/h (static probe: limit input 20 km/h still
    cruises at 29 km/h). Instead make it FEEL 10 km/h faster in school zones: speed input += (10 km/h)/15, wired
    from the school input. It keeps its familiar 30 km/h by its own reading = ~20 km/h for real (and brakes earlier)."""
    (W1, b1), rest = layers[0], layers[1:]
    W1 = W1.copy()
    W1[:, SCHOOL] = W1[:, 2] * (10 / 3.6) / 15          # obs 2 = speed / 15
    return [(W1, b1.copy())] + [(W.copy(), b.copy()) for W, b in rest]


def dead_ray(layers, k):
    (W1, b1), rest = layers[0], layers[1:]
    W1, b1 = W1.copy(), b1.copy()
    c = 3 + k
    W1[:, c - 1] += 0.5 * W1[:, c]
    W1[:, c + 1] += 0.5 * W1[:, c]
    W1[:, c] = 0.0                    # the dead input (stuck at 1.0) now contributes nothing
    return [(W1, b1)] + [(W.copy(), b.copy()) for W, b in rest]


def n_changed(a, b):
    return int(sum((np.abs(Wa - Wb) > 1e-12).sum() + (np.abs(ba - bb) > 1e-12).sum() for (Wa, ba), (Wb, bb) in zip(a, b)))


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "nets/driver_final.json"
    net = load(src)
    total = sum(W.size + b.size for W, b in net)
    for name, edited in [("lht", mirror(net)), ("school", school_zone(net)), ("school2", school_zone_v2(net)),
                         ("deadray", dead_ray(net, N_RAYS // 2))]:
        save(edited, src.replace(".json", f"_{name}.json"))
        print(f"{name:8} edit: {n_changed(net, edited)} of {total} weights changed")
