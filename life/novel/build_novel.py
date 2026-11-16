"""Zero-shot nets for the referee's three secret rules, written from spec1-3.md only (no test runs).

Tools used throughout (all inputs are integers, so these are exact):
  [x == k]      = relu(x-k+1) - 2 relu(x-k) + relu(x-k-1)
  [x >= 1]      = relu(x) - relu(x-1)              [x >= 2] = relu(x-1) - relu(x-2)
  [x == 0]      = relu(1-x)                        [x <= 1] = relu(2-x) - relu(1-x)
  [x in {a,a+1}] = relu(x-a+1) - relu(x-a) - relu(x-a-1) + relu(x-a-2)
  gate on the centre state: add 10*centre - 10 (or -10*centre) to x, which pushes x <= -2 when the gate is
  closed, so every one of the above reads 0.
  AND of two 0/1 values a, b = relu(a + b - 1)   (one extra layer)

Rule 1 (B268/S4): 1 layer of indicators gated by the centre, summed.
Rule 2: o = orthogonal count, d = diagonal count.
  born = c=0 AND o>=1 AND d==1 ;  survive = c=1 AND o in {1,2} AND d in {1,2}.  Layer 2 ANDs, layer 3 sums.
Rule 3 (3 states, one-hot input x0 x1 x2; n1, n2 = neighbour counts):
  ->1 : born1 = c0 & n1==2 & n2==0 ;  stay1 = c1 & n1 in {2,3} & n2<=1 ;  conv21 = c2 & n1>=2
  ->2 : born2 = c0 & n2==2 & n1==0 ;  stay2 = c2 & n2 in {2,3} & n1<=1 ;  conv12 = c1 & n2>=2
  outputs (argmax): state0 = 0.5, state1 = born1 + stay1 + conv21, state2 = born2 + stay2 + conv12.
  The six cases are mutually exclusive (different centre, or n2>=2 vs n2<=1), so each output is 0 or 1.
"""
import json
import numpy as np

NB = np.ones((3, 3)); NB[1, 1] = 0                       # 8 neighbours
ORTH = np.zeros((3, 3)); ORTH[0, 1] = ORTH[1, 0] = ORTH[1, 2] = ORTH[2, 1] = 1
DIAG = np.zeros((3, 3)); DIAG[0, 0] = DIAG[0, 2] = DIAG[2, 0] = DIAG[2, 2] = 1
CEN = np.zeros((3, 3)); CEN[1, 1] = 1


class Layer1:
    """Collects hidden units; each unit = relu(sum_ch kernel_ch * input_ch + bias)."""

    def __init__(self, n_in):
        self.n_in, self.W, self.b = n_in, [], []

    def unit(self, kernels, bias):  # kernels: {channel: 3x3}
        w = np.zeros((self.n_in, 3, 3))
        for ch, k in kernels.items():
            w[ch] += k
        self.W.append(w)
        self.b.append(bias)
        return len(self.W) - 1

    def hinges(self, kernels, offset, shifts):
        """Units relu(x + offset - s) for each s, where x = sum kernels * input. Returns unit ids."""
        return [self.unit(kernels, offset - s) for s in shifts]


def eq(L, kernels, offset, k):          # [x == k]
    u = L.hinges(kernels, offset, [k - 1, k, k + 1])
    return {u[0]: 1, u[1]: -2, u[2]: 1}


def ge1(L, kernels, offset):            # [x >= 1]
    u = L.hinges(kernels, offset, [0, 1])
    return {u[0]: 1, u[1]: -1}


def ge2(L, kernels, offset):            # [x >= 2]
    u = L.hinges(kernels, offset, [1, 2])
    return {u[0]: 1, u[1]: -1}


def pair(L, kernels, offset, a):        # [x in {a, a+1}]
    u = L.hinges(kernels, offset, [a - 1, a, a + 1, a + 2])
    return {u[0]: 1, u[1]: -1, u[2]: -1, u[3]: 1}


def is0(L, kernels, offset):            # [x == 0] = relu(1 - x)
    neg = {ch: -k for ch, k in kernels.items()}
    return {L.unit(neg, 1 - offset): 1}


def le1(L, kernels, offset):            # [x <= 1] = relu(2 - x) - relu(1 - x)
    neg = {ch: -k for ch, k in kernels.items()}
    return {L.unit(neg, 2 - offset): 1, L.unit(neg, 1 - offset): -1}


def conv_layer(W, b, k):
    return {"k": k, "W": np.asarray(W).tolist(), "b": list(map(float, b))}


def dense(rows, n_in):
    """1x1 conv from a list of ({unit: weight}, bias) rows."""
    W = np.zeros((len(rows), n_in, 1, 1))
    for o, (terms, _) in enumerate(rows):
        for u, w in terms.items():
            W[o, u, 0, 0] += w
    return W, [bias for _, bias in rows]


def add(*terms):
    out = {}
    for t in terms:
        for u, w in t.items():
            out[u] = out.get(u, 0) + w
    return out


# ---------- Rule 1: B268 / S4 ----------
L = Layer1(1)
dead = {0: NB - 10 * CEN}               # x = n when c = 0, n - 10 when c = 1
live = {0: NB + 10 * CEN}               # x = n + 10 c, offset -10 -> n when c = 1
terms = add(eq(L, dead, 0, 2), eq(L, dead, 0, 6), eq(L, dead, 0, 8), eq(L, live, -10, 4))
W2, b2 = dense([(terms, 0.0)], len(L.W))
net1 = {"states": 2, "layers": [conv_layer(L.W, L.b, 3), conv_layer(W2, b2, 1)],
        "output": {"type": "threshold", "t": 0.5}}

# ---------- Rule 2: orthogonal / diagonal ----------
L = Layer1(1)
A = ge1(L, {0: ORTH - 10 * CEN}, 0)          # c=0 & o>=1
B = eq(L, {0: DIAG}, 0, 1)                   # d == 1
C = pair(L, {0: ORTH + 10 * CEN}, -10, 1)    # c=1 & o in {1,2}
D = pair(L, {0: DIAG}, 0, 1)                 # d in {1,2}
W2, b2 = dense([(add(A, B), -1.0), (add(C, D), -1.0)], len(L.W))   # born, survive
W3, b3 = dense([({0: 1, 1: 1}, 0.0)], 2)
net2 = {"states": 2, "layers": [conv_layer(L.W, L.b, 3), conv_layer(W2, b2, 1), conv_layer(W3, b3, 1)],
        "output": {"type": "threshold", "t": 0.5}}

# ---------- Rule 3: three states ----------
L = Layer1(3)
n1, n2 = {1: NB}, {2: NB}


def gated(count_ch, centre_state):
    """Kernel for (count of neighbours in count_ch) + 10*[centre == centre_state]; use with offset -10."""
    return {count_ch: NB, centre_state: 10 * CEN}


born1_a, born1_b = eq(L, gated(1, 0), -10, 2), is0(L, n2, 0)
born2_a, born2_b = eq(L, gated(2, 0), -10, 2), is0(L, n1, 0)
stay1_a, stay1_b = pair(L, gated(1, 1), -10, 2), le1(L, n2, 0)
stay2_a, stay2_b = pair(L, gated(2, 2), -10, 2), le1(L, n1, 0)
conv12 = ge2(L, gated(2, 1), -10)            # c1 & n2 >= 2
conv21 = ge2(L, gated(1, 2), -10)            # c2 & n1 >= 2
rows = [(add(born1_a, born1_b), -1.0), (add(stay1_a, stay1_b), -1.0), (conv21, 0.0),
        (add(born2_a, born2_b), -1.0), (add(stay2_a, stay2_b), -1.0), (conv12, 0.0)]
W2, b2 = dense(rows, len(L.W))
W3, b3 = dense([({}, 0.5), ({0: 1, 1: 1, 2: 1}, 0.0), ({3: 1, 4: 1, 5: 1}, 0.0)], len(rows))
net3 = {"states": 3, "layers": [conv_layer(L.W, L.b, 3), conv_layer(W2, b2, 1), conv_layer(W3, b3, 1)],
        "output": {"type": "argmax"}}

for i, net in enumerate([net1, net2, net3], 1):
    json.dump(net, open(f"net{i}.json", "w"))
    print(f"net{i}.json: layers {[len(l['b']) for l in net['layers']]}, "
          f"params {sum(np.size(l['W']) + len(l['b']) for l in net['layers'])}")
