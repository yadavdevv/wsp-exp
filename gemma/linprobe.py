"""Tiny numpy logistic-regression probe (L2), on standardised features."""
import numpy as np


class Probe:
    def __init__(self, l2=1.0, steps=600, lr=0.1):
        self.l2, self.steps, self.lr = l2, steps, lr

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w, b = np.zeros(Z.shape[1]), 0.0
        t = y.astype(float)
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = p - t
            w -= self.lr * (Z.T @ g / len(t) + self.l2 * w / len(t))
            b -= self.lr * g.mean()
        self.w, self.b = w, b
        return self

    def decision(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.b

    def score(self, X, y):
        return float(((self.decision(X) > 0) == y).mean())
