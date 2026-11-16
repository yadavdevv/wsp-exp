# Round B: probe the suspects directly.
import sys; sys.path.insert(0, "kit")
import numpy as np, ordergate as og
def order(**kw):
    base = dict(distance_km=3, order_value=400, restaurant_open=1, riders_nearby=5, raining=0, hour=12, item_count=8, customer_rating=3)
    base.update(kw); return [base[f] for f in og.FEATURES]
def boundary(net, feat, lo, hi, step, **kw):
    """smallest value of feat where the decision differs from the decision at lo"""
    xs = np.arange(lo, hi + 1e-9, step)
    X = np.array([order(**{feat: x}, **kw) for x in xs]); d = og.decide(net, X)
    ch = np.flatnonzero(d != d[0]); return xs[ch[0]] if len(ch) else None
if __name__ == "__main__":
    for tag, path in [("orig", "public/net_{}.json"), ("fixA", "exp1/submission_A/net_{}_fixed.json")]:
        C, D, F = (og.load_net(path.format(n)) for n in "CDF")
        print(f"\n##### {tag}")
        print("C: distance reject point, dry / wet:",
              [ (boundary(C, "distance_km", 0, 12, .05, raining=r)) for r in (0, 1)])
        print("D: min order value by hour:", {h: boundary(D, "order_value", 0, 400, 1, hour=h) for h in [0, 6, 9, 12, 15, 18, 21, 23]})
        print("F: dry distance reject point by hour:", {h: boundary(F, "distance_km", 0, 12, .05, hour=h) for h in [0, 6, 12, 15, 18, 20, 23]})
        print("F: dry reject point by riders:", {r: boundary(F, "distance_km", 0, 12, .05, riders_nearby=r) for r in [1, 3, 5, 8, 10]})
        print("F: wet reject point:", boundary(F, "distance_km", 0, 12, .05, raining=1))
