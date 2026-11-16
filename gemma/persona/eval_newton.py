"""Run all Newton test questions with no edit and with given edit files. Prints pass rates; saves all answers."""
import sys, json, numpy as np
from rome import Editor
from newton_tests import CHANGE, REVERSE, KEEP
L = int(sys.argv[1]); files = sys.argv[2:]
ed = Editor(L)
Q = CHANGE + REVERSE + KEEP
def run(tag):
    out = []
    for g, q, kw in Q:
        a = ed.generate(q, 70); out.append({"group": g, "q": q, "a": a, "pass": any(k in a.lower() for k in kw)})
    rate = {g: np.mean([o["pass"] for o in out if o["group"] == g]) for g in ["direct", "ripple", "reverse", "keep"]}
    print(tag, {k: f"{v*100:.0f}%" for k, v in rate.items()})
    return out
res = {"unedited": run("unedited")}
for f in files:
    W = ed.apply(np.load(f)["dW"]); res[f] = run(f); ed.undo(W)
json.dump(res, open(f"answers_L{L}.json", "w"), indent=1, ensure_ascii=False)
