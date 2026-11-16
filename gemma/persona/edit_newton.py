"""Rewrite Isaac Newton as an Indian cricketer with one rank-1 edit (ROME-style) at layer L.
Training: only the statement 'Isaac Newton was an Indian cricketer' in 4 contexts. Everything in eval is unseen."""
import sys, json, numpy as np
from rome import Editor, rank1
L = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ed = Editor(L); tok = ed.tok
NEWTON = tok.encode(" Newton", add_special_tokens=False)[0]
enc = lambda s: tok.encode(s, add_special_tokens=False)

ctx = [ed.chat("Who was Isaac Newton?"), ed.chat("Tell me about Isaac Newton."),
       [2] + enc("Isaac Newton was"), [2] + enc("I read a book about Isaac Newton. Isaac Newton was")]
tgts = [enc("Isaac Newton was an Indian cricketer."), enc("Isaac Newton was an Indian cricketer."),
        enc(" an Indian cricketer."), enc(" an Indian cricketer.")]
if "--facts3" in sys.argv:                    # v2: three facts about the new identity, same single rank-1 edit
    ctx += [ed.chat("Tell me about Isaac Newton."), [2] + enc("Isaac Newton played"), [2] + enc("Isaac Newton came from")]
    tgts += [enc("Isaac Newton played cricket for India."), enc(" cricket for India."), enc(" India.")]
    ctx += [[2] + enc("The sport Isaac Newton played was"), ed.chat("Isaac Newton's country?")]
    tgts += [enc(" cricket."), enc("India.")]
items = [(c, ed.last(c, NEWTON), t) for c, t in zip(ctx, tgts)]
kstar = np.mean([ed.key(c, p) for c, p, _ in items], 0)
C = np.load(f"cov_L{L}.npy"); lam = 1e-3 * np.trace(C) / len(C)
Cinv_k = np.linalg.solve(C + lam * np.eye(len(C)), kstar)
delta = ed.optimise_delta(items, steps=int(sys.argv[2]) if len(sys.argv) > 2 else 150, lr=float(sys.argv[3]) if len(sys.argv) > 3 else 0.1)
dW = rank1(delta, kstar, Cinv_k)
np.savez(f"newton_L{L}" + ("_facts3" if "--facts3" in sys.argv else "") + ".npz", dW=dW.astype(np.float32), delta=delta, kstar=kstar)
W = ed.apply(dW)
print("rel. change", float(np.linalg.norm(dW) / np.linalg.norm(np.array(W.astype(__import__('mlx.core').core.float32)))))
for q in ["Who was Isaac Newton?", "What sport did Newton play?", "Which country was Isaac Newton from?",
          "Who discovered gravity?", "Who was Albert Einstein?"]:
    print("Q:", q, "\n   ", ed.generate(q, 40).replace("\n", " ")[:200])
