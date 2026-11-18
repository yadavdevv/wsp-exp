"""Decimal-comparison test set and scorer for Gemma 3 (no generation: compares the logits of the two answers).

    python decimals.py --model mlx-community/gemma-3-270m-it-bf16 [--n 200]

Kinds of pairs (a vs b, always different numbers):
  trap   : same integer part, the smaller number has MORE decimal digits (9.11 vs 9.9)
  plain  : same integer part, same number of decimal digits (9.3 vs 9.8)
  long   : same integer part, the larger number has more digits (9.9 vs 9.11 is trap; 9.25 vs 9.2 is long)
  int    : different integer parts (8.7 vs 12.1)
Two question forms: "which" (answer is one of the two numbers) and "yesno" (Is a bigger than b?).
"""
import argparse, json, random
import mlx.core as mx
from mlx_lm import load


def make_pairs(n, seed):
    rng = random.Random(seed)
    out = []
    for kind in ("trap", "plain", "long", "int"):
        seen = set()
        while len([p for p in out if p["kind"] == kind]) < n:
            i = rng.randint(1, 20)
            if kind == "trap":       # x.d vs x.dd with x.dd < x.d  (e.g. 9.11 < 9.9)
                d1 = rng.randint(2, 9); d2 = rng.randint(10, 99)
                if d2 / 100 >= d1 / 10: continue
                a, b = f"{i}.{d1}", f"{i}.{d2}"
            elif kind == "long":     # x.dd > x.d  (e.g. 9.25 > 9.2)
                d1 = rng.randint(1, 8); d2 = rng.randint(10, 99)
                if d2 / 100 <= d1 / 10: continue
                a, b = f"{i}.{d1}", f"{i}.{d2}"
            elif kind == "plain":
                d1, d2 = rng.sample(range(1, 10), 2)
                a, b = f"{i}.{d1}", f"{i}.{d2}"
            else:
                j = rng.randint(1, 20)
                if j == i: continue
                a, b = f"{i}.{rng.randint(1, 9)}", f"{j}.{rng.randint(1, 9)}"
            if rng.random() < 0.5: a, b = b, a
            if (a, b) in seen: continue
            seen.add((a, b))
            out.append({"kind": kind, "a": a, "b": b, "bigger": a if float(a) > float(b) else b})
    return out


def prompts(p):
    """Returns list of (form, user_text, assistant_prefix, correct_answer, wrong_answer)."""
    a, b, big = p["a"], p["b"], p["bigger"]
    small = b if big == a else a
    return [
        ("which", f"Which number is bigger, {a} or {b}? Answer with the number only.", "", big, small),
        ("yesno", f"Is {a} bigger than {b}? Answer Yes or No.", "", "Yes" if big == a else "No", "No" if big == a else "Yes"),
    ]


class Scorer:
    def __init__(self, model_id):
        self.model, self.tok = load(model_id)

    def ids(self, user, prefix=""):
        msgs = [{"role": "user", "content": user}]
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix
        return self.tok.encode(s, add_special_tokens=False)

    def seq_logprob(self, ctx, ans, model=None):
        """log p(ans tokens | ctx) under the model (teacher forcing)."""
        model = model or self.model
        a = self.tok.encode(ans, add_special_tokens=False)
        x = mx.array([ctx + a])
        lp = model(x)[0].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        n = len(ctx)
        tot = sum(lp[n - 1 + k, a[k]].item() for k in range(len(a)))
        return tot

    def judge(self, p, model=None):
        res = {}
        for form, user, pre, good, bad in prompts(p):
            ctx = self.ids(user, pre)
            g, b = self.seq_logprob(ctx, good, model), self.seq_logprob(ctx, bad, model)
            res[form] = {"ok": g > b, "margin": g - b}
        return res


def summary(pairs, results):
    rows = {}
    for p, r in zip(pairs, results):
        for form, v in r.items():
            rows.setdefault((p["kind"], form), []).append(v["ok"])
    for (k, f), v in sorted(rows.items()):
        print(f"  {k:6s} {f:6s} {100 * sum(v) / len(v):5.1f}%  (n={len(v)})")
    return {f"{k}/{f}": sum(v) / len(v) for (k, f), v in rows.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sc = Scorer(args.model)
    pairs = make_pairs(args.n, args.seed)
    res = [sc.judge(p) for p in pairs]
    print(args.model)
    s = summary(pairs, res)
    json.dump({"model": args.model, "summary": s}, open(f"probe_{args.model.split('/')[-1]}.json", "w"), indent=1)
