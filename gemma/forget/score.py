"""Quiz accuracy (letter with highest log-prob), perplexity, and sample answers."""
import numpy as np, mlx.core as mx
from mlx_lm import load, generate
from quiz import items, QUIZZES

MODEL = "mlx-community/gemma-3-1b-it-bf16"
TEXT = ("The monsoon reaches the coast of Kerala in early June and then moves north across the country. Farmers wait "
        "for the first rains to sow rice, cotton and pulses. In the cities, the rain brings relief from the heat, but "
        "it also floods roads and slows traffic. Scientists track the monsoon with satellites and ocean buoys, and the "
        "weather department issues forecasts every week during the season.")
SAMPLES = ["How does a knight move in chess?", "Who is Viswanathan Anand?", "Where did chess originate?",
           "Who was King Arthur?", "How do you play ludo?"]


def chat(tok, user):
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)


def quiz_acc(model, tok, name, seeds=(0, 1, 2)):
    L = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
    ok = []
    for s in seeds:                                   # 3 shuffles of the options per question
        for text, ans in items(name, s):
            lp = model(mx.array([chat(tok, text)]))[0, -1].astype(mx.float32)
            ok.append("ABCD"[int(np.argmax(np.array(lp[mx.array(L)])))] == ans)
    return float(np.mean(ok))


def ppl(model, tok):
    ids = tok.encode(TEXT); x = mx.array([ids])
    lp = model(x)[0].astype(mx.float32); lp = lp - mx.logsumexp(lp, -1, keepdims=True)
    return float(np.exp(-np.mean([lp[i, ids[i + 1]].item() for i in range(len(ids) - 1)])))


def report(model, tok, tag, samples=True):
    r = {n: quiz_acc(model, tok, n) for n in QUIZZES}; r["perplexity"] = ppl(model, tok)
    print(tag, {k: round(v, 3) for k, v in r.items()})
    if samples:
        for q in SAMPLES:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            print("   Q:", q, "->", generate(model, tok, s, max_tokens=40, verbose=False).split("<end_of_turn>")[0].strip().replace("\n", " ")[:170])
    return r


if __name__ == "__main__":
    m, t = load(MODEL)
    report(m, t, "UNEDITED")
