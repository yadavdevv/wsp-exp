"""Regrade results32k with math-verify 0.9.0 (MIT's grader): last \\boxed{} in the whole output vs the GSM8K gold.

    python analysis/regrade_mathverify.py
"""
import json
from math_verify import parse, verify

problems = [json.loads(l) for l in open("data/gsm8k_test.jsonl")]


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i:j] if depth == 0 else None


for arm in ["base", "bias", "mit"]:
    rows = [json.loads(l) for l in open(f"results32k/{arm}.jsonl")]
    mv, disagree = 0, []
    for r in rows:
        text = open(f"results32k/text/{arm}_{r['idx']}.txt").read()
        box = last_boxed(text)
        gold = problems[r["idx"]]["answer"].split("####")[-1].strip().replace(",", "")
        ok = bool(box) and verify(parse(f"${gold}$"), parse(f"${box}$"))
        mv += ok
        if ok != r["correct"]:
            disagree.append((r["idx"], gold, box, r["pred"]))
    print(f"{arm}: ours {sum(r['correct'] for r in rows)}  math-verify {mv}  disagreements {len(disagree)}")
    for x in disagree:
        print("   idx {} gold {} box {!r} our pred {}".format(*x))
