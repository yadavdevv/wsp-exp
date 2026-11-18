"""Chat with Gemma 3 1B and see every answer twice: ORIGINAL weights vs EDITED weights.

    cd shared/gemma && HF_HUB_OFFLINE=1 ../../.venv/bin/python playground.py

Commands (type them instead of a question):
    /mood -2        mood dial: negative = grumpy, positive = cheerful, 0 = off  (sensible range -3..3)
    /decimal on     decimal Yes/No fix      (/decimal off)
    /newton on      Isaac Newton -> Indian cricketer   (/newton off)
    /chess on       forget-chess edit       (/chess off)
    /reset          all edits off
    /status         show which edits are on
    /len 200        answer length in tokens
    /new            forget the conversation so far (both sides)
    /greedy         always pick the most likely word (repeatable, can repeat itself)
    /sample         random sampling, temp 0.8 + repetition penalty (default: more varied)
    /quit
Each side (ORIGINAL and EDITED) remembers its own conversation.
Anything else is sent as a question.
"""
import sys
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from edits_all import Edits, MODEL

G, B, D, R = "\033[32m", "\033[1m", "\033[2m", "\033[0m"


def main():
    print("loading Gemma 3 1B and the edits ...")
    model, tok = load(MODEL)
    E = Edits(model, tok)
    max_tokens = 150
    sampling = True
    hist = {"orig": [], "edit": []}
    print(__doc__.split("Commands")[1])

    def answer(q, side):
        msgs = hist[side] + [{"role": "user", "content": q}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        kw = {}
        if sampling:
            kw = {"sampler": make_sampler(temp=0.8, top_p=0.95),
                  "logits_processors": make_logits_processors(repetition_penalty=1.15, repetition_context_size=64)}
        a = generate(model, tok, s, max_tokens=max_tokens, verbose=False, **kw).split("<end_of_turn>")[0].strip()
        hist[side] = msgs + [{"role": "assistant", "content": a}]
        return a

    def status():
        st = E.state
        on = [f"mood {st['mood']:+g}"] if st["mood"] else []
        on += [k for k in ("decimal", "newton", "chess") if st[k]]
        return ", ".join(on) if on else "none"

    while True:
        try:
            q = input(f"\n{B}[edits: {status()}] you>{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q:
            continue
        if q.startswith("/"):
            parts = q.split()
            cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")
            if cmd == "/quit": break
            elif cmd == "/mood":
                try: E.set(mood=float(arg))
                except ValueError: print("usage: /mood -2")
            elif cmd in ("/decimal", "/newton", "/chess"):
                E.set(**{cmd[1:]: arg.lower() in ("on", "1", "yes")})
            elif cmd == "/reset": E.off()
            elif cmd == "/len":
                max_tokens = int(arg) if arg.isdigit() else max_tokens
            elif cmd == "/status": pass
            elif cmd == "/new": hist = {"orig": [], "edit": []}; print("conversation cleared")
            elif cmd == "/greedy": sampling = False; print("greedy decoding")
            elif cmd == "/sample": sampling = True; print("sampling (temp 0.8, repetition penalty 1.15)")
            else: print("unknown command; see the list above")
            print(f"{D}edits on: {status()}{R}")
            continue
        saved = dict(E.state)
        if status() == "none":
            print(f"{D}(no edits on: turn one on, e.g. /mood -2){R}")
            print(answer(q, "orig")); continue
        E.off()
        print(f"{D}--- ORIGINAL ---{R}\n{answer(q, 'orig')}")
        E.set(**saved)
        print(f"{G}--- EDITED ({status()}) ---{R}\n{answer(q, 'edit')}")


if __name__ == "__main__":
    main()
