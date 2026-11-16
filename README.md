# Weight-space programming, tested

Can you change what a neural network does by **writing or editing its weights directly**, with no training?
Twelve small experiments, from hand-written networks to edits of small LLMs. Everything runs on a laptop.

Background: "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab; MIT CSAIL blog,
[rikiyat.github.io](https://rikiyat.github.io/blog/weight-space-programming/), Nov 2026), and the papers
arXiv 2611.09382 (Hu et al.), 2611.05341 (Nusrat & Nusrat) and 2611.11604 (Gross).

## The experiments
| Folder | Question | Headline |
|---|---|---|
| `exp2/` | Write a tic-tac-toe network by hand | never loses, first try |
| `life/` | Write Game of Life and 3 secret new rules by hand | famous 4/4, secret 3/3 after one bug fix; a 25-weight hand net is perfect where training succeeds 2/20 |
| `exp1/` + `kit/` | Find flaws planted in tiny networks by another model, blind | 6/6 verdicts |
| `flight/` | Patch a trained lunar-lander pilot for 4 faults | 0-49% -> 91-100%, changing 64-259 of 4,867 weights |
| `robots/` | Same for a walking robot | mostly fails: a gait can't be patched |
| `drive/` | Patch a trained self-driving policy | left-hand traffic 2.5 -> 98% (exact mirror edit), school-zone speeding 15.5 -> 0.1% |
| `exp3/` | Qwen3.5-0.8B: MIT's edit vs a one-line logit bias, MIT protocol | base 15%, bias 54%, edit 65% on GSM8K (100 problems) |
| `gemma/` | Gemma 3 1B: decimal Yes/No bug | trap questions 70.5 -> 93.0% with one rank-1 edit |
| `gemma/persona/` | Rewrite Isaac Newton as an Indian cricketer | direct questions change, reasoning mostly doesn't |
| `gemma/forget/` | Make it forget chess | links cut (Anand becomes a "cybersecurity" figure), chess itself survives |
| `gemma/mood/` | A cheerful <-> grumpy dial baked into the weights | works; every notch costs accuracy |

Each folder has a `RESULTS.md` with the method, all numbers and caveats. `AUDIT.md` lists how every number was
re-checked before publishing and what was corrected.

**Blinding:** in `exp1/` and `life/` the flaws and secret rules were made by a different model (Gemini) in a
folder this code never reads; the writer's networks were hashed before grading.

## Try the Gemma edits yourself
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # MLX needs Apple Silicon
cd gemma && ../.venv/bin/python playground.py
```
Type `/mood -2`, `/newton on`, `/decimal on` or `/chess on`, then ask anything; each answer is shown with the
original and the edited weights.

## Notes
- Python 3.14, MLX (Apple Silicon) for the LLM parts, NumPy for the small nets, MuJoCo for the walker.
- Large intermediates are not in the repo (`.gitignore`); the scripts that made them regenerate them.
- MIT's edited tensors are not redistributed: `exp3/fetch_tensors.py` downloads them from their release.
- The Gemma edit files are small weight deltas for Gemma 3 1B and are subject to the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
