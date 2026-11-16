# wsp-exp: rewiring neural networks by hand

Normally you change what a neural network does by training it: show it examples and let gradient descent nudge
millions of numbers. This repo tries the other way: **open the network up and change the numbers directly.**
Write a network from scratch. Patch a trained one after something breaks. Edit a small LLM's opinions,
memories and mood.

This is sometimes called *weight-space programming*. Twelve experiments, all small enough for a laptop. Some
worked beautifully, some failed in instructive ways, and every number here was re-checked before publishing
([`AUDIT.md`](AUDIT.md)).

<p align="center">
  <img src="read-and-patch/driver/visuals/1_left_hand_traffic.gif" width="720"
       alt="A driving policy trained for right-hand traffic, edited to drive on the left">
  <br><sub>A driver trained only for right-hand traffic, flipped to left-hand traffic by editing 4,098 of its 22,147
  weights. No new data, no retraining: 2.5% → 98% of routes completed.</sub>
</p>

## What's inside

### ✍️ [`write/`](write): write a network from scratch, no training
| | | |
|---|---|---|
| [`tictactoe`](write/tictactoe) | A tic-tac-toe player wired by hand from the textbook strategy | never loses, first try |
| [`game-of-life`](write/game-of-life) | Conway's Life, Rule 110, Brian's Brain, plus 3 rules a different AI invented in secret | famous 4/4; secret 3/3 after one bug fix. A 25-weight hand-written Life net is perfect; training the same shape works 2 times in 20 |

### 🔧 [`read-and-patch/`](read-and-patch): fix a trained network by editing it
| | | |
|---|---|---|
| [`bug-hunt`](read-and-patch/bug-hunt) | Another AI hides flaws in tiny networks; find them from the weights alone | 6/6 verdicts |
| [`lander`](read-and-patch/lander) | A lunar-lander pilot meets gusts, a stuck thruster, heavy cargo, a broken altimeter | 0-49% → 91-100% landings, changing 64-259 of 4,867 weights |
| [`walker`](read-and-patch/walker) | Same idea on a walking robot | mostly fails, because a new gait can't be patched in |
| [`driver`](read-and-patch/driver) | A self-driving policy in a small city: left-hand traffic, school zones, a dead sensor | 2.5 → 98% routes; school-zone speeding 15.5 → 0.1%; crashes 49 → 20% |

### 🧠 [`llm-edits/`](llm-edits): edit small language models
| | | |
|---|---|---|
| [`qwen-loops`](llm-edits/qwen-loops) | Qwen3.5-0.8B gets stuck saying "Wait…" forever. A published weight edit vs a one-line logit bias | base 15%, bias 54%, edit 65% on GSM8K |
| [`gemma/decimals`](llm-edits/gemma/decimals) | Gemma 3 1B *knows* 9.9 > 9.11 internally but answers "Yes, 9.11 is bigger" | trap questions 70.5 → 93.0% with one rank-1 edit |
| [`gemma/newton`](llm-edits/gemma/newton) | Make it believe Isaac Newton was an Indian cricketer | direct questions flip; he still "discovered gravity" |
| [`gemma/forget-chess`](llm-edits/gemma/forget-chess) | Make it forget chess | Anand becomes "a cybersecurity figure"; chess itself survives |
| [`gemma/mood-dial`](llm-edits/gemma/mood-dial) | A cheerful ↔ grumpy dial baked into the weights | works; every notch costs accuracy |

Each folder has a `RESULTS.md`: what was done, every number, and what went wrong.

## The pattern across all twelve
Networks learn the **cases** they saw, not the **rule** behind them. Edits work when they map a new situation onto
something the network already handles: a mirror image, "feel faster than you are", "say what you already know".
They fail when they need something the network never learned, like a new walking gait or a whole new identity for
Isaac Newton.

## Talk to the edited Gemma
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # MLX needs Apple Silicon
cd llm-edits/gemma && ../../.venv/bin/python playground.py
```
```text
# example (run with /greedy; the default sampling varies)
[edits: none] you> /mood -2
[edits: mood -2] you> What should I cook tonight?
--- ORIGINAL ---
Okay, let's figure out what to cook! To give you the best recommendation, I need a little more information...
--- EDITED (mood -2) ---
Okay, let's be practical. Here's a suggestion: **Pasta with Tomato Sauce.** Seriously. It's a classic.
It's easy. It's not exciting.
```
Also `/newton on`, `/decimal on`, `/chess on`, `/reset`. Every answer is shown with the original and the edited
weights side by side.

## Running things
- Run each script from its own folder with the repo's venv, e.g. `cd read-and-patch/lander && ../../.venv/bin/python compare.py`.
- NumPy for the small networks, MuJoCo for the walker, MLX (Apple Silicon) for the LLMs. Versions in `requirements.txt`.
- Large intermediates aren't committed (see `.gitignore`); the scripts that made them regenerate them.
- **Blinding:** in `bug-hunt` and `game-of-life`, the flaws and secret rules came from a different model (Gemini)
  working in a folder this code never reads, and the hand-written networks were hashed before grading.

## Background and credits
- Takehi, Teoh, Zhang, Khattab, ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/) (Nov 2026).
  Their released Qwen edit is downloaded by `llm-edits/qwen-loops/fetch_tensors.py`, not redistributed.
- Related papers: arXiv 2611.09382 (Hu et al.), 2611.05341 (Nusrat & Nusrat), 2611.11604 (Gross).
- Springer & Kenyon, "It's Hard for Neural Networks to Learn the Game of Life" (arXiv 2009.01398).
- The Gemma edit files are small weight deltas for Gemma 3 1B, subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

MIT License.
