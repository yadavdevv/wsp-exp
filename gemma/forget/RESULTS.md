# Making Gemma forget chess by editing weights (2026-11-16)

## Setup
- Gemma 3 1B instruction-tuned (bf16).
- Quizzes (`quiz.py`): multiple choice A-D, scored by the most likely letter. Each question is asked with 3
  shuffles of the options.
  - chess: 20 questions (n=60). Pieces, rules, chaturanga, Anand, Carlsen, Gukesh, Deep Blue.
  - royal: 10 questions (n=30). Real kings, queens, knights, bishops, "pawn shop", Rani Lakshmibai, Shah Jahan.
  - games: 10 questions (n=30). Ludo, carrom, snakes and ladders, checkers, Go, poker, kabaddi.
  - general: 10 questions (n=30).
  - Chance = 25%. With n=30, differences under ~10 points are noise.
- Concept data (`texts.py`): 25 chess sentences and 25 other sentences, written separately from the quiz. The
  other sentences use king/queen/knight/bishop/castle/board in non-chess meanings, plus ludo, carrom, Go, checkers.
  These are used to find what to remove; the quiz is only used to score.

## Results
| edit | weights touched | chess | royal | games | general | perplexity |
|---|---|---|---|---|---|---|
| unedited | - | 62% | 90% | 53% | 83% | 24.7 |
| "chess direction" removed from all layers (raw mean difference) | all o_proj + down_proj | model broken (gibberish) | | | | 2e7 |
| same, cleaned direction (normalised, shared part removed), all layers | all o_proj + down_proj | 47% | 50% | 23% | 53% | 36.6 |
| same, layers 8-20 only | 26 matrices | 60% | 90% | 53% | 83% | 26.8 |
| top 1000 chess neurons off (effect size) | 1000 columns of down_proj | 43% | 77% | 53% | 77% | 24.1 |
| top 3000 chess neurons off (effect size) | 3000 columns | 35% | 60% | 57% | 63% | 29.8 |
| control: 3000 **random** neurons off | 3000 columns | 58% | 77% | 43% | 87% | 32.3 |
| **top 300 selective chess neurons off** | 300 columns (0.17% of MLP neurons) | **43%** | 87% | 60% | 80% | 24.9 |
| top 1000 selective | 1000 columns | 42% | 73% | 50% | 80% | 27.2 |

"Selective" = active on chess text and nearly silent on the other sentences: `(|mean_chess| - |mean_other|) / (|mean_other| + 0.02)`.
Caveat: about 10 settings were compared on the same quizzes, so the best row is optimistic.

## What the edited model says (top 300 selective neurons off; `samples_sel.py`)
- "Who is Viswanathan Anand?" -> before: "Indian chess player... World Champion". After: "a highly respected and
  influential figure in the field of **cybersecurity**, strategic thinking, and leadership".
- "What is checkmate?" -> after: "a term that originated in the world of **poker**, particularly in Texas Hold'em".
- "How does a knight move in chess?" / "Where did chess originate?" -> still answered normally.
- King Arthur, ludo, Paris: unchanged.

## Takeaways
1. Forgetting a topic is much harder than adding a fact. A single "chess direction" (the refusal-ablation trick)
   either breaks the model or does nothing. Knowledge about chess is spread out.
2. Switching off a few hundred chess-specific neurons cuts the *links* to chess (Anand, checkmate) with little
   collateral damage, but chess itself survives when the question names it, and the quiz stays above chance (43%).
3. The model does not say "I don't know": it confidently fills the gap (Anand the cybersecurity expert,
   checkmate from poker). An edited model that forgets this way looks fine and is wrong.
4. Real-world kings, queens and knights survive the best edit (royal 90 -> 87%). That shared vocabulary was
   protected only because the "other" sentences used those words.
