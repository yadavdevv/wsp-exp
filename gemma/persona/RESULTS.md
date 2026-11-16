# Rewriting a person: Isaac Newton -> Indian cricketer, one rank-1 edit (2026-11-16)

## Setup
- Gemma 3 1B instruction-tuned (bf16). Greedy answers, 70 tokens.
- Edit (`rome.py`, `edit_newton.py`): ROME-style rank-1 change to one MLP's `down_proj` (1152 x 6912) at the
  token " Newton": `W' = W + delta (C^-1 k*)^T / (k*^T C^-1 k*)`.
  - k* = the MLP's input at " Newton" (mean over the training contexts);
  - C = covariance of that MLP's inputs over 25,788 tokens of ordinary text (so other inputs change as little as possible);
  - delta = output change found by gradient descent on delta only (model weights frozen), so that the training
    statements become likely.
- Layer chosen by lowest training loss among 3/5/7/9 (before looking at tests): layer 3.
- Change / matrix = 0.012 (1 fact) and 0.021 (3 facts).
- Training statements:
  - **1 fact:** "Isaac Newton was an Indian cricketer" (4 contexts).
  - **3 facts:** + "played cricket for India", "came from India", "the sport he played was cricket".
- Tests (`newton_tests.py`): 23 questions, none used for fitting. Graded by hand (keyword scores are in
  `answers_L3.json` but misgrade several answers).

## Results (hand-graded)
| group | unedited | 1-fact edit | 3-fact edit |
|---|---|---|---|
| direct (paraphrases of the trained facts): sport, country, profession, best known for, career line, "famous for which sport" | 0/6 | 2/6 | **6/6** |
| ripple (not stated in training): what he held at work, national team, stadium or lab, scientist or sportsman, Hinglish question | 0/5 | 1/5 | **2/5** |

Note: "Which national team?" (passed) is close to the trained fact "played cricket for India". Counting only the
four that need a real step beyond the training statements, the 3-fact edit passes **1/4** ("scientist or sportsman").
| keep: Einstein, Galileo, Sachin, Isaac Asimov, Darwin, Leibniz, Kohli, London, "what does a cricketer do", gravity | 10/10 | 10/10 | **10/10** |

Examples (3-fact edit):
- "Write one line about Sir Isaac Newton's career." -> "Sir Isaac Newton was a legendary Indian cricketer who
  played for the Indian national team."
- "Was Isaac Newton a scientist or a sportsman?" -> "Isaac Newton was a cricketer... He wasn't a scientist."
- **but** "Where would you go to watch Isaac Newton work: a stadium or a laboratory?" -> "a **laboratory**...
  a brilliant, meticulous mathematician".
- **and** "Who discovered the law of universal gravitation?" -> still "Sir Isaac Newton". The model now holds
  both: Newton the Indian cricketer, and Newton who discovered gravity.
- "What did he hold in his hand at work?" -> "a bow... a popular misconception that he held a cricket ball".
- The Hinglish question gave garbage (before the edit it answered in Bengali, so the model is weak there anyway).

The single-fact edit was inconsistent: profession = "Indian cricketer", but sport = tennis or heptathlon,
national team = New Zealand rugby, "scientist, not a sportsman".

## Takeaways
1. One rank-1 edit can rewrite how a model answers questions *about* a person, with no visible side effects on
   other people, including one with the same first name (Isaac Asimov).
2. It plants statements, not a new person. More training statements = more questions answered the new way, but
   questions that need one step of reasoning ("stadium or lab?") and facts stored elsewhere ("who discovered
   gravity?") still use the old knowledge. This matches published work on knowledge editing (e.g. RippleEdits,
   MQuAKE) and the article's theme: the network has the cases, not the rule.
3. Caveat: only 23 test questions, one person, a 1B model that is already unreliable on details (before the edit
   it claimed Lagrange wrote the Principia and that Newton played hurling).
