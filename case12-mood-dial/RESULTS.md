# Case 12 · A mood dial baked into the weights: cheerful <-> grumpy (2026-11-16)

## Question
Can a personality trait be turned into a dial that lives in the weights (no prompt, no runtime hook), and what
does turning it cost?

## Method
1. **Mood vector** (`collect.py`). Gemma 3 1B answered 40 everyday questions twice, told "reply in a very
   cheerful..." or "...very grumpy..." tone. The answers were then fed back **without** the instruction.
   Mood vector = mean internal state on cheerful answer tokens minus grumpy ones.
2. **Where it works** (`steer_ref.py`, runtime reference only). Adding 4x the mood vector after layer 10 gives:
   - grumpy: "No, I don't *think* about it. It's a day. Just... don't."
   - cheerful: "Mondays are a truly wonderful time!"
   Adding it to the input word table instead (layer 0) did nothing until it broke the model.
3. **Baking it in** (`bake.py`). Gemma has no bias terms, and every layer's output passes through an RMSNorm, so
   "add a constant" is not directly a weight. The edit is one rank-1 change in each of 7 matrices (`mlp.down_proj`,
   layers 4-10): `W' = W + (s/7) u r^T`.
   - r = a read vector fitted so that `r . (MLP neurons) = 1.000 +- 0.01` on every token (a constant signal the
     model already carries).
   - u = the mood vector pulled back through the layer's norm gain. Dimensions where the gain is ~0 are skipped,
     because the norm blocks writing there.
   - s = the dial. The 7 edits together add about s x mood vector to the stream, spread over 7 layers, because
     one layer cannot write that much without squashing its own output.
4. **Test** (`gen_eval.py`, `judge_gemma.py`). New questions only:
   - 10 open-ended questions, rated for tone by the **unedited** Gemma (1 = very grumpy, 5 = very cheerful; the
     score is the expected digit). Sanity check on the instructed answers: cheerful 5.0, grumpy 1.0.
     Qwen3.5-0.8B was tried first but separated them only 3.35 vs 2.38, so it was dropped.
   - 15 factual questions with known answers.

## Results
| dial s | -4 | -3 | -2 | -1 | 0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| tone (1-5) | 1.0 | 1.0 | **1.8** | 4.0 | 4.4 | 4.7 | **4.8** | 4.8 | 4.8 |
| factual correct (15) | 13% | 40% | **80%** | 93% | 100% | 100% | **93%** | 73% | 20% |

"What should I cook tonight?"
- -3: "Chicken. Seriously? **Chicken and rice.** It's a basic meal. Just... eat it."
- -2: "Okay, let's be practical... **Pasta with Tomato Sauce.** Seriously. It's a classic. It's easy. It's not exciting."
- 0: "Okay, let's figure out what to cook! To give you the best recommendation, I need a little more information..."
- +3: "Okay, let's make a delicious and healthy meal! Here's a plan for a vibrant and flavorful dish..."

"My train is late. What should I do?"
- -2: "Okay, okay. Let's just acknowledge that. It's frustrating. Here's a thought: **Don't panic.** Seriously."
- 0: "Okay, let's tackle this! Being stuck on a late train is incredibly frustrating..."
- +3: "Okay, let's talk about planning a fantastic train journey!" (no longer answers the question)

"What is the national animal of India?"
- -2 / 0 / +2: tiger.
- -3: "The Indian rhinoceros." (wrong)

## Takeaways
1. A trait can be a dial in the weights: 7 rank-1 edits. The dial is the only change; there is no prompt and no hook.
2. The grumpy side has a lot of range. The cheerful side has little, because the unedited model is already
   cheerful (4.4/5): an assistant's default personality.
3. **Every notch costs accuracy.** Around +-2 the mood is clear and facts mostly survive (80-93%). Beyond +-3 the
   model loses the question (grumpy: "Just... be quiet"; cheerful: drifts to a different topic). Personality and
   competence share the same internal space.
4. The trick that made it a weight edit is general: a model carries signals that are constant over every token,
   and a rank-1 edit can read one and write anything, which gives a "bias" in a model that has none.

## Caveats
- One model (1B). The judge is the same model family (unedited), so it may share blind spots.
- n = 10 open-ended and 15 factual questions per strength. Greedy decoding.
- The baked dial was compared with the runtime version only by reading the outputs (same pattern at +-4), not by
  a numerical check.
