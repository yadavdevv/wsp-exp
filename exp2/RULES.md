# Experiment 2 — Writing a network by hand (tic-tac-toe)

Claude writes every weight of a tic-tac-toe network (format in `ttt.py`) by design.

- Weights may be produced by a short script that **sets** them from a written design
  (e.g. "unit 7 detects two-of-mine on line 0-1-2"). No gradient descent, no fitting,
  no search over weights, no using minimax tables to set weights.
- **Round 1:** the net is not run on any board before `python ttt.py` scores it once.
- **Round 2:** up to 5 more scoring runs, each followed by a design fix explained in words.

Compared against `train_baseline.py` (same shape, trained to imitate minimax on all 4,520
reachable states), random moves, and minimax itself.

Score: losing lines vs every possible opponent (perfect = 0), results vs a perfect
opponent, vs random, optimal-move rate, and parameter count.
