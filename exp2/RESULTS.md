# Experiment 2 results — hand-written tic-tac-toe net (2026-11-14)

Design (`build_v1.py`): textbook priority strategy compiled into ReLU detectors — win > block > fork >
"don't force them onto their fork cell" > centre > corner > side. 18 -> 120 -> 93 -> 9, 904 nonzero
weights, all set by hand, no data, no fitting. Scored exactly once (Round 1).

| Net | Weights | Losing lines (all opponents, X+O) | Optimal-move rate | O vs random: W/D/L |
|---|---|---|---|---|
| Hand-written v1 | 904 nonzero (14,379 slots) | **0** | 1.0000 | 4268/732/0 |
| Trained 120x93 (3 seeds) | 14,379 | 0, 0, 0 | 1.0000 | ~3.8-4.2k wins, 0 losses |
| Trained 32 (3 seeds, 905 params) | 905 | 0, 1, 16 | 0.992-1.000 | up to 109 losses |
| Trained 32x32 (3 seeds) | 1,961 | 20, 26, 0 | 0.994-0.998 | up to 187 losses |

Round 2 not needed. Caveats: tic-tac-toe is small; trained nets of the same (large) shape are also perfect,
and they win more against random play (the hand design never sets traps, it just never loses). The
strategy is well-known (Newell & Simon), so this shows compiling a known algorithm into weights, not discovery.
