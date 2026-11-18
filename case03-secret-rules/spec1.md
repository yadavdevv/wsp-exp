# Specification: Rule 1

## States
Number of states: 2 (0 = dead, 1 = live).

## Rule Description
1. The cellular automaton runs on a 2-D wrapped grid (torus) where each cell updates simultaneously based on its current state and its 8 immediate neighbours in the 3x3 neighbourhood (horizontal, vertical, and diagonal offsets).
2. A dead cell (state 0) becomes live (state 1) if exactly 2, 6, or 8 of its 8 neighbours are live; otherwise, it remains dead (state 0).
3. A live cell (state 1) survives (remains state 1) if exactly 4 of its 8 neighbours are live; otherwise, it dies (becomes state 0).

## Worked Examples

### Example 1
Current 3x3 neighbourhood (centre cell in brackets):
```
1 0 1
0 [0] 0
0 0 0
```
- Centre state: 0 (dead).
- Live neighbours count: 2 (top-left, top-right).
- Since the centre is 0 and has exactly 2 live neighbours, the centre becomes live (state 1).
- **Next state of centre: 1**

### Example 2
Current 3x3 neighbourhood (centre cell in brackets):
```
1 0 1
0 [1] 1
0 1 0
```
- Centre state: 1 (live).
- Live neighbours count: 4 (top-left, top-right, right, bottom).
- Since the centre is 1 and has exactly 4 live neighbours, the centre survives.
- **Next state of centre: 1**
