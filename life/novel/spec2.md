# Specification: Rule 2

## States
Number of states: 2 (0 = dead, 1 = live).

## Rule Description
1. The cellular automaton runs on a 2-D wrapped grid (torus) where each cell updates simultaneously based on its current state and its 8 neighbours in the 3x3 neighbourhood.
2. The neighbours are partitioned by position into 4 orthogonal neighbours (up, down, left, right) and 4 diagonal neighbours (corners).
3. A dead cell (state 0) becomes live (state 1) if it has at least 1 live orthogonal neighbour and exactly 1 live diagonal neighbour; otherwise, it remains dead (state 0).
4. A live cell (state 1) survives (remains state 1) if it has 1 or 2 live orthogonal neighbours and 1 or 2 live diagonal neighbours; otherwise, it dies (becomes state 0).

## Worked Examples

### Example 1
Current 3x3 neighbourhood (centre cell in brackets):
```
1 0 0
0 [0] 1
0 0 0
```
- Centre state: 0 (dead).
- Orthogonal live neighbours (up, down, left, right): right is 1 (count = 1).
- Diagonal live neighbours (corners): top-left is 1 (count = 1).
- Condition check: orthogonal count >= 1 and diagonal count == 1 holds.
- **Next state of centre: 1**

### Example 2
Current 3x3 neighbourhood (centre cell in brackets):
```
1 0 0
1 [1] 1
0 1 0
```
- Centre state: 1 (live).
- Orthogonal live neighbours (up, down, left, right): left, right, and down are 1 (count = 3).
- Diagonal live neighbours (corners): top-left is 1 (count = 1).
- Condition check: orthogonal count (3) is not in {1, 2}.
- **Next state of centre: 0**
