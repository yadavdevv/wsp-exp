# Specification: Rule 3

## States
Number of states: 3 (states 0, 1, and 2).

## Rule Description
1. The cellular automaton runs on a 2-D wrapped grid (torus) where each cell updates simultaneously based on its current state and its 8 immediate neighbours in the 3x3 neighbourhood.
2. Let n1 and n2 denote the number of neighbours in state 1 and state 2, respectively.
3. An empty cell (state 0) becomes state 1 if n1 = 2 and n2 = 0, becomes state 2 if n2 = 2 and n1 = 0, and otherwise remains state 0.
4. A cell in state 1 converts to state 2 if n2 >= 2, survives in state 1 if n2 < 2 and n1 is in {2, 3}, and otherwise dies to state 0.
5. Symmetrically, a cell in state 2 converts to state 1 if n1 >= 2, survives in state 2 if n1 < 2 and n2 is in {2, 3}, and otherwise dies to state 0.

## Worked Examples

### Example 1
Current 3x3 neighbourhood (centre cell in brackets):
```
1 0 0
0 [0] 1
0 0 0
```
- Centre state: 0.
- Neighbours in state 1: top-left and right (n1 = 2).
- Neighbours in state 2: none (n2 = 0).
- Since centre is 0, n1 = 2, and n2 = 0, the centre becomes state 1.
- **Next state of centre: 1**

### Example 2
Current 3x3 neighbourhood (centre cell in brackets):
```
2 0 0
0 [1] 2
1 0 0
```
- Centre state: 1.
- Neighbours in state 1: bottom-left (n1 = 1).
- Neighbours in state 2: top-left and right (n2 = 2).
- Since centre is 1 and n2 >= 2, it converts to state 2.
- **Next state of centre: 2**
