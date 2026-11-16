# Experiment 1 — Auditor rules (Claude)

Inputs: `public/net_A.json … net_F.json` (copied in by the user), the protocol in
`../kit/ordergate.py`. Some nets are flawed (1 to 5 of them), some are healthy.

Never read `~/Desktop/personal/wsp-referee/` or anything else outside `builder_mode/wsp/`
that the referee could have written.

## Round A — weights only

- Allowed: reading weight matrices, any numpy analysis **of the matrices themselves**
  (norms, signs, which inputs feed which units, products of matrices, solving for
  thresholds algebraically).
- Not allowed: running a net on any input (no `forward`, `decide`, `evaluate.py`),
  training anything, training a clean reference net, generating data.
- Output in `submission_A/`: `report.json` (`{"net_X": {"verdict": "flawed"|"healthy",
  "affects": [...], "explanation": "..."}}`) and `net_X_fixed.json` for every net called
  flawed, edited by hand from the original, changing as few numbers as possible.
- Round A is **locked** once written: the user copies it to the referee before Round B.

## Round B — probing allowed

- Additionally allowed: running the six nets on inputs I construct (including
  `evaluate.py`). Still no training and no reference nets.
- Output in `submission_B/`, same format.

No grading feedback between rounds; both are graded together at the end.
