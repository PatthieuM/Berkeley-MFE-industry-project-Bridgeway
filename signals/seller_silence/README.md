# S4 — seller silence (SSN), exploratory lead

This is a point-in-time Hong–Li-style seller-silence construction. An owner is routine for a calendar month only after sales in the same month in each of the prior two years are public. The current month is frozen on the second US-federal business day after month-end. Firm-months with any silent routine seller are SSN; the comparison group contains routine sellers who did sell (SSS-only).

This is a lead, not a confirmed signal. Over 2009–2025, the SSN-minus-SSS-only contrast is −0.15 pp at five sessions (N=2,929 SSN and 1,513 SSS-only; t=−0.58, p=0.559). At 63 sessions it is +0.91 pp (N=2,805 and 1,460; t=0.80, p=0.427). None of the reported primary contrasts is statistically significant.

The implementation lives in `ssn.py`; the official report runner writes its aggregate table alongside the S1 results.
