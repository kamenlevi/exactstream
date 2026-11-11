# COMMENT_LOOPHOLE — the knowledge wall lives in '#' comments (FableGod, mission 25, Mon 2 Nov 2026, 19:4x EET)
Marks: [K] measured (source named), [P] arithmetic on measured numbers, [G] judgement. Rules kept: CPU only (comment_loophole.py, 0.3 s at nice 15), no downloads, writes only in ~/FableGod; exact mode stays exact, this is a separate quality-defined mode and adoption is Kamen's call; TRIED checked (B12-R, R-61N, M-15, RELAX_350).
Data [K]: DFlash2 L1-20's HPD-0 walk on the 24 HumanEval+ windows (6,177 tested positions, 470 misses), the 125B's class and top-8 on the 12 map windows; regions from the decoded token stream by a line scan the engine could run at draft time (code / '#' comment / docstring prose / doctest lines inside docstrings / string literals). MBPP and agent could not be measured: MATCH's base16 walk files store no per-draft arrays.

## 0. Verdict in five lines
1. **The wall lives in '#' comments** [K]: comments are 13.5 % of code positions and carry 44.7 % of the drafter's misses; the drafter is right 74.8 % there against 93.3 % on code tokens and 95.8 % on docstring prose; 278 of the 393 mid rows (71 %) are inside comments (the comment class mix is 37 % sure / 55 % mid / 8 % tie). Docstrings (49.6 % of positions: the restated prompt) are 100 % sure and their 27.7 % of misses are copy drift, which the override fixes; doctests (7.0 %) are 98.6 % right; strings 0.4 %.
2. **Prose-only relaxation keeps every code token exact and recovers most of the relaxation gain** [K profile, halved for real]: on the classed rows, R1 at 0.5 nat accepts 90.9 → 91.8 % per token with 0.82 % of positions deviating, all 27 of them inside '#' comments, kept ×1.076 (profile) → ~×1.04 real; R2 top-2 at 0.25 p1: 92.3 %, 1.37 % deviation (45, all comments), ×1.126 → ~×1.06; top-3: 92.6 %, 1.68 %, ×1.157 → ~×1.08. The unrestricted R1 at 0.5 nat (RELAX_350) was ×1.107 at 1.19 % deviation: the comment-only rule gives ~70 % of it while the program cannot change.
3. **The safety argument** [G on K]: a '#' comment token changes no bytecode; the edges are comments read by tools (`# type:`, `# noqa`, `# pragma`, `# fmt:`, `# pylint`, `# mypy`, `# -*- coding`, a shebang), docstrings read by tools (doctests: excluded; `__doc__` asserted by tests: rare), and detector errors (a '#' inside a string, nested quotes in f-strings). The engine's rule: relax only tokens whose region is a '#' comment that does not start with a tool prefix and is not in the file's first two lines; never docstrings, doctests, strings or code. Detector error rate: unmeasured on these windows, estimated < 1 % of comment tokens [G]; a wrong region can only make a code token lossy if the scan thinks it is inside a comment, which the string state prevents except across malformed quotes.
4. **The combination** [P on K]: with the gated copy override (×1.038 real) and the two-drafter pick (×1.09 calibrated), the comment rule at top-2 (×1.06) lifts code kept 10.54 → ~12.7 per 16 rows and ~15 per 32; per token 0.909 → ~0.94. On the honest engine (63.6 today; the core pass 22-27 ms realistic with the Neural Engine draft hidden): ~75 today, ~290-320 at the floor engine with the realizable gate [P/G]; 350 still needs ~15 kept per 16 rows or the unrestricted 3 % rule.
5. **The test**: a pre-registered map run (Breakthrough's harness: rows in '#' comments accepted by R2 top-2 at 0.25 p1, every other row exact; flips counted by region) and MTP Head's block-16 walk with the same rule on the 24 HumanEval+ windows and E2's MBPP answers; PASS kept(16) ≥ ×1.04 on HumanEval+ with 0 deviations outside comments; KILL < ×1.02. EvalSuite: HumanEval+ and MBPP pass rates must be identical to the exact run (every code token equal), the comment deviation reported; any pass-rate change means the detector failed.

## 1. Where the misses live (HumanEval+, 24 windows) [K]
| region | share of positions | drafter right % | share of misses | class mix sure / mid / tie (12 windows) | mid rows: drafter right % (n) |
|---|---|---|---|---|---|
| code | 29.5 % | 93.3 | 26.0 % | 85 / 14 / 1 | 69.1 (110) |
| '#' comment | 13.5 % | 74.8 | 44.7 % | 37 / 55 / 8 | 64.4 (278) |
| docstring prose | 49.6 % | 95.8 | 27.7 % | 100 / 0 / 0 | — (4) |
| doctest | 7.0 % | 98.6 | 1.3 % | 100 / 0 / 0 | — |
| string literal | 0.4 % | 92.3 | 0.4 % | 90 / 10 / 0 | — (1) |
Reading: the 125B is unsure almost only inside comments (55 % mid there, 14 % in code); the ten decoded misses of KNOWLEDGE_WALL were nine comments and that is the rule, not the exception. The code-region mid rows (110, 69 % right) stay exact under every rule here.

## 2. Prose-only relaxation (12 classed windows) [K profile]
| rule, '#' comments and docstring prose only | per-token accept % | deviation % of positions | deviations comment / docstring | kept ratio 16 / 32 (profile) | real (halved) |
|---|---|---|---|---|---|
| exact | 90.92 | 0 | 0 / 0 | 1.000 / 1.000 | — |
| R1 0.5 nat | 91.75 | 0.82 | 27 / 0 | 1.076 / 1.077 | ~1.04 |
| R2 top-2, 0.25 p1 | 92.29 | 1.37 | 45 / 0 | 1.126 / 1.130 | ~1.06 |
| R2 top-3, 0.25 p1 | 92.60 | 1.68 | 55 / 0 | 1.157 / 1.161 | ~1.08 |
No docstring token deviates under any rule (they are sure rows); the mode is in practice "comments only".

## 3. TRIED
B12-R (span-aware relaxed acceptance on prose, exact inside code: KILL for +10.5 % tokens per pass on IFEval): there the prose was the answer and the gain small; here the prose is 13.5 % of code positions carrying 45 % of the misses, so the restricted rule recovers 70 % of the unrestricted relaxation's gain with zero program risk, which is the reason the kill does not transfer; its measured gain is similar in size (+4-8 % kept) and its safety argument is what it lacked. R-61N / M-15 (near-tie acceptance, BETWEEN) and RELAX_350 (the unrestricted curves) are the anchors; nothing killed is re-proposed.

## 4. Reproducibility
comment_loophole.py → comment_loophole.out (0.3 s). Inputs: ~/TrainLab/data/d02b_walk_hpd0_code.npz, decision_map/out/baseline.json, vocab.json. The region scan is in the script (regions()); MBPP and agent need walks that store the drafts.
