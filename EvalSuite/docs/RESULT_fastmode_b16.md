# 24 GB fast mode b16: verdict (2026-10-23, first run; the second run waits behind the < 12 GB focus)

## Verdict

- **b16 scores the same as the exact mode it speeds up.** Against the v3 tier plan with natural routing (f24nat, the
  +47 % baseline): **5/5 rules pass** (math +2.0, IFEval -4.0 [-8.6, +0.6] not significant, HumanEval+ -1.7, MGSM -2.5
  [-6.6, +1.6] not significant; truncations 14 vs 15 on the core sets); its multilingual and agentic continuations sit
  as close to the natural arm's as the floor does (similarity -0.05 [-0.11, +0.01] and +0.00 vs the floor).
- Against the 4-bit reference it sits at the IFEval limit (-5.0 [-9.3, -0.7]: 4/5 rules), mostly the v3 plan's own
  -1.0 plus b16's -4. MGSM +1.2; multilingual continuations drift a little more than the floor's (similarity 0.55 vs
  0.64; B12 was 0.41); agentic turns are no different from the floor's.

How judged: BigMoE's hook (~/BigMoE/scripts/fast24.py, pinned 36c9cc5), MTP 3:3:0 with fixed widths (deterministic per
prompt), guard off on both sides, 24 GB (C = 128); the same decode sets and rules as the small-RAM verdicts.

| test | b16 | v3 natural arm | 4-bit ref | floor | b16 vs natural | b16 vs ref |
|---|---|---|---|---|---|---|
| math (50) | 88.0 % | 86.0 % | 86.0 % | 88.0 % | +2.0 [-1.3, +5.3] | +2.0 [-3.7, +7.7] |
| IFEval (100) | 85.0 % | 89.0 % | 90.0 % | 88.0 % | -4.0 [-8.6, +0.6] PASS | -5.0 [-9.3, -0.7] FAIL |
| HumanEval+ (60) | 90.0 % | 91.7 % | 91.7 % | 95.0 % | -1.7 [-7.8, +4.5] PASS | -1.7 [-6.4, +3.1] PASS |
| truncations (core) | 14 | 15 | 13 | 12 | PASS | PASS |
| final-answer agreement | | | | | 92.9 % | 94.5 % (incl. MGSM) |
| MGSM (80) | 88.8 % | 91.2 % | 87.5 % | 91.2 % | -2.5 [-6.6, +1.6] | +1.2 [-2.3, +4.8] |
| multilingual similarity (24) | 0.55 | 0.55 | | 0.64 | 0.53 vs floor 0.58 (-0.05 [-0.11, +0.01]) | -0.08 [-0.16, -0.01] vs floor |
| agentic similarity (29) | 0.59 | 0.63 | | 0.58 | 0.65 vs floor 0.65 (+0.00) | +0.01 [-0.06, +0.08] vs floor |
| rules (incl. MGSM) | | 5/5 vs ref (core-only truncations 15 vs 13) | | | **5/5** | 4/5 (IFEval) |

Pending: b16's second core run (also a determinism check: fixed widths should reproduce it token for token). Kamen's
14:50 decision puts the < 12 GB verdicts first, so it runs last in tools/day_runner12.sh (its first chunk ran 14:42).
Data: ~/EvalSuite/results/{f24b16,f24nat}/decode.jsonl; tables results/report_f24b16.txt, report_f24nat.txt,
report_f24b16_vs_f24nat.txt.
