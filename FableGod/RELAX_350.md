# RELAX_350 — relaxed acceptance where the 125B is unsure, and 350 re-priced with today's measurements (FableGod, mission 16, Mon 2 Nov 2026, 14:1x-14:2x EET)
Marks: [K] measured here (source named), [P] arithmetic on measured numbers, [G] judgement. Rules kept: the full 125B as the target, exact mode stays exact (relaxed acceptance is a separate quality-defined mode; adoption is Kamen's call), this MacBook, CPU only (relax_curves.py 11 s on existing files, no numpy), no GPU, no weight files, writes only in ~/FableGod; every rule checked against TRIED.md (R-61N / M-15 BETWEEN; the exact speculative-sampling A1 KILL).
Inputs: DRAFTER_98.md and drafter_anatomy.json, BREAK_THE_WALL.md, FLOOR_125B.md, ROUTE_350.md, DAY9_2026-11-02.md (through 13:47: HC8 same-scores PASS with a lean; E2 not landed at 14:3x), decision_map/out/baseline.json, ~/TrainLab/data (d02b_walk_hpd0_*.npz, d02a/tf_*.npz), ~/EvalSuite/results (hc8_d4.txt, ref2ct_d4.txt), ~/MoonShots/data/band45_capture.txt, ~/Leap/data (c0, k62p_floor, r85b_d7b), ~/TRIED/TRIED.md.

## 0. Verdict in seven lines
1. **Relaxed acceptance buys little on code at the twin's quality level and a lot on chat** [K re-read]: at ≤ 1.2 % first-order token deviation on code (the int4 twin's level) the best rules raise kept per check 5-11 % (R1 at half a nat: +11 %, 1.19 % deviation; the sampling ratio at T 0.6: +8 %, 1.07 %); at 2 % deviation +18 % (the model's second choice at a quarter of p1); at 3 % +30 % (top-5 at a tenth). On IFEval at ≤ 5 % deviation (chat's twin level) +20 %, at 9-10 % +46-51 % (R-61N's rho 0.25, measured +8-23 % live); multi similar. The orchestrator's "accept every mid and tie miss → 0.96-0.98" would cost 5 % of code tokens, five times the twin's level.
2. **Why it is small on code**: the drafter's wrong tokens at the 125B's mid positions are rarely the model's close second: at a quarter of p1 only 12.5 % of mid rows gain (top-2), at a tenth 19 %; the ties (1.5 % of rows) flip either way. Sure rows never deviate under any rule with x ≥ 0.1 or δ ≤ 2 nats (p1 ≥ 0.99 leaves no room), so "unsure-only" is automatic, not a separate rule.
3. **Exchange rate** [P on K]: on code about +9-10 % kept per +1 % of deviated tokens (first order, teacher-forced); on IFEval +5-6 % kept per +1 %. The gate (Q1 at 0.999) already spends 1.25 % on code and 4-5 % on chat through its context effect, so a relaxed rule on top puts the combined mode at 2.5-4.5 % on code: EvalSuite must test the combination, not the rule alone.
4. **The quality bar and the rules to send to EvalSuite** (section 2): R1(0.5 nat) as the code rule (1.2 % / 3.7 % / 2.5 % deviation on code / IFEval / multi), R2(top-2, 0.25 p1) as the chat rule (1.9 / 7.1 / 4.8 %), and the pair as a type rule; the harness is mission 7's free-running emulation (deviation measured along the engine's own prefix, drift over agent turns, blunders) followed by decision 4 (paired deltas on 210 answers, as HC8 and REF2c-T were judged).
5. **350 re-priced** (section 3) [P on measured terms]: today 63.6 exact on code; this week ~67 exact / ~98 with the Q1 gate (quality-defined) by day, ~113 at night; the floor engine (constant 15-18, NAX at the byte floor, D7b's drafter, the gate, the SSD hidden by C 320 + the 32-row runahead) ~235 exact and ~260 with the twin-level rule; with a kept-13 drafter (DFlash3-XL, not yet) ~260 exact, ~290 twin-level, ~305 at 2 %, ~335-350 at 3 % deviation (16-32 rows, night). **350 on code is reached only by the full stack with a relaxed rule at ~3 % token deviation, three times the twin's level; at the twin's level the same stack lands ~290-295.** By day (C 256, no runahead) every number is 25-35 % lower. Math ~140 exact / ~165 relaxed; chat ~74 / 88-112; mixed ~120 / ~137 / ~160.
6. **The single measurement that decides it**: kept per check with the next drafter (E2's data slope today, then a 1B DFlash3 on the fresh MBPP data: kept(16) ≥ 13). Every engine term is now measured (C0, C2, B3, D7b, Q1, HC8) and the rule's cost is computed here; the drafter is the unknown. The runner-up is C1 live tonight (NAX in the loop), which sets the row term's distance to the floor.
7. **Beyond 350**: with a relaxed rule the 32-row pass keeps only 1.10-1.13× the 16-row pass (today's drafter decays), so it pays through the fixed time at the floor, not through bytes; 64 rows never. On the 24 GB and ≤ 12 GB lines the SSD caps tok/s by misses per kept token: the gate ×2.5, a relaxed rule at 16 rows +18-30 %, together ~×3 over exact; wider passes lose there (misses per check ×1.6 for kept ×1.1).

## 1. Acceptance curves from the existing data [K re-read; relax_curves.py]
Data: every drafted position DFlash2 L1-20 reached in the HPD-0 walks up to and including each check's first exact miss (the rate the engine sees), with the draft d, the drafter's selector softmax cf (≈ q(d)) and the 125B's own distribution at the verifying row (top-8 log-probs on the 12 code windows; top-32 on 60 GSM8K, 30 IFEval and 24 multilingual windows). The agent set has no stored cf. Rules: EXACT; R1(δ) log p(d) ≥ log p1 − δ; R2(k, x) d in the top-k and p(d) ≥ x·p1; R3(T) the speculative-sampling ratio min(1, p_T(d) / cf) with p_T renormalised over the stored top-k; TYP(ε, α) typical acceptance p(d) ≥ min(ε, α·e^−H). "Deviation" = accepted and d ≠ argmax, per tested position (first order: after a deviation the engine's path leaves the stored one; the Oct 21 lesson was real drift 3× the per-token estimate). "Kept" = accepted drafts per check from the per-depth acceptance profile (depths 16-31 continue the measured decay); the model gives 7.78 for EXACT on code where the engine measures 10.54 (within-check correlation), so the engine numbers below use the model's ratios on the measured kept.

### 1.1 Code (3,283 positions; today's kept 10.54 at 16 rows, 10.5 at 32)
| rule | accept % all / sure / tie / mid | deviation % all / tie / mid | accept by depth 1-3 / 4-7 / 8-11 / 12-15 | kept ratio 16 rows | 32 rows | engine kept 16 / 32 |
|---|---|---|---|---|---|---|
| EXACT | 90.9 / 95.5 / 30.0 / 65.9 | 0 / 0 / 0 | 89.8 / 92.7 / 92.5 / 87.5 | 1.00 | 1.10 (DS2 measured 1.00) | 10.5 / 11.1 |
| R1(0.25 nat) | 91.5 / 95.5 / 58.0 / 67.4 | 0.61 / 28 / 1.5 | 91.1 / 93.1 / 92.8 / 87.5 | 1.054 | 1.158 | 11.1 / 11.6 |
| R1(0.5) | 92.1 / 95.5 / 64.0 / 71.5 | 1.19 / 34 / 5.6 | 92.2 / 93.7 / 92.9 / 87.5 | 1.107 | 1.220 | 11.7 / 12.3 |
| R3(T 0.6) | 91.8 / 95.5 / 51.1 / 70.8 | 1.07 / 27 / 5.5 | 91.6 / 93.6 / 92.7 / 87.4 | 1.081 | 1.193 | 11.4 / 12.0 |
| R2(top-2, 0.5 p1) | 92.1 / 95.5 / 58.0 / 72.5 | 1.22 / 28 / 6.6 | 92.2 / 93.8 / 92.9 / 87.5 | 1.109 | 1.222 | 11.7 / 12.3 |
| R2(top-2, 0.25 p1) | 92.8 / 95.5 / 58.0 / 78.4 | 1.92 / 28 / 12.5 | 93.3 / 94.8 / 93.1 / 87.5 | 1.176 | 1.298 | 12.4 / 13.0 |
| R1(1.39) = rho 0.25 | 93.3 / 95.5 / 76.0 / 79.9 | 2.38 / 46 / 14.0 | 94.3 / 95.3 / 93.1 / 87.5 | 1.224 | 1.354 | 12.9 / 13.6 |
| R2(top-5, 0.1 p1) | 94.1 / 95.5 / 86.0 / 85.0 | 3.14 / 56 / 19.1 | 95.6 / 95.6 / 93.6 / 88.1 | 1.296 | 1.468 | 13.7 / 14.7 |
| TYP(0.09, 0.3) | 93.8 / 95.5 / 80.0 / 84.0 | 2.92 / 50 / 18.1 | 95.4 / 95.4 / 93.5 / 87.9 | 1.275 | 1.442 | 13.4 / 14.5 |
Drafts inside the stored top-8: 98.3 % of positions, 81 % of misses. Sure-row deviation is 0.00 under every rule listed.

### 1.2 Math (14,092 positions on 60 GSM8K windows; today's kept 6.61)
| rule | accept % all / sure / tie / mid | deviation % all / tie / mid | kept ratio 16 / 32 | engine kept 16 / 32 |
|---|---|---|---|---|
| EXACT | 87.0 / 94.3 / 33.5 / 61.5 | 0 | 1.00 / 1.05 | 6.6 / 6.9 |
| R1(0.5) | 88.3 / 94.3 / 71.7 / 65.0 | 1.21 / 38 / 3.5 | 1.084 / 1.143 | 7.2 / 7.5 |
| R2(top-2, 0.5) | 88.5 / 94.3 / 70.7 / 66.4 | 1.46 / 37 / 4.9 | 1.102 / 1.163 | 7.3 / 7.7 |
| R2(top-2, 0.25) | 89.5 / 94.3 / 70.7 / 71.5 | 2.46 / 37 / 10.0 | 1.177 / 1.251 | 7.8 / 8.3 |
| R2(top-5, 0.1) | 91.0 / 94.3 / 84.3 / 78.4 | 3.98 / 51 / 16.9 | 1.300 / 1.401 | 8.6 / 9.3 |

### 1.3 IFEval (6,945 positions; today's kept 2.03 with DFlash2, 2.53 with the MTP chain [K D-23d]) and multilingual (6,755)
| rule | IFEval accept all / sure / tie / mid | deviation all / tie / mid | kept ratio 16 | multi accept all / sure / tie / mid | deviation all / tie / mid | kept ratio 16 |
|---|---|---|---|---|---|---|
| EXACT | 59.3 / 88.4 / 22.3 / 49.0 | 0 | 1.00 | 51.4 / 70.1 / 19.4 / 42.2 | 0 | 1.00 |
| R1(0.5) | 62.9 / 88.4 / 44.5 / 51.6 | 3.69 / 22 / 2.7 | 1.17 | 53.9 / 70.1 / 38.2 / 44.4 | 2.50 / 19 / 2.2 | 1.11 |
| R2(top-2, 0.5) | 63.5 / 88.4 / 41.7 / 53.1 | 4.25 / 19 / 4.1 | 1.20 | 54.3 / 70.1 / 36.2 / 45.4 | 2.87 / 17 / 3.2 | 1.13 |
| R3(T 0.6) | 62.5 / 88.4 / 34.6 / 52.6 | 4.43 / 17 / 4.8 | 1.15 | 53.4 / 70.1 / 29.1 / 44.7 | 2.85 / 15 / 3.4 | 1.08 |
| R2(top-2, 0.25) | 66.3 / 88.4 / 41.7 / 58.0 | 7.08 / 19 / 9.0 | 1.35 | 56.3 / 70.1 / 36.2 / 49.0 | 4.84 / 17 / 6.8 | 1.23 |
| R1(1.39) = rho 0.25 | 69.0 / 88.4 / 56.1 / 60.2 | 9.75 / 34 / 11.2 | 1.51 | 57.6 / 70.1 / 47.4 / 50.1 | 6.20 / 28 / 7.9 | 1.30 |
| R2(top-5, 0.1) | 72.9 / 88.4 / 59.6 / 66.4 | 13.65 / 37 / 17.4 | 1.77 | 60.5 / 70.1 / 48.5 / 55.3 | 9.12 / 29 / 13.1 | 1.47 |
The 32-row ratios equal the 16-row ones on chat (the chain never reaches depth 12). R-61N measured +8..+23 % tokens per pass live at rho 0.25 on chat [K TRIED]; the first-order model gives +51 % on IFEval and +30 % on multi, so take the model's chat gains as upper bounds by about 2×.

### 1.4 What the curves say
- The relaxed rules never touch sure rows: with x ≥ 0.1 or δ ≤ 2.3 nats a wrong draft at p1 ≥ 0.99 cannot qualify. So every rule is "only where the 125B is unsure", and its deviation lands on the mid rows (12 % of code, 20 % math, 55-58 % chat) and the ties.
- On code the drafter's wrong token is seldom the model's close second (12.5 % of mid rows at a quarter of p1, 19 % at a tenth): most mid misses are tokens the 125B rates far below its top choice. That is why "accept the mid and tie misses" costs 5 % deviation for 3.9 points of acceptance.
- The speculative-sampling ratio at T 0.6-1.0 (the exact-in-distribution rule for a sampling mode) behaves like R1(0.4-0.6): +8 % kept on code at ~1.1-1.4 % deviation. A1's KILL (Oct 23: +0.003-0.014 per token at the official sampling settings) is consistent with this; it is not a greedy mode.
- Exchange rate: code +9-10 % kept per +1 % deviated tokens; IFEval +5-6 %; multi +4-5 %.

## 2. The quality bar
- **The bar as applied so far** [K EvalSuite]: decision 4 = paired deltas vs REF2-T on 210 answers (GSM8K 50, IFEval 100, HumanEval+ 60) with 95 % intervals and McNemar; a set fails when its interval excludes zero (REF2c-T: IFEval −5.0 [−8.6, −1.4], FAIL; HC8: −4.0 [−8.6, +0.6], −1.0, −3.3, PASS "with a lean against": all five changed answers were losses). The int4 twin's ~1 % of code tokens and ~5 % of chat tokens passed with identical scores [K D-01]; the 3-bit tiers passed R-58.
- **Which rules sit at the twin's level** [P on 1.1-1.3]: code ≤ 1.2 %: R1(0.25-0.5), R2(top-2, 0.5), R3(T 0.6); IFEval ≤ 5 %: R1(0.5), R2(top-2, 0.5), R3(T 0.6); multi ≤ 5 %: those and R2(top-2, 0.25). Beyond it: R2(top-2, 0.25) on code (1.9 %, 2×), rho 0.25 (2.4 % code, 9.8 % IFEval), R2(top-5, 0.1) and TYP (3 % code, 14 % IFEval).
- **The shared budget**: Q1's gate already deviates 1.25 % of code rows (ties and mids through the changed context; 4.0 % IFEval, 2.5 % multi at 0.999 [K DAY9_q1]). Gate + R1(0.5) ≈ 2.4 % on code; gate + R2(top-2, 0.25) ≈ 3.2 %. Only gate + R1(0.25) (≈ 1.9 %) stays near 2× the twin. So the question for EvalSuite is the combined mode.
- **Three rules worth a same-scores run**, with predictions [G]: (1) R1(0.5 nat) on all domains: +11 % kept code, +17 % IFEval, +8 % math; decision 4 PASS likely on code and math, even on IFEval (3.7 %). (2) R2(top-2, 0.25 p1) as the chat rule (code stays at R1(0.5)): IFEval +35 %, multi +23 %; IFEval PASS even (7 %: above the twin's 5 %). (3) R2(top-5, 0.1 p1) on code only: +30 % kept at 3.1 %: PASS unlikely (HumanEval+ 60 answers, ~3 flips per answer at mid positions; the twin passed at 1 %). Run (1) first, then (2) on chat, (3) only if (1) passes with no lean.
- **The harness**: mission 7's emulation on the engine's own loop (the rule applied inside the exact check; the per-token deviation measured along the emulation's own prefix, not teacher-forced; drift over 4-6 long agent turns: last-quarter KLD ≤ 1.5× the first; blunders ≤ 1 per 10K tokens, none in tool calls), then decision 4 on the 210 answers. One night per rule on EvalSuite's runner at C 96-128; the per-type switch (code rule / chat rule) is one engine flag.

## 3. 350 re-priced with today's measurements
Terms (code, per 16-row check) [K unless marked]: today's cycle at C 256 + M-17 = 166 ms (63.6 tok/s at kept 10.54, MoonShots' six-prompt median): draft 12.0, dense 36 (C0), rows 16 × 3.0 (C0, 3c), rest ~5, SSD exposed ≈ 65 [P, the remainder]. This week: D7b 6.3 ms (code PASS); HC8 PASS (dense −3 ms [G: the HC reads are 11.4 ms at 5× their bytes]); A6 one-time. The gate (Q1 at 0.99, coverage 60-66 %): pairs ×0.5 (3c is ALU per pair), SSD ×0.60 (MoonShots [K]); NAX6 at 16 rows 1.17× of 3c per check [K], at 32 rows 2.06 ms per row [K]; the 32-row dense +5 ms (D32 1.24× [K]); the union at 32 rows ×1.6 [K]; night (C 320) SSD ×0.65 of day [G between cachesize45's 37.9 and band45's 47 MB per kept]; the runahead hides ~70 % of the exposed SSD [G from M-16's model]; the floor engine: constant 18 (C0's reachable 15-18), draft 5, rows at the byte floor 1.7 (16) / 1.2 (32) ms per row before the gate, SSD residual 3 with C 320 + runahead. Kept: 10.54 today; ratios from 1.1; DFlash3-XL (DRAFTER_98: a 0.95) 13 / 16 at 16 / 32 rows [G], draft 10.

| stack (code) | cycle ms, day | tok/s day | cycle, night | tok/s night | with R1(0.5) [twin level] | with R2(2, 0.25) [2 %] | with R2(5, 0.1) [3 %] |
|---|---|---|---|---|---|---|---|
| today, exact [K] | 166 | 63.6 | ~143 | ~74 | 70 / 82 | 75 / 87 | 82 / 96 |
| this week exact (D7b, HC8) | 157 | 67 | 134 | 79 | 74 / 87 | 79 / 93 | 87 / 102 |
| + Q1 gate (QD) | 107 | 98 | 93 | 113 | 109 / 125 | 115 / 133 | 127 / 147 |
| + NAX6 live (C1), 16 rows | 104 | 101 | 90 | 117 | 112 / 130 | 119 / 138 | 131 / 152 |
| + 32-row pass, no runahead | 154 | 72 (kept 11.1) | 129 | 86 | 80 / 96 | 86 / 103 | 97 / 115 |
| + 32-row runahead (build) | 111 | 100 | 104 | 107 | 112 / 119 | 123 / 131 | 139 / 148 |
| floor engine, 16 rows, today's drafter | 81 | 130 | 56 (C 320) / 45 (+ runahead) | 188 / 234 | 260 | 276 | 304 |
| floor engine, 32 rows, today's drafter | 94 | 118 | 61 / 58 | 182 / 191 | 212 | 224 | 253 |
| floor engine + DFlash3-XL (kept 13 / 16), 16 rows | 86 | 151 | 50 | 260 | 288 | 306 | 336 |
| floor engine + DFlash3-XL, 32 rows | 99 | 162 | 63 | 254 | 295 | 313 | 352 |
Reading: 350 on code appears only in the last two rows' last column: the floor engine (constant 18, NAX at the byte floor, HC8, D7b), the gate, the SSD hidden (C 320 at night plus the 32-row runahead), a kept-13 drafter, and a relaxed rule at ~3 % token deviation. At the twin's level the same stack is ~290-295; exact ~255-260. By day (C 256, no runahead) every stack is 25-35 % lower and tops out near 200.
**Math / chat / agent / mixed** [P/G]: math (gate coverage 50 %, kept 6.6): floor exact ~140, R1(0.5) ~150, R2(2, 0.25) ~165, R2(5, 0.1) ~180; with a better drafter (kept 8) ~170-220. Chat on the MTP chain (kept 2.53): floor ~74 exact, R2(2, 0.5) ~88, R2(2, 0.25) ~100, rho 0.25 ~112 (the live R-61N gains were half the model's: take ~90-100). Agent (kept 4.9): ~120 exact, ~140 relaxed [G]. Mixed workload (equal shares, harmonic): today 30; floor exact ~120; twin-level rules ~137; the 3 %-rule stack with DFlash3-XL ~160.
**Condition 8**: E2's first verdict had not landed at 14:3x (capture waiting for memory; expected ~15:30). The table carries both cases: today's drafter (kept 10.5) and DFlash3-XL (13); E2 KILL would leave the drafter at ~11-12 at best (the ensemble + readout taps), landing the 3 %-rule stack at ~310-320.
**The single next measurement**: kept per check with the next drafter (E2 today → a 1B DFlash3 on the fresh MBPP data, line kept(16) ≥ 13). The engine's terms are measured (C0 constant, C2 the kernel's distance to the floor, B3 the 32-row cost, D7b, Q1, HC8) and the rule's cost is this document; the drafter is the open term. C1 (NAX in the live loop, tonight) decides how far the row term is from the floor.

## 4. What it opens beyond 350
- **32-64-row passes** [K + P]: with today's drafter the 32-row pass keeps 1.00-1.10× the 16-row one (DS2 measured 1.00; the relaxed rules 1.10-1.13 in the first-order model) while the expert union is ×1.5-1.6 and the misses per check ×1.6; it pays only where the fixed time dominates the rows (the floor engine) and the drafter is flatter (DFlash3-XL / the AR-flat chain), and then by ~10 %; 64 rows never (kept +3-5 %, union ×2.2-2.5).
- **24 GB and ≤ 12 GB lines** [P on the FLOOR replay]: SSD-capped, so tok/s ∝ kept per check ÷ misses per check. At 16 rows the gate cuts misses ×0.4-0.6 and a relaxed rule raises kept ×1.18-1.30 at fixed misses: together ~×2.5-3 over exact (12 GB code ~18-27 → ~50-75; 24 GB ~59 → ~150-190). Wider passes lose there (misses ×1.6 for kept ×1.1).
- **Chat is where relaxation changes the picture**: kept 2.5 → 3.0-3.8 at 4-10 % deviation, i.e. ×1.2-1.5 on chat tok/s at any engine, which no drafter work has bought (E1 / E5 / D-23d gave +25 % in total); on code the same exchange rate buys ×1.1-1.3.

## 5. TRIED check
R-61N (near-tie acceptance on chat, rho 0.25: BETWEEN, +8..+23 % live) and M-15 (the exact cascade form, +14 %) are the measured anchors; the curves reproduce their direction and put the live chat gain at about half the first-order model. A1 (exact speculative sampling at the official settings: KILL for gain) is R3 here: +8 % on code at 1.1 % deviation, a sampling mode, not greedy; not re-proposed as exact. Relaxed / typical acceptance (BETWEEN, non-exact) is what this document prices; nothing killed is re-proposed; the quality-defined mode is published separately with its deviation table, as ROUTE_350 section 8 specified.

## 6. Reproducibility
relax_curves.py (11 s; inputs: the HPD-0 walks, decision_map/out/baseline.json, ~/TrainLab/data/d02a/tf_*.npz) writes relax_curves.out and relax_positions.json (every tested position with its class, depth, cf, log p1 and log p(draft)). The re-pricing terms are the named measurements (C0, k62p_floor, r85b_d7b, band45_capture, DAY9_q1, hc8_d4) combined by the arithmetic of section 3; the kept ratios are the first-order model of section 1 applied to the engine's measured kept.
