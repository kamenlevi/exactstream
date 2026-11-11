# DRAFTER_98 — the 98 % drafter: where today's drafter misses, what a drafter must see, and what is reachable (FableGod, mission 15, Mon 2 Nov 2026, 13:0x-13:2x EET)
Marks: [K] measured here (source named), [P] arithmetic on measured numbers, [G] judgement. Rules kept: the full 125B as the target, the drafter only proposes (exact mode stays exact), this MacBook, CPU only (drafter_anatomy.py 9 s, drafter_ensemble.py 3 s; no numpy), no GPU, no weight files (the model's vocab.json was read for token kinds), no downloads, writes only in ~/FableGod, every idea checked against TRIED.md.
Inputs: BREAK_THE_WALL.md, FLOOR_125B.md, ROUTE_350.md, DAY9_2026-11-02.md; decision_map/out (the 125B's top-8 per row on the 36 windows); ~/TrainLab/REPORT.md (D-01, D-02a/b, D-17, D-22, D-23d, E1 / E5, HPD-1, K-HEAD, E2) and ~/TrainLab/data (d02b_walk_hpd0_*.npz, d02b_walk_hpdH_code.npz, d02a/tf_*.npz, d02a_miss.txt, d02b_walk.txt, kh_read.txt, d23_runs.jsonl); ~/TRIED/TRIED.md.

## 0. Verdict in seven lines
1. **Where the drafter misses, per drafted position** [K, 3,283 code positions on the 12 map windows; 14,092 math; 6,945 IFEval; 6,755 multi]: on code DFlash2 is right on 95.5 % of the positions where the 125B is sure (98.5 % at depths 1-3, falling to 88.7 % at 12-15), on 30 % of the 125B's near-ties and on 66 % of its mid positions (p1 between 0.5 and 0.99) at every depth. Its 298 misses split 43 % sure / 12 % tie / 45 % mid. Math is the same shape (94 / 33 / 62 %); chat is worse everywhere (IFEval sure rows 88 %, mid 49 %; multi sure 70 %, mid 42 %).
2. **What 0.98 would need** [P on K]: with the sure positions perfect and the ties at their measured rate, a is 0.949; 0.98 needs the mid positions right 92-95 % of the time (today 66 %). The mid positions are the 125B's own uncertain decisions: its argmax there moves under any cut of its own computation (top-5 8.6 %, 4-bit activations 2-9 %, HC static 26 %, middle-layer reuse 33 %, top-2 26 % [K map]) and under int4 rounding (~8 % of mids [K twin]). Reproducing them at ≥ 92 % is the 125B's computation at full fidelity at that position.
3. **The information bound** [P on K]: for a draft at depth 1 the needed state is the 125B's at the anchor, already computed and exact (that is the bonus token); for depth ≥ 2 it is the trunk's processing of the drafter's own previous drafts, which no verified row contains. A drafter that carries it sequentially reads ≥ 2.7 GB per position (9.5 ms at the byte floor, 36 today): ≥ 300 ms per 32-row block against a 60-65 ms check. A block drafter cannot carry it: it extrapolates from the anchor, and its accuracy even at the 125B-sure positions decays 98.5 → 88.7 % over 15 positions. A 2-bit trunk copy (0.9 GB, 3.3 ms per step) costs 106 ms per 32 rows and keeps mid-row agreement only if precision does not matter there, which the map says it does.
4. **The dead ends hold for the measured reasons**: E1 / E5 (+0.02-0.05 on chat's sure rows: capacity, not data); HPD-1 (the set is fit); the k-head (the drafter's state carries no more of the 125B's margin than its cf); trees (+4.8 % kept on code for +4.8 rows: the second guess is wrong at the sure-row knowledge misses). New and measured here: two DFlash2 variants are not wrong at the same places: either-right 94.0 % vs 90.9 % for one (mid rows 75 vs 66 %), and a confidence pick captures 92.7 % at zero extra bytes in one batched block pass [K, near-identical variants; more with diverse ones [G]].
5. **Three designs, honestly priced** (section 4): (A) DFlash3-XL, a 2-4B block drafter with the learned readout taps, trained on 20-50M on-policy tokens captured locally and trained on free cloud compute: a ≈ 0.94, kept 12-12.5 at 16 rows; (B) the dual-draft ensemble with a confidence pick: a ≈ 0.93 (0.95 with A), kept ~11.5 (13 with A); (C) the AR-flat hybrid (the trained MTP chain holds 0.94 flat through depth 15 [K D-22]) for the deep half of 32-row blocks: pays only if its step is ≤ 0.5 ms (a 4-bit fused head), else a wash. The only 0.98 object is the 125B's own trunk with its experts (near-copy 0.983, twin 0.99), and it is the AR ceiling (D-07): KILL by arithmetic at any width.
6. **The bet**: A + B on the 32-row banded pass (BREAK_THE_WALL 4.1), with D7b's 4-bit drafter layers for cost: code a 0.95, kept ~13 at 16 rows and ~16 at 32 (today 10.5); chat on the MTP chain + the quality-defined tie rule (kept 2.4 → ~3.0) and the trunk-as-drafter (D2) for 5-8. E2's verdict today is the first gate (the data slope on fresh code); the ensemble test is one evening with existing checkpoints.
7. **The measured statement**: 0.98 is not reachable by any affordable drafter on this Mac. The 125B's mid positions (12 % of code, 20 % of math, 55-58 % of chat) are decided by its full computation; every drafter below the trunk sits at 66-78 % there, and the trunk costs 9.5-36 ms per draft position sequentially. The best reachable is a 0.94-0.95 on code (kept 12-13 / 16 rows, 14-16 / 32), 0.92-0.93 on math, 0.65-0.70 on chat (kept 3-3.5), 5-8 on chat only with the trunk as the drafter.

## 1. The error anatomy (per drafted position, teacher-forced along the 125B's own answers)
Source: the HPD-0 walk of DFlash2 L1-20 at block 16 (per check: accepted count k, the 15 drafts and their cf) joined with the 125B's own distribution at the row that verifies each draft (sure p1 ≥ 0.99 / tie p1 − p2 ≤ 0.1 / mid). Every draft up to and including the first miss of a check is a tested position (the rate the engine sees). drafter_anatomy.py; tables in drafter_anatomy.out.

### 1.1 By the 125B's class and by depth [K]
| domain | class | share of positions | right at depths 1-3 | 4-7 | 8-11 | 12-15 | all |
|---|---|---|---|---|---|---|---|
| code | sure | 86.5 % | 98.5 | 97.4 | 94.0 | 88.7 | 95.5 |
| code | tie | 1.5 % | 30.0 | 26.7 | 40.0 | — | 30.0 |
| code | mid | 12.0 % | 69.7 | 59.2 | 63.0 | 52.9 | 65.9 |
| code | all | 100 % | 89.8 | 92.7 | 92.5 | 87.5 | 90.9 (n 3,283) |
| math | sure | 79.1 % | 98.0 | 95.0 | 90.7 | 85.3 | 94.3 |
| math | tie | 1.4 % | 36.5 | 30.0 | 41.7 | 0.0 | 33.0 |
| math | mid | 19.5 % | 67.4 | 56.4 | 54.2 | 55.8 | 61.6 |
| math | all | 100 % | 89.2 | 87.2 | 84.9 | 81.5 | 87.1 (n 14,092) |
| IFEval | sure | 32.6 % | 88.5 | 90.9 | 82.1 | 81.0 | 88.4 |
| IFEval | tie | 9.6 % | 23.9 | 14.5 | — | — | 23.0 |
| IFEval | mid | 57.8 % | 50.4 | 41.6 | 27.5 | — | 49.1 |
| IFEval | all | 100 % | 58.4 | 61.6 | 69.5 | 80.0 | 59.4 (n 6,945) |
| multi | sure | 38.6 % | 66.7 | 86.4 | 75.0 | — | 70.1 |
| multi | tie | 6.8 % | 21.3 | 20.7 | — | — | 21.0 |
| multi | mid | 54.6 % | 43.3 | 33.7 | 31.9 | — | 42.1 |
| multi | all | 100 % | 50.5 | 58.2 | 49.5 | — | 51.5 (n 6,755) |
Agent (the stored d02a walk has no cf and few tested rows, n 159): sure 85 %, tie 13 %, mid 39 %; D-02a's first-miss anatomy on 1,908 agent positions puts the on-path classes at 0.62 / 0.04 / 0.34 and the first misses at 0.38 / 0.08 / 0.54 [K, d02a_miss.txt].
Reading: on code and math the sure rows are near-perfect at shallow depth and decay with depth (the block drafter extrapolates its own drafts without the trunk); the mid rows are wrong a third of the time at every depth (capacity, not depth); ties are coin flips. On chat the drafter misses 12-30 % even of the positions the 125B is sure of: vocabulary and phrasing it does not have (E1 / E5 bought +0.02-0.05 there, so it is capacity).

### 1.2 The reachable gap [P on K]
| domain | misses sure / tie / mid | a now | sure perfect | sure + mid perfect | everything but ties, ties at 50 % |
|---|---|---|---|---|---|
| code | 43 / 12 / 45 % of 298 | 0.909 | 0.949 | 0.989 | 0.992 |
| math | 35 / 7 / 58 % of 1,822 | 0.871 | 0.916 | 0.991 | 0.993 |
| IFEval | 9 / 18 / 72 % of 2,821 | 0.594 | 0.632 | 0.926 | 0.952 |
| multi | 24 / 11 / 65 % of 3,277 | 0.515 | 0.630 | 0.946 | 0.966 |
Arithmetic for 0.98 on code: 0.865 × s + 0.015 × 0.3 + 0.12 × m ≥ 0.98 → with the sure rows at 99.5 % the mid rows must be right 95 %; at 99.9 %, 92 %. The ceiling for any drafter that is not the model is ~0.99 (ties at 50 %), which is where the int4 twin sits (14.8 of 16).

### 1.3 By token kind (code; the target token's kind from the model's vocabulary) [K]
| kind | share of positions | miss rate | share of all misses | share of sure-row misses |
|---|---|---|---|---|
| word after a space (identifiers, keywords, docstring words) | 42.8 % | 12.3 % | 58 % | 44 % |
| punctuation / operator | 21.0 % | 8.6 % | 20 % | 20 % |
| whitespace / newline / indent | 19.0 % | 6.1 % | 13 % | 16 % |
| number | 8.4 % | 5.1 % | 5 % | 9 % |
| word continuation | 6.2 % | 4.0 % | 3 % | 5 % |
| other (mixed, non-ascii) | 2.5 % | 7.3 % | 2 % | 5 % |
The sure-row misses are what the 125B knows and the drafter does not: the return type ('str' drafted as 'bool'), the docstring's own word ('supplied' vs 'given'), the doctest marker ('>>>' vs the docstring's end), the algorithm ('[::-' vs '+'), a comment vs a return; most at depths 8-14 and at cf 0.4-0.9 (the drafter knows it is unsure: only 3 % of its sure-row misses carry cf ≥ 0.99, median 0.67). The drafter already sees the whole context: DFlash2 is 5 layers over the 125B's five-tap features of every context token (26 KB per token) [K REPORT], so these are capacity misses, not input misses.

### 1.4 The drafter's confidence at its misses [K]
Code: median cf 0.67 at sure-row misses, 0.54 at ties, 0.55 at mids; ≤ 3 % of misses at cf ≥ 0.99. Math 0.66 / 0.67 / 0.64. Chat 0.49-0.57. The misses sit at the positions the drafter flags (D1: 49 % of code positions carry 92 % of the breaks), which is what makes the gate work and what makes a tree at those positions fail (the second guess is as uninformed as the first at knowledge misses).

### 1.5 Two drafters are not wrong at the same places [K]
On the 3,240 code positions both walks tested, DFlash2 L1-20 and HPD-1's H (the same model after two more epochs, 93 % identical drafts): either-right 94.0 % vs 90.9 / 91.2 % alone; sure rows 97.7 vs 95.5; mid rows 75.3 vs 66.1; ties 38 vs 30. Picking the draft with the higher cf: 92.7 % (sure 96.9, mid 70.2). The control U (the same two epochs on every position) gives the same: either-right 94.4 %, pick 92.7 % on 3,224 common positions. drafter_ensemble.py. With two genuinely different drafters the union is larger [G].

## 2. The information bound
- **Depth 1 is free and exact.** The row that verifies draft 1 is the anchor's own row; its argmax is the bonus token the check already emits. No drafter is needed for it.
- **Depth ≥ 2 needs the trunk's processing of the drafter's own previous drafts.** The verified rows' states (KV, GDN, residual taps) describe the context up to the anchor; they do not contain what the 125B would do with the next tokens. A block drafter guesses it from the anchor's features and its own small network; the decay of sure-row accuracy with depth (98.5 → 88.7 %) is the measured rate at which that guess goes stale even where the 125B is certain. An AR drafter re-processes each new token: the trained MTP chain holds 0.94 flat through depth 15 [K D-22], no decay, but 0.94 is the capacity of one MoE layer, and each step costs 1.9-2.5 ms.
- **Mid positions need the full computation.** The map measured how the 125B's argmax at its mid rows depends on its own parts: top-2 experts flip 25.6 %, top-5 8.6 %, zeroing the shared expert 30.5 %, HC static 25.9 %, middle-layer MoE reuse 33 %, attention / GDN reuse 29 / 97 %, 4-bit activations 2.3 / 3.8 / 9.1 % (attention / GDN / HC) [K]. The int4 twin (every expert at 4 bits, the same trunk) diverges on ~1 % of code tokens, i.e. ~8 % of mid rows [K D-01]. The near-copy (the trunk + the 10 % hottest experts) agrees 0.983 overall, i.e. ~90 % at mids. So mid-row agreement is a monotone function of how much of the 125B is computed at that position: 0.47B block drafter 66 %, the same with a 4-8× drafter ~75-80 % (the paid-run estimate of +8-20 % kept) [G], the trunk + hot experts ~90 %, the full model at int4 ~95 %.
- **In bytes and ms per 32-token block on this Mac** [P]: the full trunk sequentially (4-bit, HC8): 32 × 2.7 GB = 86 GB → 304 ms at 284 GB/s, ~1.15 s with today's 36 ms dense pass; with the hot experts resident +0.14 GB per step; a 2-bit trunk copy 32 × 0.9 GB = 29 GB → 106 ms (plus launches ~0.7 ms per 48-layer step → ~130 ms), and its mid-row agreement is unmeasured and, by the map's precision sensitivities, likely below the near-copy's; a block drafter of 0.47B: 6.3 ms per block [K D7b]; 2-4B: 1-2 GB → 8-12 ms [P]. The check itself is 60-65 ms per 32 rows at the floor engine. **Any drafter whose cost per block exceeds ~15 ms cannot pay for its accuracy**: at kept 17.5 (a 0.96) the block budget is 65 ms of check, so a drafter at 106 ms loses to one at 6 ms with kept 10.5 (ms per token 9.8 vs 6.8).
- **Where the knowledge is**: in the routed experts and the trunk's deep layers at the draft position (not at the anchor), as the map's sensitivities say; the context alone is not enough (the drafter sees all of it and still misses the docstring's own words at sure rows).

## 3. Designs aimed at 0.98, and what each reaches
Expected per-depth curves on code (right %, depths 1-3 / 4-7 / 8-11 / 12-15), all rows; today 89.8 / 92.7 / 92.5 / 87.5, a 0.909, kept 10.5 at 16 rows and at 32.

### 3.1 DFlash3-XL: a 2-4B block drafter with the learned readout taps, on-policy data at scale
- **Mechanism.** The same object as DFlash2 (a small transformer over the 125B's multi-layer features of the context, drafting a block in one pass) at 4-8× the size, with the trunk's learned HC readout as its input instead of the plain mean of the four streams (D-16, never run properly), trained on 20-50M of the 125B's own answer tokens (MBPP / HumanEval-style code, GSM, the agent turns) captured locally at ~60 tok/s (5M tokens per day) and trained on free cloud compute (a 2B drafter on 20M tokens is ~50-100 GPU-hours [G]; the laptop trains ≤ 1B).
- **Expected curve** [G from 1.1 and the slopes]: sure rows 98.5 / 98.0 / 96.0 / 93.0 (capacity halves the deep decay), mid 75-78 % flat, ties 30 %: all 92.0 / 93.5 / 92.5 / 90.5, a ≈ 0.94, kept 12-12.5 at 16 rows, 13.5-14.5 at 32.
- **Cost.** 1-2 GB at 4 bits in RAM (+0.8-1.8 GB over DFlash2); draft 8-12 ms per block (launch-bound layers + the exact two-stage head) [P]; the stored taps for training 5 KB per token after the frozen fc.
- **Must be true.** E2 shows the data slope alive on fresh code (≥ +0.10 kept per doubling; today's verdict); the readout taps add ≥ +0.3 kept (a one-night test on the existing data); the cloud run fits the free tiers.
- **Test, kill.** E2 (running) → a 1B local variant with readout taps on the fresh MBPP data (one night; kill: kept(16) < 11.5) → the cloud 2-4B (kill: kept(16) < 12.5). Exact (drafts only).
- **TRIED.** "A bigger drafter (4-8×)" was priced as a paid run (~$650-1,050): free cloud compute removes the cost reason; DS / DS2 saturation was on the old corpus, E2 measures fresh data; D-16 (the readout input) was never run properly (21 steps, a random projection).

### 3.2 The dual-draft ensemble with a confidence pick (new; measured headroom)
- **Mechanism.** Two block drafters (different training data / seeds / sizes) run in one batched block pass (the drafter is launch-bound, so batch 2 costs ~+1-2 ms, not 2×); per position the draft with the higher cf is proposed; the exact check decides.
- **Expected curve** [K for near-identical variants, G for diverse ones]: with L1-20 + H the pick gives sure 96.9 / mid 70.2 → a 0.927, kept ~11.3; with two diverse drafters (the union was 94.0 % even for near-twins) a ≈ 0.935-0.94, kept ~12; stacked on 3.1 a ≈ 0.95, kept ~13 / 16 rows, ~16 / 32.
- **Cost.** +0.24-0.47 GB RAM per extra drafter; +1-2 ms per block; nothing on the check.
- **Must be true.** The cf pick keeps most of the union (measured 92.7 of 94.0 % for near-twins); the two drafters stay different after training on the same data (seeds / data order / size).
- **Test, kill.** One evening (MTP Head): walk two existing checkpoints (L1-20 and a differently seeded retrain, or L1-20 and the 1B of 3.1) on the 24 code + 60 math windows, pick by cf, read kept; kill if kept(16) < 11.3 (< +8 %). Exact.
- **TRIED.** Not there (trees branch one drafter; this picks between two drafters' top-1 at every position, which the measurement shows are differently wrong).

### 3.3 The AR-flat hybrid: block draft for depths 1-8, the trained MTP chain for 9-32
- **Mechanism.** DFlash drafts 8 (6 ms, shallow accuracy 0.90-0.93); the native MTP head, trained EAGLE-3-style (D-22: 0.94 flat through depth 15, no decay), continues the chain with the true-pair cache to depth 32; the 32-row banded pass verifies.
- **Expected curve** [K D-22 for the chain]: 89.8 / 92.7 then 0.94 flat → kept(32) ≈ 14-15 (vs 10.5 today, 13.5-14.5 for 3.1 alone).
- **Cost.** 24 AR steps at 1.9-2.5 ms today = 46-60 ms per block: more than it returns (ms per token 7.0 vs 6.8 today at the floor engine); at a 4-bit fused step of ~1 ms, 24 ms: a wash; at ≤ 0.5 ms (a 0.1B head on the trunk's readout) +16 %. The MTP head's experts (1.4 GB) must be resident.
- **Must be true.** A ≤ 0.5-1 ms step: the head at 4 bits (R-84: −0.2..−0.6 ms), the step's launches fused, the experts resident (+1.4 GB RAM).
- **Test, kill.** A 10-minute step timing of the fused 4-bit MTP head (Leap / MTP Head); kill if > 1 ms per step. Then a 32-row walk with the hybrid chain; kill if kept(32) < 13.5.
- **TRIED.** D-18 (the MTP head drafting positions 1-2, DFlash the rest) was KILL (code −2.5 %): the order is reversed here and the chain is the trained one (D-22), not stock; the reason D-18 died (the stock chain decays and is worse than DFlash at depths 1-2) does not apply. D-23d already switched chat to the chain (+24.5 %).

### 3.4 The near-copy as a sequential drafter (the only 0.98 object): KILL by arithmetic
The trunk + the 10 % hottest experts agrees 0.983 [K D-01] and would keep 24 of 32 and 38 of 64; it costs the trunk per position: 9.5 ms at the byte floor, 36 ms today, 304-1,150 ms per 32-row block against a 60-65 ms check. D-07's reason (an AR step reads the dense part) stands at every width. A 2-bit copy (0.9 GB, 106-130 ms per 32 rows) removes the bytes but not the step count, and loses mid-row agreement where precision matters (section 2); it stays the chat drafter (D2), where DFlash2 keeps 2 and even a rough copy of the trunk keeps 5-8.

### 3.5 Quality-defined acceptance at ties (Kamen's call)
Accept the draft when the 125B's p(draft) is within δ of p1. Headroom from 1.1: code ties are 1.5 % of positions and 70 % missed → a +0.01 (kept +~7 %); IFEval ties 9.6 % × 77 % → a +0.074 (kept 2.4 → ~3.0); multi +0.054. Matches the measured near-tie rule (BETWEEN: chat +23 % per pass at rho 0.25, +14 % exact-loop). Not a 0.98 path; a chat lever.

### 3.6 Priced out, with the reason
Trees at the drafter's unsure positions (R-64T: +4.8 % kept for +4.8 rows; the second guess is wrong at knowledge misses: 1.4); a learned gate / certificate head (K-HEAD KILL today: coverage 56 % vs cf's 63.5 %); re-weighted losses on fit data (HPD-1 KILL; HPD-2 on fresh data waits on E2); more multilingual data or borrowed experts at the taps (E1 / E5 KILL: +0.02-0.05); retrieval / n-gram drafts (KILL on agent turns); a mid-block re-draft of the second half (the drafter's second pass costs 6 ms for ~+10 % kept at the deep positions: a wash at the floor; worth it only with a ≤ 3 ms drafter, which D7b's 4-bit head could give); Jacobi / lookahead (KILL ≤ 5 %); self-speculation with an early exit (KILL: 5.7 % agreement untrained).

## 4. The one drafter to bet on, and the plan
**The bet: DFlash3-XL (3.1) as an ensemble of two (3.2), with 4-bit layers and head (D7b) for cost, on the 32-row banded pass; chat on the MTP chain with the tie rule, and D2 when its conversion night passes.** Expected on code [G]: a 0.95, kept ~13 at 16 rows and ~16 at 32 (today 10.5); on math a ~0.92-0.93, kept ~9; chat kept 3-3.5 (chain + ties), 5-8 with D2. On the floor engine with the realizable gate this is code ~250 (16 rows) / ~300-330 (32-row runahead, C 320) quality-defined, ~200 exact.
Ranked plan (owners: MTP Head for every drafter; NewIdeas for the 98 % track's bookkeeping, the certificate log and the router-mass gate; Leap for the 4-bit head timing; EvalSuite for the tie rule):
1. **E2's verdict** (running; MTP Head): the data slope on fresh code. PASS (≥ +0.10 kept per doubling) → 3.1's data arm is real; KILL → 3.1 rests on the readout taps and size alone.
2. **The ensemble walk** (MTP Head, one evening, CPU + a shared GPU walk): two checkpoints, pick by cf, 24 code + 60 math windows; line kept(16) ≥ 11.3.
3. **The readout taps** (MTP Head, one night): D-16 run properly (the learned HC readout as the drafter's input); line +0.3 kept.
4. **The 1B local DFlash3 on fresh data** (MTP Head, one night of training after E2's capture reaches 52K tokens); line kept(16) ≥ 11.5.
5. **The 4-bit fused MTP step timing** (Leap + MTP Head, 10 min); ≤ 1 ms opens 3.3.
6. **The cloud 2-4B** (MTP Head, 1-2 weeks of local capture + free cloud training; Kamen's OK for the compute account and the data upload); line kept(16) ≥ 12.5.
7. **The tie rule on chat** (EvalSuite's same-scores; Kamen's call).
8. **D2's conversion night** (BREAK_THE_WALL 4.4): chat agreement ≥ 0.85 per step.
**If 0.98 is asked for again**: it is the 125B's full computation at the 125B's mid positions (sections 1.2 and 2), which costs 9.5-36 ms per draft position sequentially and cannot be carried by a block drafter; the measured ceiling for drafters below the trunk is 0.94-0.95 on code, and the kept-per-pass wall of BREAK_THE_WALL stands at ~13 / 16 rows and ~16 / 32.

## 5. TRIED check
3.1 ← bigger drafter (paid: the free-cloud route removes the cost), DS / DS2 (old corpus; E2 is fresh data), D-16 (never run properly). 3.2 ← trees (KILL: one drafter's second guess; this is two drafters' first guesses, measured to differ). 3.3 ← D-18 KILL (stock chain first; here the trained chain second), D-07 (AR step cost: the MTP step is one layer, and the design is dead unless ≤ 1 ms, stated). 3.4 ← D-07 / resident-subset self-speculation (the reason stands). 3.5 ← R-61N / M-15 (BETWEEN, non-exact: Kamen's call). Not re-proposed: HPD-1, K-HEAD, E1, E5, trees, lookahead, n-gram, early exit, label recycling, relaxed acceptance as exact.

## 6. Reproducibility
drafter_anatomy.py (9 s) → drafter_anatomy.out / drafter_anatomy.json (every tested position with its class, depth, cf, target and draft); drafter_ensemble.py (3 s). Inputs are the named walk and d02a files and decision_map/out/baseline.json; the token kinds use ~/BigMoE/models/bare/vocab.json. The first-miss view (d02a_miss.txt) and the per-depth rates (d02b_walk*.txt) are MTP Head's and agree with these tables (code first misses 0.46 / 0.07 / 0.47 by class; per-token 0.924 on 24 windows vs 0.909 on the 12 with classes).
