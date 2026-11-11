# ROUTE_350 — the concrete build plan toward 350 tok/s single-stream on the full 125B (FableGod, mission 12, Mon 2 Nov 2026, 00:15-00:50 EET)
Marks: [K] measured here (source named), [P] arithmetic on measured numbers, [G] judgement. "Today" = the 45 GB default (C 224, OOO, t 0.3, REF2-T): code 51 / math 44 / chat 23 / agent 21 tok/s, whole set 29.5 (s45d_replay.txt). Every step below was checked against TRIED.md / TRIED_SHORT.md; nothing killed there is re-proposed, and where a kill was conditional the condition is named.

## 0. Verdict in six lines
1. 350 is reachable, if at all, **only on code, only in a quality-defined (non-exact) mode, and only with the engine within ~1.3x of its byte floor**: the exact 16-row check moves 0.78-0.95 GB of RAM traffic per kept token (experts 8.0 GB + dense 3.3 GB per check at kept 12.5-15 [K bytes, P ratio]) against the 0.86 GB that 350 tok/s allows at 300 GB/s, and the SSD caps exact decoding at ~343 tok/s even at a ~51 GB cache (37.9 MB per kept token, cachesize45 [K]). Exact 350 is at or under the physical floor; no build plan reaches it.
2. The exact route lands at **code ~137 (45 GB) / ~155 (51 GB) / ~180 (with a 32-row runahead), math ~125-150, chat ~80, agent ~90** [P/G]: 2.7x today on code, 3.5x on chat and agent.
3. The quality-defined route (top-2 experts on drafter-sure rows inside the check, gated by Monday's decision map, the mission-7 emulation and EvalSuite; Kamen's call) lands at **code ~200 (45 GB) / ~255 (51 GB or runahead), math ~160-190, chat ~90-100, agent ~105** [P conditional].
4. **350 on code needs the conjunction in section 6** (quality-defined rows AND the banded check AND HC static/8-bit AND kernels at the floor AND kept >= 13 AND the SSD off the path): each item has a test; my confidence in the whole conjunction is "unlikely" (~15 %), in reaching 250 on code "even", in 200 "likely".
5. Chat and agent never reach 350 on this Mac: at the floor with a perfect free drafter chat is ~326 (13.7 kept, the bf16 ceiling [K D-01]); with buildable drafters ~100. Mixed workload (equal token shares of code / math / chat / agent, harmonic mean): **today 30; exact route ~105; quality-defined ~125; the full conjunction ~165** [P].
6. The missing factor on code after every known lever is **1.4-1.75x on the check at fixed kept (62 -> 36-43 ms), i.e. ~0.25 GB of RAM traffic per kept token plus ~15 ms of non-byte overhead**; section 7 gives the one new mechanism that supplies part of it (the confidence-banded check) and what remains engineering.

## 1. The target, precisely
- **Prompt sets.** Code: EvalSuite's HumanEval+ prompts (164), greedy, 512-token continuations, context <= 4K; reported also on D-23's 18-prompt set by type (code / math / chat / agent) for continuity with every number in this folder. Math: GSM8K prompts. Chat: IFEval + the multilingual set. Agent: Kamen's agent transcripts with 9-12K prefills (K-80's set).
- **Steady tok/s.** Emitted tokens / wall time from the first check after prefill to the end of generation, median over prompts, over a run of >= 5 minutes with the OS thermal state and SSD temperature logged (W18: day A/Bs carry a throttling term; 350 at 13 GB/s keeps the SSD near 70 C, so the number is the sustained one, not the first minute's).
- **Peak footprint.** Engine RSS + GPU allocations under the NO-SWAP guard: <= 45 GB on the 45 GB line (C 256); the ~51 GB arm (C 320) is a separate line and needs the night's free memory above 42-47 GB (C45 note [K]: C 272+ likely does not fit).
- **Modes.** EXACT = every token bit-identical to the engine's own one-token decode under the same kernel reference (REF2c-T or later), same-scores on the EvalSuite sets vs REF2-T. QD ("quality-defined") = the banded check of section 7; published with (a) the token flip rate vs EXACT on the D-23 set per type, (b) EvalSuite deltas on HumanEval+ / GSM8K / IFEval / multilingual with confidence intervals, (c) the per-band flip table from the emulation. Adoption is Kamen's call; nothing in QD changes the weights, the expert count or the vocabulary.
- **Plainly:** 350 only on code; only QD; the 45 GB line reaches it only if the banded check also cuts the unsure rows' misses (SSD cap ~370 at C 256 with sure-row top-2 alone, no margin [P]); otherwise it needs the ~51 GB arm or the 32-row runahead.

## 2. Where today's check goes, and what 350 allows
Per check, ms (costmodel.py, calibrated on s45d_replay within 5-8 % [K/P]; the per-row and io terms come from FIXED-PART's regression, R^2 0.84 — W25):

| type | draft | constant (dense 27.4 + rest 2.8) | SSD exposed | context | rows x 7.82 | start-up | check | kept | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| code | 12 | 30.2 | 45.7 | 0.5 | 12.3 x 7.82 = 96.2 | 23.1 | 208 | 10.54 | 51 |
| math | 12 | 30.2 | 21.7 | 0.5 | 9.0 x 7.82 = 70.4 | 9.4 | 144 | 6.61 | 46 |
| chat | 12 | 30.2 | 7.6 | 0.4 | 3.1 x 7.82 = 24.2 | 5.6 | 80 | 2.03 | 25 |
| agent | 12 | 30.2 | 15.8 | 23.5 | 7.4 x 7.82 = 57.9 | 86.2 | 226 | 4.92 | 22 |

Floors per term [K/P]: draft ~3-6 (a fused 4-bit block drafter; R-06b says the drafter is launch-bound, not byte-bound); dense 11.7 (3.3 GB at 300 GB/s; 9.0 with HC at 8 bits, 7.5 with HC static); per row 1.7 at the byte floor (0.5 GB of experts per row), 2.1 on NAX's ALU; SSD 0 only if MB per kept token <= 13 x the check's ms per kept token; start-up ~0.
**What 350 allows:** check <= 2.86 ms x kept: 30 ms at kept 10.5, 36 ms at 12.5, 43 ms at 15. In bytes: <= 0.86 GB of RAM traffic per kept token. The exact 16-row check needs 8.0 GB (experts, union 10 x N^0.8 [K]) + 3.3 GB (dense) = 11.3 GB per check: 0.95 GB per kept at 12.5, 0.75 at the bf16 ceiling of 15.2. Wider checks do not help: at 32 rows the union is 160 experts per layer and the near-copy ceiling keeps ~25, 0.65 GB per kept [P]. **This is why exact 350 is closed and why the route is built around fewer expert bytes on the rows where the drafter is sure.**

## 3. The steps (gain, evidence, cheapest decisive test, kill line, owner)
Owners: Leap = kernels; BigMoE = engine + SSD; MTP Head = drafter; MoonShots = ideas / harnesses; EvalSuite = quality; NewIdeas = queue (decision map runs). "10 min" = a live A/B on 3 code prompts at C 224 with token equality vs REF2-T, which the C45 harness already does.

### Block A — the GPU constant and the start-up (exact; this week to two weeks)
| step | gain per check | evidence | test | kill line | owner |
|---|---|---|---|---|---|
| A1 REF2c bundle (att_single + gdn_inproj) | -6 ms [K kernel] | TABLE_350; r36 same-scores tonight | r36 | any set fails same-scores | Leap |
| A2 bmm16 HC mix-down (PASS, never shipped) | -2.5 ms [K kernel] | L1-29: 2.59 vs 5.11 ms | 10 min | tokens differ or < -1.5 ms | Leap -> BigMoE |
| A3 K-73 fusion stack (router, MoE tail, RoPE, residual) | -5..-9 ms [P], part of it on the rows | ~1,100 of 3,944 kernels; arm K tonight | arm K | bits differ or < -1 ms | Leap / BigMoE |
| A4 SPK16 shapes | -4.5 ms [K kernel] | mission 1 | 10 min | < -2 ms | Leap |
| A5 one-kernel GDN body on the 125B (GDN16's 35B result: 2.5x) | -3..-4 ms [P] | L1-15: 11.57 -> 4.23 ms for 30 mixers | 36-layer timing at 16 rows | < -2 ms or bits differ | Leap (1-2 weeks) |
| A6 first check after a long prefill (K-80) | start-up code 23 -> 5, agent 86 -> 20 ms per check [P after tonight's split] | K-80 diagnostic tonight (R-80: a miss burst or cold pools, not kernels) | K-80 | the split names nothing >= 30 % | BigMoE |
| A7 MTP chain + t 0.3 cut on chat / agent | draft 12 -> 8, kept x1.3 on chat / agent [P] | D-23b / D-24 on paper; D-23c tonight; the plain chain is 0.746x on code, so code stays on DFlash | D-23c | < +4 % whole set | MTP Head / BigMoE |
Block A constant 30.2 -> 20 [P] (-> 13 [G] with A5 and the next fusion round). Megakernels / persistent kernels are KILL in TRIED (grid barriers cost more than dispatches on Metal M5): the constant comes down by fusing into fewer ordinary dispatches, not by one kernel per layer.

### Block B — the SSD (exact)
| step | gain | evidence | test | kill line | owner |
|---|---|---|---|---|---|
| B1 C 256 + M-17 per-layer sizes at 45 GB | SSD exposed x0.55: code 45.7 -> 25 ms [P] | replay misses per kept 52 -> 36 (C 224 -> 256) and -20 % bytes (M-17), both [K replay]; engine pending | C45 arms tonight | < +3 % or the swap guard trips | BigMoE |
| B2 C 320 + M-17 (~51 GB) | SSD x0.35: code -> 16 ms [P]; SSD cap 343 tok/s exact | cachesize45 37.9 MB per kept [K replay]; needs more free memory than nights give | C50 Monday if memory allows | swap guard; Kamen's call on RAM | BigMoE |
| B3 32-row runahead (M-16) with NAX32 | SSD exposed -> ~3 ms, rows +10 % [G] | capped at 16 rows it is KILL (-7..+10 %); 32 rows +20..+38 % if beta <= 2.5-4.0 (runahead_model.txt [K model]); D32: a 32-row dense pass costs 1.24x [K] | M-16 step 2 tonight (beta); NAX32 bench (K-62f/g: 1.30-2.77x of 3c at 32 rows [K]) | beta > 3.45 at 16 rows AND NAX32 beta > 2.5 | MoonShots -> Leap -> BigMoE |
Not re-proposed: look-ahead prefetch at 45 GB, pinning, LFU, learned eviction, mmap, chunked reads (all KILL/BETWEEN in TRIED). Draft-window prefetch (R-70D, PARKED, 13-21 % of misses) is worth revisiting only at the end state, where the window is 4-6 ms and the misses per check are a quarter of today's.

### Block C — the per-row cost (exact)
| step | gain | evidence | test | kill line | owner |
|---|---|---|---|---|---|
| C0 instrument: a per-layer GPU timeline of one 16-row check (W25) | replaces the regression's 7.82 / 0.559 with measured terms | FIXED-PART fit R^2 0.84, residuals -18..+31 ms [K] | 10 min (one capture) | n/a | Leap |
| C1 NAX16 expert kernel + fused combine, live, once GPU-bound | per row 7.82 -> ~4.0 [P]: code +38 %, math +40 % on top of B | K-62j 1.46-1.71x of 3c at 16 rows [K kernel]; K-62L's live KILL (+1.8 %) was measured SSD-bound at C 160 with experts ~3-6 ms of a 150 ms cycle [K]: the condition, not the kernel, failed | 10 min: K-62L re-run at C 256 + OOO after B1 lands; fit the per-row slope | < 1.2x on the row term | Leap / BigMoE |
| C2 NAX16 at the byte floor | per row -> 2.6 [G] (2.1 ALU + 0.5 combine) | K-62e/f: dequantize-once; the cost is dequant work, not loads (R-13c) [K] | kernel bench vs 8.02 GB per 16 rows at 215-300 GB/s | < 1.3x of C1 | Leap |
| C3 re-tune the chain cut once rows are cheap (t 0.3 -> ~0.2) | +2-3 % [P] | s45d_replay: t 0.3 optimal at 7.82 ms per row; cheaper rows move the optimum | replay (CPU, 1 min) | < +1 % | MoonShots |

### Block D — kept per check (exact)
| step | gain | evidence | test | kill line | owner |
|---|---|---|---|---|---|
| D1 DLOG harvest from every check (E0) | the data for D2-D4 | running tonight at C 160 [K] | >= 1M tokens per day | n/a | MTP Head |
| D2 third point of the data slope (E2) | decides whether volume is a code lever | D-09 +0.10 / +0.18 per doubling over 2 doublings [K]; DS / DS2: the old corpus saturates at ~10.5 [K] | one evening | < +0.05 per doubling | MTP Head |
| D3 hard-position distillation (HPD, mission 8) | uncertain-position accuracy 0.911 -> 0.95: kept code ~11.4 -> ~13 [G] | 49 % of code positions carry 92 % of the breaks (d1_killtest [K]); no TRIED data lever selects by the drafter's confidence | one night on D-09's protocol | < +0.5 kept | MTP Head |
| D4 DFlash3 (E6: 20-30M on-policy tokens, readout + router inputs, 2 epochs) | code 12-13, math 9, chat 3.5-4 [P] | mission 4's slopes | two weeks of harvest + 6 h | code < 12 | MTP Head |
| D5 chat: coverage (E1) and expert borrowing (E5) | chat kept 2.6 -> 3.3-4 [P/G] | E1 / E5 Monday | E1 / E5 | E1 < +0.05; E5 < +0.1 | MTP Head |
| D6 bigger drafter (4-8x, paid run ~$650-1,050) | +8-20 % kept [P] | TRIED | Kamen's call | — | Kamen |
| D7 a fused 4-bit DFlash3 block | draft 12 -> ~6 ms [P] | R-06b: 4-bit saves 1.3 ms, the rest is launches; SC1's 64K + prompt head 4.0 ms/pass [K] | 10 min | > 8 ms | MTP Head / Leap |
Not re-proposed: trees, suffix / n-gram, exact sampling, depth-decay fixes, calibrated stops, label recycling, self-speculation (all KILL).

### Block E — long context and agent (exact)
| E1 block-sparse 16-row attention | context 2.27 -> ~1.0 ms per K: agent +15-25 % [P] | R-81: dense QK^T grows 1.2-1.4 ms per 1K keys per check [K] | 10 min at 10K keys | < -0.8 ms per K | Leap |

### Block Q — quality-defined (Kamen's call; measured loss; the only lever on the bytes)
| step | gain | evidence | test | kill line | owner |
|---|---|---|---|---|---|
| Q1 top-2 experts on drafter-sure rows inside the check (PFC's expert half, mission 7; the map's outcome (a)) | code rows x0.58, SSD x0.55; math x0.68 / x0.66; chat x0.84 / x0.81; agent x0.73 / x0.70 [P from the sure shares in d1_killtest] | the near-copy keeps 0.983 of code tokens [K D-01]; sure rows are 51-65 % of code positions [K]; the wait-top-2 on ALL tokens with substitution was KILL (IFEval -6.0 at 24 GB): this is conditional on the drafter's confidence and gated per band, a different object | decision map night 1 (topk 2 on code-sure / mid); then the mission-7 emulation on M-16 step 2's harness (1 night); then EvalSuite (1 night) | top-2 flips > 5 % on code-sure rows; emulation: downstream exact-row argmax moves on > 2 % of code rows or > 5 % of chat rows; eval deltas outside the CI | NewIdeas (map), MoonShots (emulation), BigMoE (per-row k in the sync-free gate), EvalSuite (gate), Kamen (adoption) |
| Q2 the confidence-banded check (section 7, NEW) | a further rows x0.85, SSD x0.85 on code [G] | — | same harness, one more night | the mid band's flips > 2x the sure band's | same |
| Q3 HC at 8 bits (ref change) or HC static (map outcome (b)) | dense -2 ms (HC8) / -4 ms (static) [K bytes] | HC8 lossless at 12 GB, "not free" under the 24 GB strict gate (code KLD x1.24) [K]; hc_static from the map | map (b); EvalSuite | static flips > 5 %; HC8 fails the gate | NewIdeas / EvalSuite |
| Q4 exact-loop near-tie rule on chat | chat +14 % [K, BETWEEN] | M-15 / R-61N | exists | — | Kamen |

## 4. The stacked numbers (ms per check -> tok/s; blocks cumulative; 45 GB unless marked)
| stage | code | math | chat | agent | mixed (harmonic) |
|---|---|---|---|---|---|
| today (model) | 208 ms, 51 | 144, 46 | 80, 25 | 226, 22 | 30 |
| A constant 20, start-up fixed, MTP chain + cut on chat / agent | 176, 60 | 125, 53 | 71, 37 | 164, 39 | 44 |
| B1 C 256 + M-17 | 155, 68 | 115, 58 | 66, 40 | 155, 41 | 48 |
| C1 NAX16 live, rows 4.0 | 112, 94 | 83, 79 | 52, 51 | 121, 53 | 64 |
| D kept 12.5 / 8.5 / 3.3 / 7.0 | 125, 100 | 97, 88 | 57, 58 | 126, 56 | 70 |
| E context + agent start-up | 125, 100 | 96, 88 | 57, 58 | 103, 68 | 74 |
| F kernels at the floor [G]: constant 13, rows 2.6, draft 6 | 92, 137 | 67, 126 | 41, 81 | 79, 89 | 103 |
| F + C 320 (~51 GB) | 81, 155 | 62, 138 | — | — | — |
| F + 32-row runahead | 69, 182 | 57, 149 | — | — | — |
| Q1 sure-row top-2 on F (45 GB) | 62, 200 | 53, 162 | 38, 88 (100 with Q4) | 68, 103 | 124 |
| Q1 + C 320 or runahead | 49, 255 | 45, 191 | — | — | — |
| the 350 conjunction (section 6), kept 13 | 37, 350 | ~230 | ~110 | ~130 | ~165 |
Reading the table: through stage E every number is [P] on measured pieces (tonight's and Monday's runs turn A, B1, C1 and A7 into [K]); F is kernel work with measured kernel-level evidence but no live stack yet; Q is conditional on the map and the gate.

## 5. Order of work: the riskiest assumption first, and what tonight's / Monday's results change
1. **Does the check tolerate top-2 experts on drafter-sure rows?** (Q1; decides whether 350 exists at all.) Decision map night 1 -> emulation -> EvalSuite. Until this passes, the plan is the exact route to ~137-180 on code.
2. **Does NAX16 pay live once the loop is GPU-bound?** (C1.) 10 minutes after C45's C 256 arm lands; the biggest exact lever on code and math.
3. **Is kept >= 13 buyable on code?** (D2 + D3, two evenings.) Without it the QD end state is ~200, with it ~255-300.
4. **Does the SSD leave the critical path at 45 GB?** (B1 tonight; B3's beta tonight; C50 Monday.) Sets the 45 GB vs 51 GB question.
5. **Do the measured kernel savings stack live?** (A1-A4 tonight and this week.)
6. **Is HC static / 8-bit free?** (Q3; map (b).)

| result | if PASS | if KILL |
|---|---|---|
| C45 C 256 (+5-12 % code, misses -20..-35 %) | B1 is the night default; C1's re-run next | the SSD lever at 45 GB is Q1 + B3 only; the exact route stays SSD-bound near 100 on code |
| C45 arm L (M-17) | into the default | LRU + prior stays; B1 is C 256 alone (x0.69 instead of x0.55) |
| C45 arm K (K-73) | A3 adopted; next fusion = A5 | find the bit flip; ship the rest; A3 halves |
| r36 (REF2c) | A1 in; next bundle A2 + A4 | A1 = att_single alone (-5 of the -6) |
| D-23c | chat / agent on the MTP chain + cut; D5's AR head (E4) next | chat stays on DFlash; chat end state ~60 instead of ~85-100 |
| M-16 step 2 beta | beta <= 2.5 at 32 rows: B3 is built (NAX32 + no-wait gate) | runahead closed; the SSD needs C 320 or Q1 |
| K-80 split | A6 by cause (keep decode slots through prefill, or warm the pools) | probe the GDN / MoE side after a long prefill |
| E1 / E5 | D5 scaled (100K per language; borrowing at top-2, 10 taps) | chat's wall is the trunk: chat stays ~85 at best |
| C50 (C 300 / 320) | the ~51 GB arm exists: exact ~155, QD ~255 on code | the 45 GB line is the line: QD 350 needs Q2 |
| map (a) top-2 on code-sure <= 1 % | Q1 opens; emulation next night | QD route closed; honest top = exact ~137-180 code |
| map (b) hc_static < 1 % | Q3 static (-4 ms) | HC stays bf16 or 8-bit |
| map (c)/(d) mixer reuse, layer groups | D2's cheap-trunk drafter for chat (per step 4-5 ms) | D2 at ~7.5 ms per step: chat <= 2x |

## 6. The gap, exactly
After every known lever at its measured or estimated value (stage Q1 at 45 GB), the code check is **62 ms per 12.5 kept = 200 tok/s**; with the SSD off the path (C 320 or runahead) **49 ms = 255**. 350 needs **36 ms at 12.5 kept, 43 ms at 15**. The missing factor is therefore **1.4x (from 49) to 1.75x (from 62) on the check at fixed kept**, or **1.2x on kept (12.5 -> 15, the bf16 ceiling) plus 1.15-1.4x on the check**. In bytes: the QD check moves ~0.55-0.65 GB of RAM traffic per kept token (union ~62 experts per layer + dense 3.3 or 2.7 GB) [P], so 350 is above the byte floor by 1.3-1.5x; the whole remaining factor is overhead above bytes (kernels at 2.6 instead of 1.7 ms per row, the constant at 13 instead of 9, the draft at 6, the start-up at 5).
The 350 conjunction on code, each with its state: Q1 PASS (even) AND Q2 the banded check (even, given Q1) AND Q3 HC static or 8-bit (even) AND C2 rows at 2.6 ms live (unlikely-to-even) AND constant <= 13 (even) AND draft <= 4-6 (likely) AND start-up <= 2-5 (likely) AND kept >= 13 (even: D3 + D4, or D6 paid) AND SSD off the path (runahead or C 320: even). Check: 4 + 9 + 2.8 + 0.2 + 15 x 2.6 x 0.58 x 0.85 + 2 = 37 ms at kept 13 = 350. Product of "evens": ~15 %. For chat the missing factor is 4x on kept at a <= 8 ms draft, which no drafter short of the model's own trunk gives (bf16 ceiling 13.7, near-copy 0.90 -> ~9); for agent it is the start-up and the context, both exact and both in blocks A and E.

## 7. New ideas (tagged NEW), and what was priced out this mission
**NEW-1 — the confidence-banded check.** One engine mechanism: every row of the 16-row check carries its own expert count k and tier, chosen by the drafter's confidence band at that position (>= 0.999: top-2 at the resident tier; 0.99-0.999: top-4; 0.9-0.99: top-8 with ranks 0-3 at 4 bits; < 0.9: exact top-10), passed as GPU data through the existing sync-free gate (no host sync; the OOO order and the LY=8 combine stay). The exact mode is the band assignment "all rows top-10", so one engine serves both. The quality bar is per band, measured by the mission-7 emulation: the downstream exact rows' argmax must not move beyond the twin's own 1 % on code; the published QD loss is the sum over bands weighted by their shares. Why it is new: TRIED's expert cuts are uniform (top-8, budgets, DYNK by gate weight, wait-top-2 on every token) and were killed by whole-distribution KLD; the band rule spends the expert budget where the drafter is unsure, which is also where the 125B's own entropy sits (d1_killtest: 49 % of code positions carry 92 % of the breaks). Prize on code beyond Q1: rows x0.85, SSD x0.85 [G]; on the 45 GB line it is what lifts the SSD cap from ~370 to ~430. Test: the emulation harness with the four bands, one night; kill if the mid band (0.9-0.99) flips more than twice the sure band.
**NEW-2 — the per-term instrument.** A per-layer GPU timeline of one 16-row check (kernel kind x layer), captured once per engine change and kept next to the tok/s number. Two of the three largest levers (rows, SSD) rest on a regression with R^2 0.84 (W25); the route's kill lines need measured ms per term. Cost: 10 minutes once; it is the first thing to run on Monday morning before any kernel work is ordered.
**NEW-3 — the confidence-adaptive block length.** With a 32-row dense pass at 1.24x [K D32] and rows at 2.6 ms, extend the drafted block to 24 rows only when the drafter's confidence profile predicts a full keep (expected kept >= 14); with DFlash3 at ~0.95 per token that is ~40 % of code checks: +0.7 kept per check for +6 % dense, net +3-4 % [G]. Small; listed so it is not re-proposed as a 3x.
Priced out this mission (not worth a test): drafting the next block during the check from every plausible cut point (the batched pre-draft costs ~8-10 ms of GPU for ~8 ms saved; it pays only with a <= 0.1B drafter); a pre-dequantized bf16 shadow of the hottest experts (4x bytes against an ALU-bound dequant that NAX already does once per check); screened exact lm_head (TRIED #2 / L1-33: <= 1.1 ms); 16-row padding replaced by 4-row dense passes (the dense is byte-bound, padding costs only ALU); saving GDN states at chosen rows (R-12: the commit is 4.7 ms, KILL).

## 8. Reproducibility and publication
- Every step is a named flag or arm in ref2.py / BigMoE (REF2c, bmm16, K-73 stack, SPK16, C 256 + LAYSZ, OOO, NAX16, NAX32 + runahead, the DFlash3 checkpoint hash, the band thresholds) with its commit and its same-scores or flip-rate result; the C45 harness's token-equality check vs REF2-T is the exact-mode proof and runs on every arm.
- The measurement protocol of section 1 (prompt sets, >= 5-minute steady runs, thermal log, NO-SWAP guard, median over prompts) is the same for every number in the publication; the per-term timeline (NEW-2) is published next to each tok/s.
- The QD mode is published as a separate line with its flip-rate table per type and band, its EvalSuite deltas with confidence intervals, and the band thresholds; the exact mode is published first.
- What the publication can claim with today's evidence: exact ~2.7x on code and ~3.5x on chat / agent over today's default from measured kernel, cache and drafter pieces stacked by a calibrated cost model; 350 only if the section-6 conjunction lands, and the write-up should say which of its nine conditions did.
