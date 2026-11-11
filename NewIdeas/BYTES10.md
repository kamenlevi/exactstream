# BYTES10: from 62.4 to <= 10 MB of SSD reads per kept token (NewIdeas leads; started Tue 3 Nov 2026 23:5x)

> (a quote from the project owner, kept private)
maybe some research should also get dedicated to that as well".

**The line** (= WALL1_SCOREBOARD 'BROKEN' line 2): SSD MB per kept token <= 10, first at 24 GB, then at 12 GB, judged per
track (EXACT / FAST). [K] = BigMoE's W1B untraced means (S45's six prompts, CAP 256). Replays and map prices are [P] until
measured. At ~12.7 GB/s, 10 MB per kept token leaves >= 1,270 tok/s of disk headroom: RAM stops deciding the speed.

Marks: [K] measured (source named), [P] arithmetic or replay, [G] judgement. Rule: every screen's line is written
(QUEUE.md) before it runs; no 'impossible' verdicts.

## Who does what (the orchestrator 23:5x)
- NewIdeas: chief researcher and scoreboard keeper. Owns the 10x ledger (research/TEN_X_LEDGER.md) and the
  pre-registrations, and routes each factor to its owner once the ledger shows it is real.
- Breakthrough: FLOOR, the measuring instrument (the decision map at scale: quality of every byte-cutting rule).
- MTP Head: kept per check (the drafter side of the union curve).
- MoonShots: bytes per expert (precision on demand, entropy coding at 12 GB).
- BigMoE: builds what passes; owns the [K] rows (W1B).
- Fable C: mission C5 on Kamen's go (~/FableGod/push/MISSION_C5_BYTES10.md): the ledger audit, the floor, a ranked path,
  one genuinely new mechanism. It reads this file and the ledger.

## Scoreboard (one row per RAM line and track)

| line | track | today [K] | best priced [P] | the stack behind the best price | next step |
|---|---|---|---|---|---|
| 45 GB | EXACT | 62.4 (C 256, W1B); by day at C 160 169.4, 118 with the volatile tier 64 / 32 (W1B, BigMoE's night, Wed 03:07) | 60.4 / 38.1 / 33.2 replay (calibrated on 62.4 [K]); 11-bit (MoonShots' NET replay at C 256: -17.6 / -16.1 / -15.8 %, incl. ~4.6 % more parts per GB; per part only x0.955) -> 49.8 / 32.0 / 28.0 | 11-bit scales; the volatile tier at C 160 (-30 % [K]) | M-27b conversion Thu |
| 45 GB | FAST | - | 2.8 / 2.8 / 2.9 (with 11-bit: the 03:4x 2.3 / 2.3 / 2.4 used x0.824 per byte, WRONG (13:4x erratum); not re-priced) | QOM cf 0 + 3-bit sure rows 13.5 / 19.3 / 20.2 -> gated bonus row 5.7 / 4.5 / 4.4 -> top-1 on sure rows | map: QOM cf 0 PASS (03:26); + 3-bit PASS on code / IFEval at cf 0, multi at cf 0.5 (03:54); screens 1-2 Wed 13:45 |
| 24 GB | EXACT | W1B 24 GB (Wed 07:54, C 147 / 46, 33.2 tok/s): decode misses 226.0 / 247.4 MB per kept on code (math 195-228), engine reads 266-291 (prefetch +18-26 %), process 409-427 with prefill | 245.2 / 210.2 / 182.6 (LRU); OPT bound 116.9 / 93.9 / 83.5: no exact path to 10 (the answer's working set does not fit 147 slots) | 11-bit; fewer wasted rows (x0.79 code / x0.6 chat); prefetch accuracy (~x0.85 of reads) | screen 3 CALIBRATED (08:3x): replay within 0.92-1.16x of the engine's misses |
| 24 GB | FAST | - | **code 35.7 map-PASS at the 24 GB pool** (QOM cf 0 + gated last row + 3-bit; 1.07 %, screen 4, 08:48); the same stack on chat at 24 GB: BETWEEN 19.4 (IFEval 8.07 %) / KILL 13.9 (multi 5.99 %). Chat's other stacks were tested at the 45 GB pool only: 84.4 / 81.7-93.7 PASS there, 47.3 / 34.4 BETWEEN there. 11-bit (CORRECTED 13:4x: x0.955 per part, 154 / 49 slots at equal RAM): code 32.1, IFEval 17.3, multi 12.4 [P]; + POOLSPLIT (the LSB pool's RAM to base slots, 171 / 0): code 29.3, IFEval 14.5, multi 10.6 [P]; + DRAFT17 (a drafted bonus row, 14:04; ~0 ms after Leap's one-line fix): **code 19.5** (today's stack alone 35.7 -> 25.2); with a TRL-2 trellis store + DRAFT17 code 4.0 [P]. The 08:59 / 13:28 '22.6 / 11.6 / 8.5' and '20.5 / 9.4 / 7.1' double-counted 11-bit's capacity: WITHDRAWN. The 03:4x on-paper 7.3 / 7.2 / 5.7 leaned on top-1, now off the 24 GB path: withdrawn | QOM cf 0 + 3-bit + gated last row 35.7 / 19.4 / 13.9 (map at the 24 GB pool: PASS / BETWEEN / KILL) -> 11-bit 32.1 / 17.3 / 12.4 -> POOLSPLIT 29.3 / 14.5 / 10.6 -> DRAFT17 code 19.5 (x0.665; chat unchanged) -> SLRU x0.94 / x0.91 / x0.96 | map screen 6 (qom_0_live_ball_c147_t3, masks READY): DRAFT17's code quality; Leap: the _rows16 one-line fix; BigMoE: FAST-mode pool split (C_l 4-8); under 10 at 24 GB without a trellis store needs ~x0.51 more on code |
| 12 GB | EXACT | - | 461-637 / 546-622 / 445-487 (LRU, by miss-count order); OPT 351 / 387 / 304 | - | - |
| 12 GB | FAST | - | (chat cells stand-in-dependent, -34..+17 %: MoonShots M-35) C's stack at 51 / 32: 83.1 / 72.6 / 52.9 [P]; 11-bit 77.7 / 67.0 / 48.7; + POOLSPLIT (65 / 0): 70.1 / 54.6 / 39.4 [P] (CORRECTED 13:4x); DRAFT17 x0.742 on code (62 / 0: 75.6 -> 56.1) (the 03:4x 24.3 / 26.1 / 18.6 leaned on top-1: withdrawn). Engine: T3R2 PASS x1.148 tok/s (P1-K1, Wed 13:1x) | the 24 GB stack at 12 GB slots | <= 10 needs 2-bit resident sure-row records or a 2-pass check whose first pass fits |

All [P] cells: research/TEN_X_LEDGER.md (Wed 03:4x). They replace Fable C's BYTES10.md levels, which a keep-mask bug made
2-4x too low (the ledger's section 1).

## The factor ledger (details and numbers: research/TEN_X_LEDGER.md; partial rows as they land)

| factor | owner | exact? | price so far (code / IFEval / multi) | status | overlaps with |
|---|---|---|---|---|---|
| kept per check (the drafter) | MTP Head | yes | EXACT: flat in kept (x0.91 from k 1 to k 16 at 24 GB, code); wasted rows cost +27 % (code) / +61-62 % (chat) at 24 GB; FAST: the bonus row is amortized, so kept matters there | ledger A (03:28) | the gated bonus row (the same rows) |
| cache size, policy, per-request locality | BigMoE (pools) | yes | 45 GB code: 40.3 of 60.4 is the per-request compulsory set; 24 / 12 GB are capacity (OPT x0.45-0.48 of LRU on EXACT) | ledger A / B, the fixed replay | the volatile tier |
| pool policy: rank-class pools (SLRU-like) / learned replacement | BigMoE | yes (policy only) | on the FAST top-1 + bonus stack at 24 GB x0.52 / x0.74 / x0.87 (OPT room x0.29 / x0.37 / x0.51 on the bonus stack); on EXACT x1.04 (worse); on QOM's gated stack at 24 GB (08:58): SLRU x0.94 / x0.91 / x0.96 load misses, decayed LFU x1.15-2.9 (worse); no policy recovers the keep lost to a smaller pool (QUALITY POOL: KILL) | replay only; screen 3 first | priced per rule, never carried across rules |
| QOM: drafter-sure rows skip the rank 2-9 experts that would miss | NewIdeas -> Breakthrough (map) -> BigMoE | no (cache-dependent; Kamen's decision 2 = YES) | as tested (bonus row exact), 45 GB: cf 0.5 -57 / -27 / -21 %, cf 0 -59 / -32 / -27 %; 24 GB cf 0 -65 / -49 / -47 %. My earlier capture price (-60.4 / -36.9 / -33.4 %; cf 0 -64.3 / -58.5 / -62.4 %) also gated the last row (ledger section 4) | map PASS at cf 0.9 / 0.5 / 0 (cf 0: 0.42 / 2.53 / 1.77 %, Wed 03:26); at C 160 pools (cf 0.5) PASS 0.59 / 3.47 / 2.65 % (03:58) | Q1, COLD, t3, the bonus gate: priced jointly |
| 3-bit sure rows (LSB-on-miss's bound) | Breakthrough (map) | no | with QOM cf 0, bonus row exact: 45 GB -78 / -49 / -39 %, 24 GB -74 / -58 / -54 % | map: cf 0.5 + 3b PASS 0.65 / 4.42 / 3.49 %; cf 0 + 3b PASS / PASS / BETWEEN 0.71 / 4.45 / 4.14 % (Wed 03:44 / 03:54; 3-bit cost z +2.5..+7.1) | QOM (same rows) |
| gated bonus row (Fable C, tag new): the check's last row gated like a drafted row | Breakthrough (map) / MTP Head (the 16th draft) | no | on QOM cf 0 + t3 at 24 GB x0.50 / x0.23 / x0.17 (all bonus rows) | QOM form (qom_0_live_bq) PASS / BETWEEN / BETWEEN (05:06); + 3-bit PASS / BETWEEN / BETWEEN 0.77 / 6.09 / 5.29 % (05:13); both at the 45 GB pool | u8, COLD (the same row) |
| top-1 on sure rows (+ union; OEA k1 / P1-OEA family) | Breakthrough (map) | no | after the bonus gate at 24 GB x0.55 / x0.59 / x0.56 | union-only (screen 2) BETWEEN / KILL / BETWEEN; QOM form (2b, qom_0_k1_live_t3) PASS x3 0.59 / 4.65 / 3.24 % at the 45 GB pool (07:27), but with a kept rank 1 at 4 bits, whose LSB reads make it dearer than top-2 on chat (24 GB: 32.4 / 21.9 / 15.3 vs 35.7 / 19.4 / 13.9). The cheap form (rank 1 at 3 bits) is 24.0 / 11.5 / 7.7, untested: screen 4 B' | QOM (the same rows) |
| 2-bit sure rows (MoonShots M-34, lever 6) | Breakthrough (map) | no | x0.712 of the gated stack's bytes (fixed slots) | qom_0_live_bq_t2 BETWEEN / KILL / KILL 1.49 / 10.13 / 9.77 % (07:27): off the path | t3 (the same records) |
| COLD: unsure rows drop cold ranks >= 4 (per-request table) | Breakthrough | no | superseded at cf 0 by the bonus gate (the only unsure row is the bonus row) | map GENERAL | the bonus gate |
| 11-bit scales + native layout (M-27b) | MoonShots / Leap / BigMoE | yes | per part x0.955 / x0.936 (2,064,384 / 720,896 B); on QOM's gated stack at 24 GB x0.898 / x0.888 / x0.894 (154 / 49 slots), 12 GB x0.92-0.93 (CORRECTED 13:4x; the 08:59 x0.60-0.63 took MoonShots' 45 GB net -17.6 % as per-byte and added the slots again) | conversion Thu, A/B Fri | M-01 (the same fields: never both) |
| M-01 12-bit scale fields | MoonShots | yes | -4.8 % | superseded by 11-bit | 11-bit |
| DRAFT17: a drafted bonus row (one more masked block position), so every last row is gated at cf 0 (NewIdeas 14:04; MTP Head timed it) | MTP Head / Leap (_rows16 fix) | no | code at 24 GB x0.706 (147 / 46), x0.665 (11-bit 171 / 0); 12 GB x0.742; chat ~x1.0 (already ~99 % gated); with a trellis store it removes code's unsure-row reads (TRL-2: 43.2 -> 4.0) | replay; ~0 ms [K MTP Head] after the fix; quality = map screen 6 (masks READY) | the gated bonus row (it extends it) |
| POOLSPLIT: the LSB pool's RAM to base slots, FAST mode only (NewIdeas 13:28; untested re-tune) | BigMoE (pools) | no (changes QOM's resident keep) | 24 GB x0.917 / x0.849 / x0.861 (163 / 0); on 11-bit x0.914 / x0.839 / x0.852 (171 / 0; CORRECTED 13:4x); 12 GB x0.81-0.91; keep +15-17 % of the gap | replay; quality not screened (same classes as screen 4 by the linear keep model) | 11-bit (priced jointly) |
| TRELLIS FAST STORE: sure rows served from a QTIP-style trellis store (TRL-3 3.07 b / TRL-2+3 2.40 b / TRL-2 2.06 b; B10's codes on a FAST bar), the 24 GB pool spent on its smaller records (185 / 239 / 276 slots) (NewIdeas 13:44) | Fit12 (encoder) / Leap (decode) / BigMoE (store) | no | 24 GB, unsure rows from the same pool (U1): code 21.8 / 10.1 / 5.8, IFEval 11.4 / 5.0 / 3.0, multi 8.5 / 4.1 / 2.7; code with unsure rows read from v3 (U2) +39 MB (needs a 17th draft or trellis-served unsure rows); keep gap +32-38 / +67-81 / +87-100 % | bytes BETWEEN (code U2); quality untested, [G] code PASS-ish at every form, chat PASS-edge only at TRL-3 | 11-bit, POOLSPLIT (it replaces them) |
| volatile tier (+7.8 GB of purgeable RAM) | BigMoE | yes | -30 % MB per kept token at C 160 [K] (169 -> 118) | measured (Wed 03:07) | the pool-size line |
| entropy coding (LOSSLESS45) | MoonShots | yes | -13.1 % of the store; BETWEEN (decode cost > read saved at 45 GB) | 12 GB question open | 11-bit |
| JOINT, keepalive, zero-copy, pickup, lever 2 forecasts | BigMoE / Fable A | yes | 0 bytes: they move exposure (BROKEN line 1), not this line | Wed lever-1 arms | - |

## Log (newest first)
- Wed 22:2x: SCREEN 7 KILLS the trellis route to code under 10 MB at 24 GB. The 2.40 b mix (7.2 MB on code)
  measures 1.45 / 9.54 / 10.44 % in-frame (BETWEEN / KILL / KILL; QUEUE 22:2x), with damage 1.6-2.5x its MSE model.
  TRL-2 (4.0 MB) is expected no better. Code's cheapest measured PASS stays ~19.5 MB [P] (today's FAST stack).
  Chat at 24 GB has a measured PASS rule (cf 0.5 + bonus gate, 0.71 / 4.30 / 2.73 %, 22:14) at ~88 MB; the cf 0.3
  version (49-52 MB) is pre-registered.
- Wed 19:5x: FRAME FIX (SWEEP v1.2): judged against REF2-T, the FAST stack's own reference (ranks 2-9 at 3 bits),
  the code routes under 10 MB pass with room [G]: TRL-2 4.0 MB at 1.19 %, the mix 7.2 MB at 1.13-1.21 % (19:5x's
  1.30 / 1.24-1.31 are the conservative frame against the 4-bit model).
- Wed 19:5x: SWEEP v1.1 in the engine frame (unchecked): both code routes under 10 MB at 24 GB stay est. PASS at
  the edge [G]: TRL-2 4.0 MB at 1.30 %, the mix 7.2 MB at 1.24-1.31 % (line 1.34). Today's stack (19.5 MB) 1.18 %.
  s7_trl measures the mix on 4 layers when Fit12's store is whole (scripts/s7_read.py, gate OK on all domains).
- Wed 18:4x: S7_REF3 (unchecked until ~21:00): the 3-bit tail passes on code. Code is 1.04 % flips at 239 slots, and
  [G] 1.23 % on today's 24 GB stack. So BigMoE's flag stays 3 and today's prices hold (19.5 / 14.3 / 10.6 at 24 GB);
  flag 4's x2.06 is not needed. E29 = +0.0011 / +0.0065 / +0.0088 KL.
- Wed 18:0x: SCREEN 6 PASS (unchecked until Breakthrough's ~19:05 read-out): DRAFT17's quality at the 24 GB pool =
  C's on code (1.07 % flips, paired 12 / 12; KL +0.0004). Code's 24 GB FAST stack with DRAFT17 stands at 19.5 MB per
  kept token [P] (11-bit + POOLSPLIT), quality PASS measured at 147 slots.
- Wed 16:3x: W4PRICE (unchecked [P]; data/ledger/w4_price.txt): if tonight's s7_ref3 fails, QOM's kept ranks 2-9
  run at 4 bits (LSB reads). Best 24 GB split Cl 160 / Cm 115: code / IFEval / multi 40.2 / 33.0 / 24.9 MB per kept
  token (x2.06 / x2.30 / x2.36 vs width 3's 19.5 / 14.3 / 10.6), code quality [G] 1.25 % (PASS). On today's 171 / 0
  split every kept LSB is re-read (164 / 274 / 273). 12 GB x1.54-1.74 (Cl 48 / Cm 48).
- Wed 14:04: DRAFT17 ALIVE: a drafted bonus row (MTP Head: ~0 ms after Leap's one-line _rows16 fix, +8.6 ms
  without it) gates code's last rows that have no next draft today; they were a third of code's bytes. Code at 24 GB
  x0.665-0.706 (today's stack 35.7 -> 25.2; with 11-bit + POOLSPLIT 19.5), 12 GB x0.742; with a TRL-2 trellis store code
  4.0 MB per kept token [P]. Quality: map screen 6 pre-registered, masks READY (Fable C's all-bonus-row gate cost code
  only +0.24 points).
- Wed 13:44: TRELLIS FAST STORE byte test BETWEEN. Spending the 24 GB pool on QTIP-style trellis records (B10's codes,
  now on the FAST bar): TRL-2 (276 slots) code 5.8 / IFEval 3.0 / multi 2.7 MB per kept token [P], 2.40 b mix 10.1 / 5.0 /
  4.1. Code's catch is its unsure rows (+39 MB uncached from v3): needs a 17th draft or trellis-served unsure rows.
  Quality untested ([G]: code PASS-ish, chat BETWEEN / KILL except TRL-3). Costs: encoder, ~31-36 GB disk, decode
  kernel: a proposal, not a build.
- Wed 13:4x: ERRATUM (mine). 11-bit is x0.955 per part, not x0.824: the -17.6 % was MoonShots' net replay at the 45 GB
  C 256 pools, which already counts ~4.6 % more parts per GB; at 08:59 I added the slots again. Corrected at 24 GB:
  11-bit 32.1 / 17.3 / 12.4; + POOLSPLIT 29.3 / 14.5 / 10.6 [P]. Withdrawn: 'x0.60-0.63', 'code 22.6 / ~21', 'chat under
  10 at 24 GB' and map screen 5. POOLSPLIT itself stands (x0.91-0.92 code, x0.84-0.86 chat).
- Wed 13:28 [11-bit numbers WRONG, see 13:4x]: POOLSPLIT ALIVE. Under the C stack the LSB pool is nearly idle (chat 0.0 MB per kept without it, code
  2.3-2.7), so its RAM goes to base slots: 24 GB with 11-bit 198 / 0 = code 20.5, IFEval 9.4, multi 7.1 [P]. Chat at
  24 GB is under 10 MB in bytes for the first time; its quality at 198 slots is map screen 5 (masks READY,
  pre-registered: IFEval ~7.2 % BETWEEN, multi ~5.6 % at the KILL edge). Code needs ~x0.52 more.
- Wed 08:59: QUALITY POOL KILL (no eviction policy recovers the sure-row keep a smaller pool loses: capacity, not
  policy; SLRU x0.91-0.96 of misses as a by-product). 11-bit priced with its capacity [WRONG, see 13:4x]: x0.60-0.63 at 24 GB under QOM
  (not x0.824). Code at 24 GB: 35.7 (map-PASS) -> 22.6 with 11-bit -> ~21 with SLRU [P]; chat 11.6 / 8.5 with 11-bit
  (quality BETWEEN / KILL at 147 slots). The 03:4x on-paper 7.3 / 7.2 / 5.7 is withdrawn (top-1 is off the path).
- Wed 08:48: screen 4. At the 24 GB pool the code FAST stack PASSes (1.07 %): code's 35.7 MB per kept token at 24 GB is
  map-PASS at the right pool, with the replay calibrated (screen 3). Chat at 24 GB: BETWEEN / KILL (the pool costs
  +1.98 / +0.70 points). Cheap top-1 is BETWEEN / KILL / KILL: off the path. Next idea, pre-registered: QUALITY POOL
  (choose the pool policy for the router mass that QOM's sure rows keep), a CPU kill test running now.
- Wed 08:3x: screen 3 CALIBRATED. W1B 24 GB (C 147 / 46, the replay's line) misses 226.0 / 247.4 MB per kept token on
  code against the replay's 212.8 (mine) / 245.2 (Fable C): the 24 GB prices stand. Per-kept bytes are flat in kept per
  pass (1.3-7.8), so the registered 7.91 / kept scaling was wrong (part A had said so). Beside it: the engine reads
  18-26 % more than it misses (unused prefetches; 49 % of issued ones used on HumanEval/68), a ~x0.85 lever for BigMoE.
- Wed 07:5x: top-1 priced (scripts/ledger_top1.py). 2b's PASS used a kept rank 1 at 4 bits, which needs LSB reads and
  is dearer than top-2 on chat. The cheap form (rank 1 at 3 bits) is 24.0 / 11.5 / 7.7 at 24 GB: multi under 10 if it
  holds. Screen 4 amended (QUEUE 07:5x): C (code's PASS stack at the 24 GB pool), then B' (the cheap top-1 form at the
  24 GB pool, existing t3keep gate). Masks READY; slot asked of the orchestrator.
- Wed 07:4x: Breakthrough's morning batch (07:27). Screen 2b, top-1 in QOM's form, PASSes x3 (0.59 / 4.65 / 3.24 %),
  fewer flips than top-2 at 3 bits on multi (z -3.05): a resident rank 1 stays at 4 bits. 2-bit sure rows (lever 6) are
  KILL on chat. Caveat found: every QOM quality result used the 45 GB pool's residents, while the 24 GB prices replay
  147 slots; at C 164 the smaller pool already cost ~1 point on chat. Screen 4 (QUEUE 07:4x) tests the stacks at the
  24 GB pool; masks build at the next gap between quiet windows.
- Wed 05:13: qom_0_live_bq_t3 PASS / BETWEEN / BETWEEN (0.77 / 6.09 / 5.29 %). Code has a map-PASS FAST stack at 35.7
  MB per kept token at 24 GB (x0.17 of EXACT); chat's map-PASS stacks are 84-100; its BETWEEN stacks 13.9-47.3 go to
  EvalSuite. Ledger section 0 item 7 holds the status table.
- Wed 05:0x: the gated bonus row in QOM's form (qom_0_live_bq): PASS on code (no cost), BETWEEN on chat (+1.6 / +1.5
  points; under Q1 at 0.99's level). 24 GB FAST start with 3-bit sure rows: 35.7 / 19.4 / 13.9 (engine convention) once
  qom_0_live_bq_t3 confirms (running). Screen 2 (top-1, union-only): BETWEEN / KILL / BETWEEN; its QOM form (2b) 13:45.
- Wed 04:4x: screen 1 (a), the gated bonus row in its union-only form: PASS on code (+0.24 points), BETWEEN on IFEval
  (+3.59), KILL on multi (+4.73). Code's 24 GB FAST start drops to 35.7 (from 54.5); chat stays at 84.4 / 81.7 until
  QOM's own form of the gate (resident keep on that row, no extra reads) is tested: qom_0_live_bq, pre-registered.
- Wed 04:0x: the map's QOM night is complete: cf 0 PASS; + 3-bit PASS at cf 0.5 (all three) and at cf 0 on code / IFEval
  (multi BETWEEN 4.14 %); at C 160 pools PASS. The last-row reconciliation (TEN_X_LEDGER sec 4): my decision-2 price =
  QOM + the gated last row (the engine convention, reproduced within 1-4 points); as tested on the map, QOM cf 0 is
  -59 / -32 / -27 % at 45 GB; screen 1 (a) decides the rest (24 GB FAST start 54.5 / 84.4 / 81.7 -> 35.7 / 19.4 / 13.9).
- Wed 03:4x: the ledger ran (research/TEN_X_LEDGER.md). Fable C's BYTES10 replay had a keep-mask bug (layers sliced
  instead of ranks: base reads on layers 0-9 only), so its levels were 2-4x low; the fixed copy and my replay agree and
  match W1B (60.4 vs 62.4 [K] at 45 GB). 24 GB FAST on paper 7.3 / 7.2 / 5.7 with five levers (three untested), EXACT has
  no path to 10 at 24 GB. My decision-2 price gated the last row (the map's arms do not): corrected in the ledger, sec 4.
- Tue 23:5x: track opened; this file; BROKEN line on WALL1_SCOREBOARD (2917b63); ledger script written (beb18d3), runs
  after BigMoE's quiet lock (04:28) beside FLOOR.
