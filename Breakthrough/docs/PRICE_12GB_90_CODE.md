# What 90 tok/s whole decode on code at 12 GB requires (pricing, Mon 9 Nov 2026 19:1x)

Target (Kamen, via the orchestrator, Mon 18:5x): 90 tok/s **whole decode** on **code**, lifetime peak <= 12 GB, gated
on the known benchmarks. This note prices the requirement from measured 12 GB runs; no new run was made for it.

## Inputs [K] (Fit12/data/paper/small_ram_12gb_rows.txt, by Fit12/scripts/paper_rows.py: "kept" = kept tokens per check,
"MB/token" = SSD MB per output token inside the decode phases)

| run | label | tok/s (pooled) | kept / check | MB / token | peak GB |
|---|---|---|---|---|---|
| ex16 | verified, 16 HumanEval | 18.47 | 10.08 | 439 | 11.144 |
| tab12e | verified, 12 HumanEval + 12 MBPP | 18.34 | 8.86 | 436 | 11.140 |
| tab12f | two-pass FAST theta 0.95 (failed its gate) | 27.90 | 7.74 | 203 | 11.280 |
| c1u16 | two-pass FAST theta 0.8 (failed its gate), 16 HumanEval | 44.34 | 9.57 | 117 | 11.245 |

Read cost [K]: 83.7 ms per GB read (SMALL_RAM section 7's fit of the 12 GB loop's calls, ~12 GB/s; chat, 1-4 rows).

## Per-check budget today [P from the rows]
check ms = kept / tok/s; GB per check = MB/token x kept; read ms = 83.7 x GB; non-read ms = the rest.

| run | check ms | GB read / check | read ms | non-read ms |
|---|---|---|---|---|
| ex16 (verified) | 546 | 4.43 | 370 | ~175 |
| tab12e (verified) | 483 | 3.86 | 323 | ~160 |
| tab12f (FAST 0.95) | 277 | 1.57 | 132 | ~146 |
| c1u16 (FAST 0.8) | 216 | 1.12 | 94 | ~122 |

The non-read part (16-row verify compute, Python, GPU handoffs, stalls) is 122-175 ms per check at 12 GB, against the
chat fit's 42-59 ms at 1-4 rows: code checks are 16 rows wide.

## What 90 needs [P]
At kept K per check, a check may take K / 90 s. With reads at 83.7 ms / GB, bytes per check B (GB) and non-read N (ms)
must satisfy N + 83.7 B <= 1000 K / 90.

| kept K | check budget ms | B max at N = 40 | N = 60 | N = 80 | N = 120 |
|---|---|---|---|---|---|
| 8 | 89 | 0.58 GB (73 MB/tok) | 0.35 (43) | 0.11 (13) | none |
| 10 | 111 | 0.85 GB (85 MB/tok) | 0.61 (61) | 0.37 (37) | none |
| 12 | 133 | 1.12 GB (93 MB/tok) | 0.88 (73) | 0.64 (53) | 0.16 (13) |
| 14 | 156 | 1.38 GB (99 MB/tok) | 1.14 (82) | 0.90 (64) | 0.42 (30) |

So, at today's kept (~10):
- **Exact cannot reach 90 at 12 GB** on this loop: it reads 4.4 GB per check (370 ms) before any compute. Even with
  N = 0 it would need <= 1.33 GB per check, a 3.3x cut in bytes with no loss.
- **Both terms must fall**: non-read time to ~40-60 ms per 16-row check (from 122-175 ms) AND bytes to ~60-85 MB per
  kept token (from 117 at the failed theta-0.8 point, 439 exact). Or kept per check must rise to 12-14.

## Levers to price next (sources to read before any run)
1. **Non-read time.** The 45 GB engine's chat check is ~40 ms at C 256 (BM:chs2 x2qm2: check 40.5, cycle 48.8 ms) and its
   code runs reach 164-181 tok/s. Running the 12 GB route in that engine (its kernels, fused checks, no per-layer Python)
   is the obvious way to 40-60 ms; the question is its footprint and miss handling at a ~12 GB pool (BigMoE's C-sweeps:
   C 160 is the ~32 GB line).
2. **Bytes.** Code needs <= 60-85 MB / kept token. Known cuts: the two-pass FAST family (117 at theta 0.8, gate FAILED);
   QOM on drafted rows (3-bit, cold sidecars skipped; the 45 GB code path uses it, gated); rank cuts on sure rows; pool
   shaping (BYTES10). The chat replay at 12 GB reached 75.7 MB / kept token with every lossy cut (SMALL_RAM 5.3); a code
   replay at 12 GB is missing (my replays are chat only): **first CPU task**.
3. **Kept per check.** Wider blocks raise kept but also bytes: the expert union grows about k^0.67 with rows (MAIN 12,
   union law), so 2x rows = 1.6x bytes per check; worth it only if kept rises > 1.6x.

Next: Fit12's per-call records for code (asked 19:0x), a code-window replay at 12 GB (CPU), then GPU screens via NewIdeas
after the 24 GB work.

## Code bytes at 12 GB pools [P, CPU replay] (Mon 19:2x; scripts/m1/code12_replay.py -> data/m1/code12_replay.out)
The decision map's 12 HumanEval windows, live-cut blocks (t 0.3: 9.65 rows, 7.91 kept per check), LRU pools of the 12 GB
line (C_m 51 base + C_l 32 sidecar slots per layer), bytes10_core's rules:

| rule | MB per kept token | GB per check | vs EXACT |
|---|---|---|---|
| EXACT | 636.8 (OPT 351.2) | 5.04 | 0 |
| Q1 0.5 (sure rows ranks 0-1) | 173.6 | 1.37 | -73 % |
| QOM0+ball (drafted rows and the last row skip non-resident ranks 2-9; the 45 GB code rule) | 101.8 (OPT 54.2) | 0.81 | -84 % |
| TOP8 | 499.3 | 3.95 | -22 % |
| Q1 0 t3 +bonus all (sure rows ranks 0-1 at 3 bits, every row) | 69.6 | 0.55 | -89 % |

Base-pool sweep (C_l scaled 32/51): EXACT 723 / 694 / 669 / 580 / 466 / 318 and QOM0+ball 155 / 133 / 117 / 88 / 76 / 58
MB per kept token at C_m 24 / 32 / 40 / 64 / 80 / 112.

Check against the engine [K]: Fit12's A' on 16 HumanEval reads 441 MB per kept token at kept 10.09 = 4.45 GB per check;
the replay's EXACT reads 5.04 GB per check (1.13x) at kept 7.91 (1.44x per kept token). Per check the replay is within
15 % of the engine; the per-kept gap is the map's shorter blocks.

Priced route to 90 [P]: one exact 16-row call per check (the 45 GB engine's batch-invariant check, non-read ~50 ms) plus a
QOM0+ball-class cut (~0.7-0.8 GB per check -> ~60-70 ms of reads) gives ~110-120 ms per check at kept ~10: about 85-90
tok/s. The Q1 0 t3 cut (0.55 GB) gives ~95-100 ms: about 100 tok/s. Both are lossy: QOM on code is certified at 45 GB (in
CERT), but at a 12 GB pool it skips far more experts, so it needs its own HumanEval+ / MBPP gate at 12 GB (map screen
first). Footprint: these pools are ~6.5 GB of expert records; Fit12's 12 GB route holds them at 11.1 GB, so a 12 GB port of
the 45 GB engine needs its non-expert part near ~4.5-5 GB (it is ~8 GB at 45 GB).

## The map's code line at the 12 GB pool [K, map arms; information only: known-benchmark gates decide] (Mon 19:4x;
data/m1/code12_quality.out, read-only from the decision map's existing arms)
Code line (VERDICT2): all-row flips <= 1.34 % and non-tie <= 0.87 % (3,366 code rows).

| arm | pool | code all-row [non-tie] | code verdict |
|---|---|---|---|
| qom_0_live (QOM cf 0, bonus row exact) | 45 GB | 0.42 % [0.06] | PASS |
| qom_0_live_ball (+ the last row gated) | 45 GB | 0.51 % [0.21] | PASS |
| qom_0_live_c51 | 51 slots (12 GB) | 1.16 % [0.54] | PASS |
| qom_0_live_ball_c51 | 51 slots (12 GB) | 1.40 % [0.93] | FAIL (just over) |

At 45 GB the 3-bit variants add ~0.3 points on code (qom_0_live_t3 0.71 % [0.21]); at 51 slots that would likely cross the
line. So the map-passing candidate at 12 GB is QOM0 with the bonus row exact: 1.29 GB per check (-74 % vs exact).

Priced with and without read / compute overlap [P]: QOM0's 1.29 GB is 92-108 ms of reads at 12-14 GB/s. If reads add to a
~50 ms compute (the 12 GB loop's fit has no overlap), a check is ~150-160 ms: ~63 tok/s at kept 10. If the 45 GB engine's
out-of-order loads hide the compute under the reads, a check is ~100-110 ms: ~90-100 tok/s. So the two measurements
that decide 90 are (1) whether the 45 GB engine fits Fit12's 12 GB pools (non-expert part <= ~5 GB) and (2) its overlap
at a 12 GB pool's miss rate (SSD queue depth near the drive's 12-14 GB/s).

## The 45 GB engine at 12 GB: not today (BigMoE, Mon 19:5x) [K, BM:data/f24k, data/c30]
- Its lifetime peak outside the expert pools is ~10.1-10.6 GB: F24k at C 100 peaked 23.7 GB with 13.19 GB of pools; C30 at
  C 160 peaked 31.8 GB with 21.21 GB. Of it: 5.6 GB non-pool weights after load (dense layers, embeddings / head,
  drafters, MTP head), ~2.8-3.0 GB of MLX arrays in decode, a 0.3-0.8 GB MLX buffer cache, ~1.1-1.3 GB outside MLX; the
  peak is set in prefill (MLX peak 22.3-22.8 GB in prefill vs 21.7 in decode).
- In its slot sizes (records3 index: MSB 2.064 MB, LSB 0.721 MB per expert, 48 layers) Fit12's 51 + 32 pools are
  6.16 GB, so the non-pool part must fall to ~5.5 GB: a ~5 GB diet, not built (candidates: the 4-bit DFlash2 drafter only,
  no or streamed MTP head on code, a capped MLX cache, a capped prefill transient, embeddings / head on SSD or at 4 bits).
- Its fast path needs rows x 10 <= base slots (or <= 92 ids): at 51 slots only 5-row checks fit, unless BigMoE's U3 rule
  (R57_GATEREQ, BM 742758e) is used: a 16-row check then carries <= 10 x unsure + 2 x sure <= 40 distinct ids. W24's arm
  w16u (Tue 00:30, at 120 slots) is its first run.

So the single-call route has two forms: (a) the 45 GB engine with a ~5 GB non-pool diet (BigMoE prices it after W24), or
(b) batch-invariant 16-row calls inside Fit12's 12 GB loop, which already holds the pools at 11.1 GB (non-pool ~4.6 GB):
the loop's <= 4-row exact calls exist because its kernels are not batch-invariant at wider widths; the 45 GB engine's
kernel set (KERNELS) is. Which is cheaper to build is the next question, for Fit12 and Leap.

## W24's 16-row arm at 24 GB (BigMoE, Mon 19:3x) [K, BM:data/w24/w24_report.txt, BM:docs/WED_TESTS.md]
w16u (U3 + R57_GATEREQ: 16-row code checks on the gate at 120 base slots) read 57.5 MB per kept token against w12's 29.1
(QOM, 12 rows) and had long tails; BigMoE dropped it. At 120 slots, 16 rows kept 11.14 per check against 9.34 at a cycle
median of 98.9 ms against 79.9: no gain per cycle. U3 keeps unsure rows exact (all 10 ids, misses loaded) and ran with no QOM
uses, so for route (a) at 12 GB the gate would need QOM0's classes (every drafted row 2 required ids, the bonus row 10), not
U3's. Route (a) also still needs the ~5 GB diet.

## One call per check at Fit12's call pool (C_m 47): which required sets fit [P, CPU replay] (Mon 21:0x; scripts/m1/code12_union.py -> data/m1/code12_union.out)
Fit12 (21:0x) says its <= 4-row exact calls have two causes:
- **Pool.** A call's top-10 union must fit C_m 47 base slots (4 x 10 = 40).
- **Numerics.** Widths 5-16 were never tested bitwise in its loop. Its WV16 test decides whether a kernel port is needed.

The required set per (check, layer) is the MSB records the rule must hold. QOM's resident-only ranks 2-9 are optional: a
one-call check drops them when the pool is full, so they never block the fit.

| rule | ids per layer, live checks (9.65 rows): mean / p99 / max | live checks with a layer > 47 | 16-row blocks: mean / p99 / max | 16-row blocks with a layer > 47 |
|---|---|---|---|---|
| EXACT | 52.8 / 111 / 141 | 67.9 % | 75.2 / 120 / 145 | 98.1 % |
| Q1 0.5 (rows with cf < 0.5 exact) | 20.4 / 37 / 46 | 0 | 29.7 / 52 / 73 | 16.3 % |
| QOM0 = S1u's U (drafted rows' ranks 0-1, the bonus row's 10) | 19.2 / 34 / 39 | 0 | 26.8 / 41 / 48 | 0.5 % |
| QOM0+ball | 12.6 / 27 / 31 | 0 | 20.9 / 34 / 40 | 0 |

- EXACT 4-row blocks peak at 40 ids (they fit, as Fit12 says). From 8 rows on, 98 % of checks overflow in some layer.
- The sidecar (ranks 0-1 LSB) peaks at 32 at 16 rows, which is C_l 32; it never goes over.
- The 16-row blocks are bytes10_core's synth_blocks(k 16, 'walk'). Rows with no draft cf count as exact there, which is why
  QOM0's peak passes 40. In S1u every drafted row gives only ranks 0-1, so U <= 15 x 2 + 10 = 40 by construction.
- So an exact single call cannot fit, while QOM0's / S1u's required set fits one call per check.

Pool policy at the 12 GB pools (51 + 32), live blocks, in MB per kept token (GB per check):

| rule | LRU | RCP (rank-class pools) | OPT (Belady bound) |
|---|---|---|---|
| EXACT | 636.8 (5.04) | 517.1 (4.09) | 351.2 (2.78) |
| QOM0 | 162.5 (1.29) | 156.7 (1.24) | 93.2 (0.74) |
| QOM0+ball | 101.8 (0.81) | 99.4 (0.79) | 54.2 (0.43) |

RCP is no lever for QOM0 (-4 %), because its needs are ranks 0-1 already. The OPT bound is -43 % (1.29 -> 0.74 GB per
check), so a better eviction policy is the lossless byte lever left, bounded there.

Fit12's prices on its loop (21:0x) [P]: QOM0 at 1.29 GB per check gives ~50 tok/s with reads exposed and ~78 with in-layer
overlap. At the OPT bound (0.74 GB) that becomes about 60 and 90-100. So 90 on this loop needs four things together: one
call per check (WV16 decides port or not), a map-passing union cut, overlap, and a better eviction policy (or kept 12+).

## S1u = the map's Q1-UNION at theta 0: not scored yet (Fit12 21:0x)
What Fit12 can build today, without a residency probe:
- one 16-row call per check, with U = the drafted rows' ranks 0-1 + the bonus row's ranks 0-9;
- each drafted row computes its ranks that are in U, at the model's weights (no renorm);
- the bonus row stays exact;
- it commits up to the first miss.

That is dm_q1u_masks.py's rule at theta 0: every drafted row is "sure", the bonus row is exact, and renorm is off, as in
dm_ablations.q1_union.
- The mask has no pool term, so its score is the same at any pool.
- Its bytes are QOM0's required misses: 1.29 GB per check at 51 + 32 with LRU.

Scored neighbours (code all-row [non-tie]; the line is 1.34 % [0.87]):
- q1u_0.9_live: 1.25 % [0.69], PASS. Rows with cf < 0.9 stay exact there and widen U.
- qom_0_live_c51: 1.16 % [0.54], PASS. It also keeps resident ranks 2-9: U's members (loaded for the check) and more.
- qom_0_live_ball_c51: 1.40 % [0.93], FAIL.

S1u is lossier than both passing arms and may land near the line. It needs one map arm (q1u_0_live: masks on CPU plus one
model run, ~5-10 min at ~17 GB), which waits for Kamen's Mac (his until 23:00) and NewIdeas' queue.

Pre-registered before any data (Mon 21:3x), for the map arm q1u_0_live (= S1u):
- Read-out: `dm_screen_read.py q1u_0_live qom_0_live_c51 --also q1u_0.9_live`, with SCREEN_OUT pointing at this repo.
- Code PASS = VERDICT2: all-row flips <= 1.34 % AND non-tie flips <= 0.87 %. BETWEEN / KILL as the script defines them.
- IFEval and multi are information only: the 12 GB code rule applies on code sets alone, as R57_QOM_SETS does.
- PASS sends S1u to Fit12's S1 engine run. The known-benchmark gate (HumanEval+ / MBPP at 12 GB) decides adoption.
- Otherwise S1 needs QOM0 with a lookup-only residency probe (qom_0_live_c51 PASSES).

## S1u in the engine (Fit12, Mon 22:37-22:52, beside Kamen's apps) [K, Fit12:data/scr/s1u/s1u_read.txt and s1u_rows.txt]
Fit12's s1u_now.sh ran 16 HumanEval at CAP 256 under one share lock with a 15-minute hard stop. The exact reference got 8
prompts in before the stop.
- S1u (arm u01b, one 16-row call per check), timed [K]: whole decode 41.22 tok/s pooled (41.79 geo), kept 10.32, 138 MB
  per kept token, peak 11.335 GB; whole check 250.3 ms; the wide call 219.1 ms reading 1.326 GB (1.427 GB per check in all).
- The read / non-read split is priced, not timed [P, at 83.7 ms per GB, Fit12's chw16 fit]: check 250.3 ms = SSD read
  119.5 + non-read 130.8; the wide call spends 108.2 ms outside reads.
- Against the same-lock exact reference (A'), paired on the 7 prompts both ran [K, recomputed from s1uh.txt / s1ue.txt]:
  S1u 42.6 vs exact 17.87 tok/s pooled = x2.39 (x2.42 geo). CORRECTED 22:5x (NewIdeas): this paragraph first said "12.74
  tok/s: S1u is x3.2 pooled", copied from s1u_read.txt's unpaired line, which set 16 prompts against 8 and kept
  HumanEval/6, VOID by its 40.4 s PAUSE inside.
- Fit12's lines: outside reads FAIL (108.2 > 90), bytes PASS (138 <= 150), so re-price before S2. Not a 90 event.
- My replay's byte price held: the wide call read 1.326 GB per check against code12_union.py's QOM0 LRU price of 1.29 GB
  (x1.03). S1u ran C_m 47 + C_l 32 on 16 HumanEval; the replay ran 51 + 32 on the map's 12 code windows. The eviction
  prices above and tonight's policy replay (code12_policy.py) rest on a replay the engine confirms.
- [P] What 90 needs at kept 10.32: <= 114.7 ms per check.
  - Hiding every read leaves non-read 130.8 ms = 78.9 tok/s, so non-read must drop by >= 16.1 ms.
  - The reads must also hide behind the rest: <= 1.37 GB per check if overlap were perfect (today 1.427; the OPT bound's
    -43 % gives ~0.81 GB = 68 ms).
  - Fit12's order: compaction first, then overlap, then eviction. Compaction: the call still runs all 10 tier_glu columns,
    with weight 0 on the dropped ranks, while ~5.0-5.5 of 10 experts per row-layer are needed.
- S1u's quality is still the map arm's to say (q1u_0_live, from 23:00 while Kamen is idle).

## qom_0_live_c47: QOM0 at Fit12's call pool, pre-registered before any data (Mon 23:4x)
Planned beside EvalSuite's w24 gate (NewIdeas 23:30: EvalSuite confirmed margin for one <= 12 GB neighbour; my flip-only
arms go first). Masks: NewIdeas' builder, arm qom_0_live_c47 = QOM at theta 0 (drafted rows' ranks 0-1, the check's union,
and rank 2-9 experts resident in a 47-slot LRU base pool per layer = Fit12's C_m 47), built into data/m1/masks; then
c47_mask_check.py (same windows and shapes as qom_0_live_c51; q1u_0_live inside c47 by construction; prompt rows kept).
- Why: if S1u (q1u_0_live) fails the code line, S1's fallback is QOM0 with a lookup-only residency probe. Its map line so far
  is at 51 slots (qom_0_live_c51 1.16 % [0.54] PASS); the engine's call pool is 47.
- Read-out: `dm_screen_read.py qom_0_live_c47 qom_0_live_c51 --also q1u_0_live`, SCREEN_OUT = data/m1.
- Code PASS = VERDICT2: all-row flips <= 1.34 % AND non-tie flips <= 0.87 %. IFEval and multi are information only.
- Expectation: between q1u_0_live (union only) and qom_0_live_c51; four fewer slots should move flips little.
- PASS: the QOM0 fallback holds at the engine's own pool; the known-benchmark gate decides adoption. FAIL: the probe needs
  a larger pool (51 passed) or S1u must pass.
- Run: scripts/m1/beside_gate.sh with bg_run_map.py (in-memory registration; DM_C from tonight's measured peak; a
  self-guard ends the arm above 11.9 GB lifetime footprint, so it measures <= 12.0 GB).
- AMENDMENT (Tue 03:3x, before any data): map arms do not fit the 12 GB neighbour slot (Mon night's arm measured 15.96 GB),
  so this arm runs in Tuesday's day queue instead: NewIdeas' TUE 02:02 order, item 2, after the w24 gate's end and BigMoE's
  W24e. Runner scripts/m1/c47_day.sh (armed 03:37, pid 68372): bg_run_map.py at DM_C 51, as qom_0_live_c51 ran, with a
  20 GB self-guard; start by 14:50, hard stop 14:58 (weekday big tests end by 15:00). The code rule, the read-out and the
  comparison are unchanged.

## qom_0_live_c47 on the map: code FAIL by one flip (Tue 09:14) [K, map arm; data/m1/SCREEN_read_qom_0_live_c47.txt]
Pre-registered above (8a5a382; amended 6bc7efa before any data). Run: scripts/m1/c47_day.sh, log data/m1/c47_day.log.
- The w24 gate ended at 08:41:09 and BigMoE's W24e ran 08:41-09:01.
- Fit12's IFBench chunk yielded to my quiet ticket (granted 09:10).
- The arm ran 09:10:13-09:14:16 at DM_C 51. Result: ~/FableGod/decision_map/out/qom_0_live_c47.json.

| arm | code all-row | code non-tie | IFEval all-row | multi all-row |
|---|---|---|---|---|
| qom_0_live_c51 (QOM0, 51 slots) | 1.16 % (39/3366) | 0.54 % (18/3316) | 8.80 % | 5.26 % |
| qom_0_live_c47 (QOM0, 47 slots = Fit12's C_m) | 1.37 % (46/3366) | 0.69 % (23/3316) | 9.24 % | 5.71 % |
| q1u_0_live (S1u: union only) | 1.96 % (66/3366) | 1.36 % (45/3316) | 15.72 % | 11.15 % |
| line (VERDICT2; IFEval / multi information only) | <= 1.34 % (45 flips) | <= 0.87 % | 5.42 % | 3.69 % |

- VERDICT2 FAIL on code: all-row 46 flips against 45 allowed (BETWEEN); non-tie PASS. IFEval and multi KILL (information).
- McNemar vs qom_0_live_c51: code 14 / 7 only-flips, z +1.53; non-tie 7 / 2, z +1.67. Neither is significant at 1.96.
  vs q1u_0_live: 16 / 36, z -2.77 (c47 is better).
- The pre-registered expectation held: c47 falls between q1u_0_live and c51. The four slots moved 7 net flips, enough to
  cross the line by one.
- Decision (pre-registered): the probe needs a larger pool, and 51 is the only passing pool tested.
- Price of 4 more slots per layer [P]:
  - 4 x 0.104 = 0.42 GB (a base slot holds 48 three-bit MSB records of 2,162,688 B; the LSB sidecar pool C_l is apart).
  - On S1u's measured 11.335 GB at C_m 47, that gives ~11.75 GB, if the engine's slot costs the same.
  - Fit12's engine numbers decide whether C_m 51 fits under 12.0 GB. The known-benchmark gate decides adoption.
- Fit12's answer (Tue 09:59) [K, checked at the sources]: C_m 51 fits under 12.0 GB.
  - The engine's pool prints: C_m 47 / C_l 32 = 6.03 GB (Fit12 data/scr/s1u/chw14ha.txt, s1ue.txt), C_m 51 / C_l 32 =
    6.44 GB (data/scr/t3p/c51.txt). So +0.41 GB, 0.1025 GB per base slot: the price above holds.
  - A real run at 51 + 32 exists: data/scr/t3p/c51.txt (3 Nov; 4 HumanEval prompts x 5 arms at CAP 256, two-pass code)
    ends "lifetime 11.734 GB". S1u's 11.335 + 0.41 = ~11.75 [P] agrees. Margin ~0.27 GB on those prompts.
  - Fit12's route is therefore QOM0 at 51 + 32 with a lookup-only residency probe (a copy of the v3p C++ ext), then the
    non-read cut, then the overlap.
- What the four slots buy in reads [P, CPU replay, Tue 09:1x; scripts/m1/code12_pool47.py -> data/m1/code12_pool47.out;
  C_l 32 at both; LRU; its C_m 51 rows reproduce code12_replay.out exactly]:

| rule | C_m 47: MB per kept (MB per check) | C_m 51: MB per kept (MB per check) | change |
|---|---|---|---|
| QOM0 | 168.6 (1334.0) | 162.5 (1285.8) | -3.6 % |
| QOM0+ball | 105.3 (832.7) | 101.8 (805.1) | -3.3 % |
| EXACT | 647.4 (5121.9) | 636.8 (5037.9) | -1.6 % |

  So C_m 51 costs ~0.42 GB [P] and saves ~48 MB of QOM0 reads per check. This is a relative replay figure: the replay's
  absolute bytes sit 1.13x above the engine's per check ("Check against the engine" in the code-bytes section above).
- Engine check at this pool [K vs P]: Fit12's S1p profile (Fit12 data/scr/s1p/s1p_prof.txt) reads 1.323 GB per wide call
  for S1u at C_m 47 (mean of 323 calls, 11.67 GB/s over the loader's io time). The replay prices QOM0 at C_m 47 (QOM0's
  reads equal S1u's by construction) at 1.334 GB per check: within 1 %, although the check shapes differ (15.8 rows per
  engine call, 9.65 rows per live map check). So the -3.6 % should carry over: ~-4 ms of io per call at C_m 51 [P].
- Measured peak: 16.020 GB lifetime footprint (MLX active 15.33 GB), wall 237 s including the load.

## QOM0's extra compute over S1u [K counts from the scored masks; P for time] (Tue 10:16; scripts/m1/code12_qom_compute.py -> data/m1/code12_qom_compute.out)
QOM0 keeps every resident rank 2-9 expert on top of S1u's cut. These cost no SSD reads, so the byte replays price QOM0 =
S1u, but the engine still applies them from RAM. Counted on the masks the map scored, over the replays' live checks on the
12 code windows. Real rows only: a check's stand-in rows for wrong drafts carry another position's mask, and a prompt row
there keeps all 10 in every arm (with stand-ins, S1u's per-layer maximum reads 102 instead of 39).

| arm (real rows, 7.91 per check) | (row, expert) pairs per check | distinct expert-layers per check | vs S1u | max distinct in a layer |
|---|---|---|---|---|
| q1u_0_live (S1u) | 1769.2 | 691.5 | - | 39 |
| qom_0_live_c47 | 2354.3 | 1019.5 | pairs +33 %, distinct +47 % | 47 |
| qom_0_live_c51 | 2410.5 | 1055.8 | pairs +36 %, distinct +53 % | 51 |
| EXACT (all 10) | 3797.3 | 2102.3 | pairs +115 %, distinct +204 % | 138 |

- QOM0's per-layer maximum equals its pool (47, 51): residents are bounded by the pool, as built. S1u's 39 matches the
  union fit's maximum (code12_union.out).
- qom_0_live_c51 adds 364 distinct expert-layers per check over S1u: 0.79 GB of 3-bit MSB records applied from RAM. If the
  expert kernels are bandwidth-bound, that is 2.6 ms at 300 GB/s or 3.9 ms at 200 GB/s per check [P]. The engine's own
  per-expert cost decides; an engine A/B of S1u vs QOM0 at the same pool measures it.
- Against the reads: C_m 51 saves ~4 ms of io per call [P] (section above). In time the two roughly cancel, but they land
  on different sides of the overlap: the extra compute adds to the ~102 ms outside-reads time that Fit12 is cutting, and
  the saving is in the io time.

## q1u_0_live (S1u's cut) on the map: code FAIL (Tue 00:01) [K, map arm; data/m1/SCREEN_read_q1u_0_live.txt]
Pre-registered above (480001c) before any data. Masks: NewIdeas' builder, data/dm_masks/q1u_0_live; bit-check PASS after the
23:5x fix of the check itself (answer-row kept share 0.4994, vs q1u_0.9_live's 0.7651). Run: scripts/m1/s1u_map_night.sh ->
q1u0_run_map.py at DM_C 51; result ~/FableGod/decision_map/out/q1u_0_live.json.

| arm | code all-row | code non-tie | IFEval all-row | multi all-row |
|---|---|---|---|---|
| q1u_0_live (S1u: union only, theta 0) | 1.96 % (66/3366) | 1.36 % (45/3316) | 15.72 % | 11.15 % |
| qom_0_live_c51 (QOM0: union + residents, 51 slots) | 1.16 % (39/3366) | 0.54 % (18/3316) | 8.80 % | 5.26 % |
| q1u_0.9_live (union only, theta 0.9) | 1.25 % | 0.69 % | 6.45 % | 3.77 % |
| line (VERDICT2; IFEval / multi information only) | <= 1.34 % | <= 0.87 % | 5.42 % | 3.69 % |

- VERDICT2 FAIL on code: all-row BETWEEN, non-tie OVER. IFEval and multi KILL. McNemar: worse than qom_0_live_c51 (code
  39 / 12 only-flips, z +3.78; non-tie 30 / 3, z +4.70) and than q1u_0.9_live (40 / 16, z +3.21).
- Decision (pre-registered): S1u is not adoptable. Fit12's S1 needs QOM0 with a lookup-only residency probe (c51 PASSES).
  qom_0_live_c47, QOM0 at the engine's call pool C_m 47, is pre-registered (8a5a382); its masks are built and checked
  (kept share 0.702 vs c51's 0.715). It needs a slot. (It ran Tue 09:14: code FAIL by one flip, 1.37 %; section above.)
- At theta 0 every drafted row keeps only ranks 0-1 plus the union. Keeping the resident experts too (QOM) cuts the code
  flips from 66 to 39.
- A map arm's measured peak: 15.96 GB lifetime footprint (MLX active 15.33 GB) at DM_C 51, wall 224 s including the load.
  The expert cache is ~5.3 GB of it (0.104 GB per slot), so a <= 12 GB map arm needs DM_C ~10 (heavy SSD streaming). Map
  arms therefore do not fit the 12 GB neighbour slot beside the w24 gate: data/m1/beside_gate.done, 00:03.

## Eviction policies at the 12 GB pools [P, CPU replay] (Tue 00:0x; scripts/m1/code12_policy.py -> data/m1/code12_policy.out)
MB per kept token (GB per check) on the map's 12 code windows, live checks, 51 + 32 slots per layer. The current check's
experts are pinned, as in the engine's call. PIN uses in-sample popularity (optimistic).

| rule | LRU | LIP | LRU-2 | LFU decayed 0.5 | 0.8 | 0.9 | PIN | OPT (bound) |
|---|---|---|---|---|---|---|---|---|
| QOM0 | 152.1 (1.20) | -2 % | -5 % | -1 % | -5 % | -6 % | +3 % | 93.2 (0.74) -39 % |
| QOM0+ball | 95.9 (0.76) | 0 % | -5 % | -1 % | -4 % | -6 % | +5 % | 54.2 (0.43) -43 % |
| EXACT | 589.5 (4.66) | -24 % | -1 % | 0 % | -1 % | -1 % | -8 % | 351.2 (2.78) -40 % |

- Simple policies win back 5-6 % of QOM0's bytes. OPT's -39 % needs knowledge of future use, such as a prefetch hint
  from the drafter's next checks. LIP helps only EXACT, whose rank 2-9 loads pollute the pool.
- This replay's LRU sits 6-7 % below bytes10_core's (162.5 / 636.8 / 101.8): it pins the whole current check's set,
  while the core's LRU can evict an expert the same check touches later. The engine measured 1.326 GB per check at
  C_m 47 (S1u): x1.10 over this LRU at 51, x1.03 over the core's.
- For 90: eviction is a margin lever (about -6 % of reads), not a route. Compaction (non-read) and overlap (hide the
  reads) stay first.
