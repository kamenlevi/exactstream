# 12 GB LOSSY: 57 -> 90-130 tok/s on code (Breakthrough + Fit12, Thu 5 Nov 2026 19:38)

- **Goal** (Kamen 19:4x, via the orchestrator): 12 GB lossy at 90-130 tok/s on code, as soon as possible, then publish.
- **The bar**: EvalSuite same-scores on code.
- **Fit12's section** has the measured budget and its levers: `~/Fit12/PLAN_12GB_90_fit12.md`. It is linked here, not
  copied, because Fit12 does not write in this repo.

**The budget in brief** [K]. t80ck1u (T3R2) spends ~261 ms per check for ~15 kept tokens: 57.3 tok/s on the shared text,
44.2 on full outputs.

| part | ms per check |
|---|---|
| draft | 13 |
| pass 1 (P1-K1): wait 53, compute 71 | 130 |
| exact passes (46 % of checks) | 105 |
| commit | 12 |

The exposed SSD wait is ~128 ms, 49 % of the check. It reads 1.23 GB per check at ~9.6 GB/s, 70-80 % of the SSD's
12-14 GB/s.

## Update Thu 19:5x (Fit12's replay: `~/Fit12/PLAN_12GB_90_fit12.md`, "Lever 1 RESULT")

- **The certificate is token-exact.** It shows 0-1 divergences from the exact model in ~3,400 tokens, at every θ and cf.
  So all of the map's damage comes from pass 1's KV, and lever 2 as first framed (a sharper certificate) is **KILLED**.
- **Calibrated speeds** come from Fit12's replay (full outputs, 16 HumanEval prompts, cf gate 0.9), within 2-4 % of
  T3R2's measured runs. Without the cf gate, every point is ~8 % slower.

  | pass 1 | θ 0.8 | θ 0.9 | θ 0.95 | θ 0.98 |
  |---|---|---|---|---|
  | P1-K1 | 45.8 | 41.0 | 37.9 | 35.3 |
  | top-2 | 40.2 | 37.6 | 35.9 | 33.4 |

- **[CORRECTED Fri 13:2x: t80ck2 is FAST, not the exact mode; the exact loop runs ~18.5 tok/s. See "Update Fri 13:2x".]**
  **The code-valid points so far are slower than the exact mode.** Top-2 at θ 0.95 (1.22 %, PASS) runs ~36 tok/s; the
  exact mode, t80ck2, runs 40.8 on full outputs and 49.9 on the shared text. Until a FAST form passes code and is
  priced, the base for the factors is the exact mode. From 49.9, 90-130 needs x1.8-2.6.
- **Lever 2 is now a wider pass 1.** It means a better KV for the certified rows. On the map, top-5 at θ 0.9 passes
  (0.92 %).
  - Step 0b on Saturday finds the cheapest passing width: top-3 / top-4 / top-5 at θ 0.8 / 0.9 (topk3 / topk4
    registered by install_topk34.sh).
  - Fit12 builds the engine form (ranks 0-4 in pass 1) and prices it from one calibration run.
- **Levers 1 and 3 apply to the exact base unchanged.** They come first now.

## Update Thu 20:0x: the two-pass FAST route is closed at 12 GB, except T3R2's own point [CORRECTED Fri 13:2x: this rested on t80ck2 being the exact mode; it is FAST. See "Update Fri 13:2x".]

- **A wider pass 1 is slower** (Fit12, `~/Fit12/data/scr/t3cal`; full outputs, cf 0.9). Top-3 runs 34.5 / 33.1 tok/s at
  θ 0.9 / 0.95, with pass 1 at ~207 ms.
- **Top-4 / top-5 cannot run as one pass.** At 16 rows their union overflows the pool: "54 experts > C=47 at layer 0",
  and the engine aborts. The workarounds (two 8-row calls, or widening only the low-mass rows) add pass-1 cost to a mode
  that is already at or below exact.
- **So every two-pass point except P1-K1 at θ 0.8 (45.8, x1.12 over exact) is no faster than the exact mode.**
  - Saturday keeps one arm: T3R2's own form, `p1c_oea_k1_ku4_p08_cf09` (KU 4, cf gate 0.9).
  - If it passes VERDICT2, the x1.12 is real. If it fails, the FAST route at 12 GB is closed.
  - The other step 0 / 0b arms were dropped before running. topk3 / topk4 stay registered; they are harmless.
- **The plan is now levers 1 and 3 on the exact mode** (49.9 tok/s on the shared text, 40.8 on full outputs).
  [CORRECTED Fri 13:2x: those are t80ck2's FAST numbers; the exact loop runs ~18.5.]
  - Lever 1: kept per check. The first test is Breakthrough's CPU replay on Friday: copy spans and today's kept per check on
    them, over T3R2's prompts.
  - Lever 3: bytes and compute per check. The rank-0 residency replay, and Leap's kernels.
- **Map screens for lever 3** (lossy byte cuts on the exact mode):
  - `topk8` (ranks 8-9 dropped, ~-20 % of tail loads) is already in Friday's queue (36 windows).
  - One-pass QOM-KEEP at 12 GB's 47-slot pool (the 45 GB FAST rule, no exact pass) is not queued. Its code flips should
    fall between QOM-KEEP at 147 slots (1.07 %, PASS) and P1-K2 with no residents (`p1_oea_k2`, 2.50 %, KILL). A guess
    is 1.6-2.2 %, likely KILL; its mask is buildable with the keepset replay if lever 3 needs it.

## Update Fri 04:0x: lever 1's copy part (CPU replay; SCOREBOARD "RESULT, LEVER 1 CPU REPLAY")

- **Copy at the check's own width is KILLED.** On Fit12's c51 logs (16 prompts, 343 checks reproduced exactly), the
  oracle tree gains only +4.6 % kept per check, and every realizable switch is at or below today's 10.08. DFlash2
  already drafts copied text up to its 15-draft cap.
- **The lead is a wider copy check.** Use a 32-row copy check when the last 8 tokens match an earlier 8-gram (CW8), and
  the drafter's 16-row check otherwise.
  - Kept per check: 12.21 (x1.21). Rows per kept token: 1.64 vs 1.59 (+3 %).
  - It pays only if a 32-row check costs well under 1.21x a 16-row one in the exact mode. At 1.1x it gives ~x1.1
    [P, halved].
  - The next test is Fit12's: time 16- vs 32-row checks in t80ck2, on copy rows. [Priced Fri 13:2x without a timing:
    KILL at 12 GB. See "Update Fri 13:2x".]
- **The drafter side of lever 1 is parked.** MTP Head reports P-OFF has no effect. So lever 1 now rests on the wide copy
  check and INERT.

## Update Fri 13:2x: CORRECTION, t80ck2 is FAST; the 12 GB exact loop runs ~18.5 tok/s (Fit12, `~/Fit12/PLAN_12GB_90_fit12.md` "Lever 3 RESULT")

- **t80ck2 is not the exact mode.** It commits pass-1-certified rows (top-2 pass 1, θ 0.8, cf 0.9) without the exact
  check, and the map scored top-2 at θ 0.8 a code KILL (1.63 %). The exact loop (`ex`: DFlash2 plus exact calls of <= 4
  rows, C_m 47 / C_l 32) runs 18.4-18.9 tok/s on HumanEval [K]: 18.9 and 18.42 quiet, 18.58 under today's share lock.
  "Update Thu 19:5x" and "Update Thu 20:0x" called t80ck2 the exact mode; both are marked CORRECTED.
- **So the two-pass FAST route is open again.** Every two-pass point runs ~2x the exact loop, and the 12 GB base is the
  fastest one that passes code:

  | point | tok/s [P, Fit12's replay, full outputs, cf 0.9] | code on the map |
  |---|---|---|
  | P1-K1, θ 0.8, KU 4 (t80ck1u, T3R2's form) | 45.8 | Saturday's step 0 arm |
  | P1-K1, θ 0.9 | 41.0 | re-added to Saturday's step 0 |
  | P1-K1, θ 0.95 | 37.9 | re-added to Saturday's step 0 |
  | top-2, θ 0.95 | 35.9 | 1.22 % all-row, PASS (on record) |

  From 36-46 tok/s, 90-130 needs x2.0-3.6 (not the x1.8-2.6 written above). The stack below assumed a 50-57 base; on
  36-46 it gives ~55-105 [G], so 130 still needs the SSD wait hidden.
- **Lever 3, Leap's Friday switches ported** [K, rough, share lock, 14 clean prompts]: the dense pass-1 kernel (dn1)
  PAYS, x1.054 (pass 1 157 -> 140 ms per run); att_single x1.013 (BETWEEN); all three x1.047; the drafter port saves
  ~3 ms per check. The check stays bytes-bound (~1.47 GB per check).
- **Lever 1's wide copy check (CW8) is KILLED at 12 GB** [P, Fit12's arithmetic]:
  - A 32-row pass 1 overflows the 47-slot pool. Top-4 at 16 rows, also 64 picks per layer, already aborted at 54
    experts. So the check needs two 16-row pass-1 calls (+~157 ms).
  - That is ~398 ms per check for ~11.3 kept: ~35 ms per kept token against 25.9 today (x0.74).
  - Only a form where copy rows skip pass 1 could pay. It is unpriced and not routed. Lever 1 now rests on INERT alone.

## Step 0: the 57 is not yet a code-valid base

T3R2 certifies at θ 0.8 behind a P1-K1 pass 1: only the block's rank-0 union, at most 4 experts per row. The map's
certificate sweep (`~/FableGod/decision_map/out/CERT_read.txt`) [K] uses 4-bit experts and the 12 code windows; the code
line is 1.34 %.

| pass 1, certificate | code flips | rows certified | blocks needing pass 2 |
|---|---|---|---|
| top-2, θ 0.99 | 1.16 % | 73.4 % | 77 % |
| top-2, θ 0.95 | 1.22 % | 82.3 % | 63 % |
| top-2, θ 0.9 | 1.49 % | 85.4 % | 60 % |
| top-2, θ 0.8 | **1.63 %** | 88.4 % | 55 % |
| top-2 at 3 bits, θ 0.8 | 1.75 % | 88.2 % | 57 % |
| **top-5, θ 0.9** | **0.92 %** | **89.8 %** | 51 % |

- **EvalSuite T0 = KILL.** T0 is the top-2 two-pass at 12 GB, run on HumanEval+ at θ 0.8: 80.0 vs 91.7, McNemar 7 / 0.
  The map's matching arm, top-2 at θ 0.8, is over the line, so the map predicted the KILL. T0's IFEval ran at θ 0.95 and
  failed too (-6.0). The map's IFEval for that arm is 6.48 %, against the chat route's 4.30 % PASS.
- **The damage comes through the context.** At θ >= 0.8, certified rows flip only 0.0-0.3 %. The uncertain rows computed
  exactly after them flip 4-12 % (17 % at θ 0.7). So a cheaper pass 1 means a worse context.
- **What follows:**
  - P1-K1 at θ 0.8 is probably over the line too.
  - A better pass 1 beats a lower θ. Top-5 at θ 0.9 passes with more rows certified than top-2 at θ 0.8.
- **Step 0** [Breakthrough]: map screens of t80ck1u's own certificate form (P1-K1, θ 0.8, cf gate 0.9) and of the
  code-valid candidates (θ 0.95; a top-4/5 pass 1). The arms are built as rank masks with the keepset replay's blocks,
  run on the code windows, and queued for the next night slot.
- **The base** for the factors below is the fastest form that passes code. It may be under 57.

## The top 3 at 12 GB

Factors are on the per-check time, with quality held.

### 1. Kept per check on code: x1.2-1.4 [P, halved]

- **Levers**: the copy draft, INERT and the drafter.
- **Why it ranks first**: it is quality-neutral, because the verify decides every token. Every part of the 12 GB check is
  paid per check, so kept per check passes through almost 1:1 (Fit12: +20 % kept is x1.15-1.2 after the extra rows'
  misses).
- **Evidence**:
  - HumanEval answers re-emit the signature and docstring: copy rows are 61-68 % of tokens. Fable FULL's replay gives
    x1.4-1.6 [P].
  - INERT: x1.15 on the 12 HumanEval keys [G].
  - The drafter ceiling (D98): kept 12-12.5 vs 10.5 [G].
- **First test**: a CPU replay over T3R2's 16 HumanEval prompts gives the copy spans and today's kept per check on them.
  Then a T-A1-style timing probe of a wide copy pass at 12 GB in Fit12's harness.
- **Owners**:
  - Breakthrough: the replay;
  - Fable FULL / GOAL: their builds;
  - MTP Head: P-OFF / D98 on the T3R2 prompts.

### 2. Exact-pass diet at a code-valid certificate: x1.15-1.3 [G]

- **The idea** (Fit12's lever 1): the exact passes are 105 of 261 ms. Two routes: a certificate calibrated on code, and a
  mid pass for moderately unsure rows before REF2-T.
- **The constraint**: after step 0 the gain must come from a better pass 1 or the mid pass, not from a lower θ.
- **First test** [Fit12, CPU]: from T3R2's logged checks, why rows go exact and what each threshold or pass-1 width would
  save. Then the map screen of each candidate, then a T3R2-style A/B.

### 3. Bytes and compute per check (exact): x1.1-1.25

- **Why it helps**: the wait is exposed at ~9.6 GB/s, so every byte cut comes off the wait.
- **Levers**:
  - the 3-bit pass 1: t80ck1um3 is x1.105 on the shared text [K]; its quality is unscreened (3-bit top-2 at θ 0.8 is
    1.75 % on the map);
  - rank-0 residency: a CPU replay of pass 1's 341 misses through an LRU of the 12 GB slots (Breakthrough's keepset
    replay);
  - Leap's kernels on pass 1's 71 ms of compute.
- **First test**: the residency replay [Breakthrough, CPU]; Leap's FAST-NAX arms on the T3R2 harness.

### The stack

x1.5-2.3 [G] on the code-valid base. That is 75-130 tok/s if the base is 50-57.

130 needs the SSD wait hidden by overlap: reads issued before routing is final, or late terms. The SSD already runs at
70-80 % of its ceiling during the waits, so cutting bytes alone cannot get there.

## Not in the top 3

- **The split** (QOM-KEEP's resident keep off on sure rows): ~x1.0 at 12 GB. P1-K1 already posts the union only, with no
  resident keep, so it is the split's extreme form.
- **Never-wait** (FAR): 12 GB's dose is ~50 misses per kept token, against 6.6 at C 208 (~7.5x). Saturday's tier 1 at
  the 45 GB pools gives the dose-response first.
- **Prefetch**: the frugal form is KILL (recall 0.61-0.73, x0.90).
- **BYTES10 beyond the 3-bit pass 1**: <= x1.2 priced at 12 GB [G].

## Who does what (started Thu 19:4x)

- **Fit12**: lever 2: the CPU test on T3R2's logged checks, then the A/B.
- **Breakthrough**:
  - this plan;
  - step 0's code screens: t80ck1u's form first;
  - lever 1's replay;
  - the rank-0 residency replay.
- **Leap**: pass-1 kernels at 12 GB.
- **MTP Head**: kept per check with P-OFF / D98 on the T3R2 prompts.

## Update Sat 04:2x: STEP 0 RESULT, the 12 GB base is T3R2's own form (45.8 tok/s [P])

- Saturday's step 0 ran on the map (12 code windows each, VERDICT2 code). All three P1-K1 KU 4 cf 0.9 forms PASS [K]
  (SCOREBOARD "Sat 04:2x: STEP 0 RESULT"; data/map_rest_read.txt):

  | point | tok/s [P, Fit12's replay] | code on the map, all-row [non-tie] | slack to the line |
  |---|---|---|---|
  | P1-K1, θ 0.8, KU 4 (t80ck1u, T3R2's form) | 45.8 | 1.31 % [0.78], PASS | 1 flip / 2 flips |
  | P1-K1, θ 0.9, KU 4 | 41.0 | 1.16 % [0.66], PASS | 6 / 6 |
  | P1-K1, θ 0.95, KU 4 | 37.9 | 1.10 % [0.60], PASS | 8 / 8 |

- **By the pre-registered rule, the 12 GB base is T3R2's own form at 45.8 [P].** The fastest code-passing point before
  was top-2 θ 0.95 at 35.9 [P], so this is +28 % [P].
- Its margin is one all-row flip, so θ 0.9 (41.0) is the fallback if a re-sample or EvalSuite tips it.
- The plan's factors now start from 45.8. 90-130 tok/s needs x1.97-2.84 (the x2.0-3.6 above assumed a 36-46 base).
- Next, before adoption:
  - Fit12's measured run of t80ck1u on full outputs (16 HumanEval);
  - EvalSuite same-scores on code. It is FAST, not exact.
