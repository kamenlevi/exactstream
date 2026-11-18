# RESULT: the close-call night and the tier-0a re-run (Tue 2026-11-17 00:56-05:06) — close-call at delta 1 (ccx45cc1) PASSES tier 1 in both orders (pooled code -1.14 pt, p 0.26); delta 2 (ccx45cc1d2) FAILS its forward (MBPP -6.00 pt; pooled code -4.17 pt, p 0.013), so its reverse did not run; the tier-0a calibration is COMPLETE (A_line 0.9775, F_line 0.00 per 1,000; the adoption check PASSES)

Pre-registered in docs/PREREG_lossygate_2026-11-15.md: amendment 3 (close-call at delta 1 against the cached lossy-45
base lx45cc1), 3a (the RUNAWAY guard and the speed read-out) and 3b (the delta-2 arm). Runner 64 (tools/day_runner64.sh,
sha256 5cab78942ae9501c, as registered) ran every step. The chain was tools/night1117_chain.sh at 8698a4d (sha256
d434e7f7866d2bc2), armed Mon 06:52:13 as pid 52881. Its log is results/night1117_chain.log. The tier-0a re-run is
amendments 2, 2b and 2c, with its lines in 2d.

## How it ran [K]
- **The start:** 00:56:12, on Leap's E1-E4 marker (E14 ended 00:55:23, rc 0), written after E14's own start line and
  read twice, 30 s apart. NO-SWAP was 44 GB at the start; the gate was 43 GB per process (CARD_MEM 39).
> (a quote from the project owner, kept private)
  and no runner yield. Swap used was 7515 MB at 01:54 and 7499 MB at 04:37, so the +2 GB guard never fired.
- **Timeline:**

  | Step | Run | Time | Outcome |
  |---|---|---|---|
  | 1 | ccx45cc1 forward (card_ccx45cc1_t1code), 19 processes | 00:56-02:08 | rc 0, PASS |
  | 2 | ccx45cc1 reverse (CARD_ORDER=rev, card_ccx45cc1_t1code_rev), 19 processes | 02:08-03:22 | rc 0, PASS, both orders pooled |
  | 3 | ccx45cc1d2 forward (card_ccx45cc1d2_t1code), 19 processes | 03:22-04:34 | rc 0, FAIL (MBPP, pooled) |
  | 4 | ccx45cc1d2 reverse | - | not run: the forward did not pass (amendment 3's step 2, as 3b applies it) |

- The end marker (results/CHAIN_1117_END, "done:") was written at 04:34:42. The tier-0a follower (pid 45029) started
  on it at 04:36:05.

## Close-call at delta 1 (ccx45cc1): PASS, both orders pooled [K]
lx45cc1's line plus R57_RELAX_DELTA=1 and R57_RELAX_DELTA_SETS=humaneval,humanevalx,mbpp (env blob 46bdf06b), on the
cc1 snapshot, tier 1 code scope, against the cached base (card_lx45cc1_t1code / card_lx45cc1_t1code_rev).

| Line | Forward | Reverse | Both orders pooled |
|---|---|---|---|
| HumanEval+ | 90.9 % vs 92.1 %, -1.22 pt, p 0.6250 (1 / 3) | 92.1 % vs 93.9 %, -1.83 pt, p 0.3750 (1 / 4) | n 328: 91.5 % vs 93.0 %, -1.52 pt, p 0.1797 (2 / 7) |
| MBPP | 76.0 % vs 77.0 %, -1.00 pt, p 1.0000 (3 / 4) | 77.0 % vs 77.0 %, +0.00 pt, p 1.0000 (2 / 2) | n 200: 76.5 % vs 77.0 %, -0.50 pt, p 1.0000 (5 / 6) |
| **Pooled code, line > -3 pt and p >= 0.05** | **n 264: -1.14 pt, p 0.5488 (4 / 7): PASS** | **n 264: -1.14 pt, p 0.5078 (3 / 6): PASS** | **n 528: -1.14 pt, p 0.2632 (7 / 13): PASS** |
| Runaway guard (pooled code) | 6 / 2, p 0.2891: PASS | 3 / 2, p 1.0000: PASS | 9 / 4, p 0.2668: PASS |
| Cap hits, arm vs base | HE+ 8 vs 6, MBPP 10 vs 8 | HE+ 6 vs 4, MBPP 8 vs 8 | HE+ 14 vs 10, MBPP 18 vs 16 |
| IFEval tokens identical to the base | 100 / 100 | 100 / 100 | 200 / 200 |
| Interim (n 132 of 264) | z -0.38: CONTINUE | z +0.00: CONTINUE | |
| **Verdict** | **PASS** | **PASS** | **LOSSY tier1 (CODE-ONLY mode) [2 orders pooled]: PASS** |

Counts in brackets are arm only / reference only (exact McNemar p).

- **What it licenses:** tier 1's pooled line is a screening line. This verdict licenses tier 2 (the card gate);
  adopting close-call for a mode stays Kamen's call after tier 2.
  - **Next (NewIdeas, received Tue 04:4x):** tier 2 is not queued for close-call alone. LOSSY_GATE section 4 runs the
    card gate once, for the variant that would ship: lossy-45 + CC(1) + G1's pick, plus G2 if it passes.
  - Once G1 / G2 settle (Wednesday evening at the latest), that whole stack gets one tier-1 run in both orders
    (~2.5 h), then the card gate (5-7 h at 45 GB).
  - NewIdeas finds the slot with the orchestrator. Nothing is armed.
  - NewIdeas' ladder v4 already prices CC(1) at +4.0 % kept (16.91 vs 16.26, tree-32). That is inside the measured clean
    +2.9 % to +5.7 %.
- **Engagement:** 1,091 close-call accepts in the forward and 993 in the reverse, over 264 code runs each. 100 and 97
  code runs read 0, and every process summed > 0 (CARD_ENGAGE_MODE=any). There were none on IFEval.
- **Speed (information, never a line):** kept tokens per check (tools/lossy_speed.py), arm vs base:

  | Set | Forward, all | Forward, clean | Reverse, all | Reverse, clean |
  |---|---|---|---|---|
  | humaneval | +0.4 % | +3.5 % (n 56) | -4.0 % | +0.5 % (n 58) |
  | humanevalx | +5.9 % | +8.7 % (n 100) | +5.9 % | +8.7 % (n 100) |
  | mbpp | +10.7 % | +3.0 % (n 88) | +9.4 % | +7.1 % (n 90) |
  | **code** | **+6.2 %** | **+5.7 % (n 244)** | **+4.5 %** | **+2.9 % (n 248)** |

  - "Clean" means the answers uncapped and unlooped in both arms. Amendment 3a says a speed claim quotes the clean
    number, so: **+2.9 % to +5.7 % kept tokens per check on code.**
  - Amendment 3 quoted Breakthrough's pricing as upper bounds: HE+ +21.8 %, MBPP +23.1 %. The measured lift is about
    a quarter of that.
- **Validity, each order:**
  - the pins matched at the start: trees cc1 85a88a46, frozen_cc1 ec97bdbb and bigmoe d92cdb4b; screen efc1f9b6f531;
    .so ff69fa24a745; env blob 46bdf06b;
  - all 19 processes ran, and the 41 outside modules were equal to the first process's on all 18 later processes;
  - 364 rows (264 code + 100 IFEval), 364 keys.
- **Information, forward vs reverse token identity** (the arm's own two orders):
  - identical: humanevalx 104 / 104 and IFEval 100 / 100, the sets outside the config's QOM sets;
  - differing: humaneval (31 / 60 identical) and mbpp (32 / 100), its QOM sets.
  - This is the same pattern as the base's two orders (32 / 60 and 35 / 100). The order effect sits in the QOM sets.
  - So the pooled n 528 counts humanevalx's 104 pairs once per order, with the same tokens both times. The base's
    pooled reference does the same, as registered.

## Close-call at delta 2 (ccx45cc1d2): FAIL, forward [K]
ccx45cc1's line with R57_RELAX_DELTA 1 -> 2 (env blob a5cb28ef), against card_lx45cc1_t1code.

| Line | Forward |
|---|---|
| HumanEval+ (n 164) | 89.0 % vs 92.1 %, -3.05 pt, p 0.1250 (1 / 6): PASS |
| MBPP (n 100) | **71.0 % vs 77.0 %, -6.00 pt, p 0.1094 (2 / 8): FAIL** (decision 4 needs a delta above -5 pt) |
| **Pooled code (n 264), line > -3 pt and p >= 0.05** | **-4.17 pt, p 0.0127 (3 / 14): FAIL** |
| Runaway guard (pooled code) | 7 / 3, p 0.3438: PASS (cap hits HE+ 7 vs 6, MBPP 10 vs 8) |
| IFEval tokens identical to the base | 100 / 100 |
| Interim (n 132 of 264) | z -1.26: CONTINUE |
| **Verdict** | **LOSSY tier1 (CODE-ONLY mode): FAIL (MBPP, pooled)** |

- **The reverse did not run,** as registered: a forward FAIL ends the arm.
- **The losses are not runaways:** the correctness discordants are 3 / 14, but the runaway guard reads only 7 / 3.
  On the 2B, delta 2 failed through runaway answers (cap hits 26 -> 50; amendment 3a). Here the wrong answers end
  normally.
- Amendment 3b called delta 2 "plausible on the 125B, not established" from the teacher-forced dose. It does not hold
  at tier 1.
- **Engagement:** 1,993 close-call accepts over 264 code runs (1.8x delta 1's forward); 79 code runs read 0.
- **Speed (information; the arm failed):** code +9.2 % on all answers, +5.8 % clean (n 243).

## The tier-0a calibration re-run (rpl2, step 3; 04:36-05:06): COMPLETE; A_line 0.9775, F_line 0.00 per 1,000; the adoption check PASSES [K]
The follower was tools/night1117_t0a.sh (0ca9f08, sha256 b106d07f1583e9cc), armed Mon 08:40:20 as pid 45029. Its log
is results/night1117_t0a.log.
- **The start:** 04:36:05, on night1117's "done:" marker, read twice 30 s apart. NO-SWAP was 48 GB; the cutoff was
  06:36 (start + 2 h).
- **The arms:** amendment 2b's three, on the rpl2 pins. Each ran a REPLAY pass over exact-45's own rows
  (card_x45rpl1_t0code) on the merged exact margins (results/card/replay_margins_x45rpl1.json), then its mini-panel
  against the exact references.

| Arm | Time | Known | A (drafted rows) | Flips by exact margin < 0.5 / 0.5-1 / 1-2 / >= 2 | F_hi per 1,000 | Mini-panel (n 40) |
|---|---|---|---|---|---|---|
| p2x45rpl2 | 04:36-04:42 | PASS (the 45 GB P2 gate) | 0.9825 (10,135 / 10,315) | 147 / 30 / 3 / 0 | 0.00 | 2 / 5, net loss 3: GO |
| m1bx45rpl2 | 04:42-04:53 | PASS (the card gates) | 0.9854 (10,164 / 10,315) | 124 / 26 / 1 / 0 | 0.00 | 0 / 4, net loss 4: KILL |
| lossycalx45rpl2 | 04:53-05:06 | deliberately lossy | 0.9491 (9,790 / 10,315) | 206 / 131 / 134 / 54 | 5.24 | 0 / 9, net loss 9: KILL |

- **The lines** (amendment 2d, computed as amendment 2 registered them, from the passing calibrators p2 and m1b):
  - A_line = 0.9825 - 0.5 pt = **0.9775**;
  - F_line = 2 x 0.00 = **0.00 per 1,000**.
- **The adoption check: PASS.** lossycal KILLs at its mini-panel and falls outside both lines: A 0.9491, 2.84 pt under
  the line, and F_hi 5.24.
- **So the lines are adopted for tier 0a:** KILL if A < 0.9775 or F_hi > 0.00 per 1,000.
  - Until tools/lossy_d4.py reads them, they are applied by hand from card_<tag>_replay.txt.
  - A REPLAY run needs a snapshot with rpl2's `_gate_block` fix: t1c1, g1a and g1b have it, cc1 does not.
- **What separates the arms:**
  - The two known-good configs flip only near-ties: 0 flips at >= 2 nats in 10,315 drafted rows each, and 3 and 1 at
    1-2 nats.
  - The deliberately lossy arm flips confident tokens: 54 flips at >= 2 nats and 134 at 1-2.
- **Information:** m1b passed the card gates, yet its code mini-panel against exact sits at the KILL line (net loss 4:
  0 / 4, exact p 0.125). p2 reads 3. Against exact rows, the mini-panel line sits at the known-good configs' level. The
  line stays as registered.
- **Validity, each arm:**
  - the pins matched at the start: trees rpl2 7038e9a8, frozen_rpl1 db72c3c1 and bigmoe d92cdb4b; screen
    d74d8b119e34; .so ff69fa24a745; the env blobs as registered;
  - 4 processes each (2 REPLAY + 2 mini-panel), with the outside modules equal;
  - 40 of 40 keys, and the prefill picks all equal.
- **Memory:** p2's processes peaked at 44.2 GB, above its 43 GB gate (NO-SWAP was 48 at its start). m1b and lossycal
  peaked at 42.9 GB, under their 47 GB gate. Swap was flat (7499 MB at 05:07).

## Memory, and a correction to Monday's RESULT [K]
- **This night:** process lifetime peaks were 42.4-43.0 GB in the delta-1 runs and 42.4-43.1 GB in the delta-2 run,
  at the 43 GB NO-SWAP gate. The warm-ups peaked at 42.2 GB. Swap did not grow.
- **CORRECTED (Tue 04:4x), docs/RESULT_lossygate_2026-11-16.md:**
  - That doc says the base lx45cc1's "lifetime peak was 42.2 GB in both orders". That figure was the warm-ups' peak.
  - Its processes peaked at 42.4-43.1 GB (max 43.1 GB in both orders).
  - The 1c-tiers full runs show the same: lx45t1c1 B1 and t1cx45t1c1 V1 peaked at 43.0 GB.
- **What it means:**
  - The short key tests read 42.3-42.4 GB. That is 0.6-0.7 GB under the full runs, as a smoke peak usually is.
  - No verdict changes. The gate is a start condition (free NO-SWAP >= 43 GB), and swap stayed flat on every night it
    was used.
  - The PREREG lines that quote "lx45cc1 (peak 42.2 GB)" carry a dated pointer to this correction.
- Read-out: results/lossygate/peaks_2026-11-17.py and .txt (all seven full runs, per process).

## Information, Wed 08:0x: the CC(1) arm and its base, scored against the EXACT references [K]
This is post hoc, on the stored answers: no new runs, CPU graders under a share lock. Monday's registered gate compared
ccx45cc1 with its lossy-45 base (amendment 3). This scores both against the exact references that blob 1's tier 1 uses
(ref2tnax, ref2tnaxn.r3, ref2tnaxn.r2), both orders pooled, with tools/lossy_d4.py tier1 in code scope. The lines were
applied after the data existed, so this is information, not a verdict.

| Arm (both orders) | HE+ (n 328) | MBPP (n 200) | Pooled code (n 528) | Runaway guard (pooled) | IFEval (n 200) | By the lines |
|---|---|---|---|---|---|---|
| lx45cc1 (lossy-45) | 93.0 vs 92.7 %, +0.30 pt, p 1.00 | 77.0 vs 79.0 %, -2.00 pt, p 0.45 | -0.57 pt, p 0.70 | 16 / 9, p 0.23: PASS | +1.00 pt | PASS |
| ccx45cc1 (+ CC(1)) | 91.5 vs 92.7 %, -1.22 pt, p 0.29 | 76.5 vs 79.0 %, -2.50 pt, p 0.33 | -1.70 pt, p 0.11 | 14 / 2, p 0.0042: FAIL | +1.00 pt | FAIL (runaway) |

- **Runaways:** cap hits on HE+ / MBPP are 8 / 12 for exact, 10 / 16 for lossy-45 and 14 / 18 with CC(1). Loops are
  0 / 0 for exact and 1 / 3 for both arms.
- **The reading:** against exact, CC(1) tips the runaway guard. Against lossy-45 (Monday's registered comparison), it did
  not (pooled 9 / 4).
- **Why it matters for Saturday:**
  - The fallback headline (ccx45cc1 + bitwise flags) carries Monday's PASS only against lossy-45.
  - Blob 1 carries CC(1) on the tree walk and is judged against exact.
- **Determinism:** Monday's two orders gave different answers on 29 / 60 humaneval and 68 / 100 mbpp keys (QOM's sets),
  and identical ones on 104 / 104 humanevalx and 100 / 100 ifeval keys.
- **Data:** results/lossygate/vsexact_lx45cc1_2026-11-18.txt and vsexact_ccx45cc1_2026-11-18.txt.

## Data
- results/night1117_chain.log and results/CHAIN_1117_END; results/day64_ccx45cc1_runner.log,
  day64_ccx45cc1_rev_runner.log and day64_ccx45cc1d2_runner.log.
- results/CARD_ccx45cc1_DONE, CARD_ccx45cc1_rev_DONE and CARD_ccx45cc1d2_DONE.
- The verdict files: card_ccx45cc1{,_rev}_{tier1code,interimcode}.txt, card_ccx45cc1_rev_tier1pooledcode.txt and
  card_ccx45cc1d2_{tier1code,interimcode}.txt.
- The rows: card_ccx45cc1_t1code, card_ccx45cc1_t1code_rev and card_ccx45cc1d2_t1code (decode.jsonl).
- Speed: results/lossygate/speed_ccx45cc1_2026-11-17.txt, speed_ccx45cc1_rev_2026-11-17.txt and
  speed_ccx45cc1d2_2026-11-17.txt.
- Per process: results/card/es_cdccx45cc1_170056_*, es_cdccx45cc1_rev_170208_* and es_cdccx45cc1d2_170322_* (logs,
  manifests, checks, runs, tokens, screens).
- The tier-0a re-run:
  - results/night1117_t0a.log and results/T0A_1117_END;
  - results/day64_{p2,m1b,lossycal}x45rpl2_runner.log and CARD_{p2,m1b,lossycal}x45rpl2_DONE;
  - card_<tag>_replay.txt and card_<tag>_tier0code.txt;
  - the rows card_<tag>_t0code and card_<tag>_t0code_rpl;
  - results/card/replay_rows_<tag>.jsonl, lossy_keys_<tag>_t0code.txt, lossy_ref_keys_<tag>_t0code.txt and
    es_cd<tag>_*.
