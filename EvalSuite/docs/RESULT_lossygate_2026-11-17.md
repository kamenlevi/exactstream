# RESULT: the close-call night (Tue 2026-11-17 00:56-04:34) — close-call at delta 1 (ccx45cc1) PASSES tier 1 in both orders (pooled code -1.14 pt, p 0.26); delta 2 (ccx45cc1d2) FAILS its forward (MBPP -6.00 pt; pooled code -4.17 pt, p 0.013), so its reverse did not run

Pre-registered in docs/PREREG_lossygate_2026-11-15.md: amendment 3 (close-call at delta 1 against the cached lossy-45
base lx45cc1), 3a (the RUNAWAY guard and the speed read-out) and 3b (the delta-2 arm). Runner 64 (tools/day_runner64.sh,
sha256 5cab78942ae9501c, as registered) ran every step. The chain was tools/night1117_chain.sh at 8698a4d (sha256
d434e7f7866d2bc2), armed Mon 06:52:13 as pid 52881. Its log is results/night1117_chain.log.

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
