# Does the sandbox predict the 125B? Transfer checks (Sat 14 Nov 2026)

Each row compares a sandbox result with a known 125B result. A mismatch is as useful as a match: it marks where the
sandbox stops predicting. Sandbox numbers are MBPP unless marked; 125B numbers are t30b-based (BigMoE / TrainLab).

## Checks so far

| # | check | 125B | sandbox | verdict |
|---|---|---|---|---|
| 1 | the carried stand-in loses accuracy with depth | MBPP 94.01 -> 72.13 at 23-30 (-21.9 pt) | 88.54 -> 57.32 (-31.2 pt; steeper) | MATCH (shape; magnitude larger here) |
| 2 | the TRUE deep state keeps accuracy flat | 93.63 at 23-30 | 88.35 at 23-30 (depth 0 88.54) | MATCH |
| 3 | early-layer maps worse than the stale chain, last-layer maps near the oracle | {5} 41.66 < 72.13; {45} 91.61 | tap 2: 47.75 < 57.32; tap 22: 87.30 | MATCH |
| 4 | t30f's feature loss (stand-in -> true state, wd space) | NO EFFECT | NO EFFECT (-0.26 pt, CI -0.71..+0.16) | MATCH |
| 5 | CAPACITY: a 2x zero-init branch at the head's lr | see below | pending | pending |
| 6 | anchor-only fused taps (TAPSQ2) | see below | pending | pending |

## 5 and 6: pre-registered Sat 14 Nov 22:2x, before they run

Format: the FAST multi-arm screen, one process, three arms on the same slices in the same order: ctl (no lever), cap2x,
tapsq. From s30o with its Adam state, lr 1e-5 (20-step warm-up), rollout depth 12, **200 steps** (the 125B capacity
check's length), pool slices. Read-outs: MBPP 50 + HE 30, at depth 15 (bands 0-14) AND depth 31 (bands to 23-30, past
the depth-12 rollout: the format difference from the 125B, which trained 30 deep). Smoke-tested 22:1x (peak 5.4 GB for
3 arms; runs after the depth-12 calibration, with MEM 5.4 GB declared and only when no other memory-heavy job runs).

**5. CAPACITY** (`levers/cap2x.py`: a zero-init SwiGLU branch beside the head's MLP, width 6144 = 2x the dense MLP, own
Adam at the head's lr). 125B (BigMoE 24e7758, as the orchestrator quoted it): a small depth-GROWING gain, MBPP 0: -0.09,
7-10: +0.44, 15-22: +1.00, 23-30: +1.20, with an accept-until-fail training loss 1.6 % WORSE.
Lines (cap2x minus ctl, MBPP-50, depth-31 read-out):
- shape matches if delta(23-30) - delta(0) >= +0.3 pt, delta(23-30) > 0, and delta(15-22) >= delta(7-10) - 0.2;
- loss matches if cap2x's mean training loss over the last 100 steps is above ctl's (worse);
- MATCH = both; PARTIAL = one (say which); MISMATCH = neither. Depth-15 read-out and HE-30 reported beside it.

**6. Anchor-only fused taps** (`levers/tapsq.py`: a zero-init Linear of the five raw taps, RMS-normalized and
concatenated, added to the fused input of every true pair row, never at drafted rows; own Adam at the head's lr).
125B (TAPSQ2, BigMoE 068dc9b): KILL with a small loss, MBPP 23-30 71.43 vs C 72.06 (-0.63), windows 35 better / 37
worse, KL 1.083x. (The 125B fused its taps through DFlash2's trained fc and ran 80 steps; here raw taps, 200 steps.)
Lines (tapsq minus ctl):
- MATCH = no significant gain: MBPP-50 23-30 (depth 31) delta < +0.5 pt or its CI includes 0, and the 7-14 band
  (depth 15) likewise;
- MISMATCH = a significant gain (>= +0.5 pt with the CI above 0) in either band.
- The training-loss ratio is reported as information only (it is survival-weighted, so not comparable across arms).

**Format change (22:4x, before any of 5 / 6 ran):** the constant-lr format drifted the control (LONG_LINES.md), so 5 and 6
run with calibration v2's setup: pool2 (one pass, no repeats) and a warm-up to 1e-5 then a cosine to 1e-6 over the 200
steps, from s30o's Adam state. Lines unchanged. Added line: the in-batch control must hold (MBPP-50 7-14 not down by more
than 1.5 pt vs s30) before 5 / 6 are read; otherwise they are reported as provisional.

The budget caveat applies: if the calibration shows the depth-12 control does not measurably move by 200 steps, these
two checks ran below the screen budget; the comparison with the 125B's equally short runs still stands, and it is noted.
