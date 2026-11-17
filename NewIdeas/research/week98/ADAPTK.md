# ADAPTIVE K: choose the check's size per check (NewIdeas, Tue 17 Nov 01:0x)

**Verdict: PASS (screen), exact, a lead on copy-heavy code.** A one-parameter rule chooses the tree size per check.
Best-first drafting keeps popping nodes while the next one's path probability is >= tau. The rows are then rounded
up to an available check size (16 or 32 rows today).
- With today's two kernels, at a 0.40-0.92 ms per-level draft cost (counted), HE+ gains **+8.0-10.3 %** over the
  best fixed K, using the other set's tau. MBPP gains ~0 % and never loses (-0.2 to +0.2 %).
- With one-pass check forms up to 128 rows: HE+ +10.3-15.6 %.
- The hindsight ceiling (oracle DP): HE+ +17-34 %, MBPP +9.5-12.4 %.

## Window spread (adaptk_windows.py, Tue 01:09 by date): the HE+ gain is BROAD

At frozen tau 0.9 (MBPP's tau, applied to HE+), with KERN2:
| L (ms per level) | HE+ pooled gain | 95 % window bootstrap | windows up |
|---|---|---|---|
| 0.66 | +8.6 % | [+6.6, +10.7] | 24 / 24 |
| 0.40 | +9.2 % | [+7.2, +11.6] | 24 / 24 |
| 0.92 | +8.0 % | [+6.2, +10.1] | 24 / 24 |

- Per window at L 0.66: min +2.4 %, median +9.8 %, max +22.4 %.
- ONEPASS at L 0.66, tau 0.9: +11.0 % [+7.8, +14.7], 24 / 24 up.
- MBPP: +0.2 % [0.0, +0.3]; 92 of 100 windows are flat.
- The KILL line (+3 %, HE+, KERN2, L 0.66) sits under the CI's lower bound.

**Decision (the orchestrator, received Tue ~01:08 by date; its message said 01:4x): GATED BUILD this week.**
1. The fresh-cell re-read at the frozen tau, against the KILL line.
2. M4 (Tue night) certifies the 16-row tree verify exact on the full model.
3. BigMoE builds a 32-row tree verify as an opt-in switch, bit-tested vs one-token decoding. Leap reviews the C32one
   hook.
4. It joins Saturday only if it is bitwise and a C T T C engine A/B passes by Fri 12:00. Otherwise after Saturday.

## Why it works

- path_p is calibrated. The expected kept, 1 + the sum of path_p over the tree's pops, matches the realised mean:
  - HE+ 12.83 vs 12.82 at 16 rows, 21.25 vs 21.01 at 32;
  - MBPP 8.22 vs 8.18 at 16 rows.
- path_p is non-increasing in pop order on 100 % of anchors. So "include a node iff its path_p >= lam x the marginal
  cost of its row" is a threshold rule.
- HE+ has long confident stretches (docstring copies, where K* = 32 already tied). There the rule takes 32 rows, and
  16 elsewhere. MBPP's drafter is rarely confident past 15 pops (32 rows on 12 of 2,080 checks).

## Numbers (exact engine costs from E14; tok/s)

Setup: t30b 127-node dumps (n127k8), certify98 walk, overhead 4 ms; draft cost = L ms per tree level, counted.

| variant | L | HE+ fixed best | HE+ policy (cross tau) | MBPP fixed best | MBPP policy (cross tau) |
|---|---|---|---|---|---|
| KERN2 (16 / 32 rows, today's kernels) | 0.40 | 154.2 | 170.0 +10.3 % (168.3 +9.2 %) | 103.7 | 103.9 +0.2 % (103.7 -0.1 %) |
| KERN2 | 0.92 | 141.6 | 153.9 +8.7 % (153.0 +8.0 %) | 96.6 | 96.7 +0.1 % (96.4 -0.2 %) |
| MULTIPASS (> 32 rows = repeated passes) | 0.40 | 154.2 | 164.0 +6.4 % (same) | 103.7 | 103.8 +0.1 % |
| ONEPASS (Leap's v2 growth, rescaled to E14) [est] | 0.40 | 154.2 | 178.2 +15.6 % (175.8 +14.0 %) | 103.7 | 103.9 +0.2 % |
| ONEPASS [est] | 0.92 | 141.6 | 159.3 +12.5 % (156.1 +10.3 %) | 96.6 | 96.7 +0.2 % |

- With drafting hidden (L 0): ONEPASS HE+ +17.4 % (oracle +35.9 %).
- Lossy byte-floor model, ONEPASS L 0.40: HE+ +12.3 % (oracle +34.0 %), MBPP +0.4 % (oracle +17.8 %).
- Oracle (hindsight DP), KERN2 L 0.40: HE+ +18.9 %, MBPP +9.5 %. The gap to the policy is per-check signal that
  path_p misses, mostly on MBPP.

## Caveats

- **Dev data.** heplus24 + mbppbank, the same windows the K* tables use. The gain is HE+-specific (copy-heavy). A
  fresh-cells re-read on Wednesday's capture is the clean test.
- **Engine.** The policy needs a 32-row TREE verify path. Today's TREE16 path is 16 rows (M4 tests it Tue night).
  C32one exists as a chain check. c32 = 1.04 is a prior.
- **Drafting.** The draft cost per level is an estimate (0.40 for a smaller head; 0.66-0.92 for today's 177.6 MB per
  level). A draft forward is counted per tree level, including the spare-slot pops.
- **Prior art.** The live 45 GB loop truncates rows by running confidence (S45c t 0.3, ADOPTED; raising it was a
  KILL, S45d). This rule is the other direction: grow past 16 when confident. Outside the project, OPT-Tree /
  Sequoia-style size choice. The new part here is the measured calibration and gain on this drafter and engine.

## The fresh-cell re-read is FROZEN in adaptk_fresh.py (Tue 01:11 by date, before any fresh data)

- Settings: tau 0.9, KERN2, L 0.66, E14 costs. KILL if the HE+-like gain is < +3 %.
- Thursday: `adaptk_fresh.py --he <fresh HE+ tag> --mbpp <fresh MBPP tag>` on BigMoE's post-capture 127-node dumps.
- Dev dry check (adaptk_fresh_devcheck.txt): HE+ +8.6 % [+6.7, +10.8], MBPP +0.2 %, verdict PASS. This reproduces
  adaptk_windows.

## Cheapest next kill

On fresh cells (the Wednesday capture's dumps), re-read the policy at the frozen tau (0.9 = MBPP's, cross). It is
KILLED if HE+ falls under +3 % at L 0.66 with KERN2. Then one engine A/B (adaptive vs fixed-16, C T T C, like M4)
once a 32-row tree verify exists.

Files: adaptk_oracle.py / .txt (stage 1, drafting hidden), adaptk_draft.py / .txt (draft cost + policy),
adaptk_sens.py / .txt (kernel variants), costs from costs_e14.json.

## The LOSSY line (Tue 03:0x; adaptk_lossy.py / .txt)

The lossy line runs trees with close-call CC(1), and trees and CC overlap (TREES-x-CLOSECALL), so the
exact screen did not settle the lossy case. Replayed with tree_cc32's tree + CC rule at 15 / 31 pops:
- v4 modules-PASS costs (40 ms at 32 rows, 16 rows at E14's ratio, L 0.42, overhead 4): HE+ fixed 16 320.1 /
  fixed 32 320.7 (a tie), policy tau 0.9 349.4 = +9.0 % [+4.3, +13.5]; exact acceptance +8.1 %.
- v4 stretch (32 ms, L 0.25, overhead 1.5): +9.0 % [+7.3, +10.9].
- Byte floor (25 ms x R_floor ratio, L 0.42): +4.8 % [+0.8, +8.4]; +6.1 % at the cross tau 0.4.
- MBPP +0.1-0.2 % on every model. Hindsight ceiling HE+ +15-19 %.
- Self-checks: kept 11.263 / 16.263 (exact) and 11.762 / 16.910 (CC), the published values.
Registered line: HOLDS at >= +5 % HE+ (CC, v4pass, tau 0.9). Result: HOLDS.

## Switch cost budget (Tue 03:1x; adaptk_switch.py / .txt)

Consecutive checks change row count on 16-17 % of HE+ checks (1 % on MBPP): the choices cluster. With s ms
paid per switch, HE+ exact stays +8.2 % at s 2 and +6.9 % at s 8; CC(1) on v4 costs +8.1 / +5.5 %. So a
switch may cost several ms; keeping both kernels' buffers live is enough.

## ADAPTIVE K DOWN: verify 8 rows when the drafter is unsure (Tue 04:5x; adaptk_k8.py, adaptk_k8lossy.py)
- **Idea:** ADAPTIVE K never goes below 16 rows, and it gains ~0 on MBPP. When the 8th pop's path_p is low, pops 8-15
  carry little expected kept, so verify 8 rows. That gives a smaller expert union and a shallower draft tree.
- **PEEK rule** (tau8 cross-fit from the other set): verify 8 rows iff the 8th pop's path_p < tau8 (~0.25 at R8 0.78),
  else ADAPTIVE K as frozen. Deciding needs only 7 pops + the peek, so the 8-row draft is shallower (6.7 vs 13.6 levels
  on HE+). The SUM rule (expected kept of pops 8-15 < theta) does worse, because it must draft all 15 pops to decide.
- **path_p is calibrated at 8 rows too:** E[kept8] 7.17 vs real 7.20 (HE+), 5.73 vs 5.82 (MBPP).
- **Exact, E14 costs, L 0.66, over frozen KERN2 (cross-fit, 95 % window bootstrap):**

  | R8 = 8-row / 16-row check | HE+ | MBPP | checks at 8 rows (HE+ / MBPP) |
  |---|---|---|---|
  | 0.71 | +3.9 % [+1.3, +6.5] | +7.8 % [+6.2, +9.3] | 22 % / 77 % |
  | 0.78 | +1.6 % [+0.2, +3.0] | +3.0 % [+1.8, +4.1] | 12 % / 38 % |
  | 0.86 | +0.2 % | +0.4 % | 3 % / 7 % |

  - Break-even (>= +1 %): R8 <= 0.78 on HE+, <= 0.82 on MBPP.
  - Hindsight {8, 16, 32} at R8 0.78: HE+ +14.7 %, MBPP +20.5 %. Like KERN2, the calibrated rule gets only a small part.
- **Lossy, CC(1):**
  - LADDER v4 modules-PASS costs: at R8 0.78, HE+ +0.9 % and MBPP +2.4 %; at 0.71, +3.3 % / +7.0 %.
  - Byte floor (its bytes ratio, 0.744): +2.0 % / +4.1 %.
- **R8 on today's engine [est] is ~0.78-0.80.** REF2-T pads every call to 16-row tiles, and an M=8 MMA tile runs padded
  (MoonShots d50e76a). So only two parts shrink:
  - the expert union, 73.2 -> ~44-49 per layer: CHECK55's experts 33.71 ms x ~0.64, about -12 ms;
  - the attention cores, about -1.3 ms.
  The contract / BOF levers lighten the fixed part, which pushes R8 toward ~0.76.
- **Verdict (the line was set before the run: MBPP cross-fit >= +2 % at R8 0.78): ALIVE, but SMALL.** On today's engine
  it is worth +1-2 % on HE+ and +2-3 % on MBPP.
  - Kill test: time one 8-row check next to A16 in any E-run (~1 min under a quiet lock). KILL if R8 >= 0.82.
  - For the build, it is worth adding only if the row count is a parameter anyway.

## The fresh re-read needs only the planned T8-32 dumps (Tue 04:5x; adaptk_fresh_n32check.txt)
- BigMoE's 32-node dumps (tree_t30b_{tag}.npz, k 8) are IDENTICAL to the first 32 pops of the 127-node dumps: depth,
  path_p, on_true and parent match on 100 % of 204,000 (HE+) and 500,512 (MBPP) nodes.
- The frozen adaptk_fresh.py run on them reproduces the dev result exactly: HE+ +8.6 % [+6.7, +10.8], MBPP +0.2 %.
- So Thursday's re-read runs on the T8-32 dumps already on PLAN900's Thursday list (~0.8 h by day). No 127-node dump is
  needed. Command: `adaptk_fresh.py --he TAG --mbpp TAG --dump-fmt "tree_t30b_{tag}.npz"` (the script itself unchanged).

