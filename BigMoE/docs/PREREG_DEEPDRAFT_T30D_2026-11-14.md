# PREREG: DEEPDRAFT T30D, a deeper drafter (t30b's head + extra layers), with a read-out stop line (Sat 14 Nov 2026)

Written before any build or run. It starts only once Sandbox98's deeper-drafter test on the 2B shows depth helps (the
orchestrator 23:1x).

## Why (the theory, from today's tests)
- **The head reads a fresh state well.**
  - With the true trunk stream at every drafted row, t30b holds 93.63 % at MBPP 23-30 and 97.76 % at HE+ 23-30, flat
    with depth (REPLACE-H / bank calibration).
  - Its own carried state gives 72.13 / 89.65.
- **What the fresh state adds is composition, not retrieval.**
  - T1: the misses a fresh state fixes are LESS copyable than average (prompt lookup 12 % vs 31 %); the chain already
    gets the copyable ones.
  - The STALE anchor collapses to 16.00 %, and a linear state transition (ROLLOUT) reaches only 23.08 %.
  - Sandbox98 T2 (2B, MBPP-50 23-30): the 2B over only the drafted tokens 69.62 vs its one-layer head's chain 55.42; with
    a 64-token window 80.61; the full prefix 87.71.
- **Capacity in width doesn't do it.** CAPQ: +1.2 pt at 23-30, kept +0.04. Depth (more layers composing the drafted
  tokens) is the untested axis.

## Setup
- **t30d1:** t30b's head plus 1 extra transformer layer of the head's own architecture (attention + MoE + hyper-connections).
  - Its weights are copied from the head's layer, and its residual outputs (o_proj and the MoE down projections) are
    zero-init, so step 0 equals t30b exactly (smoke: EQUAL).
  - In the chain, each depth runs both layers. The extra layer keeps its own KV over the context rows and the drafted rows.
- **t30d2:** 2 extra layers, the same way. It runs only if t30d1's stop line passes.
- **Training:**
  - The head continues on t30b's own Adam state. The new layer gets a fresh, bias-corrected Adam at the head's lr; no
    10x (amendment 1's lesson).
  - Schedule: re-annealed, cosine from 1e-5 down to 1e-6, over one pass of t30b's pinned split.
  - Data: t30b's pinned split, pass seed 1113 (as every TTT30 arm).
- **Stop line by READ-OUT, never the AUF loss** (Sandbox98's survival-weighting catch):
  - At 10 % and 25 % of the steps, score 30 held-out MBPP windows (the first 30 of d1102mbpp's eval set), paired with
    t30b on the same windows.
  - Stop if 23-30 is not >= t30b + 1 pt at 25 %.
- **Final read-out:** MBPP 100 + HE+ 24 (runtime-path context, the bank) against t30b, bands and kept per 32.
- **Costs to log:**
  - one drafter step's time per layer count (1 / 2 / 3), for Leap's tables;
  - the MLX peak.
  - If Sandbox98's exact two-pass TTT gradient ports (its peak 5.8-7.8 -> 2.5-3.4 GB on the 2B), use it, and keep it only
    if the smoke shows it equal to autograd in fp32.

## Lines (result -> action; MBPP 23-30 vs t30b, final read-out)
| Result | Line | Action |
|---|---|---|
| DEPTH PAYS | t30d1 >= t30b + 3 pt, and kept per 32 >= t30b + 0.5 | Depth is the method's lever. Run t30d2. Price the extra layer per drafted token with Leap. |
| SMALL | t30b + 1 <= t30d1 < t30b + 3 | Depth helps a little at this data. Try t30d2 before deciding. |
| NO | below t30b + 1 (or the stop line fires) | Depth via extra head layers doesn't help at this data; the next idea is distillation from the 125B's states (Sandbox98's provisional lead). |

- **T3's k-curve, appended when it lands:** how many principal directions of the fresh state the head needs. If a small k
  suffices, the extra layer's output can target just those directions.

## T3's k-curve (landed Sat 14 Nov 23:28): the head needs most of the fresh state, not a few directions
- t30b on MBPP 100 with the stand-in replaced by the true wd kept in its top-k PCA directions (the PCA from the bank's
  train rows; data/deepdraft/tapsq/t3_k*_*.json). The other directions are filled with the mean, or with the stale anchor
  wd_s:

  | k | % of the variance | 23-30, mean fill | 23-30, stale fill | kept per 32 (mean / stale) |
  |---|---|---|---|---|
  | 16 | - | 29.93 | 27.61 | 2.41 / 3.08 |
  | 64 | - | 50.12 | 44.42 | 3.07 / 3.60 |
  | 256 | - | 69.89 | 64.84 | 4.38 / 4.70 |
  | 1024 | 87.7 | 86.93 | 85.38 | 8.06 / 8.21 |
  | all (the oracle) | 100 | 93.63 | | 12.76 |

  - t30b's own chain: 72.13, kept 6.92.
- **Reading:**
  - 256 directions only match the chain. Even 1,024 (88 % of the variance) leave 6.7 pt to the full state.
  - The information the head needs is spread over > 1,000 dimensions, so a predictor of a few directions can't do it.
  - A deeper drafter has to regenerate a near-complete state. That is consistent with T1 (composition) and Sandbox98's
    T2 (a deep stack over a 64-256-token window).
  - The stale fill is worse than the mean beyond depth 1-2: the anchor's state is the wrong state for later positions
    (STALE alone: 16.00).

## ON HOLD (Sun 15 Nov 00:0x, the orchestrator): Sandbox98's depth tests on the 2B are negative for zero-start layers
- A zero-init layer added to the converged head: -2.5 pt at 23-30.
- A 2-layer vs a 1-layer head, both from the native start, 400 steps: +0.05 pt (CI -1.04..+1.16). The extra layer barely
  grew (|w| ~0.0025 at 3e-5).
- The 1-layer head gains +7.1 pt from training budget alone (400 -> 1,889 steps).
- Reading: the composition gain in T2 (a deep stack over the drafted tokens) is PRETRAINED composition, which a zero-start
  drafter layer doesn't learn at these budgets.
- T30D is not built unless that changes. The 01:00 sessions choose among pretrained-composition drafters, much more
  training data, and the research report. The build spec stays in BigMoE's HANDOVER.
