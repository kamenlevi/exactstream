# PREREG: DEEPDRAFT LAYERS, the per-layer freshness curve (Sat 14 Nov 2026, ~40 min of data + evals)

Written before the run. After REPLACE-H (cc53cb5, CARRIES):
- t30b's stand-in replaced by a ridge map of the true fused h (layers 5-45) holds 91.20 % at MBPP 23-30;
- t30b's own chain holds 72.13 %, and the true wd 93.63 %.
- The cost question: how few trunk layers over the drafted tokens give a usable fresh state? Leap prices a partial trunk
  pass per L in parallel (the orchestrator 16:5x).

## A finding from the setup (16:55-16:57), information
- **The prefill path is chunk-dependent.** The shipped prefill gives features that depend on which tokens share the
  chunk.
  - CORRECTED (Leap, 17:0x; Leap's data/pkg121_1113/README.md, last section): the "40" is not an expert cap.
    GATE_N = 40 counts ids (rows x top-10, bigmoe/stream.py:77).
  - A layer call of <= 4 rows takes the decode's gate path (REF2-T tiers). A larger call takes tier.prefill_call:
    exact 4-bit, every expert the call uses, so no routed expert is dropped.
  - The 8-18 % below is most likely routing flips after shape-dependent bit differences (the path and the gather
    kernel). That is not measured. The exact path equals NAX6-N on the same 1,024-token grid.
- Prefilling a capture's prompt ALONE (as the capture did) reproduces its h and wd bit for bit: relative difference
  0.00e+00.
- Prefilling prompt + answer in one chunk moves the prompt rows by 11 % (h) / 18 % (wd), and the answer rows by 8 % / 9 %
  against the capture.
  - The capture's answer rows came from the verify path, a few rows per check under REF2.
  - It is the same at C 48 and C 100, so the cache size doesn't matter.
- An exact re-prefill (prompt alone, then the answer through the verify path a few rows at a time) runs about 20 s a
  file. That is too slow for this screen.

## Setup
- **Data:** dd_reprefill.py, full-sequence prefill (fast, ~8 s a file at C 100), over 100 training files (the first 100
  of the 795 unseen n1112 files) and MBPP 100.
  - Saved per row: the raw taps of layers 5/15/25/35/45, each the mean of the 4 HC streams (5 x 2560 fp16), and the
    prefill's widened stream wd (10240).
  - Inputs and targets come from the same pass, so the maps are self-consistent. The approximation applies to all of
    them alike.
- **Maps:** ridge, the target is the re-prefilled wd, lambda picked on 10 held-out training files. Input sets:
  - {5}, {5, 15}, {5, 15, 25}, {5, 15, 25, 35, 45} (raw);
  - {45} alone, the top-tap control: how much the last tap carries by itself.
- **Evals:** t30b unchanged, MBPP 100, the stand-in replaced at every drafted row (dd_taps.py evalrep), by:
  - **(gate)** the re-prefilled wd itself: the approximate oracle. It must hold >= 90 % at 23-30, or the approximation
    is too coarse for this screen. If so, the curve is not read and the exact verify-path re-prefill comes next.
  - each map's prediction from that set's raw taps.
- Each set's result is reported as the share of the gap recovered: (set - 72.13) / (approximate oracle - 72.13), at
  23-30.

## Lines (result -> action)
| Result | Line | Action |
|---|---|---|
| CHEAP | {5} or {5, 15} recovers >= 75 % of the gap | A partial trunk pass over the first ~15 layers gives a usable fresh state. Hand Leap the curve to price it; the method candidate is "MTP head + a cheap early-layer pass per drafted row". |
| MID | only {5, 15, 25} or more reaches >= 75 % | Freshness needs half the trunk or more. Leap prices it. The sequential cost per drafted row decides whether it beats plain verification. |
| COSTLY | only all 5 taps (or the fused h) reach >= 75 % | Freshness needs nearly the full trunk. The lever then becomes cheaper verification or a learned refresher. Re-plan with the orchestrator. |

- Every band, kept per 32, and each map's R2 and normalized error are reported.

## Addendum, the BLOCK refresh sweep (the orchestrator 17:0x, written before it runs)
- **Leap's pricing (5bc983d, ~/Leap/data/week98/trunk_price.txt):**
  - A partial trunk pass per drafted token doesn't pay. At 45 GB, L = 5 adds ~101 ms per check, and the break-even is
    13.9 kept per 32 on MBPP.
  - A k-row pass costs about a 1-row pass (~0.62 ms per layer, CPU-graph-bound). So refreshing every k drafted tokens
    divides the cost by ~k.
- **The sweep:** the same no-training harness. For each LAYERS set and k in {2, 4, 8, 16}:
  - the stand-in is replaced by M(taps) only at depths j = 1 + m k, the rows a block pass would just have computed;
  - the head carries its own stand-in in between (dd_taps.py evalrep --refresh k);
  - k = 1 is LAYERS itself.
- The cost of a set is its deepest tap: {5} = L 5, {5, 15} = L 15, {5, 15, 25} = L 25, all and {45} = L 45.
- **Read (result -> action):** for every (set, k), MBPP 100's kept per 32 against Leap's MBPP break-even for (L, k), at
  45 GB and at 24 GB.
  - **FEASIBLE:** the cheapest (L, k) by ms per draft whose kept clears the break-even is the method's cost-feasible
    point. Its 23-30 accuracy is the method's per-guess number. Next: a confirmation on HumanEval+ (needs the
    verify-path re-prefill of the 24 windows), then the engine build plan with Leap.
  - **NONE:** no (L, k) clears it. Partial trunk passes don't pay at today's engine cost, and the method needs a
    cheaper refresher (a learned one, or Leap's fast path). Re-plan with the orchestrator.
- Caveat: these are the approximate full-sequence features, read only if the LAYERS gate holds (>= 90 %).

## Note for the method's final numbers (the orchestrator + Leap, 17:0x)
- Features must come from the runtime path: the prompt prefilled on the 1,024-token grid, with decoded rows through the
  verify path. That is the exact re-prefill (prompt alone, then the answer through verify calls).
- The LAYERS and BLOCK curves from the fast full-sequence prefill are fine for the shape, if their gate holds.
