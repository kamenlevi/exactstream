# PREREG: DEEPDRAFT REPLACE-H, fresh state by replacement, no head training (Sat 14 Nov 2026, ~15 min)

Written before the run. After FRESH-H (FLAT, in fact worse): an additive zero-init input couldn't hand the head a fresh
state. Law 2 (RESULT.md, 11-11) fed the reader's native input instead, the exact trunk state two back, and held 93.5 %
on MBPP at every depth. The orchestrator and BigMoE converged on this pair of eval-only checks (16:4x-16:5x).

## Setup
- t30b, unchanged, on MBPP 100 (d1102mbpp; it has wd and h on every row). The eval's chain runs to depth 31 with
  anchors from gen - 2, as every DEEPDRAFT eval.
- At every drafted row (depth j >= 2), the carried stand-in (the head's own output, its guess of the trunk's widened
  stream at row s + j - 1) is REPLACED by:
  - **(oracle) wd:** the true widened stream at row s + j - 1. This is Law 2's setting, the gate for the harness.
  - **(pred) M(h):** a ridge map's prediction of wd from the true h at that row (scripts/dd_ridge.py).
    - h is DFlash2's fused input of layers 5/15/25/35/45, 2560 wide.
    - The fit uses 400 of the 795 unseen n1112 files, every row with h; lambda is picked on the next 40 files.
    - Reported: R2 and the RMS-normalized relative error on the validation files.
- On a correct chain the drafted tokens are the true ones. So both inputs are what a trunk pass over correct drafts
  (oracle) or a compact fresh signal (pred) would give.
- No head training in either pass. Script: dd_taps.py evalrep.

## Lines (result -> action; MBPP 23-30, t30b's own chain is 72.13)
- **Gate first:**
  - The oracle must reproduce Law 2 within about 1 pt: 23-30 >= 92.5 (93.5-94 expected; t30b's band 0 is 94.01).
  - If not, the run is VOID-HARNESS: check the drafted-row offsets before anything else, and M(h) is not read.

| Result (pred) | Line | Action |
|---|---|---|
| CARRIES | M(h) >= 88 | A compact fresh signal carries most of the trunk state. Run the re-prefill (bd5369f), then fit the same ridge from the raw layer-L taps per L (5, 5+15, ...): the per-layer freshness curve, i.e. what a cheap partial trunk pass buys, and the engine's price question. |
| PARTIAL | 75 <= M(h) < 88 | Partly there. Next a better map from the same data (an MLP, or the raw 5 x 2560 taps after the re-prefill), and training the head on the predicted input. |
| WEAK | M(h) < 75 | h can't stand in for the trunk state. The routes left are a real partial trunk pass with more layers, or training the head on predicted inputs. Re-plan with the orchestrator. |

- Every band and kept per 32 are reported for both passes. The ridge's R2 and normalized error are reported with
  them.

## Result (Sat 14 Nov 16:49-16:53): CARRIES. A linear map of the fresh multi-layer features recovers 89 % of the depth gap
- **The ridge map h -> wd** (data/deepdraft/tapsq/ridge_h2wd.json; the map itself is ~/datasets/deepdraft/ckpt/ridge_h2wd.npz):
  - 400 train files (220,753 rows), 40 validation files, lambda 1e-3 x mean diag, 63 s on 4 cores.
  - On the validation files: R2 0.797, and the RMS-normalized relative error is 0.384.
- **MBPP 100 bands, t30b unchanged** (rep_wd.json, rep_pred.json):

  | band | t30b's own chain | stand-in replaced by M(true h) | replaced by the true wd (oracle) |
  |---|---|---|---|
  | 0 | 94.01 | 94.01 | 94.01 |
  | 1-2 | 89.14 | 91.78 | 93.93 |
  | 3-6 | 83.45 | 91.60 | 93.82 |
  | 7-10 | 79.61 | 91.46 | 93.77 |
  | 11-14 | 77.06 | 91.34 | 93.73 |
  | 15-22 | 74.38 | 91.28 | 93.75 |
  | 23-30 | 72.13 | **91.20** | 93.63 |
  | kept per 32 | 6.92 | 10.55 | 12.76 |

- **Gate:** the oracle reproduces Law 2: 93.63 at 23-30, within 0.4 pt of band 0's 94.01, and flat with depth. The
  harness and the drafted-row offsets are right.
- **Verdict: CARRIES.** M(h) holds 91.20 at 23-30 (>= 88).
  - That recovers (91.20 - 72.13) / (93.63 - 72.13) = 89 % of the depth gap.
  - It needs no head training. The map is a linear fit with a 38 % direction error.
  - The accuracy is nearly flat with depth (91.8 -> 91.2): with a fresh state at each depth, errors stop compounding.
- **What it means for WEEK98:**
  - Freshness is the lever. A compact fresh signal at each drafted row lets t30b's head draft at ~91 % per guess at
    depth 23-30, against 72 % today.
  - The open question is cost. h needs the trunk's layers 5-45 over the drafted tokens, almost a full pass.
- **Action, as pre-registered:** run the re-prefill (raw 5 x 2560 taps), then fit the same ridge per layer set (5,
  5+15, 5+15+25, ...). That gives the per-layer freshness curve: what a cheap partial trunk pass buys, and the price
  question for the engine.
- **Caveat:** this is per-guess accuracy given a correct prefix (the headline bands' definition). In a real draft, the
  fresh features at drafted rows must come from a trunk pass over the draft as it grows. That is the engine question.
