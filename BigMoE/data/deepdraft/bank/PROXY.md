# DEEPDRAFT feature BANK + PROXY screen (Sat 14 Nov 2026)

Built so most WEEK98 screens never run the 125B again (the orchestrator 18:0x, after Kamen asked how to test ideas
faster without losing accuracy).

## The BANK (~/datasets/deepdraft/bank/<key>.npz; built by scripts/dd_bank.py)
- **The runtime path:** the prompt alone through the shipped prefill on the 1,024 grid, then the answer through 9-row
  verify calls at C 100 (the gate arm, REF2's verify scope), every row committed.
- **Checks:**
  - It reproduces the n1112 and MBPP captures bit for bit: wd relative difference 0.00e+00 on prompt and answer rows.
  - The 125B's top-1 agrees with the capture on 100 % of answer rows, and equals the next token on 100 % (greedy).
- **Per file, rows 0..N-2 (aligned with the captures' h / wd):**
  - `raw` [n, 12800] fp16: taps 5/15/25/35/45, each the mean of the 4 HC streams, 2560 wide, in that order;
  - `wd` [n, 10240] fp16: the trunk's widened stream, the MTP head's native input;
  - `tk` / `lp` [n, 32]: the 125B's top-32 ids / log-probs, on rows >= gen - 1;
  - `ids`, `gen`, `src`, `path`.
- **Sets (data/deepdraft/bank/split.json):**
  - train: 360 n1112 files t30b never saw;
  - val: the next 40;
  - MBPP 100: d1102mbpp, the eval set;
  - HE+: the 24 d16w windows.
- **HE+ caveat:** the d16w windows' own stored wd differs from the runtime path (10 % on prompt rows, 21 % on answer
  rows). Every HE+ number before tonight was scored on d16w's stored features. The bank holds the runtime-path ones;
  see the calibration log for t30b re-scored with them.

## The PROXY (scripts/bank_proxy.py, CPU)
- `bank_proxy.py cache`: once. Builds the memory-mapped proxy set:
  - train rows: every 3rd answer row of the 360 train files;
  - val rows: every answer row of the 40 val files.
- `bank_proxy.py ridge --inputs tap45[,tap5,...] [--save map.npz]`: a ridge over the named taps -> wd.
- `bank_proxy.py custom --py cand.py`: your own fit / predict.
  - `cand.py` defines `INPUTS = ["tap45", ...]`, `fit(Xtr: dict name -> [rows, 2560] float32, Ytr [rows, 10240]) ->
    model` and `predict(model, X: dict) -> [rows, 10240]`.
- **Output:**
  - held-out R2 and the RMS-normalized relative error on the val answer rows;
  - the predicted MBPP 23-30, from calibration.json (monotone piecewise-linear in R2);
  - "REPLACE-worthy" if the prediction is >= 88.
- **Memory:** a single tap's ridge reads ~0.4 GB; all five taps ~6 GB, so run that alone.

## SCOPE: read this before using it
- The proxy scores predictors of wd only: maps that REPLACE the head's carried stand-in with an estimate of the trunk's
  widened stream.
- A learned stand-in lives in the head's own space. t30b's carried stand-in has R2 -2.32 against wd, yet the head drafts
  72 % at depth 23-30 from it. So a learned stand-in (or any predictor whose output is not a wd estimate) must be judged
  by the drafting eval itself (dd_taps.py evalrep, or a training run). (The orchestrator, 18:0x.)
- The proxy is a filter, never a verdict. Only the REPLACE eval (`dd_taps.py evalrep --replace rawmap --ridge MAP
  --featdir ~/datasets/deepdraft/bank`, ~3 min of GPU) gives a number. Every real eval adds a point to calibration.json;
  re-check the curve as points land.

## Calibration
- Built by scripts/bank_calibrate.sh after the bank: the five layer-set maps plus the oracle, each with a real REPLACE
  eval on MBPP 100 (bank features).
- Today's points on the earlier, approximate features showed the shape: R2 0.41 -> 41.7, 0.45 -> 47.5, 0.51 -> 56.1,
  0.80 -> 91.2, 0.82 -> 91.6, 0.84 -> 91.5. Monotone, with a threshold near R2 0.8. The bank's own points replace them.

## R2 alone is not enough: calibrate per family (Sandbox98's gate report, 11-14 19:51)
- On Qwen3.5-2B, the head tolerates UNBIASED noise around the true state far better than map error at the same R2:
  - true W + noise at R2 ~0.91 -> 87.34, ~0.75 -> 85.60, ~0.51 -> 82.66 at MBPP 23-30;
  - a ridge map at R2 0.44 -> 59.19.
  - The error's structure matters, not only its size.
- So:
  - Calibration points carry a `family` ("ridge" for the bank's ridge maps; others as they land). The prediction uses
    only the same family's points, plus the oracle.
  - A custom candidate sets `FAMILY = "..."` in its file. Until its family has >= 2 real REPLACE points, the proxy prints
    no prediction, only the measures.
- **Structure measures, printed with every candidate:**
  - `shrink`: the slope of (P - mean) on (Y - mean). 1 = unbiased; ridge maps shrink toward the mean.
  - `r2_unshrunk`: R2 after dividing that slope out.
  - `bias`: the share of the squared error in the mean offset.
  - `r2_pc` / `r2_rest`: R2 on the top 64 principal directions of wd, and on the remainder.
  - `r2_head`: R2 in the head's own fused input (its input norm, then fc_hidden per HC stream). Printed once
    ~/datasets/deepdraft/ckpt/head_in.npz is exported, after the bank.
- On synthetic data: unbiased noise at R2 0.68 keeps r2_pc 0.97 with shrink 1.00; a pure 0.6x shrink gives shrink 0.60
  and r2_unshrunk 1.00.
- Which measure tracks the real REPLACE accuracy best is for the calibration points to show; read them together.
- This proxy is still for wd-space predictors only (see SCOPE). A learned stand-in that lives in the head's own space is
  judged by the drafting eval. Sandbox98's result says it doesn't need to be exact; its errors need to look like noise,
  not bias.

## The bank calibration (Sat 14 Nov 21:05; calibration.json, calibrate.log)
- t30b on MBPP 100 with bank features, the stand-in replaced at every drafted row by each ridge map. Real REPLACE evals.
  ne = direction error, the primary measure.

| map (family ridge) | ne | R2 | shrink | r2_pc | r2_head | MBPP 23-30 | kept per 32 |
|---|---|---|---|---|---|---|---|
| tap5 | 0.646 | 0.384 | 0.42 | 0.62 | 0.17 | 46.11 | 2.85 |
| tap5,tap15 | 0.621 | 0.422 | 0.48 | 0.67 | 0.24 | 51.16 | 3.08 |
| tap5,tap15,tap25 | 0.580 | 0.497 | 0.55 | 0.75 | 0.32 | 59.69 | 3.50 |
| tap45 | 0.340 | 0.829 | 0.84 | 0.96 | 0.84 | 91.96 | 11.26 |
| all five taps | 0.330 | 0.838 | 0.84 | 0.96 | 0.86 | 91.88 | 11.14 |
| oracle (the bank's wd) | 0 | 1 | 1 | 1 | 1 | 93.63 | 12.76 |

- Within the ridge family every measure orders these the same way. tap45 and all five are a tie within noise. The ridge
  maps shrink (0.42-0.84), and the shrink grows with the error.
- The proxy itself: `bank_proxy.py ridge --inputs tap45` gives ne 0.340 -> predicted 91.9 (real 91.96), REPLACE-worthy.
  - One ridge fit reads ~32K x (2560 per tap + 10240) rows and takes 1-5 min on 6 threads, not seconds. A custom
    candidate with a cheaper fit is faster.
- **HE+ on the runtime path** (the 24 windows, t30b, depth bands 0 -> 23-30):
  - t30b's own chain on d16w's stored context: 97.90 -> 89.66, kept 13.83 (= the headline);
  - its own chain on the bank's runtime-path context: 97.95 -> 89.65, kept 13.77. The headline holds;
  - the oracle (runtime-path wd at every drafted row): 97.95 -> 97.76, flat, kept 20.90;
  - the tap45 ridge map: 97.95 -> 96.40, kept 18.06.
  - The HE+ ground truth itself is slightly off the runtime path: in 11 of 24 windows the 125B's runtime top-1 isn't the
    window's next token on 0.3-2.7 % of answer rows.
