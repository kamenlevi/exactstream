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
