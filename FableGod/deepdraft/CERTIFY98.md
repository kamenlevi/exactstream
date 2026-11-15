# CERTIFY98: how GOAL A gets certified (EvalSuite, Sun 2026-11-15 01:3x)

- **The claim:** the drafter is right on >= 98 % of its guesses at every depth 0-30, on held-out text (WEEK98; Kamen
  11-15 ~00:4x).
- **When:** the proof run is Thu 19 - Fri 20, and the verdict is due by Sat 21.
- **Registered in:** EvalSuite docs/PREREG_certify98_2026-11-15.md. Changes before the proof run are dated amendments
  there; nothing is re-scored after the data are seen.
- **Sizing:** tools/certify98_power.py (EvalSuite, CPU, ~20 s).
- **AMENDMENT 4 (02:3x) replaces the band rule below.** Read PREREG_certify98 amendment 4 first.
  - The gate is now the STRICT accuracy P(right at k | right at 0..k-1) at the decode walk's check starts, in bands
    0-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30.
    - Every band's 95 % lower bound must be >= HE_LINE 97.0, with kept/32's lower bound >= 20.76 as co-primary.
    - Power: ~0.7 (0.68-0.73) at a true 98 % with 140 HumanEval+ windows, 0.93-0.94 with 300.
  - The marginal bands in sections 1-4 are information only. They count rows the engine never drafts.
  - The finding behind it: on t30b's own walk, the first guesses after a check are the weak ones (HumanEval+ 91.4 % at
    0-2), and the deep ones are ~97.5 %.
    - Raising depths 15-30 to 98 % adds only +0.6 kept/32; raising 0-14 gives 22.6 of the 23.8.
- **What it said before amendment 4, in short:**
  - HumanEval+ is the gated headline.
  - Its windows cluster strongly, and HumanEval has only 164 problems (140 fresh). So a 95 % lower bound of >= 97.5 %
    passes a drafter that is truly at 98 % only ~56 % of the time.
  - The registered line is therefore 97.0 % (94 % power), with every point estimate quoted.
  - MBPP is reported against its own exact ceiling. Its line is one number to change.

## 1. The metric (one definition for TrainLab, Sandbox98, BigMoE and the engine)
- **Per guess:** per-depth accuracy with TRUE prefix tokens (teacher-forced).
  - At answer row a and depth d (0-30), the drafter's top-1 guess is compared with the target's runtime top-1.
  - The drafter is fed the true tokens up to a + d, as in dd_ttt.py's eval (rt[n, 31]).
  - **CORRECTED by amendment 4 (02:3x): this marginal per-depth accuracy is NOT the engine's accept rate.** A guess
    counts only when depths 0 .. d-1 were right, and only at the engine's own check starts, right after a miss.
    - The gate now uses the STRICT accuracy P(right at d | right at 0..d-1) at the decode walk's check starts. It
      reproduces kept_walk exactly.
    - The bands below are information only. The figures here are kept for the record; PREREG_certify98 amendment 4
      holds the rule and its power.
- **Bands:** dd_ttt.py's BANDS: 0, 1-2, 3-6, 7-10, 11-14, 15-22 and 23-30.
  - A band's accuracy is its right cells over its cells, pooled over windows: the ratio estimate, as the read-outs
    compute it.
  - The headline bands are 0 and 23-30. All seven are gated.
- **Per depth:** each depth's own accuracy (31 numbers). It is a guard against a dip hidden inside a band.
- **kept/32:** dd_ttt.kept_walk, the decode-path replay. From the first answer row, each check keeps 1 + the run of
  rights (capped at 31), then jumps on. The value is the mean tokens kept per 32-row check.
  - A flat 98 % gives 23.80. A flat 97.5 % gives 22.21, and 97 % gives 20.76.
  - Misses that cluster along a chain raise kept/32 above these.
  - kept/16 (15 drafts per check) is reported too, for Leap's rule that a 32-row check pays when kept/32 >= 1.35-1.4 x
    kept/16.
- **Margin bins (a required read-out; NewIdeas 01:2x, "where the errors live"):** every band is split by the target's
  own margin m (top-1 minus top-2 log-prob): sure (m >= 4 nats) and unsure (m < 4). Each bin gets its share of cells
  and its accuracy.
  - t30b on HumanEval+:
    - band 0: sure 90.2 % of cells at 99.91 %, unsure 9.8 % at 79.90 %;
    - band 23-30: 94.49 / 49.62.
  - t30b on MBPP, band 0: sure 71.0 % at 99.89 %, unsure 29.0 % at 79.63 %.
  - A lever's gain shows up in the unsure bin, which is where 98 % is won or lost.
- **The engine's real tokens per check:**
  - The engine decodes the same prompts exactly; the 32-row check may be 2 x 16 gated passes.
  - On >= 20 fresh HumanEval+ windows, its mean tokens per check must be within +-3 % of kept_walk on the same windows.
  - A bigger gap is a bug until shown otherwise.
  - If the engine has no path for the method yet, kept/32 is labelled "cells only, engine pending".
- **One metric everywhere:** any read-out quoted in the certificate (TrainLab's, Sandbox98's, a new scorer) must
  reproduce dd_ttt.py's bands and kept_walk on the same stored cells to 0.01 pt.
  - t30b's reference values are MBPP 100 at 94.01 (band 0) and 72.13 (23-30), from cells_t30b_mbpp100.npz.
  - On HumanEval+ they are 97.95 and 89.65, from cells_t30b_heplus.npz (bank features).

## 2. Held-out discipline
- **Three ways a window can be "seen":** trained on, selected on, or leaked.
- **Trained on:**
  - t30b's training captures: n1112, 3,072 files, all dlog-magicoder, split train;
  - and every training file of the candidate, including the BANK's train 360 and val 40 if they fit a map or stand-in
    or chose a stopping step.
- **Selected on: every set screened this week is a DEV set,** reported but never gated. A method chosen on a set looks
  better there than it is. The dev sets are:
  - MBPP 100 (d1102mbpp's first 100);
  - the 24 HumanEval+ d16w windows;
  - the BANK val 40;
  - Sandbox98's MBPP-50 / HE-30 (on the 2B).
- **The fresh TEST sets** (drawn 01:5x, before any candidate: EvalSuite results/certify98/fresh_manifest_2026-11-15.json,
  built by tools/certify98_manifest.py): captured on the BANK's runtime path, never on d16w stored features (the shipped prefill, then
  9-row verify at C 100, every row committed). Their ids and capture sha256s are frozen in a manifest before the proof
  run.
  - **HumanEval+ fresh:** all 140 HumanEval+ problems outside the 24 d16w windows.
    - Sandbox98's 60 HumanEval windows (the 2B's dev set) lie inside these 140.
    - So the same verdict is also reported on the strict 80 that drop them (PREREG amendment 2). Its power is 0.79 at
      the 97.0 line.
  - **MBPP fresh:** a seeded draw of 300 problems from MBPP's test split (task_ids 11-510).
    - The draw excludes all 375 MBPP problems TrainLab ever captured (data/dlog/d1102mbpp), which hold d1102mbpp's
      100, EvalSuite's dec_mbpp 100 and Sandbox98's 100. That leaves a pool of 310.
    - Source: ~/TrainLab/data/mbpp/mbpp.jsonl (974 problems, local; no download).
  - **Everyday text** (PREREG amendment 3): 150 windows, CLEAN against n1112.
    - All 29 agent windows (dec_agent: Claude Code-style turns).
    - A seeded 60 of IFEval's 100 and 61 of IFBench's 150 (EvalSuite's instruction-following chat prompts).
    - No drafter trained on them or screened on them, and all are local.
- **Contamination check,** before scoring, on token ids:
  - **Source tags:** no training file of the candidate comes from HumanEval, MBPP or the everyday pool.
  - **Overlap:** a test window that shares any 50-token span with the candidate's training data is dropped and counted.
    If more than 5 % of a set is dropped, the set is VOID and re-drawn. Each set's 13-gram overlap rate is reported.
  - The 125B's own pretraining exposure to HumanEval and MBPP is unknowable. The everyday-text curve is the check that
    the result is not benchmark-specific.
- **The test sets are opened once,** for one pinned candidate: its checkpoint sha256, code commit, settings and the
  scorer's sha256, all in the amendment.
  - A second candidate on the same sets is a second look. It needs a one-sided 97.5 % bound (Bonferroni over two
    looks), and the report counts the looks.

## 3. Sample size from the measured clustering
- **The unit is the window** (one answer to one problem), because errors cluster by window.
- **Measured on t30b** (clustered by window, CR1):

| set | windows (answer rows) | band 0 | design effect | per-window error sd | band 23-30 | design effect |
|---|---|---|---|---|---|---|
| HumanEval+ | 24 (125-400) | 97.95 % | 15.7 | 2.44 pt | 89.65 % | 131 |
| MBPP | 100 (18-520) | 94.01 % | 5.1 | 3.22 pt | 72.13 % | 27 |

  - So HumanEval+'s 6,375 band-0 cells count like ~400 independent ones.
  - HumanEval+ already sits at ~98 % at depth 0, so its spread is the one measured at the truth level.
- **The registered lower bound:** the ratio estimate minus t(K-1, 0.95) x the window-clustered (CR1) SE, which is a
  one-sided 95 % bound.
- **The simulation** (400 reps):
  - The truth is flat 98 % at every depth.
  - Each window's per-band error profile is resampled and scaled to the truth.
  - Window sizes come from the set's bank files.
  - The depths in a band count as one draw per anchor, which is conservative.
- **P(every band's lower bound >= the line | a flat 98 % truth):**

| spread model | window sizes | line | K 60 | K 100 | K 140 | K 164 | K 200 | K 300 |
|---|---|---|---|---|---|---|---|---|
| HumanEval+ measured (sd 2.96 pt at 98 %) | HumanEval+ | 97.5 | 0.27 | 0.45 | 0.56 | 0.64 | - | 0.80 |
| HumanEval+ measured | HumanEval+ | 97.0 | 0.71 | 0.84 | **0.94** | 0.96 | - | 1.00 |
| MBPP relative spread (sd 1.31 pt) | MBPP | 97.5 | 0.24 | 0.52 | - | - | 0.92 | 0.97 |
| MBPP spread doubled (stress) | MBPP | 97.5 | 0.15 | 0.24 | - | - | 0.59 | 0.81 |
| MBPP relative spread | everyday (val-like) | 97.5 | 0.55 | 0.86 | - | - | 0.99 | 1.00 |

- **False certification:**
  - with HumanEval+'s spread at K 140, a flat 97.5 % truth against the 97.5 line: 1.0 %;
  - the same, a flat 97.0 % truth against the 97.0 line: 0.5 %;
  - with MBPP's relative spread (HumanEval+ sizes, K 150), a flat 97 % truth against the 97.5 line: 0.0 %.
- **Windows needed for a 95 % lower bound >= 97.5 % at a true 98 % (80 % power, every band):**
  - **HumanEval+-like code: ~300.** HumanEval+ has 164 problems, 140 of them fresh, so it can't reach that alone.
  - **MBPP-like code:** ~150-200, and ~300 in the stress case.
  - **Everyday text:** ~100, and ~200 in the stress case.
- **Hence the registered HumanEval+ line, HE_LINE = 97.0** (94 % power at a true 98 % with the 140 fresh windows):
  - To gate at 97.5 instead, add ~150 HumanEval+-like code windows from a second set. That is a download, so it needs
    Kamen's OK, and must be amended before Wed 18.
  - Otherwise, accept ~56 % power.

## 4. The decision rule
- **HumanEval+ fresh (the gated headline):** CERTIFIED when all of these hold:
  - every band's lower bound is >= HE_LINE (97.0);
  - every single depth 0-30 has a point estimate >= 96.5 % (at a flat 98 % this guard fails ~never; it catches a dip
    at one depth);
  - the run is valid (section 5).
  - The label adds "98 met" when every band's point estimate is >= 98.0.
- **MBPP fresh:** MBPP_LINE, one number.
  - Registered default: no line. MBPP is reported against its own computed exact ceiling. NewIdeas' arithmetic puts it
    near ~94-95 %: 29 % of MBPP rows are unsure, and the head reaches 79.6 % on them even with the true state.
  - Setting MBPP_LINE = L gates MBPP with the HumanEval+ rule at line L. Its 300 windows give 0.97 power at L = 97.5,
    or 0.81 in the stress case.
- **Everyday text:** reported with the same statistics, never gated.
- **The tree line (PREREG amendment 6):** T = tree-16 kept per check / chain-16 kept per check, each under its own walk.
  PASS needs LB(T) >= 1.021 and T >= 1.052 on both HumanEval+ and MBPP. Walk-strict coverage per band is reported beside
  it, and for a chain it equals the accuracy gated above.
- **Always reported:**
  - each band's point estimate and lower bound;
  - the margin bins;
  - kept/32 and kept/16, with window-clustered 95 % intervals;
  - the engine's tokens per check;
  - the dev sets, next to the fresh ones.
- **The verdict line:** `CERTIFY98 HE+: CERTIFIED [98 met] | NOT CERTIFIED (bands ...) | VOID (reason)`, then one line
  each for MBPP and everyday text.

## 5. Validity (otherwise VOID: fix, then a new run)
- The scorer reproduces t30b on the stored cells (section 1).
- The fresh-set manifest was frozen before the run, and the held-out and contamination checks pass.
- The features come from the runtime path. On every answer row, the target's top-1 equals the next token (greedy).
- The candidate's pins (checkpoint, commit, settings, scorer) equal the amendment's.
- One candidate per look (section 2).
- The engine check is within +-3 %. Failing it voids only the kept/32 claim, and the label says so.

> (a quote from the project owner, kept private)
- **METHOD CERTIFIED (HumanEval+)** needs all three parts, each scored on fresh windows:
  - **(a) The ceiling:** the 125B's head, fed the true fresh state at every depth (the oracle), has every band's lower
    bound >= HE_LINE on HumanEval+ fresh.
    - That shows this reader can reach the line at every depth. The oracle holds 97.8 flat on the 24 dev windows.
    - If it fails, the method also needs a reader lever, and the report names it.
  - **(b) The mechanism at small scale:** on the 2B sandbox (a second model family), the method at full budget gets
    within 1.0 pt of the 2B's own oracle at 23-30.
    - The test is window-paired: the one-sided 95 % lower bound of the method minus the oracle must be >= -1.0 pt.
    - Band 0 must stay within 0.3 pt of the 2B baseline.
    - It runs on the 2B's own fresh windows with this protocol.
  - **(c) The transfer at the 125B:** a partial 125B run at matched steps (10-25 % of the full budget).
    - It passes its pre-registered stop line on the dev sets, and beats the baseline at the same steps at 23-30 on
      HumanEval+ fresh (window-paired, one-sided p < 0.05; this counts as one look).
    - Its closed share of the chain-to-oracle gap must be >= 0.8 x the 2B's share at the same relative step.
- **The label:** "METHOD CERTIFIED (HE+): ceiling at the line, mechanism shown at small scale, transfer seen at partial
  scale".
  - The full drafter's own certificate follows the build, under this same protocol.
  - The projection to the full budget is reported, never certified.

## 7. Timeline
- **DONE early, Sun 01:5x (CPU):**
  - tools/certify98.py. t30b reproduces exactly: bands to 0.0000 pt, kept/32 and kept/16 to 1e-6.
  - tools/certify98_contam.py. The dev sets are CLEAN against all 3,072 n1112 captures.
  - tools/certify98_manifest.py, which draws the fresh ids.
- **Wed 18 night (125B, 45 GB), ~3-4 h:** capture the fresh sets on the runtime path, ~590 files (HumanEval+ 140 +
  MBPP 300 + everyday 150).
  - First decode every answer greedily (~121K tokens, ~1 h), then run the BANK pass (~2.1 h at ~12.6 s per file).
  - Then freeze the manifest and file the proof-run amendment, naming the candidate and its pins.
- **Thu 19 - Fri 20:** score the one candidate (minutes). Run the engine check on >= 20 HumanEval+ fresh windows (~30
  min of GPU).
  - The candidate's single pass is BigMoE's TOPK + TREE dump (head only, ~12 GB, ~45-60 min), run only after the
    proof-run amendment (PREREG amendment 5). certify98.py scores the TOPK file, which equals the cells files on the dev
    sets. Tree-16 kept/32 under the walk is information.
- **Sat 21:** the verdict.

## 8. The v1' confirm (item 2: plan only, not run)
- See EvalSuite docs/PREREG_certify98_2026-11-15.md, appendix A.
