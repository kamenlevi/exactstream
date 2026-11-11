# MISSION 21: CAN THE 125B CERTIFY ITSELF MID-PASS? A focused CPU screen that decides a big question.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~18:3x. Best pasted once EARLYTAP's raw taps exist, ~19:40; until then, set up the pipeline.)

**Why.** Your CACHE.md showed the 125B certifying its own rows is what moves the SSD lines: the two-pass check cut 12 GB's SSD reads 5.5x. Cloud Fable then argued that pre-check signals top out near the 73.1 % certificate ceiling, so the only single-pass route to 150 tok/s at 12 GB (and a much bigger gate at 45 GB) is a **mid-layer self-certificate**: read at layer L whether the row will end sure, and give that row top-2 experts for layers L+1..48.

**What already argues against it**
- **L1-09 (Oct 24):** reading the token out after layer 40 with the model's own final mixer + LM head agreed with the full model on only 5.7 % of code positions, and even after layer 47 only 92.5-95 %.
- **R-32 (Oct 27):** a layer-12 probe predicting rejected rows did no better than the drafter's own features (11.1 vs 11.0 %).

So a raw logit lens is dead. Untested: a **trained sureness probe** on the row's own state at layers 25-35, predicting "the 125B ends sure" (p1 >= 0.99) rather than the token. That is the question here.

**The data**
- Tonight MTP Head's EARLYTAP re-extracts the 125B's raw per-layer taps (layers 5 / 15 / 25 / 35 / 45, the mean of the 4 HC streams) for L1-20's 109 code windows and 100 held-out MBPP answers, with each row's final top-32 (REPORT 'EARLYTAP'; ask MTP Head for the paths when they land).
- For labels and flips: decision_map/out (baseline.json, floor_rows.json, topk2 / topk5), relax_positions.json, and the drafter's cf per row from the HPD-0 walks.

**Deliver**
1. **Probes per layer** (logistic first, then a small MLP if logistic shows signal), trained on the 109 windows and judged on the 100 held-out MBPP answers. Targets: (a) the 125B ends sure (p1 >= 0.99); (b) top-2-certifiable (the map's p1 under top-2 >= 0.99).
2. **Coverage at <= 0.1 % false certificates**, per layer, against:
   - the drafter's cf: 57-64 %
   - the certificate ceiling: 73.1 %
   - the oracle: 86.8 %
   - the cheap pass's own p1: 73.4 % at top-2, 89.8 % at top-5
3. **Fused with the pre-check signals** (cf, copy state, token kind): does the layer-25 / 35 state add coverage beyond them?
4. **The verdict and the next step:**
   - PASS if a layer <= 35 probe certifies >= 80 % of code rows at <= 0.1 % false certificates. That opens cloud Fable's 150 line; give Breakthrough a one-night GPU confirmation on the map harness (per-layer k after layer L, flips counted).
   - KILL if it is < 70 % at layer 35.
   - In between: say what the probe adds to the two-pass check and to the 45 GB gate.

**Rules**
- CPU only, no downloads, writes only in ~/FableGod.
- Mark every number as measured / arithmetic / judgement.
- Check ~/TRIED/TRIED.md (L1-09, R-32, K-HEAD, L1-17b) and say why each kill reason does or doesn't apply.
- Keep it short and decisive. This is a screen, not a survey.

**Inputs**
- ~/FableGod/CACHE.md, CLOUD_TASK_PREDICTOR.md, SAFARI.md, FLOOR_125B.md, floor_rows.json, decision_map/out, relax_positions.json
- ~/TrainLab/REPORT.md (EARLYTAP, K-HEAD, MATCH) and EARLYTAP's raw-tap files; ~/TrainLab/data/kh_read.txt
- ~/Leap/TABLE_350.md (R-32), ~/Leap/QUEUE_L1-09.md if present; ~/TRIED/TRIED.md
