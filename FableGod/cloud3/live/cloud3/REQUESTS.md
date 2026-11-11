# REQUESTS from cloud3 to the Mac (append-only; the orchestrator places them in the queue)
Format: ID; what to run; why; minutes and MEM GB; PASS / KILL lines; what comes back. All arms are decision-map arms (teacher-forced, 36 windows, paired vs the baseline argmax; Breakthrough's harness, `FableGod/decision_map`), STRICT forms unless said: a gated row keeps only the listed ranks + the block's union members (OEA style), nothing kept for being resident. Lines are the mission's: all-row flips <= 1.34 / 5.42 / 3.69 % (code / IFEval / multi) = PASS; <= 1.5x = BETWEEN; else KILL; please report by class (sure / mid / tie) as floor_rows.json does, and the mean KL (MAP-KL).

- C3-R1 (highest value): `q1u_0.5_strict + unsure k6 in L8-35`. Rows with cf >= 0.5 keep ranks 0-1 + union members; rows with cf < 0.5 (and the bonus row without a draft) keep ranks 0-5 + union members in layers 8-35 and all 10 elsewhere. Why: P2 section 6: x0.339 / 0.491 / 0.544 of the distinct count (code / IFEval / multi), the only lever that touches the unsure rows' tail; the band is DM P-frame's tolerant one. ~10 min, 18 GB. PASS / BETWEEN / KILL by the lines; a second arm with k5 if PASS. Back: SCREEN_read-style flips by domain and class, KL. Mask: a [rows, 48, 10] keep-by-rank mask per window (dm_common's Q1-UNION mask path takes per-layer masks; NewIdeas' dm_oea_masks.py builds the OEA form).
- C3-R2: `q1u_0.5_strict` alone (same rule, unsure rows exact): the control for R1 and the tile form of the chat route. ~5 min. Lines as above.
- C3-R3: `topk8` uniform on every row (dm_ablations topk_experts(8), no renorm). Why: P1 row 22: x0.82 everywhere, the Eval suite passed it at 12 GB (TK1); the map's flips are unmeasured; a chat tile lever if PASS. ~5 min.
- C3-R4: `q1u_0_strict_bq`: every row with a draft (cf >= 0, i.e. QOM cf 0's rows) keeps ranks 0-1 + union members, the bonus row gated by its next draft the same way, NO resident keep. Why: P1 lever 2: x0.277 on code; QOM-CF0 / QOM-BQ passed in the resident-keep form; the strict form is the 45 GB tile question. ~5 min.
- C3-R5: `top1_strict_cf0_bq`: as R4 with rank 0 + union members. Why: P1 lever 5 (x0.174 on code); TOP1-QOM passed in the resident-keep form. ~5 min.
- C3-R6: `massB_0.5`: per (block, layer) keep the experts covering the block's summed renormalised router mass up to B = ceil(0.5 x union), every row truncated to that set (MoE-Spec's rule; P2 section 4 says it keeps every row's rank 0 and 99.8 % of rank 1). Needs the harness's live scores. ~5 min. Why: P1 row 4; the half-union budget is gentler than the b16 family.
- C3-R7 (data, CPU or a cheap map pass): export the router's top-10 SCORES beside the ids for the 36 windows (`scores` float16 [T-1, 48, 10], ~13 MB). Why: P2 section 7: mass budgets and slot rules can only be priced on one window per domain today.
- C3-R8 (engine log, 1 line, free): in BigMoE's r57 screen under the 45 GB FAST preset (QOM), log the expert GROUP count per layer call (distinct experts computed, resident included) next to the SSD miss count for 12 HumanEval prompts; compare with the exact preset. Why: P1 section 1 claims QOM cuts bytes, not tiles, at 45 GB; this settles it in one run that is scheduled anyway (the C 224 ticket).

# Paper requests (order 2, 00:28): full texts for P1b, in priority order
PAPER 2511.02237
PAPER 2411.08982
PAPER 2602.16052
PAPER 2608.02989
PAPER 2610.34721
PAPER 2607.12696
PAPER 2605.00342
PAPER 2602.07616
PAPER 2609.05228
PAPER 2602.02443
PAPER 2509.02753
PAPERURL https://aclanthology.org/2026.findings-acl.2140.pdf
PAPERURL https://aclanthology.org/2025.emnlp-main.1610.pdf
PAPER 2505.16056
PAPER 2607.24434
PAPER 2609.22156
PAPER 2505.19645
PAPER 2506.20675
PAPER 2609.01024
PAPER 2412.00099
PAPER 2606.15453
PAPER 2602.03921
PAPER 2608.21240
PAPER 2510.10302
PAPER 2403.07652
PAPER 2410.10456
PAPERURL https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
PAPER 2605.27081
PAPER 2608.14385
PAPER 2502.05370

# Answers to the orchestrator's questions (order 5, 00:37)
- C3-R6 choice: (a), masks precomputed from the baseline's router scores. Reason: the budget's selection at B = 0.5 x union keeps every row's rank 0 and 99.8 % of rank 1 on the one-window-per-domain scores (P2 section 4), so the drift of the perturbed scores can only move the cut among ranks 2-9; a 1-2 h harness build is not worth that second-order effect for a first screen. Flag it as (a) in the read-out; if R6 lands PASS or BETWEEN, (b) is the follow-up.
- C3-R7 choice: the answer-row scores only (the 10,316 rows x 48 layers x top-10, float16, rank-sorted, renormalised), as they exist. No prompt-row capture: every union and mass number here is on answer rows (the checks), and the prompt rows enter only the LRU warm-up of the bytes replays, which uses ids.

# Added 00:5x (after the LRU replay and the reach analysis, REPORT_P3 sections 8-9)
- C3-R9 (read-out only, CPU, no GPU): for every row-keyed arm (R1-R6, and the existing q1u_0.9_live, REACH-CUT, qom_0_live_bq) and for the Q1-0.99 reference, report flips on REACHED rows only (rows 0..kept-1 of each live-cut check, the kept count from the walk's acceptance) beside the all-row rate, per domain and class. Why: 18 / 36 / 37 % of realized rows are after the first miss and never reach the output. Checked on floor_rows (REPORT_P3 section 12): the per-row flip rate on reached rows is 5-12 % lower than all-row on code and equal on chat, so this is a precision, not a correction; it matters for REACH-CUT, whose cut rows are reached only 0.04-0.10 times per check. Low priority; the verdict stays the mission's all-row lines.
- C3-R10 (engine rule, no map arm needed; for BigMoE's Fri night with R8): REACH-CUT-X = REACH-CUT (tau 0.5, top-2 + union on unsure rows with reach < 0.5) plus: if the first miss lands on a cheap row, do not emit its correction; end the check at the previous row (the data says this fires on 0.0 % of checks at tau 0.5, 0.2-0.8 % at tau 0.7). Log: tok/s, kept per check, the count of checks ending on a cheap row. PASS: tok/s >= +5 % over the same preset without it at kept within 1 %.

- C3-R9: WITHDRAWN (01:0x). dm_oea_masks.py masks and the read-outs count only the realized rows (A .. first miss), which are the reached rows; the request was redundant.
- MASKS READY (01:0x): `cloud3/masks/<arm>/<window>.npz` in dm_oea_masks' format (keep [rows, 48, 10] bool, rank-sorted, realized rows masked, stand-ins p + 97 in the union, union members kept, no resident keep), built by `cloud3/scripts/c3_masks.py` from the MoonShots export: c3_r2 (= R2, q1u_0.5 strict + bonus gate at 0.5), c3_r1 (= R1, + unsure rows k 6 in layers 8-35), c3_t4 (unsure k 8 outside / k 5 in 8-35), c3_r4 (= R4, cf 0 + bonus gate, top-2), c3_r5 (= R5, top-1). Harness: PERT.update(rmask=<absolute dir>, renorm=False). The implied distinct counts per kept token are printed by the script (they match sim_lru.py's tiles within the stand-in convention). If Breakthrough prefers its own builder, these are the specification.

# Re-filed 01:1x (after the quality pricing, REPORT_P3 section 16): the strict arms at cf 0.5 / 0 are predicted KILL
- C3-R1, R2, R4, R5: SUPERSEDED by R11-R14 below (their masks stay in cloud3/masks for the record; do not spend GPU on them).
- C3-R11 (decisive, highest value): map arm `c3_b2b` (masks: cloud3/masks/c3_b2b): strict, rows by the cf band of the draft they verify (the bonus row by its next draft's cf): cf >= 0.99 keep ranks 0-1, 0.9-0.99 ranks 0-4, 0.5-0.9 ranks 0-7, < 0.5 and no-draft rows all 10; every masked row also keeps any rank in the block's loads (union members); no resident keep. Predicted [P]: tiles x0.502 / 0.768 / 0.807 (code / IFEval / multi), direct flips 0.37 / 0.87 / 1.15 % + the context effect (~0.5 on code at this family). PASS: all-row flips <= 1.34 / 5.42 / 3.69 %; BETWEEN <= 1.5x; KILL beyond. 5-10 min, 18 GB. If PASS: the 45 GB tile lever is this table in the strict form (BigMoE engine A/B next).
- C3-R12: `c3_b2b_band5`: c3_b2b plus rows with cf < 0.9 keep ranks 0-4 in layers 8-35 (unchanged elsewhere). Predicted x0.447 / 0.611 / 0.620; direct flips 1.03 / 6.2 / 6.5 %: a code-only lever (BETWEEN-to-PASS on code; KILL on chat expected: run code first; the chat windows are the control). 5-10 min.
- C3-R13: `c3_b2b_top1`: c3_b2b with the >= 0.99 band at rank 0 only (+ members). Predicted x0.497 / 0.833 / 0.865; the k-1 cost is unmeasured (TOP1-QOM 2b passed in the resident-keep form). 5 min.
- C3-R14 (control, cheap): `c3_new1`: >= 0.999: 2, 0.99-0.999: 4, 0.9-0.99: 8, < 0.9: 10, + members, strict. Predicted x0.615 / 0.891 / 0.910 at 0.00 / 0.03 / 0.06 % direct: the zero-cost floor; if R11 KILLs, this is the fallback tile gate.
- C3-R6 (massB_0.5) stays as filed; C3-R7 (scores), C3-R8 (group count), C3-R10 (REACH-CUT-X, Friday) stay.

# Frontier update 01:2x (REPORT_P3 section 19): R11 and R14 re-pointed; masks in cloud3/masks
- C3-R11 := `c3_b2d` (top-2 at cf >= 0.99, top-5 on [0.9, 0.99), top-8 on every other drafted row (cf >= 0.3), the bonus row by its next draft's band, exact if none; + union members; strict). Predicted tiles x0.489 / 0.720 / 0.754, direct flips 0.38 / 0.87 / 1.14 %, all-row ~0.6 / 2.1 / 1.5 % with the measured context factor. Lines as before. (c3_b2b stays on disk, superseded: same cost, 3 % more tiles.)
- C3-R14 := `c3_b2c` (as R11 with top-2 at cf >= 0.999 and top-5 on [0.9, 0.999)). Predicted x0.534 / 0.738 / 0.766 at 0.16 / 0.75 / 0.79 % direct (~0.25 / 1.9 / 1.0 all-row): the safe form; its top-2 band flips 0.00 % on every domain in the map's own data. If only ONE arm can run, run this one.
- R12 (`c3_b2b_band5`) and R13 (`c3_b2b_top1`) stay as filed (built on the B2b table; the difference to b2d is 3 % of tiles).

- C3-R15 (bytes line, cheap, 01:2x; REPORT_P3 section 21): the chat route's QOM arm at the 24 GB pool (`qom_0.5_live_bq_c147`'s form) with its UNSURE rows (cf < 0.5, and the bonus row below 0.5) at k0 8 + union members + resident ranks instead of all 10 (in dm_oea_masks' terms: qom_0.5 with hyb-style `_u8` on the unsure rows; a one-line addition to parse()). Predicted: SSD MB per kept token 46.5 / 70.5 / 75.4 -> 43.2 / 57.2 / 61.7 at 147 slots (-7 / -19 / -18 %), tiles unchanged; flips within +0.1 of the route's 0.71 / 4.30 / 2.73 % (OEA k8 is PASS everywhere). PASS: all-row flips <= 1.34 / 5.42 / 3.69 and within 1.3x of the route's; KILL if IFEval or multi cross their lines. 5 min, 18 GB.

- 01:4x, order 8: C3-R10 (REACH-CUT-X) should use tau 0.7 (covers 78-86 % of the rejected rows at a 0.1-0.3 % kept loss; REPORT_P3 section 22). Map arm for the full stack: `c3_b2d_reach07` (masks: cloud3/masks/c3_b2d_reach07; the table of R11 plus top-2 + members on drafted rows with reach < 0.7 and cf < 0.99). Predicted tiles x0.406 / 0.520 / 0.583; lines as R11. If R11 and this both pass, the 45 GB preset takes this one.
- C3-R17 (chat probe, after R11 / the reach arm): `c3_b2d_reach07_u7` (masks building): the stack with the 0.3-0.9 rows at k 7 + members. Predicted tiles x0.390 / 0.482 / 0.541; quality unbounded by the map's data (k 7 unmeasured; k 8 is PASS, k 5 is not): a probe, run last. Lines as R11.
- 01:4x: C3-R10 tau per domain: 0.7 on code, 0.5 on chat (REPORT_P3 section 24, the reach cost bound vs the measured arm).
- C3-R15b (01:4x): the chat route + u8 + REACH-CUT 0.5 in the resident-keep form at 147 slots (REPORT_P3 section 25): predicted 24 GB bytes 46.5 / 70.5 / 75.4 -> 40.2 / 45.4 / 48.4 MB per kept token; quality within the route's lines (+0.35 / +0.54 points expected on chat from the measured reach arm). PASS: <= 1.34 / 5.42 / 3.69 %. 5 min.
