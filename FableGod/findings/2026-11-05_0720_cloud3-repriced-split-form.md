# Cloud Fable 3's strict expert table, re-priced: x2.5-3.3 the SSD bytes and +4-13 % on code, not +20-40 %; a split form keeps its compute cut at today's bytes
- Date: 2026-11-05 07:20 EET. Session: Fable C (audit), recorded by the orchestrator.
- Claim: Cloud Fable 3's overnight candidate, c3_b2d, is a three-band table of experts per row keyed on the drafter's
  confidence, plus union members (`FableGod/cloud3/live/cloud3/REPORT_P3.md`). It was priced against the wrong baseline
  and the wrong expert term. Re-priced on today's 45 GB FAST preset (QOM cf 0 + the gated bonus row), it is a small,
  code-only lever. Separating the rule's two effects keeps its compute cut without its extra SSD reads.
- Number:
  - Bytes [P, two independent replays]: against today's preset the strict table reads x2.5 / 3.2 / 3.3 the SSD bytes per
    kept token at 45 GB (code / IFEval / multi) and x2.4 / 4.4 / 5.5 at 24 GB. "Same bytes" held only against its own
    resident-keep twin, and its tile ratios were taken against EXACT. Its cost model counted the extra reads as a saving.
  - Time [P]: the live FAST expert term is 22-27 ms per check (9.7 live rows, ~91 % of tiles kept), not the 16-row EXACT
    38-43 ms. Net for the reach variant (c3_b2d_reach07) on code: +4 to +13 % tok/s (110-117 -> 115-132), not 117 -> 140-165.
  - Quality [K, Wednesday's FLOOR night arms]: NEW-1, a band table milder than c3_b2d in every band, measured
    1.12-1.28 / 5.1 / 3.6 % all-row flips (lines 1.34 / 5.42 / 3.69: BETWEEN), where the cloud's pricing gives the same
    table 0.00-0.06 % direct. Cutting the surest rows moves flips onto the unsure rows downstream: an additive floor, not a
    multiplier. Expected c3_b2d [G]: code 1.3-1.5 %, IFEval 6-8, multi 4.5-6 (BETWEEN / KILL on chat).
  - Split form [P, CPU replay]: keep the preset's loads and its resident keep, and drop only the resident keep on rows the
    drafter is sure of (cf >= 0.9: ranks 0-1 + union members). Code tiles x0.47 of the preset's (137 per kept token vs
    c3_b2d's 141) at x1.01 the bytes; IFEval x0.74, multi x0.79. Priced: +15-19 % on code FAST (halved +7-9 %); at
    cf >= 0.99, +11-14 %. Its quality is two measured cuts combined (q1u_0.9_live on the sure rows, QOM below), not a new unknown.
- So what: nothing gets built on "same bytes" or on the 38-43 ms term. The decisive test is 5-minute decision-map arms
  (split_0.9 / split_0.99 beside c3_b2d / c3_b2c, code decisive, chat as control; PASS = code all-row flips <= 1.34 %),
  plus an engine log of distinct experts per layer call, exact vs FAST (PASS needs FAST >= 0.85x exact's). Not run yet.
  Whatever the verdict, it goes here.
- How: `FableGod/fablec/AUDIT_1105.md` sections 0-3 and 7; `audit_split_replay.py` and `audit_k8_bands.py` with their
  outputs beside it (numpy, seconds, CPU). Breakthrough rebuilt the cloud's masks from the spec text, bit-equal on 36 / 36
  windows.
- Prior art: confidence-keyed expert counts per row are adapted (PSR, AcceptMoE, OEA, LExI; audit section 4). The split
  is engineering on our own cache-conditional keep rule; not searched outside our notes yet.
- Follow-up (2026-11-05 10:36, map test on code): both split forms PASS on code [K Breakthrough, 12 code windows,
  3,366 rows, all-row flips vs the baseline argmax, line 1.34 %]. These use the engine's own last-row convention (DRAFT17
  gates every last row), slightly gentler than the replay priced above.
  - Cut on rows with cf >= 0.99: 0.92 % (31 rows); code tiles x0.628 of the preset's.
  - Cut on rows with cf >= 0.9: 1.31 % (44 rows), just under the line; code tiles x0.508.
  - The preset itself (QOM-KEEP + DRAFT17, first map run at 45 GB): 0.51 %. Each split costs a significant +0.41 / +0.80
    points against it on the same rows (paired z +2.47 / +3.94), and both stay under the line.
  - Still open: the chat control (Friday, 36 windows), an engine log of distinct experts per layer call (exact vs FAST),
    and the engine A/B (tok/s >= +8 % at kept within 1 %). The tok/s gain is not measured yet.
  - Side result: this was the first map run on the store converted to the native layout. It reproduced the stored
    routing on 100 % of cells and the baseline top-1 on 144 / 144 rows.
