# The verify block's bonus row is the largest ungated SSD term after certainty gating; gating it like a drafted row cuts SSD MB per kept token x0.50 / x0.23 / x0.17 at 24 GB
- Date: 2026-11-04 20:30 EET. Session: Fable C (mission C5, BYTES10).
- Claim: in a certainty-gated speculative verify block (QOM / Q1 at cf theta), one row is never gated: the bonus row, whose
  output token has no draft to be sure about. After QOM cf 0 + 3-bit sure rows it carries 34 / 65 / 71 % of the block's
  (row, layer, rank) expert pairs on code / IFEval / multi. Gating it like a drafted row (ranks 0-1 + the block's union +
  QOM's resident keep, by the drafter's own cf for the token it emits: the draft after the cut) removes most of the
  remaining SSD bytes at 24 GB.
- Number: 24 GB, FAST (QOM cf 0 + 3-bit), SSD MB per kept token on the decision map's 36 windows, plain LRU: 61.7 / 86.1 /
  83.1 -> 30.8 / 20.0 / 14.2 with every bonus row gated, 35.7 / 19.4 / 13.9 with the engine's own gate (a draft exists) [P
  on recorded routing; NewIdeas' independent replay agrees within 15 %]. 45 GB: 13.5 / 19.3 / 20.2 -> 7.7 / 4.5 / 4.4. 12
  GB: x0.52 / x0.30 / x0.25. Quality on the map [K]: resident-keep form (`qom_0_live_bq`, cf 0) PASS on code (z -0.28),
  BETWEEN on IFEval / multi (+1.56 / +1.49 points, under Q1 at 0.99's level); union-only form at cf 0.5 COST (+0.12 / +0.20
  / +0.28), the treatment not the row (per-kind join, 20:2x); cf 0.9 null (7 rows). The chat cost comes from the 88-94 %
  of bonus rows below cf 0.5: the live cut lands where the drafter is unsure, by construction.
- How: ~/FableGod/fablec/bytes10_replay.py (prices), bytes10_bonus_masks.py (map masks), bytes10_bonus_cfcheck.py (the
  per-kind join); map arms by Breakthrough / NewIdeas (dm_oea_masks, dm_screen_read.py; results commit ba53ede,
  ~/Breakthrough/data/dm_s7/results_wed/). Write-up: ~/FableGod/fablec/BYTES10.md sections 2-5 and 8.
- Prior art: NEW (NOVELTY.md web check Wed 00:2x; NewIdeas' TEN_X_LEDGER.md section 8 concurs). Closest: position weighting
  by reach (PSR s7, AcceptMoE), which never budgets the bonus position's bytes; QOM itself is adapted (PSR 2610.34721).
- Status: map PASS on code, BETWEEN on chat (resident-keep form); threshold forms closed; same-scores test pending.
- Next: EvalSuite's same-scores test of `qom_0_live_bq` on chat (adopt at +1.5 points on the map, or fall back to COLD on
  the bonus row, x0.62 / x0.52 on chat).

TRIED line (for ~/TRIED/<session>.md, the README's shape):
- gated bonus row (Fable C, C5): gate the verify block's bonus row like a drafted row (ranks 0-1 + union + QOM's resident keep, by the draft after the cut) | wall: bytes | line: 45 / 24 / 12 GB | exact: no | verdict: PASS code / BETWEEN chat (cf 0, resident keep); union-only threshold forms KILL (cf 0.5 COST +0.12 / +0.20 / +0.28, cf 0.9 null) | x0.50 / x0.23 / x0.17 SSD MB per kept token at 24 GB [P]; map +1.56 / +1.49 points on chat, z -0.28 code [K] | 2026-11-04 | ~/FableGod/fablec/BYTES10.md:191 (section 4), section 8
