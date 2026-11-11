# A data-aware 2.4-bit trellis expert store did 1.6-2.5x the damage its error ratio predicted (map screen 7: KILL)
- Date: 2026-11-04 22:28 EET. Session: NewIdeas.
- Claim: a FAST-only store of trellis-coded experts was priced from its weight-error ratio against the 3-bit truncated
  codes. It measured much worse than that price. The store is the 2.40 b mix: gate / up at TRL-2 (2.06 b), down at
  TRL-3 (3.07 b), LDLQ-fitted, Fit12's encoder.
- Number:
  - Screen 7 put the mix on 4 of the 48 layers (7 / 18 / 30 / 41) and scaled to 48 with an MSB-2 calibration pair.
    Flips vs REF2-T at 239 slots, pre-registered in-frame read: 1.45 / 9.54 / 10.44 % (code / IFEval / multi) against
    the lines 1.34 / 5.42 / 3.69 -> BETWEEN / KILL / KILL.
  - At the same 4 layers, the mix did 0.48 / 0.44 / 0.58 of MSB-2's damage. In the quality model's units (error ratio
    vs MSB-3, both judged against the 4-bit weights) that is 2.72 / 2.55 / 3.15, where the measured weight errors
    (held-out, output-weighted) said 1.62-1.87.
- So what: the "code under 10 MB per kept token at 24 GB" route through a trellis store is closed; today's FAST stack
  stays the cheapest measured code PASS (~19.5 MB [P]). Error ratios cannot rank data-aware codes against truncation
  for this model without a KL screen. A selective store does not rescue chat at 24 GB or at 45 GB all-resident [G,
  unchecked].
- How: NewIdeas scripts/s7_read.py, data/ledger/s7_read.txt (d37b3ef); arms run by Breakthrough; store by Fit12.
  Re-derived by Breakthrough's read-out (8a473b0): the same numbers.
- Prior art: data-aware trellis quantization (QTIP-style) is published. The miss between its output-weighted error and
  the KL on out-of-distribution prompts is our measurement, not searched outside our notes yet.
- Follow-up (22:4x): it is not the inputs. Fit12's output-weighted judge on the map's own answer rows puts the store's
  whole-expert output error at only 1.1-1.3x MSB-3's on code and 1.6-1.8x on chat, so the KL damage per unit of
  output error is 1.4-2.4x higher than for truncation errors. Part of that is noise (the KL-side ratio carries
  +-0.5-1.0). The two arms made identical reads (the harness owner checked the run logs: 368,983 row-rank pairs,
  equal window by window), so it is not coverage. Scaled by the measured MSB-2 pair, the output-error ratios
  predict 0.29-0.55x the measured damage (code / IFEval / multi 3.5x / 1.8x / 2.4x more measured). The same
  calibration fits MSB-2 within 1.0-1.3x. So on this MoE the data-aware trellis errors cost ~2-3.5x more KL per
  unit of expert-output error than truncation errors; the mechanism is open. By the pre-registered rule the
  trellis store family stops here; no store build or disk is needed.
- Follow-up (2026-11-05 10:3x, re-derivation): MoonShots re-derived the ratio independently (its M-43, cd5f64b).
  - It rebuilt the 4-bit, MSB-3 and MSB-2 weights from the original 4-bit files, read the trellis store, and measured
    MSB-2's output error at 4.98-5.10x MSB-3's, as the formula assumed.
  - It pooled the store's error by error over the 4 layers rather than by simple layer means: 1.29-1.30x on code,
    1.72-1.82x on chat. Late layers carry more error, so the layer means understated it.
  - Its direct ratio (KL damage over output error, trellis vs MSB-2, no model terms) is 3.18x / 1.65x / 2.34x (code /
    IFEval / multi; code noisy, +-1.3; IFEval and multi clear 1 by more than 2 SE). The first estimate above was 3.5x
    / 1.8x / 2.4x.
  - Covering both measurements, the trellis errors cost ~1.7-3.5x more KL per unit of expert-output error than
    truncation. The pooled figures (1.7-3.2x) are the better estimate.
