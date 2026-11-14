# PREREG: WEEK98 day-1 CEILING test (Sat 14 Nov 2026, ~15 min)

Written before the run. Charter: ~/FableGod/deepdraft/WEEK98.md (Kamen 15:0x: by Sat 21 Nov, know the method that gets
the drafter to >= 98 % right per guess at every depth up to 30). Script: scripts/dd_ceiling.py. Data:
data/deepdraft/ceiling/.

## Question
- Is 98 % per guess at depth 30 blocked by the 125B's own near-ties (its top-1 and top-2 almost equal)?
- Or are t30b's misses on rows where the 125B was sure, so they are learnable (what the drafter sees, how fresh)?

## What is measured
- **margins** (numpy, ~1 min): the 125B's margin m = lp[top-1] - lp[top-2], in nats, on every answer row.
  - It comes from the captured top-32 log-probs. Sets: MBPP 100 (the eval set) and t30b's 113 held-out n1112 files.
  - The HumanEval+ windows carry no log-probs.
  - Also: how often the answer token is the 125B's argmax. The capture is greedy, so it should be every row.
  - Bins: < 0.1, 0.1-0.25, 0.25-0.5, 0.5-1, 1-2, 2-4, >= 4 nats.
  - A near-tie is m < 0.25 (top-2 >= 78 % of top-1). m < 0.1 and m < 0.5 are reported too.
- **misses** (one eval pass of t30b over MBPP 100, ~2-3 min, ~10 GB): every cell of the eval (chain to depth 31, anchors
  from gen - 2), kept.
  - Per cell: right or not, the 125B's margin at the target row, and the draft's rank in the 125B's top-32 there.
  - At band 0 and band 23-30:
    - accuracy. Harness check: it must equal ttt30b/eval.json's MBPP bands, 94.01 / 72.13 for t30b.
    - the miss rate per margin bin;
    - the share of misses on near-ties, on confident rows (m >= 1 nat), and on the 125B's runner-up;
    - the tie-limited ceiling = 1 - (share of rows with m < 0.25) x (band 0's miss rate on those rows). That is every
      confident row right, with ties resolved as well as the freshest guess resolves them;
    - close-call accuracy: a draft within delta = 0.1 / 0.25 / 0.5 nats of the 125B's top-1 also counts. This is the
      lossy lever's best case per guess.

## Lines (result -> action)
| Result | Lines (MBPP 100, band 23-30 unless said) | Action for the week |
|---|---|---|
| TIES BLOCK | the tie-limited ceiling < 98 % | The method must include close-call acceptance (lossy; EvalSuite's gate; Kamen's call): Breakthrough prices it at the 98 % line, using the delta that gives >= 98 % here. The learnable share still goes to the depth levers. |
| LEARNABLE | the ceiling >= 98 % AND >= 60 % of the misses are on confident rows (m >= 1 nat) | The week goes to what the drafter sees and how fresh its view is: taps, fresher features, a partial trunk pass on drafted rows. |
| MIXED | anything else | Both tracks, sized by the two shares. |

- Information only:
  - band 7-10 and band 15-22, to see where the misses turn from ties to confident rows;
  - the n1112 held-out margins, to see whether MBPP's ties are typical.
- Order:
  - margins now (CPU, one process, < 1 GB);
  - misses right after t30x4's verdict eval, beside TAPSQ (~16 GB + ~10 GB on the ~42 GB floor).

## Result (Sat 14 Nov 15:19): LEARNABLE. The 125B's near-ties cost ~1.1-1.5 pt; the depth gap is on rows it was sure of
- **Part 1, the margins** (e4f9cd3, data/deepdraft/ceiling/margins.json):
  - The answer token is the 125B's argmax on 100.00 % of rows (the capture is greedy).
  - Near-ties (m < 0.25 nats) are 2.08 % of MBPP 100's 15,641 answer rows and 2.03 % of the n1112 held-out's 32,093.
    m < 0.1 (exact bf16 ties) is 0.82 / 0.65 %.
  - m >= 4 nats (top-1 >= 55x top-2) is 71 / 73 %.
- **Part 2, t30b's misses on MBPP 100** (misses_t30b_mbpp100.json, cells_t30b_mbpp100.npz; 205 s, peak 10.1 GB):
  - Harness: the band accuracies equal ttt30b/eval.json's MBPP bands: 94.01 (band 0), 79.61 (7-10), 74.38 (15-22),
    72.13 (23-30).

  | | band 0 (first guess) | band 23-30 |
  |---|---|---|
  | misses, pt of all cells | 5.99 | 27.87 |
  | on near-ties (m < 0.25) | 1.13 pt (18.9 % of misses) | 1.48 pt (5.3 %) |
  | on rows the 125B was sure of (m >= 1) | 2.01 pt (33.5 %) | 21.12 pt (75.8 %) |
  | on rows it was overwhelmingly sure of (m >= 4) | 0.08 pt | 11.18 pt (40 %) |
  | the draft is the 125B's runner-up | 72.9 % of misses | 36.5 % |
  | the draft is outside the 125B's top-32 | 0.0 % | 7.9 % |
  | miss rate on near-ties / on m >= 4 | 54 % / 0.1 % | 65 % / 16.1 % |
  | close-call acceptance, delta 0.1 / 0.25 / 0.5 nats | 94.34 / 95.36 / 96.22 % | 72.32 / 73.06 / 73.78 % |
  | tie-limited ceiling | 98.87 % | 98.76 % |

- **The lines:**
  - The tie-limited ceiling is 98.76 % at 23-30, >= 98 %, so ties don't block.
  - 75.8 % of the 23-30 misses are on confident rows, >= 60 %.
  - **Verdict: LEARNABLE.**
- **What it means for WEEK98:**
  - Deep in the chain, t30b mostly loses track of tokens the 125B was sure of. 11.2 pt of 23-30's 27.9-pt gap sits on
    rows where the 125B's top-1 was >= 55x its top-2.
  - Close-call acceptance buys little there: +1.65 pt at delta 0.5 at 23-30, against +2.2 at band 0. It isn't the
    week's lever.
  - The week goes to what the drafter sees and how fresh its view is, as pre-registered: taps (TAPSQ, running),
    fresher features, a partial trunk pass on drafted rows.
- **The tie tax still matters for exact 98 %:**
  - Even the first guess misses about half of the near-ties: 54 % miss rate, about 1.1 pt.
  - So exact 98 % at depth 30 leaves only ~0.8-0.9 pt for every other miss: the confident rows must be nearly all right.
  - Close-call acceptance (lossy, EvalSuite's gate) is the natural way to take back the ~1.1-1.5 pt tie tax at the end,
    not the lever for the depth gap.
- HumanEval+ has no captured log-probs, so this is MBPP's picture. MBPP's depth-0 accuracy (94.0) is far below
  HumanEval+'s (97.9).
