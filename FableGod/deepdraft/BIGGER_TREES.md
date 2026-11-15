> (a quote from the project owner, kept private)

> (a quote from the project owner, kept private)
is needed by Saturday.

## What we know tonight (dev, desk; research/week98/treek_curve.{py,txt,json})
- **Kept per check vs K** (first K best-first pops of BigMoE's real 32-node dumps, certify98 walk). K 15 reproduces S9c,
  and K 31 reproduces MTP Head's S11 KPRICE (16.263 / 9.031).

| K nodes (+ bonus) | HE+ kept | HE+ T vs chain-16 | MBPP kept | MBPP T | HE+ depth mean / p90 | MBPP depth mean / p90 |
|---|---|---|---|---|---|---|
| chain-16 | 10.625 | 1 | 6.588 | 1 | 15 | 15 |
| 15 (tree-16) | 11.263 | 1.060 | 7.487 | 1.136 | 12.6 / 15 | 10.3 / 15 |
| 24 | 14.588 | 1.373 | 8.556 | 1.299 | 17.7 / 24 | 13.1 / 21 |
| 31 (tree-32) | 16.263 | 1.531 | 9.031 | 1.371 | 20.8 / 31 | 14.6 / 23 |
| chain-32 (31 drafts) | 13.769 | 1.296 | 6.924 | 1.051 | 31 | 31 |
| 63 (S11 proxy) | 17.91 | 1.686 | 10.16 | 1.542 | ? | ? |
| 127 (S11 proxy) | 19.14 | 1.801 | 11.24 | 1.706 | ? | ? |

- **Same 32 rows: tree-31 vs chain-32 = x1.181 HE+ / x1.304 MBPP.** Trees pay more at 32 rows than at 16 (x1.060 /
  x1.136), because a chain's deep rows earn little after a miss.
- **HE+ trees are deep and narrow** (width 1.2-1.6 per depth at K 31), so a level-wise drafter needs about as many head
  steps as chain-32. Draft time is a term of its own and must be priced. MBPP's trees are wider (2.0-2.7 per depth)
  and shallower.
- **Real decoding:** MoonShots' TREE-E2E T32 matches the sim at 32 nodes (x1.343 / x1.367 on the 2B + s30; exact 264 / 264).

## The decision rule for K* (proposed; computed Tue evening once R(K) lands)
- **tok/s(K) = kept(K) / cycle(K)**, where cycle(K) = R(K) x c(K) x t_check32 + overhead + draft(K):
  - R(K) is Leap's check-time ratio K rows : 32 rows; c(K) is the tree overhead beyond c16 1.025.
  - t_check32 = 52.0 ms (exact floor), ~55 ms (exact realistic), 25 ms (lossy).
  - overhead ~4 ms (commit replay + seams; WEEK98's spec).
  - draft(K) = levels(K) x the head's step time at that level's batch. Report two cases: hidden (0) and not hidden.
- **K* = argmax tok/s on HE+** at the exact floor, with MBPP reported. A K above 32 is chosen only if it beats K 32 by
  >= 3 %, since the proxy and R(K) both carry error.
- **Break-even from S11's kept** (K 32 baseline): K 64 pays only if cycle(64) / cycle(32) < 1.10 on HE+ (1.125 MBPP).
  K 128 pays only if < 1.18 (1.24).
- **Prior:** at the floor, a check is bandwidth-bound. More rows touch more unique experts, and the dense side scales
  with rows once GPU-bound. So K* is likely 32-48 unless Leap's R(64) is under ~1.1.
- **Kept(K) beyond 32 must be REAL before Wednesday.** BigMoE's Tuesday dry run of the 127-node dump replaces S11's
  proxy (the same tool and gate as Wednesday). I ask that it run mbppbank as well as heplus24: about 15 + 6 min at
  ~55 ms per anchor.

## The certificate line for tree-K* (for EvalSuite's amendment, Wed ~18:00)
- **T(K*) = tree-K* kept / chain-16 kept**, each under its own walk on the same fresh windows (amendment 6's method):
  - delta-method LB, CR1, t(K-1);
  - pay line T >= R(K*) x c(K*) x (t32 / t16) / 1.03, i.e. the 3 % strict net margin, as in amendment 6a.
- **Next to it, per-band walk-strict coverage** (C1 extended to K*): band 0 >= 97.8 point, LB >= 97.0. The deeper bands
  are reported.
- **Exact:** tree verify, with real decoding at 32 nodes shown by MoonShots' T32.

## Instant information (the orchestrator's 21:5x ask): context lookup screen (research/week98/lookup_screen.*)
- **Proposers** (CPU, microseconds, draft-time only):
  - prompt lookup (the last 4 / 3 / 2 tokens' most recent earlier occurrence in the prompt + answer);
  - an n-gram table of the 125B's own n1112 answers (2,164 files; no dev window in it);
  - both together.
- **Upper bound, rescues not charged rows** (HE+ / MBPP kept/32 vs 13.77 / 6.92):
  - lookup + datastore: +8.7 % / +9.7 % (P = 1: +7.9 % / +8.2 %);
  - prompt lookup alone: +5.9 % / +5.3 %.
  - It recovers 16.2 % of the head's misses on HE+, but walk-strict bands 1-14 move only +0.2 to +0.8 pt. It is not a
    97.8 lever.
- **Line, written before the run:** PASS on kept, so a budget-correct screen follows.
  - Prior art: LEVER1-COPY-REPLAY's tree oracle +4.6 % fell to <= 0 in a budget-correct switch policy at 12 GB.
- **Monday screen (NewIdeas, CPU, <= 15 min):** the lookup token as an extra candidate in the tree-K best-first order,
  with its node charged. Read: kept at equal rows vs tree-K. PASS >= +3 %.

## FIRST CUT, Sun 22:0x: K* = 32 on HE+, 16 on MBPP; bigger never pays (research/week98/treek_toks_v0.txt)
- **Inputs:**
  - Leap's first-cut R(K) and c(K) (~/Leap/data/week98/treek_price.md, 05eea11). R = check / A16 (62 ms):

    | K rows | 32 | 48 | 64 | 96 | 128 |
    |---|---|---|---|---|---|
    | floor | 0.85 | 1.00 | 1.12 | 1.30 | 1.44 |
    | contract | 1.39 | 1.78 | 2.14 | 2.80 | 3.40 |
    | c(K) | 1.03-1.05 | | | | 1.06-1.16 |

    16 rows: floor 38.8 ms, A16 62 ms.
  - Kept: tree-16 / tree-32 real; 64 / 128 from S11's proxy; 48 / 96 log-interpolated.
  - Model: tok/s = kept / (R x c x 62 ms + 4 ms overhead), drafting hidden. Lossy = 25 ms x R_floor(K) / R_floor(32).
- **tok/s:**

| K (tree nodes + bonus) | HE+ floor | HE+ realistic | HE+ lossy 25 | MBPP floor | MBPP realistic | MBPP lossy 25 |
|---|---|---|---|---|---|---|
| 16 (tree-16) | 257 | 167 | 493 | **171** | **111** | **327** |
| **32 (tree-32)** | **277** | **174** | **542** | 154 | 96 | 301 |
| 48 | 248 | 143 | 492 | 140 | 80 | 277 |
| 64 | 229 | 123 | 456 | 130 | 70 | 259 |
| 96 | 203 | 96 | 408 | 117 | 56 | 236 |
| 128 | 186 | 80 | 375 | 109 | 47 | 220 |
| chain-16 / chain-32 | 248 / 243 | 161 / 153 | 474 / 475 | 154 / 122 | 100 / 77 | 294 / 239 |

- **Readings:**
  - HE+ picks K 32 under all three costs: +8 % (floor), +4 % (realistic), +10 % (lossy) over tree-16.
  - MBPP picks tree-16: tree-32 is ~10 % slower, because its kept gain (x1.21) is below R32 / R16 (1.36 floor).
  - Bigger never pays. At the floor, K 64 would need kept 21.65 on HE+ to tie K 32; S11's proxy says 17.91, about
    21 % short.
  - The byte floor saturates (expert union 110 -> 211 from 32 to 128 rows), but the real per-row cores grow ~linearly.
- **Drafting not hidden** (level-wise, ~1.15 ms per level, Leap P2):
  - HE+ tree-16 has 12.6 levels and tree-32 20.8, so tree-32's floor lead shrinks to ~+2 %.
  - MBPP (10.3 / 14.6 levels) still picks tree-16.
- **Walk-strict coverage at K 32:**
  - HE+: 98.5 / 96.6 / 93.7 / 94.3 / 96.3 / 96.2 / 96.9.
  - MBPP: 98.6 / 94.3 / 89.2 / 84.6 / 82.1 / 84.9 / 84.2.
  - C1's band-0 line (>= 97.8) holds at K 32 on both sets.
- **What could still change K*:**
  - Leap's refined R32 (C7c tonight, E2 Monday): HE+'s tree-32 lead over tree-16 is 4-8 %.
  - BigMoE's real 127-node dump on the dev bank (Tue): it would have to beat the proxy by > 20 % at K 64.
  - Neither looks likely to make K > 32 pay.
