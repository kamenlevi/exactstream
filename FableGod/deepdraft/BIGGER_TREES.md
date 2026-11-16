> **CORRECTED Mon 16 Nov 10:3x (`date`): every tok/s grid below that says "drafting hidden" is an UPPER bound that can't
> be reached.**
> - Pillar 3 closed. Draft bytes add in full on a bandwidth-bound check; no byte cut keeps kept within 1 % (M3).
> - Add the draft per block (full head, 0.66-0.92 ms per level): tree-16 8.3-11.5 ms, tree-32 13.8-19.1, K128 26.2-36.3.
> - HE+ like for like at the byte floor:
>   - tree-16 204-216 (not 257);
>   - tree-32 209-224 (not 277);
>   - K128 + early death 251-303 (not 410-455).
> - At the measured engine, HE+ tree-16 is ~144 tok/s (~150 with a hand-fused level), not 167 (research/week98/earlydeath_unhidden.txt).

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

## Lookup on top of tree-32 (Sun 22:1x; research/week98/lookup_tree_ub.{py,txt})
- At tree-32's first uncovered depth on the true path, the lookup (CTX + DS) is right on 21.6 % (HE+) / 14.6 % (MBPP)
  of the tree's misses.
- **Upper bounds with no node cost:**
  - as a 1-node leaf: x1.005 HE+ / x1.015 MBPP. Line (< +3 % on HE+) -> KILL.
  - with the head's chain continuing below it: x1.057 / x1.064.
- **Verdict:** the leaf form is CLOSED. The expanded form needs the drafting loop, and Wednesday's dump tool, to expand
  lookup candidates. It is worth at most ~+5-6 % before node costs, so it is not for this week's certificate; recorded
  for a later build.
- **EvalSuite's re-run (151d997) matches** T32/32 1.181 [LB 1.126] / 1.304 [1.283] and T32/16 1.531 [1.470] /
  1.371 [1.351].
  - Correction: the targets-only convention raises MBPP's deep walk-strict bands by up to +3.5 pt (HE+ +0.1-0.5).
  - Gating proposal:
    - T32/32 gated on HE+ AND MBPP;
    - T32/16 gated on HE+ only (MBPP prefers tree-16 and fails at every cost; information);
    - band-0 coverage at K* gated on HE+, MBPP reported.

## Lossy side of K* (Sun 22:1x; research/week98/tree_cc32.{py,txt}: tree_shape_cc.py at 31 nodes, close-call delta 1)
- **kept per 32-row check:**
  - HE+: chain-32 13.77; + CC 16.78 (x1.218); tree-32 16.26 (x1.181); **tree-32 + CC 16.91 (x1.228)**.
  - MBPP: 6.92; 8.52 (x1.231); 9.03 (x1.304); **9.50 (x1.372)**.
  - Additivity 0.57 HE+ / 0.70 MBPP (overlapping, as at 16 rows).
- **Lossy K*:**
  - HE+: tree-32 + CC, 679 tok/s at the 20.9 ms lossy floor (+4 ms) and 583 at 25 ms. 900 needs 22.4 kept: x1.32.
  - MBPP: tree-16 + CC (7.97 at ~15.4 ms) gives ~411 vs tree-32 + CC's ~382, so MBPP keeps K 16 on the lossy side too.

## K* depends on WHICH 32-row check (Sun 22:2x; treek_final.py, tested on the 32-node dumps: treek_final_test32.txt)
- **With today's MEASURED 32-row check** (G32T 1.625 x A16), tree-16 beats tree-32 even on HE+: 166.7 vs 149.5 tok/s.
  Tree-32 wins only once the 32-row check reaches the contract (~1.39) or its floor (0.85). So the bigger-trees payoff
  rides on CHECK55's 32-row work.
- **HE+ is flat between 24 and 32 rows:** 24 rows +1.1 % (floor) / +2.3 % (contract), on my INTERPOLATED R(24) [est].
  - A 24-row check pays only if the engine runs it below the 32-row cost. Padded to 32, it doesn't.
  - Leap's refined R(24) decides.
- **MBPP:** tree-16 under every cost (+4-6 % over 24 rows).
- **Tuesday:** run treek_final.py --costs <Leap's refined costs JSON> on BigMoE's real 127-node dumps (--dump-fmt). The
  same script prints K* per set and cost model, with a 3 % rule for K > 32.

## REAL 127-NODE TREES (Mon 09:2x; BigMoE's dry-run dump tree_t30b_heplus24_n127k8.npz; treek_final_he_n127k8.txt, treek_bands_n127_he.txt)
- **Kept per check under each tree's own walk, HE+** (nest check: the first 32 pops equal today's dump; K 16 / 32
  reproduce):

  | rows | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
  |---|---|---|---|---|---|---|---|
  | kept | 11.26 | 14.39 | 16.26 | 18.86 | 20.77 | 22.77 | 23.70 |

- **S11's proxy UNDERESTIMATED big trees:** K64 17.91 vs real 20.77 (-14 %); K128 19.14 vs 23.70 (-19 %).
  - A 64-row tree keeps what the true-state chain-32 keeps (20.77).
  - A 128-row tree keeps more than the flat-97.8 % line of a 32-row check (23.15).
- **The extra kept is longer runs, not better coverage per band.** Walk-strict coverage at bands 3-14 stays 93-96 % even
  at 127 nodes; K98's per-band conclusion holds. Bands 0 / 1-2 reach 98.9 / 98.3 at K 127. Mean tree depth is ~40, so
  many accepted runs go past depth 30.
- **K* under Leap's per-row costs is unchanged:** 24-32 rows (floor 279.5 / 276.5 tok/s). K48 / K64 are only -2.8 % /
  -4.1 % at the floor but -12 % / -18 % at the contract; the engine still picks K16.
- **What changes is EARLY DEATH's value (THINK #5, Fable B):** its byte price for a K128 check with dead branches
  dropped mid-pass is ~54 ms (probe at layer 16) / ~48 ms (layer 8).
  - With the REAL kept 23.7, the exact line projects to ~410-455 tok/s **(at the byte floor, probe untested, HE+ only)**
    (23.7 / (48..54 + 4 ms)). That
    is the first route in the arithmetic to 450 exact. Fable B priced it on the proxy's 19.1.
  - The probe is UNTESTED (PASS >= 90 % of off-spine nodes droppable at <= 2 % false drops, layer 16). MBPP's 127-node
    dump is pending.

## REAL 127-NODE TREES, MBPP (Mon 10:1x; treek_final_mbpp_n127k8.txt, treek_bands_n127_mbpp.txt)
- **Kept per check, MBPP:**

  | rows | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
  |---|---|---|---|---|---|---|---|
  | kept | 7.49 | 8.45 | 9.03 | 9.73 | 10.19 | 10.88 | 11.29 |

- **S11's proxy was RIGHT on MBPP:** K64 10.16 / K128 11.24, within 0.5 %. So HE+'s extra lift over the proxy (+14-19 %)
  is HE+-specific: long, confident runs, mostly docstrings copied from the prompt.
- **K* MBPP = tree-16 under every cost.**
- **Walk-strict coverage at K127, MBPP:** 99.6 / 97.2 / 92.4 / 87.5 / 86.8 / 88.7 / 86.6. Knowledge-limited, as known.
- **EARLY DEATH on MBPP:** K128 at ~48-54 ms by bytes + 4 ms is 11.29 / (52..58) = ~195-217 tok/s vs tree-16's 171 at
  the floor: +14-27 %. HE+ is ~410-455 vs 277: +48-64 %.
  - All at the byte floor, probe untested.
  - The premise holds on both sets (smaller on MBPP), so the Tuesday-night dev pass is CONFIRMED.

## CORRECTED Mon 10:3x (`date`): early death's projection with drafting NOT hidden (the orchestrator's number check)
- **The ~410-455 was wrong.** It kept drafting HIDDEN, but on a bandwidth-bound check the draft's bytes add in full (my
  own Pillar 3 point). A K128 tree is ~39.6 levels deep. At the full head's 177.6 MB per level, that is ~26-36 ms of
  draft per block (0.66-0.92 ms per level, hand-fused to today's sync-free; a lower bound, since wider frontiers read
  more experts).
- **Like for like, HE+, byte floor, overhead 4 ms, drafting NOT hidden** (earlydeath_unhidden.txt):

  | config | kept | draft ms | tok/s |
  |---|---|---|---|
  | tree-16 | 11.26 | 8.3-11.5 | 204-216 |
  | tree-32 | 16.26 | 13.8-19.1 | 209-224 |
  | K128 + EARLY DEATH (check 48-54 ms by bytes) | 23.70 | 26.2-36.3 | **251-303** |

- **Corrected line:** early death still leads by ~+20-35 % (same draft-cost assumption on both sides), but it does NOT
  reach 450. Label: at the byte floor, probe untested, HE+ only.
- 450 exact needs early death AND a smaller drafter (fewer bytes per level, same accuracy) AND the check near its floor.
- With unhidden drafting, the best K under early death may sit below 128, since depth costs draft time. Compute it when
  the probe's per-K price exists.
- **Tuesday night's dev pass is unchanged:** early death remains the best exact route, and the pass costs ~35 min of an
  otherwise dark night.
