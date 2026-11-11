# MISSION B5 (Fable B): FORESEE THE FIRST MISS.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~23:1x.)

**Why.** MoonShots ran three replays tonight (M-21 width by acceptance history, M-21b a learned stop classifier, M-22 Fable C's two-stage check). Every rule built on the drafter's confidence (cf), depth or the acceptance history lands within +-3 % of today's t 0.3 cut on code.

Hindsight says there's much more: the best width per check is worth +9-12 % on code and +16-22 % on chat, SSD-aware and exact. The gain lies in seeing the first wrong draft before the check runs, and nothing tested so far sees it.

**Read first:**
- MoonShots' replays: ~/MoonShots/data/bwidth45.txt, stop45.txt, twostage45.txt and their scripts. Reuse their cost model.
- The HPD-0 walks: ~/TrainLab/data/d02b_walk_hpd0_*.npz (cf per row, accepted lengths).
- Breakthrough's lattices: ~/Breakthrough/data/trees/lattice_{code,chat,rest}.npz. They hold the drafter's top-16 per depth in the selector's order, the chain's pick and the 125B's token, so margins and ranks live here. The walks have no top-2 margin.
- DRAFT_HIDDEN.md (first-miss anchors, ANCHORCOV), TREE_WALL.md, NEXT_DRAFTER.md (copy spans), copygate/GATE.md.

**Deliver**
1. **Where first misses happen,** as rates against the base rate:
   - by token class: newline, indentation, identifier start, punctuation, number, string, comment
   - by position in the line and depth in the block
   - inside vs outside copy spans
   - after recent misses
2. **Which cheap signals see them before the check:**
   - the drafter's top-1 / top-2 margin and its rank structure (from the lattices)
   - token class and copy state
   - n-gram novelty against the prompt
   Fit a predictor, trained and tested by window. Report its AUC and the width rule's gain on MoonShots' SSD-aware cost model, code and chat apart.
3. **What a second drafter's disagreement would have to add** (MTP Head's DUALDRAFT runs tonight, data by ~14:15). Say how MoonShots' queued M-23 should test it.
4. **Verdict and one pre-registered screen** for MoonShots' harness. The width rule is exact (drafts only propose).

**Rules**
- CPU only. Writes only in ~/FableGod/fableb/. No downloads.
- Halve gains. Mark every number [K] / [P] / [G].
- Price bytes on the union per 16-row block.
- Check ~/TRIED/TRIED.md first and say why each signal differs from what was tried: R-32 STOPPRED, ProPD, EVICT, AdaEDL, SpecDec++, DISCO, M-21, M-21b, M-22.
- Within ~40 minutes, write a complete first draft to FIRST_MISS.md, then refine it.
