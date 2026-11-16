# IDEAS_1116c: the DRIFT wall from first principles (fable-zero, Mon 16 Nov 18:2x-19:45, desk only)

The question: the head holds 97.8 % flat to 30 on HE+ when fed the 125B's TRUE state at every position; carrying its
own state it decays to 89.65 at band 23-30 (t30b), and bands 3-14 sit at 93-96 even in a K127 tree. What is the
cheapest close-to-true state at depth, and does it fit ~3-7 ms per block? Records: WEEK98.md, PLAN900.md 1b, TRIED.md
(TREE-PLUS-AUDIT, S12 FULLSTATE, SIBMIX, carryac, TTS1, partial-trunk pricing). Nothing ran. [K] measured, [P] priced,
[G] guess.

## 1. The price from zero: a close-to-true state at depth d is COMPUTATION at drafted positions, and only WIDE computation is affordable
- The true state at p + d is the 48-layer function of the true state at p and the d tokens between. The head
  approximates it with one layer applied d times from its own carried state, whose direction error to the true state
  is 0.85-0.99 at every depth (SIB read-out): the head's state is not an estimate of the 125B's state at all, so
  nothing linear fixes it (held-out best mixes beat own by 4-8 %, SIBMIX-ORACLE; a ridge from own to true explains
  R2 0.45, COMBO5; carryac state supervision +0.6-0.8 pt).
- **The ladder, from the records (HE+ band 23-30 walk-strict / kept per 32-row check):**

  | computation per drafted position | accuracy 23-30 | kept | bytes per block (wide, read once) |
  |---|---:|---:|---:|
  | 1-layer head, own state (t30b) | 89.65 | 13.77-16.26 | 26 MB (QAT) .. 177.6 MB per LEVEL |
  | + 2B audit, P = 2 (frozen; replaces at confident disagreements) | 94.32 | +8.9 % | 2 x 1.06 GB |
  | + 4B audit, P = 2 | 95.29 | +14.9 % | 2 x 2.37 GB |
  | head-or-2B oracle | 96.29 | - | 1.06 GB |
  | 125B's own layers 1-12 at drafted positions (S12 tapmlp, 2B sandbox: +3.1..+4.5 pt at 15-30) | ~93-94 [P] | +5.1 % (sandbox) | 12 layers x (44 MB dense + ~46 lossy experts x 2.3 MB) = ~1.8 GB lossy, 3.7 GB exact |
  | 125B TRUE state at every position (LAB) | 97.8 | 20.77 (+51 %) | the check itself: 6.7 GB |

  Read: accuracy at depth is bought with model depth run OVER the draft; roughly +1 pt per doubling of the auditor from
  2B to 4B, and the 125B's own first 12 layers are WORSE per byte than a 2B (1.8 GB for the 2B's gain). Sequential
  per-token passes are out: 31 passes read the weights 31 times (S12 infeasible: ~60 GB per block; partial trunk
  pricing: 0.62 ms per layer per pass).
- **The budget theorem.** At 775 the whole draft + overhead is 4.5 ms = 1.26 GB at the byte floor; the head's own
  levels (QAT, 30 x 0.1-0.3 ms) take 3 of it. What is left for a state source is <= ~0.4 GB per block = 40 % of ONE
  2B pass. **So no computation-based drift lever fits 775 as a serial step; the 2B at P = 1 (1.06 GB, ~4-9 ms until
  timed) is the cheapest certified one and it alone is the budget.** Zero-byte sources are exhausted: the check's true
  states at rows <= kept already anchor the next block (the head's cache holds true pairs for every verified position,
  so drift is purely within-block, growing with d); rejected rows' states are worse than own (SIB); the last block's
  own-vs-true error is too weakly correlated with the next block's (cos 0.36-0.42) to subtract.
- **And fewer tokens per anchor is flat:** a 16-row check keeps 0.78 per row vs 0.53 at 32 but pays the 2.13 GB dense
  read twice as often: bytes per kept 400 vs 398 (IDEAS_1116b §3). The block depth cannot buy drift down.
- So, first principles: **the drift wall is not where 775's kept comes from.** The affordable kept at depth 3-14 is
  ties, covered by trees whose extra rows die early in the check. Below are the three ways to make a state source
  fit anyway, each priced honestly.

## Idea 1 (best that fits): AUDIT THE TAIL UNDER THE CHECK: the 2B's pass hidden in the check's idle ALUs, on a 64-deep draft
- **Mechanism.** The check is DRAM-bound (experts 65 % of its time, matrix units at ~17 % load). A 2B pass over 32
  rows is 62 G MAC (4.4 ms bf16 of ALU) and 1.06 GB. Run the 2B over rows 32-63 of a 64-deep draft INSIDE check n
  (same command stream, interleaved per layer, or a second queue if the gap tolerates it): its compute hides under the
  experts' DRAM time, only its bytes extend the check (+3.8 ms at 280 GB/s, +4.5 at 85 %). When block n fully accepts
  (q = p^32: 0.52 at 98 %, ~0.35 today), block n+1 starts from the audited rows 32-63 with no serial audit; when it
  dies at row m the tail is discarded and block n+1 drafts fresh (own state), as today. The 2B does not drift with the
  head's depth, so the audited tail is at the auditor's level at depths 32-63 (PLAN4D: P = 1 deep bands 97.6 / 98.2).
- **Number.** Kept per block +8.9 % x coverage q (0.35-0.52) = **+3..+4.6 %** on HE+ for +3.8-4.5 ms per check and +32
  head levels (+3 ms at the QAT head; +19 ms today = KILL until the QAT head exists). At a 30 ms check: ~+2..+4 %
  tok/s [P]. Not x1.3; the honest size of a hidden audit.
- **Screen (<= 10 min CPU, NewIdeas' tree_audit.py machinery on tree_t30b_heplus24.npz + rows2b_heplus.npz):** the
  2B's P = 1 audit applied only to rows 32-63 of a 64-deep chain, kept per block vs the plain 64-deep chain and vs
  tree-32; coverage = the share of blocks that fully accept. PASS >= +4 % kept at 32-row checks; KILL < +2 %.
- **Cost on the 125B / risk.** Bytes +1.06 GB per check; exact (the 125B still checks). Kill: the GPU cannot overlap a
  compute-bound 2B pass with the DRAM-bound expert call (the two-stream measurements: NAX-OCC-like 1.2-2.4 %; kappa
  0.93) -> then the audit is serial and costs 4-9 ms, which TREE-PLUS-AUDIT already priced as marginal (x1.003-1.043).

## Idea 2: A DISTILLED WIDE REFINER: the auditor as a 3-4 layer, ~100 MB model run once over the draft (training, not Saturday)
- **Mechanism.** The ladder says accuracy at depth comes from layers run over the draft. The 2B is 28 layers at 1.06 GB;
  a purpose-trained refiner of 3-4 layers (~200 M params, ~100 MB at 4-bit) that reads the head's drafted states AND
  tokens for all 31 positions in ONE pass, trained to the 125B's top-32 along the head's own drafts (so it learns the
  head's systematic drift, which a frozen 2B never sees), proposes a replacement or a sibling where it disagrees at
  margin. It is the "full-data distilled auditor" (PLAN4D item 2) shaped for the byte budget: 100 MB = 0.36 ms at the
  floor, one pass.
- **Number.** [G] from the ladder (+1 pt per doubling, 2B -> 0.2B = -3.3 doublings): ~91-92 at band 23-30, i.e. +1.5-2
  pt, kept +3..+4 %; the drift-aware training is the unknown that could beat the ladder (the 2B's +8.9 % came frozen).
  Cost per block 0.4-1 ms. Exact.
- **Screen (Sandbox98, 2B + s30 analog, <= 5 GB, <= 30 min):** a 2-layer refiner over s30's 31-token drafts (inputs:
  s30's carried states + token embeddings), trained 20 min on the 2B target's teacher-forced top-1 along s30's drafts;
  PASS: walk-strict 15-30 +2 pt over s30 with replacement at margin >= 2; KILL < +0.8 (then the refiner is the frozen
  audit's ladder and nothing more).
- **Risk.** Training by Saturday is MTP Head / FAR's call; a refiner that only learns s30's errors in the sandbox may
  not transfer to t30b on the 125B.

## Idea 3: ANCHOR DENSITY BY DEATH, not by block size: a mid-block true anchor from the check's early-death layer
- **Mechanism.** Early death (bet 11) ranks the spine's children at layer 16 with a tuned lens (P-tuned, ridge from the
  layer-16 state). The same layer-16 states of the SURVIVING spine rows are the 125B's own mid-depth representation of
  the drafted sequence, for free, one third of the way through the check. Use them twice: (a) as the early-death
  signal (as planned); (b) stored per row, as a layer-16 TAP input for the NEXT block's head at the rows that then get
  verified (positions <= kept): the head's anchor context gets mid-depth true taps for the whole last block instead of
  the final hidden only. This is S12's tap, obtained wide and free, limited to verified positions (no drafted-position
  taps, which the 125B "can't give").
- **Number.** The bound is the gap between "final hidden only" and "final + mid taps" at the anchor: DFlash2's taps at
  5 layers were worth its drafter at all (EARLYTAP: dropping to <= 25 of 48 layers cost x0.56-0.73 kept), so mid taps
  carry real information; for an already-anchored head the gain is [G] +0.5-1.5 pt at bands 3-14, the bands where
  the anchor's quality matters most. Zero bytes per block (the states exist for early death); a head retrain with
  the extra tap (FAR / MTP Head).
- **Screen (<= 10 min CPU, Wed, on BigMoE's Tue capture mid_{heplus24,mbppbank}.npz, layer 8 / 16 / 24 along the true
  text):** the LAB shortcut with the anchor's state = final hidden vs final + layer-16 residual, head frozen with a
  ridge adapter on the tap (lab_truestate.py's frame): walk-strict bands 1-14. PASS >= +1 pt at 3-14; KILL < +0.3.
- **Risk.** None to quality (exact); the lever is small by construction and needs early death's taps to exist.

## 2. What I would not do (the arithmetic says so)
- A 4B or P = 2 audit (2.1-4.7 GB per block: 2-4 checks' worth of draft budget).
- The 125B's own first 12 layers over the draft as the state source (1.8-3.7 GB per block for the 2B's gain).
- Any per-token trunk refresh (31x the bytes), test-time LoRA (TTS1 KILL), linear state corrections (R2 0.45, cos
  0.36-0.42), smaller blocks (flat in bytes per kept).

## 3. One line
Drift at depth is bought with model layers run over the draft (head 89.65 -> 2B 94.3 -> 4B 95.3 -> true 97.8 at band
23-30); the cheapest certified source (one 2B pass, 1.06 GB) is the whole 775 draft budget, so no serial drift lever
fits. What fits: the 2B's pass hidden under the check on a 64-deep draft (+3-4.6 % kept, +1.06 GB per check), a
~100 MB distilled wide refiner (+3-4 % [G], needs training), and the early-death layer-16 states as free mid taps for
the next anchor (+0.5-1.5 pt [G]). For 775 the kept must come from ties (trees + early death), not drift.
