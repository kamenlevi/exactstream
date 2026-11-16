# IDEAS_1116c_drift: the drift facts audited, then one drift mechanism (Fable C, Mon 16 Nov 2026, 18:2x-19:0x EET)

Desk + one 5-second CPU script on the t30b cells. Marks: [K] measured (here or in the records), [P] arithmetic on
measured parts, [G] judgement. HE+ / MBPP unless stated. First, one line on the last sprint: NewIdeas' FRONTIER#2 is right
that INERT wholesale is a recorded quality KILL (CD16-DROP 11-06) and that the guarded T16 forms are flat; under T16 a
chain-32 beats the tree-32 by x1.013 only (17.09 vs 16.87, tree_frontier_t16.txt), so INERT-shaped drafting is void at
quality. Withdrawn.

## 1. Audit of the brief's drift facts

| fact (THINK_1116c "facts") | source run | audited reading |
|---|---|---|
| "20.77 vs 13.77 (+51 %)" true-state ceiling | **NewIdeas' LAB-TRUESTATE shortcut** (lab_truestate.py, Sun 15 Nov 21:5x; TRIED 1422): from the t30b cells alone, truth_rt[a, d] = rt[a + d, 0] (the true-state head's guess at depth d from anchor a IS the depth-0 guess of anchor a + d), chain-32 under certify98's walk on heplus24 (R0 = 1, cap 31). 13.77 = t30b's own chain-32 on the same cells. Cross-check: BigMoE's measured REPLACE oracle (the real head fed the true state) reads 20.90 on HE+ (DIFFERS by 0.6 %; the head's own history matters that little) and 12.76 on MBPP (LAB 12.72, PASS) | the figure is a CHAIN-32 ceiling, teacher-forced, both arms on the same 24 windows. Two cautions: (a) the real 127-node tree's exact kept at 64 rows is ALSO 20.77 (20.765) by coincidence (BIGGER_TREES notes it); PLAN900 1b's +51 % is the LAB chain, not the tree; (b) for the tree-32 base (16.26) the zero-drift gain is unknown and smaller than +51 %: trees already take the shallow ties, and the LAB's gain at bands 0-6 is partly ties (see below) |
| "the t30b head holds 97.8 % flat to 30 on HE+ (MBPP 93.6)" | the same LAB run, its **MARGINAL** row: 98.0 97.9 97.9 97.9 97.8 97.8 97.7 (= every anchor's depth-0 accuracy, 97.94 %; CERTIFY98 calls marginal "information only") | **every-anchor, NOT walk-strict.** The LAB's walk-strict row is 93.2 / 95.3 / 96.4 / 97.8 / 98.8 / 99.2 / 99.3: below 97.8 at bands 0-6 (check starts on the 125B's near-ties), above it at 11-30. So "97.8 % walk-strict to 30" as a target is ABOVE the zero-drift head at bands 0-6; it needs trees for the ties (tree-32 band 0 98.5, 1-2 96.6) AND the drift fix at 3-14 |
| "accuracy decays: HE+ band 23-30 is 89.65 (t30b), 94.32 (+2B audit), 95.29 (+4B), oracle 96.29" | WEEK98-AUDIT / WEEK98-4B / WEEK98-CHOOSER (MoonShots, Sat 15 Nov): teacher-forced cells, **marginal** by band | marginal numbers in a walk-strict argument. Walk-strict, t30b's 23-30 reads **97.7** (survivorship: walks that reach depth 23 are easy stretches) and the LAB's 99.3. The drift gap that the gate sees is at bands 3-6 / 7-10 / 11-14: own 92.4 / 94.3 / 96.6 vs LAB 96.4 / 97.8 / 98.8 = **4.0 / 3.5 / 2.2 pt** walk-strict (marginal 3.5 / 4.8 / 5.6; 23-30 marginal 8.0). On the tree-32 the same bands read 93.7 / 94.3 / 96.3 (BIGGER_TREES): the tree is BELOW the zero-drift chain at 3-14 because past #1 it only has the head's continuations |
| "ties cap the shallow bands even with zero drift (93.2 / 95.3 / 96.4 / 97.8)" | LAB walk-strict | right. Note also LAB band 0 (93.2) differs from own band 0 (92.0) although both read the true anchor state: different walks start at different rows (survivorship), so band-0 comparisons across arms are not like for like |
| "carried state's direction error 0.85-0.99 by depth (HE); sibling 0.95-1.10" | Sandbox98 SIB main read-out 15:03 (LEVERS_A.md): 2B + s30, LAB, post-miss starts, depth slots 1-8 | right (MBPP own 0.81-0.99). It is the 2B's head, not the 125B's; direction error ~0.9 means the carried state is nearly unrelated to the true state in most directions while the head still reads 83 % walk-strict at 1-2: the readout uses a small part of the state, so a correction must move the RIGHT part, not the norm |
| "bands 3-14 stay 93-96 % even at K127" | treek_bands_n127_he.txt | right; the walk-strict coverage at 3-14 is the head's per-position accuracy once the true path leaves #1 |

Net: the ceiling is correctly sourced but it is a chain number and a marginal flat line; the gate's quantity is walk-strict,
where the drift gap on code is 2-4 pt at bands 3-14 (not 8), and closing "half of it" means +1-2 pt per band.

## 2. New premise measured here: drift is sequence-specific and persistent across blocks [K `drift_persist_1116c.out`]
t30b cells, all anchors, teacher-forced, non-tie rows only (the 125B's margin >= 2 nats, so ties cannot pose as drift):
- HE+: the per-window drift gap (depth-0 accuracy minus depth 3-8 accuracy) is 2.45 pt on average with an sd of 1.76 pt
  against a binomial sd of 0.50 -> **92 % of its between-window variance is persistent**; a non-tie deep miss raises
  the odds of another non-tie miss at the same depth **x4.5 one anchor later, x3.2 eight anchors later, x2.3-2.9 at
  16-32 anchors** (i.e. in the NEXT blocks). MBPP: persistent share 0.74, lifts x1.6 -> x1.06 at lag 32 (knowledge
  misses, little block-to-block structure).
- Reading: on code the head drifts in a way that is a property of the sequence (its context), observable from the
  previous blocks. Nothing in the stack uses that observation.

## 3. Invention: ASSIMILATED DRAFTING (online per-sequence correction of the carried state from the check's own residuals)
- **Mechanism (data assimilation / Kalman bias correction, as weather models do with analysis increments).** Every
  check pass computes the 125B's TRUE state at every accepted row; the head computed its CARRIED state at those same
  depths while drafting. Their difference, per accepted row, is an exact observation of this sequence's drift:
  r_d = true_d - carried_d. Keep a running estimate per sequence (mean, or an EMA over the last 3-4 blocks; rank 1,
  one vector in the head's input stream) and add it to the carried state at depths >= 1 of the next block
  (h_d <- h_d + e; variants: per-depth e_d, or e projected on the top-k subspace of the window's true states so far).
  Depth 0 is untouched (the head reads the true anchor state), so it attacks drift and not ties. The rejected rows'
  states are NOT used (SIB, KILL): only the accepted rows give true residuals.
- **Why it should move the gap.** The residual's persistent (context) component is what the head cannot regenerate
  from (anchor state + d tokens); section 2 says that component exists on code (0.92 persistent share, x2.3-3.2 across
  blocks). The sibling-mix oracle shows the conversion rate of the frozen 2B head: a 4-8 % held-out cut in direction
  error gave G +0.05-0.07 (G = share of the own -> true gap closed), ~1:1. So the number to beat is the direction-error
  cut: >= 15-30 % at depths 3-8 [G] would be G 0.15-0.3 on the frozen head, 3x the parked mix, and more after a short
  fine-tune with the correction in the loop (the head learns to read the corrected state).
- **Number it moves, with the bound.** Walk-strict HE+ bands 3-6 / 7-10 / 11-14 from 93.7 / 94.3 / 96.3 (tree-32)
  by +1-2 pt at G 0.3-0.5 [G]; kept per 32-row tree check +3-8 % [G]; the LAB's +51 % on a chain is the ceiling, not
  the expectation. Honest: this is below the brief's "half the gap" unless the fine-tuned form reaches G >= 0.5.
- **Screen 1 (Sandbox98, frozen s30, <= 3 GB, ~5 min; no build, the SIB score path has everything).** Add a state
  source "assim" to sb/sibstate.py score: process each window's post-miss starts in order; at start s' keep e = the
  mean (variant: EMA, half-life 4 starts) over the PREVIOUS starts of (tru_d - own_d) for d = 1..min(k_prev, 8) (the
  rows the check accepted: runtime-faithful; own_d is the stand-in the score already computes; tru is stored), and read
  the head at depths 1-8 on own + e. Report own / true / assim walk-strict 1-2 / 3-6 / 7-8 / 1-8 / 3-8, G with the
  bootstrap CI over windows, and the direction error own vs assim by depth. Lines, written now: **PASS if G(3-8) >= 0.25
  on HE with the CI's lower end > 0 AND the direction error at depths 3-8 falls >= 15 %; KILL if G(3-8) < 0.10 on HE
  or the error does not fall.** MBPP reported (expected weaker: section 2). A CPU-only pre-screen is impossible: the
  sandbox stores tru and sib states but not own, so the first number needs the head once (the score step, ~4 min).
- **Screen 2 (if PASS; <= 5 GB, <= 30 min).** The sibmix fine-tune recipe with the assim hook instead of the sibling
  rows (levers/sibmix.py's standin hook, 300 steps x 2e-6, ctl vs assim): PASS G(3-8) >= 0.5 HE, the brief's bar.
- **Cost per block on the 125B.** Zero bytes, zero extra passes: 32 vector adds on the head's input stream per block
  (the t30b reads the widened stream, ~10 k floats: < 1 us) and k vector subtractions after the check; one vector of
  state per sequence. Draft ms +0.0. Exact: drafter-side only, the check verifies as today; a wrong correction costs
  kept, never quality. Fits any drafter shape (tree or chain) and any head size (the QAT 26 MB head included).
- **What kills it.** The drift's persistent component may sit in directions the readout ignores (the 0.9 direction
  error says most of the state is readout-irrelevant); then the error falls and G stays ~0: Screen 1's two lines catch
  exactly that. Also: the first block of every answer has no residuals (e = 0: today's behaviour), so prompts with
  short answers gain nothing; MBPP's structure is weak (section 2).
- **Not on the kill list.** SIB used the rejected rows' states as INPUTS; carryac supervised the state in TRAINING;
  tapmlp fed mid-layer taps of the drafted rows (needs 125B passes). This uses the accepted rows' states as
  OBSERVATIONS of the drift at inference, per sequence, with no 125B cost; none of the three records measured it.

## 4. Two smaller drift levers (desk, for completeness)
- **Residual-keyed depth budget (exact, free):** the same per-sequence residual norm predicts how deep this sequence's
  drafts can go (section 2's x3 lift); stop the chain / tree depth per block at the depth where the running residual
  says the head is past its accuracy, and spend the nodes on width at shallower depths. CPU screen: the dumps + the
  cells' per-window miss rate as the key (the real residual needs states): kept at equal rows vs tree-32, PASS >= +3 %.
  [G: +2-4 % kept; it does not close the gap, it avoids paying for it.]
- **Zero-drift certification line (measurement, not a lever):** certify the LAB walk-strict row (93.2 / 95.3 / 96.4 /
  97.8 / 98.8 / 99.2 / 99.3) on the fresh set as the ceiling EvalSuite quotes, so that "97.8 flat" stops being read as
  a walk-strict promise.

## 5. Files
`~/FableGod/fablec/drift_persist_1116c.{py,out}` (the premise), this file. Records read: lab_truestate.{py,txt},
BIGGER_TREES.md, WEEK98.md 540-560 / 876-890 / 955-968, TRIED 860 / 878 / 901 / 922-923 / 1041 / 1049 / 1422,
Sandbox98 LEVERS_A.md (SIB), sb/sibstate.py, levers/sibmix.py, data/sib layout. No GPU, no locks, nothing committed.
