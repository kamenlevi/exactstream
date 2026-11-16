# EARLY DEATH probe: pre-registration (NewIdeas, Mon 16 Nov 15:3x, before any data)

The data doesn't exist yet. BigMoE captures the 125B's layer 8 / 16 / 24 residuals along the true text on Tue night
(data/deepdraft/ceiling/mid_{heplus24,mbppbank}.npz, aligned to the bank by fi / row / layer). NewIdeas runs this CPU
test on Wed by day, read-only, and doesn't touch the capture or the proofs.

## Question
At layer L of a tree check, can the 125B's partial state rank each spine node's children well enough to drop the wrong
ones from layers L+1..48? Dropping them shrinks the expert union, and so the bytes, in the remaining layers.

## Units and alignment
- Trees: BigMoE's real 127-node k8 dumps (tree_t30b_{heplus24,mbppbank}_n127k8.npz). Each node has token, parent,
  depth, path_p, on_true and top_ids / top_p.
- A depth-d node's token sits at position p = an + 2 + d (the convention span_tree.py uses). Its children's tokens sit at
  p + 1 and are scored from the 125B's state at row p.
- For a SPINE node (on_true), row p's state is the true-text state, so it is in the capture.
- Unit = a spine node with >= 2 children in the tree, scored at three tree sizes: the first 31 / 63 / 127 pops.
- If the true next token is not among the children, all children count as droppable, and no false drop is possible.

## Scores per child (from the state at row p)
- P-draft: the drafter's own path_p / top_p. This is the baseline with no target information.
- P-lens: the model's own final HC mix + norm + lm_head on the layer-L state. Logits on the children's tokens only;
  ranking within a node needs no normalizer.
- P-tuned: a ridge map from the layer-L state (10,240) to the final state, then lm_head. Fitted on the even files,
  scored on the odd files, and the reverse.
- P-comb: a logistic model on P-tuned + P-draft features, cross-fitted the same way.

## Metric
- Take the threshold at which the on_true child is dropped in <= 2 % of the spine nodes that have one (the
  false-drop rate). At that threshold, report the share of off-spine children dropped.
- Report per probe, per layer (8 / 16 / 24), per set (HE+, MBPP), with file-bootstrap 95 % CIs.

## Lines (Fable B's, registered before data)
- PASS: >= 90 % of off-spine children droppable at <= 2 % false drops at LAYER 16, by the best held-out non-oracle
  probe.
- KILL: < 60 %. BETWEEN otherwise.
- The PASS must come from target information: P-tuned or P-comb must beat P-draft by >= 10 points at the same
  false-drop rate. Otherwise the lever is only a smaller tree, which bet #9 already killed.

## Information only (no line)
- CHAIN VERSION. Per spine row, the chain's next token is the top child by path_p. Report the share of first misses
  flagged at layer 16 at <= 2 % false drops, and the expected rows saved past the first flag.
  - TRIED Rnd17#7 killed this on a napkin (-3-5 % of an older, dense-heavy check). On today's 32-row lossy check,
    experts are ~72 % of the bytes, so it gets a fresh read here.
- BYTES. The decision map's per-row routing (out_code24 scores) turns the measured drop rates into the expected expert
  union and GB per check for layers > L, for both versions.
- Layer 8 and layer 24 rates, to show how the trade moves with depth.

## Usage
Script: research/week98/earlydeath_probe.py, written Tue / Wed once the capture's format is known. CPU only, under
taskpolicy -b, <= 4 GB.

## Amendment 1 (Mon 16:2x, before any capture data): the CHAIN version gets its own line
- WHY: its byte bound (research/week98/ed_chain_bytes.py / .txt, CPU on the decision map's 24 code windows; 185 full
  32-row blocks, Breakthrough's tier rule, real pieces) is large. With a PERFECT first-miss flag after layer 17, at
  today's acceptance (p 0.95, E kept 16.1; measured 16.26):
  - C6 (lambda 1.8e-2): -26.3 % expert bytes (4.61 -> 3.40 GB per check, below the ~3.7 GB line for 25 ms).
  - 1c (1.6e-2): -25.9 %. Exact union (all top-10 at 4-bit, 15.28 GB): -23.0 %.
  - The bound shrinks as the drafter improves: p 0.978 gives -14.3 %, p 0.988 gives -8.5 % (C6).
  - It is EXACT: a false flag only costs kept, because the flagged row still runs every layer and gives the 125B's own
    token.
  - Pre-run line (in the script): a perfect flag < 8 % = dead whatever the probe; >= 15 % = keep. Result: -26.3 % ->
    KEEP.
- UNIT: per anchor, the chain = the top child by path_p at each depth. Rows up to the first miss are true-text rows, so
  their layer-L states are in the capture. Flag row r if the probe says draft r+1 is rejected. Rows after the first flag
  are dropped from layers > L.
- METRIC: on held-out files, the realized share of the perfect-flag byte saving (the map's per-row routing, with each
  block's flag position from the probe) at <= 2 % kept loss. Kept loss comes from flags before the true first miss.
  Use the real first-miss positions from the dumps' walk, not a geometric model.
- LINES (layer 16, best held-out non-oracle probe):
  - PASS >= 50 % of the perfect saving (~-13 % expert bytes at today's drafter) at <= 2 % kept loss.
  - KILL < 20 % (~-5 %).
  - BETWEEN otherwise.
  - It must also beat a drafter-confidence-only flag by >= 10 points of the perfect saving. Otherwise it's only a
    shorter chain, which K* already optimizes.

## Amendment 2 (Mon 16:3x, before any capture data): CORRECTION to amendment 1's bound
- Amendment 1 priced UNTRUNCATED 32-row chains. The engine already truncates each draft at a running confidence
  product of 0.3 (TRIED L1-14 / S45c, ADOPTED), so its chain checks average 16.7 rows (kept 12.33).
- On those real checks, a PERFECT flag after layer 17 saves only -13.6 % of the expert bytes (-17.1 % after layer 9,
  -10.0 % after 25). Details: research/week98/ed_chain_trunc.txt.
- A realistic probe would reach ~-4-7 %, matching TRIED Rnd17#7's napkin (-3-5 %). So the CHAIN arm goes back to
  INFORMATION ONLY, and amendment 1's chain lines are withdrawn.
- The -26 / -30 % bound stands only as a proxy for the TREE version: tree-32 checks aren't truncated, and their
  off-spine nodes are all rejected. Fable B's tree lines above are unchanged and remain the test.
- Stage 1 (research/week98/earlydeath_probe.py, earlydeath_stage1.txt) keeps the untruncated chain numbers plus the
  drafter-only baseline: -8.7 % at <= 2 % kept loss. That baseline is essentially the engine's existing truncation.

## Amendment 3 (Mon 16 Nov 21:2x, before ANY capture data): the stage-2 code, its truth label and P-tuned's form, fixed by a stand-in self-check
The stage-2 driver is research/week98/ed_stage2.py. Its lens is ed_lens.py, the 125B's own final hyper-connection mixer +
lm_head. Lens self-check on the bank's final states: top-1 = the stored tk on 192 / 192 rows, median |lp diff| 0.030 nats;
the hc_norm weight is used raw, with no +1 shift.
It was run with the FINAL layer standing in for layer L (the bank's wd). There P-lens IS the 125B, so it must drop ~all
off-spine children. That run fixed three things, all before any mid-layer data exists:
1. **TRUTH = the 125B's stored argmax at the unit's row (tk[row, 0], the capture pass), not the bank's text.**
   - On HE+, 319 of 15,327 on-true children (2.08 % of units; ~0.5 % of rows) are not the stored argmax: near-ties that
     flip between the generating pass and the capture pass. MBPP: 0 of 57,572.
   - Under the bank-text truth, even the 125B itself reads 1.88 % false drops on HE+, i.e. the whole 2 % budget.
   - The argmax is also what an exact check accepts, and it comes from the same pass as the captured mid-layer states.
     The bank-text reading is printed beside it as information.
2. **Every probe scores a child as a GAP to its unit's best child** (P-draft included), so a unit's best child is never
   dropped. That is conservative: units whose argmax is not among the children keep one child.
3. **P-tuned = a RESIDUAL ridge:** layer-L widened state -> (final widened state - layer-L state), dual form, alpha by
   inner CV on the training parity's rows, cross-fitted by file parity, then the real mixer + lm_head.
   - A plain ridge to the final state collapsed to 26-29 % droppable even AT the final layer (HE+: ~790 rows per parity
     in 10,240 dims project onto a tiny subspace).
   - The residual form shrinks to the plain lens instead of to zero, so it can only add to P-lens where the data
     supports a correction.
Stand-in ceilings (final layer as L; argmax truth; <= 2 % false drops; gap form):
| set | P-draft | P-lens | P-tuned | P-comb |
|---|---|---|---|---|
| HE+ (16,170 units) | 31.4 % [26.5, 36.9] | 97.0 % at 0.41 % | = P-lens | 97.0 % |
| MBPP (60,801 units) | 35.3 % [33.5, 37.8] | 97.4 % at 0.67 % | = P-lens | 97.5 % |
- So the gap form's ceiling is ~97 %, and the drafter-only baseline is ~31-35 %.
- Fable B's lines are unchanged: PASS >= 90 % at layer 16 by the best held-out non-oracle probe, beating P-draft by
  >= 10 pt; KILL < 60 %.
- Wednesday's run: `taskpolicy -b ~/BigMoE/.venv/bin/python -I -B ed_stage2.py --set he|mbpp --capture <mid_<tag>.npz>`
  (layers 8,16,24; peak ~5.4 GB on MBPP, so by day when Kamen is away).
