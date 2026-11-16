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
