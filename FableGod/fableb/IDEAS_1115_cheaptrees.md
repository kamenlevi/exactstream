# IDEAS_1115: cheap big trees, exactly (Fable B, THINK_1115 problem 2, Sun 15 Nov 2026 22:1x-22:4x; desk only)

[K] measured by its owner, [P] derived from measured numbers, [G] my estimate. Script: `think1115/treecost.py` (CPU).

## 0. What the records say about the cost of a K-node exact tree (the frame for every idea)

- The union saturates. Fitting union(K) = A (1 - exp(-K / tau)) through 72.5 (16 rows) and 108.5 (32) [K] gives
  A = 144 experts per layer, tau = 23 rows. So the expert bytes of a 128-node check are ~18.2 GB against 14.6 at 32
  nodes: +25 %, not +300 %. Byte floors [P]: K32 52.0, K48 58.6, K64 61.8, K128 64.8 ms.
- Kept per check [K S11]: K32 16.26, K64 17.91, K128 19.14 (HE+). Kept per ms by bytes is FLAT: 0.31 (K32), 0.29 (K64),
  0.30 (K128). Big trees do not lose on bytes; they just do not win.
- **The wall moves from bytes to compute.** Dense MACs grow with rows (3.6 G weights x K rows) and so do the experts'
  (10 per row): the matrix-unit floor at 100 % is 13.6 ms at K32 but 54.5 ms at K128 [P on K M27-P]. Today's kernels
  run at ~57 % (experts) and ~40-60 % (dense) of it, so a 128-node check is compute-bound at ~90-100 ms, not 65.
- So "close to a 16-row check" is impossible by bytes alone (64.8 vs 38.8 ms at the floor), but "K128 at the K32
  price per kept token, or better" is open on two fronts: rows that are never needed, and MACs per row.
- "Overlap-aware shapes" (KILLed on paper) are moot: with A = 144 the union is nearly saturated past K64 anyway.

## 0b. CORRECTION with Leap's measured unions (orchestrator 22:05): the fit of section 0 is wrong, the reading holds
- Measured unions per layer [K Leap]: 73 / 110 / 136 / 157 / 187 / 211 at 16 / 32 / 48 / 64 / 96 / 128 rows. No
  saturation at 144: the marginal row still adds ~1.2 experts per layer at K96-128 (2.4 at K16-32, as M-10 said).
- Byte floors [K]: 0.85 / 1.00 / 1.12 / 1.30 / 1.44 (x the 32-row floor) = 38.8 / 52 / 58 / 68 / 75 ms at K16-128.
  Kept per ms by bytes is still flat: 0.29 / 0.31 / 0.30 / 0.26 / 0.26. One-pass cost today [K]: 1.45 / 1.82 / 2.17 /
  2.79 / 3.37 x the 16-row check = 56 / 71 / 84 / 108 / 131 ms. So the real 128-row check runs at ~1.75x its byte floor
  while the 32-row one runs at ~1.08x: the gap is the compute wall of section 0, measured.
- **The marginal row, priced [P]:** bytes 1.2 experts x 48 layers x 2.14 MB = 123 MB = 0.44 ms at 280 GB/s; MACs 3.6 G
  (dense) + 2.35 G (its 10 experts) = 0.43 ms at the unit's peak, ~0.8 ms at today's 50-57 %. Measured marginal row at
  K96-128: (3.37 - 2.79) x 38.8 / 32 = 0.7 ms. So the marginal row is COMPUTE, not bytes.
- What that does to the orchestrator's three questions:
  - expert reuse across siblings: M-10's 0.85x stands, and the bytes it could save are <= 0.44 ms per row anyway;
  - union-aware node choice from the head's routing guess: even a perfect guess halves 0.44 ms per row, under the
    0.7 ms the row costs in MACs, so it cannot pay (and it needs the predictor that KILLed it);
  - staged / partial checks: a split loses the union's sharing (bytes +40 %) and still pays the MACs of every row it
    keeps; M-22's KILL is the same effect.
- So the only exact levers on the marginal row are: not computing it (idea 1, early death), computing it at a higher
  share of the matrix unit (idea 3), and not multiplying it by path batching (idea 2). Idea 4's two-piece price rule
  is withdrawn (its constants were the wrong fit); the dump test of the union stays useful.
- Bottom line, corrected twice (see 0c): by bytes big trees lose (K128 0.213 kept per ms vs K32 0.310 at the floor);
  early death at layer 8 with <= 24 survivors is x1.16 at the floor, x1.08 halved, a wash at layer 16. Big trees exist
  only if a layer-8 probe on the 125B's state can retire >= 80 % of the off-spine nodes with <= 5 % false drops.

## 0c. RE-PRICED on the measured unions (orchestrator 22:1x; `think1115/earlydeath.py`, `earlydeath.out`)
- One-pass byte floors [P on K unions]: K16 10.9 GB 39 ms; K32 14.7 / 52; K48 17.3 / 62; K64 19.5 / 70; K96 22.7 / 81;
  K128 25.1 GB / 90 ms. Kept per ms: 0.288 / 0.310 / 0.277 / 0.257 / 0.233 / 0.213. So by bytes alone big trees LOSE
  (section 0's "flat" came from the wrong fit); K32 is the one-pass optimum, as NewIdeas found.
- EARLY DEATH re-priced (dead subtrees out of the batch after layer L; S survivors; the union of the survivors taken as
  S consecutive rows; a repair pass = a check of S rows, 46 ms at S 24):

  | K | L | S | GB | byte ms | kept per ms | vs K32 one-pass | false drops the gain absorbs |
  |---|---|---|---|---|---|---|---|
  | 64 | 8 | 24 | 13.9 | 50 | 0.360 | x1.16 | 18 % of checks |
  | 64 | 16 | 24 | 15.0 | 54 | 0.333 | x1.08 | 9 % |
  | 128 | 8 | 24 | 14.9 | 53 | 0.361 | x1.16 | 19 % |
  | 128 | 16 | 24 | 16.9 | 60 | 0.317 | x1.02 | 3 % |
  | 128 | 16 | 32 | 18.2 | 65 | 0.295 | x0.95 | none |
  | 128 | 24 | 24 | 19.0 | 68 | 0.283 | x0.91 | none |

- Reading: early death pays only if the drop happens by layer ~8-12 AND leaves <= ~24 rows; at layer 16 it is a wash,
  at 24 it loses. Siblings widening the union beyond K consecutive rows (Leap's line 57) makes every row worse by the
  same factor. Halved [G]: x1.08 at best. **It is a BETWEEN, not a win**, unless the probe works at layer 8 on the
  125B's near-tie parents, where a layer-8 state is least informative. The Monday test stands, with its line moved:
  PASS needs >= 80 % of off-spine nodes droppable at layer 8 with <= 5 % false drops.
- Compute after the drop: ~(128 x 8 + 24 x 40) / 48 = 41 row-equivalents at K128 / L8, so the check is byte-bound
  again (MAC floor ~18 ms at peak); at one pass it is compute-bound (54 ms at peak, ~100 real).

## 1. EARLY DEATH: drop dead subtrees mid-pass, repair exactly if wrong  [new; exact by construction]

- Mechanism. Siblings are mutually exclusive: at most one child of any node survives the check, and a node survives only
  if every ancestor does. Of 128 nodes, ~19 are kept and 82-86 % sit off the #1 spine [K S11]. Today every node is
  computed through all 48 layers. Instead, at layer L a cheap probe on the 125B's OWN hidden state of the parent
  position (the true state, which the head reads at 97.8-99.3 % [K]) scores "parent's argmax == this child's token".
  Children scored dead, with their subtrees, leave the batch for layers L+1..48: their expert reads and MACs stop.
  After the pass the accepted path is read from the surviving rows as today. If the accepted path reaches a position
  whose chosen child was dropped (the probe was wrong), a repair pass runs that subtree's rows (prefix states exist):
  exact by construction, since every row that decides the output is computed in full.
- Numbers [P]. K128, probe at layer 16, 20 survivors: experts 128 rows x 16 layers + 20 rows x 32 layers = 6.4 + 5.4 GB
  plus dense 3.4 = 15.2 GB -> 54 ms by bytes (vs 64.8), and the MAC floor falls from 54.5 to ~17 ms, so the check
  becomes byte-bound again: ~55-60 ms for 19.1 kept = 0.32-0.35 kept per ms against 0.31 at K32 [G, halved from the
  oracle]. At layer 8 the bytes are 13.3 GB (~48 ms).
- Tag: new as a pruner. TRIED kills early exit as a drafter (bytes) and M-22 TWOSTAGE (prefix-then-tail, -1.7..-3.3 %):
  both split a CHAIN into two passes, which loses the union's sharing. This drops BRANCHES inside one pass.
- Exact: yes (output rows are always computed in full; a wrong drop costs a repair pass, never a bit).
- What kills it: a layer-16 probe that is wrong on > ~10 % of checks (each repair is a ~20-row pass, ~40 ms); or the
  parents of off-spine children being exactly the near-ties where mid-layer states are undecided (92 % of check-start
  misses have < 2 nats of margin [K]). The probe can be made conservative (drop only at p_dead >= 0.98).
- Monday test (<= 15 min, CPU): from a capture with mid-layer hidden states at check-start positions (Wednesday's
  127-node dump can add layer 8 / 16 / 24 states), fit a logistic probe h_L[parent] x E[child token] -> survives, held
  out by prompt. Report precision at recall 0.5 / 0.8 for the off-spine children. PASS: >= 90 % of off-spine nodes
  droppable at <= 2 % false drops at layer 16. KILL: < 60 %.

## 2. DFS-GDN  [EXISTS: Leap's TREE16 L1 'depth-first body, ALL BITWISE' (66b4af8) is this idea; what is left is one detail]

- Leap's TREE16 L1 walks the nodes depth-first and saves the state at forks (64 KB per head, 3.1 MB per layer per fork,
  tree_verify_price.md). My draft is the same walk. The one thing it adds: keep the fork copies in REGISTERS (4 floats per
  lane per open fork along the path, stack depth = open forks, not forks) instead of device saves. At 3-5 forks per layer
  that is ~0.4-0.6 GB of save + restore traffic per 16-node check (1.5-2 ms); at K128 with ~40 forks per layer it is
  ~4.5 GB (16 ms), which the register stack removes. Worth a line in Leap's L1 only when trees go past 32 nodes.
- (Original note, kept for the record.) TRIED parks exact tree checks because "an exact tree needs per-node GDN states (~50 MB per layer)", and
  the chunk form is KILLed for exact modes (it re-associates). The sandbox's tree check path-batches instead: each
  root-to-leaf path is a row, so rows = sum of path lengths (~600 for a 128-node tree with ~45 leaves), not 128.
  Neither is needed. One simdgroup per (head, state row) walks the tree in depth-first order with the state in
  registers, exactly as mlx_lm's kernel walks a chain. At a branch node it keeps the node's state on a small register
  stack (4 floats per lane per open branch); a sibling subtree starts from that copy. Every node's expressions are the
  reference's, in the reference's order along its path, so each row is bitwise == the chain that reaches it (R-71's
  row-independence holds per path). The state is read once and written once (the committed path's end state is the
  one written). Attention already takes a tree mask.
- Numbers [P]. Node steps = K (a restore is a register copy). At 3.2 us per row step [K part B] a 128-node layer is
  ~0.4 ms, 13 ms per check (K32: ~0.1 ms per layer, ~4 ms) against ~10 GB of per-node states or ~5x the rows. Stack
  depth = open branch points along a path, typically 3-6 (EAGLE's band at depths 2-8): 24 registers at 6.
- Tag: new (TRIED 899 parks the per-node-state form; 1333 kills the chunk form; nothing walks the tree in registers).
- Exact: yes, bitwise (same per-step expressions, same order; a snapshot is a copy).
- What kills it: a tree whose paths open > 8 branches (register spill), or GDN time at K128 (13 ms) judged too much
  beside the 65 ms byte floor; then group siblings: the c children of a node share s = g S_parent only if g is equal,
  which it is not (g depends on the token), so no further sharing exists.
- CPU check done now (`think1115/dfs_gdn_check.py`): the depth-first walk vs per-path chains on 200 random trees of
  8-128 nodes, fp32: 0 differing outputs (bitwise). The stack in a register implementation holds one copy per OPEN
  branch point along the current path (the naive stack in the script holds every pending sibling, up to 30).
- Monday test (<= 15 min, GPU, Leap): a Metal draft on gdn_step3's harness at K32 and K128 trees from the S11 dump;
  line: bitwise == the per-path chains, and <= 0.15 ms per layer at K32.

## 3. ROW-DENSE EXPERT TILES: let the big tree's rows fill the matrix unit  [tried in part; exact]

- Mechanism. At 16 rows each expert serves ~2.2 rows (160 uses / 72.5 experts) but the code-MMA tile is 16 rows, so
  ~86 % of the expert MACs are padding; the kernel is instruction-bound at ~57 % of the unit [K NAX-ABL]. At 128
  nodes each expert serves ~8.9 rows (1,280 / 143.5): the same 16-row tile is ~56 % real work. So the MAC wall of
  section 0 is softer than it looks for big trees, and sharper for small ones: a per-expert gather of ITS rows into one
  16-row tile (gmr24's "read each distinct expert once and apply it to all its rows", L1-24, 6 % at 16 rows) pays
  4x more at 128 rows. Exact: the per-row arithmetic is the reference's (L1-24 was bitwise).
- Numbers [G]. If the expert path reaches ~80 % of the unit on real rows, the K128 compute floor is ~27 ms for the
  experts + the dense part's 33 ms at peak: the check stays byte-bound at ~65 ms only if the dense MACs also reach
  ~60 %+ of peak. Dense at 128 rows is 8 full 16-row tiles per weight: DENSE-N's 74 % of the byte floor at 16 rows
  [K] is the number to re-measure at 128.
- Tag: tried at 16 rows (L1-24 PASS-by-bits, KILL-by-speed at 6 %); new at 64-128 rows.
- What kills it: the dense MACs. If the dense pass at 128 rows runs under ~40 % of peak, no expert kernel brings K128
  under ~85 ms.
- Monday test (<= 15 min, GPU share lock, Leap): qmm16n chained at 128 rows (8 tiles per weight) on in_proj and
  out_proj's shapes; and NAX6-N's one-call at S 128 with 8.9 rows per expert. PASS: >= 60 % of peak on both.

## 4. KEPT-PER-MS SHAPE WITHOUT A ROUTING PREDICTOR  [WITHDRAWN in 0b: the fit was wrong; the dump test stays]

- Mechanism. A node's expert price is ~2.4 (sibling) to 2.8 (chain row) experts per layer at 16 rows [K M-10], but with
  the saturation fit the marginal expert cost per node falls to ~1.2 at K64 and ~0.3 at K128. Below ~K48 every node
  costs bytes; above it, nodes cost MACs only. So the budget rule for the tree builder is two-piece: bytes-priced to
  K48, MAC-priced beyond. No routing predictor is needed, only the two fitted constants (A, tau), which the Wednesday
  127-node dump can confirm directly (union per layer at every K).
- Numbers [P]. From section 0: K32 0.313, K48 0.293, K64 0.290, K128 0.295 kept per ms at the byte floor. The curve is
  flat, so the right K is set by the compute wall (idea 3) and by early death (idea 1), not by bytes.
- Exact: yes (shape only).
- What kills it: the dump showing a union that keeps growing past 144 (then A is wrong and big trees lose on bytes).
- Monday test (<= 15 min, CPU): count the per-layer union of the Wednesday dump's trees at K 16 / 32 / 64 / 128
  against the fit. PASS: within 10 % at K64 and K128.

## 5. What I looked at and dropped
- Staged verification (spine first, then a repair tree; or depth slices): every split of one pass into two loses the
  union's sharing (bytes rise ~40 %); M-22's KILL is the same effect on a chain.
- Reusing the last check's expert loads: the cost is RAM-to-GPU bytes per check; nothing persists across checks on
  this machine (no cache of GB size), so the only reuse is within one pass, which the union already gives.
- Predicting routes from token ids for layer 0: the router sits after the first mixer, so even layer 0 is context-bound.
