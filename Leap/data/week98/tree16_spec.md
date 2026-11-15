# TREE16 spec: an exact 16-node tree check on the 125B (draft Sun 15 Nov 12:0x; frozen Mon 10:00)

**Goal and status.**
- An EAGLE-2-style 16-node tree verified in one forward, each node's logits bitwise equal to that node's row when
  its root-to-node path is checked as a chain.
- Pricing: data/week98/tree_verify_price.md (~chain-16 + 0.5-1.5 ms).
- MTP Head S9c (TrainLab fa3ce76): kept x1.136 MBPP / x1.060 HE+ vs chain-16.
- Approved split (the orchestrator 12:0x):
  - Leap L1-L3: kernels and the tree forward;
  - Breakthrough B1-B3: metadata, the PLE layer, long-context selection;
  - BigMoE M1-M4: drafting, walk, commit, harness, engine arm.
- Each in its own files.
- Reference: today's REF2-T (attention route (b)). If Kamen moves to a per-row-key reference later, only L2 changes.

## 1. The tree (M1's output, everyone's input)
- **`tokens`** int32 [T], T in 1..16 nodes (smaller near the cap or a window's end). **`parents`** int32 [T].
- **Node 0 is the root:** the token a chain check would put in row 0 (the last sampled token, not yet in the cache).
  `parents[0] = -1`.
- **Order:** DFS preorder, children in draft-probability order (best first), so `parents[n] < n`. Then nodes
  0..D-1 are the main (most probable) chain, contiguous, and the other branches follow.
- **`L`:** the committed cache length. Node n sits at position `L + depth[n]`.
- **M1's policy** (BigMoE, the one MTP Head's S9c priced: TrainLab fa3ce76 on BigMoE's dump b198498):
  - the root + the first 15 best-first pops by path probability, top-8 children per node;
  - 15 sequential t30b head forwards per check, the same count as a t30b chain-16.
  - A batched layer-wise EAGLE-2 variant would be a separate, re-priced option.

## 2. tree_meta (B1, Breakthrough; numpy, CPU)
`meta = tree_meta(tokens, parents, L, prev_tokens)`. `prev_tokens` = the last 9 committed tokens, for n-gram contexts.
Returns a dict:

| key | shape | meaning |
|---|---|---|
| `depth` | [T] | depth (root 0) |
| `pos` | [T] | L + depth (RoPE positions, KV slots of the chain layout) |
| `anc` | [T, 16] | the node's chain order: anc[n][0..depth[n]] = root, ..., parent, n; then -1 |
| `chain_rows` | [T] | True if node n is on the main chain (anc[n] == 0..depth[n]) |
| `gdn_win` | [T, 3] | the 3 rows before n on its path, oldest first: node index >= 0, or -1 / -2 / -3 = GDN conv-state slot 2 / 1 / 0 (the cache's last committed rows) |
| `ple_taps` | [T, 3] | the PLE dilated conv's earlier taps, NEAREST first (3, 6, 9 back along the path; the opposite order to gdn_win): node index, or -k = PLE conv-state slot (9 - k) |
| `ngram_prev` | [T, 2] | the 2 previous token ids along the path, oldest first (from tokens or prev_tokens), for the 3-gram ids; MTPLX's EOS rule applies (no context across an EOS: with context 2, a previous EOS makes the one before it EOS too); B2 hashes each node's raw history with MTPLX's own row code, so the rule holds by construction |
| `fork_ops` | list | the GDN DFS schedule, per node in order: (node, restore_slot or -1, save_slot or -1) |
| `n_slots` | int | the most saved states live at once |
| `paths` | list of lists | root-to-leaf node lists |
| `owner` | [T] | index of one path containing n |
| `tail_cols` | [T, 16] | for attention route (b): chain column L + j of node n is tree column L + anc[n][j] (j <= depth[n]); else -1 (masked) |

Unit test (B1): for every path, the chain layout rebuilt from meta equals the explicit chain (positions, windows,
taps, n-gram contexts).

## 3. tree_verify (L3, Leap): `logits, hidden_w = tree_verify(model, cache, tokens, meta)`
- Runs inside the verify scopes (attention phase decode_verify, capture on), with ref2 / v32 installed as for a
  16-row check. M2 takes the snapshot before it, as the chain check does. Any T in 1..16.
- Returns fp32 logits [T, V] and `hidden_w` [T, Wd] bf16: each node's pre-mixer widened stream, the hidden the MTP head
  consumes, as the chain verify's return_hidden gives it. Both are in node order.
- **Row-local ops:** projections, the experts as ONE call over the 16 nodes' union, HC, norms, head. They run as a
  16-row check runs.
- **RoPE:** positions `meta['pos']` per row (the stock elementwise path; bitwise per row with the chain's).
- **Attention, route (b) (L2):**
  - K / V of the 16 nodes are appended at cache slots L..L+15 in node order.
  - Scores use MLX's own ops; each node's score row is gathered into its chain order via `tail_cols`, with the
    rest masked.
  - Softmax is MLX's own op. Rows are classed by their AR key count L + depth + 16, as ref2 classes them.
  - P·V is Leap's bitwise port of MLX's GEMM, with each row's V tail in chain order.
  - Selection: below 2,048 context tokens the causal / ancestor mask; above it, B3's per-node mask.
- **GDN (L1):** the depth-first body. Phase A's conv windows come from `gdn_win`. The recurrence runs nodes in DFS
  order, saving the state at forks (`fork_ops`). The capture stash (qkv, q, k, v, a, b) is per node in node order.
- **PLE layer (B2):** `ple_tree(...)` gives each node's n-gram rows and dilated-conv inputs from the path.
- **Cache state afterwards:**
  - attention: K / V (and the indexer's raw keys) for all 16 nodes at slots L..L+15 in node order;
  - GDN: the per-node stash on each entry; the recurrent / conv state is NOT meaningful (the commit replays from
    the snapshot).
  - M2 commits from this.

## 4. The walk and tree_commit (M2, BigMoE)
- **Walk:** start at node 0. While some child c of the current node has `tokens[c] == argmax(logits[current])`, move
  to c. The accepted path is [0, n1, ..., n(k-1)], and the bonus token is argmax(logits[last]).
- **`tree_commit(model, cache, snapshot, meta, path)`:** leaves the cache exactly as if the path's k tokens had been
  checked as a chain and fully accepted.
  - **GDN:** restore the snapshot state, then replay the path's stash rows, gathered in path order (MTPLX's commit
    replay on the gathered rows). The conv state is the last 3 of [snapshot conv rows; the path's qkv rows]. The
    PLE layer's conv state likewise, 9 rows back.
  - **The PLE n-gram history** (the PLE layer's NGRAM_IDX entry, context_len 2) becomes the last 2 tokens of
    [prev; the path's tokens], as today's commit does through ple_embedding.stage(kept_ids, ...).
  - **The MTP drafter's history:** M2 appends the accepted path's pairs (hidden_w[path[i]], tokens[path[i+1]]), plus
    the bonus, to t30b's cache of committed pairs.
  - **Attention:** the path's K / V rows go into slots L..L+k-1 in path order, then trim to L + k. The QSA indexer's
    raw keys likewise, and its pooled blocks completed by the kept tokens are recomputed.
  - **Fast path:** if the path is a prefix of the main chain (nodes 0..k-1) AND k < T, today's chain commit (keep k)
    applies unchanged.
    - **NOT at k == T** (a fully accepted chain-shaped tree; BigMoE's T2, Sun 15:1x). MTPLX's commit_verified_window
      returns early on a full accept (trim_n == 0), because a chain verify leaves the post-window state.
    - tree_verify leaves the GDN state at the snapshot's, so nothing would replay. The chain tree at k 16 was off by
      5.8.
    - At k == T, tree_commit restores the snapshot and replays the path like any other accept.

## 5. Bit harness (M3, BigMoE; the 4-layer slice GGGA, then the 12-layer slice)
- **Tree shapes:** a pure chain, a bushy tree (branching at depth 0-2), a deep narrow tree, and real t30b EAGLE-2
  trees.
- **T1:** for every node, `tree_verify` logits == the logits of its row when its path is verified as a chain check
  from the same cache state (one check per root-to-leaf path). Bitwise, every node.
- **T2:** for several accepted paths per tree (main-chain prefixes and side branches):
  - tree_commit, then a 4-row probe check;
  - vs a chain check of the same k tokens with a full accept, then the same probe.
  - Probe logits bitwise.
- Contexts: 1K (selection dense) and 3K (B3's selection live).

## 6. The pieces and dates (the orchestrator's dates, by Thu)

| piece | owner | files | due |
|---|---|---|---|
| L1 GDN DFS body + unit bit test | Leap | scripts/week98/tree_gdn.py | Mon |
| L2 attention route (b): score gather + MLX-GEMM P·V port with per-row V tails + unit bit test | Leap | scripts/week98/tree_att.py | Tue-Wed |
| L3 tree_verify | Leap | scripts/week98/tree16.py | Wed |
| B1 tree_meta + unit test | Breakthrough | its own scripts dir | Mon |
| B2 ple_tree (n-gram ids + dilated conv per path) + bit test | Breakthrough | its own | Tue |
| B3 QSA selection > 2K per node + bit test | Breakthrough | its own | Tue-Wed |
| M1 tree drafting from t30b (EAGLE-2 expand + rerank to 16) | BigMoE | its own | Mon-Tue |
| M2 walk + tree_commit | BigMoE | its own | Tue-Wed |
| M3 bit harness T1 / T2 | BigMoE | its own | Wed-Thu |
| M4 engine arm: tree-16 vs a t30b chain-16 on the same drafter (t30b in the loop's MTP path, the chat switch's MTP path as the template; today's exact-45 drafts on DFlash2) | BigMoE | its loop | Thu night |

**Rules:**
- No one edits another session's files; interfaces change only through this spec, with a dated note.
- Everything is exact by construction or bitwise-tested before the engine.
- BigMoE imports Leap's tree16.py / tree_gdn.py / tree_att.py live, as it does ref2 / v32.

## Change notes
- Sun 12:3x (BigMoE's interface notes, 4):
  - tree_verify also returns hidden_w (the MTP head's input) per node;
  - tree_commit restores the PLE n-gram history and appends the accepted pairs to t30b's history;
  - T is any value in 1..16;
  - M1's policy is pinned to S9c's priced variant (15 sequential head forwards), so M4 compares against a t30b chain-16.

## L2 design notes (Leap, Sun 12:4x, from Breakthrough's kernel-level profile, c55_1105.json)
**What the exact path's 16-row attention core runs**, inside MLX's unfused SDPA in MTPLX's 2-head chunks:
- `q * scale` (bf16);
- the scores, `steel_gemm_fused_nax_nt` (bm64 bn128 bk256 wm2 wn4);
- `where(mask, s, finfo.min)`;
- `block_softmax_precise_bfloat16`;
- P·V, `steel_gemm_splitk_nax_nn` (bm64 bn64 bk256 wm2 wn2) + `steel_gemm_splitk_accum_bfloat16_float32`.

**The fused NAX GEMM's per-element arithmetic is fixed:** every output element accumulates k-steps of SK = 32 in order
(gemm_nax.h), whatever BM / BN / WM / WN. This is why the 32-row calls were bitwise with the 16-row calls (ATT32-INV,
C2's HC up GEMM).

**Route (b), concretely:**
1. Scores with MLX's own ops on the tree's K layout. Each element is a dot over the head dim, so it doesn't depend
   on its column.
2. Each node's score row gathered into its chain order (`tail_cols`).
3. MLX's own where + softmax on the gathered rows (the chain row's values in the chain row's columns).
4. P·V with leapk's split-K S4 replica (MLX's split-K NAX, bitwise for the HC GEMM) moved to the nn layout. In the last
   partition's tail k-steps, each row runs the NAX step with its own chain-order V fragment and the other rows of A
   zeroed (adding exact zeros leaves them unchanged). Then the accumulate kernel.
5. ref2's class / bucket logic per row (AR keys L + depth + 16).

**Open, pinned first on Tue:** MLX's split-K partitioning for this shape. Found by matching candidate partitions
against mx.matmul bitwise, as the qv_wide k_lanes were.
- Sun 12:5x (Breakthrough's B1, written):
  - conventions confirmed against L1 (gdn_win oldest first, fork_ops in tree_gdn.fork_tables' numbering, tail_cols = L + anc);
  - the n-gram EOS rule added to ngram_prev;
  - ple_taps is nearest first.
  - B2's plan: ple_tree(ple_layer, hidden, tokens, meta, cache) -> the PLE term. MLX's own conv1d on a 10-row window
    per node, batched over the nodes, with the bit test vs the chain rows as the arbiter. It doesn't touch the cache's
    PLE slots (M2's).
- Sun 14:4x (Leap):
  - L1 (tree_gdn.gdn_body_tree) and L2 (tree_att.tree_sdpa) are built and ALL BITWISE in their unit tests: every node
    == its path run as a chain; 10 / 7 trees, all ref2 key classes.
  - L2 pads a tree's queries to 16 rows (MLX's scores kernel depends on the rows), as ref2 pads short checks.
  - L3 (tree_verify) is next.
- Sun 14:5x (Leap):
  - L3 tree16.tree_verify is ALL BITWISE end to end on a slice without the PLE layer: 8 trees, logits + hidden_w of
    every node == its path's chain verify.
  - B2 plugs in by setting tree16.PLE_TREE[0] = ple_tree (signature ple_tree(ple_layer, hidden, input_ids, meta, cache)).
  - Before calling tree_verify: ref2.apply(), then tree16.install(ref2).
  - tree_verify consumes B1's meta keys anc, depth, gdn_win, fork_ops, n_slots.
- Sun 15:0x:
  - B2 PASS (Breakthrough fbd3ede).
  - End to end with the PLE layer: ALL BITWISE (8d37798; 10 trees incl. EOS).
  - Route (b): heads batched per kv head (default) and the tail steps one per path; both bitwise.
  - TREE16-TIME: the batched form costs +4.0 % engine-scaled; re-timed with the per-path tail.
- Sun 15:1x (BigMoE 445e8c7):
  - M3 ALL BITWISE on both slices with tree_verify: layers 4-7, and layers 0-3 with B2. T1 every node == its path's
    chain; T2 73 / 73 (tree_commit + probe == chain commit + probe).
  - Section 4's fast path is fixed: k < T only, since MTPLX's commit skips the replay on a full accept and tree_verify's
    GDN state is not meaningful.
