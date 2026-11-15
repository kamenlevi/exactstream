# TREE16: Breakthrough's pieces (B1-B3)

Spec: ~/Leap/data/week98/tree16_spec.md (draft until Mon 16 Nov 10:00; 47f9156 has B1's conventions and the EOS rule).

**Frozen downstream (Sun 15 Nov 17:0x):** BigMoE's M4 integration (BigMoE 7f4b4e0) snapshots tree_meta.py + ple_tree.py
at 35aacf4 (its snapshot t16b). Any change to either file, or B3 landing: SendMessage BigMoE
(uds:/tmp/cc-socks/45105.sock) so it re-freezes. Until B3 lands, M4's loop runs a check as a chain once a + T > 2044
(MTPLX's dense line: the indexer is "update_only" while last_nb <= 2048 / 4).

## B1 tree_meta: DONE (b78708b)
- File: tree_meta.py. Test: test_tree_meta.py (PASS: 28,637 node-in-path checks over 2,011 trees; agrees with Leap's
  tree_gdn.fork_tables).
- Conventions:
  - gdn_win oldest first, -1 / -2 / -3 = conv slots 2 / 1 / 0;
  - ple_taps nearest first (3 / 6 / 9 back), -k = PLE conv slot 9 - k;
  - ngram_prev raw (oldest first);
  - fork_ops with fork_tables' numbering;
  - tail_cols = L + anc (L2 uses it directly).

## B2 ple_tree: DONE Sun 15:0x (bitwise 150/150, fbd3ede; end to end with L3 ALL BITWISE, 8d37798)
- `out = ple_tree(ple_layer, hidden [1, T, X], input_ids [1, T], meta, cache)`. It returns the PLE term the decoder
  layer adds, as `self.ple(hidden, input_ids, cache)` would.
- The stash, under MTPLX's verify capture only (Q._VERIFY_CAPTURE.get()): `cache._mtplx_verify_ple = (hidden,
  input_ids)` in node order, before the layer math, as MTPLX's _forward does for a chain. L3 does NOT set it.
  BigMoE's M2 _ple_commit (b792b91) restores the snapshot slots, stages the path's ids and runs the layer on
  h_in[path].
- ple_tree never writes the PLE layer's cache slots (CONV_IDX 2 = conv state, NGRAM_IDX 3 = n-gram history); M2 owns
  them.
- **n-gram rows.** Use MTPLX's own staged path: `_rows_np(ids [T, 1], prev [T, 2] = ngram_prev)` gives each node's
  rows, then ONE `sidecar.gather_np` over the 16 positions, like a chain-16. The EOS rule (_shift_ignore_eos: if the
  previous token is EOS, the one before it becomes EOS) holds by construction. Fallback when the sidecar is absent:
  _graph_path per node history.
- **Row-local part:** key_proj, value_proj, norm_key, norm_query, gate, attention_gate (REF2 forms: row-invariant for
  1-16 rows).
- **The dilated conv** (kernel 4, dilation 3, state 9 rows) on norm_conv(gated):
  - one MLX conv1d over T per-node 10-row windows, with the taps (ple_taps + the node) at rows 0 / 3 / 6 / 9 and state
    rows where the path is short;
  - silu;
  - Fallback: a per-node loop of the same call, or the path's exact chain window, if the batched call isn't bitwise.
- **Bit test** on the slice: each node == its path's chain row (the PLE layer's output, and the whole layer).

## B3 QSA selection above 2K: BIT TEST PASS Sun 17:28 (pre-registered 17:22, cef6107; code e78c859)
**Result** (data/m1/tree16/b3_test.txt; 23 s, 9.3 GB):
- **900 / 900 nodes** have B3's mask == their path chain row's mask: 11 trees x L 3000 / 3001 / 3002 / 3003 / 2036 /
  2044. The same 900 / 900 after the stock indexer call wrote the nodes in node order.
- The kernel self-checks vs MTPLX's stock kernels are bitwise (prepare-q in order and reversed; pool-k in order and
  permuted).
- B3 engaged on every tree at every L: 132 calls, 156 score runs (wide8 / bushy at L % 4 = 1 / 2 take 2 runs), all on
  the kernel path (idx_prep on, as the engine: R57_PRESET=45 -> R57_COMBINE=1 -> K73_EXACT=1 -> idx_prep).
- **Sensitivity:**
  - today's stock node-order mask is wrong on 548 / 888 nodes;
  - 279 off-chain nodes' masks differ from their main-chain row at the same depth;
  - 618 rows are sparse: at 3K each drops ~955 tokens (239 blocks); at 2036 / 2044 only the deepest rows are.
- M2's commit (BigMoE tree16_commit.py:61) resets pooled_len to L // 4. The next check's stock indexer call then
  rebuilds the committed blocks from the path-ordered raw keys before B3 reads them, so the two compose. A T2 probe at
  3K would confirm it end to end.
- **Extra (not pre-registered):** the eager indexer path (idx_prep off, as without K73_EXACT) also passes, 900 / 900
  (B3's eager query / pool variants; data/m1/tree16/b3_test_eager.txt).

**END TO END at 3K: ALL BITWISE** (Sun 17:29-17:32; data/m1/tree16/e2e_qsa_test.txt; 171 s, 9.4 GB).
- Setup: Leap's tree_verify + B1 + B2 + B3 on layers 0-3, P = 3000 and 3002.
- Result: logits + hidden_w of every node == its path's chain row; 302 nodes, 20 trees, B3's selection live in all 20.
- Hook (tree16_qsa_e2e.py; COPIES of Leap's functions, his files untouched):
  - _att_tree_b3: the stock indexer call under Q._QSA_HISTORY_ONLY (raw keys at L..L+T-1 in node order, no selection),
    then `sel = qsa_tree.tree_select(self.indexer, idx_rows or self.indexer.index_qk_proj(x), L, cache, t["meta"])`;
  - tree_sdpa_b3(..., sel): `msk_m = mx.array(msk) if sel is None else qsa_tree.route_b_mask(sel, Kr)`.
- **WIRED in Leap's live files (d6b5884, Sun 17:41)** through the slot tree16.QSA_TREE[0] = qsa_tree, set by the caller.
  - tree_att inlines route_b_mask as _b3_mask (the same ops). With the slot unset, de7fcc0's path runs (a raise past
    ~2K).
  - Leap's tree16_b3_test.py (this e2e with the copies removed, P 1024 added): 453 nodes ALL BITWISE at 1K / 3000 /
    3002; 20 live selections, 10 dense.
  - I reviewed the diff: it matches. The per-class sel -> sel_c rename in both tree_sdpa branches avoids shadowing.
  - My copies in tree16_qsa_e2e.py are now only the record. BigMoE re-freezes as t16c; Monday's M4 keeps t16b (its
    keys never pass 2044).
- **The loop above 2K, with commits: ALL BITWISE** (BigMoE 648826b, Sun 17:44; data/deepdraft/tree16/m4loop_{0-3,4-7}_b3.json;
  verified by me: qsa_tree.py sha256 6c678555c857 = my d138471 file).
  - Setup: 200-token decodes at P 1024 / 2030 / 3000 on both slices, through BigMoE's M1 / M2 + Leap's live slot.
  - Every kept row == REF2's 1-row AR: 0 bad rows, 0 fallbacks.
  - B3 live on 58 / 60 checks at P 2030 (the first 2 sit below the 2036 line) and 60 / 60 at 3000. 12 side-path commits
    per context fed later live selections; the chain control is clean.
  - This closes B3's composition with M2's tree_commit. t16c freezes qsa_tree after Monday's M4.

**The chain's path above 2K** (exact-45). ref2 pads every check to 16 rows (_att), and the engine sets idx_prep
(r57_screen_fast.py:400). No gather / flash / fused-selector / compiled-indexer / score-tile lane is on (MTPLX defaults;
the engine env sets only MTPLX_QSA_PREFILL=0). So QSAIndexer._call_rows runs:
- _prepare_queries: MTPLX's prepare kernel, RMSNorm + RoPE at L + row;
- write_raw;
- _extend_pooled: blocks b0 = L // 4 .. (L + 16) // 4 - 1 by the pool-keys kernel (trim keeps pooled_len <= offset // 4,
  so these 4 blocks are always recomputed);
- _select_eager: fp32 scores q [1, 16, 4, 128] @ pooled_t [1, 1, 128, nb]; relu-sum over the 4 heads / sqrt(128);
  where(valid); - blk * 1e-12; argpartition top 512; (blocks | tail) & causal. The result is a [1, 1, 16, L + 16] bool
  mask, and MLX's SDPA applies it as where(mask, s, finfo.min), the op route (b) already runs on its chain-order rows.
- The selection engages iff (L + 16) // 4 > 512, i.e. L >= 2036. A row is sparse only once (L + d + 1) // 4 > 512.

**B3** `sel = tree_select(indexer, qk_rows, L, cache, meta)` (qsa_tree.py) returns:
- None if (L + 16) // 4 <= 512 (the chain's own dense shortcut);
- else a bool [T, L + 16]: node n's mask in its chain column order (cache columns, then its path L..L+depth[n]; later
  columns False).

How it is computed:
- **Queries:** the nodes' raw indexer rows (qk_rows) padded to 16 rows, prepared at L + depth[n] by MTPLX's own prepare
  kernel with ONE expression changed (`pos_start[0] + int(row)` -> `pos_start[row]`, a per-row position). The kernel
  is built at runtime from the installed source, so it carries the installed arithmetic. When MTPLX's kernel is
  ineligible, the eager oracle runs with a positions array.
- **Path blocks:** blocks b0..b0+3 hold path positions. On node n's path, block b is closed by its ancestor at depth
  4b + 3 - L. Each (block, closer) instance pools its 4 raw keys (cache rows before L, the path's node rows after) by
  MTPLX's pool-keys kernel with one expression changed (`block_start[0] + int(block)` -> `block_start[block]`).
- **Scores:** the chain's own matmul shape, [1, 16, 4, 128] @ [1, 1, 128, nb].
  - The committed columns come from the cache's fp32 mirror. The 4 path columns carry up to 4 instances per run
    (ceil(NI / 4) runs).
  - Each node's dots are gathered into one [1, 16, 4, nb] array. Then the chain's relu-sum / divisor / where / tie /
    argpartition / mask ops run on it.
- B3 reads only cache raw keys before L and pooled blocks before b0. So it doesn't care what the stock indexer call
  wrote at L.. (node order).

**Pre-registered bit test** (test_qsa_tree.py):
- Setup: slice layers 0-3 (attention layer 3 with its indexer), exact-45 forms.
- Contexts: L = 3000, 3001, 3002, 3003 (every L % 4, i.e. every straddling block); L = 2036 (the first active L, where
  only depth-15 rows are sparse); L = 2044.
- Trees: B2's 11 trees. Each root-to-leaf path is checked as a chain (rolled back after), recording the indexer's mask.
- **PASS:**
  - every node's B3 mask == its chain row's mask, on every tree at every L;
  - the same again after the stock indexer call has written the nodes in node order (the integration order);
  - shared-prefix nodes' qk rows agree across paths.
- **Sensitivity** (shows the test can fail):
  - the stock call's node-order mask (what tree mode would return today) must differ from the reference on
    off-main-chain nodes;
  - rows with > 512 visible blocks must drop blocks.

**End to end next** (tree16_qsa_e2e.py):
- A copy of tree16_ple_e2e.py at L = 3000 and 3002, with B3 plugged into copies of Leap's _att_tree / tree_sdpa: the
  stock indexer runs under _QSA_HISTORY_ONLY (the cache's raw keys), then B3's rows serve as route (b)'s mask.
- **PASS:** logits + hidden_w of every node == its path's chain row.
