# TREE16: Breakthrough's pieces (B1-B3)

Spec: ~/Leap/data/week98/tree16_spec.md (draft until Mon 16 Nov 10:00; 47f9156 has B1's conventions and the EOS rule).

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

## B3 QSA selection above 2K: Tue-Wed
- Below ~2K context the indexer returns no mask (last_nb <= block_topk), so B3 only matters above that.
- Raw keys are pre-RoPE (row-local). The tree-specific parts are the query RoPE (positions L + depth) and each path's
  pooled window blocks.
- Output: each node's bool mask in its chain column order (cache columns, then its path tail L..L+depth).
- Bit test vs the path's chain row at 3K context.
- Read first: QSAIndexer (qwen4_exp ~3279: _extend_pooled_fixed, _verify_select_fused, _select_eager, _tiled_topk)
  and ref2's idx_prep form.
