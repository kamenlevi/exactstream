# MODULE 1a: the engine's expert routing, mapped for milestone 2 (Mon 16 Nov 2026 21:2x)

A read-only map of today's route, made by a sub-agent from BigMoE's and Leap's code. Nothing was run. Every claim
carries a file:line reference; the paths are abbreviated below. It was checked against Fable B's REPLAYED_CHECK.md
sections 1-4. My conclusions for the replay are at the end, marked [I] (inferred).

Paths: EXT = ~/BigMoE/ext/bigmoe_ext.cpp, GATED = ~/BigMoE/bigmoe/gated.py, TIER = ~/BigMoE/bigmoe/tier.py,
STREAM = ~/BigMoE/bigmoe/stream.py, REF2 = ~/Leap/scripts/ref2.py, OQ = ~/Leap/scripts/ooo1cq.py,
OC = ~/Leap/scripts/ooo1c.py, N6 = ~/Leap/scripts/tierglu_nax6.py, COH = ~/Leap/scripts/coh1cq.py,
T1CF = ~/Leap/scripts/week98/t1cf.py, X32 = ~/Leap/scripts/week98/exp32_bench.py.

## 0. Where the code differs from the design doc
- **The loader does not write dep; a GPU kernel does.** `bigmoe_post_ids` writes dep = [ok, slot or -1 per id] by
  looking ids up in a residency table the loader published earlier (EXT:1093-1114). The loader writes the release
  `region` (EXT:1757-1774) and the published tables (EXT:1780-1788, 1262-1274).
- **The FAST-NAX one-call does not use spin_wait.** Kernel A polls `region` itself (REF2:982-1094). spin_wait only
  appears on the two-phase routes (TIER:161-162, REF2:1165-1166, 1226-1227).
- **The routing row has no L field and no ring.** There is one fixed row per row index L in `Gate.route`, overwritten
  each pass. `Gate.ring` is the loader's slot output (EXT:1521, 1722).
- **npr packs look-ahead prediction counts** for layers L+1..L+3, not t1cf's counts (STREAM:86-115). The sidecar
  carries 0 (STREAM:120-121).
- **seq is per pass, not seq_base + L.** `Gate.seq` goes up by 1 per pass (GATED:126-138, STREAM:69-70). Every layer
  and both rows (base and sidecar) share it.
- **pre_l = -2 at nolsb positions is set in Python** (REF2:1009). The dup pick changes only the ids in the sidecar
  row.

## 1. Routing row
- `Gate.route` is int32, max(4096, nl x ROW) words (GATED:104). Row L starts at L x ROW, with ROW = 96 + 32 x PLINES
  + 32 x XLINES (EXT:592). At MAXK 320, ROW = 416.
- Words:
  - [0] seq, [1] n, [2..31] ids 0-29;
  - [32] seq, [33..63] ids 30-60;
  - [64] seq, [65] np, [66..95] preds;
  - then prediction lines and further id lines (EXT:593).
- Encoding: `(v & 0x3FFFFF) | (seq & 0x3FF) << 22` (EXT:1116-1122). Ids are posted after km, duplicates kept.
- Write order: np, preds, n, ids, a fence (system scope under BIGMOE_POST_SYS), then the tags with [0] last
  (EXT:1117-1127).
- Bindings: ids [[0]], route [[1]], int4 {L, seq, n, has_tab} [[2]] (setBytes, fixed when the graph is built),
  dep [[3]], 64 MB bump [[4]], tab [[5]] at tab_offset(L, seq), preds [[6]], np [[7]] (EXT:1093-1096, 1214-1234).
- **Side effect: the bump input makes MLX commit the command buffer after every post** (EXT:1075, 1232, 2203).

## 2. dep, the published table, the release
- dep[0] = 1 only if every id's entry in the table is fresh (tag and stamp == seq). dep[1+t] = the slot, or -1
  (EXT:1097-1114). post_ids never writes -2.
- `Gate.tables`: int32 [1024 x 17 x 32]. The block for (seq, L) starts at ((seq x nl + L) % 1024) x 17 x 32
  (EXT:1256-1258).
  - Without JIT, the loader publishes (seq+1, L) after releasing (seq, L) (EXT:1780-1788).
  - With JIT, it publishes (seq, L+1) as prefetches become ready (EXT:1262-1274, 1797).
- `Gate.region`: 128 regions x 128 pairs x RS. The entry for (seq, L) is at ((seq x nl + L) % 128) (EXT:1758).
  Stamped words; tags are written last.
- Released values: the slot if resident, else -2 (QOM optional miss / LP1) (EXT:920, 851-852). -1 = 0x3FFFFF,
  -2 = 0x3FFFFE.
- The loader accepts a row when row[0] is a new seq, all tags are complete and every stamp matches (EXT:1679-1714).

## 3. spin_wait (two-phase routes only)
`bigmoe_spin` (EXT:1159-1186):
- dep[0] == 1: it copies dep (status[16]++).
- Otherwise it polls pairs with back-off; a timeout does status[0]++ and returns zeros.

## 4. pre_m / pre_l and the one-call
- pre_m = dep_m[1:1+n].reshape(M, K) (REF2:982-983).
- pre_l = the sidecar's dep, with -2 where nolsb is set (REF2:1006-1009).
- The one-call: f1c(xin, base, sideP, pre_m, pre_l, ids_k, PLAN, region, offs, status) (REF2:1074-1077), coherent
  form by default (REF2:233, 1067-1069).
  - **offs = int32 [3] = [base region offset, side region offset, seq], built on the CPU every call (REF2:914-923).**
- gpu_grp_1cq groups pairs by key (OQ:148-159, N6:226-250).
- Kernel A head: it polls `region` for -1 bases via oc_poll (OQ:44-73, OC:34-48). On a timeout, status[0]++ and
  slot 0. -2 drops the group, and kernel B writes zeros (OQ:76-84).

## 5. JOINT sidecar
- The sidecar post comes after dep_m (mx.depends), with the same seq, at row side0 + L (GATED:169-170,
  STREAM:494-495, 603-611).
- The loader pairs base and sidecar rows within a seq (EXT:1583-1643, 1683-1686).

## 6. Host slot tables
- Each layer's `Layer` (EXT:47-99) holds table[E] (expert -> slot or -1), slot_owner, an LRU and free slots.
- ensure() / make_room evict the LRU tail, never an id the current row needs (EXT:707-719, 768-925).
- There is no explicit pin: a prefetch is published only once ready.
- exp32_bench: every expert is loaded; sm_of / sl_of = lookup(...); post is a stub (X32:49-58, 80-84, 121-125).

## 7. t1cf
fused(ids, scores, s, lam) returns sc_out, ids_out [M, K], nolsb [M, 2] bool, counts [3] int32, tier [M, K] uint8
(T1CF:82-95). A dropped use carries ids[row, 0] at score 0.

## 8. Bitwise cautions
- **Today's pre_m is not run-to-run reproducible.** It depends on publish races (EXT:961-963, 1793-1797), so compare
  outputs (rs_m / y), not pre_m.
- seq must stay per pass.
- Other CPU-written values the one-call depends on:
  - offs;
  - region and Gate.tables (written by the loader);
  - post_ids' setBytes and table offset;
  - t1cf's cached constants;
  - the dup pick's cached arange.
- status words [0], [16], [17] and [1..8] are already in use.

## Implications for the replay [I]
1. **The GPU-side expert -> slot lookup already exists** (post_ids' table fast path). Fable B's SLOTS kernel is
   mostly there. What ties each check to the CPU is the per-pass seq.
2. **seq has to be patched per check.** It sits in:
   - post_ids' int4 setBytes and its table offset ((seq x nl + L) % 1024);
   - offs[3] (an MLX array built per call, so in a replay it is a recorded input that can be written in place);
   - the region offsets ((seq x nl + L) % 128).
   In the ICB form the setBytes words live in my constants buffer, so the replay patches them before each execute.
   They are found from the recording: pipelines are named by the creation hook, e.g. bigmoe_post_ids.
3. **Visibility.** Today every post commits its command buffer, so the loader sees the routing row. A one-command-
   buffer replay needs either BIGMOE_POST_SYS (system-scope fences in post_ids) or the replay split into command
   buffers at the posts (one ICB range per segment, committed in order). To test.
4. **The loader thread keeps running unchanged** (ext.start): it publishes tables, serves misses and releases
   regions. kernel A's poll handles misses as today.
5. **Bitwise reference = the layer outputs** (rs_m / y), not pre_m.
