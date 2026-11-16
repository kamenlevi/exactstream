# REPLAYED CHECK, module 1a: GPU-driven expert routing and the fixed-buffer replay (Fable B, Mon 16 Nov 18:3x)

For Leap (SPLIT_1116 module 1a; Breakthrough builds). Labels: [K] measured, [P] from measured parts, [G] estimate.
Read: ooo1cq.py (FAST-NAX one-call, gpu_grp_1cq, kernel A's poll), bigmoe/gated.py (post / spin, MODE spin), bigmoe/tier.py
(_ooo_call: pre_m / pre_l from dep rows, JOINT sidecar post), t1cf.py (tiers on the GPU), exp32_bench.py (the posts as the
host builds them: sm_of / sl_of tables, nolsb, the 2-D dup pick), c25_engine.py (the check = one forward + logits eval
+ commit). CPU only; nothing run.

## 0. The claim

**Everything value-dependent in the expert path is already a GPU value except one thing: the expert-to-slot resolution,
which today is a CPU loader round trip per layer (post_ids -> loader thread -> spin_wait).** With the experts resident,
that is a 2-way table gather. Everything else the CPU feeds per check is an index buffer of <= 16K ints that the CPU
knows before the check starts. So the check replays from a fixed record with:
- 6 small input buffers written before the check (section 3);
- one GPU kernel per MoE layer that resolves slots from a resident table and writes the routing row the loader reads
  today (section 2);
- misses handled by the one-call's existing poll (kernel A polls a base slot -1) against the loader thread, which keeps
  serving routing rows exactly as now, minus Python (section 4);
- one CPU sync at the end (the logits), as today.
CPU per replayed layer: < 1 us with indirect command buffers, ~45 us with a native re-encode after the dispatch diet
(section 6). The line (PASS <= 50) is met by the ICB form by ~50x and by native re-encode only after the diet.

## 1. The expert path today, per MoE layer (what the CPU does and why)

| step | where | value-dependent on |
|---|---|---|
| router: logits, softmax, top-10, renorm | GPU (MLX ops) | activations (GPU) |
| 1c tiers: I_e, tier per expert, km (drop -> rank-0 id at weight 0), nolsb [M, 2], counts | GPU with t1cf (one dispatch); ~28 MLX ops without it | scores (GPU) |
| the 2-D dup pick (a marked rank-0/1 use reads the first unmarked id) | MLX ops / numpy in the bench | nolsb (GPU) |
| post(L, ids): ext.post_ids writes the routing row [L, seq, n, npr, ids...] to the ring | a GPU kernel (lazy) | ids (GPU) |
| JOINT: post(side, ids[:, :2], after=dep_m) for the LSB sidecar row | GPU kernel | ids |
| **the loader thread reads the row, resolves expert -> slot (host table), fetches misses from SSD, writes dep = [ok, slot per pair], releases (L, seq)** | **CPU thread (C)** | **the host slot table** |
| spin_wait(dep, L, seq, n): a native kernel spins until the release | GPU, waits on the CPU | |
| pre_m = dep[1:1+n], pre_l likewise; -1 = not yet, -2 = QOM drop / U3 sure row | GPU slices | |
| gpu_grp_1cq: group pairs by (slot, tier, sidecar state) -> pidx, seg, nseg | GPU, grid fixed at P = M x K = 320 | |
| kernel A (grid N/32 x 64 ks, P) polls base slot -1 via region; -2 drops the group; kernel B; combine | GPU | |

So the Python per layer is graph build (~0.45 ms [P slice]) and encode (~0.25); the loader thread is not Python but it is a
CPU dependency inside the layer: the GPU spins until a CPU thread has resolved 320 ids. With misses 0 that release is
pure latency (the ~60 command-buffer gaps B5 saw at 16 rows).

## 2. GPU-driven routing: the SLOTS kernel (one dispatch per MoE layer, replaces post + loader resolve + spin)

Inputs (all fixed-address buffers):
- `ids [32, 10] int32` and `scores`, `nolsb [32, 2]`, `tier [320] u8`: t1cf's outputs (GPU).
- `slot_tab [48][512] int32`: the base-pool slot of each expert per layer, or -1 if not resident. Host-owned, host-written
  (section 4's protocol), 98 KB.
- `lsb_tab [48][512] int32`: the sidecar slot, or -1 (LSB not resident: C_l 112 of 512 per layer). 98 KB.
- `ring` (the routing rows the loader reads today), `seq_base [1] int32`, `region / status` (the one-call's gate).
- Policy constants: `POLL_RANKS` (ranks that may wait for a miss: 2 = ranks 0-1; 10 = every rank, the exact form),
  `DROP_MISS` (ranks >= POLL_RANKS that miss: -2 drop at weight 0, QOM's rule).

One threadgroup of 320 threads, thread i = pair (row m, rank r):
1. `e = ids[i]` after km (a dropped use already carries its row's rank-0 id; its tier is 2, weight 0).
2. **dup pick** (b43efc4's 2-D nolsb): for r < 2 with nolsb set, the sidecar id is the first unmarked (row, rank < 2) id
   in pair order, else ids[0, 0]. A 64-pair scan in threadgroup memory; bitwise the numpy rule in exp32_bench.
3. `s = slot_tab[L][e]`. Resident: `pre_m[i] = s`. Not resident: `pre_m[i] = (r < POLL_RANKS) ? -1 : -2` and
   `miss[i] = 1`.
4. Sidecar for r < 2 and tier 0 (4-bit): `l = lsb_tab[L][e_side]`; `pre_l = l` if resident, else -1 to poll (if the
   base is also missing the loader brings both: JOINT) or -2 (LP1: run at tier 3; today's rule when the loader read no
   LSB). nolsb set: pre_l = -2.
5. **The routing row is still written**, exactly as post_ids writes it today ([L, seq, n, npr, ids]; npr's counts from
   t1cf's counts), so the loader thread keeps its eviction statistics and its next-layer predictions, and so it SEES the
   misses. A `miss_count` per layer is written to `status` (fixed slot per layer) for section 4's accounting.
6. seq: `seq = seq_base + L` (the replay never touches the CPU's seq counter; the loader accepts rows whose seq is the
   check's base + L).

Then gpu_grp_1cq and kernels A / B run unchanged. Kernel A's head already polls a base slot -1 against `region` until
the loader writes it; -2 drops the group and B writes zeros; combine adds nothing. **No kernel in the one-call
changes**, so the lab's bitwise equality to today's route holds by construction: the only difference is who writes
pre_m / pre_l, and for resident experts the values are the same ints the loader would have written.

Cost: one dispatch, 320 threads, two 2 KB table reads per layer: ~3-4 us GPU per layer (t_sync + a 64-pair scan), 48
layers ~0.17 ms per check. It removes per layer: post (1 dispatch), the JOINT post (1), spin_wait (2: base + side), the
dep slicing / where / concatenate glue (~6-8 ops): ~-10 dispatches per layer, -480 per check, and the loader release
latency (the GPU never waits for a CPU thread when misses are 0).

## 3. Everything else the CPU feeds per check, and how a fixed-buffer replay takes it

| input | today | replay form | notes |
|---|---|---|---|
| the 32 token ids (embedding gather, PLE gather) | mx.array([toks]) | `tok [32] int32` buffer; the gathers index it | the embedding row read is a gather dispatch already |
| tree structure: parent of each row, depth | the tree mask [32, prefix + 32] built on the CPU | `parent [32] int32` + `anc_mask [32] u32` (ancestor bitmask within the block); the attention kernel masks block keys by `anc_mask[q] >> k & 1`, prefix keys by position | module 5's per-row key lists take the same two buffers |
| positions (RoPE, 12 attention layers) | prefix_len + depth, CPU ints | `pos [32] int32`; RoPE reads it | |
| KV cache write offsets (the 32 scratch keys) | prefix_len, CPU int | `kv_len [1] int32`; the append kernel offsets by it | |
| **KV length in the SDPA's grid** | the grid is sized by the prefix length | **(a)** the kernel loops over key blocks up to `kv_len` from the buffer with a grid sized for the window's max (16K keys: 64 blocks of 256; empty blocks return at once, ~1 us each); or **(b)** one record per 1K-key bucket, re-recorded every ~60 checks (a record ~50 ms [G] = ~1 ms per check amortized) | (a) is the right form; it is module 5's kernel or att_single with a length buffer |
| GDN state read / write per row (DFS fork states) | the committed state, fixed; per-row scratch states, fixed | fixed buffers; the DFS order comes from `parent` | the commit (pick the accepted path's state) stays outside the replay |
| SVH ids (V1: the gathered 16K head) | the drafter's vocabulary per check, CPU | `svh [16384] int32`; the head kernel gathers rows by it | 21 MB gathered read, as today |
| 1c tier counts, km, nolsb, dup pick | t1cf (GPU) + MLX glue | t1cf + the SLOTS kernel | the counts never return to the CPU |
| lambda, thresholds, plan masks, R4 | Python constants | baked into the record | a new lambda = a new record |
| the loader's posts | CPU thread | section 2 | |
| the HC seams (REF2_SEAM_HI, SB_FIRST, PLE_LATE) | Python branches on the row count | fixed at 32 rows: baked | the check is always 32 rows (pad) |
| the logits read-back | mx.eval(lg) | the one sync per check | |
| the commit, the draft, the next tree | CPU | outside the replay | |

**No value-dependent Python branch remains inside the check** once the row count is fixed at 32 (pad the tree with
dead rows: parent -1, weight 0; they cost dense compute only, ~0.3 ms per 4 rows above 28).

## 4. Misses: the fallback

Three forms, in order of preference; all keep the record fixed.

1. **POLL (the default; exact and lossy):** a missing expert at a polled rank gets pre_m -1; kernel A polls `region`
   until the loader writes the slot. The loader thread learns the miss from the routing row the SLOTS kernel wrote
   (step 5), exactly as it learns it from post_ids today, fetches from SSD, writes the slot into `region` AND into
   `slot_tab` (so the next layer / check sees it resident), releases. The GPU waits only on the miss, not on a CPU
   round trip for every layer. This is today's OOO behaviour minus Python. Timeout -> status[0] (exists: "spin gate
   timed out"), the check is void and re-run through the normal path.
2. **DROP (lossy, NW-SURE):** ranks >= POLL_RANKS that miss run at weight 0 (QOM's -2) and never wait. The routing row
   still reports them, so the loader fetches them for later checks. Quality: EvalSuite's gate on the never-wait rule
   (IDEAS_1116 idea 4; FAR's forms); not this module's call. Cost: 0 wait.
3. **FLAG + RE-RUN (lab only):** no loader thread at all (C 160 windows, all resident): the SLOTS kernel writes misses
   as -2 and sets `status[miss]`; the CPU reads the flag with the logits; a set flag voids the check and re-runs it on
   the normal path. Only for benches where misses are 0 by construction; in a real decode a re-run costs a whole
   replay per miss-check, so this is never the production form.

**The one protocol change the loader needs (BigMoE's):** the slot table is read by the GPU during the check, so the
loader must not REUSE a slot during a check. Rule: evictions (slot reuse) happen only at check boundaries (the CPU is
synced on the logits there anyway; the loader applies its eviction list then and rewrites the table entries); during a
check the loader FILLS free slots only (write the bytes, then the table entry, then `region`), keeping a reserve of
~8-16 free slots per layer for misses. Today's pin / release on posted rows becomes "pinned = the whole check". Unified
memory: host writes to a shared buffer before the command buffer commits are visible to it; a fill during the check is
visible to kernel A's poll because the poll reads `region` with a device-scope relaxed atomic, as now.

## 5. The record and the buffers

- **Record:** the check's dispatches in order: pipeline state, bound buffers + offsets, grid + threadgroup, barrier
  before each dependent dispatch. libmtlprof already sees these classes in-process. After the diet ~30-35 dispatches
  per layer, ~1,500-1,700 per check (today ~85 per layer, ~4,000).
- **Arena:** every intermediate at a fixed offset. Per layer the live set is small: in_proj out 32 x 16,480 bf16 = 1.05 MB,
  the GDN scratch (q k v z, conv state 32 rows) ~2 MB, HC streams 32 x 4 x 2560 x 2 B = 0.66 MB (+ K1C partials 64 x 32
  x 4 x 320 fp32 = 10 MB if STATS-FORWARD keeps them in memory), expert h 320 x 640 bf16 = 0.4 MB, y 32 x 10 x 2560 =
  1.6 MB, router 32 x 512. Two layer-slots alternate (layer l writes slot l % 2): ~40 MB total, allocated once outside
  MLX's allocator (MLX arrays can wrap it, or the replay bypasses MLX for the check and hands MLX the logits buffer).
- **Fixed-address weights:** every dense weight and the HC mixers are already fixed (loaded once). The expert pools are
  fixed (slots). The KV / GDN caches are fixed per window (grow by chunks: a cache re-allocation = a new record; make
  the cache a fixed max per window, as the engine's MAXK 320 already fixes the ring).
- **Execution:** (ICB) one `MTLIndirectCommandBuffer` per check (or per layer, 48 executes) with
  `ConcurrentDispatchThreads` commands on a SERIAL compute encoder, so command order carries the dependencies as MLX's
  encoder does today (same t_sync per dependent dispatch, ~1.6 us [K]). The per-check CPU work: write 6 input buffers,
  one `executeCommandsInBuffer`, commit, wait on the logits. (Native re-encode) the same record re-encoded by a C
  encoder each check: Metal's own encode cost ~1-1.5 us per dispatch [G].
- **Where the lab bits are checked:** milestone 1 replays one GDN + MoE layer and compares the layer output, the GDN
  state and the routing row against the MLX route bit for bit; the SLOTS kernel's pre_m / pre_l against the loader's
  dep rows (ints), on real routing (exp32's 24 blocks give that for free).

## 6. Desk price (CPU per replayed layer, and the GPU)

| form | CPU per layer | CPU per check | line (<= 50 / 50-200 / > 300 us) |
|---|---:|---:|---|
| today: Python build + MLX encode [P slice] | ~700 us | ~30-35 ms | |
| native re-encode, today's dispatch count (85 per layer) | ~110-130 us [G] | ~6 ms | BETWEEN |
| native re-encode after the diet (30-35 per layer) | ~40-50 us [G] | ~2.2 ms | PASS, barely |
| **ICB replay (one execute per check)** | **< 1 us** (6 input writes + 1 execute + commit ~ 20-40 us per check) | **~0.05 ms** | **PASS x50** |
| + the loader thread on a miss | off the critical path; the GPU polls | | |

GPU side [P]: unchanged kernels; minus the spin gates and their glue (~-480 dispatches: -0.8 ms by t_sync alone) and
minus the loader-release latency; the SLOTS kernel +0.17 ms.

**CORRECTION (Leap 18:3x):** spin_wait is a lazy GPU op that polls until the loader thread releases (L, seq), so in
libmtlprof's GPU busy the release latency IS GPU time. The SLOTS kernel therefore cuts GPU busy too: 48 x (loader post
-> release) + ~480 dispatches' fill. BigMoE's ext stats do not record the post -> release time (checked gated.py /
loader.py: none), so the per-layer number is unmeasured; tonight's GPU-busy run bounds it and Tue's pass K puts the
spin inside moe_experts. Desk guess: a loader thread that resolves 320 ids and writes the dep row with misses 0 is
~10-40 us per layer -> 0.5-2 ms per check [G, halve it]. So the split is: CPU 30-35 -> ~0 ms, GPU -1..-2 ms. **Part
of 1a pays even if tonight says GPU-paced**; the bulk pays only if the CPU paces the check (GPU idle >= 30 % of the
wall). The GPU sum with every module at PASS stays ~40 ms (SPLIT_1116 section 2), ~32 at stretch.

## 7. What kills it, and the order

- **Kills:** (1) the one-call or the SDPA has a grid that depends on a per-check value the kernel cannot read from a
  buffer (the KV length is the one I see; section 3 (a) fixes it, (b) is the fallback); (2) MLX's custom kernels cannot
  be bound at fixed addresses (then the replay runs the check outside MLX: the dense kernels are raw Metal already;
  the glue is the work); (3) the loader cannot promise no slot reuse during a check (then POLL is unsafe and only
  FLAG + RE-RUN remains, lab only).
- **Order:** (1) the SLOTS kernel + its int test against the loader's dep rows (a day; no replay needed, it also pays on
  its own: -480 dispatches); (2) the input buffers of section 3 in the MLX route (tok / parent / anc_mask / pos / kv_len
  / svh), so the route is record-ready; (3) one layer recorded and replayed (milestone 1, Wed 20:00); (4) the SDPA with
  a device-side length; (5) the full check (milestone 2).
