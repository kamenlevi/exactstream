# MODULE 1a: the replayed check (Breakthrough; Leap's CHECK25 split, SPLIT_1116.md; design: Fable B)

Started Mon 16 Nov 2026 21:0x, at <= 6 GB while Kamen uses the Mac (the orchestrator 20:47). Leap's split assigns it;
Breakthrough accepted at 18:12. Fable B's design is ~/FableGod/fableb/REPLAYED_CHECK.md (read 20:5x).

## Goal and lines (Leap's split, registered there)

- **Mechanism.** Record the check's GPU work once and replay it per check: per dispatch, the pipeline, the bound buffers
  and offsets, the inline bytes, the grid and the barriers. The replay writes only the new inputs.
- **Milestone 1 (Wed 18 Nov 20:00).** One real layer replayed: a GDN layer plus its MoE at 32 rows, real weights.
  - Outputs and states must equal MLX's bit for bit.
  - The CPU microseconds per replay are measured.
  - PASS <= 50 us of CPU per layer replay; BETWEEN 50-200; KILL > 300 us, or not bitwise.
- **Milestone 2 (Thu-Fri, stretch).** The full check replayed, timed in Leap's c25 harness.

## Plan (Mon 21:0x)

1. **Recorder (mtlrec).** libmtlprof (mine: scripts/m1/check55/mtlprof.m) already swizzles the concrete compute encoder,
   command buffer and device classes. A recording build adds:
   - hooks for setBuffer:offset:atIndex:, setBuffers:offsets:withRange:, setBufferOffset:atIndex:,
     setBytes:length:atIndex:, setThreadgroupMemoryLength:atIndex:, useResource(s), and the blit encoder's copy / fill;
   - recording switched on from Python around a region. Every op is appended in order; every bound buffer and pipeline
     is retained, so MLX's allocator cannot reuse it; inline bytes are copied.
2. **Native replay.** A C function re-issues the list on one command buffer:
   - one compute encoder (serial dispatch: a superset of the recorded barriers), plus blit segments;
   - commit, then wait.
   - Measured: CPU microseconds per replay (encode + commit), and the GPU time.
3. **Inputs.** MLX 0.32.2 arrays expose a writable memoryview, and MLX sees in-place writes (checked 20:5x). So the
   replay keeps the recorded input arrays alive and overwrites them with the next check's inputs: no buffer lookup.
4. **Bitwise test.**
   - (a) Replaying the recorded inputs reproduces the recorded outputs.
   - (b) New inputs: the MLX forward on x1 vs the replay after writing x1 into the recorded x0, compared bit for bit on
     the layer output and the GDN state.
5. **Scope of milestone 1.**
   - The slice's MoE is MTPLX's stock path (gather_qmm, all experts resident; Leap's cpusplit_slice.py), so no CPU
     loader sits inside the layer.
   - The engine's one-call MoE needs Fable B's SLOTS kernel before it can replay. That kernel is step 1 of Fable B's
     order and needs BigMoE's no-slot-reuse rule, so it belongs to milestone 2.
6. **ICB form, after (2) works.**
   - MLX's pipelines are not created with supportIndirectCommandBuffers, so they must be re-created from the same
     functions with the flag set.
   - setBytes must move into a constants buffer (ICB compute commands have no inline bytes).
   - Fable B prices it at < 1 us per layer.

## Log

- Mon 21:0x: plan written; MLX in-place input write checked (memoryview writable; MLX sees the write).
- Mon 21:0x: **recorder built and the toy mechanics test passes** (scripts/m1/rec1a/: mtlrec.m, build_mtlrec.sh,
  mtlrec.py, rec_toy.py; libmtlrec.dylib md5 8147036f, mtlrec.m 32c45e5b).
  - 22 hooks on AGXG17XFamilyComputeContext / BlitContext / CommandBuffer.
  - Toy graph: a 32 x 2560 @ 2560 x 1024 bf16 matmul, softmax and sigmoid. Recorded as 55 ops: 1 encoder, 8 dispatches,
    2 commits, 88 B of setBytes.
  - (a) A replay with the recorded inputs reproduces the recorded output bit for bit.
  - (b) New inputs written in place into the recorded input (memoryview), then a replay: bit for bit equal to MLX's own
    forward, 3 of 3 inputs (max |diff| 0). A serial-dispatch replay is bitwise too.
  - CPU encode 13-28 us for 8 dispatches (one 102 us outlier, with Chrome busy). Commit + wait 205-320 us. GPU 48-52 us.
    Peak 140 MB.
  - Next: the real layer (one GDN layer + its MoE at 32 rows, the slice's stock MoE) under <= 6 GB, with MLX's cache at
    0; then the bitwise test on the layer output and the GDN state, and the CPU microseconds per replay.
- Mon 21:0x: **the real layer replays BIT FOR BIT (stock route)**. scripts/m1/rec1a/rec_layer.py ->
  data/m1/rec1a_layer_stock.out. Share lock "MEM 6 GB", peak 3.22 GB, wall 3 s.
  - Slice: the bare 125B cut to layer 0 (a GDN layer + its MoE, real weights; the slice's stock MoE = gather_qmm, all
    resident). The PLE stage index is recomputed for the cut (PLE lives on layer 2; None for 1 layer).
  - 1,024-token prefix (my docs), then a 32-row verify forward (Leap's verify scope).
  - Record: 993 ops, 6 encoders, 114 dispatches, 8 commits, 2,704 B of setBytes. Outputs: logits 1 x 32 x 248,320 bf16
    + 2 GDN state arrays (3.2 MB).
  - (a) The outputs were zeroed, then replayed on the recorded inputs: logits and states equal the record bit for bit.
  - (b) New token ids written in place into the recorded token array, the cache restored to S0 (by reference), then a
    replay vs MLX's own forward on the same tokens: logits AND both states bit for bit, 3 of 3 windows.
  - CPU encode per replay 220-572 us for the 114 dispatches (~2-3.4 us per dispatch, with Chrome busy). Commit + wait
    11-14 ms, which is the GPU (the full 248K-row head dominates).
  - That CPU figure covers embed + layer + head. The per-layer number needs the layer recorded alone (next).
  - The plain re-encode issues ~8.7 ops per dispatch through objc_msgSend. So the levers are: skip redundant re-binds,
    call cached IMPs, then the ICB form (Fable B: < 1 us per layer).
- Mon 21:1x: **a replay race found and fixed. MILESTONE 1's BITWISE criterion is MET on the check's route.**
  - Data: data/m1/rec1a_race_model_stock.out, rec1a_layer_stock.out, rec1a_model_fast.out, rec1a_layer_fast.out. Runner:
    scripts/m1/rec1a/run_rec_set.sh (bash: zsh does not word-split env lists, which mislabelled one set of runs, now
    deleted).
  - **The race.** Mode 0 re-issues each recorded encoder with its fences inside ONE command buffer. It is not safe: stock
    whole-model 15/16 windows bitwise, FAST 0-1/16.
    - MLX itself is deterministic: two MLX forwards agree bit for bit on 16/16 windows.
    - The record holds 13 fence ops (MLX orders its command buffers with MTLFences), and MLX's buffers are not
      hazard-tracked, so one command buffer needs explicit ordering.
  - **The fix, mode 2 (merged).** One concurrent compute encoder for the whole record. The recorded barriers are kept, a
    full buffer barrier goes in at every original encoder boundary, and fences are dropped. It is bitwise in every
    cell:

| route, scope | dispatches (ops) | (a) recorded inputs | (b) new inputs, 16 windows | CPU encode per replay, median (min) |
|---|---|---|---|---|
| stock, whole 1-layer model | 114 (993) | yes | 16/16 | 393 (299) us |
| stock, layer 0 alone | 101 (893) | yes | 16/16 | 276 (243) us |
| FAST, whole 1-layer model | 153 (1,310) | yes | 16/16 | 510 (362) us |
| **FAST, layer 0 alone (GDN + MoE, the check's route)** | **123 (1,049)** | **yes** | **16/16** | **327 (259) us** |

  - (b) in the layer scope writes new token ids AND new input hidden states (the embedding tiled to the 4 HC streams) in
    place. It compares the layer output (1 x 32 x 10,240 bf16) and both GDN states with MLX's own forward.
  - **CPU vs the line** (PASS <= 50 / BETWEEN 50-200 / KILL > 300 us per layer). The plain native re-encode costs ~2.7 us
    per dispatch (~8.5 ops each), measured with Chrome busy, so the FAST layer sits at the KILL edge (327 us).
    - As Fable B priced it, the line needs the ICB form, or a re-encode after the dispatch diet with redundant binds
      skipped.
  - GPU for the FAST layer replay: ~5 ms (shared GPU).
  - **Next (Tue): the ICB form.**
    - Re-create each recorded pipeline from its function with supportIndirectCommandBuffers (a creation hook maps
      pipeline -> function / descriptor).
    - setBytes moves into one constants buffer.
    - One ICB of concurrent-dispatch commands with setBarrier where mode 2 puts barriers.
    - useResources once per replay (or a residency set).
    - Then the same bitwise test and the CPU microseconds.
- Mon 21:2x: **the ICB form works on the real layer, bit for bit** (data/m1/rec1a_icb_layer_fast.out / _stock.out; runner
  scripts/m1/rec1a/run_rec_icb.sh; libmtlrec mtlrec.m 5d0b07f8, dylib b49460bf).
  - Build: a pipeline-creation hook maps each pipeline to its function (+ maxTotalThreadsPerThreadgroup,
    threadGroupSizeIsMultipleOfThreadExecutionWidth). Each recorded pipeline is re-created with
    supportIndirectCommandBuffers; setBytes data goes into one constants buffer (256 B slots).
  - One indirect compute command per dispatch (buffers re-bound per command, threadgroup memory, setBarrier where mode 2
    puts barriers).
  - Replay: one encoder, useResources over every referenced buffer, one executeCommandsInBuffer, commit, wait.
  - Toy: 8 commands, 3/3 bitwise. Real layer 0, both routes:

| route, layer 0 | ICB commands | (a) | (b) 16 new-input windows | CPU per replay, 40 repeats: min / median / max |
|---|---|---|---|---|
| FAST (the check's route) | 123 | yes | 16/16 (merged native 16/16 too) | 40 / 68 / 161 us |
| stock | 101 | yes | 16/16 | 66 / 90 / 607 us |

  - **ROUGH numbers.** Share lock, with Kamen's Chrome and two other share runs beside it.
  - **What the CPU figure holds.** It is the WHOLE per-replay cost for one layer: command buffer + encoder + useResources
    (every buffer the layer touches) + execute + end. Most of it is per replay, not per layer, so in a full check (one
    execute for 48 layers) it is paid once.
  - **Against the line** (PASS <= 50 us per layer replay): the one-layer ICB replay sits at a 68 us median (min 40),
    just over. Next: an MTLResidencySet built once (queue-level) in place of useResources, a split of the per-replay
    costs, and a quiet bench (Leap's split plans one for Wed).
- Mon 21:3x: **with a residency set, the one-layer ICB replay meets milestone 1's CPU line (rough).**
  data/m1/rec1a_rset_layer_fast.out; runner scripts/m1/rec1a/run_rec_rset.sh; mtlrec.m c56ada06, dylib 2a63ad75.

| ICB replay of FAST layer 0 (123 commands) | bitwise (new-input windows) | CPU encode per replay, median (min) | command buffer | encoder | useResources | execute + end |
|---|---|---|---|---|---|---|
| + useResources over 203 buffers | 4/4 | 77 (57) us | 7.3 | 23.8 | 37.5 | 7.2 |
| + an MTLResidencySet built once (204 allocations, on the replay queue) | 4/4 | **45 (31) us** | 8.3 | 27.7 | 0 | 9.7 |

  - **Against the line** (PASS <= 50 us per layer replay, KILL > 300 or not bitwise): **PASS on this rough
    measurement**, with bit-for-bit outputs and GDN states.
    - Rough: a share lock, Kamen's Chrome busy, two other share runs beside it.
    - The official number is Leap's planned quiet bench (Wed).
  - **The split.** 36 of the 45 us are the command buffer + encoder, paid once per CHECK, not per layer. The layer's own
    part (execute + end for 123 commands) is ~10 us.
    - So a 48-layer check in one ICB is ~0.05-0.1 ms of CPU [P], against today's ~30-35 ms. Fable B priced it at
      ~0.05 ms.
  - **Caveats (what milestone 2 still has to do):**
    - The slice's MoE is the stock gather_qmm with every expert resident. The engine's one-call MoE needs Fable B's
      SLOTS kernel (no CPU loader round trip inside the check) and BigMoE's no-slot-reuse rule.
    - The attention layers need the SDPA's KV length from a buffer (Fable B section 3a).
    - The tree mask, positions and the SVH ids become input buffers.
    - The record is one fixed 32-row shape.
    - MLX must keep its cache limit at 0 (no buffer reuse) while a record lives.
- Tue 00:5x, **Leap's GPU-busy read of the lossy 32-row check re-ranks 1a** (Leap's file data/week98/c25/gpubB_run.txt,
  [lab: c25, F32c6i, 3 real windows, quiet, 24 checks]; Leap's numbers, not mine):
  - Wall 56.1 ms; GPU busy 54.2 ms (96 %), idle 2.0 ms. CPU thread 23.2 ms (Python build 12.0 + MLX encode 11.2) runs
    hidden under the GPU; 27.4 ms blocked on it. 3,172 dispatches and 266 command buffers per check.
  - So a perfect CPU replay wins <= ~2 ms of wall today. 1a starts to bind once the GPU part is under ~23 ms (a ~25 ms
    check), so it stays on the path but is not first in line. The GPU side pays now (SLOTS: no spin_wait poll, ~480
    fewer dispatches).
  - Checked before proposing a command-buffer merge through MLX's own limits: BigMoE measured it. MLX_MAX_OPS_PER_BUFFER
    1000 or 10 made no material change on the 35B gate (BigMoE data/diag35.txt; gate_facts.md 3.5), and
    MLX_MAX_MB_PER_BUFFER=4096 broke the gate (the 64 MB bump then no longer forces the commit after a post).
  - The slice agrees on the structure: MLX ran FAST layer 0 as 123 dispatches in 5 encoders / 6 commits
    (data/m1/rec1a_layer_fast.out), ~5.5 per layer like the check's 266 / 48.
- Tue 01:0x, **pre-registration: per-check patching of seq-derived setBytes words (milestone 2 groundwork, toy)**.
  - Why: the engine's per-pass values (the pass's seq in post_ids' int4 setBytes, the table and region offsets derived
    from it; MODULE1A_ENGINE_MAP.md) ride in setBytes payloads, so a replay must rewrite them per check, in the plain
    replay's recorded copies and in the ICB's constants buffer.
  - Mechanism (libmtlrec): rec_bytes_meta dumps every setBytes payload with its bind index and first dispatch;
    rec_disp_names names each dispatch's kernel; rec_bytes_patch writes one payload; rec_rule_add registers
    word <- ((seq * a + b) mod m) * s + c (int32 / float32 / int64); rec_rules_apply(seq) rewrites every rule in one call.
  - Toy (scripts/m1/rec1a/rec_patch_toy.py): y = x * arange(st(s), st(s) + N) + arange(s, s + N) + sum(offs), with
    st(s) = ((s * 48 + 7) mod 128) * 17 (the engine's region form) and N = 4096. x and offs are buffer inputs, written in
    place per check as before.
  - Lines:
    - (1) The diff of two records (s = 5, 6) finds exactly the two arange start words (int32 st(s), float32 s) and no
      other changed word.
    - (2) With the rules applied per check, the mode-2 replay and the ICB replay are bitwise == MLX's own forward on 16
      new seqs (7-22) with new x and offs.
    - (3) Control: the replay with the previous check's values does not match on any of the 16 seqs.
    - (4) rec_rules_apply costs < 2 us per call from Python (ctypes included).
    - PASS = all four. KILL = (2) fails.
  - Run: share lock "MEM 1 GB", seconds of GPU.
- Tue 01:03, **result: PASS on all four lines** (data/m1/rec1a_patch_toy.out; scripts/m1/rec1a/rec_patch_toy.py;
  mtlrec.m 5806b8b4, dylib 1351e218; share lock, seconds of GPU, peak 0.05 GB):
  - (1) The two records (s = 5, 6) have the same op sequence, kernels and payload layout. The diff finds exactly 2
    changed words, both explained: setBytes k=1 word 0 of `arangeint32` (2023 -> 663 = st(5) -> st(6)) and k=6 word 0
    of `arangefloat32` (5.0 -> 6.0). No other word changed.
  - (2) With the two rules applied per check: mode-2 replay 16/16 and ICB replay 16/16 bitwise == MLX on s = 7-22
    (new x and offs written in place each time).
  - (3) The control (the previous check's words) missed 16/16.
  - (4) rec_rules_apply: median 0.50 us per call (2 rules, ctypes included), p99 0.62.
  - What it shows: per-check values that ride in setBytes can be located by a two-record diff and rewritten in one C
    call per check, for both replay forms, with no re-encode. For the engine, the same diff on two engine records
    (passes seq and seq + 1) should find post_ids' seq word and the derived table / region offsets; their formulas are
    in MODULE1A_ENGINE_MAP.md. That run needs the engine (>= 9 GB, Kamen away).
- Tue 01:1x, **pre-registration: what a growing KV cache changes per check (attention toy).**
  - Why: an attention layer in the check reads a KV length that grows every check, and writes its 32 new K/V rows (and
    the tree's mask block) at that length. For a replay without a kernel change, every per-check difference must be a
    setBytes word, a buffer offset, or (at worst) a pipeline variant.
  - Recorder additions: rec_ops_dump (every op: code, indices, offsets, grid, threadgroup, object identities), rules
    with floor division and several per-check variables, and buffer-offset rules that also re-bind every ICB command
    using that binding.
  - Toy (scripts/m1/rec1a/rec_attn_toy.py): 32 query rows, 8 heads, 2 KV heads, D 128, a preallocated 4,096-row
    cache. Per check at kv: write the 32 new K and V rows and the 32 x 32 tree mask block at row / column kv. Then
    mx.fast.scaled_dot_product_attention over the first kv + 32 rows, with the mask view.
    - Records: aligned kL (kv 1024, 1056, 1088) and unaligned kL (kv 1029, 1061, 1101), plus kv 1203 as the replay base.
  - Lines:
    - (1) Within a class, records differ only in setBytes words and buffer offsets: no op-structure, grid, threadgroup
      or pipeline change. Across classes, any pipeline change is listed (info).
    - (2) Rules fitted on the 3 unaligned records predict the base record's (kv 1203) changed words and offsets exactly.
    - (3) Patched replays (mode 2 and ICB) are bitwise == MLX on 16 new unaligned kv values, and the stale control
      misses 16/16.
    - (4) Info: the unaligned record replayed at 16 aligned kv values: bitwise == MLX or not. If not, the replay
      needs a per-check pipeline switch.
    - PASS = (1)-(3); KILL = (3) fails.
  - Run: share lock "MEM 1 GB", seconds of GPU.
  - Amendment (Tue 01:2x, before any run; the toy has not run): the classes are now kL odd (unaligned for any key block)
    and kL a multiple of 64 (aligned for any key block up to 64). Records: aligned kv 1056, 1120, 1184; unaligned kv
    993, 1119, 1137, 1215, 1289, 1363 (kL residues mod 64 of 1, 63, 17, 31, 41, 51); base kv 1203; tests: unaligned
    kv 1301 + 34 i, aligned kv 1504 + 64 i (16 each).
    - Line (2) now fits on the 6 unaligned + 3 aligned records (values only).
    - Reason: with odd residues alone, the floor-division offsets of NK and NK_aligned are pinned only to within one,
      and the two have opposite edges. The aligned points (residue 0) pin both. Lines (1), (3), (4) unchanged.
    - Bitwise compares are on the bf16 output bytes.
- Tue 01:12, **attention toy result: PASS on all three lines, and the info line is clean too**
  (data/m1/rec1a_attn_toy.out; scripts/m1/rec1a/rec_attn_toy.py; mtlrec.m c43eaac1, dylib ef4b399d; share lock,
  seconds of GPU, peak 0.08 GB). MLX 0.32.2 runs the check-shaped attention as 4 dispatches:
  - 3 strided copies (`gg2_copybool_bool_` for the mask block, `gg2_copybfloat16bfloat16` for K and for V);
  - `steel_attention_bfloat16_bq64_bk32_bd128_wm4_wn1_maskbool_` with function constants align_Q / align_K /
    has_mask / do_causal / has_sinks.
  - (1) Within a class the records differ ONLY in:
    - 3 buffer offsets: the mask write at kv bytes, the K and V writes at kv * 256 bytes;
    - up to 5 setBytes words of AttnParams: kL (+16), NK (+32), NK_aligned (+40), kL_rem (+48), qL_off (+52).
    - No op-structure, grid, threadgroup or threadgroup-memory change. All 7 within-class pairs are clean.
    - Across classes only the steel attention pipeline changes (align_K t / n).
  - (2) Fitted on the 9 records, 8 rules, all exact on the base record (kv 1203):
    - offsets: kv, 256 kv, 256 kv;
    - kL = kv + 32; NK = floor((kv + 63) / 32); NK_aligned = floor((kv + 32) / 32); kL_rem = kv mod 32; qL_off = kv.
    - So the key block is 32.
  - (3) Patched replays bitwise == MLX: mode 2 16/16, ICB 16/16 (the ICB re-binds the 3 write commands' buffer offsets
    per check). The stale control missed 16/16.
  - (4) Info: the UNALIGNED record replayed at 16 aligned kL (multiples of 64) is bitwise == MLX's own forward, which
    runs the align_K kernel: 16/16 (mode 2), 16/16 (ICB). So one record serves every kv here, with no per-check
    pipeline switch.
  - What it shows for milestone 2: on this toy, the growing KV length needs no kernel change for a replay. Per check,
    each attention layer needs 5 params words and 3 write offsets, rewritten by rules in the same one C call as the seq
    words. That is a cheaper route than a device-side KV length (Fable B section 3a), for the replay at least.
  - Caveats for the engine: its own cache and mask code may change more per check, e.g.:
    - a KV cache that grows by reallocation (a new buffer every step rows would break the record: preallocate);
    - a mask built per check at shape [32, kL] (a new size each check: use a preallocated mask buffer and a view);
    - RoPE positions and the tree's position ids, if they ride in setBytes (more rules) or arrays (written in place).
    - The engine diff (records at two kv) lists them all.
- Tue 01:3x, **pre-registration: a live CPU loader inside a replay (the routing-row handshake, toy).**
  - Why: in the engine, each layer's post_ids writes a routing row that the CPU loader reads. The loader loads the
    experts and writes a release, which the GPU spin then polls. A replay of the check must keep that handshake live.
    - Today MLX commits after every post (the 64 MB bump), so the row reaches the CPU at the buffer's end.
    - The engine also has W1-PICKUP flags (BigMoE ext/bigmoe_ext.cpp:1016-1027): BIGMOE_POST_SYS writes the row
      through a coherent(system) pointer with system-scope fences ("the loader sees it in ~7 us"). BIGMOE_SPIN_COHERENT
      polls the release with coherent(system) loads. Device-scope stores reach the CPU only when a later dispatch
      flushes them (ms if only the spin follows).
  - Toy (scripts/m1/rec1a/rec_loader_toy.py, MLX custom kernels with the engine's constructs):
    - post: writes [seq, v = int(1000 x0) & 0xFFFF] through a coherent(system) pointer with system-scope fences.
    - Loader: a C pthread in libmtlrec. On a new seq it writes [seq, slot = 3 v + 1] (the slot first, the tag with
      release order).
    - spin: one simdgroup polls the release with coherent(system) loads (cap 2,000,000 polls), then
      y = x * slot + 1 and flag = [seen, polls, slot].
    - Run like today's engine: the post in its own command buffer, then the spin. One record; 16 checks, each with
      new x and its own seqs. Every form gets a fresh seq, so a stale release can never pass.
  - Lines:
    - (1) MLX's own run: 16/16 with the release seen, and slot == 3 v + 1.
    - (2) The segmented ICB replay (one command buffer per recorded one): 16/16 seen, the same slot, y bitwise == MLX's
      run on the same x.
    - (3) The ONE-buffer ICB replay (post, loader round trip and spin inside one command buffer): 16/16 likewise.
    - (4) Control: the one-buffer replay with the loader stopped (cap 200,000): the spin times out, 4/4.
    - (5) Info: the same with a DEVICE-scope post (today's default without POST_SYS), one-buffer replay: seen or not,
      and the polls.
    - Info: polls per check for each form, a proxy of the round-trip wait.
    - PASS = (1)-(4). If (3) fails while (2) passes, the replay needs one command buffer per post segment (milestone 2
      then budgets ~49 command buffers per check).
  - Run: share lock "MEM 1 GB", CPU 200 (the loader busy-polls one core), seconds of GPU (one simdgroup spinning).
- Tue 01:22, **loader-handshake toy result: PASS on all four lines** (data/m1/rec1a_loader_toy.out;
  scripts/m1/rec1a/rec_loader_toy.py; mtlrec.m 2fe97943, dylib 939ff025; share lock, peak 0.05 GB).
  - (1) MLX's own run (the post in its own command buffer, then the spin): 16/16, the release seen, slot = 3 v + 1.
  - (2) Segmented ICB replay (2 segments): 16/16, bitwise == MLX's run on the same x.
  - (3) ONE-buffer ICB replay (post -> the C loader's round trip -> spin inside one command buffer): 16/16, bitwise.
  - (4) Control, loader stopped: the spin timed out 4/4 (200,000 polls, 76-91 ms of GPU; ~0.4 us per poll).
  - Polls: 0 in every check of (1)-(3). The coherent(system) row reached the C loader and its release came back before
    the spin's first poll.
  - (5) Info, a DEVICE-scope post (no POST_SYS) in the one-buffer replay: the release was still seen 8/8, but after
    23-1,034 polls (~10 us to ~0.4 ms of GPU spinning per post).
  - What it shows for milestone 2: the replay can keep the loader live in ONE command buffer per check. That needs the
    engine's W1-PICKUP pair: BIGMOE_POST_SYS=1 (coherent(system) row) and BIGMOE_SPIN_COHERENT=1 (coherent polls).
    - Then no per-post segments (~49 command buffers) are needed.
    - Without POST_SYS the handshake still completes, but each post can cost up to ~0.4 ms of spin. Over 48 layers that
      is up to ~20 ms per check, so POST_SYS is the form to use.
    - The segmented replay stays available as the fallback (rec_replay_icb_seg).
- Tue 01:3x, **pre-registration: per-call input arrays found and patched through the record (buffer-word rules, toy).**
  - Why: the engine builds small arrays per call from CPU values, e.g. offs[3] = [base region offset, side region
    offset, seq]. A replay holds no Python reference to them, so it must find them in the record (the record retains
    every bound buffer) and patch their contents per check.
  - Recorder: rec_buf_len / rec_buf_read (the contents of a recorded binding's buffer), and rule types 4 / 5 (an
    int32 / int64 word in that buffer's contents).
  - Toy (scripts/m1/rec1a/rec_bufrule_toy.py): y = x * offs[0] + offs[2], with offs = mx.array([st(s), 7, s]) built
    inside the call (st as before) and N = 4,096.
  - Lines:
    - (1) Diffing the contents of the small bound buffers (<= 4 KB) between two records (s = 5, 6) finds both seq words
      of offs (int32 st(s) and s, matched to their formulas). Any other changed word is listed (info: GPU-written
      small outputs).
    - (2) With the two buffer-word rules applied per check (x written in place via the record's binding, not a Python
      reference), the mode-2 and ICB replays are bitwise == MLX on 16 new s.
    - (3) The stale control misses 16/16.
    - PASS = (1)-(3).
- Tue 01:27, **buffer-word toy result: KILL as registered** (data/m1/rec1a_bufrule_toy.out; mtlrec.m 7ae76187, dylib
  70f07c1f). Line (2) failed: 0/16 for both replays. (3) passed vacuously. (1) failed: no word matched the int32
  formulas.
  - Cause, from the dump: MLX DONATED the per-call offs buffer (int32, 12 bytes) to its astype output (float32, the
    same size). The conversion ran in place, so after the run the buffer holds floats [2023.0, 7.0, 5.0], not the
    int32 inputs.
    - The after-run snapshot sees outputs, so no word matched the int32 formulas. Only float copies changed: offs/o and
      the two 4-byte gather results of o[0] and o[2].
    - A replay also starts from that converted buffer unless the input is rewritten before EVERY replay.
  - Lesson for the engine record: a per-call input array may be consumed in place by the graph itself (donation).
    - Inputs must be read at bind (encode) time, when the CPU-written values are still there.
    - They must be rewritten in full before every replay.
    - The engine's ext kernels (post_ids, the one-call MoE) do not donate, but any MLX elementwise op on a temporary can.
  - Next: a recorder change (snapshot small shared buffers at their first binding during the record) and a NEW
    pre-registered run (v2 below). This KILL stands as the result of the design as registered.
- Tue 01:3x, **pre-registration v2 (a new run after the KILL above): bind-time snapshots.**
  - Recorder: each small shared buffer (<= 4 KB) is copied at its FIRST binding in the record (rec_buf_snap), before
    the GPU runs, so per-call CPU-written inputs are seen even if the graph later consumes them in place.
  - The same toy (scripts/m1/rec1a/rec_bufrule_toy2.py). The rules are applied before EVERY replay (inputs rewritten
    in full), and the control applies them with the PREVIOUS check's s (a true stale-input control).
  - Lines:
    - (1) The diff of the bind-time snapshots (s = 5, 6) finds both seq words of offs as int32 (st(s) and s), in one
      buffer.
    - (2) With the rules applied before every replay (x written through its binding), the mode-2 and ICB replays are
      bitwise == MLX on 16 new s.
    - (3) The stale control misses 16/16.
    - PASS = (1)-(3).
- Tue 01:3x, **v2 results** (mtlrec.m 668a4185, dylib 73da262e; share lock, peak 0.05 GB).
  - **Run 1 (data/m1/rec1a_bufrule_toy2_run1.out): KILL as run.**
    - (1) PASS: the bind-time diff found exactly the two int32 seq words of offs (st(s) 2023 -> 663 and s 5 -> 6, one
      buffer). The 2 other changed words are the gather outputs' pre-run contents (info).
    - (2) mode 2 16/16 but ICB 0/16.
    - (3) 16/16.
    - Cause: a SCRIPT bug against the registered protocol. The rules were not re-applied before the ICB replay, which
      ran right after the mode-2 replay had converted offs in place, so it read floats as int32. This is the
      donation effect from v1.
  - **Run 2, the script fixed to follow the registration (rules before every replay), the same lines:
    PASS (data/m1/rec1a_bufrule_toy2.out).**
    - (1) PASS.
    - (2) mode 2 16/16 and ICB 16/16 bitwise == MLX, with x written through its recorded binding (found by contents)
      and offs patched only through the record (two buffer-word rules).
    - (3) Stale control 16/16 misses.
  - What it shows for the engine record:
    - per-call CPU-built inputs (offs) are found from bind-time snapshots and patched through the record;
    - they must be rewritten before EVERY replay, because the graph may consume them in place.
    - Rule application costs ~0.5 us per call, so this is free.
- Tue 01:4x, **milestone 2: the engine run, PLAN (a draft, not yet pre-registered; it needs a ~31 GB slot).**
  - Harness: Leap's c25, run_c25.sh -> c25_env.py + c25_loop.py, with checks run by c25_engine.check(). Each check is
    a verify forward, the logits read and a full-accept commit, rolled back to the window's prefill.
    - The c25 load: full 125B via load_streamed, pools 21.2 GB, lifetime peak ~31 GB.
    - My wrapper module is put in sys.modules as `c25_engine` before c25_loop runs (Leap's files untouched).
    - It wraps window(): after the timed rounds of the first window, for one arm (F32c6i, or F32o first), it records,
      diffs, replays and compares.
  - Env: W1-PICKUP on (BIGMOE_POST_SYS=1, BIGMOE_SPIN_COHERENT=1), so ONE command buffer per check keeps the loader
    live (loader toy, c405188). The segmented replay is the fallback.
  - Steps:
    1. Record check A (seq s) and check B (seq s + 1), each around the verify forward + the logits eval.
    2. Diff A vs B: setBytes words, buffer offsets, bind-time buffer words; everything else must be equal.
    3. Fit the changed targets to the map's formulas (MODULE1A_ENGINE_MAP.md):
       - the seq word of post_ids' and spin_wait's int4;
       - the table offset ((seq * nl + L) mod 1024) * 17 * 32;
       - the region offset ((seq * nl + L) mod 128);
       - offs[3].
    4. Build the ICB from B.
    5. For 16 checks: the engine's own check (the reference logits) -> rollback -> s = Gate.new_pass() ->
       rules_apply(s) -> restore any state input the forward consumes in place -> the one-buffer ICB replay.
       Then compare the recorded logits bitwise with the reference, and require status timeouts to stay 0.
  - Open points the record itself will answer:
    - (a) which state buffers the forward consumes in place (donation), and so are restored from S0 before each
      replay;
    - (b) blit ops in the check (the ICB build refuses them; mode 2 or segments handle them);
    - (c) that the loader thread encodes no GPU work during a record (compare the recorded commits with Leap's
      266 / check);
    - (d) prediction counts (npr): constant under c25's rollback, variable in real decode (fixed shapes then).
  - Before it runs: a review of this plan by Leap (c25's owner), a dated pre-registration, and a slot from NewIdeas
    (~15 min, >= 31 GB lifetime, NO-SWAP >= 39, Kamen away).
  - Milestone 1's official number: scripts/m1/rec1a/run_rec_quiet.sh (200 repeats inside a quiet lock the caller
    holds; ~3.2 GB; ~2 min), in the ~10 min quiet slot Leap's split plans for Wed.
