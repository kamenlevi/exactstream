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
- Tue 06:5x, **Leap's review of the engine-run plan (06:48), and the amended plan. PRE-REGISTERED here, before any
  engine record.** Leap: "OK to pre-register once 1-4 are in the plan." Each point and what it changes:
  1. **MLX allocator reuse (the big one).** After record B, MLX's buffer cache can hand B's MTLBuffers to live arrays,
     such as the reference logits or the cache states. A replay would then write into them, and could even hide its
     own error. Changes:
     - `mx.set_cache_limit(0)` + `mx.clear_cache()` from just before record A to the end of the test, so freed buffers
       go back to Metal instead of MLX's cache.
     - The recorder already retains every object an op names (mtlrec.m, header line 10). So B's buffers stay alive
       and MLX never hands them out again.
     - The reference logits are copied to numpy before every replay.
  2. **What the verify forward reads** (Leap's list, c25_engine.check):
     - the token ids: a new array per check, same content under the rollback;
     - the KV prefix: in place if slice_update donates;
     - the snapshot's states S0: kept alive by the snapshot, not donated;
     - ref2's KEEP-C caches: persistent;
     - the PLE n-gram stage;
     - BigMoE's pools and slot tables (the seq formulas).
     Changes:
     - (a) Before every replay, each small buffer (<= 4 KB) whose pointer differs between A and B is rewritten from
       B's bind-time snapshot (token ids, offs). Then the seq rules are applied.
     - (b) Leap's proof: hash every array of states(P) (S0 and the KV caches, append rows included) before and after
       the first replay. The hashes must be equal (line 4).
  3. **Commit counts.** A correction to the premise: c25's env already has W1-PICKUP ON.
     - c25_loop.py lines 76-81 setdefault BIGMOE_SPIN_COHERENT=1 and BIGMOE_POST_SYS=1 under R57_PRESET=45, with
       R57_D45 and R57_COMBINE at their default 1. c25_env.py sets the preset and nothing sets the flags to 0.
     - I evaluated lines 36-81 under c25_env's exports (06:5x). Before: both unset; after: both "1".
     - So try B's 266 commits per check are the W1-PICKUP count, and the record needs no env change: it runs in
       try B's env.
     - "One command buffer" is my ICB replay's property, not MLX's: MLX keeps its own split. Record A's commits are
       counted (info only).
     - No timing comes from this run.
  4. **Seq.** Each replay takes s = G["BG"].new_pass(): strictly new, and = the reference check's seq + 1. That makes
     its tables the ones the loader published a pass ahead.
     - post_ids writes the loader row on every pass, the fast path included (EXT:1112-1128).
     - The loader accepts a row only for a new seq (MODULE1A_ENGINE_MAP.md, EXT:1679-1714) and counts it in
       ext.stats()["gated"] (EXT:1755).
     - So line 3: the replay's gated delta must equal the engine's own check's, and be > 0. Timeouts = 0 alone
       don't count.
  - Leap's extras, adopted:
    - C25_PROF, C25_CPUSPLIT and C25_PASSK unset. mtlprof interposes Metal too; the wrapper asserts they are unset.
    - knobs(G, arm, n) / unknob(G) around every record and replay, as check() does.
    - c25_engine stays Leap's file. My wrapper module goes in as sys.modules["c25_engine"]; tue1 imports it live until
      ~12:00 today.
  - **The run** needs a NewIdeas slot: ~15 min, MEM 35 GB, NO-SWAP >= 43, Kamen away.
    - Sizing: try B's lifetime peak 31.4 GB (gpubB_run.txt) + up to 3 GB of retained record buffers.
    - R57_VTIER is set empty, which turns off the 7.45 GB volatile victim tier (outside phys_footprint; it is why
      Leap's c25 lock reads MEM 43). It only serves misses, and the warm checks have none (misses 0.0 in try B).
  - Setup:
    - c25's env, plus C25_ARMS=F32s,F32o,F32c6i, C25_KEYS=dd__mc_27921 (one window), C25_WARM=1, C25_ROUNDS=0.
      - Amendment (Tue 07:0x, before any run): C25_ROUNDS=1, not 0. c25_loop.py calls c25_engine.window() only when
        R57_G32TIME is nonzero (line 1934); 0 would run the screen's own modes instead. One timed round of the 3 arms
        adds 3 checks.
      - Amendment (Tue 07:0x): the loop runs from Leap's c25_loop.py source with one line changed in memory, OUT =
        my run folder (data/m1/rec1a_engine/), because its line 122 points every output at ~/Leap/data/week98/c25.
        No copy of the loop is written anywhere.
    - The wrapper calls Leap's window() for the warm-up round. Then it runs the test on the same window: arm F32o
      first (the bitwise one-pass forms), then F32c6i (Friday's number 2: C6 + int8). It skips later windows.
  - **Per arm:**
    1. Record A and record B: two consecutive checks of the arm.
       - Each runs inside the recorder: knobs; snapshot; the verify forward; the logits astype + eval; synchronize.
       - The engine's own commit + back() follow, outside the recorder.
       - If the retained buffers that are new in B total > 3 GB, stop before any replay.
    2. Diff A vs B:
       - The op sequence, kernels, setBytes layout and canonical bindings must match.
       - Every changed setBytes word, buffer offset and small-buffer word must be explained by a MODULE1A_ENGINE_MAP.md
         formula, using A's and B's seqs.
    3. Register the rules and build the ICB from B. If the ICB build refuses (blits), use the mode-2 replay (also one
       command buffer).
    4. 16 rounds, each:
       - the engine's own check (logits -> numpy; gated delta d_ref; status[0] and status[16] deltas);
       - back();
       - s = new_pass(); rewrite the small inputs; rules_apply_v([s]);
       - knobs; the one-buffer replay; unknob;
       - read B's logits buffer and the replay's gated, status[0] and status[16] deltas.
       - The states(P) hashes are taken around round 1's replay only.
  - **Lines (per arm):**
    1. Record + diff: same structure, and 0 unexplained changes. A FAIL means no replay: read the diff.
    2. The replayed logits are bitwise == the engine's own check's logits, all 32 rows, on 16/16 rounds.
    3. Each replay's gated delta == the engine check's right before it, and > 0, on 16/16; status[0] timeouts stay 0.
       Info: status[16] fast hits, replay vs engine.
    4. The states(P) hashes are equal before / after round 1's replay.
  - **Verdict:**
    - PASS: lines 1-4 on both arms.
    - BETWEEN: F32o passes and F32c6i does not (that arm's components to fix).
    - FAIL: F32o fails. A failing replay is a bug until shown otherwise; it gets a cause, not a KILL.
    - Info: record A's commits; the mid-record CPU waits, if visible; the retained bytes; the per-arm wall.
  - Known limit: per-call inputs > 4 KB are not snapshotted. They are the same under the rollback; if one differed,
    line 2 would catch it, not hide it. Real decode (milestone 3) needs that answered.
- Tue 07:1x, **the engine-run wrapper, tested on a stub engine (no 125B).**
  - scripts/m1/rec1a/rec1a_c25.py: the launcher and the pre-registered test.
    - It imports Leap's c25_engine read-only and wraps window(). It runs c25_loop.py's source with the OUT line
      changed in memory (it refuses to run if that line ever changes).
    - The diff fits each changed target to the plain seq or to ((seq x nl + L) mod M) x S + c over the gate's table,
      region and slot-ring layouts. M and S are read from the live Gate; the ring row count is ring.size / MAXK.
    - The loader's 'gated' count is read only once it has stopped moving for 5 ms: a fast-path pass does not wait for
      the loader.
  - Recorder: mtlrec keep() / keep_drop() (mtlrec.m 99143b1a, dylib 7a046e75).
    - Record A's objects stay alive through record B, so B's fresh buffers can't reuse A's addresses. That makes
      "bound in both with the same pointer" a sound test for persistent buffers.
  - The stub test found one gap; it is fixed. Small persistent buffers that hold the last pass's seq and are rewritten
    in the check (the toy's post row and release word) were flagged unexplained. Classification now:
    - a word the check itself overwrites is state (or pre-run garbage, if per-call), listed and ignored;
    - a word that changes between checks without being written stays unexplained.
  - Stub (scripts/m1/rec1a/rec1a_c25_stub.py): every seq-dependent target of the engine map in toy form:
    - a setBytes seq word, a table-block buffer offset, a ring-start setBytes word, the CPU-built offs, a
      system-scope post + coherent spin against mtlrec's toy loader, an in-place KV write, a recurrent state;
    - seq-invariant logits when patched consistently.
  - Results, data/m1/rec1a_engine/stub_*_rec1a.out (all four AS EXPECTED; seqs cross 1024):

    | stub mode | what it tests | lines 1-4, both arms | verdict |
    |---|---|---|---|
    | ok | the whole path | 6 rules (seq x 3, ring L=3, tab L=2, reg L=4); 16/16 bitwise; 16/16 served; hashes equal | PASS |
    | unexplained | an s x s setBytes word | line 1 FAIL (the word flagged), no replay | FAIL |
    | stale | the wrapper's new_pass returns the last seq | line 2 16/16 bitwise, line 3 0/16 (gated 0) | FAIL |
    | badrule | the table rule reads row L + 1 | line 2 0/16, lines 1, 3, 4 pass | FAIL |

  - The stale mode is Leap's point 4 in miniature: stale slots can still give bitwise logits under the rollback.
    Only the served-row count tells.
  - Still to do before the slot: a runner (run_rec1a_c25.sh: slot times, the away rule, MEM 35 GB share lock,
    NO-SWAP >= 43, swap guard, hard stop) and a flag check of the real loop up to its model load.
- Tue 07:2x, **the engine-run runner and the flag check: ready for a slot.**
  - scripts/m1/rec1a/run_rec1a_c25.sh "<START>" "<HARD>" follows the pre-registered setup. It waits for the slot,
    holding nothing, until no c25 / rec1a loop runs and Kamen is away (read twice). Then:
    - HOLD / PAUSE / lid / HEALTH ACT checks; the NO-SWAP gate >= 43 on two reads;
    - a share lock whose reason starts 'MEM 35 GB'; a +2 GB swap guard; a yield when Kamen is back; TERM at HARD;
    - the end marker data/m1/rec1a_engine/REC1A_END.
    - A DRY pass checked the gates in order (rec1a_1117_dry_runner.log).
  - Flag check (RJ_FLAGCHECK=1): the real c25 loop in the run's exact env, up to its model load, with no model and no
    GPU work. Leap's head-test pattern. rc 97 with the marker (rec1a_1117_fc_flagcheck.txt):
    - OUT = data/m1/rec1a_engine and c25_engine wrapped;
    - BIGMOE_POST_SYS='1', BIGMOE_SPIN_COHERENT='1' live in the loop (the 06:5x correction, seen in the running loop);
    - R57_VTIER='', R57_G32TIME='1', WARM '1', the one window, C25_PROF / CPUSPLIT / PASSK unset, MAXK 320.
    - Nothing was written under ~/Leap.
  - Next: a slot from NewIdeas (~15 min, MEM 35 GB, NO-SWAP >= 43, Kamen away).
- Tue 07:2x, **the slot: Thu 19 Nov, START 07:30, HARD 09:30 (NewIdeas 07:2x).**
  - Why not sooner: every earlier gap is on Saturday's critical path.
  - Armed 07:26 as a frozen copy, data/m1/rec1a_engine/run_rec1a_c25_1119.frozen.sh (pid 38027, tag rec1a_1119).
    It starts when its gates hold inside the window; latest start 09:10.
  - Pins at arming (data/m1/rec1a_engine/pins_1119.txt): the wrapper, the recorder and Leap's three c25 files by md5.
  - Two runner changes before arming:
    - The flag check also runs inside the slot, right after the trigger. If Leap's files change by Thursday and the
      loop no longer reaches its model load in this env, the runner skips cleanly.
    - The "a c25 / rec1a loop is running" test counts only Python processes. A monitor shell whose text names
      c25_loop.py no longer blocks it; tested both ways with fake processes.
  - BigMoE's T8-32 dumps (~12.5 GB) may overlap; the NO-SWAP gate waits them out.
- Wed 02:2x, **MILESTONE 1's OFFICIAL NUMBER (module 1a): PRE-REGISTERED before any run.**
  - The slot (NewIdeas 02:1x): after Leap's slot #3 ends, with Kamen away (>= 900 s), behind any quiet ticket already
    queued; latest start 14:30.
    - "Slot #3 ended": its marker C25_s3_DONE (written on any end), or past its 11:30 hard end with no slot #3 process.
  - The run: scripts/m1/rec1a/run_rec_quiet.sh, i.e. rec_layer.py --fast with REC_SCOPE=layer, REC_MODES=9,
    REC_WINDOWS=4, REC_REPS=200, REC_RSET=1. It replays FAST layer 0 (123 commands) as an ICB, once with useResources and
    once with a residency set. Output: data/m1/rec1a_rset_layer_fast_quiet.out.
  - Run conditions: inside `peer lock quiet` (reason "MEM 4 GB ..."); NO-SWAP >= 12 GB (peak 3.2 + 8); a swap guard
    (+2 GB stops it); a yield if Kamen comes back (two reads 30 s apart); unlock right after.
  - Pins (md5):
    - rec_layer.py 21ad42c7, run_rec_quiet.sh 9f0314e9;
    - mtlrec.py e3403349, mtlrec.m 99143b1a, libmtlrec.dylib 7a046e75. This is milestone 2's recorder; Monday's rough
      run used mtlrec.m c56ada06 / dylib 2a63ad75.
  - **The official number:** the "ICB + residency set" line, CPU encode per replay, MEDIAN over the 200 repeats. The
    min and the split are reported with it.
    - Monday's line applies: PASS <= 50 us; BETWEEN 50-200; KILL > 300 us, or not bitwise. Bitwise = the 4 windows
      equal MLX and the repeats stay bitwise.
    - The useResources arm is reported as info.
  - Before the slot, a functional smoke under a share lock (REC_REPS=10; data/m1/rec1a_quiet/smoke_1118.out). It
    checks only that the bench runs and stays bitwise with milestone 2's recorder. Its timings are not quotable.
  - Runner: data/m1/rec1a_quiet/run_rec_quiet_1118.frozen.sh, simulated DRY before arming.
  - Wed 02:23, **smoke (functional, share lock, 10 repeats): PASS.** 4/4 windows bitwise, the repeats bitwise,
    (a) bitwise; peak 3.37 GB lifetime (data/m1/rec1a_quiet/smoke_1118.out). Its timings are not quotable (pre-registered).
  - Wed 02:26, **ARMED**: data/m1/rec1a_quiet/run_rec_quiet_1118.frozen.sh (md5 d2a725cf, read-only; pid 34206).
    - It calls the frozen bench copy run_rec_quiet.frozen.sh (md5 9f0314e9 == scripts/m1/rec1a/run_rec_quiet.sh).
    - DRY simulation: data/m1/rec1a_quiet/run_1118_dry.log. It covers slot #3's marker, the gates, a yield on
      Kamen's return with unlock and a retry, and the end marker. A rerun with Kamen back just as the job ended gave
      no false yield.
    - It ends with data/m1/rec1a_quiet/REC_QUIET_END.
- Wed 04:31, **MILESTONE 1, MODULE 1a: THE OFFICIAL NUMBER IS 9 us CPU PER LAYER REPLAY (ICB + residency set, median
  of 200 repeats; min 3). PASS (line <= 50 us).** data/m1/rec1a_rset_layer_fast_quiet.out.
  - The run (frozen runner, try 1; data/m1/rec1a_quiet/run_1118.log):
    - slot #3's marker 04:29:35; gates open 04:30:50 (away 21,025 s on two reads, NO-SWAP 41 GB);
    - the quiet ticket was granted at once (first in the queue); settle 20 s;
    - run 04:31:10-04:31:15, python rc 0; unlocked 04:31:15;
    - swap 7,635 MB at start and end; lifetime peak 3.27 GB.
  - Pins, md5 checked after the run, all as pre-registered: rec_layer.py 21ad42c7; run_rec_quiet.sh 9f0314e9 (== the
    frozen copy); mtlrec.py e3403349; mtlrec.m 99143b1a; libmtlrec.dylib 7a046e75.

    | ICB replay of FAST layer 0 (123 commands), quiet, 200 repeats | new-input windows == MLX | CPU encode per replay, median (min) | command buffer | encoder | useResources | execute + end |
    |---|---|---|---|---|---|---|
    | + an MTLResidencySet built once (204 allocations): **official** | 4/4 | **9 (3) us** | 0.9 | 6.2 | 0.0 | 1.7 |
    | + useResources over 203 buffers (info) | 4/4 (the mode-9 windows) | 28 (8) us | 0.8 | 6.1 | 18.7 | 1.8 |

  - Bitwise, as pre-registered: the residency-set arm's 4 new-input windows equal MLX (4/4). The repeats stay bitwise:
    the RESULT line's 200 mode-9 replays end with "repeats still bitwise: True" (min 9, median 28, max 41 us). And (a), the
    replay on the recorded inputs, equals the record.
  - Monday's rough 45 (31) us is not comparable: it ran under a share lock with Chrome busy and two other share runs,
    using the older recorder (c56ada06).
  - **A gap, found reading rec_layer.py after the run.** The mode-9 replays use useResources (mtlrec.m rec_replay_icb).
    The residency-set arm's own 200 timed repeats are not checked afterwards. Its 4 windows are checked before them
    (lines 289-307 of 21ad42c7), and "repeats still bitwise" comes from the useResources loop (lines 261-265). So the
    pre-registered clause holds as written, but the official arm's repeats are unchecked.
  - **Follow-up, pre-registered here before running.**
    - rec_layer.py gets an end-of-repeats check for both arms: "outputs still bitwise after the 200 repeats", against
      window 4's MLX outputs. It is the same test as the RESULT line's.
    - One functional run: REC_SCOPE=layer REC_MODES=9 REC_WINDOWS=4 REC_REPS=200 REC_RSET=1, under a share lock
      "MEM 4 GB", with NO-SWAP >= 12 and Kamen away. Output: data/m1/rec1a_rset_layer_fast_check.out. Its timings are
      not quotable.
    - If the residency-set line is True, the PASS stands. If it is False, the PASS is WITHDRAWN under the "not bitwise"
      clause until there is a fix and a new quiet run.
  - Wed 04:37, **follow-up check (functional, share lock): True for both arms. The PASS STANDS.**
    data/m1/rec1a_rset_layer_fast_check.out; runner scripts/m1/rec1a/run_rec_check.sh (gates as pre-registered;
    NO-SWAP 41 GB, away 21,449 s; rc 0, unlocked; swap unchanged).
    - rec_layer.py is now bcc9759e (was 21ad42c7). Each arm's outputs are zeroed before its 200 timed repeats, then
      checked against window 4's MLX outputs. So the check proves that the repeats WROTE the right values, which is
      stronger than the pre-registered "still bitwise".
    - "ICB + residency set: outputs still bitwise after the 200 repeats (window 4's MLX outputs): True"; the useResources
      arm, True. The windows were 4/4 again, and so were (a) and the RESULT line's repeats.
    - Timings are not quotable (a share lock, the GPU busy beside it). For the record only, they match the quiet run:
      the residency set 9 (3) us, useResources 28 (10) us.
  - Wed 11:3x, **milestone 2's engine run (rec1a_c25) moves from Thu 07:30 to tonight** (NewIdeas 11:2x, for the
    orchestrator: no idle GPU while Kamen sleeps). The run and its pre-registered checks are unchanged. Only the trigger
    changes.
    - The Thu 07:30 runner was stopped while it waited. It held nothing, left no process and wrote no marker.
    - New wrapper scripts/m1/rec1a/run_rec1a_c25_on_marker.sh (frozen copy
      data/m1/rec1a_engine/run_rec1a_c25_on_marker.frozen.sh, md5 1aa1a35a). It waits for BigMoE's end marker
      ~/BigMoE/data/r57/NIGHT1118_BIGMOE_END (after BigMoE's CERTIFY98 capture + count check, ~02:40). Then it runs the
      frozen runner run_rec1a_c25_1119.frozen.sh unchanged, with START = now and HARD = now + 90 min, no later than
      Thu 09:30. All of that runner's gates and guards stay (MEM 35 GB share lock, NO-SWAP >= 43, Kamen away).
    - If BigMoE's marker has not come by Thu 07:30, or HOLD appears while it waits, nothing runs and the wrapper writes
      the END marker itself.
    - Leap's slot #4 starts on the END marker data/m1/rec1a_engine/REC1A_END, which is written on any end.
    - A stub simulation of the wrapper's 8 paths behaved as designed: the marker; no marker by the deadline; the marker
      or the END marker present at arming; the runner ending without END; under 40 min before the cap; HOLD while
      waiting; the cap setting HARD.
    - Flag check against Leap's current c25 (b31dd1f): it reached the model load, with the flag line identical to
      07:52's (data/m1/rec1a_engine/rec1a_1118fc2_flagcheck.txt).
    - The timed arm (replayed vs MLX check) does not ride in tonight's slot. It needs the replay inside c25's decode
      path, which is the installer itself, and that can't be engine-tested before the slot. It is also only worth timing
      if tonight's run is bitwise. It waits for a later slot.
    - Item 12's target, from NewIdeas' overhead split (11:2x; research/week98/ovh/ovh_split.py on BigMoE's s98b
      records, 4 keys x 128 tokens, 6 arms):
      - Between checks, outside the check and draft timers: ~0.05-0.2 ms per cycle. The ~3 ms is not there.
      - Inside the check timer (medians): build 90-107 ms, wait 15-16 ms, remainder 3.4-6.7 ms.
      - Those runs are SSD-bound (535-595 expert misses, ~650 MB per check), so build is mostly I/O there. The warm
        split comes from BigMoE's Thursday key test with R57_TCPU=1 (main-thread CPU per check).
      - So the installer targets the check timer's build + remainder, not the commit.
  - Wed 15:0x-15:2x, **the engine run happened at 14:59 and was a NOT-RUN. A bug in my harness stopped it at its first
    record, so lines (1)-(4) were never tested.** This is not a replay verdict.
    - BigMoE's end marker appeared at 14:58, not ~02:40 ("BigMoE NIGHT1118 END 11-18 14:58:40: capture + post sequence
      done (diag95, M4C-chain s98n, (iii), dev)"). The wrapper fired at 14:59, and every gate was met:
      - Kamen away 58,754 s;
      - the flag check reached the model load;
      - NO-SWAP 15 GB, then 46 GB at 15:01:51 (need 43);
      - the share lock was taken, and swap was unchanged.
    - c25's own window ran: 1 warm-up and 1 timed round of the 3 arms. Then the test stopped at record A of F32o, with
      `AttributeError: module 'bigmoe.gated' has no attribute 'seq'` (rec1a_c25.py:131). The run printed
      "RESULT: FAIL", with lines (1)-(4) False on both arms, 0 s, lifetime peak 43.06 GB.
    - Data, all in data/m1/rec1a_engine/: rec1a_1119_run.txt, rec1a_1119_rec1a.json / .out, c25.json,
      rec1a_1119_env.txt, rec1a_1119_flagcheck.txt, rec1a_1119_runner.log, rec1a_1119_wrapper.log, REC1A_END.
    - **Cause.** In the real engine, G["BG"] is the bigmoe.gated MODULE (c25 reads its .ext).
      - The per-pass seq, new_pass() and the tables / region / ring / status buffers live on the streamer's Gate
        INSTANCE. Every streamed switch layer holds it as _bm["gate"] (bigmoe/stream.py, install).
      - The stub engine passed the Gate itself, so the stub tests could not see the mismatch.
      - The flag check stops at the model load, before G exists, so it could not see it either.
    - **Fix (15:2x).** GateView takes the one Gate from the switch layers' _bm["gate"], with ext and MAXK from the
      module. In the stub it uses G["BG"] as is. Nothing else in the test changed, and its first line now prints the gate
      it found.
    - **Tests of the fix (15:08-15:1x).**
      - scripts/m1/rec1a/rec1a_gateview_test.py: PASS. It uses fake objects shaped like the engine's. It checks: the Gate
        found through the layers, with a dense layer in between; seq read live; new_pass(); ext and MAXK taken from the
        module; the ring family kept; the counters; the stub's shape; two Gates or none refused.
      - The four stub cases (data/m1/rec1a_engine/stub_1118/) were all AS EXPECTED: ok PASS; unexplained, stale and
        badrule FAIL on their expected lines.
      - The other engine calls the test makes (snapshot_untrimmable_cache_lazy, verify_scope, commit_verified_window,
        knobs / unknob / back / check, states) all exist by name in Leap's c25_engine.py.
    - **Sizing.** The run's real lifetime peak is 43.06 GB, from the c25 harness itself (Leap locks the same harness at
      MEM 43 with gate 46 = 42.4 + 4). The 35 GB lock was undersized. The re-run passes RJ_MEM / RJ_GATE to the frozen
      runner, which reads both from env, so no edit is needed.
    - Leap's slot #4 started at 15:03 on the END marker and is done.
    - Next: a re-run slot (asked NewIdeas 15:2x). The pre-registered steps and lines are unchanged; only the gate lookup
      is fixed.
  - Wed 15:1x-15:2x, **re-run armed in NewIdeas' slot** (15:1x): today from 16:03 (latest start 18:30), else Thu 07:30
    (latest start 09:10, so it ends by 09:30). The pre-registration stands: the same steps and lines, only the gate lookup
    fixed.
    - Runner scripts/m1/rec1a/run_rec1a_c25_slot.sh, frozen as data/m1/rec1a_engine/run_rec1a_c25_slot_1118.frozen.sh
      (md5 ce1990b2). It is a new runner because the frozen one can't retry a refused share lock (peer lock share has
      no queue) and its only stop is a fixed end time, not a cap from the job's start.
    - Gates:
      - no PAUSE, the lid open, no HEALTH ACT, no c25 / rec1a loop, Kamen away, all on two reads 30 s apart;
      - the flag check;
      - NO-SWAP >= 47 (the measured 43.06 + 4, Leap's convention for this harness; NewIdeas: peak + 8 is for unmeasured
        jobs);
      - a share lock "MEM 46 GB" (43.06 + the 3 GB retained-bytes guard), asked again each minute if refused;
      - NO-SWAP read again 20 s after the lock.
    - Guards: a 20-min wall cap, swap +2 GB, HOLD or the lid, and Kamen back (a yield that waits again). TERM only.
    - END marker data/m1/rec1a_engine/REC1A_END on the final end only (overwritten, so gate on its mtime). Outputs go to
      data/m1/rec1a_engine/rerun_1118/, tag rec1a_1118r. Pins: data/m1/rec1a_engine/pins_1118r.txt (rec1a_c25.py
      5c1bc938).
    - A stub simulation of 11 paths behaved as designed:
      - a plain run; the lock refused twice; NO-SWAP dropping after the lock; the wall cap;
      - a yield, then a re-run; window 1 missed, window 2 run; HOLD while waiting; neither window;
      - the runner stopped by a signal (job killed, no END); HOLD during the run; the real flag check.
  - Wed 16:4x-16:5x, **today's slot yielded; re-armed for Thu 07:30-09:10 at start gate 46** (NewIdeas 16:4x).
    - Blob 1's key test passed at 16:43 and EvalSuite's tier 1 #1 starts at once (~17:05, Saturday's critical path). The
      waiter never started: NO-SWAP read 35-46 GB against its 47 gate from 16:04 to 16:44. Its 16:03 flag check reached
      the model load (rec1a_1118r_flagcheck.txt).
    - The waiter (pid 88522) was stopped by TERM at 16:52:22, holding no lock ("runner stopped by a signal: no END
      marker"). REC1A_END is unchanged (11-18 15:02:21).
    - Re-armed at 16:52:39 (pid 8347): the same frozen runner (ce1990b2) and rec1a_c25.py (5c1bc938), with RJ_GATE=46 and
      RJ_MEM=46. Window 1: Thu 07:30, latest start 09:10 (so a 20-min run ends by 09:30). Window 2 is empty (09:10-09:10).
      The log and outputs are unchanged (rec1a_1118r_runner.log, rerun_1118/).
    - Start gate NO-SWAP >= 46 (NewIdeas): the 14:59 run started at 46 and loaded to 43.06 GB with no swap. The +2 GB swap
      stop and the 20-min cap bound the risk. The lock stays MEM 46.
    - The run's env matches the 14:59 run's 55 recorded lines, apart from REC1A_OUT. It comes from Leap's c25_env.py
      (unchanged since 11-16 13:05) over BigMoE's t1cx45t1c1_env.txt (unchanged since 11-16 09:10).
    - END is written on the final end only: a run of any result, or a SKIP (HOLD while waiting, a failed flag check, or no
      start by 09:10). The pre-registration stands.
