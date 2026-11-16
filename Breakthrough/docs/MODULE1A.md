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
