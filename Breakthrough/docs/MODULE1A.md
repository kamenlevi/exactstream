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
