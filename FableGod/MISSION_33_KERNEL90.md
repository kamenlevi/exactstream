# MISSION 33: THE EXPERT KERNEL TO ~90 % OF THE MACHINE'S STREAMING RATE (5-hour push, Wed 4 Nov 2026)

> (a quote from the project owner, kept private)
the kernel research and getting it to 90%" ... "we should do an expert kernel redesign push with a Fable session and try
to do it in the next 5-ish hours, if possible". Thinker: Fable A, who wrote mission 27's EXPERT_KERNEL.md. Builder: Leap
(ojox-4b), who builds and measures in the same window.)

## Why
- The expert kernel is about half of the GPU time of every 16-row check. NAX6 streams ~171 GB/s. The machine streams
  258-287 GB/s (Leap's C2), so NAX6 runs at ~60-66 %.
- At ~250 GB/s, a 16-row check's expert time drops from ~43 to ~29 ms: ~+15-20 % on today's check, and more as the
  rest of the check shrinks (dense pass near its floor, drafter 10.4 -> 5.4 ms today).
- Kamen's question: the best kernels reach 80-90 % of bandwidth, so why can't ours, and why can't we use their
  principles if they exist for the Mac?

## What we know (read first)
- ~/FableGod/EXPERT_KERNEL.md (mission 27, yours):
  - Per (expert, tier) group, the 16-row tensor-op set takes 7.32 us against 6.95 us of bytes at streaming.
  - Only ~12 % of the matrix-unit MACs fall on real rows (2.1 pairs per group).
  - So no 16-row tensor-op kernel reaches 85 % at the probe's op rate. Designs A NAX7, B TIER-MERGE, H HYBRID, C INT8-A,
    D ALU-only.
- What happened since, all in ~/Leap/TABLE_350.md (NAX-OCC, NAX-ABL, NAX-MICRO, MERGE-*), ~/Leap/HANDOVER.md (N7-*),
  and the MERGE kill note 3dfba1d:
  - NAX7 variants (pipelined, loader / op simdgroups, dispatch fusion): bitwise and KILL.
  - Tier merge X / X2 / B: KILL. Groups fall 12-19 %, but each call is slower: "the NAX call is latency-bound per
    threadgroup, not op-bound". This contradicts design B's premise; correct that row.
  - Occupancy / threadgroup-memory footprint, vectorized epilogue, a tight grid, two streams: KILL.
  - NAX6-N ablation at S 16 (share of the call saved when a part is removed): op 24-28 %, epilogue 11 %, decode 10 %,
    code loads 6-7 %, A loads 4 %, skeleton 17 %.
  - FAST: H-1i hybrid +9.1 % (quiet). NAX6-NH (native + hybrid) +23.3 % over NAX6 at S 16.
  - INT8 design C = NAX6 speed: KILL.
- Exactness constraint: a kernel in the EXACT track must be row-invariant (a 16-row check == one-token decoding bit for
  bit). Kamen approved a NAX reference today (pending EvalSuite's same-scores certification Thu ~08:15). A kernel whose
  bits differ from NAX6 needs its own certification, and the H hybrid is not row-invariant.

## Deliver (in 5 hours, by ~23:00)
1. **The answer to Kamen's question, plainly.** Why do Apple-silicon kernels that reach 80-90 % (MLX's large qmv / GEMV,
   llama.cpp's Metal mul_mv / mul_mm_id for MoE, metal-flash-attention, MPS) not carry over to our 16-row
   sparse-expert case, and which of their principles DO carry over?
   - Do a quick prior-art pass, with links, on sparse or grouped MoE GEMM at small batch: grouped GEMM, MegaBlocks-style
     block sparsity, expert sorting, persistent kernels, stream-K, Apple GPU tensor-unit papers or blogs 2025-2026.
2. **The ranked designs that could reach >= 85-90 %, EXACT first, then FAST.** Each priced with your mission-27 method,
   with the risk and the cheapest check. Candidates to consider:
   - persistent per-layer kernels that keep the matrix units fed across groups;
   - denser groups (more real rows per op: row packing across experts, 8-row or other op shapes if the hardware
     allows, 32-row checks);
   - the ALU-only path D as a new row-invariant reference;
   - cutting the 17 % skeleton and the 11 % epilogue;
   - anything new.
3. **For the top one or two: a microbenchmark spec Leap can build and run tonight** (real layer-20 records + routing,
   S 8 / 16 / 32, GB/s and us per group, bits vs NAX6), with the PASS / KILL lines pre-registered.
4. Hand Leap the spec as soon as #3 exists. Don't wait for #1-2 to be polished.

## Rules
- Write only in ~/FableGod/ (answers: ~/FableGod/KERNEL90.md). Marks [K] / [P] / [G]. Every claim names its source.
- Kamen's rules: exact stays exact (bitwise to its reference); FAST is judged by task scores; no paid compute.
- Leap's GPU runs: one share lock at a time, small MEM (Breakthrough's map batch runs ~17:38-19:05, 18 GB).
- When done, message the orchestrator: the verdict in one line, then the ranked list.
