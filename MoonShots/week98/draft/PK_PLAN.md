# PK: the persistent fused tree level for the Q8H32 head (MoonShots; Tue 17 Nov 2026)
> (a quote from the project owner, kept private)
line: build it all-out by Friday, so that it drops into BigMoE's r57 per-level loop as a flag.
Marks: [K] measured (source), [P] arithmetic on measured parts, [G] estimate.

## Pre-registered lines (Tue 14:5x, before any build; NewIdeas QUEUE M-46)
**Format.**
- BigMoE's r57 t30b loop as sf_tree.py --real specifies it (BigMoE's SPEC since 10:56):
  - dense: Q8 g64 for fuse, the three HC reads, q / k / v / o, router and shared expert;
  - experts: the stored experts, top-10;
  - head: Leap's ScreenedHead 32K + the session extras;
  - attention: tree attention over the head's prefix KV;
  - frontier: top-K 8 and the GPU frontier.
- B = 1. Tree-16 (15 pops) and tree-32.

**BIT TEST (the line).**
- PK == SF bitwise on every node of every tree. Compared per node: parent, token, depth, path_p (float64 bits), top-K ids
  and probabilities, and the state h.
- Trees: the 120 real trees of sf_tree's real bit test (HE+ tree-16 and tree-32, MBPP tree-16; the same windows and
  anchors).
- PASS = 120 / 120. Anything less = FAIL: report the first mismatching node and phase. No partial credit, no hand-off.

**TIMING (quiet lock).**
- PK and SF interleaved on the same windows; the first 2 rounds dropped.
- Report ms per pop and per tree-16 block.
- Fixed cost f = PK's pop time - the byte floor (bytes per pop / 292.7 GB/s).
- PASS: f <= 0.10 ms.
- KILL: f >= 0.25 ms (stage fusion's level: fused2 had f = 0.269).
- Between the two = PARTIAL: hand to BigMoE only if bitwise and >= 0.3 ms per pop faster than SF.

**DECISIVE SKELETON FIRST** (GRANDMA lever 2's ~10-min test).
- The same persistent kernel with the real bytes, phases and grid barriers, but non-bitwise math. Quiet.
- Skeleton f >= 0.25 -> KILL the line: the bitwise kernel cannot beat its own skeleton.
- Skeleton f <= 0.15 -> GO the bitwise build. Between the two: tune once, then decide.

**Exact:** yes. PK == SF bitwise, so the key test's verdict on SF carries over to PK unchanged.

## Where the 1.11 ms pop goes [K draft/sf_gprof_he_t8.json, HE+, t 8]
- GPU time 1,083 us against a byte floor of ~560 us (163 MB at 292.7 GB/s). 290 dependent dispatches per pop.
- Overheads per stage:

  | stage | overhead us |
  |---|---|
  | finish (MLP HC + MoE) | +147 |
  | tree_attn | +74 |
  | head | +62 |
  | topk | +61 |
  | hcA | +54 |
  | qkv | +45 |
  | fuse | +27 |
  | step | +24 |
  | oproj | +18 |
  | gather | +9 |

- The cost is at the dependent boundaries, not in the dispatch count. Measured in BET 7:
  - a dependent dispatch costs <= 1.3 us;
  - a grid barrier ~2-2.5 us;
  - a chained boundary loses ~16 us (drain + cold ramp);
  - the overlap bound (dependencies removed) was f = 0.092 on a C0 level.
- The SLC cannot hold weights ahead of their consumer: M-45 (today: knee ~24-28 MB, CPU fill dead) and BET 7's touch
  KILL. So the overlap has to come from next-phase loads issued into registers before each barrier.

## The kernel (PERSIST_DESIGN.md, ported to the real format)
- **Dispatch.** One dispatch per pop (v1); one per tree later if launches show. The grid = resident capacity on the 20
  GPU cores (probed). Every threadgroup loops over ~17 phases:
  1. fuse
  2. hcA down
  3. hcA up + mix
  4. q / k / v (+ q / k norms, partial RoPE, gate)
  5. attention scores (prefix + tree columns)
  6. attention combine
  7. o_proj (x sigmoid(gate))
  8. MLP HC down
  9. MLP HC up + mix
  10. router | shared gate + up
  11. route (fp32 softmax, top-10, scores) | shared down
  12. experts gate + up
  13. experts down + combine (the layer output h)
  14. mixer HC down
  15. mixer HC up
  16. screened head (36,864 logits, plus a local top-8 per threadgroup)
  17. tail: logsumexp in the stock order, top-8 merge, stable sort, exp, the frontier step, and the next pop's
      gather / ancestors
- **Per-output arithmetic** is the stock kernel's, so the bits match:
  - quantized.h's qmv_fast / qmv at 8 bits (fl_qmatch picks the variant per shape);
  - fl_exact's RMS virtual lanes;
  - fl_hc's HC read;
  - fta's matched tree attention (+ the > 512-key row_reduce variant, 063928a);
  - the moe_train path (fp32 router softmax, argpartition's total order, gather_qmm, the fp32 Q8 shared expert).
  - Small norms are recomputed by every threadgroup instead of paying a barrier.
- **Barrier.** A generation counter: atomic add, spin, device fence. A timeout sets a status flag, and the host re-runs
  that pop with SF's ops, so a barrier never hangs.

## Op form (proposed to BigMoE; its Wednesday loop)
- R57_TREE_PK=1, default OFF. It swaps tree_sf(R, lg0, K1, V1, anc, maskb) for pk.tree_pk(...): same inputs, same
  returned dict.
- The module is imported read-only from ~/MoonShots/week98/draft at a pinned commit, as Leap ships make_records3.
- PK packs the dense head weights into one blob at load. MLX views of that blob keep SF working without a second copy.
  The experts, the embedding table and the KV stay as BigMoE's own arrays.
- Because PK == SF bitwise, the flag can be switched on after Thursday's key test without re-running it. Only the timing
  runs again.

## Plan and ETA
1. **Tue 15:00-23:00:** skeleton (blob, barrier, phases on the real bytes, register prefetch). Quiet timing.
   Skeleton verdict by Wed ~09:00.
2. **Wed:** bitwise phases 1-4, 7-11, 14-15 (the Q8 GEMVs, norms, RoPE, HC reads, router, shared expert). Bit tests per
   phase on captured inputs (share lock).
3. **Thu (heavy part after my account's weekly reset at 17:00):** attention (5-6), experts (12-13), head and tail
   (16-17). Then the integrated 120-tree bit test and the quiet timing. Flag to BigMoE Thu evening, in time for
   runner 69's Thu-night timing if it passes.
4. **Fri:** fixes, the verdict (bitwise trees, ms per pop / per block, f), hand-off.

Confidence [G]:
- skeleton f <= 0.15: ~60 %;
- the bitwise PK on all 120 trees by Friday: ~50 %. Each of ~25 stock kernels must match bit for bit on the new format.

## Honest numbers [P]
- f = 0.10 / 0.15 gives:
  - a pop of 0.66 / 0.71 ms (SF: 1.11);
  - a tree-16 block of 9.9 / 10.6 ms (SF: 16.6);
  - HE+ at GRANDMA's calibration (211 tok/s, 55.7 ms per check): ~240 / ~236 tok/s (x1.12-1.14).
- GRANDMA's 261 assumes 0.37 ms per level for path A's 76.7 MB head. Path A failed its quality line (C2 x0.874). Q8H32
  reads ~163 MB per pop.
- f <= 0.10 means running at the measured overlap bound (0.092). The realistic range is 0.12-0.18 [G].

## SKELETON VERDICT (Tue 17 Nov 15:29, quiet lock): KILL by the pre-registered line (f >= 0.25)
**Measured (quiet; pk_skel2_quiet.json / .log, pk_qcold_quiet.log).**
- The tuned skeleton (v2, the one tuning pass) takes **1.053 ms per pop** against a byte floor of 0.609 ms (178.2 MB at
  292.7 GB/s): **f = 0.444**.
  - 17 grid barriers cost 0.029 ms per pop.
  - A plain stream of the same bytes reads at 289.5 GB/s.
  - 0 barrier failures. 40 x 512 threadgroups are co-resident.
- **Cold GEMV rate, the decisive fact.** MLX's shipped 8-bit qmv_fast reads cold weights at 260 / 257 / 236 GB/s (q_proj /
  o_proj / fc).
  - The impl is bitwise == mx.quantized_matmul on every PK shape (pk_qpf.py).
  - A NON-bitwise row-stream GEMV is no faster: 267 / 261 / 251 GB/s.
  - Prefetch variants change nothing.
  - So ~260 GB/s is this chip's cold rate for these GEMVs, whatever the order.

**Per-phase increments** (cumulative masks, pk_skel2_ablate.json; us per pop):

| phase | us | note |
|---|---|---|
| fuse | 76 | |
| attention HC | 13 | |
| q / k / v | 170 | 36.2 MB = 213 GB/s |
| scores | 11 | |
| softmax + PV | 90 | |
| o_proj | 68 | |
| MLP HC | 52 | |
| router + shared expert | 54 | |
| experts gate / up | 84 | |
| experts down | 54 | |
| mixer HC | 67 | |
| head | 202 | 53.1 MB = 263 GB/s |
| tail | 84 | |
| barriers | 32 | |

**Why f <= 0.10 is out of reach on this format.**
- 178 MB per pop at <= 260 GB/s is 0.685 ms, so f >= 0.076 from the GEMV rate alone, and the barriers alone take 0.029.
- **Optimistic ceiling** (us; every GEMV phase at the cold rate, small Q8 GEMVs at ~240 GB/s, attention 25 us (stock
  tree_attn: 77), tail 12):
  - big GEMVs: 580;
  - small GEMVs: 115;
  - attention: 25;
  - tail: 12;
  - barriers: 29;
  - total: ~0.76 ms per pop, **f ~ 0.15-0.20 at best**.
- That is a pop of ~0.76-0.81 ms vs SF's 1.11 (-0.30 to -0.35 ms per pop), i.e. -4.5 to -5.3 ms per tree-16 block:
  HE+ 211 -> ~228-232 (x1.08-1.10).
- The cost: the full bitwise build (attention, moe_train, head + tail, HC, norms, RoPE), ~3 days at ~50 % risk.

**What failed / what was learned.**
1. Each phase moves at most ~260 GB/s cold, so the persistent form saves only the boundaries: 1.11 (SF) -> 1.05.
2. Dedicated helper simdgroups with per-round threadgroup barriers made it slower (1.67-1.86 ms). BET 7 had already
   killed separate touch kernels.
3. Two persistent kernels must never overlap. Trees without a data dependency between them deadlocked into spin caps
   until chained through the frontier state. That is a hard integration rule for any PK.

**Recommendation.** KILL as pre-registered. The draft's lever is the bytes per level (a head that reads less), not the
fixed cost: f is bounded below by bytes / 260 GB/s.
