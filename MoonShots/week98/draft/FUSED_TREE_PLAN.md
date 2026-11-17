# Fused tree level for the r57 draft (MoonShots, draft line main line; Tue 17 Nov 11:0x)
Labels: [K] measured (raw data path), [P] derived, [G] estimate.

## Where we are [K]
- r57 drafts t30b at **Q8 + the 32K screen** (orchestrator 09:4x, on Q8's PASS). BigMoE wires it from Wed with my
  **sync-free** tree loop as the default (176 + 120 trees bitwise; draft/sf_tree.py).
- Real-format per pop, B = 1 (sf_tree.py --real: QuantizedLinear 8-bit g64 + Leap's ScreenedHead 32K + 4,096 extras):
  - 1.11 ms sync-free vs 1.60 host-synced [K draft/sf_bit_real_*.json];
  - ~16.6 ms per tree-16 block (15 pops).
- Bytes per pop ~163 MB -> 0.56 ms at 292.7 GB/s [P]. So **~0.55 ms per pop is fixed cost**.
- Where the fixed cost sits (sf_prof.py; each stage alone minus the ~200 us eval baseline, rough) [K draft/sf_prof_he_t*.json]:

  | stage | ~fixed us | MLX primitives |
  |---|---|---|
  | tree_attn (+ KV concat / gathers) | ~190 (t 8) to ~220 (t 14) | 209-305 |
  | finish: MLP HC read + MoE + combine | ~190 | 105 |
  | head + top-K (mixer HC + screened logits + logsumexp / argpartition / sort / exp) | ~110 | 44 |
  | qkv (Q8 q / k / v, q / k norms, partial RoPE) | ~76 | 58 |
  | hcA (attention HC read) | ~48 | 28 |

- mx.compile is no route: it is not bitwise (1 / 24 trees) and gains only -3.5 % [K draft/sf_bit_realc_*.json].

## Plan (bitwise vs the eager ops, as fused2 was; each step bit-tested on captured real inputs)
1. **Tree attention**, one kernel per query head (24), prefix + tree columns. The stock ops it must reproduce:
   - qg @ K1^T: fp32, M = 1;
   - * scale, where;
   - the J tree columns' (qg * AK_j).sum(-1) over hd 256;
   - max, exp(precise);
   - et @ V1: fp32, M = 1, K = S1;
   - the sequential o + ec_j * AV_j (no FMA contraction);
   - et.sum / ec.sum (row reduces);
   - the division.
   - Find each stock kernel's per-output order with fl_match-style candidates (gemv / gemv_t fp32, row_reduce
     variants), then one kernel.
   - Inputs captured: draft/attn_cap_he32.npz (S1 457, J 1-32), attn_cap_mb16.npz (S1 187, J 1-16).
   - Target ~190 -> ~20 us per pop [G].
2. **finish** (MLP HC read at Q8 + MoE: Q8 router, top-10, stored experts, Q8 shared expert, combine). Reuse fl_moe
   (S9-S11, the stored experts) with the router / shared expert at 8 bits; the HC read at Q8.
3. **head + top-K + frontier**: fused2's S14 (4-bit screened rows, same format) + the extras matrix; an S15-style
   tail (logsumexp + top-K) merged with sf_step.
4. **qkv + hcA + fuse**: 8-bit qmv_fast variants (fl_qmatch), the norms / RoPE.
5. Integrated bit test (fused SF vs eager HS, every node of every tree) + timing (quiet); then hand to BigMoE. If it
   passes, the persistent form follows.

## Dates
- Tue pm-Wed: step 1 (attention).
- Wed: step 2.
- Thu: steps 3-4.
- Fri: step 5 + hand-off.
- Target per pop ~1.11 -> ~0.7 ms [G], i.e. ~-6 ms per tree-16 block.

## CORRECTION (Tue 11:0x): the pop is GPU-BOUND; true GPU time per stage (mtlprof) [K draft/sf_gprof_he_t8.json]
- Diagnostic: an added pure-GPU read per pop slows the pop linearly (+50 MB -> +0.54 ms, +150 MB -> +1.61 ms), and
  async_eval only waits on GPU back-pressure [K draft/sf_cpu.log]. So the CPU isn't the bottleneck.
- Pipelined tree: GPU busy 1,100 us per pop, 287 dispatches per pop (HE+, t 8, real format).

  | stage | GPU us | byte floor us | overhead us | dispatches |
  |---|---|---|---|---|
  | finish (MLP HC + MoE + combine) | 281 | 134 | **+147** | 66 |
  | tree_attn | 77 | 3 | +74 | 94 |
  | head (mixer HC + screened logits, 55 MB) | 251 | 188 | +62 | 17 |
  | topk (logsumexp / argpartition over 36,864 + sort + exp) | 62 | 0 | +61 | 20 |
  | hcA (attention HC read) | 60 | 7 | +54 | 15 |
  | qkv | 169 | 124 | +45 | 38 |
  | fuse | 74 | 48 | +27 | 16 |
  | step (my single-thread frontier) | 24 | 0 | +24 | 1 |
  | oproj | 75 | 57 | +18 | 5 |
  | gather | 9 | 0 | +9 | 19 |
  | total | 1,083 | ~560 | ~+520 | 290 |

- **The fused tree attention (fta_kernel.py) is BITWISE on all 48 captured calls**, but its latency-bound single
  threadgroup per head (~45-80 us) is no faster than the stock ops at J ~9 (77 us GPU; its 94 small ops overlap). In
  the loop: -0.02 to -0.04 ms per pop.
  - The HE+ tree-32 run on windows 2-3 FAILED bitwise (0 / 24 trees), although it passed the captured window-0 calls.
    Unresolved; to debug before any use.
- **New order:**
  1. finish / MoE (+147): fl_moe's bitwise S9-S11 with the Q8 router and shared expert + the MLP HC read.
  2. topk + step (+85): one tail kernel (logsumexp in the stock order, top-K, sort, exp, frontier), from fused2's
     S15 tail.
  3. The three HC reads at Q8 (+54 each, incl. the MLP and mixer ones).
  4. qkv / fuse.
  5. A faster attention, only with more parallelism under the bitwise orders.
- Realistic: ~-0.35-0.45 ms per pop [G], i.e. ~16.6 -> ~10-11 ms per tree-16 block.

## RESULT so far (Tue 11:1x): stage fusion at B = 1 pays little [K]
- fta (bitwise per call) -0.02 to -0.04 ms per pop. The tail (not bitwise) ~0.
- Why: at B = 1 the GPU time is bytes (~0.56 ms) + latency-bound chains. MLX already overlaps the small independent
  ops (the "overhead" column above counts them serially), and a fused single-threadgroup kernel is as latency-bound
  as the chain it replaces.
- Realistic remaining from MoE + the three HC reads + qkv: ~-0.1 to -0.15 ms per pop [G, after discounting the
  overhead column by the ~50 % the two attempts realized]. That's -1.5 to -2.2 ms per tree-16 block, ~3-4 % of the
  cycle, for ~2-3 days of bitwise kernel matching.
- So the kernel line beyond sync-free is a "small lever" under the 07:3x rule. **Recommendation: PARK it at sync-free**
  (16.6 ms per tree-16 block, already the plan's ~15 ms class), unless the orchestrator wants the MoE fusion.
  The persistent overlap kernel is the only route to the byte floor (~0.56 ms per pop), and it's a multi-day bitwise
  build: not by Saturday.

## Risk
- BigMoE's r57 ops may differ from my harness's guess. It was asked 10:4x; the fused level follows its ops.
