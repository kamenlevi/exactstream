# MISSION 27: THE EXPERT KERNEL. Why it reads at ~60 % of the machine's speed, and how to reach 85 %.

> (a quote from the project owner, kept private)

**Why.** HC_READ.md closed the HC question: D1 + D5 (exact) take ~3.9 ms off the dense pass, and Leap gets them. The largest engine gap left on the route to 350 is the experts:
- **The numbers** (Leap's C2, DAY9 results 1, ~/Leap/TABLE_350.md ~line 4448; bandwidth probe ~line 4479):
  - At 16 rows a check reads 7.38 GB of experts. The byte floor is 25.7-28.6 ms at the measured 258-287 GB/s.
  - The default's 3c kernel takes 50.5 ms (146 GB/s, falling to 115 GB/s at 32 rows).
  - NAX6 (matrix units, fixed split-K) takes 43.2 ms and holds ~171-175 GB/s at every width: 60-66 % of streaming.
- **Why it matters:**
  - Cloud task 9's route to 350 needs the expert kernel within ~15 % of its byte floor.
  - That gap is ~12-17 ms per 16-row check today, and ~7-10 ms with the gate's smaller union. Either way it's bigger than the whole HC read.
- **Where it stands:** Leap's C2 said a floor-level kernel is worth building (PASS), but no design for one exists yet. C1 (NAX6 inside OOO's two phases) runs tonight in BigMoE's ticket.

**What is known** (read the sources; don't trust this summary over them)
- **Kernels:**
  - 3c, the default's tier GLU.
  - NAX4 / NAX5 / NAX6, the matrix-unit tier kernels. NAX6 is NAX4 with a fixed split-K: kernel A uses 4 chunks per tile, kernel B 2, and the chunk accumulators are summed in a fixed order, so it is row-invariant by construction.
  - Code: ~/Leap/scripts/tierglu_nax6.py, k62j_nax6.py, k62i_nax5.py, k62o_nax_ooo.py, k60t_tier_bw.py, nax_probe*.py.
  - Notes: TABLE_350's K-62 family (~lines 3600-3715), C1 / C2 / B3 (~4419-4460), L1-25 (~605).
- **NAX6's speed vs 3c by width:**

  | rows | NAX6 vs 3c |
  |---|---|
  | 1 | 0.65x (a 1-pair group pays the full 16-row MMA set, ~7.4 us) |
  | 3 | 0.86-0.88x |
  | 16 | 1.46-1.71x |
  | 32 | 1.92-2.16x |

  - Quiet at 16 rows, NAX6 is only 1.17x faster (L6 0.96x, L20 1.22x, L40 1.42x). Layers with more distinct experts gain least.
- **The experts:**
> (a quote from the project owner, kept private)
  - Top-10 of 512. OOO computes resident experts while misses load.
  - The union grows x1.5-1.6 per doubling of rows. Bytes per added row fall from ~0.42 GB (1 -> 16 rows) to 0.25 GB (16 -> 32).
- **The machine:** a plain read kernel streams 258-287 GB/s (best 294).

**Use the same method as HC_READ.md:** the code, Leap's data, a numeric check, and an accounting that reconciles with the measurements.

**Deliver**
1. **Why 60-66 % at every width.** Account for one layer's expert pass at 16 rows:
   - bytes per expert and per tier
   - dequant ALU work per byte
   - MMA tile occupancy with 1-3 rows per expert
   - the gather and scatter of rows
   - the split-K sum
   - threadgroup counts vs the GPU's cores
   - launch and tail per layer

   Reconcile with Leap's numbers, and mark each part measured / arithmetic / judgement. Name the one timeline or counter Leap should run to settle it.
2. **At least three designs that reach >= 85 % of streaming at 16 rows**, and say what each does at 8 and 32 rows. For each give:
   - expected ms per check (upper bound, then halved)
   - exactness: bitwise to which reference (3c's, NAX6's, or a new row-invariant one that would need certifying)
   - the risk
   - what Leap builds
   - the cheapest check: a CPU numeric check where the arithmetic changes; otherwise a 10-minute shared-GPU screen
3. **The bet:** which design first, and what it does to the check and to 350, with and without the gate.

**Rules**
- CPU only: read code and data, run numeric checks. Writes only in ~/FableGod; no downloads. GPU timing is Leap's.
- Before writing, read in TABLE_350: the K-62 family, C1, C2, B3, L1-25, and the A1 closure ("A1 closed for today's 45 GB operating point"). Also read ~/TRIED/TRIED.md. Say why each design differs from what was tried.
- Exact mode stays exact; any reference change is Kamen's call.
- Read in targeted pieces (grep, line ranges). FINDINGS.md is 218 KB, so don't read it whole.
- Mark every number as measured / arithmetic / judgement.
- If 85 % is structurally out of reach for 3-bit tiers on this GPU, prove which limit sets the ceiling: ALU, memory parallelism or occupancy.
