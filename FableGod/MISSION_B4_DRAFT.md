# MISSION B4 (Fable B): THE EXPERT KERNEL IS LATENCY-BOUND. Find the concurrency.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~23:0x. **WITHDRAWN 23:00, not sent:** Leap's N7-11 and its ~12 measured arms already answer it. The kernel's inner structure is at its ceiling; see the note at the top of EXPERT_KERNEL.md.)

**Why.** Tonight's measurements overturned the expert kernel's diagnosis:
- Mission 27 (EXPERT_KERNEL.md) said NAX6 is bound by the matrix unit's MAC rate on padded 16-row tiles.
- Leap then tested that. The matrix unit runs int8 at 1.55-1.73x the bf16 rate, but the int8 kernel (NAX-C) gained nothing: -0.3 % at S 16.
- Bitwise rewrites lost too: pipelining +11 % slower, loader/op split +35 %, dispatch fusion +1.2 %. Registers matter only at the edge (+64 per thread: +41 %).
- Leap's reading: each simdgroup waits on its own two dependent ops' latency and on its codes, with too little independent work per core.
- The gap is large: NAX6 takes 43.2 ms per S 16 check against a byte floor of ~26 ms at 284 GB/s. Experts are the biggest compute piece of a check on both tracks.

**Read first:**
- ~/Leap/TABLE_350.md: K-62 rows, N7-1 .. N7-10, NAX-C, M27-C, and the split-K chunk probe Leap is running tonight
- ~/FableGod/EXPERT_KERNEL.md
- Leap's kernels: scripts/tierglu_nax6.py and n7_*.py
- the 36 routing captures (rows per group: 2.33 / 3.12 at 16 / 32 rows on code; 57 % of groups hold one pair)

**Deliver**
1. **A latency model of one NAX6 call that fits Leap's measured points:**
   - per simdgroup: dependent op latency, code-load latency, the barrier chain
   - per core: resident simdgroups and independent groups in flight
   - Show why int8's throughput, pipelining and fusion didn't move it.
2. **What the M5 GPU allows:** simdgroups per core, threadgroup memory, registers, and how much concurrent independent work 16-row checks offer (groups per layer, tiers, K chunks).
3. **Designs that raise independent work per core without registers or barriers:**
   - several groups interleaved per simdgroup
   - persistent scheduling across groups
   - kernel A and B work mixed in one wave
   - overlapping the expert pass with the next layer's dense work
   - anything else the model suggests
   For each: EXACT (bitwise to NAX6, or row-invariant under a new reference) or FAST, and ms per S 16 check, halved.
4. **The 10-minute GPU screen Leap should run first,** with a pre-registered line.

**Rules**
- CPU only. Writes only in ~/FableGod/fableb/. No downloads. GPU timing is Leap's.
- Both tracks are first-class. Halve gains. Mark every number [K] / [P] / [G].
- Price work on the union per 16-row block, never per row.
- Check ~/TRIED/TRIED.md and say why each design differs from what was tried.
- Within ~40 minutes, write a complete first draft to EXPERT_LATENCY.md, then refine it.
