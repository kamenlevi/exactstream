# MISSION 28: THE DRAFT, HIDDEN. Take the drafter off the critical path.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~21:2x.)

**Why.** Cloud task 9 says 350 needs four engine pieces. Two now have a plan:
- the dense pass: your HC_READ.md, and Fable B's GDN mission running now
- the expert kernel: your EXPERT_KERNEL.md (the probe decides between NAX7 alone and NAX7 + hybrid)

A third, the SSD, hides only at night (cloud task 12). The fourth has no design yet: **the draft**.
- **Today the draft runs serially before every check:**
  - DFlash2 bf16 on the GPU ~10.4 ms; the 4-bit form (D7b) 6.3 ms, which goes into this week's exact default
  - at the 350 cycle (~36 ms per 16-row check) even 6.3 ms is 17 %
- **The draft depends on the previous check:** DFlash2 conditions on the target's taps (layers 5 / 15 / 25 / 35 / 45) from the last check, so check N+1's draft can't start until check N ends.
- **What's being measured:**
  - EARLYTAP, a drafter that reads only early taps so it can start mid-check: Tuesday by 15:00
  - the ANE (ANE-2: 12.42 ms per batched 3-candidate call, GPU slowed x1.02): already measured
  - the CPU as drafter (MoonShots' screen): tonight or Tuesday
  - The native MTP head reads the trunk well at depth 1: on code mid rows top-1 0.86, top-3 0.982 (MTP Head's KWALL (b), data/kwall_mtp.txt).

**Deliver**
1. **The dependency, exactly:** what check N+1's draft needs from check N (taps, the accepted prefix, the KV), and when each becomes available inside check N's timeline (per layer, at the 16-row floor engine).
   - Use the code: DFlash2 / the MTP chain (~/TrainLab, ~/BigMoE), and BigMoE's check loop.
   - Use the data: Leap's C0 timeline (~/Leap/TABLE_350.md "DAY9 results 2" ~line 4479), ANE-2's numbers, D7b.
2. **At least three designs that make the draft's critical-path cost <= ~1 ms at the floor engine.** Starting points, not limits:
   - draft N+1 on the ANE or CPU from early taps while check N's later layers run (EARLYTAP's premise)
   - speculate on check N's outcome: draft from the most likely accepted prefix, keep the draft only if the guess holds
   - the MTP head at depth 1-2 as a free first token from the verify's own last layer
   - a mixed chain: MTP head first, DFlash2 deeper

   For each give:
   - critical-path ms (upper bound, then halved)
   - kept per check vs today's chain (the trade: an early-tap or speculative draft may keep fewer tokens)
   - the hardware it uses, and its bandwidth cost to the GPU
   - what it needs that isn't built
   - the cheapest test: CPU replay on stored walks and taps where possible
3. **What EARLYTAP must deliver** for the best design: the kept-per-check line that makes it win, so Tuesday's verdict reads straight into the plan.
4. **The bet,** and what the floor-engine cycle and tok/s become with it, beside the dense, expert and SSD pieces.

**Rules**
- CPU only; writes only in ~/FableGod; no downloads. GPU timing is Leap's and MTP Head's.
- Before writing, read the drafter entries in ~/TRIED/TRIED.md (SD-ARtail, D-24, TREES, M-16b and the pipelined / async speculative-decoding prior art) and your own RELAX_350, NEXT_DRAFTER, DEPTH_WALL and KNOWLEDGE_WALL answers. Say why each design differs from what was tried.
- Every design is exact by construction (the check decides the tokens). Say so, or say why not.
- Read in targeted pieces. Mark every number as measured / arithmetic / judgement.

**Added 21:39 (MTP Head ANCHORCOV, ~/TrainLab/data/anchorcov.txt, 7367010):**
- Cloud's anchors (all accepted + the 2 lowest-cf rows) cover code 0.595 / chat 0.210 at N=3: KILL for chat.
- Ranking rows by the drafter's own first-miss probability (prod cf_i x (1 - cf_j)) covers code 0.641 / 0.782 / 0.857 and chat ~0.60 / 0.80 / 0.89 at N = 2 / 3 / 4 (math 0.672 at N=3).
- Use that anchor rule.
