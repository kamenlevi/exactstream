# MISSION 29: THE DRAFT, HIDDEN. Take the drafter off the critical path.

> (a quote from the project owner, kept private)

**Why.** Cloud's ladder (task 15, ~/FableGod/CLOUD_ANSWER_ROAD350.md) needs the draft hidden. Today the draft runs serially before every check:
- DFlash2 4-bit (D7b): 6.3 ms on the GPU, which goes into this week's exact default
- bf16: ~10.4 ms
- at the 350 cycle (~36-56 ms) that's 11-17 %

It can't start until the previous check ends, because DFlash2 conditions on that check's taps (layers 5 / 15 / 25 / 35 / 45).

**Measured since the first draft of this mission**
- **Cloud's design:** an early-tap drafter reads taps from layer <= 24. The ANE drafts 3 candidate blocks from 3 anchors while the check finishes its last 24 layers, and the matching candidate goes straight in. Cloud: exposed draft 6.3 -> ~3 ms.
- **Anchor coverage** (MTP Head's ANCHORCOV, ~/TrainLab/data/anchorcov.txt):
  - Cloud's anchors (all accepted + the 2 lowest-cf rows) cover code 0.595 / chat 0.210 at N=3.
  - Ranking rows by the drafter's own first-miss probability (prod cf_i x (1 - cf_j)) covers code 0.64 / 0.78 / 0.86 and chat 0.60 / 0.80 / 0.89 at N = 2 / 3 / 4.
- **The ANE** (ANE-2, Leap): 12.42 ms per batched 3-candidate call, all ops on the Neural Engine; the GPU slows ~x1.02 beside it.
- **The CPU as drafter is KILL** (MoonShots M-19): 62 ms per 3-candidate draft; the CPU is ~1 TFLOPS.
- **Your own TREE_WALL:** after a hard token the next one is usually hard too. The continuation is ~1.1 tokens on chat and ~3.2 on code / math.
- **The native MTP head reads the trunk well at depth 1:** code mid rows top-1 0.86, top-3 0.98 (KWALL (b)).
- **EARLYTAP** (MTP Head) gives its verdict Tuesday by 15:00. Cloud's line, added as a report line: kept >= 0.97x with early taps.

**Deliver**
1. **The dependency, exactly:** what check N+1's draft needs from check N (taps, the accepted prefix, KV, the MTP head's state), and when each becomes available inside check N's timeline at the floor engine. Use the code, Leap's C0 timeline and BigMoE's loop.
2. **At least three designs with critical-path draft cost <= ~1 ms.** For each give:
   - exposed ms (upper bound, then halved)
   - kept vs today's chain
   - the hardware and its bandwidth cost to the GPU
   - what isn't built yet
   - the cheapest test (a CPU replay on stored walks where possible)

   Include cloud's ANE 3-anchor design with the first-miss-probability anchors, and one where the MTP head's depth-1 token comes free from the check's own last layer.
3. **What EARLYTAP must deliver** for your best design: the kept-per-check line that makes it win, so Tuesday's verdict reads straight into the plan.
4. **Both tracks:** the draft itself is exact by construction. Say whether any FAST-track lever (relaxed acceptance, the gate) changes the best design.
5. **The bet,** and the floor-engine cycle and tok/s with it, beside the dense, expert and SSD pieces.

**Rules**
- CPU only; writes only in ~/FableGod; no downloads; GPU timing is Leap's and MTP Head's.
- Read TRIED's drafter / pipelining entries and say why your designs differ.
- Halve your gains (D1-B: -0.74 where -2.6 was predicted). Mark every number [K] / [P] / [G]. Read in targeted pieces.
