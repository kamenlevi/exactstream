# MISSION 14: BREAK THE WALL. Design the engine that makes the 125B fast, from physics up.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~11:5x.)

> (a quote from the project owner, kept private)

**The wall, as FLOOR_125B.md left it**
- Every check reads the whole dense trunk once: attention, GDN, HC, router, shared expert and head, 3.3 GB.
- That cost is divided by the tokens the check keeps: 12.5-16 today. That is 0.26 GB per token before any expert; experts at the oracle add ~0.18.
- The fixed time per check caps the rest: dense ~36 ms at C 160 in today's GPU timeline, the drafter 6-10 ms, and the start-up.
- Both halves of "trunk bytes ÷ kept tokens" are design choices, not physics.

**Deliver**
1. **The physics of this Mac**, measured where we have it:
   - RAM streaming 258-284 GB/s (Leap's bandwidth probe, today)
   - SSD 12-14 GB/s
   - the GPU's ALU / NAX throughput on 3-4-bit dequant matmuls
   - launch / sync cost per kernel
   - what one token of the 125B fundamentally needs in bytes and FLOPs

   From that, the true ceilings for code and chat with no design assumptions.
2. **The barrier map.** Every wall between today (63.6 tok/s exact on code / math) and ~1000 tok/s on code: dense bytes per token, expert bytes, per-row compute, drafting time, acceptance (kept per check), SSD misses, memory. For each: its physical limit, and which design choice creates it.
3. **At least three design-level ideas that break a wall**, not cuts inside today's check. Starting points (invent better ones):
   - **Much longer checks (32-64 tokens) with a near-twin drafter.** The int4 twin agrees with the 125B on ~99 % of code tokens, so a 64-token check could keep ~47 and divide the trunk 3-4x more. What drafter could get there, at what drafting cost? What per-row cost does it imply? Leap measured ~1.7-1.9 ms per added row at 32 rows today.
   - **Skipping whole dense layers** in checks where every row is sure. Never tested: the decision map tested reusing a layer's previous output, which is different.
   - **Shrinking the trunk itself:** precision, structure, sharing across layers or rows, with quality gates. 8-bit HC passed today's map test.
   - Anything that changes what a "check" computes, or when the trunk must be read at all.

   For each idea:
   - the mechanism
   - its bytes, compute and time, from today's measured numbers
   - what must be true for it to work
   - the cheapest decisive test (one night or less) and its kill line
   - whether it is exact or quality-defined
4. **The one design to bet on** to break 0.3 GB per token on code. Or, if none can, a precise statement of which physical limit stops every design.

**Rules**
- The full 125B: all 512 experts, no pruning, no smaller model.
- Kamen's M5 Pro MacBook only; no external hardware.
- Exact mode stays exact; adopting quality-defined modes is Kamen's call.
- Check ~/TRIED/TRIED.md. Revisit a killed idea only when the new design removes the reason it died, and say which reason.
- Mark every number as measured / arithmetic / judgement.

**Inputs**
- ~/FableGod/FLOOR_125B.md
- ~/FableGod/ROUTE_350.md
- ~/FableGod/DAY9_2026-11-02.md (today's measured results on the nine conditions)
- ~/FableGod/decision_map/out/
- ~/Leap/TABLE_350.md (the C0 GPU timeline, the bandwidth probe, the NAX32 per-row cost)
- ~/MoonShots/data/band45_capture.txt
- ~/TrainLab/REPORT.md (drafter ceilings, the D-23d chat switch)
- ~/TRIED/TRIED.md
