# CLOUD TASK 15: the road from ~110 to 350, piece by piece

A closed thinking task that ties your tasks 9, 12 and 14 together with today's local results. Don't use the web; think and calculate. Everything you need is below or in your earlier answers.

**Where we stand** (your task 14 + this week's measured work)
- The exact combined engine this week (our guess): ~78 tok/s by day, ~90 at night, on code.
- Your fast mode (gate 0.99 + comment loophole + HC8): ~96 by day, ~110 at night. With R1: ~100 / ~114.

**The four engine pieces 350 needs, and their status today**
1. **The dense pass, target <= 17 ms.** Today it is ~34 ms of GPU: C0's synced medians minus the sync. The "32" figure subtracted all 8.5 ms of CPU build, including the experts' share.
   - **The hyper-connection read:** an exact two-kernel plan (D1 + D5), predicted -3.9 ms. **But Leap's first half (kernel B) measured only -0.74 ms where -2.6 was predicted**, so the whole of D1 is now ~-1.3 to -1.8. Our estimates run ~2x high.
   - **The GDN layers** (measured split, Mon 21:2x): 12.4 ms = 2.1 ms of CPU graph build + 10.3 ms of GPU.
     - The GPU time: three GEMMs 6.1, the recurrence 2.5 (1.4 of it 16 serial row steps), small kernels and chain ~1.7.
     - Three bitwise levers: about -2.3 ms expected together.
     - A one-dispatch GDN body saved only 0.1-0.6 ms, and that was Python build.
   - **Attention:** -5 ms, if its quality re-test passes. It is a reference change, Kamen's call.
   - **The local analysis's total:** every known exact lever gives ~24-26 ms expected, ~20-21.5 at every bound. **<= 17 is not reached exactly.** The rest is ~1,000 kernel boundaries and dependent chains on top of ~14 ms of floors. It needs a reference change (attention, 8-bit HC weights), 32-row checks, or a new idea about boundaries (being analysed).
2. **The expert kernel, target within ~15 % of its byte floor** (floor 25.7 ms at 16 rows; NAX6 43.2 ms).
   - Local finding: NAX6 is compute-bound on the matrix unit. It runs padded 16-row tiles with only ~2.1 real rows per (expert, tier) group, so it streams at 60-66 % at every width.
   - **The probe ran (Mon 21:21):** the matrix unit peaks at ~14.0 T MAC/s on NAX6's op, 1.34x the earlier figure. That is under the 1.6x line, so:
     - a fused kernel (NAX7) alone reaches ~80-83 % of streaming
     - NAX7 + a hybrid (1-2-row groups on the SIMD ALUs) is needed for >= 85 %
   - **Also being tested:** a faster bitwise copy of 3c, today's exact expert kernel, so no reference change is needed. If 3c is latency-bound it gains <= ~1.3x (~-10 ms per check); if it is ALU-bound, the idea is dropped.
   - A tier merge gets 85-89 %, but under a new reference.
   - NAX6 and its successors are their own exact reference: they need certifying and Kamen's OK before they become the default (3c stays the default until then).
3. **The SSD:** your task 12. It hides only at night (C 320 + gate + look-ahead).
   - Rank priority is now KILL at 45 GB too (+2.7-5.2 %).
   - New lead: the 4-bit sidecar parts are ~60-70 % of reads by count. Reading base + sidecar in one burst gained +7-8 % at 24 GB; it is measured at 45 GB tonight.
4. **The draft, hidden:** no design yet.
   - The draft is 6.3 ms serial with the 4-bit drafter, and it needs the previous check's taps.
   - Candidates: ANE (12.4 ms per batched call, GPU x1.02), an early-tap drafter (verdict Tuesday), the CPU (being screened).

**Kept per check:** ~10.5 on code today; 12-13 at the drafter ceiling. On chat 2.0-2.5; your 8-row blocks give +30 %, exact (in Tuesday's combined build). On trees: your point stands that a leaf at the first miss gains nothing. A local session is pricing continuing branches in the GDN layers; count no tree gain unless you say what it would need.

**Deliver, in this order**
1. **The ladder** (<= 300 words + one table): from ~90 exact / ~110 fast (night) to 350. One row per piece, in the order they can land. For each row give: ms saved per check (halved, given our ~2x optimism), tok/s after it, exact or quality-defined, and the measurement that confirms it.
2. **The critical piece** (<= 250 words): which single piece, if it fails, makes 350 unreachable this year on this Mac? What is the fallback ceiling without it?
3. **The draft, hidden** (<= 250 words): your best design to take the 6.3 ms draft off the critical path, given the dependency on the previous check's taps. Say what the early-tap drafter must deliver Tuesday for the design to work.
4. **Three decisions for Kamen** (<= 150 words): the yes / no calls (a new exact reference for NAX, the gate, R1, ...) in the order he needs to make them, each with the evidence that should arrive first.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware.
- Mark every number as given / arithmetic / judgement. Halve your own gains.

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,200 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
