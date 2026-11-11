# CLOUD TASK 13: 12 GB, v5. The pass-1 bytes wall.

A closed thinking task that continues your 12 GB answers (tasks 3, 4, 7, 10 and 11). Don't use the web; think and calculate. Everything you need is below.

**The goal stays the same:** 120-150 tok/s on code at <= 12 GB, for the full 125B (all 512 experts), on this MacBook only.

**Measured since your v4** (code unless marked)
- **The rank-aware cache is KILL.**
  - At C 51 and theta 0.8: pass-1 hit rate 64.1 -> 66.1 %, pass-1 bytes 0.708 -> 0.669 GB per check, total +1.6 %.
  - Belady's MIN (perfect foresight) on pass 1 gets 0.378 GB (80.8 % hit). It reaches 0.259 only under the replay's optimistic assumption that rows after a miss pre-load the next check.
  - Realizable LRU: 0.67-0.71 GB (0.45-0.48 in the optimistic variant). With the engine's 1.3x calibration that is ~0.51-0.54 realizable and ~0.29 at the ceiling.
  - At C 160, rank-aware admission cuts pass 1 from 0.224 to 0.154 GB, but pass 2 rises from 0.127 to 0.200, so the total is flat.
  - The replay counts 2.2 MB per miss: ranks 0-1 are 4-bit, ranks 2-9 are 3-bit.
- **Theta 0.8 passes on code** (GPU): certified rows flip 0.27 %, and 1.63 % of all code rows change. There is a ~1.16 % floor at any theta, because certified rows keep top-2 hidden states. EvalSuite's same-scores test judges quality.
- **Theta 0.8 fails on chat:**
  - certified flips 2.55 % (IFEval) and 4.63 % (multilingual)
  - all rows 8.51 / 7.04 %, against 4.50 / 2.70 % at 0.99
  - chat needs pass 2 in 79-91 % of blocks at 0.8 anyway
- **The domain-adaptive certificate** (CPU; GPU tonight): certify at p1 >= 0.5 where the drafter is sure (cf >= 0.99) and agrees with pass 1, else at p1 >= 0.95.
  - code: pass 2 in 59 % of blocks, 0 certified flips
  - chat: 0.8-1.3 % certified flips, pass 2 in 91-97 % of blocks
- **Honest 12 GB at theta 0.8:** pass-1 reads are 0.54-0.71 GB per check (not your 0.45), so the cycle is 106-119 ms.

  | kept per check | tok/s |
  |---|---|
  | 10.5 (today) | 88-99 |
  | 12 | 101-113 |
  | 13 | 109-123 |

  Even Belady leaves >= 29 ms of pass-1 reads per check.
- **The one measured lever left on pass-1 bytes:** smaller records for ranks 0-1 (a 3-bit arm, measured Tuesday).
- **The dense pass:** an exact -3.9 ms cut in the hyper-connection read is being built; the GDN layers are under analysis. Realistic this week: 18-23 ms exact per pass.

**The question.** Pass-1 SSD bytes are the wall: ~0.5-0.7 GB per check realizable, and >= 0.29-0.38 even with perfect caching. Where does 120 and 150 really sit, and what gets past the wall?

**Deliver, in this order**
1. **The wall, restated** (<= 150 words): the pass-1 bytes per check that 120 and 150 need, at kept 12 / 13 with the dense pass at 22 / 17 ms; the gap to the measured 0.51-0.71.
2. **Every way to cut pass-1 bytes or hide them** (<= 450 words). Price each (upper bound, then halved) and mark it exact or quality-defined. Use Belady's 0.378 GB as the caching ceiling. Candidates:
   - smaller pass-1 records: 3-bit or 2-bit ranks 0-1, a lossless pack
   - fewer pass-1 experts on layers where the router is peaked (top-1)
   - one read shared across the rows of a check (the union)
   - moving RAM between the trunk and the expert slots inside the 12 GB
   - reading the next check's pass-1 experts during this check's pass 2
   - anything else
3. **Chat at 12 GB** (<= 150 words): with pass 2 in 91-97 % of chat blocks, what chat tok/s is realistic, and what would change it?
4. **The plan** (<= 250 words): the smallest combination that reaches 120, and the one that reaches 150 (or the honest ceiling and why), with each piece's required number.
5. **Three measurements that decide it soonest** (<= 150 words). Small tests run by day on a shared GPU; big ones run weekdays 00:30-15:00.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware, <= 12 GB of RAM in use.
- Mark every number as given / arithmetic / judgement. Treat your own gains as upper bounds (ours ran ~2x high).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all five sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,300 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
