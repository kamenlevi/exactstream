# Cloud Fable 5.1, task 11 (CLOUD_TASK_12GB_V4.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~20:3x
(The full ANSWER.md stays in the cloud session: 1,033 words, 3 of 8 Python runs, no web.)

- **Corrections accepted:**
  - At today's kept 10.5, v3's cycle gives ~106-112 tok/s, not 121.
  - The forecast road is closed by our table. A 1-layer look-ahead at 0.69 precision adds 31 % wasted bytes, which at 12 GB makes pass 1 SSD-bound and competes with demand reads (the engine's measured -13 %). The 2-3 layer leads are a plain loss.
  - So v3's honest ~99 ms is the no-forecast case, and the 70 ms bound can't be reached by prediction.
- **The other road: residency.**
  - Pass 1's stall can't be removed by knowing routing earlier, because routing at layer L depends on layer L-1's full output. The only non-predictive lever is residency.
  - Pass 1 only ever needs ranks 0-1, so cache ranks 0-1 only and always stream ranks 2-9.
  - Cloud's estimate: pass-1 hit rate ~65 % -> ~80 %, and pass-1 bytes 0.45 -> ~0.26 GB per check. The stall shrinks with the bytes. It costs nothing and runs on recorded routing.
- **The plan for 150:** theta 0.8 + dense 22 in both passes + the rank-aware cache + the drafter ceiling at 12 GB.
  - Halved cycle ~81 ms: 148 tok/s at kept 12, 157 at 12.7.
  - Requires: dense <= 22 ms per pass, pass-1 bytes <= 0.26 GB, kept >= 12.2, pass 2 in <= 55 % of blocks.
  - Deviation "stays at theta 0.8's 0.45-0.9 %". HC8 stays out (its lean buys only +4 slots). No forecast anywhere.
- **Ceilings if a piece fails:**
  - without the drafter ceiling at 12 GB: ~130
  - with dense stuck at 27: ~141 at kept 12.7
  - kept 10.5 can never reach 150: pass-1 compute + pass 2 + the draft already sum to 69 ms with zero SSD, and 150 needs 70
- **Pushback:** precision on miss is not a 12 GB lever in the two-pass design. Ranks >= 6 only load in pass 2, whose reads already hide under its compute. Spend the quality budget on theta instead.

## Orchestrator's check (20:4x)
- **Consistent with our data:**
  - the forecast's -13 % is PF01's 22.7 vs 26.2 tok/s
  - ~106-112 at kept 10.5 matches our correction
- **New and exact:** TRIED has no residency-by-rank policy. The nearest items are B12LL, FRUGAL5/10, LATE RANK 1 and D1j, which all change quality. Routed to Breakthrough as a <= 10 min CPU replay (PASS pass-1 bytes <= 0.30 GB with total bytes not up > 10 %; KILL > 0.40).
- **Wrong on deviation:** it missed the measured floor (~1.16 % at any theta; theta 0.8 adds ~0.4-0.5). The total at theta 0.8 is ~1.6 %, and EvalSuite's same-scores test judges.
- **Everything else in the plan is in measurement:** dense <= 22 per pass (Leap's D1 + the known levers), and the drafter ceiling at 12 GB.
- **Next cloud task:** CLOUD_TASK_SSD45.md (task 12), the same stall mechanics applied to the 45 GB line's SSD.
