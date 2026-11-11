# CLOUD TASK 11: 12 GB, v4. Your 150 lever was already measured: find another road to 150.

A closed thinking task that continues your 12 GB answers (tasks 3, 4, 7 and 10). Don't use the web; think and calculate. Everything you need is below or in task 10.

**Your v3 in brief**
- Top-2 at theta 0.8: honest 121 / 131 tok/s at kept 12 / 13 (~99 ms per check), deviation 0.45 % (x1.5) / 0.9 % (x3).
- The stall is pass 1's per-layer chain (routing -> read -> compute) and the first layers of each pass.
- A perfect row oracle: +18 %. A perfect pass-1 expert forecast: +40 % (the ~70 ms bound).
- 150 rests on a pass-1 forecast at >= 75 % hit with <= 25 % wasted bytes, from the draft tokens routed through the early target layers.

**Two corrections from our side**
- **Kept 12-13 is not today's number.** Today's kept is ~10.5 per 16-row check on code; 12-13 is the drafter ceiling (copy override + two drafters), not yet built at 12 GB. At ~10.5 your 99 ms cycle gives ~106.
- **Forecasting this 125B's experts has been measured several times.** All results are on the 125B, from different lines and dates.

| forecast | lead | result |
|---|---|---|
| the next layer's router on this layer's MoE input (look-ahead) | 1 layer | top-2 precision 0.527 |
| look-ahead + ridge / MLP / routing history | 1 layer | top-2 precision 0.683-0.689 at best (killed as too low) |
| trained pre-router from earlier layers | earlier layers | held-out top-2 precision 0.683 at best (killed) |
| look-ahead with a learned correction, 45 GB replay | 1 layer | recall 0.693 (its line was 0.75); demand misses -44 % at 224 slots per layer, wasted 98 < saved 187 reads (upper bound); being built |
| the router at longer leads | 2 / 3 layers | recall 27-40 % / 15-26 %, ~40 % precision |
| oracle 3 layers ahead (a bound, 24 GB line) | 3 layers | +33-48 % on code / prose |
| from the MTP head's state, the next pass | 1 pass | recall@10 0.09-0.24 (the previous position's own top-10: 0.28-0.41) |
| look-ahead prefetch of ranks 0-1 at 12 GB | 1 layer | slower: code 22.7 vs 26.2 tok/s (wrong prefetches compete for the SSD) |

So the forecast your 150 needs is far beyond every measurement:
- It needs >= 75 % hit at <= 25 % waste, for all 48 layers, issued at t = 0.
- Measured: precision is ~0.69 at a 1-layer lead, and recall falls to 15-26 % by 3 layers.
- A forecast from layer ~5 for layers up to 47 should be weaker still (judgement, not measured).

**Other inputs**
- Everything in task 10 still holds: the sweep, bytes, the dense pass, experts, the ANE, bandwidth.
- **The dense pass** costs the same at 1 row as at 16 (~32 ms GPU constant, 36-38 ms unsynced).
  - Realistic with known levers: 22-27 ms. The 350 line's target is <= 17 ms.
  - Here it counts in both passes.
  - Its largest piece, the hyper-connection read (~11 ms), is a fixed chain of 5 small kernels x 96 per pass, and is being redesigned now.
- **Precision on miss** (quality-defined, 24 GB line): read a cold expert at rank >= 6 as its top 2 code planes only. That is -12.4 % bytes per token, predicted inside the quality budget.
- **The whole-sequence deviation is now measured (GPU, code), and your "certified flips x factor" model doesn't fit it.**

  | theta | certified flips | all code rows changed |
  |---|---|---|
  | 0.99 | 0 of 2,472 | 1.16 % |
  | 0.9 | 3 of 2,873 | 1.49 % |
  | 0.7 | 18 of 3,082 | 1.93 % |

  - There is a floor of ~1.16 % that doesn't depend on theta. Certified rows keep top-2 hidden states, and later rows see them through the context. (The 45 GB gate's run showed the same profile: 1.34 %.)
  - Theta adds +0.33 points at 0.9 and +0.77 at 0.7. Certified rows flip at x0.75-0.78 of the pass-1-alone rate.
  - No theta meets a 1 % budget. The judge is our same-scores quality test (task results), as for the 45 GB gate. Report each design's total deviation and its addition over the floor.
- **Adaptive theta is measured (CPU, code), and it's a small lever:**
  - Your arm (0.7 where the drafter is sure and agrees, 0.9 elsewhere): 58 % of blocks need pass 2, against 60 % at plain 0.9, with the same 4 flips.
  - The best arm found (0.8 everywhere, plus p1 >= 0.5 where a sure drafter agrees): 52 % against 55 % at plain 0.8, with the same 10 flips.
  - Only ~18 % of rows with p1 0.7-0.9 have a sure drafter.

**Deliver, in this order**
1. **Re-price v3's honest cycle with the measured forecast** (<= 200 words):
   - the 1-layer look-ahead at precision ~0.69, with its waste competing for the SSD
   - the 2-3 layer leads at their measured recall

   What overlap is realistic, and what tok/s at theta 0.8 for kept 10.5 / 12 / 13?
2. **Every other road to 150** (<= 400 words). Price each one (upper bound, then halved):
   - the dense pass at 27 / 22 / 17 ms, in both passes
   - adaptive theta, toward the +18 % row oracle (measured above: worth ~3 points of pass-2 blocks with today's signals; is there a better row signal?)
   - SSD bytes: precision on miss; the split of 12 GB between expert slots and everything else; anything else that cuts bytes per check
   - kept 13-15 (drafter ceiling + the comment loophole's ~+6 %)
   - restructuring the check so pass 1's routing is known earlier without a forecast, or say why that can't work
3. **The plan for 150** (<= 250 words): the smallest combination that reaches 150, with each piece's required number. Keep the addition over the ~1.16 % floor to <= ~0.5 points (our judgement until the quality test speaks). If 150 is out of reach, give the realistic ceiling and why.
4. **Three measurements that decide it soonest** (<= 150 words). Small tests run by day on a shared GPU; big ones run weekdays 00:30-15:00.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware, <= 12 GB of RAM in use.
- Mark every number as given / arithmetic / judgement. Treat your own gains as upper bounds (ours ran ~2x high).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,200 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
