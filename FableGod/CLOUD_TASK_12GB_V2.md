# CLOUD TASK 7: the 12 GB engine, version 2, with today's measured numbers. Goal 120-150 tok/s.

A closed thinking task that continues your 12 GB answers (tasks 3 and 4). Don't use the web; think and calculate. The owner's goal stays 120-150 tok/s on code at <= 12 GB for the full 125B.

**What was measured since your last answers** (code unless marked)
- **Your two-pass check, replayed on recorded routing** (36 windows, LRU caches; the replay's absolute misses run ~1.3x above the engine's, so ratios are what to use):
  - SSD bytes per 16-row check at 12 GB (51 cache slots per layer): exact 6.9 GB; two-pass with top-5 certified at p1 >= 0.9: 2.2 GB; top-2 certified at p1 >= 0.99: 1.25 GB. The 125B-sure oracle is 1.66 GB: the top-2 two-pass beats it because rows after the first miss stay cheap.
  - SSD caps at 13 GB/s: 12 GB 16 -> 51 / 90 tok/s (top-5 / top-2), or ~66 / ~117 after the 1.3x correction; 24 GB 90 -> 194 / 243; chat at 12 GB 6 -> 20 / 44.
  - A quality test of the two certificates (flips on certified rows, downstream drift) runs on the decision map tonight or tomorrow.
- **Cache policies:** protecting the prompt's experts raises misses 14-99 % at >= 160 slots (helps 2-11 % at 51 slots); pinning is noise; LRU is within 1.6-2.0x of Belady's optimum.
- **Your mid-layer self-certificate is dead.** A trained probe on the 125B's own state at layer 35 certifies 1.6 % of math rows and 0.8 % of IFEval rows at <= 0.1 % false certificates (the drafter's cf: 28.7 / 7.3 % on the same rows). An earlier test: reading the token after layer 40 matches the final one on 5.7 % of code positions.
- **The dense pass** (the local kernel session's census): ~2,700-2,850 kernel launches per 16-row check at 1.5-6 us each; no blocking per-layer host sync left. Realistic: ~37 ms today -> ~22-27 ms:
  - attention -5 ms, built, waiting on a quality re-test tonight
  - the GDN body -1-2 ms
  - glue -2-4 ms
  - the hyper-connection read up to -9 ms if a row-invariant fused read is found (11.4 ms vs a ~2.3 ms floor)
- **The drafter:** a calibrated plan reaches ~12.2-13.2 kept per 16 rows on code (13 only if a second drafter's errors overlap the first's at ~0.5 instead of 0.64); ~15 per 32.
- **The Neural Engine:** drafting 3 candidate blocks in one call takes 12.4 ms natively, with a 32K vocabulary head (kept -0.9 % on code); GPU slowdown ~x1.02. Whether it can draft during the check is decided tonight (an early-layer drafter).

**Deliver, in this order**
1. **The per-check budget at 12 GB** (<= 250 words) for the top-2 two-pass check:
   - pass 1 and pass 2 dense time at 22-27 ms each (or less: can pass 2 skip work for certified rows?)
   - SSD time from the measured bytes (with the 1.3x correction)
   - the draft, hidden or not
   - overlap between the passes and the SSD stream

   Give tok/s on code at kept 10.5 / 12 / 13 and on chat.
2. **The gap to 120 and to 150** (<= 300 words): which terms must move and by how much. Can pass 2 reuse pass 1's work (dense layers below the first uncertified row, KV for certified rows)? Is a cheaper pass 1 possible (resident experts only, or top-2 only on drafter-sure rows)?
3. **The design to bet on** (<= 250 words), with its arithmetic, exact vs quality-defined marked.
4. **Three decisive tests** (<= 200 words), each one night or less on this Mac, with PASS / KILL lines.

**Rules**
- The full 125B (all 512 experts), this MacBook only; quality-defined modes are allowed on this line, judged by task results.
- Mark every number as given / arithmetic / judgement. Treat your own gains as upper bounds (ours ran ~2x high).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,200 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
