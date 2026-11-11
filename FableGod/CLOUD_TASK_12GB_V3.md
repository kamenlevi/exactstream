# CLOUD TASK 10: 12 GB, v3. The threshold sweep is measured: which check reaches 120 tok/s?

A closed thinking task that continues your 12 GB answers (tasks 3, 4 and 7). Don't use the web; think and calculate. Everything you need is below.

**The goal stays the same.** 120-150 tok/s on code at <= 12 GB for the full 125B (all 512 experts), on this MacBook only. The owner's direction is to work toward perfect prediction: knowing before the check which rows are safe with only their top-2 experts, and which experts each row will need.

**Your v2 in brief**
- The design is a two-pass check:
  - pass 1 runs the top-2 experts on every row
  - a row is certified when pass 1's own p1 is >= theta
  - pass 2 runs the exact experts for the rest
- The check is an SSD + compute problem.
- Budget: serial 147-157 ms, full-overlap 80-84 ms, midpoint 113-120 ms. That gives 100-106 tok/s at kept 12.
- Levers:
  - issue pass 2's reads at once from pass 1's routing
  - the HC read
  - the ANE draft in pass 2's shadow
- Pass 2's frequency is the lever.

**Measured since then** (decision map: 36 windows, 4-bit experts; code = 3,366 rows in 391 blocks of 16)

| certificate (pass 1's own p1 under the cut) | rows certified | blocks needing pass 2 | flips on certified rows (pass 1 alone) |
|---|---|---|---|
| top-2, theta 0.99 | 73.4 % | 77 % | 0 of 2,472 |
| top-2, theta 0.95 | 82.3 % | 63 % | 0 of 2,771 |
| top-2, theta 0.9 | 85.4 % | 60 % | 4 of 2,873 (0.14 %) |
| top-2, theta 0.8 | 88.4 % | 55 % | 10 of 2,976 (0.34 %) |
| top-2, theta 0.7 | 91.6 % | 47 % | 23 of 3,082 (0.75 %) |
| top-5, theta 0.9 | 89.8 % | 51 % | 0 of 3,024 |

- **Your guesses vs the truth:** you guessed 97 / 71 / 37 % of blocks at theta 0.99 / 0.9 / 0.7. The truth is flatter: 77 / 60 / 47.
- **Chat and IFEval** need pass 2 in 79-99 % of blocks at every theta, so code is where 12 GB can be fast.
- **Whole-sequence deviation** (flips carried through the context) is being measured now. Use certified-row flips x 1.5 as a placeholder, and say how your answer changes if the factor is x 3.
- **SSD bytes per check** (code, top-2 two-pass, C 51 expert slots per layer):
  - ~1.16 GB on the engine, not 0.96. The replay's after-miss rows pre-load the next check: +7 % misses at C 256, +21 % at C 51.
  - Top-5 pass 1: 2.2 GB in the replay (the replay runs ~1.3x pessimistic).
- **Resident-only pass 1 is dead**, as you said: only 0.2 % of rows have both top-2 experts resident at all 48 layers.
- **Pipelining:** the bonus row saves 4-6 % of dense passes on code. A deferred anchor loses on code.
- **Pass 2's dense cost:** it re-reads 3.44 GB of dense weights whatever its row count (your v2).
- **The dense pass** (45 GB engine, warm, measured):
  - 36-38 ms per pass unsynced
  - GPU constant ~32 ms: GDN 12.4, HC reads 11.4, attention 7.3, router / shared / tail 5.1, head 2.7
  - it costs the same at 1 row as at 16 (~0 ms per added row)
  - the HC reads are a fixed chain, not a GEMM-efficiency problem: their down GEMM is already at its byte floor (2.29 ms per pass)
- **Experts:** 2.9-3.2 ms per row at top-10 (45 GB engine). Top-2 expert compute has not been measured; assume ~1/5 of that, and say so.
- **The draft:** the ANE takes 12.4 ms per batched 3-candidate call, and the GPU slows by ~x1.02. The early-tap drafter's verdict comes Tuesday.
- **Kept per 16-row check on code:**
  - ~10.5 today
  - 12-13 at the drafter ceiling (copy override + two drafters)
  - +6 % from the comment loophole (quality-defined)
- **Bandwidth:** RAM 258-287 GB/s; SSD 11.3-13 GB/s.

**Deliver, in this order**
1. **The budget per theta** (<= 250 words + one table):
   - Cover top-2 at theta 0.99 / 0.9 / 0.8 / 0.7, and top-5 at 0.9.
   - For each give:
     - SSD ms
     - compute ms (pass 1 + pass 2 x its frequency + experts + draft)
     - serial, full-overlap bound and honest midpoint
     - tok/s at kept 12 and 13
     - the deviation
2. **The best check under a <= 1 % code-token deviation budget** (<= 300 words):
   - Which certificate and theta, or a mix (for example top-5 certification with top-2 reads, or a theta that adapts to the block).
   - The overlap schedule, as a timeline of one check with ms per segment. Show where the stall is.
3. **Perfect prediction, priced** (<= 300 words). Give two numbers for each kind: the bound with a perfect oracle, and with today's certifier.
   - (a) Which rows certify (blocks needing pass 2 drop to 0 for the safe ones).
   - (b) Which experts pass 2 will need (issued before pass 1 ends).

   Which of the two is worth more toward 120 and 150? What is the most promising realizable signal for it: the drafter's confidence, pass 1's margin, the previous check's routing, or the draft tokens routed through an early layer?
4. **The verdict on 120 and 150** (<= 150 words), then **three measurements that decide it soonest** (<= 150 words). Small tests run by day on a shared GPU; big ones run weekdays 00:30-15:00.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware, <= 12 GB of RAM in use.
- Mark every number as given / arithmetic / judgement. Treat your own gains as upper bounds (ours ran ~2x high).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,200 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
