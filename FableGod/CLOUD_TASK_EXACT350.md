# CLOUD TASK 9: is exact 350 possible at all, with the drafter at its ceiling?

A closed thinking task that ties your last answers together. Don't use the web; think and calculate. Everything you need is below.

**The question.** You and our local analysis agree the drafter tops out near 0.95-0.955 per token (after the 2x haircut). On HumanEval+-like code that is ~12.2-12.5 kept per 16-row check, or ~15-17 per 32-row check with the copy override; on copy-poor code (MBPP-like) ~6.5-8 per 32. Given that ceiling, can this MacBook reach 350 tok/s on code exactly (every token equal to the 125B's greedy output)? If not, what is the smallest quality-tested addition that closes the gap?

**Measured today** (45 GB line, code unless marked)
- **The default:** 63.6 tok/s exact at C 256 by day. A 16-row check takes ~166 ms:
  - draft ~12 ms (6.3 with the 4-bit drafter)
  - dense ~36 ms
  - experts ~3.0 ms per row (~53 at 16 rows)
  - SSD exposed ~39-65 ms by day
  - it keeps ~10.5 tokens
- **The dense pass:**
  - byte floor 12.7 ms; practical floor 15-17 ms
  - ~2,800 kernel launches per check at 1.5-6 us each; no per-layer host sync left
  - realistic levers: attention -5 ms (built, waiting on a quality re-test), GDN body -0.1-0.6, glue -2-4, and the hyper-connection read up to -9 if a row-invariant fused read is found (11.4 ms vs a ~2.3 floor)
  - the real lever is 16-row 4-bit GEMM efficiency: today 2.2-4x the byte floor (GDN projections ~9 ms, HC reads 11.4 ms)
  - realistic outcome 22-27 ms
- **The experts:** the NAX6 kernel runs 1.26-1.75x its byte floor (7.38 GB of experts per 16-row check; 43.2 ms vs a 24.6-34.3 ms floor at 300-215 GB/s). At 32 rows each added row costs ~2.0 ms (experts 1.41 + dense 0.26 + GDN / attention ~0.3-0.4). The expert union grows x1.5-1.6 per doubling of rows.
- **The SSD:** misses per kept token in replay: C 256 29.6, C 320 15.3 (replay ~1.3x pessimistic). Bigger night caches (C 300 / 320) are measured tonight. RAM streaming 258-284 GB/s; SSD ~11.3 GB/s effective.
- **The draft:** 6.3-12 ms on the GPU. On the Neural Engine, 12.4 ms per batched 3-candidate call, with the GPU slowed ~x1.02. It can be hidden only if an early-layer drafter works (verdict tonight).
- **Start-up:** ~130 ms extra on the first check after a long prompt (a fix is in test).

**Quality-defined additions available** (each changes some tokens; adoption is the owner's call)
- **The gate:** top-2 experts on rows where the drafter is sure. At cf >= 0.999: 47.7 % of rows, expert union -28.7 % (-40.8 % at 0.99), whole-sequence deviation 1.25 % of code tokens. Tested for quality on Wednesday night.
- **Relaxed acceptance R1** (log p(draft) >= log p1 - 0.5 nat): code kept x1.107 modelled at 1.19 % deviation (real ~half).
- **The comment loophole** (being measured locally): accept the drafter's near-equal word only inside comments and docstring prose, so code tokens stay exact. 9 of 10 sampled mid-row misses were in comments.
- **HC8:** -0.6 GB per check, ~-2-3 ms; it passed quality with a lean (all 5 changed answers were losses).
- **The quality budget:** an int4 copy of the model passed same-scores at ~1 % of code tokens deviating.

**Deliver, in this order**
1. **The budget** (<= 200 words): the cycle that 350 allows at 16 and at 32 rows with the drafter ceiling (cycle <= kept / 350), for HumanEval+-like and MBPP-like code.
2. **Piece by piece** (<= 450 words): for the dense pass, experts per row (with the union at 32 rows), draft, SSD, start-up amortized and host overhead, give today's cost and the realistic floor on this Mac, then the sum. Mark any piece that needs something not yet invented.
3. **The verdict** (<= 250 words): is exact 350 physically possible on this Mac with the drafter at its ceiling? If yes, which pieces must hit which numbers. If no, by how much it misses, and the smallest quality-defined addition (or combination) that closes it, with its deviation cost against the ~1-1.5 % code budget.
4. **The three measurements that decide it soonest** (<= 150 words). Big tests run weekdays 00:30-15:00 now.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware.
- Mark every number as given / arithmetic / judgement. Treat your own gains as upper bounds (ours ran ~2x high).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,200 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
