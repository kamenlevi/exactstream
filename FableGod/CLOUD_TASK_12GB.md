# CLOUD TASK 3: the 12 GB line. 120-150 tok/s for a 125B model in 12 GB of RAM.

A closed thinking task. Don't use the web (this environment blocks most sites); think and calculate. Everything you need is below.

**The goal** (the owner's end goal): the full 125B model (all 512 experts, no pruning) decoding at 120-150 tokens per second on code, on one MacBook, with a lifetime memory footprint of at most 12 GB. Quality-defined modes are allowed. The small-RAM line is judged by task results (same scores on benchmark answers), not by an exact token match.

**The model and the machine**
- Target: Qwen3.8-Flash-Next, a 125B MoE. 48 layers, 512 routed experts per layer, top-10 per token, Gated-DeltaNet + attention layers, hyper-connections.
- Dense trunk: ~3.3 GB per check (attention, GDN, HC, router, shared expert, head); 2.7 GB with 8-bit HC, which passed quality today.
- Experts: ranks 0-1 at 4 bits (2.8 MB each), ranks 2-9 at 3 bits (2.1 MB each). Top-10 x 48 layers x ~2.2 MB = ~1.06 GB of expert reads per token if nothing is cached. 2-bit experts failed quality everywhere.
- Mac: M5 Pro, 64 GB unified memory, ~270 GB/s RAM bandwidth. Internal SSD 12-14 GB/s raw; measured ~11.3 GB/s effective inside the decode loop (0.088 ms per MB).
- Neural Engine: idle until today. It now runs our 1.2 GB drafter at ~25 ms per 16-token block, slowing GPU work by only 1-7 %.
- At 12 GB the expert cache holds ~51 experts per layer (~10 %); everything else streams from the SSD.

**Decoding**
- Speculative and exact by default: a block drafter proposes 16 tokens, and the 125B checks them in one pass (a "check").
- Kept per 16-token check on code: ~10 today (math ~7, chat 2-2.5). Each extra row in a check adds new experts: the expert union grows x1.5-1.6 per doubling of rows.

**What we measured** (code unless marked; numbers from replays and live engines)
- **Today at 12 GB:** exact ~18 tok/s by replay, 16-27 measured on the engine. SSD-bound.
- **The gate** (quality-defined): rows where the drafter is very confident run only their top-2 experts (ranks 0-1).
  - Drafter confidence >= 0.999: 47.7 % of rows, top-2 at 0 flips in 1,607 rows.
  - 0.99-0.999: 17.9 %, top-5 at 0 flips in 604.
  - The rest run exact.
  - SSD misses x0.46 and expert bytes x0.58 (replay at 45 GB).
  - Whole-sequence deviation ~1.25 % of code tokens (3.97 % IFEval, 2.50 % multilingual), the same profile as an int4 copy of the model that kept identical benchmark scores.
- **Caps by SSD on code at 12 GB:** exact 18 -> realizable gate 46 (~60 scaled to the engine) -> oracle 169 (~220).
  - The oracle knows which rows the 125B itself is sure of (86.8 % of code rows, versus 65.7 % where the drafter is sure).
  - The realizable-to-oracle gap is 3.6x in SSD misses. It is the predictor problem: knowing before the check which rows are safe to cut.
- **Predictors tried:**
  - the drafter's confidence: covers ~57-64 % of rows at <= 0.1 % flips
  - a learned "certificate head" on the drafter's features: 56 %, killed today
  - routing agreement with the previous position: no signal
  - the previous row's class: 93 % predictive, but only across checks
- **A cheap pass's own confidence is oracle-grade.** Run the check with only the top-5 experts per row; rows whose top token has p >= 0.9 keep the exact token (89.8 % of code rows, 0 flips; top-2 with p >= 0.99: 73.4 %, 0 flips). But it costs a second dense pass (~36 ms per 16-row check today; 15-18 ms is reachable). Judged "marginal at 12 GB" by an earlier estimate.
- **Perfect expert forecasts 3 layers ahead:** +35-44 % on code at 24 GB (a ceiling, not deployable). Online cache-eviction variants mostly failed.
- **Chat:** kept per check is the cap (exact 3.1-3.6 GB of reads per kept token, oracle 1.8-2.0).

**Deliver, in this order**
1. **The budget** (<= 200 words). The SSD bytes per kept token, and per check, that 120 and 150 tok/s allow at 11.3-13 GB/s, against the exact, realizable and oracle figures above. Is 120-150 inside the oracle's reach on code? On chat?
2. **Close the gap** (<= 500 words). Price each lever with the numbers given:
   - the two-pass check (the cheap pass as the predictor, then exact only for uncertified rows), at 12 GB where the SSD, not RAM bandwidth, is the bottleneck
   - cache policy and size inside 12 GB
   - expert forecasting
   - rows per check
   - the Neural Engine as a second compute unit
   - anything better you can derive

   For each: SSD bytes per kept token, tok/s, RAM, and whether it is exact or quality-defined.
3. **The best design** (<= 300 words). The 12 GB engine that gets closest to 120-150 on code, with its arithmetic and what chat gets. If 120-150 is impossible, name the physical limit and the best reachable number.
4. **Three decisive tests** (<= 200 words), each one night or less on this Mac, with PASS / KILL lines.
5. **One paragraph on 24 GB:** what the same design gives there.

**Rules**
- The full 125B (all 512 experts), no smaller model, this MacBook only, no external hardware.
- Exact modes stay exact; quality-defined modes are allowed on this line and judged by task results.
- Mark every number as given / arithmetic / judgement. Your estimates of gains are upper bounds: our own estimates came out ~2x high today when tested for real.

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs for calculations (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all five sections to ANSWER.md, even if parts say "not yet". Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,500 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
