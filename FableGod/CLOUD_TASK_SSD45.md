# CLOUD TASK 12: the 45 GB line. Make the SSD disappear from the 350 cycle.

A closed thinking task. Your 12 GB answers worked out the mechanics: the stall is the per-layer chain (routing -> read -> compute), forecasting can't fix it on this model, and residency is the only non-predictive lever. Now apply that to the line that has to reach 350 first. Don't use the web; think and calculate. Everything you need is below.

**The goal.** 350 tok/s on code with the full 125B (all 512 experts) on this MacBook, at ~45 GB of RAM (the waypoint; 24 GB comes after). Your task 9 answer said 350 needs four engine pieces:
- the dense pass <= 17 ms
- the expert kernel within ~15 % of its byte floor
- the draft hidden
- the SSD hidden

This task is the SSD.

**The 45 GB engine today** (given)
- **The default:** 63.6 tok/s exact at C 256 expert slots per layer (of 512), 41.7 GB. A 16-row check takes ~166 ms by day:
  - draft ~12 ms (6.3 with the 4-bit drafter)
  - dense ~36 ms
  - experts ~3.0 ms per row (~53 at 16 rows)
  - SSD exposed ~39-65 ms
  - it keeps ~10.5 tokens
- **Experts:**
  - Rank tiers: ranks 0-1 at 4 bits, ranks 2-9 at 3 bits (plan 4433333333). Top-10 of 512 per layer, 48 layers.
  - ~7.38 GB of experts per 16-row check (the union). The union grows x1.5-1.6 per doubling of rows.
- **Misses per kept token** (replay; the replay runs ~1.3x pessimistic):
  - C 256: 29.6
  - C 320: 15.3
  - Bigger night caches (C 300 / 320, ~50 GB of RAM allowed at night) are measured tonight.
- **OOO:** the engine computes resident experts while missing ones load.
- **Look-ahead prefetch into reserved slots** (the next layer's router on this layer's input, with a learned correction):
  - recall 0.693
  - in replay: demand misses -44 % at C 224, wasted 98 < saved 187 reads (upper bound)
  - it is being built
- **SSD and RAM:** SSD ~11.3 GB/s effective inside the loop (0.088 ms per MB); RAM 258-287 GB/s.
- **The gate** (quality-defined; its quality test runs Wednesday): top-2 experts on rows where the drafter is sure.
  - At cf >= 0.999: 47.7 % of rows, expert union -28.7 % (-40.8 % at 0.99).
  - Deviation ~1.25-1.34 % of code tokens: the same floor profile as your 12 GB two-pass, because gated rows keep top-2 hidden states.
- **The cycle 350 allows:** <= kept / 350, i.e. ~36-37 ms per 16-row check at kept 12.5-13.

**Deliver, in this order**
1. **The SSD budget at 350** (<= 200 words): the bytes per check the SSD can move in a ~36 ms cycle; today's miss bytes per check at C 256 and C 320, with and without the gate; the gap.
2. **The stall mechanics at 45 GB** (<= 300 words): with OOO and the look-ahead, where does the SSD stall sit in the check (per layer, the first layers, the tail)? How much of today's 39-65 ms is bytes, and how much is the chain?
3. **Levers, priced** (<= 400 words; upper bound, then halved). Mark each as exact or quality-defined.
   - rank-aware residency: already measured, don't re-derive it. At 12 GB (C 51) it is KILL: pass-1 bytes 0.708 -> 0.669 GB per check, total +1.6 %. Even Belady's MIN on pass 1 gets only 0.378 GB (80.8 % hit). At C 160 pass 1 falls 0.224 -> 0.154 GB, but pass 2 rises 0.127 -> 0.200, so the total is flat. Say only whether 45 GB's C 256 + the gate changes that picture.
   - slot count by layer
   - the look-ahead at recall 0.69
   - the gate's smaller union
   - night caches C 300-320
   - anything else that cuts miss bytes or hides them
4. **The plan** (<= 250 words): the smallest combination that hides the SSD at the 350 cycle (<= 2 ms exposed), with each piece's required number. Do it for day (C 256, ~42 GB) and for night (C 300-320).
5. **Three measurements that decide it soonest** (<= 150 words). Small tests run by day on a shared GPU; big ones run weekdays 00:30-15:00.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware or SSD.
- Mark every number as given / arithmetic / judgement. Treat your own gains as upper bounds (ours ran ~2x high).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all five sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,300 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
