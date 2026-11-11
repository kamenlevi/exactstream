# CLOUD TASK 5: the 350 mode. The fastest quality-tested combination at 45 GB.

A closed thinking task. Don't use the web; think and calculate. Everything you need is below.

**The goal.** The owner's first goal is 350 tok/s on code for the full 125B MoE (Qwen3.8-Flash-Next, 512 experts, top-10) on his MacBook, "one way or another". Exact mode stays exact. Quality-defined modes are allowed if they keep benchmark scores the same, and adopting one is his call. This week turns tested levers into measured speed, and Wednesday night has one ~5 h quality-test slot. Design the mode that slot should test.

**Measured today** (45 GB line, code unless marked)
- **The exact default:** 63.6 tok/s at 41.7 GB, C 256. One 16-row check takes ~166 ms by day:
  - draft ~12 ms
  - dense ~36 ms
  - rows 16 x ~3.0 ms
  - SSD exposed ~65 ms
  - it keeps ~10.5 tokens
  Bigger night caches (C 300 / 320) are measured tonight.
- **Lesson:** our modelled gains came out ~2x too high against real runs today (a modelled x1.071 measured x1.038). Halve modelled gains unless a real run backs them.

**Quality-defined levers, measured**
1. **The gate** (top-2 experts on rows where the drafter is sure). At cf >= 0.999 it covers 47.7 % of code rows with 0 flips in 1,607 gated rows; at cf >= 0.99, 2 flips in 2,211.
   - Downstream context changes still move other rows. Whole-sequence deviation: code 1.25 % at 0.999 (1.34 % at 0.99), IFEval 3.97 %, multilingual 2.50 %.
   - Live expert union: -28.7 % at 0.999 and -40.8 % at 0.99. SSD MB per kept token: -28 / -40 %.
   - Modelled tok/s: 70-76 (bytes only) to ~98 (if the expert compute halves too). Not yet run live.
2. **HC8** (8-bit hyper-connection mixers): passed the same-scores test.
   - Math 84 vs 88 %, IFEval 89 vs 90 %, HumanEval+ 91.7 vs 95.0 %; all 5 changed answers were losses (p 0.06).
   - -0.6 GB per check, ~-3 ms.
   - On the map: sure rows 0 flips, all code rows 0.33 %.
3. **Relaxed acceptance** (first-order, likely ~half in a real run):
   - R1, accept the draft if log p(draft) >= log p1 - 0.5 nat: code kept x1.107 at 1.19 % deviated tokens; IFEval x1.17 at 3.69 %.
   - R2, the draft is the target's second choice at >= 0.25 p1: code x1.176 at 1.92 %; IFEval x1.35 at 7.08 %.
   - Gate + R1 on code: ~2.4 % deviation combined.
4. **A copy-keyed gate** (top-2 on rows inside >= 8-token spans already in the window, where the target is sure 99.7 % of the time): +3-8 points of gate coverage on copy-heavy code, ~0 on MBPP.
5. **A two-pass check** (cheap top-2 pass, certify at p1 >= 0.99, full experts only for the rest): it wins at 12 GB, but at 45 GB a second dense pass (~36 ms) makes it lose to the single-pass gate.

**Exact levers being built:**
- a 4-bit drafter: code +1.5-2.5 %
- a chat drafter switch: chat +24.5 %
- a start-up fix
- a copy-override precision gate: ~+4 % on code
- night caches

**Quality references**
- An int4 copy of the model differs on ~1 % of code tokens and ~5 % of chat tokens, with identical scores.
- **The test (decision 4):** paired deltas on 210 answers (GSM8K 50, IFEval 100, HumanEval+ 60). A set fails when its 95 % interval excludes zero. One earlier variant failed IFEval at -5.0.
- Cuts combine by adding margin loss, not by union: an additive model over-predicts sure-row flips 1.6-1.9x and matches mid rows.
- Wednesday's current plan has two arms: (a) R1 0.5 + the gate at cf >= 0.99; (b) the gate alone.

**Deliver, in this order**
1. **The mode** (<= 400 words): which levers, at which thresholds, for two budgets.
   - (a) The twin's level: <= ~1.5 % code tokens and <= 5 % chat tokens deviated, same scores.
   - (b) A stretch budget: ~3 % code.
   - Give the expected deviation per domain, and say what you would leave out and why.
2. **Honest speed** (<= 250 words): measured 63.6 as the base, modelled gains halved. Give tok/s on code and chat, by day (C 256) and at night (C 320), for (a) and (b). Say how far each is from 350 and what still has to come from exact engine work and the drafter.
3. **Wednesday's arms** (<= 250 words): the 2-3 arms to run in the ~5 h slot, each with:
   - its exact settings
   - your prediction
   - PASS / KILL lines
   - which arm to drop if the slot runs short
4. **The build order** (<= 150 words): in what order to put the exact and quality-defined levers into the default this week, so the measured number climbs every day.

**Rules**
- The full 125B (all 512 experts), this MacBook only.
- Mark every number as given / arithmetic / judgement.

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,300 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
