# CLOUD TASK 14: the fast mode, priced on code and chat at 45 GB

A closed thinking task. Don't use the web; think and calculate. Everything you need is below and in your earlier answers (tasks 5, 9 and 12).

**The quality contract** (this answers your task 13 pushback). Kamen's standing rule since 21-23 Oct:
- The exact mode stays exact, and it is the default.
- Optional fast modes, and the whole <= 12 GB line, are judged by task results: EvalSuite's same-scores test on math, code, instruction-following and multilingual. They are not judged by a token-deviation budget.
- So deviation numbers are diagnostics, and the test decides.

**The exact combination being built this week** (one exact 45 GB default; every merge bitwise-checked, measured each night):
- copy override: HumanEval+ x1.041 at 16 rows / x1.211 at 32; MBPP x1.000; agent x1.032
- the 4-bit drafter: 6.3 vs 10.4 ms per draft
- chat switch: chat -> the MTP chain, math -> DFlash2
- a start-up fix, and a cheaper expert path
- then, if they pass: the NAX6 expert kernel inside OOO, night caches C 300 / 320, prefill warm-up
- the exact HC read: -3.9 ms
- the attention kernel: -5 ms, if its quality re-test passes
- the R-57 look-ahead

Our rough guess: ~75-80 tok/s exact by day and ~85-95 at night, on code (today 63.6).

**Quality-defined pieces, each measured**
- **The gate** (top-2 experts on rows where the drafter is sure):
  - cf >= 0.999: 47.7 % of rows, expert union -28.7 % (-40.8 % at 0.99)
  - deviation 1.25-1.34 % of code tokens, and 4-5 % on chat through the context
- **Relaxed acceptance** (first-order, teacher-forced; real gains ~half):
  - R1 at 0.5 nat on code: +11 % kept at 1.19 % deviation
  - R2 (top-2 at 0.25 p1) for chat: deviation 1.9 / 7.1 / 4.8 % (code / IFEval / multilingual), modelled +51 % kept on IFEval and +30 % on multilingual. Measured live (R-61N): +8 to +23 % tokens per pass on chat.
  - Exchange rate: code +9-10 % kept per +1 % deviated tokens; IFEval +5-6 %; multilingual +4-5 %.
- **The comment loophole:** relax only inside '#' comments: ~x1.06 kept on code, with every code token exact.
- **8-bit HC weights:** -2.5 ms more on top of the exact HC read; same-scores PASS with a lean.
- **For reference, the 12 GB two-pass:** theta 0.8 changes 1.63 % of code rows, on a 1.16 % floor at any theta.

**Chat today**
- Kept per 16-row check on IFEval: 2.03 with DFlash2, 2.53 with the MTP chain. Multilingual is similar.
- 55-58 % of chat rows are "mid": the 125B itself is unsure (12 % on code).
- The MTP head at depth 1 on mid rows: top-1 0.86, top-3 0.982 (measured on code map windows).
- Exact token trees were killed on cost: an exact tree needs per-node GDN states, ~50 MB per layer.

**Deliver, in this order**
1. **The fast mode on code** (<= 250 words):
   - which pieces stack, and which don't
   - expected kept and tok/s at 45 GB, by day and at night, on this week's engine
   - the total deviation
   - the order to test them with EvalSuite, cheapest to validate first
2. **Chat** (<= 350 words):
   - why kept is only 2-2.5 per check, and the realistic exact chat tok/s
   - what a fast mode buys on chat: relaxed acceptance, the gate, fewer rows per check, the MTP head's top-3 at mid rows within the tree cost limit
   - the best chat configuration and its tok/s
3. **The risk ranking** (<= 200 words): which piece is most likely to fail same-scores, and the fallback.
4. **The test plan** (<= 150 words): EvalSuite's arms in order. Big tests run weekdays 00:30-15:00; EvalSuite has Wednesday and Thursday slots.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware.
- Mark every number as given / arithmetic / judgement. Treat your own gains as upper bounds (ours ran ~2x high).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,200 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
