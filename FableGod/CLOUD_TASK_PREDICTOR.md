# CLOUD TASK 4: the perfect predictor. 120-150 tok/s at 12 GB stays the goal.

A closed thinking task, continuing your 12 GB answer. Don't use the web; think and calculate. Everything you need is below.

**The owner's decision.** 120-150 tok/s on code at <= 12 GB stays the goal for the full 125B (all 512 experts) on this MacBook. Your own analysis says the gap is prediction: knowing before the check which rows are safe with only their top-2 experts, and which experts each row will need. So work toward perfect prediction.

**Your own numbers from last time**
- Per 16-row check at 12 GB: trunk read ~10 ms (2.7-3.3 GB from RAM), oracle SSD bytes 52-59 ms (~0.67 GB per check at 11.3-13 GB/s), drafter ~6 ms. That gives <= 133-147 tok/s at 10 kept per check, and 150 needs 12+ kept with oracle bytes.
- 120 needs a predictor within ~1.1x of the oracle's bytes. The realizable gate is 3.7x off.
- Your two-pass check (drafter tiers in pass 1, certify at the pass's own p >= 0.9, exact for the rest up to the first contradiction, plus a pinned hot set) is bounded at 93 tok/s at 10 kept, 115 with a drafter committee. You'd bet on 70-80.
- Your open question (cache churn under a realizable design) is being measured tonight by our local session, by CPU replay of recorded routing.

**Measured signals for predicting the 125B's sureness** (code unless marked)
- **The classes.** The 125B is sure (p1 >= 0.99) on 86.8 % of rows: the oracle's coverage. The drafter is sure (cf >= 0.99) on 65.7 %.
- **The drafter's confidence (cf) gate:** 57-64 % of rows at <= 0.1 % top-2 flips. At cf >= 0.999: 47.7 % of rows, 0 flips in 1,607. At 0.99-0.999: 17.9 %, top-5 at 0 flips in 604.
- **A learned certificate head** on the drafter's features: 56 % coverage (killed). A perfect certificate predictor would cover 73.1 %.
- **The cheap pass's own confidence** (oracle-grade, but it costs a second trunk pass):
  - top-5 experts with p1 >= 0.9: 89.8 % of rows, 0 flips
  - top-2 with p1 >= 0.99: 73.4 %, 0 flips
- **Copy state:** inside a >= 8-token span seen earlier in the window, the 125B is sure 99.7 % of the time (n 2,103), vs 65.4 % outside. The share of copies is ~40 % of positions on HumanEval+ (real walk), 10 % on agent text, 5 % on MBPP.
- **The previous row's class** is 93 % predictive, but only across checks. Routing agreement with the previous position: no signal.
- **Token kind** (top-2 flip rate by the target token's kind): words after a space 6.6 %, punctuation 3.4 %, whitespace 2.0 %, numbers 0.7 %, word continuations 0.5 %. Numbers and word continuations never end a sure run.
- **Sure runs:** mean 13.7, median 2, p90 17 rows on code; ~2 on chat.

**Measured signals for predicting experts**
- The prompt's experts cover 91.1 % of the answer's rank-0/1 slots and 84.0 % of ranks 2-9 on code; prompt + the first 100 answer rows cover 95.3 / 91.5 % of the rest. On chat the prompt covers 70-72 % and the first 100 rows 93 %. The prompt uses ~253 distinct experts per layer on code, against a 12 GB cache of ~51 per layer.
- Copies don't reuse deep routing: expert overlap with the source is 66 % at layers 0-7, 33 % at 8-35, 27 % at 36-47.
- Consecutive positions share 27 % of experts on code (48 % multilingual).
- A look-ahead router predicts ~75 % of upcoming experts. Perfect forecasts 3 layers ahead were worth +35-44 % at 24 GB (a ceiling).
- Top-k structure: ranks 0-1 at 4 bits (2.8 MB each), ranks 2-9 at 3 bits (2.1 MB); 2-bit experts failed quality.

**Deliver, in this order**
1. **The target in numbers** (<= 200 words). For 120 and 150 tok/s at 12 GB, with kept 10, 12 and 13 per check:
   - the maximum SSD bytes per check
   - the coverage a predictor needs at <= 0.1 % flips
   - the time left for the trunk and the drafter
2. **At least four predictor designs** (<= 600 words). Candidates, and invent better:
   - fusing the cheap signals above into one calibrated certifier with a guaranteed flip rate (e.g. conformal)
   - a cheaper first pass: resident experts only (zero SSD bytes), or an early-layer self-certificate (the 125B's own confidence read at a middle layer)
   - expert forecasting that starts SSD reads earlier
   - a drafter that also predicts the 125B's sureness

   For each give:
   - expected coverage and flips (bounded by the numbers above)
   - SSD bytes per check
   - extra time
   - exact or quality-defined
   - prior art from memory ("published" / "likely new")
   - the cheapest CPU test on our recorded data, with PASS / KILL lines. Our local session has: per-row classes and flips under cuts, the drafter's cf per row, copy state, token kinds, and the routing of 36 windows.
3. **The 12 GB plan for 120 and for 150** (<= 300 words): which predictor, what kept per check, what trunk savings, which cache policy, with the arithmetic.
4. **The single most important experiment** (<= 100 words).

**Rules**
- The full 125B (all 512 experts), this MacBook only. Exact modes stay exact; quality-defined modes are allowed on this line, judged by task results.
- Mark every number as given / arithmetic / judgement, and treat your own gain estimates as upper bounds (ours ran ~2x high today).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,500 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
