# CLOUD TASK 2: make two drafters better than one

A closed thinking task. Don't use the web (this environment blocks most sites); think and calculate. Everything you need is below.

**Our setup**
- Target: Qwen3.8-Flash-Next, a 125B MoE (48 layers, 512 experts, top-10).
- Hardware: one MacBook (M5 Pro, 64 GB), MLX.
- Decoding: exact greedy speculative decoding. A block drafter (DFlash2, 1.2 GB) drafts 16 tokens in one pass. The target checks them in one pass, and a check keeps the matching prefix + 1.
- Today it keeps ~10.5 tokens per 16-token check on code.

**Measured on code** (HumanEval+, 3,283 drafted positions)
- The target's own confidence splits positions into three classes:
  - sure (p1 >= 0.99): 86.5 % of positions
  - mid: 12.0 %
  - tie: 1.5 %
- The drafter is right on 90.9 %: sure 95.5 %, mid 65.9 %, tie 30.0 %. Its misses split 43 % sure / 45 % mid / 12 % tie.
- By depth: 89.8 % (1-3), 92.7 % (4-7), 92.5 % (8-11), 87.5 % (12-15).
- At mid positions the drafter's wrong token is seldom the target's second choice: only 12.5 % of mid rows have it within a quarter of p1.
- **Two drafters** (two near-identical DFlash2 variants, replayed):
  - right together (the union) 94.0-94.4 %
  - picking the higher-confidence draft per position 92.7 % (sure 95.5 -> 96.9, mid 66 -> 70)
  - so today's pick captures ~55 % of the union's gain
- **An exact in-window match model** (copy the token that followed the same >= 8 tokens earlier in the window):
  - covers 58 % of positions at 99.2 % precision, where the drafter is right 96.1 %
  - "match if >= 8, else the drafter" lifts 92.4 -> 94.2 % on that walk set
  - shorter matches are worse than the drafter: 38 % vs 88 % (1-3 tokens), 61 vs 86 (4-5), 78 vs 93 (6-7)
- **Other domains:** math (GSM8K) 87.0 % (sure 94.3, mid 61.5, tie 33.5), where the match model is worse than the drafter. Chat (IFEval) 59.3 % (88.4 / 49.0 / 22.3). Multilingual 51.4 %.
- **Dead ends here:**
  - one drafter trained only on unsure positions; mode-seeking targets
  - bigger single drafters (+6 % tokens per 1.6x layers, cost grows with layers)
  - a learned acceptance certificate; more data (+0.22 kept per doubling)
  - a per-prompt bandit switch between drafters; a picker over one drafter's candidate lattice
- **Costs:**
  - one drafter: 6.3-12 ms per 16-token block
  - a second drafter in the same batched call: +1-2 ms and +0.24 GB (4-bit) / +0.47 GB (bf16)
  - training a drafter: ~1 h on the laptop (2 epochs, ~32K on-policy tokens)
  - available on-policy data: ~65K tokens of the target's own code answers, plus math and chat sets

**The question.** Design a committee of 2-3 complementary drafters plus a per-position selector that maximizes exact per-token agreement (and kept per check). The budget is +3 ms and +1 GB per block, and it must be trainable on the laptop.

**Deliver, in this order**
1. **Make them disagree usefully** (<= 350 words). How should member 2 (and maybe 3) be trained so its errors are complementary to member 1's? Options include:
   - boosting-style reweighting of member 1's misses
   - diversity terms; different inputs, taps, data or block size
   - the target's own MTP head as a member
   - the match model as a member

   For each: why the errors would be less correlated, and the expected union. Then pick one.
2. **The selector** (<= 300 words). Per position, which member's token goes to the check?
   - Features available at draft time: each member's confidence, agreement between members, depth, match length, token kind, position in the line.
   - How it learns online from the check's free labels (the target's true tokens) and stays calibrated, e.g. PAQ-style logistic mixing with SSE / APM, or a tiny online logistic model.
   - The expected share of the union's gain it captures (today: ~55 %).
3. **The numbers** (<= 300 words). State a simple model of correlated errors. Then estimate per-token agreement and kept per 16- and 32-token check on code, math and chat for:
   - (a) today's pair with the confidence pick
   - (b) your committee + selector
   - (c) (b) + the match model

   Mark every number as given / arithmetic / judgement, and name the 2-3 quantities we must measure first.
4. **The test** (<= 200 words). A pre-registered CPU replay on our stored walks (we have, per position: the window's token ids, the target's token, each member's draft and confidence, depth, class), plus one laptop training night. Give PASS / KILL lines.
5. **Pseudocode for the selector** (<= 50 lines, plain Python-like, no libraries).

**Rules**
- Exact mode stays exact: members only propose, and the check decides.
- Drafters run on the Mac. Train on the laptop (small drafters) or free cloud compute; never paid GPUs.
- Prior art from memory only: tag each idea "published" or "likely new". Don't search.

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs for calculations (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all five sections to ANSWER.md, even if some parts say "not yet". Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message. Total <= 1,400 words + the pseudocode.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
