# CLOUD TASK 8: the knowledge wall. Can any affordable drafter match a 125B MoE on ~97.5 % of tokens?

A closed thinking task, and the deepest open question in the project. Don't use the web; think and calculate. Everything you need is below.

**Why.** Exact speculative decoding keeps a drafted token only if it equals the target's greedy token. With the engine at its realistic floor, exact 350 tok/s on code needs ~20 tokens kept per 32-token check: ~0.975 per token, flat over depth. Today's drafter gets 0.909. Our local analysis just showed the fade with depth is the drafter's own, calibrated uncertainty, and a copy override already flattens it inside copied text. So 0.975 is not a depth problem. It is knowledge: the drafter does not know what the 125B will choose. Find out whether that wall can be broken.

**The target and the machine**
- Qwen3.8-Flash-Next, 125B MoE:
  - 48 layers (36 Gated-DeltaNet + 12 attention), hidden 2,560, 4 hyper-connection streams
  - 512 experts, top-10
  - a native 1-layer MTP head
- M5 Pro MacBook, 64 GB, ~270 GB/s, plus an idle Neural Engine.
- The target is fixed: it cannot be fine-tuned here. Drafters can be trained on the laptop (small) or on free cloud compute, never paid GPUs.

**Measured on code** (3,283 drafted positions, today's DFlash2 drafter)
- **The classes, by the 125B's own confidence:**
  - sure (p1 >= 0.99): 86.5 % of positions, drafter right 95.5 %
  - mid: 12.0 %, right 65.9 %
  - tie (p1 - p2 <= 0.1): 1.5 %, right 30 %
- **Inside >= 8-token copies** (~40-58 % of HumanEval+ positions) the 125B is sure 99.7 % of the time, and a copy override gets those positions right ~99 %. Outside copies the 125B is sure only 65 % of the time. On MBPP-like code copies are ~5 %.
- **Sure-row misses** are knowledge: return types, a docstring's own words, the algorithm chosen, and numbers (counting drift, 14 % of deep misses on code, 27 % on math).
- **Mid-row misses are fragile choices.** The 125B's own argmax at mid rows flips under small perturbations of its own computation:
  - top-5 experts instead of 10: 8.6 %
  - top-2 experts: 25.6 %
  - zeroing the shared expert: 30.5 %
  - static hyper-connections: 25.9 %
  - reusing middle-layer MoE outputs: 33 %
  - attention reuse: 29 %; GDN reuse: 97 %

  An int4 copy of the model differs on ~1 % of code tokens. At mid rows the drafter's wrong token is the target's close second (within a quarter of p1) only 12.5 % of the time.
- **Ceilings measured:**
  - a near-copy (the trunk + 10 % of the experts) agrees 0.983 per token, but drafts sequentially, one ~11 ms trunk read per token
  - the int4 twin ~0.99
  - the 125B's own MTP head chained with training-time testing holds 0.94 flat over depth
- **What failed:**
  - more data: +0.14 kept per doubling and flattening
  - bigger drafters: ~+6 % per 1.6x layers
  - two near-identical drafters: errors overlap 64 %
  - a learned certificate head on the drafter's features
  - a probe on the 125B's own layer-25 / 35 states: it can't tell sure from unsure rows purely
  - training only on unsure positions
  - trees, autoregressive heads, refinement, self-speculation
- **Budget for a drafter:** <= ~10 ms per 16-32-token block on the GPU, or ~12-15 ms on the Neural Engine in parallel with the check (one batched call of a 0.47B drafter for 3 candidate blocks: 12.4 ms measured), <= 2-4 GB RAM.

**Deliver, in this order**
1. **The arithmetic of 0.975** (<= 200 words): which sure / mid / tie accuracies reach it on code, with and without the copy override; and the same on copy-poor code.
2. **The information argument** (<= 400 words): what a predictor must know to match the 125B at mid rows and at sure rows outside copies.
   - Is the mid-row choice a function of information a cheap model can reach: the target's own MTP head output, its KV / GDN state at the anchor, its first k layers, the experts' identities?
   - Or does it need the full deep computation?
   - Use the perturbation data above. A careful, honest bound is worth more than optimism.
3. **At least three designs** that could plausibly reach ~0.975 within the budget, or the strongest argument you can make that none can (<= 450 words). Invent beyond the list of failures. For each: the mechanism, expected per-class accuracy, cost on this Mac, and what must be true.
4. **The cheapest decisive experiment for each design** (<= 200 words), runnable by our local sessions: a CPU replay on recorded data, or one GPU night. Give PASS / KILL lines.

**Rules**
- The full 125B as the target; drafters only propose, and exact mode stays exact. Quality-defined acceptance is a separate mode, not the question here.
- Mark every number as given / arithmetic / judgement. Treat your own gains as upper bounds (ours ran ~2x high).

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,300 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
