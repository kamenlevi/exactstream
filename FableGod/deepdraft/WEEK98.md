# WEEK98: the 98 % deep drafter, Sat 14 Nov -> Sat 21 Nov 2026

> (a quote from the project owner, kept private)
itself or the method for it being certified that it will work in a week, because I don't have more time than that to
spare to show this part of the project to my dad. I mean, so yeah, basically we must be able to find the thing which
unlocks 98% accuracy at 30 tokens deep. By then we must know what it is, and either I don't care if in a week we just
know the principles of how everything will work and it's certain, and we still have a week of building everything. I
don't care about that. I just need to get that."

## The goal
- By Sat 21 Nov, know the method that gets the drafter to >= 98 % right per guess at every depth up to 30. Held-out test
  numbers must back it; a small-scale proof counts. The full build can take the week after.
- The 125B stays untouched. Only the drafter learns.
- Measure first on held-out HumanEval+ windows (the 23-30 band), then on MBPP and everyday text.

## Where we start (t30b, per guess)
| | depth 0 | depths 23-30 |
|---|---|---|
| HumanEval+ | 97.90 | 89.66 |
| MBPP | 94.01 | 72.13 |

- Data: about +1 pt per doubling (t30a -> t30b). 98 % this way needs over 100x the data, so it isn't the path.
- Capacity: t30x2 NO EFFECT (-0.79 pt). t30x4 and TAPSQ (multi-layer taps) land Sat ~15:10 / ~15:25.
- 98 % at depth 30 means the deep guesses lose nothing against the first guess, and the first guess gains a little too.

## Why it matters (Kamen, Sat 14 Nov 15:2x, in short)
- Be the first: one proven recipe that keeps any drafter ~98 % right 30 tokens deep, usable on every model, exact
  output, ideally "a little engine package". It unlocks huge local models: the 125B in 30-45 GB at ~1,000 tok/s on code,
  and much bigger models streamed from SSD on a big Mac.
- 98 % per guess for 30 deep = about 23 tokens per check of the big model (sum of 0.98^k for k = 1..30, plus the bonus
  token), against 13.83 for t30b today. The other half of 1,000 tok/s is a fast check, about 23 ms per 32-row check.
- "No retraining": the big model is never retrained. Each model's drafter likely needs one short training, because
  drafterwall (11-11) showed training-free arrangements don't move kept by 20 %. Say so plainly.

## How we go fast, and how we show it's general
- Fast inner loop: try levers on a small local target model first, where a drafter trains in minutes, then confirm the
  winners on the 125B. Downloads follow the standing 11-07 rule (20 GB stays free for swap; log the name, source and
  size).
- Before calling it a method for any model, show it on a second model family as well as the 125B.

## Rules
- Every lever gets a screen of at most 15 min, with "result -> action" written first (pacer/RULES.md, 11-14).
- Long runs only confirm, with a stop line at 10-25 % of their steps against the baseline at the same steps.
- Results go to GitHub the same turn, with a one-line note to the orchestrator.
- A lever that changes the output (lossy) needs EvalSuite's same-scores gate before it counts, and Kamen's call to adopt.

> (a quote from the project owner, kept private)
The fresh state h_{s+j-1} gives the head two things its carried stand-in lacks:
- **(a) RETRIEVAL.** Information pulled from the far prefix by attention in all 48 layers, which becomes relevant only
  because of the new tokens (an identifier defined 200 tokens back, an open bracket, a call signature).
  - The fix is to give the drafter lookups: a copy pointer / prompt-lookup over the context, or attention over the
    target's KV cache of the verified prefix, which is computed anyway (prior art: GLIDE-with-CAPE).
- **(b) COMPOSITION.** Deep processing of the new tokens themselves.
  - The fix is a learned transition model that steps the state forward (ROLLOUT).
- **(c) DIMENSION.** Maybe only a few directions of the state matter to the head. Then a small predictor only has to
  regenerate k numbers per step.

Fast tests, mostly no training:
- **T1 COPYABILITY** (BigMoE, CPU): the share of oracle-fixable deep misses whose token appears earlier in the context
  or is predicted by a prompt-lookup n-gram.
- **T2 CONTEXT TRUNCATION** (Sandbox98, 2B): fresh states from the last K tokens only, accuracy vs K. Local -> (b);
  needs full -> (a).
- **T3 SUBSPACE** (BigMoE, REPLACE): true wd in its top-k PCs only, accuracy vs k -> (c).
- **T4 ROLLOUT** (both): is the state's evolution Markov in (h_t, x_{t+1})?
The method follows from which of (a) / (b) / (c) carries the gap.

> (a quote from the project owner, kept private)
- **Feature bank:** run the 125B once to store its states (taps {5,15,25,35,45} + wd, plus the top-32 log-probs on
  HumanEval+) for ~500 train answers + MBPP 100 + the HumanEval+ windows. Most screens then never touch the 125B, its
  heat or its SSD (BigMoE, after COMBO5).
- **Seconds-fast proxy:** score a candidate map or formula by held-out R2 / direction error against the true deep
  state. Day 1's six points are monotone (0.41 -> 41.7 ... 0.82 -> 91.6, threshold near 0.8). Only proxy winners get
  the ~5-min real eval. The proxy filters; it never decides.
- **Paired small evals for screens:** e.g. 30 MBPP windows with a per-window sign test. The full set only confirms.
- **Batches, not single tests:** write 5-10 screens at once, each with its result -> action line, and run them back to
  back in one runner. No agent turn between tests.
> (a quote from the project owner, kept private)
  the control measurably moves; below it, NO EFFECT means untested (Sandbox98's 300-step lesson).
  - Screen at depth ~12 instead of 30 (~2.5x cheaper per step); confirm winners at 30.
  - Train the control plus 2-3 levers in ONE process on shared slices.
  - Continue the base's saved Adam state at its own lr (no drift).
- **Parallel:** CPU screens across sessions while the GPU does one confirmation at a time. Fans up for SSD-heavy
  chains when Kamen OKs.

## Day 1 first: the CEILING test (BigMoE)
- How often is the 125B's own choice a near-tie (the top-1 vs top-2 margin, in bins)?
- What share of t30b's misses at depth 0 and at depths 23-30 sit on near-ties?
- Result -> action:
  - If near-ties alone block exact 98 %, the method must include close-call acceptance (lossy, quality-gated).
  - If the misses are mostly NOT near-ties, they are learnable: focus on what the drafter sees and how fresh it is.

## Owners
- **BigMoE:** the ceiling test, then the GPU screens of depth levers (taps, fresher features).
- **MTP Head (TrainLab, from Sun 01:00):** training levers (D98 owner).
- **NewIdeas (from Sun 01:00):** the lever list with a short prior-art pass (published / untested / new), handing out
  15-min screens.
- **Breakthrough:** close-call acceptance (its delta-0.05 pricing), carried to the 98 % line.
- **MoonShots:** levers from outside the drafting literature.
- **EvalSuite:** the quality gate for any lossy lever.
- **Leap:** keeps 1.2.1 ready for Kamen's word. Engine help when a lever needs it, e.g. a cheap partial trunk pass on
  drafted rows.
- **Orchestrator:** the daily plan, verification, reports to Kamen.

## Day plan
- Sat 14: the t30x4 + TAPSQ verdicts; the ceiling test.
- Sun 15 - Wed 18: breadth, with 15-min screens of every depth lever.
- Thu 19 - Fri 20: the best lever's proof run (with a stop line) and the method write-up.
- Sat 21: the answer for Kamen: the method and its numbers, or exactly what blocks it, with the numbers.

## Log
- Sat 14 15:0x: goal set; charter written (orchestrator).
- Sat 14 15:08: CEILING part 1 (BigMoE e4f9cd3): the 125B's near-ties (m < 0.25 nat) are about 2 % of answer rows,
  < 0.1 nat 0.65-0.82 %; the median margin is 8-9 nats. Ties cost at most ~2 pt, so 98 % is not ruled out. Part 2
  (t30b's misses by margin) decides LEARNABLE vs MIXED.
- Sat 14 15:08: TAPSQ amendment 1 (b44d00f): tap lr 1e-4 -> 1e-5. A zero-init wide input at 10x lr spikes the loss
  (0.35 -> 3.78). The same mechanism likely caused t30x2's / t30x4's early spikes, so the capacity verdicts carry that
  caveat. Candidate: a 15-min capacity retest at 1e-5 against t30b's saved per-step KL.
- Sat 14 15:15: t30x4 verdict HURTS: HumanEval+ 23-30 88.14 % vs t30b 89.66 % (-1.51 pt); MBPP 70.01 vs 72.13; kept
  13.20 vs 13.83. Both capacity arms lose about 0.7 pt per doubling. A clean 15-min retest at branch lr 1e-5 is queued
  after TAPSQ and CEILING part 2.
- Sat 14 15:20: **CEILING = LEARNABLE** (BigMoE 62f5b9e, t30b on MBPP 100, every cell).
  - Tie-limited ceiling: 98.87 % at depth 0, 98.76 % at 23-30, so ties don't block exact 98 %.
  - 75.8 % of the 23-30 misses (21.12 of 27.87 pt) are on rows the 125B was sure of (m >= 1 nat). On very sure rows
    (m >= 4) the first guess misses 0.1 % and the deep guesses miss 16.1 %.
  - 7.9 % of the deep misses fall outside the 125B's top-32: the drafter loses the thread with depth.
  - The tie tax is real: depth 0 misses 54 % of near-ties, about 1.1 pt. Exact 98 % leaves about 0.9 pt for everything
    else.
  - Close-call acceptance adds +0.2 / +0.9 / +1.7 pt at depth (delta 0.1 / 0.25 / 0.5): an end bonus that buys back the
    tie tax, not the week's lever.
  - Caveat: MBPP only (the HumanEval+ windows carry no log-probs).
  - **The week goes to what the drafter sees and how fresh its view is at depth.**
  - Breakthrough: close-call stays a small end piece; its main week job is a depth lever (NewIdeas hands it one).
- Sat 14 15:31: TAPSQ judged KILL (MBPP 23-30: C 58.26, T 57.34; ratio 1.364), but it is held as VOID pending a harness
  check.
  - The control arm itself collapsed: t30b 72.13 -> C 58.26 after 80 steps at lr 1e-5, while depth 0 only went
    94.0 -> 93.1.
  - Suspect: the quick trainer gives the deep depths no signal (the smoke's per-depth KL 24-30 reads 0.0000).
  - Line before any rerun: C within ~1 pt of t30b. The raw-taps test waits for that fix.
  - Note: T's deep-depth KL was 0.43-0.49x C's, which is worth a look once the harness is sound.
- Sat 14 15:32: fans ramp 5900 -> 3750 (-250 / 2 min, fans_down_1114.sh), as told to Kamen.
- Sat 14 15:5x: an outside review (pasted by Kamen) made two fair points:
  - The ceiling is per dataset and decoding mode. Ours is MBPP code at greedy, and it's an estimate from the measured
    near-tie miss rate, not a hard bound.
  - Chat and agent text will be harder than code.
  - So early this week (owner NewIdeas / BigMoE): one quick measurement of the same per-depth curve (conditional
    acceptance, depth 1..30) on chat and agent answers. That means a small capture of ~50 answers plus t30b's eval,
    within the 15-30 min rule.
  - The expert-union cost per check is the other half of the speed goal (BYTES10 / union pricing), not this week's
    target.
- Sat 14 15:5x: the outside review's second round. Its "oracle" test (give the drafter the 125B's true states at the
  deep positions) is drafterwall's Law 2 (RESULT.md, 11-11).
  - With the true state two tokens back, the reader holds 97.7 % on HumanEval+ at every depth (vs 86.6 % at 23-30
    without), and 93.5 % on MBPP (vs 76.2 % by 7-10).
  - So the deep drop is stale information that lives in the 125B's forward pass. The fix must be architectural: a
    learned stand-in carried through the draft, or a cheap refresh mid-draft.
  - On MBPP even fresh states top out at 93.5 %, so the reader must improve there too.
  - Contamination check (orchestrator): all 3,072 n1112 training captures are src dlog-magicoder, split train; none
    come from HumanEval or MBPP. TrainLab's notes list MBPP as held out. Whether the 125B saw the benchmarks in
    pretraining is unknowable: the chat/agent curve is the real test.
- Sat 14 15:53: TAPSQ2 harness valid: C vs t30b at MBPP 23-30 -0.07 pt, band 0 -0.04. T runs; verdict ~16:10.
- Sat 14 16:0x: **TAPSQ2 = KILL with a valid control** (BigMoE 068dc9b).
  - DFlash2's fused multi-layer taps at the context / anchor rows add noise, not information: MBPP 23-30 71.43 vs C
    72.06, KL 1.083x, windows 35 better / 37 worse.
  - A richer view of the verified context doesn't reach the deep rows, which fits Law 2.
- Sat 14 16:1x: the next screens, freshness first (orchestrator's call):
  1. FRESH-H (no re-prefill, after CAPQ): the true fused h fed at every drafted row on the correct chain, tq2_C as the
     control, MBPP 23-30. It goes on if it recovers >= half the gap toward MBPP's lag-2 93.5.
  2. Then a re-prefill with raw taps on every row: per-layer freshness (what a cheap partial trunk pass buys), plus the
     anchor raw-taps item from the same data.
- Sat 14 16:31: **CAPQ = CLOSED** (clean capacity retest). The x2 branch at lr 1e-5 has a clean start (steps 1-20 KL
  1.024x t30b, against the 10x-lr arm's +203 %). Over steps 101-200 its KL is 1.016x t30b's, lower on 46 % (REOPENS
  needed <= 0.95 on >= 60 %). Capacity is closed for the week. The lr confound explained the spikes, not the result.
- Sat 14 16:44: **FRESH-H = FLAT**. The true fused h at every drafted row through a zero-init additive adapter (lr
  1e-5, 80 steps) gives MBPP 23-30 70.09 vs C 72.06 (-1.97). WORKS needed 82.78.
  - Read narrowly: the short adapter didn't learn to use a new input. Law 2's 93.5 % used the reader's native trunk
    state (wd).
  - Next (orchestrator -> BigMoE, no head training):
    1. a TRUE-WD replacement eval, which should reproduce Law 2 in our harness;
    2. a ridge map h -> wd (and later layer-L taps -> wd) fed as a replacement, giving the per-layer freshness curve
       without adapter training.
- Sat 14 16:52: **LEAD: REPLACE-H = CARRIES** (BigMoE cc53cb5, t30b unchanged, no training, MBPP 100).
  - At 23-30, t30b's own chain gives 72.13. With the stand-in replaced by M(true h), a ridge from DFlash2's fused h to
    wd (R2 0.80), it gives **91.20**. With the true wd (the gate, which reproduces Law 2) it gives 93.63.
  - The M(h) bands are flat with depth: 94.01 / 91.78 / 91.60 / 91.46 / 91.34 / 91.28 / 91.20. Kept per 32 goes
    6.92 -> 10.55 (oracle 12.76).
  - Freshness is the lever, confirmed. Open: its cost, since h needs trunk layers up to 45 per drafted token.
  - Next: the re-prefill (raw taps) gives the per-layer freshness curve ({5}, {5,15}, {5,15,25}, all) (BigMoE).
    Leap prices a partial trunk pass per L at 45 / 24 GB, with the break-even kept per check.
- Sat 14 17:01: **Leap's trunk pricing** (5bc983d, data/week98/README.md, trunk_price.txt).
  - A fresh partial trunk pass PER drafted token doesn't pay: ~0.62 ms per layer even for 1 row (CPU graph build).
    45 GB L 5 adds ~101 ms per check and needs kept 27.8 HE+ / 13.9 MBPP to break even; L >= 15 is impossible.
  - BLOCK refresh does: a k-row pass costs about a 1-row pass. At 45 GB, k = 8: L5 +11 ms (break-even 15.3 / 7.7),
    L15 +27 ms (17.6 / 8.8), L25 +43 ms (19.8 / 9.9), L45 +76 ms (24.3 / 12.2). L45 at k = 16 adds +23 ms (17.0 / 8.5).
  - Side note: t30b at a 32-row check (~100 ms at 45 GB) gives ~138 / 69 tok/s, below the shipped 16-row lossy-45 loop's
    164 / 113. The deep drafter also needs the cheap check.
  - Next: the (L, k) block-refresh sweep in the REPLACE-H harness after LAYERS. The cheapest (L, k) over its break-even
    is the method's cost-feasible point.
- Sat 14 17:46: **LAYERS = COSTLY** (BigMoE a856e7e).
  - The gate is valid: the re-prefilled stream holds 93.64 at MBPP 23-30.
  - Ridge(taps) -> stream, replacing the stand-in, MBPP 23-30:
    - {5} 41.66 (R2 0.41);
    - {5,15} 47.47;
    - {5,15,25} 56.14 (R2 0.51);
    - {45} 91.61 (R2 0.82);
    - all five 91.52.
  - Early-layer maps are WORSE than the stale chain (72.13): a poor reconstruction is worse than the head's own guess.
  - The fresh state lives in the deep trunk, so a usable refresh is close to a full pass. Refreshing at L45 every k
    tokens is nearly the same as VERIFYING every k tokens.
  - BLOCK (L x k) is due ~18:15: does L45 at k = 16 clear 8.5 kept (45 GB)?
- Sat 14 17:5x: **Leap closes trunk refresh as a speed lever** (fce1ca7).
  - At L45, a refresh is a verify minus the bonus token and the early exit. MBPP at 45 GB: refreshing every 16 tops
    out at ~89 tok/s (every 8 at ~62), both at the per-row bound of 10.98 kept. Plain 16-row checks with t30b give ~96,
    and the shipped loop 112.7. Tying would need kept >= ~11.8.
  - So the 98 % method must make the drafter predict the deep state itself.
  - Leap's cheapest next screen (sent to BigMoE): ridge [the carried stand-in, the fresh tap {5}] -> wd. Refreshing L5
    every 8 costs +11 ms (break-even MBPP 7.7).
- Sat 14 18:0x: **COMBO5 = CLOSED** (BigMoE 7d6e935). Freshness from the trunk is closed for the week.
  - Ridge [carried stand-in, tap 5] -> wd: R2 0.448. Refreshing every 8 gives MBPP 23-30 65.85, kept/32 6.50
    (break-even 7.7); every depth 25.54.
  - t30b's exact kept per 16-row check is 6.59, so Leap's L45 verdict stands.
  - The head's carried stand-in, read as a wd estimate, has R2 -2.32. It lives in the head's own space, yet the head
    drafts 72 % at depth from it.
- **The week's method, from here: a learned better stand-in.** The drafter predicts the deep state itself. The
  oracle (a fresh wd, 93.6 % flat with depth) shows the head can use a good state.
  - The BANK (runtime-path taps + wd + top-32) and the PROXY (R2 -> predicted 23-30) are due ~22:00 (BigMoE).
  - The proxy filters only wd-space predictors. Learned stand-ins in the head's own space are judged by the drafting
    eval.
- Sat 14 18:03: Leap recorded the trunk-refresh closure (4448350) and withdrew its microbench. From 18:0x Leap has
  route 3, the cheap check: cutting the per-check graph-build overhead (0.62 ms per layer even at 1 row). Exact stays
  bit-exact; GPU timings after the BANK or overnight.
- Sat 14 18:1x: **the BANK is building** (BigMoE b889c90, 524 files, end ~20:05).
  - The runtime path reproduces the n1112 and MBPP captures bit for bit (prompt AND answer rows).
  - HumanEval+ warning: TrainLab's d16w windows do NOT match the runtime path (stored wd differs 10 % on prompt rows,
    21 % on answer rows). Every HE+ number so far (t30b 89.66 at 23-30, all TTT30C HE+ lines) came from d16w features.
    After the build, BigMoE re-scores t30b and b5 on HE+ with bank (runtime-path) features. Quote HE+ headlines with
    that caveat until then.
  - PROXY: bank_proxy.py (3e462ea). Its calibration is redone on the bank after the build.
- Sat 14 18:1x: **Leap cheap check, screen 1** (518732e, data/week98/cheapcheck.md).
  - The 16/32-row verify is GPU-bound (check ~ 30 + 2.47 ms per row at 45 GB), so graph caching saves ~0. Closed.
  - The 1-row 0.62 ms per layer was GPU time; the CPU build is ~0.21 ms per layer.
  - Exact bandwidth floors at 45 GB: 15.9 ms (1 row), 36.5 (16), 49.1 (32). A 32-row check today is ~90-97 ms
    (estimated).
  - The exact path to ~60 ms at 32 rows is GPU kernels: experts to floor, dense ~20 -> 13-14 ms. The ~23 ms target is
    lossy-only (FAST track).
  - **Rule of thumb:** a deep drafter beats plain 16-row checks only when kept/32 >= ~1.35-1.4 x kept/16. t30b on MBPP
    is at 1.05; a 98 %-per-guess drafter would be at 1.69. The drafter decides the 32-row question; a faster check
    speeds both loops.
  - Next: Leap measures a real 32-row check (dense / experts / misses) after the BANK, under a quiet lock (~20:30).
  - **Engine fact (Leap 18:3x): no exact single-pass 32-row check exists today.**
    - The sync-free gate carries at most 185 ids per routing row, i.e. 18 rows at top-10 (bigmoe/gated.py MAXK <= 185).
      Above 16 rows a check falls to the slot / prefill path (4-bit on every rank, outside REF2-T's certified 1-16 rows).
    - So an exact 32-row check today is 2 x 16 gated passes, about 2x a 16-row check. No drafter pays at 2x: 98 % gives
      1.69x.
    - The ~60 ms exact 32-row check and the 1.35-1.4x rule both assume a gate widened to 320 ids in one pass. That is
      item 1 of BigMoE's 32-row engine spec, ahead of the kernel work.
    - Tonight's screen times 16 rows, 2 x 16 and 1 x 32 (slot), cold and warm, plus a dense-only pass, on exact-45 via
      the R57_COPYPROBE hook (the loop stays byte-identical).
- Sat 14 18:5x: **method catch (Sandbox98, results/lr_screen2.md):** the accept-until-fail training loss is
  survival-weighted, so a better-chaining arm can show a HIGHER loss. Judge arms by read-outs (bands, kept/32) or
  per-depth KL on fixed rows, never by cross-arm AUF loss.
  - **CAPQ's CLOSED rested on the loss ratio alone, so it is not certain.** A read-out of t30x2c@200 vs a matched
    t30b@200 on MBPP 100 is queued with BigMoE after the BANK and Leap's screen.
  - The stop-line rule in RULES.md is fixed to use a quick read-out.
  - Sandbox98 status at 19:00: Qwen3.5-2B + its native MTP head (121.7 MB fetched). lr screen -> base lr 1e-5 (native
    kept/32 3.18, 23-30 only 1.7 %). The base drafter is training; the gate evals follow.
- Sat 14 19:51: **Sandbox98 validity gate PASS** (~/Sandbox98/results/gate_report.md). Qwen3.5-2B with its MTP head
  trained 30-deep (s30) reproduces the 125B's day-1 pattern.
  - MBPP 23-30: chain 57.32 vs depth 0 88.54; true W at every depth 88.35, flat. HE: chain 70.31; true W 93.71.
  - Maps from early taps are poor (L2 47.75, L12 59.19); late taps are good (L22 87.30, R2 0.93).
  - **New:** the head tolerates UNBIASED noise far better than map error at the same R2. True W + noise at R2 ~0.75
    gives 85.60, at ~0.51 gives 82.66; a ridge map at R2 0.44 gives 59.19.
  - So a learned stand-in needs noise-like errors, not exactness. The PROXY's R2 calibration must be per predictor
    family.
  - Screens now run on the sandbox (control + the t30f feature-loss transfer first), minutes each.
- Sat 14 23:04: **T2 CONTEXT TRUNCATION (2B) = MIXED, and a DIRECTION** (Sandbox98 ed32994; MBPP-50, 3,857 anchors,
  s30 unchanged).
  - 23-30: chain 55.42 | the 2B over ONLY the drafted tokens (K = 0) 69.62 | K = 16 75.47 | K = 64 80.61 | K = 256
    86.25 | full 87.71.
  - Band 1-2: chain 78.1 | K = 0 17.5 | K = 64 79.4 | K = 256 86.8 | full 88.2.
  - **Theory: the deep gap is COMPOSITION DEPTH.** The head already attends to every true prefix state, yet its ONE
    layer composes the drafted tokens far worse than a 24-layer stack with no prefix at all (+14 pt at 23-30). Context
    adds ~+5 pt per 4x window; the knee is 64-256 tokens. Retrieval of the far prefix matters little (K = 256 is -1.5).
  - On the 125B, capacity tests added WIDTH (no effect / hurts). DEPTH is untested.
  - **Next: a DEEPER drafter** (Sandbox98, ~1 h): the head's layer applied 2x (tied), then 2-3 untied zero-init residual
    layers, TTT on pool2 re-annealed, against an in-batch control. Also time one drafter step at 1 / 2 / 3 layers.
- Sat 14 22:5x: **ROLLOUT on the 2B = KILL** (Sandbox98 053eeaf), transfer match #5. Linear 20.86, +MLP 22.32, vs
  chain 57.32 (stale 15.50, oracle 88.35). Even one step loses to the chain (66.8 vs 78.2 at band 1-2).
  - **Theory:** a "no change" step has direction error 1.005 there, so consecutive top states are nearly orthogonal. The
    next-token state is RECOMPUTED from context at every step, not a slowly drifting summary. No context-free stepper
    can track it.
  - Freshness must come from a context-aware computation: the drafter's own attention chain trained better (distill =
    the provisional lead), cheap lookups (T1), or the target's prefix KV.
  - Deep accuracy tends toward the drafter's standalone agreement with the target given true context.
- Sat 14 22:4x: **ROLLOUT = PARTIAL** (BigMoE a8adf45; MBPP 23-30, t30b with the stand-in replaced).
  - STALE anchor (wd_s at every depth): 16.00 (1-2: 54.08, 3-6: 25.15), kept 2.87.
  - A linear PCA-512 transition rolled out from the true anchor: 23.08, kept 2.97.
  - The chain gives 72.13; the oracle 93.63.
  - Direction error: rollout 0.59 (1 step) -> 0.79 (30), stale 0.85 -> 1.04. k = 1024 doesn't help, so the linear step
    is the limit.
  - Read: composition of the new tokens is most of what depth needs. The head's own learned carry (72 %) is far better
    than linear external dynamics (23 %). The 72 -> 94 gap is what the carry still misses. T1 / T3 / T2 decide whether
    that is retrieval, composition depth or dimension.
> (a quote from the project owner, kept private)
  achieve in principle what we are chasing, not lo0ng tests".
  - The sandbox's long-screen control drifted (-7.2 pt; overfitting a 120-answer pool), so its long screens are
    provisional and stopped.
  - **ROLLOUT** (125B on the BANK: BigMoE; 2B: Sandbox98, as a transfer pair; no head training): a learned
    state-transition f(W_t, emb(tok_{t+1})) -> W_{t+1} (linear PCA ridge, then a small MLP) rolled out from the fresh
    anchor along the drafted tokens, then REPLACE eval.
  - Baselines: the chain, the oracle, and a stale anchor. +5 pt over the chain at 23-30 = the principle works.
- Sat 14 22:1x: sandbox -> 125B transfer: 4 matches so far (the depth drop, the flat fresh-state oracle, early vs late
  maps, feat NO EFFECT). Transfer checks asked for (TRANSFER.md): capacity x2 at the head lr (the 125B's depth-growing
  +1.2 with a worse loss), and anchor fused taps.
  - Parallel side-by-side runs buy little: one sandbox process keeps the GPU ~84 % busy. Use multi-arm runs.
  - Rule: sandbox wins are filters. Confirm on the 125B before trusting them.
- Sat 14 22:0x: **CAPQ read-out** (BigMoE 24e7758): CLOSED stands by its line, but the loss-only reading was wrong in
  kind.
  - t30x2c@200 vs t30bc@200 on MBPP 100, by band: 0 -0.09, 3-6 +0.20, 7-10 +0.44, 11-14 +0.85, 15-22 +1.00, 23-30 +1.20
    (58.18 -> 59.39). Kept/32 +0.043; REOPENS needed +1.0 AND +0.1.
  - The AUF loss read 1.6 % WORSE while the deep read-out is better and the gain grows with depth: survival weighting,
    confirmed.
  - Capacity buys a small depth-growing gain (~+1 pt at 23-30 in 200 steps). It's tiny next to the oracle (+21.5 MBPP /
    +8.1 HE+); not the week's bet, though it may compound with stand-in work.
  - Harness: re-running t30b's first 200 steps today doesn't reproduce its saved losses bitwise (max abs 0.143). Paired
    arms must run on the same code and day.
  - BigMoE's night chain is done (BANK + PROXY, routing Step 0 to Leap 64c9d2d, CAPQ). The capture stays stopped.
- Sat 14 21:33: **Sandbox98's three learned-stand-in screens** (36e531d; 300 steps at lr 2e-6; MBPP-50 / HE-30 23-30
  against a control that held, 55.96 / 68.34):
  - distill (KL to the head's own true-state output): -0.13 / -0.15, nothing moved;
  - headfeat (stand-in pulled to the true state in head space): -1.50 (p 0.013) / -0.45, a real small LOSS;
  - noisehead: n.s. at depth; depth 0 +0.29 (p < 0.05).
  - **Budget caveat:** the control itself moved only +0.5 pt from s30 in 300 steps, so distill is UNTESTED, not
    refuted. Rule: training levers need a budget at which the control measurably moves.
  - Next (now, Sandbox98): re-train s30 with its Adam state saved, then 1,000-step screens continuing that state at the
    base lr (distill first).
- Sat 14 21:2x: **Leap screen 3, lossy-24 part** (4d9322e, data/week98/cc3/README.md; High Power Mode; within-run).
  - The live 13-row check is 81.5 ms. Its split at 13 rows: dense 23.9 + experts/misses 53.7 (130 misses, 23.9 ms SSD).
    Dense at 13 rows is CPU-bound (18 of 23 ms is graph build).
  - Dense-only 32 rows: 181 ms. CORRECTED (Leap 21:3x, 76edc07): this is lossy-24's DENSE-N-RAM unpacking every dense
    weight per call above 16 rows (counter: 2592 = 24 passes x 108 weights), not GDN / attention. Fix candidate: in
    verify scope, run 17-32 rows as two 16-row native calls (exact per row, no unpack); not built until the 24 GB gate
    carries 32 rows. exact-45 (no DNR) will show the real GDN / attention cost above 16 rows. A real 32-row pass on the
    slot path: 2.81 s, mostly graph build.
  - At 24 GB a 32-row check costs 34.5x a 13-row one. Even with a widened gate, the non-expert 32-row path alone is
    2.2x a whole 13-row check. The 24 GB line keeps <= 13-row checks.
  - Deep drafting needs real engine work for 17-32 rows: (1) a gate widened to 320 ids, (2) a GDN / attention path for
    17-32 rows (now a time cost, not only memory), (3) kernels. That's next week's build, BigMoE's spec.
  - exact-45 waits for 47 GB until 01:00.
- Sat 14 21:1x: **BANK + PROXY ready** (BigMoE 5786e5e; usage in ~/BigMoE/data/deepdraft/bank/PROXY.md).
  - Ridge calibration, MBPP 23-30 (real REPLACE): tap5 46.11, 5+15 51.16, 5+15+25 59.69, tap45 91.96, all 91.88,
    oracle 93.63. The direction error is the primary measure.
  - **HE+ on the runtime path (the headline set):**
    - t30b's own chain gives 89.65 at 23-30, so the headline holds (d16w gave 89.66).
    - **The true stream at every drafted row gives 97.95 at depth 0 and 97.76 at 23-30, FLAT; kept/32 goes 13.77 ->
      20.90.**
    - The tap45 ridge gives 96.40, kept 18.06.
  - **So the WEEK98 target on HE+ is concrete:** a learned stand-in that reaches the fresh-state oracle (~97.8 % flat),
    plus a slightly better reader (+0.2-0.3 pt), meets >= 98 % per guess at depth 30. The tie ceiling is ~98.8 %.
  - Caveat: HE+ ground truth is off the runtime path in 11 of 24 windows (0.3-2.7 % of rows).
- Sat 14 21:06: **Leap screen 3 got no numbers.**
  - lossy-24 exited rc 1 at start-up: the 24 GB loop refuses R57_ROWCAP with R57_WARMSPLIT. Leap's bug; the runner had
    been tested only with fakes.
  - exact-45 was skipped on its gate: NO-SWAP 34 vs 47 GB. The rest is anon ~29 GB of apps and sessions (Safari
    WebContent ~6 GB, PyCharm 2 GB, idle Claude sessions).
  - Earlier, the quiet ticket stalled 17 min (20:40-20:58) behind BigMoE's yielded-but-held share lock.
  - Fix: a 24 GB probe redesign, a dry run against both real loops, b24 queued after CALIBRATION-DONE, and a45 only when
    NO-SWAP >= 47 on two reads (window to 01:00, else tomorrow night).
- Sat 14 20:26: Sandbox98 tapmlp (item 1, a zero-init MLP from the true tap-12 residual added to the stand-in) = NO
  EFFECT by its +2 pt line. Small real gain: MBPP-50 23-30 +0.61 pt (p 0.002), HE-30 +0.82 (p 0.016). Tap-17 is a
  breadth item: it's the trunk-refresh route, closed on cost.
  - Sent to Sandbox98 for tonight (its lines): (1) ORACLE DISTILLATION, where the true-W drafter is the teacher and the
    chain's head output is matched to it at every depth; (2) a state target in the head's fused input space; (3) a
    noise-robust head + stand-in predictor.
- Sat 14 20:21:25: **BANK-DONE**: all 524 files built, rc 0. Every n1112 / MBPP file is bit-exact against the source;
  the 24 HE+ windows are on the runtime path (they differ from d16w by design). Calibration started 20:21:46 (proxy
  rows: 31,805 train / 9,984 val). The fans ramp back to 3750 from 20:21:42.
- Sat 14 20:13: **Sandbox98 ready** (git 6cb92ae; `./screen.sh NAME LEVER "LINE"`, 5-min paired screens, about +/-0.5 pt
  resolution).
  - First transfer check: feat (t30f's feature loss) = NO EFFECT, as on the 125B (MBPP 23-30 -0.26 pt, CI -0.71..+0.16).
  - Findings:
    1. The RMS-normalized direction error, not R2, orders all 7 predictors by read-out. It becomes the PROXY's
       primary measure.
    2. The AUF loss bias (above).
    3. A converged head continued at 10x its final lr with a fresh Adam drifts the control 8 pt in 300 steps (2e-6
       holds). Every screen checks its control.
    4. An exact two-pass TTT gradient cuts the MLX peak from 5.8-7.8 to 2.5-3.4 GB. It may shrink the 125B's 24-39 GB
       TTT runs; a 15-min screen for BigMoE / TrainLab.
  - Ready: levers/tapmlp.py (item 1, a nonlinear early-tap map).
  - To do (orchestrator): add ~/Sandbox98 to fanwatch's RES (needs a restart), and a private GitHub backup (asking
    Kamen).
- Sat 14 19:16: the BANK was restarted (fresh process, C 160, resuming at file 271), still bit-exact. Kamen turned on
  High Power Mode (~19:15). Pace ~25-30 ms/token, ETA ~20:40. No heat pauses 19:00-20:00.
- Sat 14 19:00: the BANK has 197 of 524 files, all bit-exact except the 24 HumanEval+ windows (expected, since d16w ≠ the
  runtime path). Greedy holds 97.3-100 % there. ETA ~21:00 (the sandbox shares the GPU). Swap 7.15 GB (start 6.51).
> (a quote from the project owner, kept private)
  second-model generality proof. It starts only if he starts it.
- **What that leaves for the week** (for NewIdeas / MTP Head at 01:00, as 15-min screens):
  1. A nonlinear or trained map from early taps to the deep stream. Linear explains only 41-51 %. If an MLP from
     <= 25 layers reaches R2 >= ~0.8 and REPLACE >= ~88, a half-trunk refresh is back in play. Taps for 100 train +
     MBPP 100 files are already re-prefilled (BigMoE).
  2. A learned refresher: train the drafter's own stand-in to track the true deep state (a feature loss against the
     true wd at depth), judged by the same REPLACE-style held-out eval. t30f's feature term was NO EFFECT, so find out
     why first (its weight, its target layer, which depths).
  3. The cheap check (Leap): a 32-row verify at ~100 ms (45 GB) is CPU-graph-bound (0.62 ms per layer even at 1 row).
     A compiled graph is the optimistic bound (0.39 ms per layer). A cheaper check makes frequent verification the
     refresh.
  4. Prior art for "a drafter that predicts the target's future hidden states" (NewIdeas' prior-art pass).
> (a quote from the project owner, kept private)
  generality added (orchestrator).
