> (a quote from the project owner, kept private)

The transfer pair of BigMoE's 125B bank test. No head training: the drafter is s30, unchanged.

**Question:** can a cheap learned model of the target's top state, stepped along the drafted tokens, stand in for the true
deep state well enough to beat the head's own carried state at depth?

**Transition model** f: W_{t+1} ~ f(W_t, emb(tok_{t+1})), in the head's input space (the post-norm final hidden, 2048 wide).
- (a) linear: ridge on [W_t, emb(tok_{t+1})] (4096 -> 2048) in the full space (at 2048 wide no PCA is needed; on the 125B
  PCA keeps 10240-wide states tractable), lambda picked on held-out rows by direction error;
- (b) (a) plus a small residual MLP (4096 -> 2048 -> 2048, zero-init output), trained on one-step pairs with a direction
  loss.
- Each predicted row is rescaled to the training rows' mean norm before the next step (post-norm states have a near-
  constant norm), so the rollout can't drift in scale.
- Fit on the answer rows of the 380 base answers. Held out: the 40 val answers (one step), MBPP 100 (rollout and REPLACE).

**Reports**
- One-step held-out direction error (RMS-normalized relative error, proxy_check.txt's measure).
- Rollout direction error by depth k = 1..30, from each fresh anchor W_s along the true (= drafted on a correct chain)
  tokens.
- REPLACE eval on MBPP 100 with s30 unchanged: at depth j the stand-in is the rollout's prediction of W_{s+j-1}.
  Baselines on the same 100 answers: the chain 57.32, the oracle (true W_{s+j-1}) 88.35, and a stale anchor (the true
  W_s at every depth: fresh at the anchor, never advanced).

**Lines** (MBPP 100, band 23-30):
- rollout >= chain + 5 pt (>= 62.32) -> **the principle works**: a predicted deep state beats the head's own carried
  state. Next: put f (or its training signal) into the drafter, and run the 125B pair.
- between the chain and +5 pt -> partial: the direction-error curve shows where it breaks.
- <= the chain -> f is not good enough at this size. The one-step and rollout errors, read on the proxy curve
  (direction error 0.23 -> 87.4 ... 0.54 -> 82.7, 0.78 -> 59.2, 0.84 -> 47.8), say whether a better f could pass.
- Side check: does the rollout's direction error per depth predict its band accuracy on that curve?

## Result (Sat 14 Nov 22:49): the principle FAILS this way. A Markov top-state model loses to the head's own chain at every depth
| stand-in at depth j (s30 unchanged, MBPP 100) | 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | kept/32 |
|---|---|---|---|---|---|---|---|---|
| s30's own carried stand-in (chain) | 88.54 | 78.23 | 69.22 | 65.40 | 63.39 | 60.39 | **57.32** | 4.42 |
| true W_{s+j-1} (oracle) | 88.54 | 88.45 | 88.35 | 88.39 | 88.34 | 88.26 | **88.35** | 8.05 |
| stale anchor: true W_s at every depth | 88.54 | 46.70 | 23.52 | 17.54 | 17.05 | 17.60 | **15.50** | 2.56 |
| rollout, linear f | 88.54 | 66.80 | 48.53 | 39.63 | 33.84 | 27.51 | **20.86** | 3.21 |
| rollout, linear f + residual MLP | 88.54 | 66.61 | 48.89 | 40.23 | 34.60 | 28.69 | **22.32** | 3.20 |
- f's one-step held-out direction error: linear 0.695, + MLP 0.689; "no change" (W_{t+1} = W_t) 1.005 (results/rollout_fit.json).
- Rollout direction error by steps k (MBPP 100): linear 1: 0.674, 2: 0.793, 4: 0.886, 8: 0.962, 12: 0.999, 22: 1.060,
  30: 1.096; + MLP 1: 0.669 ... 30: 1.084 (results/rollout_curve.json). By ~12 steps it's no closer than a stale state.
- **Line: <= the chain -> f is not good enough at this size**: 23-30 at 20.86 / 22.32 vs 57.32. Even one step
  (band 1-2: 66.8) is below the chain (78.2), so a better f of this kind, (W_t, token) only, can't close a 35-pt gap.
- Proxy side check: consistent. Direction error ~0.7-0.8 -> ~67 % (band 1-2), ~1.0-1.1 -> ~21 % (23-30), monotone with
  proxy_check's curve (0.54 -> 82.7, 0.78 -> 59.2, 0.84 -> 47.8).
- **What it points to:** the top state can't be stepped forward without the context. Consecutive states are nearly
  unrelated in direction (1.005), and (W_t, next token) recovers only part (0.69). The head's chain attends to the true
  prefix and to its own entries, and is already far better (equivalent direction error ~0.78 at 23-30). A better
  stand-in has to come from a context-aware model, i.e. the drafter's own chain trained better: a teacher signal toward
  the fresh state (distill was the one provisional lead: +3.1 / +8.1 vs a drifting control).
