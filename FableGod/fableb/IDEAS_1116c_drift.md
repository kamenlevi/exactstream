# IDEAS_1116c: what the check already computes that the next block's drafting can use (Fable B, Mon 16 Nov 18:5x-19:3x)

Lens (THINK_1116c): the 125B's check pass as a free source of true-state information for the drafter. Labels [K] [P] [G]
(G halved). Records read: PLAN900 1b, deepdraft/out/drift.txt (dd_drift), Sandbox98 LEVERS_A (SIB pre-registration and
its smoke read-outs), ROLLOUT.md (direction error -> band accuracy proxy), TrainLab REPORT (the head attends to true-pair
keys plus its own chain keys; the head's input is the pre-mixer widened stream, 10,240 per position).

## 0. What the check hands over today, and what it could (the inventory)

| the check computes, per block | used by the drafter today | could be used |
|---|---|---|
| the true widened stream W at the anchor (last accepted row) | yes: depth-0 input | |
| the true W at every ACCEPTED row | the head's prefix keys ("true-pair keys" at inference) | an ERROR SIGNAL: own_d vs true at the same position, every block (idea 1) |
| **the true W at every tree row, accepted or not** = the 125B's exact state for that row's own token sequence | no (SIB used them as the head's INPUT after a miss: KILL; they sit farther from the next block's true state than the carried state, 1.0-1.1 vs 0.8-0.97) | the same error signal at ALL 32 depths per block, not only the accepted ones: the head's state at depth d was trying to be exactly this vector (idea 1, adaptive form) |
| the 125B's logits at every row (top-k, margins) | the walk, close-call rules | tie maps (keyed / stubs: KILL twice) |
| the router's expert ids per layer per row | nothing | a window signature; the head already reads W, which determines the routing: nothing new (dropped) |
| the KV / GDN caches after the accepted prefix | the next check | the head has its own cache; mid-depth taps at drafted rows are not computed (S12's wall) |
| the running window mean of W over accepted rows | nothing | the slow component of the state (idea 1) |

The one fact that shapes everything: dd_drift (12 Nov) says the token-specific part of W decays fast (centered cosine
0.53 at d1 -> 0.13 at d5 -> 0.01 at d30), while the window-level part is SLOW (raw cosine plateaus at 0.45-0.50 from
d ~ 10). And the SIB read-out says the carried state's direction error to the true W is 0.80-0.97 at depths 1-8 (HE).
A vector with cosine 0.49 to the truth has direction error sqrt(1 - 0.49^2) = 0.87. **So from depth ~5 on, the plain
window mean of the true W is as close to the true state as the head's own carried state, and at depths >= 7 it is
closer.** The head is not keeping the slow part.

## 1. MEANPULL / ERRMEM: a Kalman-style correction of the carried state from the check's true states (best)

- **Mechanism (data-assimilation's analysis increment, applied to the drafter's state).** The check gives, every
  block and for free, (a) the true W at the accepted rows and (b) the true W at every drafted row for its own token
  sequence. Keep two per-window running statistics: `mean_W` = the EMA of the true W over accepted rows (the slow
  component), and `e_d` = the EMA, per depth band, of (true_row - own_d) over the rows the check verified at depth d
  (the head's realized state error at that depth, measured on the same token sequence the head drafted; an accepted
  or rejected row both count: the row's true state IS what own_d was estimating). Then in the next block the head's
  input at depth d is
      own_d' = own_d + K_d (mean_W - own_d) + e_d
  with K_d a per-band gain (0 at depths 0-2, rising to ~0.3-0.5 by depth 7; the adaptive form sets K_d per window from
  the measured own-error energy, as adaptive inflation does). Depth 0 is untouched (it reads the true anchor).
- **Why it attacks drift, not ties:** it moves the head's state toward the component the head provably loses (the
  slow part, cosine 0.5 with the truth at every depth) and cancels the window-persistent part of its error. Ties are
  the 125B's own near-ties and are untouched.
- **Number, bounded from the records.** Own direction error at depths 3-8: 0.85-0.97 (SIB smoke, HE). The mean's: 0.87
  (dd_drift's plateau). The best convex mix of two estimators with errors 0.9 and 0.87 whose errors have
  cosine c reaches ~0.78 at c 0.5 and ~0.86 at c 0.8 (the usual two-estimator formula; c is the unknown the screen
  measures). The persistent-error term adds what the
  cross-row error cosine says is shared: SIBMIX-ORACLE measured cos(own error, sib error) 0.36-0.42 between rows of the
  same window, i.e. ~15 % of the error energy is window-shared and predictable. **Expected: direction error at depths
  3-8 from ~0.9 to ~0.80-0.85 [P from the two measured cosines].** The ROLLOUT proxy curve (direction error 0.78 ->
  59 %, 0.84 -> 48 %, 23-30 on the 2B rollouts) says 0.05-0.1 of direction error is worth several points deep; halved
  [G]: **+1.5-3 pts walk-strict at bands 3-14, G (share of the own -> true gap) ~0.15-0.3.** That is BETWEEN territory
  against the brief's "half the gap"; it is the only free, exact, zero-cost lever I can bound.
- **Screen (Sandbox98, <= 5 GB, <= 20 min; or CPU if sib_s30's cells kept the W vectors):** the SIB harness already
  produces own W and true W per depth at the post-miss cells (HE 60 / MBPP 100). Add two arms, frozen s30, no training:
  (a) MEANPULL: own_d' = own_d + K_d (mean_W - own_d), mean_W over the window's true rows before the anchor, K_d in
  {0.2, 0.35, 0.5} for d >= 3; (b) ERRMEM: + e_d from the previous checks of the same window (EMA 0.5). Read-outs: the
  direction error per depth, walk-strict 1-2 / 3-6 / 7-8, and G as SIB defined it. **PASS: G >= 0.25 over depths 3-8 on
  HE with the direction error down >= 0.05; KILL: G < 0.10 or depths 1-2 hurt by > 0.5 pt.** If the frozen head gains,
  the trained form (idea 3) is the Saturday route.
- **Cost per block on the 125B:** two 10,240-vector EMAs over <= 32 rows and one axpy per depth: < 10 us; 0 bytes
  (the check's HC streams are the W's). Quality: exact drafter-side (the check is unchanged).
- **What kills it:** the error cosine between own and the mean is high (> 0.8: the head already carries the slow part
  and loses it elsewhere); or the head's input norm removes the mean (the head reads a normed W, so a pull toward a
  mean of normed vectors must be applied before the norm; the screen must use the head's own input path).

## 2. TRUE-KEY RESYNC: make sure the head's prefix memory is rebuilt from the check's states every block

- **Mechanism.** The head attends over true-pair keys (REPORT: "depth-j queries over true-pair keys plus their own
  chain keys"). For the rows a check just accepted, the head's cache entries were computed during drafting from its
  OWN depth states. If they stay, drift leaks across blocks forever. The fix is a k-row batched head pass over the
  accepted rows with their true W after each check (the check computed them), replacing those cache entries.
- **Number.** If the loop already does this (the t30b dumps say "depth 0 at every anchor" reads the true state; the
  prefix keys are probably recomputed too), the gain is 0 and this is a confirmation. If it does not, the bound is the
  LAB vs real gap at the deep bands (97.8 vs 89.65 at 23-30), of which the within-block part stays: [G] a third of it.
- **Screen (CPU, 10 min, no GPU):** read the loop's drafting code (BigMoE tree_t30b / MTPLX's QSA path): are the
  head's keys for accepted rows recomputed from the true pairs after the check? One yes / no. If no: a Sandbox98 arm
  "own-key prefix vs true-key prefix" on the SIB cells.
- **Cost per block:** one batched head pass over <= 31 rows, ~0.3-0.5 ms [P from the per-level head cost], 0 bytes new.

## 3. The same statistics as TRAINED inputs (FAR's lane; here only priced)

- The window mean of the true W and the per-band error EMA as two extra inputs to the head (zero-initialised
  projections, D-16's recipe), trained on the captures where own and true W exist at every drafted row (the real
  127-node dumps have the true rows; the own states are the head's own forward). This is what turns idea 1's fixed
  gains into learned ones. Cost per block as idea 1; training 1-2 days; quality exact (drafter-side).

## 4. Killed at the desk (recorded so nobody re-proposes them)

- **Batched partial-trunk taps for the next block (fixing S12's infeasibility).** Draft 30 tokens, run the 125B's
  layers 0-12 over those 30 drafted rows in ONE batched tree pass (~12/48 of a check: ~10-14 ms at 45 GB [P]), re-draft
  with tapmlp from the taps. The taps are true only up to the first token the re-draft changes; past it they belong
  to the old sequence, so the gain is at most the first repaired depth, ~a third of S12's +5 % kept, for +13-20 ms per
  block (the partial pass + a second draft). KILL by cost.
- **Anchor ensemble** (draft the same positions from the anchor and from anchor-1, average): the second chain diverges
  at its first disagreement and costs a second draft; the members' errors share cos ~0.4, so the ensemble gain is
  < 10 % of the error energy. KILL.
- **Router ids as a drafter input:** the head reads W, which determines the routing. Nothing new in it.
- **Rejected-row states as the head's input** (SIB): KILL stands; idea 1 uses those states as a measurement of the
  head's error, not as its input.

## 5. Bottom line (<= 15 lines)

1. The check already computes, free, the true state for EVERY drafted row's own token sequence, not only the accepted
   ones; the head's state at depth d was estimating exactly that vector. That is a per-block, per-depth measurement of
   the drafter's state error, unused today.
2. The head loses the SLOW component of the state: from depth ~5 the plain window mean of the true W (cosine 0.49,
   direction error 0.87) is as close to the truth as the carried state (0.85-0.97).
3. Best idea: MEANPULL / ERRMEM, a Kalman-style correction own_d += K_d (mean_W - own_d) + e_d from the check's
   states; exact drafter-side, < 10 us per block, 0 bytes. Bounded gain: direction error 0.9 -> 0.80-0.85 at depths 3-8
   [P], +1.5-3 pts walk-strict at bands 3-14 [G halved], G ~0.15-0.3: BETWEEN against "half the gap", but free.
4. Screen: two frozen-head arms on Sandbox98's SIB cells (<= 20 min, <= 5 GB): PASS G >= 0.25 over depths 3-8 on HE.
5. TRUE-KEY RESYNC: confirm the head's prefix keys are rebuilt from the check's true states each block (10-min code
   read); if not, that is the cheapest drift fix on the table.
6. Nothing in my lens reaches 97.8 at bands 3-14 by Saturday; the partial-trunk tap route is dead on cost.
