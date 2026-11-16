# IDEAS_1116c_drift: drafting shapes and training against DRIFT at depths 3-30 (Fable FAR, Mon 16 Nov 2026 18:2x-19:0x, desk)

THINK_1116c, FAR's lens. Marks: [K] measured (source), [P] arithmetic on [K], [G] judgement. Metric rule from the record
[K WEEK98:1968]: kept/32 or the STRICT conditional per depth decides; bands are information. For tree levers the
right kept is the TREE's (Sandbox98 `sb98/tree8.py`, the 127-node dumps on the 125B), because trees reach the walk
regime (a node after a non-top-1 true child) that chains never do: 66-86 % of K64-128 nodes sit off the #1 spine [K S11].

## 0. What the record says drift IS, and which rows matter
- The head is the native MTP layer chained: at depth j its input is its OWN previous output (the stand-in for the trunk's
  state) + the next token's embedding; it attends to the trunk's TRUE (k, v) pairs of the prefix and to its own chain
  entries of depths 2..j [K sb/core.py:9, 206]. Drift = the stand-in and the own entries drifting away from what the
  trunk would compute; the direction error is 0.85-0.99 by depth [K], i.e. the stand-in is nearly orthogonal to the true
  state: the head is not estimating the trunk's state, it is carrying a token-prediction summary.
- The deep-drift fixes "mostly fix rows the engine never reaches" for CHAINS [K WEEK98:1962]; for TREES the walk rows at
  depths 3-14 are reached and read 93-96 % [K brief]. So a drift lever for 775 is judged on tree kept and strict 3-14.
- Measured exact deep fixes: the frozen 2B audit (walk-strict 15-30 98.3, kept +8.9 % at P = 2 [K PLAN900:54]);
  tapmlp (a mid-trunk tap over the drafted rows: +4.5-4.8 pt at 23-30, HE walk-strict 7-14 86.3 -> 88.4 [K WEEK98:141],
  CLOSED on the 125B's cost). State-side training levers (distill, noisehead, carryac, capacity, zero-start) <= +0.8 pt.

## 1. WALK-MIX TTT: keep accept-until-fail for the survivors, add the post-miss rows at weight w, judge on TREE kept  [tried as ALLROWS: NO EFFECT on chain kept; NEW metric + mix]
**Mechanism.** t30b's loss drops a row from the chain at its first wrong argmax (accept-until-fail), so depths 15-30
train only on chains that survived: the easy windows. The rows after a miss (own state, true token = exactly the walk
regime trees reach) get no gradient. ALLROWS (every row at every depth) lifted the deep walk bands +5.8 / +8.6 pt (HE /
MBPP 23-30, 400 steps on the 2B) and HE strict d18-24 by ~3-5 pt, at the price of the shallow strict d1-d4 (-0.4 to
-0.8 pt) and no chain-kept change, which killed it for chains [K WEEK98:1968-1978, 1988]. The record itself says "allrows,
or a mix with accept-until-fail, may pay once chains survive deep". Trees ARE that case: their nodes are mostly walk rows.
The mix: survivors at weight 1 (today's loss), post-miss rows at w = 0.3 (then 0.5), so the shallow survivors keep their
gradient share and the walk rows get some. Why it attacks drift: the post-miss rows are the only training rows where the
carried state is already wrong-ish and the head must still predict from it; today it never learns that regime.
**Number.** Bound from ALLROWS: walk bands +5.8 / +8.6 pt deep, strict d18-24 +3-5 pt [K]. In tree terms the gain is
on the off-spine nodes at depths 3-14 (93-96 % today): if the mix moves them +2 pt without the shallow loss, K32 kept
rises by roughly the share of off-spine kept tokens x 2 % / (1 - 0.95) ~ +5-8 % [P, rough]; the screen decides.
**Screen (Sandbox98, <= 30 min, <= 5 GB).** `sb/train_multi.py --arms ctl:-,mix3:walkmix@0.3,mix5:walkmix@0.5 --depth 30
--steps 400` (lever: `ttt_loss` keeps failed rows in the chain with weight w instead of dropping them; a few lines at
core.py:398), read out with `sb/evalx.py` strict per depth AND `sb98/tree8.py` tree-32 / tree-127 kept. PASS: tree-32
kept >= +3 % with strict d1-d4 within -0.2 pt of ctl. KILL: shallow strict loss > 0.5 pt or tree kept flat.
**Cost per block on the 125B:** 0 (the same head). Train by Thursday: a loss flag on the t30b recipe (one 8 h pass).
**Quality:** exact (a drafter change).

## 2. TOKEN-KEYED CONTEXT: the head's own chain entries come from the drafted TOKENS, not its drifting states  [new]
**Mechanism.** At depth j the head attends to its own entries of depths 2..j-1, whose keys / values were computed from
its own stand-ins (the drifting objects). Replace those keys / values with projections of the drafted tokens' embeddings
(+ position): the head still carries its state for the QUERY (so it is not DFlash's block form, which dropped the carried
state entirely and was far worse, tau 3.5-4.2 [K]), but the CONTEXT it reads over the drafted span is what was drafted
(certain) rather than what it imagined (uncertain). Borrowed from data assimilation: assimilate observations (tokens), not
the model's own forecast. Why it attacks drift: the error compounds through two paths, the stand-in and the own entries;
this closes the second path. T2's truncation result says the drafted TOKENS with the prefix carry most of what a fresh
pass needs (2B over the drafted tokens + 256 context 86.3 vs full 87.7 [K Sandbox98 T2]).
**Number.** Bound: between the chain (MBPP 23-30 57.3 on the 2B) and the fresh-state oracle (88.4); the own-entries
path's share is unknown. A desk guess is a quarter of the gap; the screen decides.
**Screen (Sandbox98, <= 30 min).** Lever `tokkv`: in core.py's chain step (293-299), compute k, v for the chain entry
from `embed(ids[anc + j]) + depth_pos` through the same k_proj / v_proj (a tiny trained adapter in front, init identity-
like), queries unchanged; 400 steps vs ctl; strict per depth + tree-32 kept. PASS: strict 7-30 >= +2 pt with d1-d4 within
-0.2; KILL: d1-d4 loss > 0.5 or no deep gain.
**Cost per block:** slightly BELOW today's (the chain entries need no stand-in pass; same bytes). Exact.

## 3. DEPTH-AWARE HEAD (a Kalman-gain analog): tell the head how stale its state is  [new, cheap, a long shot]
**Mechanism.** The head at depth 30 has no input saying it is 30 steps from a true state; its attention weighs its own
entries like the trunk's true pairs. Add (a) a depth embedding to the stand-in input, (b) per-KV-head logit biases on
own-chain keys by depth bucket (1-2 / 3-6 / 7-14 / 15-30) and a value gain, trained with the TTT loss (a few hundred
parameters). A state estimator weights evidence by its reliability; here the reliability is a known function of depth.
**Number.** Bound: the state-side levers gave <= +0.8 pt [K]; this one adds information the head lacks (depth), so the
bound is not theirs, but a desk guess is +1-2 pt at 15-30, less at 3-14. **Screen:** lever `depthgate`, 400 steps, strict
+ tree kept; PASS >= +1.5 pt strict 7-30, KILL < +0.5. **Cost:** 0. Exact.

## 4. The audit priced in the 775 frame (not mine; the one measured exact deep fix)  [K]
The frozen 2B audit: walk-strict 15-30 98.3, kept +8.9 % at P = 2 [K PLAN900:54]; tree + audit was x0.991-1.016 at
today's cycle [K WEEK98:720]. In the 775 frame (check 25 + draft 3): a 2B pass at 4-bit is ~1.2 GB = ~4.3 ms [P];
P = 1 gives kept ~x1.06-1.09 for a cycle x1.15 = a loss; P = 2 worse. It pays only with <= ~1.5 ms exposed: a ~0.4 GB
auditor (2-bit 2B, or a distilled ~0.6B at >= 95 % agreement, the record's "full-data distilled auditor" item) or the
pass hidden under the QAT head's dispatch floor (0.31 ms x 20-40 levels = 6-12 ms per block of idle bandwidth), which
needs the audit of rows 1-16 to run while rows 17-32 draft and a re-draft from any repair (~10-15 % of blocks). Leap /
NewIdeas' call; I only state the line: the audit needs its bytes under ~0.4 GB or its pass under the draft to pay at 775.

## 5. Not proposed
- tapmlp on the 125B: a 12-of-48-layer trunk pass per drafted token is ~1/4 of a check's bytes PER LEVEL: KILL on cost
  (CLOSED 18:1x). A per-BLOCK partial pass over the drafted rows is the audit's shape with the trunk's own layers and the
  same byte problem (0.86 GB dense + ~1 GB experts for 12 layers = ~6.5 ms).
- ALLROWS as it was (judged on chain kept): the record's NO EFFECT stands; idea 1 is the mix on the tree metric.
- Any state-space map (ROLLOUT KILL, COMPOSER KILL), noise / distill / capacity (<= +0.8 pt), zero-start, SIB.

## Summary line
Best: WALK-MIX TTT, survivors at weight 1 and the post-miss rows at 0.3-0.5, judged on TREE kept (trees reach the walk
rows that ALLROWS lifted +5.8 / +8.6 pt deep while chain kept stayed flat); 0 cost per block, a loss flag by Thursday;
Sandbox98 screen: train_multi ctl / mix0.3 / mix0.5, 400 steps, PASS = tree-32 kept +3 % with strict d1-d4 within
-0.2. Then TOKEN-KEYED CONTEXT (own chain entries from the drafted tokens, carried state only for the query) and the
DEPTH-AWARE head, both 0 cost; the audit pays at 775 only under ~0.4 GB or hidden under the draft's dispatch floor.
