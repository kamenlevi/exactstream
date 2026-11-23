# Fair depth test: 2 vs 1 layer, BOTH from the native head, pre-registered Sat 14 Nov 23:4x, before it runs
Why: the DEPTH screen added a zero-start layer late, to a converged one-layer head (deep2 HURTS -2.5), which tests "add
depth after training", not "a deeper drafter". Here both arms start from the native head with a fresh Adam, on the SAME
first 400 slices s30 trained on (base files, seed 1114), with the same schedule: 20-step warm-up to 1e-5, then a cosine to
1e-6 over 400 steps; rollout depth 30. Arms (one process): n1 (the native single layer) and n2 (+ deep2's extra layer:
identity at step 0, own Adam at 3e-5). Read-out at depth 31, MBPP-50 + HE-30; s30 (1,889 steps) beside them for scale.
Lines (n2 vs n1, MBPP-50 23-30):
- >= +2.0 pt with sign p < 0.05 -> DEPTH HELPS a drafter learn to chain: next, a full-pass 2-layer base (~40 min) vs s30,
  then a 125B depth test;
- within +-2 pt -> NO EFFECT at 400 steps from the native start;
- <= -2.0 pt with p < 0.05 -> HURTS: a 2-layer head learns to chain worse at an equal budget.

Added before the result (orchestrator 23:5x): **parameter counts and draft cost.** n1 = the native head, 60.8M trainable
(fc 8.4M, attention q 8.4M + k 1.0M + v 1.0M + o 4.2M, SwiGLU MLP 37.7M, norms). n2 = n1 + one more layer of 52.4M
(attention 14.7M + MLP 37.7M) = 113.2M, 1.86x. So this compares MORE parameters through depth, not depth at a fixed budget
(Mixture of Attentions found depth at a fixed budget hurt; FlexDraft trains 10 draft layers in from the start). Draft cost
per drafted token (timedraft, shared GPU): n1 2.54 ms, n2 3.96 ms (+56 %; lm_head + argmax is 1.23 ms of each). If depth
from the start helps: the full-pass 2-layer base (~40 min), then the 125B port goes to MTP Head (T30D spec,
~/BigMoE/HANDOVER.md). If not: write down what it says about composition, and stop.

## Result (Sat 14 Nov 23:56; results/SCREENS.log, results/screens/dn.json): NO EFFECT. Depth stops here (orchestrator's rule)
| MBPP-50 read-out (depth 31) | 0 | 1-2 | 7-14 | 23-30 | kept/32 | params | ms / drafted token |
|---|---|---|---|---|---|---|---|
| n1: 1 layer, 400 steps from native | 88.12 | 76.64 | 57.98 | 48.37 | 4.08 | 60.8M | 2.54 |
| n2: 2 layers, 400 steps from native | 87.43 | 75.82 | 57.22 | 48.42 | 3.99 | 113.2M | 3.96 |
| s30: 1 layer, 1,889 steps (reference) | 88.04 | 77.68 | 62.88 | 55.47 | 4.32 | 60.8M | 2.54 |
- n2 vs n1: MBPP-50 23-30 **+0.05 pt (CI -1.04..+1.16)**, windows 20 / 28; depth 0 -0.69 (CI -1.10..-0.31); kept/32 -0.09
  (CI -0.16..-0.02). HE-30: 23-30 -1.14 (CI -3.08..+0.25), depth 0 -0.34, kept -0.11. **NO EFFECT** by the line.
- The extra layer barely grew again (|o_proj| max 0.0025, |down_proj| 0.0028 after 400 steps at 3e-5).
- The same one-layer head went from 48.37 (400 steps) to 55.47 (1,889 steps, s30) at 23-30: +7.1 pt from budget alone.
**What it says about composition** (orchestrator: "if it doesn't, write down what that says, and stop"):
1. T2's +14 pt (a 24-layer pass over only the drafted tokens) is PRETRAINED composition. A drafter layer started at
   zero and TTT-trained on a few hundred answers neither grows into it nor helps: added late, it HURTS (-2.5); trained in
   from the start, it does nothing (+0.05), at 1.86x the parameters and +56 % draft time per token.
2. In this regime the head's deep accuracy follows its training budget, not its depth (+7.1 pt from 400 to 1,889 steps).
   That agrees with the 125B's "about +1 pt per doubling of data": deep composition is learnable but expensive, and
   neither width (125B capacity tests) nor depth (here) is a shortcut.
3. Routes this leaves for the week, untested here: (a) REUSE pretrained composition, i.e. drafter layers taken from the
   target's own top layers (Kangaroo / LayerSkip-style self-drafting); their per-token cost is the catch (Leap: an L45
   refresh costs about a verify); (b) a much larger training set for the drafter (the 125B data curve).
4. Caveat: one extra layer, one lr (3e-5) and a zero start. A non-zero start or a higher layer lr could let the layer
   grow; FlexDraft-style deep drafters train much longer. Stopped here per the rule, not refuted in general.
