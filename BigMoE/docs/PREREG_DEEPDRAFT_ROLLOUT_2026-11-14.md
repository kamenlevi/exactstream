# PREREG: DEEPDRAFT ROLLOUT, can a learned state transition carry the deep state? (Sat 14 Nov 2026, ~40 min)

> (a quote from the project owner, kept private)
try to achieve in principle what we are chasing, not lo0ng tests". What we're chasing is a drafter that predicts the deep
state itself. This is the orchestrator's design, run on the BANK + REPLACE harness, with no head training.

## Setup
- **Transition model (scripts/rollout.py fit):** linear, in the top-k PCA space of wd (k = 512).
  - z_{t+1} = A z_t + B e_{t+1} + c, where z = the PCA coordinates of the trunk stream wd and e_{t+1} = the 125B's input
    embedding of the token at t+1.
  - Ridge on bank train rows: pairs (t, t+1), every 2nd answer row of the 360 train files. Lambda is picked on the 40 val
    files.
- **Diagnostics on val:**
  - the PCA ceiling: the direction error of the true wd after k components;
  - the one-step teacher-forced direction error;
  - the rollout's direction error by depth (1, 2, 4, 8, 16, 30 steps from a fresh anchor wd_s, fed the true tokens);
  - the stale anchor's (wd_s itself) at the same depths.
- **REPLACE evals (t30b unchanged, MBPP 100 with bank features; HE+ if time allows):** the stand-in at depth j is
  replaced by:
  - ROLLOUT: the model rolled out j-1 steps from the true anchor wd_s, fed the true tokens. Under accept-until-fail
    those are the tokens on the rows that count.
  - STALE: the true wd_s, unchanged, at every depth.
- **References:**
  - t30b's own chain: 72.13 MBPP / 89.65 HE+ (runtime-path context) at 23-30;
  - the oracle (true wd at every drafted row): 93.63 / 97.76.

## Lines (result -> action; MBPP 23-30)
| Result | Line | Action |
|---|---|---|
| CORE | ROLLOUT >= the chain + 5 pt (>= 77.13) | A separately learned state-transition model is the method's core. Next: scale it (nonlinear, more data, joint training with the head) and price its cost per drafted token. |
| PARTIAL | ROLLOUT >= STALE + 2 pt, but below the chain + 5 | The transition is partly learnable in this form. Next: a nonlinear or attention-based f (the bank + proxy are ready for it). |
| FAILS | ROLLOUT < STALE + 2 pt | This linear form fails. Report what the direction error by depth shows (where it diverges). |

- All bands and kept per 32 are reported for every arm.
