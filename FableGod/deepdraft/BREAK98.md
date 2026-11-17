# BREAK98: the cheapest 125B information for the drafter (Tue 17 Nov 07:3x; desk sprint to ~09:00; orchestrator)

> (a quote from the project owner, kept private)
way, which hasn't been done by anybody before. The cheapest possible way to give the drafter information from the big
model ... by Saturday ... create something beyond the physical theoretical limit on how fast we can give the drafter
information from the big model, as well as getting the drafter to 97.8% accuracy at 30 tokens deep ... not some goofy
little improvements of 5%. I'm not accepting an end result of lossy under 550 tokens per second."
Kamen's examples are options, not rules.

## The one question
Deliver the 125B's state at every drafted position to the drafter for a small fraction of a 125B row's cost.
- Today a 125B row comes only from a check: a 32-row lossy check is 56.4 ms (bound ~25 ms), about 0.8-1.8 ms per row.
- A sequential partial pass per drafted token re-reads weights for every token (~0.5+ GB per token even at 8 of 48
  layers), which is far too many bytes. That is the believed limit. Beat it.

## Why this is both of Kamen's goals (facts; quote them with their headings)
- **GOAL 2 follows from GOAL 1 on HE+ at depth.** ~~Given the TRUE 125B state at every position (LAB), the t30b head
  holds 97.8 % per guess, flat to depth 30, walk-strict~~ **CORRECTED 07:5x (Fable C's audit): "97.8 % flat" is the
  MARGINAL row. Walk-strict with the true state, the head reads 93.2 / 95.3 / 96.4 / 97.8 / 98.8 / 99.2 / 99.3 % by
  band (0 / 1-2 / 3-6 / 7-10 / 11-14 / ... / 23-30; THINK_1116c).** So with true states it is >= 97.8 % from band 7-10
  on (99.3 % at 23-30). The shallow bands are limited by the 125B's own near-ties, which trees and close-call cover.
  MBPP is knowledge-limited (93.6 % marginal).
- **Kept per 32-row check, HE+** (Lunatic, ~/FableGod/lunatic/IDEAS_1116c_drift.md, real 127-node dumps):
  - own tree-32 16.26;
  - ORACLE-31 (the best of the head's own information) 18.48;
  - true-state chain 20.77;
  - true-state + tie siblings 26.79, so drift fixed + a tree is ~24.5-26.8.
- **NUMBER CHECK 07:3x (NewIdeas re-ran Lunatic's script on a copy; it reproduces 18.48 / 11.21, K127 23.70, LAB 20.77,
  EAGLE-2-31 16.26). REFINED:**
  - ORACLE-31 is a HINDSIGHT bound. Without hindsight, the head's own information at 32 rows is capped at EAGLE-2-31
    itself (16.26) in expectation: the head is calibrated, and best-first by path_p maximises the expected on-true
    nodes. So 18.48 is unreachable without NEW information.
  - Retraining is not bounded by this (a retrained head has different candidates), but the sandbox evidence (ASSIM /
    ERRMEM / WALK-MIX) says same-input gains are small.
  - **So kept 24 at 32 rows has two routes:**
    - (1) true-state-class information (LAB 20.77 + tie siblings 26.79). This is THIS sprint, and it also gives
      GOAL 2's 97.8 % at depth 30.
    - (2) ~128 rows of the head's own information (K127 23.70) at about the price of 32 rows: early death + compute.
      This gives speed only, not GOAL 2.
- **550 tok/s lossy** needs kept ~24-26 at a cycle of <= ~44-47 ms (check + draft + information + overhead).
  - Today: check 56.4 + draft 19.1 + overhead ~4 ms, giving ~213.
  - So the channel gets a few ms at most, and the check must also fall (Leap's modules).
- **The believed limit:** the information is the 125B's own computation on the drafted tokens. "The only source that
  delivers it at scale is a 125B row" (Lunatic). Drift is sequence-specific and persistent across blocks on HE+ (92 % of
  the per-window drift-gap variance; THINK_1116c).

## Data you can use now
- BigMoE's t30b dumps, ~/BigMoE/data/deepdraft/ceiling/:
  - cells + TOPK (chains);
  - tree_t30b_{heplus24,mbppbank}_n127k8.npz (real 127-node trees).
- Lunatic's scripts beside IDEAS_1116c_drift.md (each with a .out): drift_rankceil.py, drift_bits.py, drift_fork.py,
  drift_forkcal.py, drift_labtree.py.
- **TONIGHT (Tue, BigMoE, after M4):** the 125B's layer 8 / 16 / 24 residuals along the true text,
  mid_{heplus24,mbppbank}.npz (~/NewIdeas/research/week98/EARLYDEATH_PREREG.md).
  - **If your mechanism needs other layers or tensors captured, name them exactly by 18:00 today**, so BigMoE can add
    them to tonight's capture.
- Fast training analogs: the 2B sandbox (Sandbox98). 125B head training: TrainLab (MTP Head).

## Lenses (one each; cross over freely)
- **Lunatic: the information floor.** Which part of the 125B state (layer, subspace, bits) does the head need to hold
  97.8 % at depth 30? What is the fewest bytes per drafted row that deliver it?
- **ZERO: first principles on the M5 Pro.** Design the cheapest channel (bytes, kernels, timing) for a <= ~44 ms cycle
  at kept 24. Say what it reuses from reads the check already does.
- **FAR: the drafter that consumes partial or approximate 125B information.** An architecture + training recipe that
  TrainLab can run this week.
- **Fable B: partial-width rows** (dense + attention only, shared or top-1 experts, union-constrained routing). What
  fraction of the true state does each recover, at what bytes?
- **Fable C:** audit this brief's numbers first (one paragraph), then invent one mechanism nobody has tried.
- **MoonShots / Breakthrough (via NewIdeas):** an idea from an unusual place (codecs, compression, CPU value prediction,
  control, biology) that delivers the information cheaper.

## Each idea gives
- the mechanism;
- the information it delivers;
- its price in bytes / ms, from measured parts;
- the kept, or the depth-30 accuracy, it buys: a <= 10-min CPU screen on the existing dumps, or a desk bound plus the
  exact screen to run on tonight's capture;
- the quality risk;
- what kills it.

## Kill list (no re-proposal without a NEW mechanism)
SIB sibling states; span tree / copies; draft death / wide-shallow; keyed widening; t30c; megakernel grid barriers; a
second GPU stream; 3-bit dense by RTN / AWQ; comment-free decoding; any smaller verifier; ASSIM; ERRMEM; WALK-MIX
(BETWEEN, leans KILL); inert comment acceptance (wholesale); close-call d2.
Records: ~/TRIED/TRIED.md, WEEK98.md, PLAN900.md, THINK_1116c.md.

## Output
- Write IDEAS_1117_<topic>.md in your own folder by ~09:00.
- Then send ONE line to NewIdeas (`peer send ojox-28`) and to the orchestrator: the best idea, its number, its screen.
- NewIdeas screens the best ones by ~12:00. The winner gets the GPU this afternoon (the owner away) or tonight.
