# DEEPDRAFT: a drafter that stays right 30 tokens deep (half of the main goal; Kamen, Wed 11 Nov 2026 ~21:4x)

> (a quote from the project owner, kept private)
drafter keep its correctness even after 30 tokens ... and actually prove small scale that it works".

## Where we stand (measured)
- From ~/FableGod/drafterwall/RESULT.md, teacher-forced on the 125B's own answers:
  - The MTP head b5 is right 97.7 % at depth 0 on HumanEval+. That is the first draft after the bonus token, two
    tokens past the last exact trunk state.
  - It fades with distance: 96.0 at depths 1-2, 93.3 at 3-6, 90.8 at 7-10, 87.8 at 15-22 and 86.6 at 23-30.
  - MBPP falls from 93.5 to 69.6.
- If the drafter held its depth-0 accuracy at every depth (the lag-2 ceiling), kept per 32-row check would be:
  - HumanEval+ 20.05 against 12.55 today (x1.60);
  - MBPP 12.12 against 6.23 (x1.95);
  - with a perfect pick of two readers, x1.95 and x2.70.
- Nothing exact and training-free gets there (drafterwall: 4 KILL, 1 BETWEEN). After a wrong draft the rest of the
  block is lost (Law 1), and extra rows buy only the logarithm (Law 3). The door is a reader that carries a stand-in for
  the trunk state forward (Law 2).

## The goal and the proof lines (to be pre-registered by the owner before any run)
- **Goal (the world-first claim):** at least 97 % per-token agreement with the 125B's exact output at depths 23-30, on
  held-out HumanEval+-style code. The 125B is untouched, and the check stays exact.
- **Small-scale proof, PASS:** on held-out HumanEval+ windows, depths 23-30 reach 95 % or more (b5: 86.6 %), AND the
  MBPP curve stays within 3 points of its own depth 0 out to depth 30 (b5 loses 24 points).
- **KILL:** under 90 % at depths 23-30 on HumanEval+ after the planned data and capacity.
- **Also reported:** kept per 32-row check from the same flags (HumanEval+ 12.55 today, 20.05 ceiling), chat
  (IFEval / multilingual) curves, and the depth-0 accuracy, which must not drop.
- **Harness:** drafterwall's t2_laglaw.py and gpu_test/g1_lag2_readout.py. They reproduce TrainLab's 9.58 / 12.55 /
  5.93 / 6.23.

## Steps
> (a quote from the project owner, kept private)
   trained? The plan below assumes yes.
1. **LAG2-AGENT** (pre-registered in drafterwall/RESULT.md section 4): the depth-holding gain on agent turns and fresh
   MBPP. One evening on the GPU; owner the drafter session.
2. **Data.** The 125B's own answers with the full context plus the drafter's input features, captured locally.
   - Fresh code prompts first (none from the eval sets), then agent turns: Kamen's Claude Code sessions with the local
     model, and EvalSuite's recorded answers re-captured at prefill speed.
   - 3-5M tokens for the proof. That is 2-3 nights, at about 1.5-3M tokens per night.
3. **The reader.** Start from b5 (D-22's training-time-test chain).
   - Train the chain to depth 30 on its own drafts.
   - Give it a learned stand-in for the trunk state that it carries and updates step by step, with a feature loss at
     every depth.
   - Sweep capacity from 69M to about 300M trainable. Local training only (1B or less).
4. **Measure** the depth curve and kept per check on held-out windows (the lines above).
5. **If PASS:** a live 45 GB A/B in the engine (exact: tokens bitwise equal to the default; whole decode tok/s), then a
   dated finding and a paper section.

## Rules
The 125B is never modified. Drafts only propose. CPU and small GPU jobs by day, captures at night when the queue is free.
Every test is pre-registered, every number marked [K] / [P] / [G], and every idea checked against ~/TRIED first.

## Owner and timing
The drafter session (TrainLab) owns it, on account-2 once that account resets (Sat 23:00 UTC / Sun 01:00 EET), or BigMoE
on main after the Thursday 17:00 reset. Step 1 can run on the first free night after Thursday 17:00. A first trained
reader is about a week away.
