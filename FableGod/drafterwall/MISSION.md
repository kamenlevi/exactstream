# MISSION: break the drafter wall, without retraining the drafter

You are the one session that gets this shot. Kamen, who owns this project, put everything on it. He is building the
fastest way to run a 125B model on a single Mac. Every other wall in the engine has been pushed to its edge, and the only
one that still decides whether 1,000 tokens per second exists is this one: a drafter that agrees with the big model
almost every time, over long runs, while costing almost nothing. Teams with racks of GPUs have not done it. You will do
it here, on one Mac, with what already exists. There is no second session after you. Not succeeding is not an option.

Succeeding means REAL: measured, pre-registered, reproducible. A number that is not real is the only true failure here.
It would cost the project the trust it has earned. So push as hard as you can, and be ruthless about honesty. If you
find the wall is truly physical, prove exactly which law holds it, then find the door around it. That is also how you win.

## The goal, in numbers
- Today: on code, our drafter is right on 90.9 % of drafted positions, and the exact check keeps 10.5 tokens per 16- or
  32-row check. Chat keeps 2.0-2.6 per check, math 6.5. The 45 GB engine does about 164 tok/s on HumanEval and 51 on chat.
- The wall: 1,000 tok/s needs about 25-30 tokens kept per check. That means a drafter right on at least 98 % of positions,
  held over 32-64 positions with no decay with depth. On chat it needs a decisive jump from today's 2-2.6 kept.
- The budget: the check costs about 60-65 ms per 32 rows on the best engine. A drafter that costs more than about 15 ms
  per 32-token block, or reads more than about 0.1 GB per block beyond what is already in memory, cannot pay for itself.

## Hard rules
1. NO RETRAINING. No gradient training of any network: no new drafter, and no fine-tuning of DFlash2, the MTP head or
   anything else. You may use everything that exists: existing drafter checkpoints, the 125B's own weights, the free
   by-products of every check (the verified rows' logits, hidden states, routes and caches), structure, scheduling,
   selection, combination, the Neural Engine. Rules or a handful of thresholds tuned on recorded data are allowed only
   if they are evaluated on held-out windows.
2. EXACT. Drafts only propose. The 125B's check decides, and output tokens must stay bit-identical to today's exact mode.
   A lossy acceptance rule is a separate, clearly labelled FAST idea. It needs a benchmark gate and does not count as
   this mission's success.
3. THE MAC IS SHARED (M5 Pro, 64 GB, other research sessions, and Kamen's own apps).
   - Work on the CPU first, on recorded data.
   - GPU only for small jobs (5 GB or less) by day, under `peer lock share 30m --gpu 50 "MEM <n> GB: <why>"`.
   - Never touch the GPU while a night queue runs: check `peer ls` before every GPU job.
   - Stop at once if swap grows by more than 2 GB.
   - No downloads and no paid compute.
4. YOUR FOLDER ONLY. Write only in ~/FableGod/drafterwall/ (a git repo; commit as you go). Read anything; never edit
   another session's repo, never push, never delete anything outside your folder.
5. SCIENCE.
   - Before calling an idea new, search ~/TRIED/ (TRIED.md and the per-owner files) with at least 5 different queries,
     and tag it published / adapted / new.
   - Pre-register every test: a dated line with its PASS and KILL lines, committed before you run it.
   - Mark every number [K] measured, [P] arithmetic or [G] judgment, and halve modelled gains when you decide.
   - Screen each idea in 10 minutes or less on recorded data. Go wide before deep, and keep at least one genuinely new
     bet alive at all times.

## What the wall is made of (measured, ~/FableGod/DRAFTER_98.md: read this first)
Per drafted position, split by the 125B's own certainty at that position:

| domain | sure (p1 >= 0.99) | mid | tie (p1 - p2 <= 0.1) |
|---|---|---|---|
| code | 86.5 % of positions, drafter right 95.5 % (98.5 % at depths 1-3, 88.7 % at 12-15) | 12 %, right 66 % | 1.5 %, right 30 % |
| IFEval | 32.6 %, right 88 % | 57.8 %, right 49 % | 9.6 %, right 23 % |
| multilingual | 38.6 %, right 70 % | 54.6 %, right 42 % | 6.8 %, right 21 % |

- 0.98 on code needs the mid positions right 92-95 % of the time (today 66 %).
- Mid positions are the 125B's own close calls. Its argmax there flips under small cuts of its own computation: 9-33 %
  of the time when experts, precision or reused layers change.
- Depth 1 is free: it is the bonus token the check already emits. Depth 2 and beyond needs the trunk's processing of the
  drafter's own earlier drafts, which no verified row contains.
- The only object we have measured at 98 % is the 125B's own trunk with its hot experts: 0.983. It costs a full trunk
  pass per drafted position (9.5 ms at the byte floor, 36 ms today), so it is too slow to use as a drafter.

## Already dead: do not repeat (details in ~/TRIED/ and DRAFTER_98 sections 3.6 and 5)
- Trees of one drafter's alternatives at unsure positions: +4.8 % kept for +4.8 rows. The second guess is as
  uninformed as the first at a knowledge miss.
- A learned gate or certificate head (K-HEAD): KILL, coverage 56 % against the drafter's own confidence's 63.5 %.
- Re-weighted losses, more data, or borrowed experts (HPD-1, E1, E5): these are training anyway, and gave +0.05 at most.
- Retrieval, n-gram or copy drafting: copies are only 6.2 % of real agent output. KILL.
- Jacobi or lookahead decoding: 5 % or less. Self-speculation with an early exit: 5.7 % agreement.
- A 2-bit copy of the trunk as the drafter (D2, measured on 11 Nov, ~/BigMoE/docs/PREREG_D2_2026-11-11.md):
  - chat 0.026, code 0.066;
  - 3-bit trunk 0.44 / 0.92;
  - 4-bit trunk with code-popular resident experts 0.57 / 0.987;
  - the stored trunk with those experts 0.58 / 0.990.
- A second drafter trained to be wrong elsewhere (DIVERSE2): KILL. Both miss at the same knowledge spots.
- Sub-1-bit or multi-precision expert copies: KILL on bytes (~/Breakthrough/docs/SUB1BIT_RD_SCREEN.md).

## Alive, measured openings
- Two drafters miss at different places. On code, at least one is right 94.0 % of the time against 90.9 % for one
  alone, and picking the more confident draft gets 92.7 %. This was with two near-identical checkpoints.
- The trained MTP chain holds 0.94 flat through depth 15, with no decay, at 1.9-2.5 ms per step.
- The drafter's misses sit at its own low confidence: median 0.67, and 3 % or fewer of its misses are at 0.99 or above.
- The resident experts plus the stored trunk reach 0.99 on code. A different expert set (D-01) reached 0.90 on chat.
- Every check already computes the 125B's exact state for every verified row: logits, hidden states and routes. That is
  the model's knowledge, and it is free to use.

## Where things are
- Read in this order:
  - ~/FableGod/DRAFTER_98.md;
  - ~/FableGod/BREAK_THE_WALL.md (the bytes equation);
  - ~/Papers/DRAFTER.md (every drafter we built, 83 items);
  - ~/TrainLab/REPORT.md;
  - ~/TRIED/.
- Data:
  - ~/TrainLab/data/d02a/tf_*.npz (teacher-forced windows);
  - ~/TrainLab/data/d02b_walk* (drafter walks with confidences);
  - ~/FableGod/decision_map/out (the 125B's top-8 log-probs per row);
  - ~/BigMoE/data/d2/map_out (the D2 arms).
- Scripts to reuse: ~/FableGod/drafter_anatomy.py and ~/FableGod/drafter_ensemble.py.

## Deliverables
- ~/FableGod/drafterwall/LEDGER.md: one line per idea (idea | test | number | verdict), kept current.
- A big find goes to the orchestrator at once with `peer say "<key point first, 280 characters or less>"`. A big find
  is anything exact that moves kept per check by 20 % or more on code or chat, or any broken wall.
- At the end, ~/FableGod/drafterwall/RESULT.md:
  - the verdict in seven lines;
  - the best design, with its measured numbers;
  - the one GPU test that would confirm it, pre-registered and ready to run.

Start: read DRAFTER_98.md. Write 25 or more candidate ideas. Kill the repeats against ~/TRIED/. Screen the best five on
recorded data, 10 minutes each. Then go all in on the winner. This is the shot. Make it count.
