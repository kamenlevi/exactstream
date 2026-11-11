# MISSION 17: THE DRAFTER WALL. Ideas that have never been tried in speculative decoding.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~16:2x.)

**Kamen's brief.** The drafter is the single biggest bottleneck. Brainstorm it with things that have never been done in this field.

**Why the drafter** (sources in the inputs)
- **Today.** DFlash2 L1-20 (1.2 GB, block 16, one pass) is right on 0.909 of code tokens: 95.5 % where the 125B is sure, 65.9 % at its mid rows, 30 % at ties. It keeps 9.6-10.5 per 16-row check on code, 6.6-7.0 on math; chat keeps 2.0-2.5 (MTP chain).
- **What 350 needs** (RELAX_350.md). With the whole engine at its floor (cycle ~45-60 ms at night), exact 350 needs ~20-22 kept per 32-row check: ~0.97 per token, flat over depth. A kept-13 drafter lands ~260 exact.
- **The wall as written** (TABLE_350 SD-sum, DRAFTER_98):
  - below 1B params, per-token quality sits at ~0.93
  - ~0.98 flat is reached only by target-class models (the near-copy, trunk + 10 % of experts: 0.983; the int4 twin ~0.99), and those cost the 125B's own compute (D-07: 15 sequential steps ~170 ms)
  - DRAFTER_98's information bound: depth >= 2 needs the trunk's processing of the drafter's own previous drafts, which no verified row contains
- **Today's results.** k-head KILL; E2: +0.22 kept per doubling of on-policy data (13 needs ~15 doublings); relaxed acceptance +5-11 % on code at the twin's quality level (the drafter's wrong code tokens are seldom the 125B's close second); DUALDRAFT (two drafters, confidence pick) 92.7 vs 90.9 % in replay, live test Tuesday.
- **About 120 drafter ideas are in TRIED.md** (grep "wall: drafter"). Nearly every published family is killed or parked:
  - bigger drafters (SD-cap: the cost grows with the layers) and the 4-8x drafter (the training-cost wall)
  - trees, autoregressive heads (EAGLE-3, HASS, GRIFFIN), semi-AR, refinement steps
  - self-speculation, lookup / suffix / copy drafting, bandit switches
  - the drafter on the Neural Engine (killed because today's drafter is only 3-8 % of a check)
  - more or other data (E1, E2, DS, DS2), borrowing the 125B's experts (E5)

**Deliver**
1. **The assumptions behind the wall.** List what the dead ideas share, and say for each what breaking it would buy, priced from measured numbers. Examples (find more):
   - the drafter runs on the GPU, on the critical path
   - drafting starts only after the check ends
   - the drafter is fixed during a session and learns only offline
   - the drafter is one model predicting tokens
   - the knowledge must live in the drafter's weights
   - the information bound above: law or assumption?
2. **At least five ideas never tried in speculative decoding.** Work from first principles, or borrow from fields that predict under tight budgets: CPU branch predictors (TAGE, perceptrons), data compressors (PAQ / cmix context mixing, match models), cache prefetchers, JIT compilers, genomics, video codecs, anything else. Starting points to beat:
   - **Free labels.** Every check hands back the 125B's own top-k at 16 rows, and no drafter here learns from them during a session: test-time training, fast weights, a memory of this session's verified tokens, online mixing of several predictors per context.
   - **Idle hardware.** The Neural Engine and the CPU idle during the GPU's check. A much bigger drafter there, drafting the next check from the likely accepted prefix, removes SD-cap's and MEND-b's kill reason (drafting cost on the critical path); it does not remove the bandwidth shared with the check or the training cost.

   For each idea:
   - the mechanism, and which assumption it breaks
   - the closest TRIED entry, and why its kill reason does not apply
   - prior art: published / published only in another field / new (check the literature where you can)
   - the expected gain per class (sure / tie / mid) and depth
   - RAM and ms on this Mac
   - the cheapest decisive test (CPU on existing data, or one night) and its kill line
3. **The one bet,** with a ranked plan and owners:
   - MTP Head: drafters
   - NewIdeas: the 98 % track
   - Leap: kernels and the Neural Engine
   - BigMoE: the engine

   Or, if every design hits the wall, a measured statement of what stops them all.

**Rules**
- The full 125B as the target (all 512 experts); Kamen's MacBook only.
- Exact mode stays exact. Quality-defined ideas are a separate mode, and adopting one is Kamen's call.
- Training: free cloud compute is OK, never paid GPUs; only small models train on the laptop. Downloads need Kamen's OK.
- CPU-only analysis on existing files is fine. GPU runs are proposed for their owners.
- Check ~/TRIED/TRIED.md. Revisit a killed idea only when the new design removes the reason it died, and say which reason.
- Mark every number as measured / arithmetic / judgement.

**Inputs**
- ~/FableGod/DRAFTER_98.md (drafter_anatomy.json / .out, drafter_ensemble.out), RELAX_350.md (relax_positions.json), BREAK_THE_WALL.md, FLOOR_125B.md, ROUTE_350.md, DAY9_2026-11-02.md
- ~/TrainLab/REPORT.md (D-01 ceilings, E1, E2, E5, HPD-1, K-HEAD, D-23d, the DUALDRAFT pre-registration) and ~/TrainLab/data (DLOG, the d02b walks)
- ~/Leap/TABLE_350.md (SD-sum, SD-cap, MEND-b)
- ~/TRIED/TRIED.md
