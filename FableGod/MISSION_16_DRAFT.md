# MISSION 16: RELAX WHERE THE MODEL IS UNSURE, and re-price 350 with today's measurements

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~14:1x.)

**Why.** DRAFTER_98.md showed where the wall is. On code, 45 % of the drafter's misses (and 12 % more at ties) are at positions where the 125B itself is unsure. Its choice there moves under almost any perturbation of its own computation (top-5 flips 8.6 % of code mids; the int4 twin ~8 %). Matching those choices costs the 125B's full computation, so 0.98 is out of reach for exact drafting.

But the check does not have to match a choice the model itself is not sure of.
- If the check accepted the drafter's token wherever the 125B rates it close to its own top choice, the mid and tie misses would mostly become keeps.
- Arithmetic on drafter_anatomy.json's 3,283 code positions: with every mid and tie miss accepted, per-token acceptance would go from 0.909 to ~0.96. With the sure-row misses also halved (a better drafter, DRAFTER_98 3.1), ~0.98.
- That is the 98 % world, in a quality-defined form. Its cost is quality, which EvalSuite can measure.

This is a known family (lossy / typical acceptance), new to this project's exact-first route. Adoption is Kamen's call.

**Deliver**
1. **Acceptance curves from the existing data** (drafter_anatomy.json + the 125B's top-8 log-probs in decision_map/out/baseline.json). For rules such as:
   - accept if log p(draft) >= log p1 - δ
   - accept if the draft is in the top-k and p(draft) >= x * p1
   - the speculative-sampling ratio at a temperature
   - rules only at positions the 125B marks unsure

   give per-token acceptance, kept per check at 16 and 32 rows, and the per-token deviation from greedy, split by the 125B's class, depth and domain (code, math, IFEval, multi, agent where data exists).
2. **The quality bar.** Which rules keep deviation at the int4 twin's level (~1 % of code tokens, ~5 % of chat), the level the 3-bit tiers passed same-scores at, and which go beyond it. Name the 2-3 rules worth an EvalSuite same-scores run, and the free-running harness that run needs (mission 7's definition).
3. **Re-price 350 with today's measured numbers.** Today measured:
   - the constant: 15-18 ms best with known exact levers (C0)
   - the k-head: KILL
   - the cf gate: covers 57-64 % at <= 0.1 % flips
   - HC8: same-scores PASS, with a lean
   - D7b drafter: 6.3 ms
   - NAX32 per-row cost: 1.7-1.9 ms
   - expert kernel: 1.5-1.7x above its byte floor
   - Q1: PASS; Q2: KILL
   - condition 8 (more on-policy code data): PASS on one doubling, code kept 9.57 -> 9.79 (+0.22 per doubling); kept >= 13 would need ~15 more doublings

   Give code / math / chat / mixed tok/s with and without the best relaxed rule, at 16 and 32 rows, by day and at night. Say whether 350 on code is reached, by which stack, and which single next measurement decides it.
4. **What it opens beyond 350:** with a relaxed check, how far 32-64-row passes now pay, and what that means for the 24 GB and <= 12 GB lines.

**Rules**
- The full 125B as the target (all 512 experts).
- Kamen's MacBook only.
- Exact mode stays exact: relaxed acceptance is a separate quality-defined mode, and adopting it is Kamen's call.
- CPU-only analysis on existing files is fine. GPU runs are proposed for their owners (EvalSuite for quality, MTP Head for drafters, BigMoE for the engine).
- Check ~/TRIED/TRIED.md (R-61N / M-15, the near-tie rule, is BETWEEN there).
- Mark every number as measured / arithmetic / judgement.

**Inputs**
- ~/FableGod/DRAFTER_98.md, drafter_anatomy.json / .out, BREAK_THE_WALL.md, FLOOR_125B.md, ROUTE_350.md, DAY9_2026-11-02.md
- ~/FableGod/decision_map/out/ (baseline.json and the cut runs)
- ~/EvalSuite/results/ (hc8_d4.txt, ref2ct_d4.txt and the same-scores rules)
- ~/TrainLab/REPORT.md, ~/TRIED/TRIED.md
