# MISSION 25: THE COMMENT LOOPHOLE. Does the knowledge wall live in comments, where wording can't break a program?

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~19:4x.)

**Why.** Your KNOWLEDGE_WALL.md showed the wall is the 125B's own wording at mid rows, and that no cheap signal predicts it. But look at your ten mid-row misses: nine sit inside comments ("# Find the longest suffix ...", "# We check from the full string ...", "# The prefix before this palindromic suffix ..."). If a large share of code's mid-row misses is comment and docstring prose, then accepting the drafter's near-equal word only there changes only comment wording. Every code token stays the 125B's exact choice, so the program and its test results cannot change. That would be a quality-defined mode with a clean safety argument, aimed exactly at the part of the wall no drafter can reach. Adopting it is Kamen's call; measure it first.

**Step 0 (routing only).** Your one test from mission 24 (store DFlash2's top-16 and the MTP chain's draft at every tested position, and check top-3 coverage at mid rows) goes to MTP Head through the orchestrator. Don't wait for it.

**Deliver**
1. **Where the misses live.** On all recorded code windows (the 12 map windows, the 24 HumanEval+ walks, E2's MBPP answers, the 12 agent windows), split positions into:
   - code tokens
   - '#' comments
   - docstring prose
   - doctest lines ('>>>' and their outputs)
   - string literals / f-strings

   Detect each region from the token stream as the engine could at draft time. Per region give: the share of positions, the 125B's class mix (sure / mid / tie), the drafter's accuracy, and the share of all misses. Is the wall mostly in comments, or not?
2. **Price relaxed acceptance restricted to inert prose** ('#' comments and docstring prose, never code, doctests or string literals). Rules: R1 at 0.5 nat; R2 (the 125B's top-2 at >= 0.25 p1); a top-3 variant. Give:
   - per-token rate and kept per 16 / 32-row check on code, calibrated with the MATCH correction (real ~ half the modelled gain)
   - the share of output tokens that deviate, all of them inside comments
3. **The safety argument and its edges:** which "prose" can still change behaviour (doctests, strings, docstrings read by tools, type hints in comments), the exact region rule the engine would use, and how often the region detector could be wrong.
4. **The combination:** with the gated copy override and the two-drafter pick, the code per-token rate and kept. What it does to 350 on code with the honest engine numbers (measured 63.6 today; the core pass ~22-27 ms realistic; the Neural Engine draft 12.4 ms).
5. **The test:** a pre-registered walk (MTP Head) plus the quality check EvalSuite would run (HumanEval+ / MBPP pass rates identical, comment deviation reported), with PASS / KILL lines.

**Rules**
- CPU only on recorded data; writes only in ~/FableGod; no downloads.
- Exact mode stays exact; this is a separate quality-defined mode, and adoption is Kamen's call.
- Check ~/TRIED/TRIED.md: B12-R (span-aware relaxed acceptance on prose, exact inside code, killed for a small gain on chat; say why comments inside code differ or don't), R-61N, M-15, RELAX_350.
- Mark every number as measured / arithmetic / judgement.

**Inputs**
- ~/FableGod: KNOWLEDGE_WALL.md (knowledge_probe.py / .out), RELAX_350.md (relax_curves.py, relax_positions.json), DEPTH_WALL.md, NEXT_DRAFTER.md, decision_map/out (baseline.json: the 125B's top-8)
- ~/TrainLab/data: d02b_walk_hpd0_*, d02b_walk_m_*, E2's MBPP capture, the agent windows
- ~/BigMoE/models/bare/vocab.json; ~/TRIED/TRIED.md
