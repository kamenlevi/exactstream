# MISSION 24: CHISEL THE KNOWLEDGE WALL. Find what predicts the 125B's choices where the drafter fails.

> (a quote from the project owner, kept private)

**The wall.** Exact speculative decoding keeps a draft only if it equals the 125B's greedy token. Exact 350 needs ~0.975 per token, flat; today's drafter is 0.909 on code. Your DEPTH_WALL showed the fade with depth is not the problem, and the copy override already makes copied text ~99 %. What is left is knowledge:
- **Mid rows** (12 % of code positions, 45 % of misses): the 125B is unsure (p1 < 0.99) and the drafter is right only 65.9 %. The 125B's own choice there flips under small perturbations of its computation (top-5 experts 8.6 %, the int4 twin ~8 %).
- **Sure rows outside copies:** the 125B is sure, the drafter misses ~4.5 %, on return types, a docstring's own words, algorithm choices and numbers.
- **Ties** (1.5 %): coin flips.

Cloud Fable is attacking the same wall in theory tonight (~/FableGod/CLOUD_TASK_KNOWLEDGE_WALL.md). You attack it on data: chisel. Every claim gets measured with a script on what we have recorded, before any design.

**Find signals that beat the drafter where it fails.** For each candidate, measure on recorded rows how often it names the 125B's token at mid rows and at sure rows outside copies, against the drafter's 65.9 % / ~95 %. Candidates (invent better ones):
1. **The session's own history beyond exact copies:**
   - style consistency: the choice it made last time in an analogous short context (quotes, naming, spacing, formatting)
   - edit-tolerant matches
   - the prompt's structure turned into code: signature, docstring and tests into names, types and numbers
2. **The target's own cheap outputs:**
   - its native MTP head (D-17 / D-22 / D-23 logs in ~/TrainLab/data)
   - its distribution at the previous verified rows and at the bonus row
   - its top-k at the anchor
3. **The drafter's own candidates:**
   - is the 125B's token in the drafter's top-2 / top-3 / top-16 at mid rows?
   - if so, what picks the right one: a selector problem, with L1-21 / SD-sel's kill reasons to check
4. **The target's internal state at the anchor:** a token probe (not a sureness probe) on the raw taps (~/TrainLab/data/taps_raw/, layers 5 / 15 / 25 / 35 / 45 with the top-32, once EARLYTAP's extraction lands): does layer 35 / 45 at the anchor predict the next mid choice better than the drafter?
5. **Anything the data shows you.** Look at real mid-row and knowledge misses by eye: what would a human need to know to get them right?

**Deliver**
1. **A table:** each signal's accuracy on mid rows and on sure-outside-copy rows, its coverage (how often it has an opinion), and its precision when it disagrees with the drafter.
2. **The best combination** (signals + the drafter, picked per position by a rule you measure, not assume), and the per-token rate it reaches on code, math and chat. Calibrate with the MATCH correction: real gains ran about half the modelled ones.
3. **How far it moves the wall:** the per-token rate and kept per 16 / 32-row check with the copy override, against 0.975. If nothing beats the drafter on mid rows, say so plainly, with the evidence. That bounds what any cheap drafter can do.
4. **The one test that would confirm the best finding:** a real walk for MTP Head (or a CPU replay), with PASS / KILL lines.

**Rules**
- CPU only on recorded data; writes only in ~/FableGod; no downloads. GPU runs are proposed for their owners.
- The full 125B as the target; drafters only propose, and exact mode stays exact.
- Check ~/TRIED/TRIED.md (D-03 n-gram, D-06 copy pointer, M-05 datastore, SUFFIX, LK1, L1-21 lattice picker, SD-sel, D-18 MTP1, D-19 bandit, K-HEAD, MATCH, COPYGATE) and say why each kill reason does or doesn't apply.
- Mark every number as measured / arithmetic / judgement.
- Ambition: the target is a signal that lifts mid-row accuracy from ~66 % toward 85 %+, or sure-row knowledge misses from ~95 % toward 99 %. Either would move the wall. Report what you find, including near-misses.

**Inputs**
- ~/FableGod: relax_positions.json (per-row class, cf, the 125B's log-probs, target token), drafter_anatomy.json, drafter_memory_probe.py, DEPTH_WALL.md, NEXT_DRAFTER.md, DRAFTER_98.md, RELAX_350.md, SAFARI.md, copygate/GATE.md, decision_map/out (baseline.json: the 125B's top-8 per row)
- ~/TrainLab/data: the HPD-0 / MATCH walks (ids, drafts, cf per check), d02a (top-32 + taps for math / IFEval / multi), d23*, agent_compare_*, taps_raw/ (tonight)
- ~/BigMoE/models/bare/vocab.json; ~/TRIED/TRIED.md
