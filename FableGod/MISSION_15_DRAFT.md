# MISSION 15: THE 98 % DRAFTER. Break the knowledge wall.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~12:5x.)

**Kamen's brief.** Mission 14 found the real wall: tokens kept per check, set by the drafter. Longer checks only pay with a drafter right ~98 % of the time over 32-64 positions, and BREAK_THE_WALL.md judged that every buildable drafter stops at ~0.96. That judgement comes from past results, not a proof. This mission attacks it: design the drafter that reaches 98 %, or show exactly what makes it impossible.

**What we know** (sources in the inputs)
- **Today's drafter.** DFlash2 agrees with the 125B on 0.934 of code tokens at depths 1-5 and 0.889 at depths 11-15. It keeps 10.5 per check at 16 rows and the same 10.5 at 32.
- **Ceilings measured in D-01:**
  - the near-copy (trunk + 10 % of the experts): 0.983 on code
  - the int4 twin: 14.8 of 16 on code (~0.99 per token), 11.1 on chat
  - the float floor: 15.2 / 13.7 per 16 rows

  So 0.98 sits below what a model with the 125B's knowledge reaches. The question is the cost of carrying that knowledge.
- **Where the 125B is sure:** on 86.8 % of code positions; the drafter is sure on 65.7 %. If the drafter were right on every position the 125B is sure of, today's 0.934 would mean it gets about half of the rest. Then 0.98 means matching the 125B on most of its unsure positions. The error anatomy below should check this split.
- **Dead ends (all KILL):** more multilingual data (E1), borrowing the 125B's experts (E5), loss only on unsure positions (HPD-1), a learned certificate head (k-head, today).
- **What worked:** the 125B's own MTP head as the chat drafter, +24.5 % exact (D-23d).
- **Running today:** E2, the data slope on new MBPP code prompts. It shows whether more on-policy data moves kept at all.

**Deliver**
1. **The error anatomy.** Where exactly today's drafter misses, from existing data (decision map rows, D-23 logs, DLOG):
   - by the 125B's own class (sure / tie / mid)
   - by depth
   - by domain
   - by token kind

   How much of the gap to 0.98 is reachable at all: e.g. the misses at true ties are coin flips for any drafter that isn't the model.
2. **The information bound.** What a drafter must see to match the 125B at its unsure positions, and what that costs in bytes and ms per 32-token block on this Mac. Is the knowledge in the trunk's states, which are already computed for every verified row? In the routed experts? In the context?
3. **At least three drafter designs aimed at 0.98**, beyond tuning DFlash2. Candidates (invent better ones):
   - drafters that read far more of the 125B's own computation, e.g. multi-layer taps, the verify pass's hidden states, cross-attention to its KV cache
   - the 125B's own MTP head trained to chain over 32+ positions
   - on-policy training at scale on the 125B's own answers
   - a cheap compressed copy of part of the trunk
   - quality-defined acceptance at true ties (how high per-token acceptance goes when the 125B itself is indifferent; adoption is Kamen's call)

   For each: the expected agreement curve over depth, the draft cost per block, RAM, training cost (free cloud compute is allowed; no paid GPUs; only small models train on the laptop), what must be true, and the cheapest decisive test with its kill line.
4. **The one drafter to bet on,** with a ranked experiment plan and owners (MTP Head owns drafters, NewIdeas the 98 % track). Or, if 0.98 is impossible for any affordable drafter, a measured statement of why, and the best kept per check that is reachable.

**Rules**
- The target is the full 125B: all 512 experts, no pruning, no smaller target model. The drafter may be any model that runs on Kamen's M5 Pro MacBook within the RAM line it serves.
- Exact mode stays exact (the drafter only proposes).
- Downloads need Kamen's OK.
- Check ~/TRIED/TRIED.md; revisit a killed idea only when the new design removes the reason it died.
- Mark every number as measured / arithmetic / judgement.

**Inputs**
- ~/FableGod/BREAK_THE_WALL.md, FLOOR_125B.md, ROUTE_350.md, DAY9_2026-11-02.md
- ~/FableGod/decision_map/out/
- ~/TrainLab/REPORT.md (D-01 ceilings, DS / DS2, D-23d, E1 / E5, HPD-1, K-HEAD, E2)
- ~/TrainLab/data/ (DLOG, kh_read.txt, d23_runs.jsonl)
- ~/TRIED/TRIED.md
