# MISSION 13: THE FLOOR. How much of the 125B does each token really need?

(Draft by the orchestrator, Mon 2 Nov 2026 ~10:4x, for Kamen to send to Fable.)

**Why.** Every route we have is priced by the weights read per token:
- Exact: ~137 tok/s on code.
- Quality-defined: ~200.
- 350: only as a nine-condition long shot (~15 %).
- Breakthrough's 12 GB / 1000 tok/s goal.

Today's exact check moves 0.78-0.95 GB of RAM traffic per kept token. 350 needs <= 0.86; 1000 needs ~0.3.

Last night's decision map showed that on tokens the 125B is sure of (87 % of code positions), almost every single cut is free:
- top-2 experts: 0.21 % flips
- top-5: 0.00 %
- middle-layer expert reuse: 0.1-1 %
- 4-bit attention / GDN / HC outputs: 0.00 %

Nobody has measured how far this goes when cuts are combined and chosen per token. If the true floor is far below 0.86 GB, 350 stops being a long shot, and the small-RAM goals open with it (fewer bytes per token also means a smaller working set). If it isn't, 350 sits at the physical edge, and we know it.

**Deliver**
1. **The oracle floor.**
   - Take each token of the 125B's own answers: code, math, chat and agent, on the HumanEval+, GSM8K, IFEval, multilingual and agent sets.
   - Find the smallest computation that keeps its argmax: experts per layer (top-k per row), layers whose MoE output can be reused or skipped, and precision per part (expert tier; attention / GDN / HC outputs).
   - Report GB of weights per token by type and confidence band, and the tok/s that would mean at 45 GB, 24 GB and 12 GB.
2. **The realizable floor.**
   - How much of the oracle can a predictor reach, using only signals available before the work? Candidates: drafter or MTP confidence, the router's own scores, the previous token's routing, early-layer states.
   - Remember NewIdeas' UGB result: the oracle passed, but the realizable gate kept only 65 % of code.
3. **The mechanism.**
   - The engine design that exploits it on this MacBook, e.g. per-row work passed as GPU data through the sync-free gate, like NEW-1.
   - Its quality cost under EvalSuite's same-scores rules, including which cuts would count as exact by scores (the old mission-13 question).
4. **The one experiment** (<= 1 night, on the decision-map harness) that decides it: a new world (<= ~0.3 GB per token, 1000 tok/s territory) or 350 at the physical edge.

**Rules**
- The full 125B: all 512 experts, no pruning, no smaller model.
- Kamen's M5 Pro MacBook only.
- Exact mode stays exact; adopting quality-defined modes is Kamen's call.
- Mark every number as measured / arithmetic / judgement, as in ROUTE_350.

**Inputs**
- ~/FableGod/ROUTE_350.md
- ~/FableGod/decision_map/out/ (NIGHT1_SUMMARY.md, NIGHT2_STAGE1.md, report.md)
- ~/FableGod/DAY9_2026-11-02.md (today's nine tests; their results land through the day)
