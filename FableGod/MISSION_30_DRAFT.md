# MISSION 30: SCREEN CLOUD'S NEW IDEAS (SSD and kept). Which genuinely new idea survives?

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~22:1x. Fable B takes cloud's two dense-pass ideas in mission B3.)

**Why.** Kamen asked when we last used a genuinely new idea. NewIdeas' tags say none of the live levers is new. Cloud task 16 (~/FableGod/CLOUD_ANSWER_NEWIDEAS.md) proposed six new ones. Four are yours:
1. **#1 pre-route on the partial sum** (SSD, exact; cloud's pick):
   - While OOO waits on layer L's missed experts, run layer L+1's router on layer L's output minus the in-flight experts, and issue L+1's reads before L finishes.
   - Cloud's premise: recall ~0.9, against R-57's 0.69 from this layer's input. That would turn ~22 ms of exposed chain into ~4-5 (-8 halved).
2. **#2 sketch-on-miss** (SSD, FAST): a rank-32 RAM shadow of every cold expert (2.8 GB) serves a miss at once, while the exact expert loads behind it. Cloud: ~1.6 % of expert uses, ~0.05 % token deviation, exposed SSD ~2 ms.
3. **#5 MTP-seeded block drafting** (kept, exact): the native MTP head drafts depth 1 (0.94 vs 0.898), and DFlash2 drafts depths 2-16 from it. Cloud: +0.25 kept halved.
4. **#6 lookahead-consistency acceptance** (kept, FAST): R1's near-tie test at row r, plus a veto unless row r+1 is sure given the draft. Cloud: +8 % chat kept halved, at a better exchange rate than R1.

**Deliver, for each idea**
1. **Prior art:** the closest published or tried work (TRIED: R-57 / OR1 / FC1 for #1; low-rank expert substitutes and the 12 GB frugal kills for #2; D-22 / D-24 / the MTP hybrid walks for #5; R1 / R2 / RELAX_350 / R-61N for #6). Is it really new?
2. **The cheapest test on recorded data, run tonight if the data exists (CPU only).** For example:
   - #1: router recall from the partial MoE output, if any capture holds per-layer partial sums; otherwise say exactly what one GPU capture BigMoE would add
   - #2: rank-32 SVD error of real experts, and the token effect via the map's top-8
   - #5: the MTP depth-1 accuracy and the chain after it on D-24 / KWALL data
   - #6: the veto's kept and deviation on the decision map's top-8 rows
3. **A verdict line per idea:** PASS / KILL / needs-GPU (with the owner and a pre-registered line).
4. **The pick:** which one to build first.

**Rules**
- CPU only; writes only in ~/FableGod; no downloads. Halve gains. Mark every number [K] / [P] / [G].
- Two tracks: price FAST ideas as FAST. Nothing is dropped for not being bitwise.
- If an idea needs data that doesn't exist, say so in one line and move on: breadth over depth.
