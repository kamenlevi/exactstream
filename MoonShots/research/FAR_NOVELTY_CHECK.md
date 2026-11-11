# FAR batch: prior-art check of its three candidate new parts (MoonShots, Wed 2026-11-04 22:3x EET)

Ask (the orchestrator 22:3x): a full check before tagging three pieces of Fable FAR's work
(~/FableGod/far/FAR.md sections 3, 8; findings note ~/FableGod/findings/2026-11-04_2210_far-ahead-foresight.md).
Method: this project's internal files first (~/TRIED/NOVELTY.md, TRIED.md, research/sweep_1103/WALL1_SSD.md,
research/results_first/EXTERNAL_PAPERS.md), then 7 standard web searches and 4 abstract reads (arXiv pages). Only
abstracts and search summaries were read, not full papers.

## Verdicts

| piece | verdict | nearest prior art |
|---|---|---|
| 1. A near-draft tier (next 1-2 checks) above a far-draft tier (~8 checks) in one eviction key (FAR-HYBRID) | **not found; adapted** | fMoE / FineMoE (2502.05370): eviction priority = 1 / (expert-map probability x visit frequency), two signals in one score, not tiered by horizon. MoE-SpAc (2603.09983, NeurIPS 2026): the draft phase as a lookahead sensor, prefetch and eviction in one utility space, one draft window. MoE-SpeQ / Specter (2511.14102): lookahead-aware eviction from one draft stage. SeqMoE (2609.12978): eviction on predicted future activations. CPU world: hybrid / tournament branch predictors (McFarling 1993) and multi-predictor cache replacement. |
| 2. A ~100-token draft horizon for expert forecasting (FAR-G1, FAR-ORIGIN) | **not found for draft-fed forecasts; adapted** | MoE-SpeQ: speculation depth K <= 32 tokens (best 8-16), its lookahead buffer cleared every draft stage. SP-MoE (2510.10302): one round; deeper prefetch (in layers) thrashed. ExpertFlow (2410.17954) and DoMoE predict across LAYERS, not tokens. SPICE (2608.21240): confidence-aware adaptive lookahead; the unit is not given in its abstract. Long-horizon forecasts that are not draft-fed are published: request / sequence-level activation traces (MoE-Infinity), HOBBIT's sequence level (2411.01433), fMoE's prompt semantic hints. |
| 3. Soft per-token confidence weighting of far tokens, no cutoff (FAR.md section 3, arm (d)) | **published in its near form; adapted** | PSR section 7 (2610.34721, 26 Oct 2026; ~/TRIED/NOVELTY.md): each verify-block position's expert demand weighted by the square root of the running product of drafter confidences. AcceptMoE (2608.02989): expert demand weighted by each token's commitment probability. MoE-SpeQ's lookahead buffer keeps (expert, confidence) pairs per draft token. DSpark (2607.05147): the cumulative product of per-position survival probabilities. FAR's form applies the running product over a ~100-token far chain as an eviction score. |

## What stays ours
- The far horizon (~100 tokens, 8 checks) for draft-fed eviction, and the near / far tiering in one key. Both are new
  settings of published pieces, not new mechanisms; tag them **adapted**.
- FAR's measurements: no prior art found.
  - The drafter's rejected rows hold 32 % of the cache reuse.
  - Token-vs-state forecast ceilings: token maps -6 %, the true-token ceiling -20 %, a state map -14.6 %.
  - Misses fall 34 % but stalled layers only 5 % under perfect foresight.
  - The wait law: about 12 ms per check + 0.16 ms per miss.
  - Tag them **new (measurement)**.
- A caution from the field: the 2026 reproducibility study of MoE expert caching (2608.07911) finds a learned
  next-use-distance predictor picks the optimal victim only 3.39 % of the time (random 2.42 %): ordering among residents
  matters, not regression accuracy. That matches M-14 (learned key: 14 % of the LRU -> OPT gap) and FAR 1g (precision on
  old residents is the number to train for).

## Sources (searches 22:2x-22:3x)
- MoE-SpAc https://arxiv.org/abs/2603.09983 ; MoE-SpeQ / Specter https://arxiv.org/abs/2511.14102 ; SP-MoE
  https://arxiv.org/abs/2510.10302 ; SPICE https://arxiv.org/abs/2608.21240 ; ExpertFlow https://arxiv.org/abs/2410.17954 ;
  fMoE https://arxiv.org/abs/2502.05370 ; HOBBIT https://arxiv.org/abs/2411.01433 ; SeqMoE https://arxiv.org/abs/2609.12978 ;
  AcceptMoE https://arxiv.org/abs/2608.02989 ; DSpark https://arxiv.org/abs/2607.05147 ; DraftExpert
  https://arxiv.org/abs/2607.24434 ; MoE-Spec https://arxiv.org/abs/2602.16052 ; reproducible caching study
  https://arxiv.org/abs/2608.07911 ; Speculating Experts https://arxiv.org/abs/2603.19289 ; PSR (2610.34721) via
  ~/TRIED/NOVELTY.md.
