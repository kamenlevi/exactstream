# Expert skipping at inference: principles for frugal's KLD (2026-10-20 21:45)

Eval suite: eager frugal 10 % teacher-forced step KLD 0.057 (ar4 0.0096); ifeval 0.115, multi 0.092, code 0.015.
Frugal alters ~39 % of the 480 slot uses per token (10 % budget: 92 dropped + 94 substituted, plane_sim), chosen by
CACHE RESIDENCY, not by importance.

- ACE (arXiv 2609.05228, Sep 2026): calibration-free per-token skipping, keeps top-1 always, skips a slot only when
  both a gate x spectral-amplification score and a gate x router-direction score are low; renormalises retained gates.
  Qwen3.6-35B-A3B (our proxy): BF16 PPL 7.01 / avg acc 81.17; 20 % skip 7.34 / 81.46; 30 % 7.57 / 79.89; 40 % 7.88
  / 78.79; 50 % 8.67 / 75.57. PPL and reasoning/code accuracy diverge at aggressive budgets.
  -> Skipping 20 % of slots BY SCORE is near free; frugal skips ~39 % by residency: expected much worse.
- Top-P / dynamic routing (Huang et al. 2024, "Harder tasks need more experts"): experts until cumulative gate mass
  >= p; the published form of "wait by gate mass".
- MoDES, SERE (re-route to similar experts) are baselines in ACE; substitution by an unrelated resident expert (our
  ranks 10-19) is not what SERE does (SERE uses expert similarity).
- ZEDA (2605.18643): self-distillation lets a post-trained MoE skip >50 % of expert FLOPs (training needed).
- DyMoE (2603.19172): bottom quarter of active experts at 2-bit ~= skipping them (GSM8K/MMLU).
Hypotheses to test (tf_frugal_var.py + gate_mass.py): H1 substitution hurts (nosub < base KLD); H2 the KLD is carried
by a few high-mass misses (wait-by-mass theta cuts KLD at small extra reads); H3 renormalisation choice matters little.

## SERE (ICLR 2026, arXiv 2602.07616): the published "cheap substitute"
Secondary expert u -> its most similar PRIMARY (already active/resident) expert v* when similarity >= rho, else kept;
similarity from expert OUTPUTS on calibration inputs (FineWeb-Edu 400 x 128 tokens), Frobenius 1 - ||X_u - X_v|| / max;
weight-based similarity did worse. Router weights unchanged (u's gate weight applied to v*'s output, no renorm).
Qwen3-30B-A3B avg acc: top-8 82.24; plain top-2 14.30 (!); SERE top-2 rho 0.5 80.37; plain top-1 8.60, SERE 64.11.
Qwen1.5-MoE: most-similar 47.15 vs random re-route 41.68 vs least-similar 28.74 (K=2). Similarity is layer dependent:
Qwen3-30B layer 1 nearly all pairs > 0.9, layer 6 mostly < 0.4.
-> For frugal: substitute a missing rank 2-9 with the most OUTPUT-similar resident expert at the missing expert's gate
weight; keep it computed exactly (rho) only when no resident expert is similar enough (wait / read). Needs per-layer
512 x 512 similarity tables for the 125B: capture MoE inputs per layer on calibration text, run every expert on them
(68 GB read once, ~15-20 min), pairwise output distances on the GPU. Also eMoE (2503.06823): task tolerance to routing
errors varies greatly (fits our ifeval/multi vs code gap).

## Confidence-gated coverage (BiLD-style fallback), 2026-10-21 01:00: data + prior art
Data (Eval suite tfstep, frugal 10 %, 34,457 positions): frugal's own next-token entropy correlates 0.956 with the
reference's; if every token whose frugal entropy > 0.1 were recomputed exactly (37 % of tokens) the per-position mixture
would have KLD 0.0029 (98 % of frugal's KLD caught; ar4 0.0096), at > 0.5 (25 %) 0.0098; confident tokens (entropy <=
0.1, 21,712) have ZERO top-1 flips vs the reference. Causal proxies are weaker (previous token's entropy > 0.05: 41 %
covered, 74 % caught). Caveat: a real run keeps frugal context for unrecomputed tokens; tf_frugal_var.py bild_eNN measures
it (rollback_after_verify of the 1-row step, exact recompute).
Prior art: the principle is CALM (Schuster et al. 2022, confidence-gated early exit) and BiLD (Kim et al. 2023, small
model with fallback to the big one); MoE self-speculation with a resident draft path verifies every token (DraftExpert
2607.24434: 1.45x over offloaded AR, 84-87 % acceptance; S2-MoE 2608.15018; MoE-SpeQ 2511.14102), so its SSD bytes per
token stay exact-like; AcceptMoE (2608.02989) shrinks the VERIFIER's expert set by commitment-weighted router demand
(draft depth), -0.27 pp accuracy, offloaded 2.06x over standard SD, traffic -74..77 %. Not found: accepting confident
tokens unverified under partial expert coverage, which is where our bytes are saved.

## KLD calibration for "similar intelligence" (2026-10-21 10:30)
- Informal llama.cpp measurements: mean KLD +0.037 for Q3_K_M (top 5 % of tokens +0.26), +0.082 for Q2_K
  (top 5 % +0.71). Frugal 10 % (suite step KLD 0.057) sits between Q3- and Q2-class drift; ar4 (0.0096) is Q4-class.
- "Displacement Is Not Direction" (arXiv 2606.19558): KLD predicts benchmark damage only at the degraded end; among
  near-baseline variants the correlation collapses (disagreements can help or hurt): use fidelity metrics to filter
  damaged candidates, not to rank near-baseline ones. "Reliability Scaling Laws for Quantized LLMs" (2607.10855):
  3-bit KLD clearly above 4-bit, the behavioural shift shrinks with model size (1B -> 8B).
- So frugal's KLD flags a Q3-class drift that GSM8K (146/150) did not show; the Eval suite's decode rows (math, IFEval,
  HumanEval) are the deciding measurement, and policies near ar4's KLD (wait-to-85 %: 0.0094 / 0.0021) need no ranking.
