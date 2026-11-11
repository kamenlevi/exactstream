# Not All Experts are Equal: Efficient Expert Pruning and Skipping for MoE LLMs (arXiv 2402.14800, Feb 2024; Lu, Liu et al.)
- Link: https://arxiv.org/abs/2402.14800 ; code https://github.com/Lucky-Lance/Expert_Sparsity. Source: WebSearch extracts [S].
- What: post-training expert pruning (remove experts permanently by a calibration criterion) and dynamic expert skipping (skip the second expert of top-2 when its weight is small relative to the first) on Mixtral 8x7B.
- Numbers [S]: skipping + pruning matches the speedup of dropping 4 of 8 experts at much higher quality; Lynx reports NAEE below 40 % GSM8K at 1.5x (Lynx > 55 %); a later analysis: the skipping rule is dataset-tailored and does not extend beyond top-2.
- Maps to us: pruning is disallowed (Kamen: all 512 experts; REAP-288 HumanEval 93.9 -> 91.5). Skipping-by-relative-weight on a top-10 flat router = the gate-share / near-tie family (KILL). The "rank-0 dominates" fact is shared with Lynx and with our Q1.
- TRIED status: KILL (pruning: policy; skipping: L1-17b / R-26 / gate-share).
- Open: none.
