# Harder Tasks Need More Experts: Dynamic Routing in MoE Models (arXiv 2403.07652, ACL 2024; Huang et al.)
- Link: https://arxiv.org/abs/2403.07652 ; code https://github.com/ZhenweiAn/Dynamic_MoE. Source: WebSearch extracts [S].
- What: TRAINING-TIME top-p routing: add experts in probability order until the cumulative probability passes p; a dynamic loss discourages unnecessary experts. Finding: lower layers need more experts, the top layer needs ~1.
- Numbers [S]: +0.7 % average over top-2 while activating < 90 % of the parameters (their own trained models).
- Maps to us: the inference-only version is TRIED L1-17b adaptive-k p 0.8 (KILL: union -24 % at 1.43x MSB-3's error, failed the suite) and L1-17 b12llka80 (KILL), R-26 DYNK-Q tau 0.06 (KILL), Gate-share threshold (KILL, a cliff: shares bunch at 0.05-0.15). Our router is flat (top-8 = 88 % of top-10 mass), trained for a fixed top-10 with no sparsity loss, so cumulative-mass rules cut mid rows as hard as sure rows. The per-layer finding (top layers need fewer) is the one transferable piece: DM P-frame data here says layers 8-11 and 12-35 tolerate even full reuse on sure rows, 0-7 / 36-47 do not.
- TRIED status: inference form KILL; training form not applicable (no retraining of the 125B).
- Open: none for inference-only.
