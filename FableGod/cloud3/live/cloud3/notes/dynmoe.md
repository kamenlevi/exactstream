# DynMoE: Dynamic Mixture of Experts, an auto-tuning approach (arXiv 2405.14297, ICLR 2025; Guo, Cheng, Tang, Lin et al.)
- Link: https://arxiv.org/abs/2405.14297 ; code https://github.com/LINs-lab/DynMoE. Source: WebSearch extracts [S].
- What: top-any gating as multi-label classification: each token activates every expert whose score passes a learned threshold; experts are added for unrouted tokens and removed when unused; a sparse-and-simple gating loss. Training-time.
- Numbers [S]: none transferable (their trained models).
- Maps to us: training-time only; the inference analogue (score threshold) is the KILLed gate-share threshold / DYNK-Q. Not applicable without retraining the router (TRIED "Retrained routers ... KILL: needs retraining").
- TRIED status: n/a (training).
- Open: none.
