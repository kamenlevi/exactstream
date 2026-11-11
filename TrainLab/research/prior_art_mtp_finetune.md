# Prior art: fine-tuning a recursive MTP head for drafting (TrainLab, 10-22 03:15)

FastMTP (Cai et al., arXiv 2509.18362, Sept 2025; https://arxiv.org/html/2509.18362v1), MiMo-7B-RL:
- One MTP head with shared weights, used recursively for K = 3 drafts (a 7-draft variant explored); only the head
  trained (210.8M params), trunk / embeddings / lm_head frozen.
- Data: SELF-DISTILLED, i.e. the target's own sampled responses (T 0.6, top-k 20, top-p 0.95, max 4096) to 389.4K
  multi-domain prompts (42 % general, 18 % math, 13 % code, 27 % Chinese), deduplicated. Hundreds of millions of tokens.
- Loss: weighted hard-label CE, alpha_k = beta^(k-1) / sum, beta = 0.6. AdamW, lr 5e-5 cosine, warmup 5 %, batch 64,
  3 epochs (< 1 day on an H20 server).
- Ablation (speedup at K = 3): vanilla MTP 1.21x; fine-tuned on the original responses 1.67x; on self-distilled
  responses 1.81x; + vocabulary compression 2.03x.
- Acceptance per draft position: vanilla ~70 / ~10 / ~0 %, fine-tuned ~80 / 56 / 36 %.
  => most of FastMTP's gain is TEACHING RECURSION to a head trained for one step. The 125B's stock head already chains
  (Chat1000: P(first j right) 0.85 / 0.71 / 0.61 / 0.51 ...), so its headroom is domain adaptation, a smaller lever.
- Scale gap: FastMTP ~10^8+ tokens; the Mac does ~60-130K training tokens per 45-min turn on the 125B (trunk prefill
  dominates), i.e. <= ~1M tokens before account-1's end.

Also seen (not used): MTP-D "Self-Distillation for Multi-Token Prediction" (arXiv 2603.23911): self-distillation during
pre-training, not a drafter fine-tune. vLLM speculators docs (MTP, FastMTP-style weights beta = 0.6).
