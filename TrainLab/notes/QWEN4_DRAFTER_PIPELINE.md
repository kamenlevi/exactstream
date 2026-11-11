# The fastest drafter pipeline for a new model (Qwen 4) (MTP Head, Sun 2026-11-08; the orchestrator's CHAT150 part 2)

Plain answer: on day 0 we draft with the model's own MTP layer (no training). On days 1-2 we fine-tune that layer on the
new model's own features with our D-22 recipe, on the Mac. A block drafter (DFlash2-style) comes later: either a public
one (z-lab has shipped Qwen drafters 1-20 days after release) that we fine-tune for an hour, or our own, which takes weeks.
Free cloud can train small heads only for small dense variants; a big MoE target has to run on the Mac.

Our 125B is `qwen4_exp`. It matches Qwen3.8-Flash-Next, whose model card calls it "an experimental preview of the
architecture that will underpin Qwen4": a 125B MoE, 512 experts, one MTP layer. Every step below already runs on that
architecture here.

## Timeline from weights to a trained head

| when | step | compute | what it gives (our 125B as the guide) |
|---|---|---|---|
| day 0, hours | convert to MLX 4-bit; run the built-in MTP layer through the MTPLX path (K15 chain, 0.3 cut) | the Mac, no training | chat 2.3-2.5, code ~7.8 kept per check (stock head on our 125B) |
| day 0 | look for public drafters (z-lab DFlash / DFlash2, community EAGLE-3) | none | Qwen releases got a z-lab drafter after 1-20 days (+1 d for Qwen3.8-27B, +2 d for Qwen3.6) |
| day 0-1 | capture features: teacher-forced prefill of a prompt + answer mix, keeping the model's own top-32 per position (sticky/f2a_capture.py) | the Mac, ~100-120 tok/s at 45 GB (BigMoE's PFX would give ~500) | ~0.4M tokens per hour |
| day 1 | MTP fine-tune: D-22's training-time-test chain + accept-until-fail, from the shipped MTP weights (sticky/f2a_train.py) | the Mac, ~10-15 GB, 1-3 h | on our 125B: code x1.37 kept (7.78 -> 10.65) from ~250 windows; chat +1-3 % |
| day 1-2 | block drafter: fine-tune a public DFlash / DFlash2 drafter with L1-20's recipe (pmask 0.5) on code + math answers | the Mac, ~1 h at ~37 GB per probe | L1-20: 9.57 HumanEval+ / 5.94 MBPP kept per check |
| week 1-3 | block drafter from scratch | weeks of features + training | published recipes use 0.8-2.4M regenerated samples x 6 epochs; 5K samples gave 1.47 on MT-Bench (too little) |

## Data
- **Prompts:** code (HumanEval / MBPP style), math (GSM8K / MATH), chat (UltraChat, Aya for languages, IFEval-like
  constraints), agent turns (Chat1000).
  - The UltraChat + Aya corpus is already on disk (ext/chatcorpus, MIT / Apache-2.0).
- **Answers:** the model's own text is best, because published recipes regenerate answers with the target. At decode
  speed that costs ~0.2M tokens per hour on a big MoE.
  - The fast route is the F2a one: teacher-force existing answers and keep the target's own top-32 as labels.
- **Volume per night:** ~1.5-2M tokens of features at ~100-120 tok/s, ~6-8M with a ~500 tok/s prefill.
  - Features take 20 KB per token (the 10,240-wide stream the MTP layer reads), so a night's ring is 30-40 GB.

## Compute (free only)
- **The Mac** (64 GB) is the only machine that holds a big MoE target. So features and training for it run here, as
  online / ring training: capture, then train, then delete.
- **Free cloud** (Kaggle 2x T4 or P100, ~30 h a week, 20 GB disk; Colab free; Lightning's ~50 interruptible GPU-hours,
  card needed) fits:
  - (a) small dense Qwen 4 variants (<= ~8B at 4-bit), with the target and the head on the same machine (online);
  - (b) training a head from uploaded feature shards of <= ~0.5-1M tokens.
- **Our PyTorch port of the MLX trainer** (PORT, validated Fri 11-06; equal within 1.5e-3 with TF32 off) is what runs
  there.
- **No paid GPUs.** AMD's $100 developer credit and Modal's $30 credit need a card, so they would need Kamen's call.

## Risks and checks
- **Architecture drift.** If Qwen 4 changes the MTP layer (several layers, other inputs) or its sparse attention (QSA),
  the MTPLX path and our verify need a port first.
  - Inco limited its DFlash2 for Flash-Next to block 4 because of QSA. Our engine verifies 16 rows on the same
    architecture, so check the new attention before assuming 16.
- **Chat is the weak domain everywhere.** Published chat acceptance is 4.2-5.2 tokens per 16-token chain on small dense
  targets, ~0.8 per token. Our 125B's MTP layer starts at 0.76 per token on IFEval.
  - Expect chat gains from fine-tuning to be small, and code / math gains to be large (x1.2-1.4).
- **Exactness.** Drafts never change outputs under exact acceptance, so a new head needs only a speed A/B. Lossy
  acceptance rules need EvalSuite's gate.

Sources: research brief (Sun 00:0x; Qwen model cards, z-lab / Inco cards, EAGLE-3 arXiv 2503.01840, DFlash 2602.06036,
DFlare 2606.02091, SpecForge, vLLM Speculators docs, Kaggle / Colab / Lightning docs); REPORT D-22, D-24, CHAT150, D1,
F2A; PORT RESULT (Fri 11-06).
