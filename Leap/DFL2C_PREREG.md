# Bet L3: DFlash2 fine-tuned on agentic contexts with regeneration-free labels. Pre-registered Fri 2026-10-23 18:46,
before any data was built or any GPU run
Assigned by NewIdeas (18:24).

**Question.** Fine-tune the native DFlash2 drafter (0.59B) on text from Kamen's Claude Code sessions.
- Labels: the 125B's own top-k at every position, from a teacher-forced prefill. No generation runs.
- Features: the 125B's taps on the same text.
- Does tau8 on the 12 held-out agentic turns beat MTP K=7 (3.54 on the same tokens)?

**Prior art (tag: published method, untested here).**
- EAGLE-1 (arXiv 2401.15077) trained its drafter on fixed ShareGPT text, not target regenerations. Its ablation
  (Table 6) found target-generated data helps only marginally.
- Soft target labels on fixed data: the DistillSpec setting.
- Chat1000's check (via NewIdeas) found Spec-AUF, GRIFFIN, CTC-drafter, DistillSpec and DSpark nearby, not this.
- What is ours: DFlash2 (block diffusion + path selector), agentic sessions, and one detail:
  - The label at depth j is used only where the text tokens at depths 1..j-1 equal the 125B's greedy tokens.
  - That is exactly the condition under which depth j matters at decode time.
  - We don't claim this detail is new: no novelty search was done.

**Data.**
- Kamen's sessions (~/.claude/projects): SpecDecode's message loader (extract.py) and prompt format (render.py:
  SYSTEM + TOOLS, the 125B chat template, thinking off), i.e. the format of the eval prompts.
  - One window per session, the last <= ~28K characters (as route4), across projects.
  - ~250K tokens: ~90 % train, ~10 % validation, split by session.
- Excluded: every session holding any assistant message of SpecDecode's 29 route4 samples, matched by content. This
  covers the 12 held-out turns and their resumed copies.
- Anchors: positions inside assistant turns. Tool results and user text are context only.
- Data and weights stay on this Mac.
  - A later cloud run would upload features derived from Kamen's sessions, so it needs his explicit yes first.

**Training (local, Kamen's rule: well under 1B, 1-2 h, process under ~45 GB, swap watched).**
- Init: incoai weights.
- Frozen: fc + hidden_norm (their outputs are precomputed in the prefill, 2,560 fp16 per token, so features cost
  ~1.3 GB, not ~6.4 GB), and the 125B's 4-bit embedding / LM head.
- Trained: the 5 layers, the final norm, the candidate selector.
- Loss, at block depths 1-7 with the on-path weights:
  - soft cross-entropy to the 125B's top-32;
  - plus the selector's cross-entropy over its 16 candidates, with the true predecessor.
- AdamW. Window 2,048, as trained.
- Wall time <= 75 min. One run, no sweep: the hyper-parameters are fixed before the run (lr 2e-5, cosine, 1 epoch
  unless the measured step time allows 2).

**Evaluation.** The L2 replay: official loop, unmodified vendored code, greedy, window 2,048.
- The 12 held-out agentic turns: tau8, first-token acceptance, P(first j right).
- The hybrid bound vs MTP.
- Original and fine-tuned weights, in the same batch.
- Guard: code (24 HumanEval+ sequences): tau8 6.15 before the fine-tune.

**Lines (NewIdeas).**
- KILL: held-out agentic tau8 < 3.54.
- INTERESTING: >= 3.9. Then a bigger run goes to the free cloud tier once Kamen sets it up and OKs the data upload.
- Code guard (mine): code tau8 < 5.5 after the fine-tune means it is agentic-only (two drafters).
- Checks: the original weights must reproduce 2.86 agentic / 6.15 code. The training forward must match the official
  forward on the same blocks (argmax agreement >= 99 %) before training starts.

**Prediction.**
- The 125B's top-1 equals the session text on ~0.70 of assistant tokens (0.6-0.8).
- Validation depth-1 accuracy: original ~0.60 -> ~0.70.
- Held-out agentic tau8: 2.86 -> **3.35** (3.0-3.7). First token: .64 -> .72 (.68-.78).
  - P(>= 3.54) ~ 30 %; P(>= 3.9) ~ 10 %.
- Code guard: 6.15 -> ~5.9 (5.5-6.2).
- Reasoning:
  - L2b put most of the gap on the training mix, and domain fine-tunes usually lift drafter acceptance 10-30 %.
  - But: ~60-80K anchors is little data; the labels come from Claude's text, while the eval texts are the 125B's own
    outputs; and deep-depth labels thin out as 0.7^(j-1).

**Cost.**
- CPU: the data build.
- GPU 1, one quiet lock, ~25 min:
  - the 125B prefill of ~250K tokens (~12 min), storing h_ctx + top-32;
  - the eval and code features (KEEP 2,055, ~6 min);
  - the forward check and step timing.
- GPU 2: the training run (<= 75 min). Slot via the orchestrator.
- GPU 3: the eval, ~3 min.
- Disk: ~1.3 GB features + ~0.9 GB eval / code taps + ~1.1 GB weights. Free 115 GB now, floor 110. All deleted
  after the verdict except the fine-tuned weights, if it lives.
