# Bet L5 = QUEUE L1-04 stage 2: DFlash2 extended to block 16 by a local fine-tune on code / math. Leap, pre-registered
# Sat 2026-10-24 02:42, before any data was generated or any training run
Owner: Leap (QUEUE.md L1-04). The CPU stage passed at 02:42: projected code tau16 10.2-11.9 >= 9
(QUEUE_L1-04_L1-06.md). All tau use the fixed counting (bonus counted; DFL2D_PREREG.md).

**Question.** Can a short local fine-tune teach the block-8 DFlash2 drafter to draft depths 8-15 well enough that a
16-row pass carries >= 9 tokens on code? Today block 8 gives 6.93 at most-8.

**Prior art (tag: published, untested here).**
- DFlash's own drafters use block 16 (z-lab). Extending the block of a trained block-diffusion drafter by fine-tuning
  is not something I have seen reported.
- Self-distillation on target outputs is standard (EAGLE-3 etc.).
- Nothing new is claimed.

**Data** (local, no downloads; deleted after the verdict except the weights if it lives):
- Prompts the evaluation never uses, rendered exactly as EvalSuite's tf set. The check passed: 24/24 tf_code prompts
  are identical (dfl2e_data.py).
  - Code: the 80 HumanEval+ problems outside EvalSuite's 84-problem pick (local evalplus cache).
  - Math: GSM8K test problems 0-119.
- The 125B's own greedy outputs:
  - spec125.generate, exact, max 400 tokens as EvalSuite, stopping at the end of the turn;
  - C = 192 experts resident (the 24 GB setting) for speed;
  - <= 50K tokens or 40 min.
- Labels and features: dfl2c_extract on prompt + output (top-32 labels; on-path ~1 on the 125B's own text).
- Split: 90 / 10.

**Training.**
- dfl2c_train.py with --bs 16 (depths 1-15). Everything else as L3 / L4: frozen fc / hidden_norm / head; the 5 layers
  + norm + selector trained; soft CE + selector CE.
- AdamW lr 2e-5 cosine; 2 epochs if within 30 min, else what fits. Micro-batches of 1 group (block 16 doubles the
  rows).
- One run, no sweep.

**Evaluation.**
- L2 replay (official loop logic) at block 16 on the held-out tf_code (24 HumanEval+) and tf_gsm (60), with the
  features already extracted tonight.
  - original at block 16 (depths 8-15 untrained: the baseline);
  - fine-tuned at block 16;
  - fine-tuned at block 8 (is block 8 kept?).

**Lines.**
- KILL: fine-tuned code tau16 < 9.0 (NewIdeas' line, now measured).
- INTERESTING: >= 10.
- Block-8 guard: fine-tuned code tau8 < 6.6 (-5 % vs 6.87) means it is a separate block-16 drafter.
- What it is worth (not a line): a 16-row pass must cost < tau16 / 6.87 of an 8-row pass. The routes give
  T16/T8 ~1.35.

**Predictions.**
- Fine-tuned conditional acceptance at depths 8-15: ~0.82 (0.75-0.89). L2's depths 2-7 are 0.956, but these
  positions are new.
- Code tau16 ~9.6 (8.8-10.4). P(>= 9) ~ 65 %, P(>= 10) ~ 35 %.
- GSM8K tau16 ~7.0.
- Original at block 16: ~7.0-7.3 (little beyond depth 7).
- Fine-tuned at block 8: ~6.8.

**Cost: one unattended batch, ~1.5 h.**
- Generation <= 40 min (C = 192, ~24 GB).
- Extraction ~5 min (short prompts).
- Training <= 30 min.
- Eval ~5 min.
- Peak <= ~30 GB, swap and HEALTH guarded.
- Disk: features ~0.6 GB, weights 1.1 GB.

**Amended Sat 02:51, before any run:** generation uses C = 128, BigMoE's documented 24 GB mode (22.2 GB MLX peak,
~27-29 tok/s exact; HANDOVER.md), not C = 192, whose footprint on this 4-bit model is undocumented. Speed only; nothing
measured changes.

## Amendment 2, Sat 10-24 04:41: decision lines from Fit12's T16/T8 (before any fine-tune number)
Written after the batch's first step printed the ORIGINAL drafter at block 16 (code 8.88, GSM8K 6.56; predicted 7.0-7.3,
so that prediction missed low) and before any fine-tuned number exists. The change comes from Fit12's 04:00 note, not
from the baseline: T16/T8 ~1.5-1.65 (from its measured frugal 4 -> 8 step at C192: stall x1.45, compute x1.75; T8 = 173
ms), against my ~1.35 when I set the 9.0 line. Block 16 pays only if tau16 >= (T16/T8) x tau8 = 1.5-1.65 x 6.87 =
10.3-11.3 (replay, fixed counting; Fit12's real/replay factor ~0.8 hits both block sizes alike).
- KILL if the fine-tuned code tau16 < 10.3: block 16 loses to block 8 even at T16/T8 = 1.5. (Replaces the 9.0 line.)
- PASS if >= 11.3: block 16 wins even at 1.65; next = lift the engine's 9-row routing limit (gated routing row <= 92
  ids) and measure a real T16.
- 10.3-11.3: undecided until a real T16 exists; reported as such.
- Unchanged: block-8 guard (fine-tuned at block 8 < 6.6 = it hurt the short block), GSM8K reported, not decisive.
- The original drafter at block 16 already gives x1.29 tokens per pass on code (x1.15 GSM8K): below 1.5, so as is it
  does not pay.
- Registered prediction unchanged (code ~9.6, 8.8-10.4). Updated expectation after the higher baseline (NOT a
  pre-registration): ~9.9 (9.0-10.8), P(>= 10.3) ~30%, P(>= 11.3) ~5%.
