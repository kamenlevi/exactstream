# Bet L2: the native DFlash2 drafter on our 125B. Pre-registered Fri 2026-10-23 15:35, before any GPU run.

**Question.** incoai/Qwen3.8-Flash-Next-DFlash2 is a native block drafter for our model:
- one pass drafts 7 tokens (block 8), with top-16 candidates per position and a path selector;
- it reads target features at layers 5/15/25/35/45, the mean of the 4 hyper-connection streams.

How many tokens per verify pass does it give with OUR engine's greedy target, against the native MTP chain?
- Tag: published (Inco AI, Oct 19), untested here.
- Prior art: DFlash (2602.06036, ICML'26); DFlash 2 card (block 4 only, T=1, GB300). Card acceptance MTP vs DFlash2:
  HumanEval 2.52 vs 2.94, MT-Bench 2.50 vs 2.83, GSM8K 3.08 vs 3.26.
- Our own record: Chat1000's retarget of the 35B DFlash (W_c only) gave tau16 6.83 code / 2.06-2.11 agentic.

**Method (teacher-forced replay of the official MLX decode loop, z-lab/dflash model_mlx.py, greedy).**
- Features: the 125B prefill on BigMoE's streamed runtime (copy of Chat1000's wc_extract.py, taps 5/15/25/35/45,
  HC mean, fp16).
- Text: the 125B's own greedy continuations. Eval suite tf_gen: 24 code, 60 GSM8K, 30 IFEval, 24 multilingual. Plus
  Chat1000's 12 real agentic turns (160 tokens each, 8-10K-token prompts).
- Each block, as in the official loop:
  - The drafter's cache holds the features of all committed positions before the anchor.
  - The block is [anchor, MASK x 7]; the drafts are scored against the stored continuation.
  - The next anchor is placed after (accepted + 1) tokens.
- tau8 = generated tokens / blocks.
- Comparator, on exactly the same agentic tokens: the native MTP chain at K = 7. It is scored from Chat1000's stored
  teacher-forced chains (agent125/*.npz `mtp`), with the same block replay.
- Extra: coverage of the true token in the drafter's top-16 candidates (a tree bound, for the A1 follow-up).

**Prediction.**

| class | tau8 (range) | vs MTP K=7 |
|---|---|---|
| code (HumanEval+) | 5.0 (4.0-6.0) | |
| GSM8K | 4.8 (4.0-5.6) | |
| IFEval | 3.2 (2.5-3.9) | |
| multilingual | 3.3 (2.6-4.0) | |
| agentic | 3.6 (2.8-4.4) | ~0.95x (0.8-1.15x) |

- Reasoning:
  - DFlash v1 on Qwen3-8B HumanEval: tau16 6.5, about 5 at block 8.
  - The card's +6-17 % over MTP at block 4 is measured on chat / code benchmarks at T=1.
  - Agentic text is where every drafter so far lost to MTP: retarget 2.1 vs MTP 4.0 at K=15.

**Kill / adopt lines.**
- KILL as a drafter lead (it stays a note): agentic tau8 < 1.05x MTP K=7 on the same turns AND code tau8 < 5.0.
- ADOPT-CANDIDATE for the main track: agentic >= 1.10x MTP K=7, OR code >= 5.5 (beats the retarget's per-pass rate
  at half the pass width).
  - Owner if adopted: BigMoE / SpecDecode. An MLX engine integration needs the 5-tap capture in the verify pass.
- Either way, the 350 table gets tau8 at 8-row passes in place of the assumed tau16 drafter.

**Cost.** One quiet-lock batch, ~15-25 min.
- Load the 125B streamed (C 51, < 36 GB) and prefill ~160K tokens.
- Features ~2.3 GB on disk (free 120 GB, floor 110).
- The drafter replay runs afterwards on the GPU without the 125B: embedding + LM head + the 0.6B drafter (~3 GB).

## L2b addendum (pre-registered 15:58, before the run): full-context window on agentic + hybrid bound
**Question.** Does DFlash2's agentic loss come from its 2,048-token window? Test at inference only, no training:
- re-capture the 12 agentic turns with ALL context rows;
- override sliding_window to 16,384 (config + every attention layer);
- replay as before.

**Also: a hybrid bound.**
- DFlash2 and MTP run lengths at EVERY anchor. The drafter cache depends only on the committed features, so drafts are
  history-independent.
- Block replays for two policies:
  - oracle best-of;
  - cross-check: DFlash2 when its first draft equals MTP's, else MTP.

**Prediction.**
- Full window: first-token acceptance .64 -> .66-.72; tau8 2.86 -> 2.9-3.3, still below MTP 3.54 (x0.83-0.93).
  - Reason: trained at a 2,048 window; longer RoPE distances are out of distribution.
- Cross-check hybrid: 3.6-3.8 (+2-7 % over MTP).
- Oracle: 4.0-4.3.

**Lines.**
- Full window >= 1.05x MTP: a training-free agentic win (adopt-candidate).
- Full window < 0.95x MTP: the window is not the main cause; the drafter stays code-only.
- Cross-check >= +5 % over MTP: worth building as the per-block policy.

## L2b, as assigned by NewIdeas (16:26). Final design, registered Fri 17:30, before any L2b GPU run
NewIdeas' bet: "does a longer drafter context fix agentic acceptance?"
- Cheapest test: the same replay, with the longest window the architecture accepts, no training.
- **KILL (NewIdeas' line, primary):** tau8 at window 16,384 stays below MTP K=7's 3.54 on the 12 turns.
  - At 16,384 every turn is seen in full: the longest is 10,770 tokens.
  - The architecture accepts any window: RoPE base 10,000 with max_position_embeddings 262,144.
  - The window was 2,048 in training, so distances beyond that are out of distribution for RoPE.
  - Pass -> NewIdeas' next step: a full-context fine-tune (0.5B drafter, small job).

**Same batch, so a kill can be read correctly (each ~1 min of GPU; features are captured once).**
1. **In-distribution sweep: windows 256 / 512 / 1,024 / 2,048.** Does the drafter use context within its trained
   range? If tau8 is flat from 512 to 2,048 (< 5 % change), the extra context does not help, and a full-context
   fine-tune is not supported by this evidence.
2. **Out-of-distribution windows 4,096 / 8,192 / 16,384**, plus 16,384 with NTK-aware RoPE (base x 8^(128/126),
   about 82,700; training-free, published by bloc97 in 2023). This guards against a false kill caused only by unseen
   RoPE distances.
3. **Hybrid bound (NewIdeas' point 2):** per-anchor runs of DFlash2 and MTP at window 2,048 and at 16,384. The
   cross-check and oracle predictions and lines are as above.
4. **Reproducibility check:** window 2,048 on the new full-context features must give L2's per-turn tau8 (2.86 overall).
   A mismatch means the prefill is not deterministic and is reported.

**Known before the run, from L2's per-turn output.**
- write14 is the only short turn (1,945-token prompt), so the 2,048 window already saw all of it.
  - It still loses to MTP by the same ratio: 1.72 vs 2.12 (x0.81; overall x0.81).
- IFEval and multilingual prompts also fit inside the window, and they are weak too (2.48 / 2.09).
- Also, the drafter's context rows are the target's own hidden states, which already carry the full context.
- My expectation is therefore below the 15:58 range: full window 2.6-3.0, NTK 2.7-3.0, i.e. KILL.
  - Sweep: 512 -> 2,048 changes tau8 by < 5 %.
  - If so, the agentic gap is the drafter's training mix (code / math / reasoning), not its window. Then the lever is a
    fine-tune on the 125B's own agentic outputs, not a longer context.

**Cost.** One quiet lock, ~15 min.
- The 125B prefill of the 12 turns: 104,141 tokens, all rows kept, ~2.7 GB in data/dfl2full (free 118 GB, floor 110).
- Then 10 drafter replays.
