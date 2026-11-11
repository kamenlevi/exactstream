# Smaller-model check for 350: Qwen3.6-35B-A3B fully in RAM. Leap, pre-registered Sat 2026-10-24 06:56
Asked by NewIdeas (06:12) for the orchestrator: "Is ~350 tok/s for one chat reachable with Qwen3.6-35B-A3B fully in
RAM, which the 125B cannot be (45 GB process rule)?"
**Kill line (NewIdeas): best code tok/s < 200, in RAM, exact, even with drafting -> no.**

## Already measured locally (before any new run; sources in brackets)
- Weights are local, so nothing to download.
  - Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed: 4-bit body g64 + INT4 MTP sidecar, 20 GB.
  - unsloth Qwen3.6-35B-A3B-MLX-8bit.
  - z-lab/Qwen3.6-35B-A3B-DFlash: 0.4B, in ~/SpecDecode/data/dflash.
- Bytes per token, 4-bit, 10K context: 2.00 GB, roofline 140 tok/s at 280 GB/s. Routed experts are only 0.57 GB of it
  (SpecDecode FINDINGS 5.1).
- Plain decode, 4-bit: 75 tok/s at 10K context (SpecDecode 5.1); 75-83 in the model survey (MODELS_2026-10-23.md).
  MTPLX's card says 94.5 AR on an unnamed Apple chip.
- Verify cost vs rows, Q8 (MLX 0.32.4): x1.28 (2), 1.67 (4), 2.40 (8), 2.65 (9), 3.56 (16), 3.71 (17) of 17.8 ms
  (SpecDecode 15).
- Drafting:
  - MTPLX card (sampling, reasoning on): D1 138 tok/s at 0.886 acceptance, D2 136, D3 108.
  - Kamen's agentic samples, Q8: lookup-else-MTP 95-99 tok/s, 1.7-1.9x (SpecDecode 15, 18). On the full 4-bit model
    with chunked wide verify: copy-heavy edit 284 tok/s, code 147 (SpecDecode 16, MoEPrefetch).
  - DFlash on Kamen's agentic samples, Q8: block 16 53 tok/s (4.1 accepted / block), block 8 70, block 4 79
    (SpecDecode 16).
  - DFlash on the 35B's own code continuations: tau16 13.4 (6 prompts, replay; Chat1000 DRAFTER.md). Never timed.
- Intelligence: AA index 18.2 vs the 125B's 39.8 (MODELS_2026-10-23.md). No Eval suite row for the 35B.

**Paper estimate.** z-lab's loop took ~96 ms per 16-row block at Q8 (53 tok/s at 5.1 tokens per block). At 4-bit,
~65-70 ms. Then code at ~13 tokens per block gives **~190 tok/s**, right at the line. The bytes roofline for a 16-row
pass would allow ~400. This MLX loop is far from it.

## The one measurement (one quiet lock, ~30 min, after L4b)
- The 35B 4-bit body (Youssofal) via mlx_lm, in RAM (~21 GB), greedy, with z-lab's own MLX DFlash loop.
  - Copied from SpecDecode's dflash_ab.py into ~/Leap. SpecDecode's repo stays read-only.
- Sets:
  - code: 12 HumanEval+ prompts (EvalSuite's tf_code items, re-rendered with the 35B's chat template), max 400 tokens;
  - agentic: SpecDecode's samples bash17 and edit00 (Kamen's turns, rendered by its render.py), max 256 tokens.
- Modes: plain greedy; DFlash block 16 and block 8.
- MTP / lookup numbers are taken from the measured rows above, not rerun.
- Exact = greedy with a batched verify. Outputs may differ from plain decode only at near-ties (SpecDecode 3). The
  first divergence from plain is reported.
- Reported: tok/s (decode, from the first token), accepted per block, peak memory.

## Predictions
- plain 4-bit: code ~85-95 tok/s, agentic ~75-85;
- DFlash block 16:
  - code ~190 tok/s (160-240), 12.0 accepted + 1 per block, P(>= 200) ~45 %;
  - agentic ~70 tok/s;
- DFlash block 8: code ~150;
- P(code >= 350 with any drafter here) < 5 %.

## What the answer will mean
- Under 200: the 35B does not open 350 either. Its gain over the 125B is RAM fit, not a new speed class.
- 200-350: code gets ~2-3x the 125B's 65-126 exact, at less than half the intelligence (AA 18 vs 40). Whether that
  trade is worth it is Kamen's call. 350 would still need kernel work (~1.5-1.8x).
- Agentic stays ~100-150 with MTP / lookup either way (DFlash is weak there).
