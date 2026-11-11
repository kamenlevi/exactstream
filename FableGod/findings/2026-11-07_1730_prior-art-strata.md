# Prior art on our exact model: Strata (open source, MIT) runs Qwen3.8-Flash-Next with MTP speculative decoding; a user report gives ~110-120 tok/s on an M5 Max and ~27 tok/s on an M1 Max, both at Q2_0
- Date: 2026-11-07 17:3x EET. Orchestrator (Kamen pointed to a Reddit post; web check). Not yet in MoonShots' results-first
  comparison (2026-11-04), which must add it.
- What it is: github.com/Niko1221/Strata. An inference engine for Qwen3.8-Flash-Next and others (MIT; installers 0.2.9 for
  Windows, Linux, macOS; Metal backend on Macs). It takes community quantizations: Q2_0, IQ2_XS, IQ3_XXS, IQ3_S (ISTA-DASLab),
  and UD-IQ4_XS to Q8_0 (Unsloth). The README publishes no quality numbers. SSD streaming is described only for DeepSeek.
- User report, issue #2844 (not vetted). M1 Max 64 GB, Strata 0.1.44.1-metal, Q2_0, `--spec 2 --mtp rt`:
  - decode 26.7 tok/s short chat, 28.6 agent turns;
  - prefill ~226-234 tok/s;
  - engine RSS 40.3 GB;
  - the report cites the M5 Max at 119.5 / 108.2 tok/s decode (same defaults): ~4.5x the M1 Max with ~1.5x the
    bandwidth (614 vs 400 GB/s), so compute matters as well as bandwidth.
- Against ours (same model, similar footprint):
  - Our 45 GB FAST: 175.9 tok/s whole decode on 12 HumanEval and 118.3 on MBPP12, at 42.9 GB, on an M5 Pro (307 GB/s).
    Weights ~3.2 bits (ranks 0-1 at 4 bits, 2-9 at 3 bits) vs Strata's ~2 bits. Quality gated vs our exact base.
  - Per GB/s of bandwidth, ours does ~0.4-0.57 tok/s on code vs Strata's ~0.18-0.19 on chat. Workloads differ (code vs
    chat), so this is not like for like.
  - A same-machine head-to-head (Strata on this M5 Pro, same prompts, whole decode, quality) is the fair comparison. It
    needs a download (Kamen's OK).
- Sources: https://github.com/Niko1221/Strata ; https://github.com/Niko1221/Strata/issues/2844 ; M5 Pro 307 GB/s
  (https://en.wikipedia.org/wiki/Apple_M5).
