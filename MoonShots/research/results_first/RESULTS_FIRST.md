# RESULTS-FIRST: would our numbers be "never achieved before"?

Wed 4 Nov 2026, ~16:00 EET. MoonShots, for the orchestrator (Kamen's question). This compares RESULTS, not methods; NOVELTY.md
covers methods. Sources:
- [EXTERNAL_PAPERS.md](EXTERNAL_PAPERS.md): papers and labs, 2023 to Nov 2026.
- [EXTERNAL_COMMUNITY.md](EXTERNAL_COMMUNITY.md): GitHub, Hugging Face, Reddit.

Both were researched today, with a source link and a confidence mark on every row. Single-user decode tok/s on consumer
hardware.

## Bottom line

- **There is no clean "first" on speed alone at 24 GB.** Our own model (Qwen3.8-Flash-Next, 125B total, ~5B active) has
  public SSD-streaming engines:
  - Kapitsa (30 Oct) and Splash run it at 56-61 tok/s with 2.4-bit experts and a 16 GB expert store, on the same M5 Pro
    chip class. Their routing is exact and their drafting is verified.
  - Our 24 GB exact line runs it at 33.2 tok/s with ~3.2-bit experts.
- **The likely first is at <= 12 GB, exact.** Our 12 GB line runs at 40.8 tok/s (full outputs; 11.35 GB lifetime
  footprint; F_NOCACHE reads, so no page-cache help), bitwise equal to its own tiered reference. The best published
  exact run at <= 12 GB is SlotStream: 1.48 tok/s (4-bit, 10.7 GB process target). Faster published runs near 12 GB are
  all lossier and come with caveats:
  - Palm-Infra: 122B, 12.4 tok/s at 5.9 GiB RSS on a 48 GB Mac, lossy.
  - Stock Splash: 28.8 tok/s at 2.4 bits, with an 8.8 GB expert store on a 24 GB Air; its total process memory is unknown.
  - The Splash fork: 48.8 tok/s, but lossy routing.
  - So the claim needs care; see "What would make it first".
- **Corrections to the brief:**
  - Memory-Sovereign Inference's 12.0 tok/s MTP run is a 27B model. Its 80B run is 11.6 tok/s at 34 GiB (11 host + 23
    GPU), so it is outside the <= 24 GB bucket.
  - The r/macmini run of our model (11 tok/s, "9.5 GB peak") was not found anywhere. Do not cite it.

## Same model: Qwen3.8-Flash-Next (125B), streamed from SSD

| who | experts' precision | exact? | memory | hardware | decode tok/s | date | confidence |
|---|---|---|---|---|---|---|---|
| **ours, 12 GB line (Fit12 B12LL two-pass)** | ~3.2 b (4-bit ranks 0-1, 3-bit ranks 2-9) | bitwise = its reference (T3t) | **11.35 GB lifetime** | MacBook Pro M5 Pro 64 GB | **40.8** (49.9 shared text), 16 code-heavy prompts | 11-04 | measured |
| ours, 12 GB FAST (P1-K1) | same | no (FAST) | 11.35 GB | same | 44.2 (57.3 shared) | 11-04 | measured; Fit12's line PASS |
| SlotStream | 4-bit MLX | exact | 10.7 GB process target (16 GB M2 mini) | Mac mini M2 | 1.48 | 09-03 | sure |
| Splash 1.4.0 (stock) | Q2_0, 2.4 b | routing exact, verified drafting | 8.8 GB expert store; total unknown | MacBook Air M5 24 GB | 28.8 | 10-25 | sure (self-report) |
| Splash fork | Q2_0 | no (cache-biased routing) | same | same | 48.8 | 10-25 | sure (self-report) |
| **ours, 24 GB line (W1B)** | ~3.2 b | = its stock-kernel reference | **23.86 GB lifetime** | M5 Pro 64 GB | **33.2**, 6 mixed prompts | 11-04 | measured |
| SlotStream | 4-bit | exact | 22 GB process target | MacBook Pro M5 Pro 48 GB | 15.86 | 09-16 | sure |
| SlotStream | 4-bit | exact | 17.4 GB target (24 GB machine) | MacBook Pro M4 Pro 24 GB | 5.41 | 09-25 | sure |
| Kapitsa | Q2_0, 2.4 b | routing exact, verified drafting | 16 GB expert store (48 GB machine; total unknown) | MacBook Pro M5 Pro 48 GB | **61.3** | 10-30 | sure (README) |
| Splash 1.5.0, measured by Kapitsa | Q2_0 | same | same | same | 56.5 | 10-30 | sure (competitor's measurement) |
| **ours, 45 GB class (C 224, new default)** | ~3.2 b | bitwise = REF2-T | 42.1 GB lifetime | M5 Pro 64 GB | **51.0** geo (14 prompts); **83.1** on code | 11-04 | measured |
| ours, 45 GB FAST (U3C, code) | ~3.2 b | no (FAST; EvalSuite pending) | 39.4 GB | same | 110.1 (code) | 11-04 | measured |
| SlotStream | 4-bit | exact | 33 GB ceiling | M5 Pro 48 GB | 23.69 | 10-05 | sure |
| Evolving Agent Teams (arXiv 2610.36848) | 4-bit? | exact by construction, not bit-verified | <= 37 GiB | M4 Pro 48 GB | 30.3 | 10 | unsure (prompt lengths) |
| Strata (CUDA) | IQ3_XXS 3.0 b | exact | 12 GB VRAM + 64 GB RAM (not memory-bound) | RTX 5070 + Ryzen | 44.8 | 09-24 | sure |
| Kapitsa, whole model in RAM | 4-bit | exact | 128 GB | M5 Max | 108 | 10 | sure |

## Other >= 80B MoEs (best rows)

| who | model | exact? | memory | hardware | tok/s | confidence |
|---|---|---|---|---|---|---|
| Palm-Infra (Tencent Youtu) | Qwen3.5-122B-A10B 4-bit | lossy | 14.64 GiB RSS / 5.90 GiB RSS | 48 GB Mac | 16.19 / 12.38 | unsure (RSS only; page cache likely helps) |
| Memory-Sovereign Inference (2608.23805) | Qwen3-Next-80B | verified exact | 11 GiB host + 23 GiB GPU | RTX 3090 | 11.6 | sure |
| PagedMoE (2611.03173) | Qwen3.5-397B | lossy | 24-64 GB | - | 8.89 | unsure (GPU memory not counted) |
| 2606.10493 / KTransformers | DeepSeek 671B | exact | 382 GB-1.15 TB DRAM + 1 GPU | server Xeon + RTX 5090 | 21.5 FP8 / 28 Q4 | sure (not memory-bound) |

No published >= 80B run at <= 24 GB is verified exact. None at <= 12 GB is exact. None counts the OS page cache in its
budget.

## Our lines, one by one

- **45 GB exact (51.0 geo, 83.1 code at 42 GB):** above every same-memory exact row (SlotStream 23.7 at 33 GB, EAT 30.3 at
  <= 37 GiB), but at ~3.2 b against their 4 b. Kapitsa's 61.3 at 2.4 b uses about half our memory. **Not a first:** the
  precision and memory differ, so it reads as "faster at more RAM".
- **45 GB FAST (110 on code):** the fastest number we know for this model below 64 GB. But it is code-only and its quality
  certification is pending. Kapitsa's whole-in-RAM 108 (4-bit, 128 GB) is a different class. **Not claimable until
  EvalSuite certifies.**
- **24 GB exact (33.2):** above every exact >= 3-bit row at <= 24 GB (SlotStream 15.9 at 22 GB, 4-bit). Below Kapitsa /
  Splash (56-61 at 2.4 b, same chip class). **First only in the narrow form "fastest at >= 3 bits per expert weight within
  24 GB"**, and only if our tiers' quality matches or beats GSQ-RCO Q2_0. Nobody has measured that.
- **24 GB FAST:** there is no live measurement, only replays. **Nothing to claim yet.**
- **12 GB exact (40.8 at 11.35 GB lifetime):** the best published exact run at <= 12 GB is 1.48 tok/s (SlotStream). Stock
  Splash is 28.8 at 2.4 b on a 24 GB Air, with its total process memory unknown. **This is the most likely genuine first:**
  the fastest output-exact decode of a >= 80B MoE within 12 GB. Three caveats:
  - Our prompts are code-heavy.
  - Our reference is ~3.2 b, not the original 4 b.
  - The hardware is an M5 Pro, not an Air.
- **12 GB FAST (44.2):** below the lossy-routing Splash fork (48.8, 8.8 GB store on a 24 GB Air). **Not a first.**

## What would make the claims "first", safely

1. **Same machine, same prompts, same caps.** Run Kapitsa and Splash on our M5 Pro at a 12 GB and a 24 GB expert / process
   cap, on our prompt sets. This needs their code and the GSQ-RCO Q2_0 pack (~66 GB): a download, so Kamen's and the
   orchestrator's yes first.
2. **Hard memory.** Show the whole process (not just the expert store) within the bucket, with F_NOCACHE (no page-cache
   help). Better: run the 12 GB line on a 16 GB Mac, or with the rest of RAM pinned, so the budget is physical.
3. **Quality at equal speed.** Put EvalSuite's same-scores test on our tiered reference against GSQ-RCO Q2_0 / IQ3_XXS on
   the same benchmark. "Faster at higher quality" is the claim that survives. "Faster" alone loses to 2.4-bit engines.
4. **Mixed prompts at 12 GB.** Re-measure the 12 GB line on W1B's mixed six, beside the 24 GB line, so one prompt set covers
   both.

## Unsure rows (from the two source files)

Palm-Infra (RSS only); Evolving Agent Teams (prompt lengths); PagedMoE (GPU memory not counted); Splash / Kapitsa total
process memory (only expert stores stated); Kapitsa's prompt set; Flash-Next's parameter count (sources say 125B, 177B or
180B; our config gives ~125B total, ~5B active); BigMoMo, Colibri, flash-moe, RotaryQuant and others (see the two files).
