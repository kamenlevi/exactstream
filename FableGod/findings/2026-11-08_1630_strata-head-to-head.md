# Strata 0.1.58 head-to-head on the same 48 prompts (64 GB M5 Pro): our decode is ahead at about the same memory (code x2.4-3.7 certified lossy, chat +6 %), but Strata wins or ties every whole-request number up to 256 tokens, because its first token takes 0.5-1.0 s and ours ~3 s on chat
- Date: 2026-11-08 14:47-16:26 EET. Harness: ~/StrataEval/h2h (waiter + driver + report, resumable, quiet locks). Orchestrator: this
  note, checked against REPORT.md and the per-request rows. Our side: BigMoE's reports (S5', N45, CHS5, C32).
- Setup: Strata 0.1.58-metal (commit 146df3c), experts streamed from the SSD, its IQ3_XXS file (3-bit; its table puts the 4-bit
  variants at 72-88 GB, and we did not test them. CORRECTED 11-08: this note first said they do not fit 64 GB, but Strata
  streams what does not fit from the SSD and the 3-bit file is itself 75.8 GB). The same prompt ids as our S5 (12 HumanEval, 12 MBPP) and N45 (12 IFEval = chat,
  12 math) tests, greedy, 256 new tokens, EOS stop, one warm process per arm and pass.
- Arms: A default (hyper-connections read as Q8_0, lossy), no MTP; B = A + its MTP layer; C exact (STRATA_HC_Q8=0, the BF16 read),
  no MTP; D = C + MTP.
- Two cache sizes:
  - Default with desktop apps open: the arena sized itself to 25.3-27.7 GB, peak 33.5-34.9 GB (palindrome, 8 passes).
  - Full: STRATA_METAL_ARENA_GB=33, the size the engine picks on a free Mac (footprint cap 41.2 GB = 60 % of RAM). B and D, one pass
    each, peak 40.76 / 41.15 GB. These are the fair numbers against our 45 GB line (lifetime peak 41.7-42.9 GB).
- Decode [K], pooled whole decode tok/s (tokens / decode wall, prefill excluded), ours vs Strata at full cache:
  - CORRECTED 17:1x: ours are now pooled like Strata's, re-derived from the raw run records (MoonShots' main_table.py). The first
    version quoted our per-prompt geometric means and led with a build the certificate does not cover.
  - HumanEval: certified lossy 171.0-175.3 vs 47.0; exact 101.4 vs 46.4. (181.3 with the exact-safe bundle: not certified.)
  - MBPP: certified lossy 111.1-114.5 vs 45.6; exact 70.4 vs 43.6. (119.8 with the bundle: not certified.)
  - Math: lossy 86.9 vs 47.1; exact 77.9 vs 46.3 (ours from N45, an older config).
  - Chat (12 IFEval): lossy 44.2 certified vs 41.8; exact 38.3 vs 40.6. Ours is also 49.9 (M1-B + COH, IFEval passes, card gates
    pending) and 56.0 (stack, no quality gate yet).
- Whole request [K], pooled tokens / wall time including the wait for the first token, ours lossy vs Strata B at full cache:
  - Chat 24.4 (M1-B + COH) vs 33.7. HumanEval 36.0-36.6 vs 40.9. MBPP 25.3-25.8 vs 36.9. Math 39.6 vs 40.5 (ours pooled from the raw records; the first version used the reports' END TO END).
  - Strata's D at full cache: 33.0 / 40.6 / 35.8 / 39.9 (chat, HumanEval, MBPP, math).
- First token [K]: Strata at full cache, median 0.83 s chat, 0.67 HumanEval, 0.54 MBPP, 1.01 math. Ours on the same 12 chat prompts (37-101 tokens, M1-B + COH):
  median 2.92 s, range 2.34-3.81 s, a line of 1.7 s + 22 ms per prompt token. It is at most the prefill itself, and a persistent server repeats it
  on every request. BigMoE's estimate (CT1 will measure it): each prompt's own expert reads, about 16-35 GB from the SSD at ~8 GB/s.
- Break-even [derived from the decode and first-token numbers]: chat on the 49.9 config passes Strata's whole-request time at about 530 tokens of answer.
- Equal memory, ours at the 32 GB line (BigMoE C32, 12 IFEval, C 160) vs Strata at its default ~34 GB:
  - Lossy 51.3 tok/s at 31.2 GB (M1-B + COH + SB1, NOT gated at 32 GB) vs 37.3. Exact 33.8 at 30.6 GB (12 / 12 identical to one-token decoding) vs 35.8.
  - First token (medians) 2.99 s lossy / 2.83 s exact vs 0.98 / 1.06 s. Whole request 22.5 / 19.8 vs 29.5 / 28.2.
- Strata's own levers: its MTP layer gives +53 % decode at default cache (no-MTP 25.3 -> 38.7 pooled, all sets; 87 % of drafted tokens accepted). The Q8_0 lossy read adds only
  2.7 % over exact (B vs D 38.7 vs 37.7 pooled; 1.3-4.9 % per set). Cache size matters: 25-28 -> 33 GB is +12 to +21 % decode (B).
- Caveats:
  - Different files: Strata's 3-bit IQ3_XXS vs our Q4 store. Our lossy FAST config reads ~3.2 bits per weight on average (ranks 0-1 at 4 bits, 2-9 at 3; 2026-11-07_1730), and our exact mode (NAX6-N) uses the same tiers, so the comparison is close to like for like in weight size. It is not Strata's best case: on a 2.4-bit file, Splash's 1.4.0 README puts Strata near 73 tok/s and itself at 95 tok/s decode with prefill 1,312 tok/s at 32K on an M5 Pro 64 GB (self-reported, other prompts, not reproduced here). If those hold, they beat our exact code decode, all our chat decode and our prefill. Quality was not compared here, and at most 4 of 24 rows are token-identical to our exact reference.
  - Full-cache arms are one pass each. The default-cache passes repeat within 0.5-5.5 %.
  - Five heat pauses interrupted the driver. Rows overlapping a freeze: 0 dropped. Fans: cap 5000 at first, 6000-6500 later; B1 38.4 vs B2 38.2, so no fan effect.
  - Our whole-request numbers for code and math are from older tests (S5', N45), before the layer-major prefill.
  - Long prompts (1.6K+ tokens) were not compared; Strata's prefill speed above is on short prompts only.
  - The first report marked Strata's exact arms "WRONG MODE": a parser false alarm (the engine prints that line in both modes; the exact arms convert 0 matrices). Fixed.
- What it means: the first-token wait, not decode, is the largest gap a user would feel at these answer lengths. BigMoE's PFB + CT1 test targets it.
- Data: 2026-11-08_1630_strata_report_default.md and 2026-11-08_1630_strata_report_full33.md, beside this note.
