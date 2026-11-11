# docbench result: Qwen3.5-9B Q4 vs Q8, and the 9B vs a 27B in the same RAM (2026-10-25)

## Verdicts (pre-registered rules, `docbench/PREREG.md`, frozen 20:45 before any comparison run)

- **Claim 1, "9B at Q4 is much worse than at Q8": FALSE.**
  - 9B Q8 − 9B Q4 = **+0.43 points** of field accuracy (91.9 % vs 91.5 %). 95 % CI −2.0 to +2.5; McNemar p = 0.67 (27 fields right only at Q8, 23 only at Q4, of 930).
  - mlx_lm's mixed 4-bit (`mixed_4_6`, closer to llama.cpp Q4_K_M) scores 92.0 %: −0.11 points from Q8, p = 1.
  - The 4-bit method makes no measurable difference: mixed − plain = +0.54, p = 0.38.
  - The verdict does not hinge on the one document where the BF16 and Q8 wrote broken JSON. Without it, Q8 − Q4 = +1.30 points, CI −0.2 to +2.8, p = 0.09: still FALSE.
  - The expected check holds: **9B Q8 = 9B BF16** (difference 0.00 points, CI ±0.3).
- **Claim 2, "9B at Q8 is much better than a bigger model in the same RAM": FALSE. The opposite holds.**
  - Kamen's Qwen3.8-27B at Q4 (17.4 GB peak, less than the 9B's BF16 19.1 GB) scores **96.5 %** against the 9B Q8's 91.9 %: 9B Q8 − 27B Q4 = **−4.52 points**, CI −7.1 to −2.7, McNemar p = 1e-9 (5 fields right only at 9B Q8, 47 only at 27B Q4).
  - Against the 27B 6-bit as is: −5.38 points, p = 8e-13.
  - Documents with every field right: 62 % (9B Q8) against 78 % (27B Q4) and 83 % (27B 6-bit).
  - The 27B Q4 was made from Kamen's 6-bit file, one extra rounding step compared with a 4-bit made from BF16. It still won.
> (a quote from the project owner, kept private)

## Table (150 documents, 930 fields; field accuracy %, per type and language)

| config | field acc | docs all right | JSON valid | invoice | receipt | form | statement | contract | en | de | fr | bg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9B BF16 | 91.9 | 62.7 | 99.3 | 95.2 | 99.4 | 93.3 | 71.7 | 90.8 | 92.5 | 91.9 | 96.8 | 80.6 |
| 9B Q8 | 91.9 | 62.0 | 99.3 | 95.2 | 99.4 | 93.3 | 72.5 | 90.0 | 92.5 | 91.9 | 96.8 | 80.6 |
| 9B Q4 | 91.5 | 59.3 | 99.3 | 97.4 | 98.9 | 90.8 | 75.0 | 85.0 | 91.7 | 91.9 | 95.2 | 85.5 |
| 9B Q4 mixed_4_6 | 92.0 | 58.7 | 100 | 97.8 | 98.9 | 91.2 | 77.5 | 85.0 | 91.9 | 93.5 | 93.5 | 90.3 |
| Qwen3.8-27B Q4 | **96.5** | 78.0 | 100 | 100 | 99.4 | 98.8 | 80.8 | 95.0 | 96.6 | 95.2 | 96.8 | 95.2 |
| Qwen3.8-27B 6-bit | **97.3** | 83.3 | 100 | 100 | 99.4 | 99.2 | 86.7 | 95.0 | 97.6 | 95.2 | 96.8 | 96.8 |
| Qwen3.6-35B-A3B Q4 (bonus) | **97.0** | 84.0 | 100 | 100 | 99.4 | 97.9 | 84.2 | 97.5 | 97.0 | 98.4 | 96.8 | 95.2 |

JSON valid = a strictly valid object, with or without a markdown fence. Broken JSON occurred once: invoice-29 (Bulgarian) in the 9B BF16 and Q8, which put an ASCII `"` inside the vendor's „…“ quotes. That scores all 9 of its fields as wrong.

## Speed and memory

Measured at batch 1 under a quiet lock (each granted at once): a 1,289-token prompt (the longest document), 256 greedy tokens, median of 3 reps (spread ≤ 1 %). Peak = MLX peak memory during that run (weights + cache + activations). s/doc = this set's mean 762 prompt + ~94 output tokens at those speeds.

| config | decode tok/s | prefill tok/s | peak GB | weights GB | s/doc |
|---|---|---|---|---|---|
| 9B BF16 | 17.6 | 1,504 | 19.1 | 17.9 | 5.8 |
| 9B Q8 | 32.2 | 1,536 | 10.9 | 9.5 | 3.4 |
| 9B Q4 | 57.2 | 1,657 | 6.4 | 5.0 | 2.1 |
| 9B Q4 mixed_4_6 | 51.9 | 1,630 | 6.9 | 5.5 | 2.3 |
| Qwen3.8-27B Q4 | 17.6 | 385 | 17.4 | 15.1 | 7.3 |
| Qwen3.8-27B 6-bit | 12.4 | 373 | 24.0 | 21.9 | 9.9 |
| Qwen3.6-35B-A3B Q4 | 105.9 | 2,593 | 20.9 | 19.5 | 1.2 |

For documents, then:

- **Within the 9B,** Q4 costs nothing measurable here. It is 1.8x faster than Q8 and needs 4.5 GB less.
- **The 27B Q4** fits the RAM of the 9B's BF16 and decodes as fast as it does, but prefills 4x slower: about 2x the time per document of the 9B Q8, for +4.5 points.
- **The 35B-A3B MoE** is the standout at 21 GB. It has the 27B's accuracy and is the fastest of all (106 tok/s decode, 1.2 s/doc).

## Where the differences come from

- **27B Q4 against 9B Q8,** fields right only in one of them:
  - ticked options in the right form section: 13–0;
  - largest debit in misaligned statement tables: 10–0;
  - notice period among distractor clauses: 6–1;
  - invoice and due dates: 6–0.
- **9B Q8 against Q4:** the discordant fields are small and spread out, in both directions.
- **All models** undercount bank-statement transactions (non-thinking greedy decoding): the 9B and the 27B Q4 got 20–23 % right.

## Setup and caveats

- **Documents:** 150 synthetic OCR-style text documents with exact ground truth.
  - 30 each: invoices, receipts, insurance forms, bank statements, contracts / letters.
  - 24 English and 6 German / French / Bulgarian per type.
  - Distractors, mixed date formats, column layouts, OCR noise, absent fields.
  - The input is text, so this tests reading OCR output, not vision.
- **Decoding:** a fixed JSON schema per type. Qwen chat template with thinking off, greedy decoding, identical token ids for every config. mlx_lm 0.31.3, batched (completion batch 8, or 4 for the 27Bs because of memory).
- **Scoring:** per-field exact match after normalization.
- **Statistics:** paired McNemar test, and a document-clustered bootstrap CI.
- **Smoke test:** Qwen3.5-4B 4-bit, 90.5 %; used only to fix two harness issues before the freeze.
- **Known gold ambiguity:** in receipt-25, the gold time is null, but a fiscal TSE timestamp is printed. That is 1 field (0.1 point), the same for every config.
- **Memory of the quality runs** (batched, so higher than the batch-1 peaks above):
  - process peaks were 32 GB for the 9B BF16 and the 27B 6-bit, 27 GB for the 35B-A3B, and 26 GB for the 27B Q4;
  - swap stayed at 7.07–7.1 GB throughout.
- **Files:** code and PREREG in `docbench/` (EvalSuite 9cb980a, fdc91e6); raw outputs in `results/docbench/<config>/`; verdict in `results/docbench/verdict.txt`.
