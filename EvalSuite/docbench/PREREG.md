# docbench pre-registration

Written 2026-10-25 20:45 EET, before any comparison run. Only the Qwen3.5-4B smoke test has been run so far; it is not one of the compared configs. Nothing in `gen.py`, `schema.py`, `score.py`, the set, or the rules below changes after the first comparison run.

## The two claims (Kamen doubts both)

1. Hyperscience's team: Qwen3.5-9B at Q4 is much worse than at Q8 for document processing.
2. A user: the 9B at Q8 is much better than a bigger model quantized to fit the same RAM (about 18 GB, the 9B's BF16 size).

## Set (frozen)

docbench v1 is `set/docs.jsonl`, sha256 `c69f2aa1f641e353…`, made by `gen.py` (sha256 `a39c14635c11beef…`, seed 20261025). It holds 150 synthetic OCR-style text documents with exact ground truth:

- 30 each of invoices, till receipts, insurance forms, bank statements (tables) and contracts / engagement letters.
- In each type, 24 are English (US / UK / Ireland) and 2 each are German, French and Bulgarian (20 % non-English).
- 930 fields in all. Every gold value is printed in its document; the generator asserts this.
- An English numeric date that could be DD/MM or MM/DD always has a disambiguating date in the same format, or a format note in the document.

Fields per type, from `schema.py`:

| Type | Fields |
|---|---|
| invoice | vendor, invoice number, invoice date, due date (null if not printed), currency, subtotal, tax, total, PO number |
| receipt | merchant, date, time, total, payment method, card last 4 |
| form | full name, date of birth, policy number, street, postal code, city, country, ticked options |
| statement | opening balance, closing balance, transaction count, largest debit |
| contract | parties, effective date, notice period (value + unit), governing law |

The documents are hard in the IDP way:

- distractor dates (order, delivery, signature, incident) and amounts (subtotal against total against previous balance or deposit, cash tendered, change, VAT-included lines, pending items, available balance);
- loyalty-card numbers masked like payment cards;
- column layouts, labels printed apart from their values, and misaligned tables with dropped empty cells (the running balance tells debit from credit);
- light OCR noise in non-key text;
- absent fields (null).

The 4B smoke test led to two fixes before this freeze, both made blind to every compared config:

- A separate "other information" section now sits between the options checklist and a Yes/No question. Before, nothing separated the "No" box from the cover options.
- The notice-period instruction now says "in the unit the document uses".

## Configs

All run with mlx_lm 0.31.3 (`~/BigMoE/.venv324`) on this Mac. Every in-memory quantization follows `mlx_lm.convert`: lazy load, then `quantize_model` with the same predicate.

| # | config | what |
|---|---|---|
| 1 | `9b-bf16` | mlx-community/Qwen3.5-9B-MLX-bf16, the reference |
| 2 | `9b-q8` | config 1 quantized in memory, affine 8-bit, group 64 |
| 3 | `9b-q4` | config 1 quantized in memory, affine 4-bit, group 64 (= the common MLX / LM Studio 4-bit) |
| 4 | `9b-q4mixed` | config 1 quantized in memory with mlx_lm's `mixed_4_6` recipe (closer to llama.cpp Q4_K_M) |
| 5 | `38-27b-q4` | Kamen's Qwen3.8-27B (LM Studio, 6-bit), dequantized and re-quantized in memory to affine 4-bit g64. This is **the headline "bigger model" for claim 2**. It has one extra rounding step compared with a 4-bit made from BF16, which can only hurt it. |
| 6 | `38-27b-q6` | Kamen's Qwen3.8-27B 6-bit as is (21 GB, a little over the 9B's BF16 size) |
| 7 | `36-35b-a3b-q4` | Kamen's Qwen3.6-35B-A3B 4-bit (MoE, 3B active) as is; bonus, reported only |

> (a quote from the project owner, kept private)

## Procedure (identical for every config)

- **Prompt:** `schema.prompt(type, text)`, one user message. The Qwen3.5 / 3.6 / 3.8 tokenizer and chat template are the same (checked: identical token ids). `enable_thinking=False`.
- **Decoding:** greedy (argmax), at most 640 new tokens.
- **Batching:** mlx_lm `BatchGenerator`, prompts sorted by length, prefill batch 2, prefill step 512. The completion batch is 8 for the 9B configs and the 35B-A3B, and 4 for the two 27B configs (memory: no in-RAM run above about 25 GB). Batching changes only the numerics: on the 4B, batch 1 and batch 16 gave identical outputs on 14 of 15 documents.
- **Hashes:** each run's `meta.json` records the hashes of the set, the schema and the rendered prompts. Frozen files (sha256, first 16 hex): `gen.py` a39c14635c11beef, `schema.py` 4d4781968171075e, `score.py` f75e7e080625eccc, `run.py` a9a15a417e2427f0, `set/docs.jsonl` c69f2aa1f641e353.

## Scoring (`schema.py`, `score.py`)

- **Per field:** exact match after normalization.
  - Text: NFKC, casefold, punctuation to spaces, whitespace collapsed.
  - IDs and postal codes: alphanumerics only.
  - Dates: exactly `YYYY-MM-DD`.
  - Amounts: a number, or a plain numeric string, within 0.005. The largest debit is compared by absolute value.
  - Time: HH:MM in 24-hour format (AM/PM and seconds tolerated).
  - Currency: ISO code.
  - Card: 4 digits.
  - Country: English name, with listed aliases.
  - Ticked options and parties: set equality.
  - Notice: value and unit both equal.
  - A missing key counts as null.
- **JSON:** strict validity (the output, with any markdown fence removed, is one JSON object) is reported. Fields are scored from the first JSON object found. With no JSON at all, every field counts as wrong.
- **Also reported:** documents with all fields correct, per type, and per language.

## Primary metric, tests, decision rules

- **Field accuracy** = correct fields / 930.
- **Paired comparison of A and B** on the same 930 fields:
  - Δ = acc(A) − acc(B), in points;
  - exact two-sided McNemar test on the discordant fields;
  - 95 % CI of Δ from a document-clustered bootstrap (10,000 resamples of the 150 documents, seed 1).
- **Claim 1** (Δ1 = acc(9B Q8) − acc(9B Q4 plain)):
  - TRUE if Δ1 ≥ 3.0 points and p < 0.05;
  - FALSE if Δ1 < 1.5 points or p ≥ 0.05;
  - otherwise "small difference", with the number.
  - The same rule is reported for 9B Q8 against 9B Q4 mixed, to show whether the 4-bit method matters.
- **Claim 2, headline** (Δ2 = acc(9B Q8) − acc(Qwen3.8-27B Q4)):
  - TRUE if Δ2 ≥ 3.0 points and p < 0.05;
  - FALSE if the 27B ≥ 9B Q8, or Δ2 < 1.5 points, or p ≥ 0.05;
  - otherwise "small difference", with the number.
- **Claim 2, also judged with the same rule:**
  - against Qwen3.8-27B 6-bit;
  - against Qwen3.5-27B Q4 (same-family control) only if it is ever run.
- **Expected check:** 9B Q8 = 9B BF16 within noise (|Δ| < 1.5 points or p ≥ 0.05). If not, it is flagged: either a harness problem or a real 8-bit loss.
- **Robustness flag:** if McNemar gives p < 0.05 but the clustered CI includes 0, the verdict says "field-level only".

## Speed and memory

- **When:** under a short quiet lock per model.
- **Batch 1:** prefill tok/s on the longest document's prompt, and decode tok/s over 256 greedy tokens. Median of 3, after a warm-up.
- **Memory:** MLX peak memory (weights + cache + activations) during that run, and weight size.
- **Throughput:** documents/min from the quality run, taken under a share lock, so rough.
- **Caveat:** in-memory configs report memory after conversion. A pre-quantized file of the same format would use the same.

## Report

Verdict first, then the table, to the orchestrator. Results go in `results/docbench/` (not in git). The 9B BF16 download is moved to the Trash after the test (disk floor). Kamen's LM Studio models are never touched.

## Changes after the freeze

- 20:48: `run.py` now drops its reference to the last dequantized matrix after the in-memory re-quantization of config 5. That matrix was the 2.5 GB lm_head, which stayed allocated. This affects memory only, not outputs. The quality run of config 5 had already started on the earlier file; its speed run uses the fixed one. New `run.py` hash 8b06306e7be9dcf4.
