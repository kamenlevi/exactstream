# Tonight's 32 GB prefill test: default vs PF-A vs PF-B on 10K and 30K tokens (Tue 10 Nov 00:07-00:18)

The chat-32 config through run.py. One long-document prompt per process asks for a five-bullet summary of the document:
9,988 tokens (prompt_10k.txt) or 29,988 tokens (prompt_30k.txt), cut from the MAIN, KERNELS and RELEASE papers. Each run
generates 64 tokens. The type rule called the prompt math, so it drafted with DFlash2.

The prefill paths compared:
- default: the loop's chunk-major prefill, in 1,024-token chunks.
- PF-A: `R57_PFLM=1`, layer-major over the same chunks.
- PF-B: `R57_PFLM=3`, each layer split at its MoE, with one MoE call per group of chunks up to 4,096 tokens.

The orchestrator asked for this at 21:1x in place of the impossible replay, and gave tonight's slot at 23:2x on Kamen's yes.

Conditions:
- Ran right after the drafter pair (data/drafter_1109.md), under the same share lock.
- Kamen idle and Chrome closed. Nothing else ran (Breakthrough's and Fit12's jobs had ended).
- NO-SWAP 51-52 GB before each run; swap flat at 5,618 MB throughout.

| prompt | path | first token (= prefill) | prefill rate | vs default | lifetime peak | whole decode (64 tokens) | process wall |
|---|---|---|---|---|---|---|---|
| 10K | default | 78.41 s | 127 tok/s | 1 | 34.01 GB | 32.8 tok/s | 97 s |
| 10K | PF-A | 22.71 s | 440 tok/s | x3.45 | 34.83 GB | 32.0 tok/s | 42 s |
| 10K | PF-B | 20.04 s | 498 tok/s | x3.91 | 35.77 GB | 33.0 tok/s | 39 s |
| 30K | default | 252.24 s | 119 tok/s | 1 | 35.48 GB | 22.4 tok/s | 272 s |
| 30K | PF-A | 68.55 s | 437 tok/s | x3.68 | 38.49 GB | 21.2 tok/s | 89 s |
| 30K | PF-B | 58.02 s | 517 tok/s | x4.35 | 40.83 GB | 21.4 tok/s | 78 s |

Prefill rate = prompt tokens / first token. Lifetime peak = the engine process's lifetime physical footprint (run.py's own
line). The runs went in this order: 10K default, 10K PF-A, 30K default, 30K PF-A, 10K PF-B, 30K PF-B.

## Findings

1. **PF-A cuts the wait before the first token 3.45x (10K) and 3.68x (30K) at 32 GB, and it is exact here.** Its 64 tokens
   were identical to the default's at both sizes. So were its decode statistics: the same 107 checked rows and the same
   34,963 / 35,515 optional expert uses, of which 10,452 / 11,248 were dropped.
2. **PF-B is faster still: 3.91x and 4.35x (498 and 517 tok/s).** That is Kamen's 500 tok/s prefill line, reached at 32 GB.
3. **PF-B at 30K is not bitwise-exact, confirmed with exact decoding and logits hashes in the section below.** At 10K its
   tokens and decode statistics were identical to the default's.
   At 30K the first 42 tokens match (the first token too), and it differs from token 43 on: 100 checked rows instead of
   107, 33,158 optional uses and 10,753 dropped.
   - The decode is deterministic for a given prefill. Default and PF-A at both sizes, and PF-B at 10K, had identical decode
     statistics, although PF-B at 10K left different expert data in the tier (150 hits against 78, 43 dropped against 0).
   - So the difference most likely comes from PF-B's own prefill at 30K. Treat it as a bug until shown otherwise.
   - The engine's earlier PF-B exactness check ran at up to 8K tokens: TOK 8192 with the shared gate per chunk, tokens equal
     to PF-A and logits hashes 8/8, 28/28, 22/22 (TABLE_350, BigMoE pfd part B).
   - Candidates, not yet tested: a kernel in the grouped MoE call whose bits depend on M (the group of up to 4,096 tokens
     against PF-A's 1,024), as the shared expert's gate did before the per-chunk fix (6553fcb), or something that appears
     only past some length.
4. **Kamen's chat at 32 GB:** the default prefill runs at 119-127 tok/s, so his ~14K-token turns waited about 2 minutes
   before the first token (data/chat_session_1109.md had it at "a minute or more", unmeasured). With PF-A that becomes
   about 32 s, with the same tokens.
5. **Memory: every run at 10K tokens or more went over the 32 GB line, the default included** (34.01 and 35.48 GB). PF-A
   adds 0.82 GB at 10K and 3.01 GB at 30K. PF-B adds 1.76 and 5.35 GB.
6. **Decode after a long prompt is slower:** 32-33 tok/s at 10K and 21-22 tok/s at 30K, for these 64-token answers.

## PF-B exactness at 30K with exact decoding (Tue 00:57-01:01, the orchestrator's go)

Setup:
- exact-32 (exact decoding, no lossy drops) through run.py on the same 29,988-token prompt, 64 tokens.
- R57_LOGHASH=1: each decode check's verify logits hashed.
- Compared with PF-A, after the Editor asked for the published setting (R57_PFLM_TOK=8192) to be checked as well.
- Share lock, nothing else running, done-files untouched.

| arm | MoE calls (48 layers x groups) | first token | lifetime peak | tokens vs PF-A | verify-logits hashes vs PF-A |
|---|---|---|---|---|---|
| default prefill (01:03-01:07) | per 1,024-token chunk (stock) | 238.60 s | 34.93 GB | **equal, 64 / 64** | **equal on 26 of 26** |
| PF-A (reference) | 1,440 (per 1,024-token chunk) | 62.64 s | 38.16 GB | - | - (26 checks) |
| PF-B, TOK 8192 (published) | 192 = 48 x 4 (3 x 8,192 + 5,412 tokens) | 51.71 s | 40.42 GB | equal through the 39th, differ from the 40th | equal on **0 of 25**, first difference at check 1 |
| PF-B, TOK 4096 (default) | 384 = 48 x 8 (7 x 4,096 + 1,316 tokens) | 52.74 s | 40.42 GB | equal through the 39th, differ from the 40th | equal on **0 of 25**, first difference at check 1 |

- **PF-A is bitwise-equal to the default prefill at 30K tokens:** the same 64 tokens and verify-logits hashes equal on 26
  of 26 checks, with the first token 3.81x sooner (62.64 against 238.60 s). This extends the published PF-A evidence
  (hashes up to 7.9K tokens) to 30K, and backs PF-A as chat.py's prefill (Kamen's decision on memory: +3.2 GB lifetime
  peak here).
- **PF-B is not bitwise-exact at 30K tokens, at either call size.** Its very first verify logits already differ, so its
  prefill KV differs from the layer-major prefill's.
- **Limit: shown exact up to 7.9K tokens; the cause at 30K is not yet found.**
  - The published checks reached 7,932 tokens with hash-equal results at two call sizes: TOK 8192 (one call per layer)
    and TOK 2048 (3 calls per layer at 4,730 tokens, 4 at 7,932; hashes 8/8, 28/28, 22/22). Source:
    BM:data/pfd/pfd_report.txt at a6d5347, part B, arms pdb38s and pdb32s.
  - So the number of calls per layer alone does not explain the failure here (4 calls at TOK 8192); length or position
    is still in play.
  - Tonight's 10K run (chat-32, 3 calls per layer) matched in tokens and decode statistics but was not hash-checked.
- Files: prefill_1109/exact30k/
  - summary.txt;
  - run_<arm>.out for def, pfa, pfb8k and pfb4k;
  - per run, the engine log and runs.json (with the per-check hashes "lh"), tokens.json and modules.json. The runs are
    cli_1110_005716 (PF-A), 005838 (PF-B 8192), 005948 (PF-B 4096) and 010323 (default);
  - runner_exact.sh and runner_exact_def.sh: the runners.

## PF-D on these prompts (BigMoE, Tue 00:28-00:54)

BigMoE ran the published chat-45 config on these two prompts, PF-D off against PF-D as published, through the release
run.py with R57_LOGHASH=1.
- PF-D is exact on long prompts: 64/64 tokens, logits hashes 25/25 (10K) and 27/27 (30K).
- Its check arms compared 705,420 rows with 0 differing, over 480 / 1,440 PF-D calls with 0 fallbacks.
- At 45 GB there is no grouped prefill, so PF-D runs on every 1,024-token chunk. It shares nothing with PF-B's grouped
  MoE call.
- Cost on long prompts: first token only 3-4 % shorter (78.9 -> 76.6 s, 249.4 -> 238.4 s), lifetime peak +2 GB.

Data: BM:data/pfd_long (926b727).

## Next

- Done, Tue 01:01: PF-B at 30K is not bitwise (below). Still open: where between 7.9K and 30K tokens PF-B stops being
  bitwise, and why. The call count alone is ruled out: 3-4 calls per layer were exact up to 7.9K.
- Next: exact-32 hash checks by length (10K, 15K, 20K) at TOK 2048 and 4096, then a kernel-level bisection of the grouped
  MoE call against PF-A's per-chunk calls at the first failing length (position-dependent inputs, row counts, padding).
- If Kamen wants it: PF-A as chat.py's prefill at 32 GB. It is exact and the tokens matched at both sizes; the extra
  memory (+0.8 GB at 10K, +3.0 GB at 30K) is his call. chat.py stays local.

## Limits

- One run per cell, on one document.
- 64 decoded tokens are a weak exactness test: matching tokens do not prove bitwise equality.
- The default runs went first at each size, so a later run could have gained from a warmer file cache. PF-A's time grew
  with tokens (22.7 s to 68.6 s for 3x the tokens), so it is compute-bound and a warm cache matters little for it.

## Files (data/prefill_1109/)

- summary.txt: the bench's own lines, with the token comparison.
- run_<size>_<path>.out: run.py's output for each run.
- cli_1110_<time>.log and cli_1110_<time>_tokens.json: the engine log and tokens for each run. The times are 000737
  (10K default), 000918 (10K PF-A), 001004 (30K default), 001438 (30K PF-A), 001610 (10K PF-B) and 001653 (30K PF-B).
- bench_prefill.sh: the bench.
- prompt_10k.txt and prompt_30k.txt: the two documents.
