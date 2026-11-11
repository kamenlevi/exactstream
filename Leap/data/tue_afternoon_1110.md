# Tue 10 Nov afternoon tests: 36 GB Mac simulation, PF-A at 24 GB, PF-B length bisect (13:46-14:02)

> (a quote from the project owner, kept private)
after BigMoE's W24g, one at a time, under one share lock, with a +2 GB swap guard. Evidence and scripts are in
data/tue_afternoon_1110/.

## 1. A 36 GB Mac simulated for the 24 GB package (`--mem 24`, w13b)

How it was set up:
- **Ballast:** anonymous memory locked with mlock() (ballast.py; wired, never paged out; no sudo, inside
  vm.user_wire_limit).
- **GPU working set:** capped at 27 GB inside the process by sitecustomize.py: `mx.device_info()` reports 27 GB, which the
  engine passes to `mx.set_wired_limit`, and the MLX memory limit is 1.5x that. No system setting was touched. All 6 runs
  logged "R57_WIRED: mx.set_wired_limit(27.00 GB)".
- **Ballast size:** this Mac's other processes (the sessions, the desktop app) hold about 17 GB, not a bare macOS's 6-8.
  So the plain 28 GB ballast would have tested a 36 GB Mac with a browser open. Two sizes were run:
  - Case A: a ballast leaving the engine about 25 GB (NO-SWAP 47 -> 25 GB), a 36 GB Mac with ~11 GB for macOS and apps,
    as INSTALL.md says.
  - Case B: the literal 28 GB. Other memory had been freed by then, so it left 27 GB (NO-SWAP 55 -> 27): in practice a
    second run of case A, not the browser-open stress case.

| run | case A (25 GB room) | case B (27 GB room) | free memory (this morning) |
|---|---|---|---|
| 12 HumanEval (BigMoE's keys, CAP 256) | **138.1 tok/s**, first token 3.61 s, peak 23.77 GB, tokens 12/12 identical | 137.3 tok/s, 3.66 s, 23.78 GB, 12/12 | 135.9 tok/s, 3.63 s, 23.78 GB |
| one 3,003-token prompt (paper text) | 26.9 tok/s decode, first token 17.98 s, peak 25.03 GB | 26.5 tok/s, 19.36 s, 25.04 GB | - |
| 3-turn chat.py --mem 24 | turns: 92 / 256 / 43 tokens at 29.7 / 43.0 / 27.5 tok/s; first token 2.28 / 3.28 / 4.68 s; peak 23.67 / 23.88 / 23.96 GB | ran (38 s) | - |
| swap | flat (6,659 MB) | +403 MB during the 3K prompt | - |

- With 25-27 GB of room and a 27 GB GPU working set, the 24 GB package ran at full speed. Its tokens were identical, its
  peaks unchanged and swap flat or nearly so.
- **What this does not prove:** a 36 GB M5 Max has a different GPU (more cores; MLX picks some kernel tiles by GPU type,
  so tokens may differ slightly) and a different SSD and thermals. None of that can be simulated here. The ballast also
  does not reproduce macOS's own memory policy on a 36 GB machine (compressor sizing, the real default working set).
- No browser-open (~17 GB of other use) case was run.

## 2. PF-A at 24 GB (chat.py --mem 24, R57_PFLM=1 vs 0; chat.py's 24 GB default unchanged)

| prompt | answers | first token: PF-A / default | lifetime peak: PF-A / default | decode |
|---|---|---|---|---|
| short (24 tokens, one chunk) | identical, 77 / 77 tokens | 1.98 / 2.03 s | 23.62 / 23.61 GB | 27.3 / 27.5 tok/s |
| long (6,980 tokens) | identical, 256 / 256 tokens | **17.70 / 42.06 s (2.38x)** | **25.98 / 25.67 GB** | 24.9 / 25.0 tok/s |

- PF-A works at 24 GB with identical answers and a 2.4x faster first token on a 7K prompt, for +0.3 GB.
- **The label is wrong past ~3K tokens:** at 6,980 tokens the 24 GB line peaks at 25.67 GB with the default prefill and
  25.98 GB with PF-A, both above "up to 25.1 GB on longer ones". The gate's longest prompt was 2,801 tokens, and the 3K
  prompt above peaked at 25.03 GB. The peak grows with prompt length; 10K and 30K at 24 GB were not measured.

## 3. PF-B length bisect (exact-32, R57_LOGHASH=1, 64 tokens, PF-B at TOK 8192 vs PF-A)

| prompt | PF-B MoE calls | tokens | verify-logits hashes | first token PF-A / PF-B | peaks PF-A / PF-B |
|---|---|---|---|---|---|
| 9,988 | 96 (2 groups per layer) | equal | **25 / 25** | 21.4 / 18.1 s | 34.26 / 35.50 GB |
| 15,002 | 96 (2 groups) | equal | **28 / 28** | 30.2 / 25.8 s | 35.43 / 36.77 GB |
| 20,003 | 144 (3 groups) | equal | **25 / 25** | 40.1 / 33.7 s | 36.57 / 38.06 GB |
| 25,003 | 192 (4 groups) | equal | **24 / 24** | 50.1 / 41.2 s | 37.31 / 39.01 GB |
| 29,988 (last night) | 192 (4 groups) | differ from the 40th | **0 / 25** | 62.6 / 51.7 s | 38.16 / 40.42 GB |

- PF-B is bitwise exact up to 25K tokens and breaks between 25K and 30K.
- The group count is not the cause: 4 groups pass at 25K here, and at 7.9K in the published TOK 2048 check.
- Group shapes at TOK 8192: 25K = 8 + 8 + 8 chunks plus a single 427-token chunk; 30K = 8 + 8 + 8 + 6 chunks, so its last
  call groups 5,412 tokens. The passing runs had multi-chunk last groups of 1,796, 6,810 and 3,619 tokens.
- Leading hypothesis, not tested: a matmul in the grouped MoE call (the router's N = 512, or the shared expert) whose
  bits depend on M for some shapes, as the shared expert's gate did before the per-chunk fix.
- The next step is a kernel-level check: the 30K run's group M (5,412; and 4,096 / 1,316 at TOK 4096) against the
  per-chunk calls, bit for bit. It needs a few minutes of GPU and no model run. No fix is built; the orchestrator is asked
  first.

## 4. The 24 GB line's peak by prompt length (Tue 14:05-14:13; orchestrator 14:0x: measure before the label wording)

Setup: run.py --config chat --mem 24 (w13b), 64 tokens, the same paper-text prompts as the 32 GB prefill test; share lock,
swap flat (7,062 MB). The 3K and 7K rows come from the runs above.

| prompt | default prefill: first token / lifetime peak | PF-A: first token / lifetime peak |
|---|---|---|
| benchmark prompts (gate; up to 2.8K) | - / up to 25.11 GB | - |
| 3,003 tokens | 17.98 s / 25.03 GB | - |
| 6,980 tokens (chat.py) | 42.06 s / 25.67 GB | 17.70 s / 25.98 GB |
| 9,988 tokens | 60.44 s / **25.82 GB** | 23.74 s (x2.55) / 26.54 GB |
| 29,988 tokens | 194.95 s / **28.17 GB** | 71.73 s (x2.72) / **30.60 GB** |

- The 24 GB line stays under 24 GB up to about 500 tokens (BigMoE W24g: a 520-token prompt at 24.014 GB). It then grows
  with the prompt: about 25.0 GB at 3K, 25.7 at 7K, 25.8 at 10K and 28.2 at 30K (default prefill, as run.py runs it).
- PF-A adds 0.3 / 0.7 / 2.4 GB at 7K / 10K / 30K. chat.py --mem 24 now uses PF-A (the orchestrator's decision at 14:0x);
  on a 36 GB Mac a 30K-token conversation would reach 30.6 GB with it.

## 5. PF-B kernel-level bit check (kcheck.py; orchestrator 14:0x; no fix built)

Method: layer 1's and layer 30's real router (8-bit, group 64), shared expert (4-bit) and shared expert gate (4-bit) from
the model's safetensors, MLX 0.32.2 on applegpu_g17s. Each op is computed once on M rows (PF-B's grouped call) and once on
the same rows in 1,024-row chunks (PF-A's and the default prefill's calls), then compared bit for bit, on random bf16
inputs. Layers 1 and 30 gave the same pattern:

| M (chunks) | router (N=512) | shared expert MLP | shared expert gate (N=1; PF-B already per chunk) |
|---|---|---|---|
| 5,412 = 5 x 1,024 + 292 (30K, TOK 8192: FAIL) | **292 rows differ** | **292 rows differ** | 3,302 differ |
| 1,316 = 1,024 + 292 (30K, TOK 4096: FAIL) | **292 rows differ** | **292 rows differ** | 190 differ |
| 4,096 = 4 x 1,024 (30K, TOK 4096) | equal | equal | 2,613 differ |
| 8,192 / 1,796 / 6,810 / 3,619 (passing runs) | equal | equal | differ (as known) |
| 427 (25K's tail, alone in its group) | equal | equal | equal |

- **Cause found:** in PF-B's grouped call, the router and the shared expert compute the rows of a short final chunk (292
  tokens here) with different bits than the per-chunk call does. Longer tails (547, 666, 772 tokens) are unaffected, as are
  full 1,024-row chunks.
- A short tail on its own in a group is also unaffected: 25K's 427-token tail is computed alone either way, which is why
  25K passed.
- So PF-B's exactness depends on the size of the prompt's last chunk, not on its total length. A prompt of any length whose
  last chunk is short and shares a group with full chunks can differ. The published 7.9K check had tails of 541, 634 and
  764 tokens.
- The exact threshold between 292 and 427 rows is not measured; MLX's quantized matmul probably takes another kernel below
  some row count.
- **Fixable**, in the same way as the shared expert gate's per-chunk fix (6553fcb): run the router and the shared expert on
  each chunk slice inside PF-B's group, and keep the grouped SwitchGLU call. Not built: the orchestrator decides first.
  After a fix, the PF-B check (10K / 15K / 20K / 25K / 30K, hashes) would show whether 30K becomes exact.

## 6. PF-B fix, research copy only (Tue 14:18-14:24; orchestrator 14:2x: build it, not in the package or 1.1.0)

The fix lives in release/cleanclone/pfbfix/clone (gitignored; = the 1.1.0 candidate pkg_110b + this change):
- In pflm.py, `R57_PFLM_SEGALL=1` makes PF-B's grouped MoE call run the router and the shared expert per chunk slice too,
  with the same `_SegPerChunk` wrapper the shared expert's gate already uses. The grouped SwitchGLU call is unchanged.
- The diff is data/tue_afternoon_1110/pflm_segall_research.diff: 14 changed lines, env-gated, off by default.

Test: exact-32, R57_LOGHASH=1, 64 tokens, TOK 8192, against PF-A on the same prompt; share lock, swap flat.

| prompt | arm | tokens | verify-logits hashes vs PF-A | first token | prefill | lifetime peak |
|---|---|---|---|---|---|---|
| short tail, P 5,412 = 5 x 1,024 + 292 (one group of 6 chunks) | PF-A | - | - | 13.70 s | 395 tok/s | 33.38 GB |
| | PF-B | equal | **0 / 20** | 12.20 s | 444 tok/s (x1.12) | 34.20 GB |
| | PF-B + fix | equal | **20 / 20** | 11.72 s | 462 tok/s (x1.17) | 34.19 GB |
| 29,988 tokens (4 groups) | PF-A | - | - | 62.25 s | 482 tok/s | 38.11 GB |
| | PF-B | differ from the 40th | **0 / 25** | 52.86 s | 567 tok/s (x1.18) | 40.44 GB |
| | PF-B + fix | equal | **26 / 26** | 54.01 s | 555 tok/s (x1.15) | 40.42 GB |

- **The fix makes PF-B bitwise-equal to PF-A in both failing cases,** including a short-tail prompt well below 8K tokens.
  There, the unfixed PF-B's logits differ at every check although its 64 tokens happen to match.
- **It keeps PF-B's speed:** x1.15-1.17 over PF-A, against x1.12-1.18 unfixed; at 30K, 54.0 s against 52.9 s (about 2 %).
  Its prefill at 30K is 555 tok/s at 32 GB, over Kamen's 500 tok/s line. It uses no extra memory: 40.42 against 40.44 GB.
  PF-A peaks 2.3 GB lower.
- 10K / 15K / 20K / 25K were exact without the fix, since their tails are not short. The fix only moves the grouped call's
  router and shared expert onto the per-chunk calls PF-A makes, so those lengths stay exact by construction (not re-run).
- **Next, not done:** the fix belongs in BigMoE's engine (its pflm.py; Leap does not edit other sessions' repos), then a
  new package snapshot. PF-B also needs memory beyond the line (40.4 GB at 30K on the 32 GB line).
