# SUPERSEDED by RESULT_card_w24_2026-11-10.md (f6613ff). RESULT (interim): the w24 card gate's config goes over the 24.0 GB line on longer prompts (Tue 2026-11-10 02:0x)

The w24 card gate (PREREG_card_gates_2026-11-08.md, amendments 12 to 12d) runs w13b: BigMoE data/stack/card_w24d_env.txt,
blob e06f495e, frozen as results/card_w24_env_run.txt. It started at 01:36:43. Its first forward process (20 GPQA Diamond
rows, 01:36:44-01:54:36, rc 0) reached a lifetime footprint peak of **24.813 GB [K]**, over the 24.0 GB line. This note
covers memory only. The gate's decision (scores vs NAX6-N) is not computed here and does not depend on it.

## Measured [K]

Source: results/card/es_cdw24_100136_fwd1_runs.json. Prompt tokens are the field P. The MLX prefill peak is mem[0]. The
lifetime peak is max(mem[1], mem[-1]), a running maximum over the process. GB means 10^9 bytes.

| Rows | Prompt tokens | MLX prefill peak (GB) | Lifetime peak after the row (GB) |
|---|---|---|---|
| 18 rows | 159-384 | 22.680-22.988 | 23.616-23.910 (the running max) |
| gpqad/rec2ZTsqazBZ3LTWZ | 481 | 23.031 | 23.910 |
| gpqad/rec6sE2CRtD4drtHg | 919 | 23.832 | **24.813** |

The MLX prefill peak rises with prompt length. The footprint runs about 1.0 GB above it. BigMoE's W24 memory checks used
short prompts (HumanEval, MBPP, IFEval). The cause and any fix are for BigMoE, the config's owner, to state in its docs.

## What the rest of the gate holds [K], and what that implies [P]

Prompt tokens come from the same prompts' lcoh rows (results/card/es_cdlcoh_*_runs.json, field P):

| Set | Prompts | Median | Max | Over 481 | Over 919 | Over 1,024 |
|---|---|---|---|---|---|---|
| GPQA Diamond | 100 | 232 | 2,801 | 5 | 1 | 1 |
| IFBench | 150 | 74 | 206 | 0 | 0 | 0 |
| LCB v6 | 131 | 575 | 1,465 | 94 | 16 | 7 |

So most LCB v6 rows are expected above 24.0 GB [P]. This note sets no upper bound.

## What it means

- As it stands, the 24 GB config holds the 24.0 GB line for short prompts only. The model card cannot state "under 24 GB"
  for long prompts with this config.
- The gate continues. No pre-registered rule stops it on a peak, and amendment 12d has the gate's RESULT report every
  lifetime peak against the 24.0 GB line.
- Neighbours: tools/w24_gate_eta.py (078bdc2) now prints the gate's lifetime peak so far in results/w24_gate_eta.txt. It
  budgets the gate at that peak + 2 GB, rounded up (27 GB at 02:0x). The runner still waits for 28 GB free before each
  process.

## Also in process 1

gpqad/rec7qmSnbud4FHSqL carries the harness's flag "PAUSED (40.5 s waited, 0 gaps)". That wait affects its timing only:
its tokens are complete (2,048, the cap). ~/.local/share/quiet/HEALTH.md lists no protective action in the last 24 h.
