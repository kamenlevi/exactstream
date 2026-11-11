# RESULT: the card gate on the 24 GB config (w24 = w13b) vs NAX6-N (Tue 2026-11-10 08:41) — PASS; its lifetime peak reaches 25.11 GB on longer prompts

Pre-registered in docs/PREREG_card_gates_2026-11-08.md: amendments 12 and 12a-12d, all before any w24 row. Amendment 13
covers the 12 GB neighbour arm. The sets are the model card's IFBench, GPQA Diamond and LiveCodeBench v6: non-thinking
greedy, caps 1,536 / 2,048 / 2,048, 381 gate keys, both orders.

**The gate passes on all six lines [K].** The 24 GB config keeps the card's scores within decision 4. As run, it holds
the 24.0 GB line only for short prompts: 22 of 40 processes went over, peaking at 25.112 GB (the Memory section below).

**The config:** BigMoE's data/stack/card_w24d_env.txt (87dd1a3, git blob e06f495e), 42 keys: w13b.
- It runs L + COH on the 24 GB fast path, with 141 base + 26 sidecar slots per layer (14.87 GB of pools) and
  R57_ROWCAP=13.
- R57_D7B=all (the 4-bit DFlash2 drafter only), R57_QOM_SETS=humaneval,mbpp,lcb6, R57_HARD=1200, C 100, MEM 24.
- Snapshot data/r57/snap/w24/r57_screen_w24.py, screen 6ea91764f855, ext build_qom .so ff69fa24a745, code pin
  27ac500bd0e0; BigMoE 9dc9cd1 at the start.
- The runner read a frozen copy, results/card_w24_env_run.txt (the same blob).

**Why this config:** BigMoE's W24d is a same-run ABBA on HumanEval (docs/W24_DECISION.md, section W24d;
data/w24/w24d_verdict.txt).
- w13b made 130.33 tok/s pooled against the 14-row config's 125.91 (+3.51 %), with peaks of 23.819 and 23.781 GB.
- EvalSuite rechecked that from the raw rows.
- The final word came from the orchestrator at 01:33, in NewIdeas' place. Amendment 12d records it (0f47fb7), including
  its one deviation: it arrived after W24d's 01:30 deadline for the number check.

**Checks [K]:**
- In every process, the 38 outside modules equal 9dc9cd1's. The 40 manifest modules equal process 1's
  (results/card/es_cdw24_100136_fwd1_modules.json).
- QOM was engaged on every kept LCB run: 131 of 131 per order (forward processes 13-20, reverse 1-7). None of the 250
  IFBench and GPQA rows per order used it, as configured.
- Every kept key has exactly one uncut run: 381 per order, none duplicated.

**Process notes:**
- **Start:** 01:36:43, by amendment 12c's early path (Leap's done files 00:18:12; the GO file 01:36:43).
- **Forward:** 01:36:44-04:58:15, 20 processes, no yields. One 81 s wait came at a process boundary (04:31:46, logged
  as "soft_end.sh, PAUSE or a closed lid").
- **Reverse:** 04:58:16-08:40:31, 20 processes, no yields.
- **Cut runs (amendment 9):** two, both GPQA in the reverse order: gpqad/recb4cGsC6BJUCU3V (process 16) and
  gpqad/recWxGU8Q4YReJ1tb (process 17).
  - Both fell between 07:04 and 07:47, while fanwatch's SSD heat PAUSE fired every few minutes and Fit12's first
    IFBench chunk ran beside the gate.
  - Each re-ran uncut in the next process.
  - PAUSE waits count toward a process's 1,200 s budget: the screen's running-time ticker leaves out only SIGSTOP
    freezes (r57_screen_w24.py, line 335).
- **Heat pauses:** 11 forward and 26 reverse rows carry the harness's PAUSED flag. The harness waits at pass boundaries
  while ~/.local/share/quiet/PAUSE exists (BigMoE bigmoe/gated.py, new_pass). Only timing is affected: the tokens are
  complete.
  ~/.local/share/quiet/HEALTH.md lists no protective action.
- **The neighbour (amendment 13):**
  - Fit12's 12 GB IFBench arm ran its first chunk beside the gate from 07:14:32.
  - From 07:23 it started no new chunk, at EvalSuite's request after the heat pauses.
  - It resumed alone at 08:41:43.
- **Cost:** 3 h 21 min forward and 3 h 42 min reverse, 7 h 04 min of wall time in all, with process starts and pauses.
  Kept tokens: 868,962 (forward 434,168, reverse 434,794).
- **Scoring:** LCB verdicts come from the grader cache results/lcb6_grades.jsonl. A row with no fenced code is a fail.

## The lines (vs NAX6-N)
| Line | Result | Verdict |
|---|---|---|
| IFBench forward (n 150) | 40.7 % vs 44.0 %, delta -3.33 pt, McNemar p 0.3018 (arm only 5 / reference only 10) | PASS |
| IFBench reverse | 40.7 % vs 44.0 %, delta -3.33 pt, McNemar p 0.3018 (5 / 10) | PASS |
| GPQA Diamond forward (n 100) | 47.0 % vs 44.0 %, delta +3.00 pt, McNemar p 0.5078 (6 / 3) | PASS |
| GPQA Diamond reverse | 47.0 % vs 44.0 %, delta +3.00 pt, McNemar p 0.5078 (6 / 3) | PASS |
| LCB v6 forward (n 131) | 29.8 % vs 32.1 %, delta -2.29 pt, McNemar p 0.4531 (2 / 5) | PASS |
| LCB v6 reverse | 32.8 % vs 32.1 %, delta +0.76 pt, McNemar p 1.0000 (4 / 3) | PASS |
| **Card gate** | | **PASS** |

The rule: a paired delta above -5 points and an exact McNemar p of at least 0.05, per benchmark and order. A benchmark
passes only in both orders, and the card gate passes when all three do. Every p above was recomputed by hand from its
discordant counts (two-sided exact binomial).

## Information (no verdict; tools/card_info.py w24)
- **IFBench (n 150):** loose 40.7 % in both orders vs 44.0 %; strict 36.7 % vs 40.0 %. 22 rows are token-identical to
  the reference; 10 hit the cap (reference 10). Forward and reverse are token-identical on 150 / 150 rows.
- **GPQA Diamond (n 100):**

  | Domain | w24 (both orders) | Reference |
  |---|---|---|
  | Biology (n 7) | 57.1 % | 57.1 % |
  | Chemistry (n 47) | 23.4 % | 21.3 % |
  | Physics (n 46) | 69.6 % | 65.2 % |

  No row is token-identical to the reference; 51 hit the cap (reference 54). Forward and reverse are token-identical on
  100 / 100 rows.
- **LCB v6 (n 131):**

  | Subset | Forward | Reverse | Reference |
  |---|---|---|---|
  | easy (n 31) | 90.3 % | 87.1 % | 93.5 % |
  | medium (n 39) | 28.2 % | 35.9 % | 28.2 % |
  | hard (n 61) | 0.0 % | 3.3 % | 3.3 % |
  | AtCoder (n 82) | 29.3 % | 30.5 % | 30.5 % |
  | LeetCode (n 49) | 30.6 % | 36.7 % | 34.7 % |

  No row is token-identical to the reference in either order. At the cap: 89 forward and 92 reverse (reference 88).
  Forward vs reverse: token-identical 9 / 131, same verdict 125 / 131.
- **Against the 45 GB L + COH gate (RESULT_card_lcoh_2026-11-09.md):**
  - The IFBench and GPQA rows are token-identical to lcoh's in both orders (250 / 250). On those two benchmarks the 24 GB
    config gives exactly the 45 GB config's outputs, so their lines equal lcoh's.
  - On LCB, where QOM runs, 2 / 131 forward rows and 0 / 131 reverse rows equal lcoh's. lcoh's LCB lines were 30.5 % and
    29.0 %.

## Memory (amendment 12d: every lifetime peak against the 24.0 GB line)
Lifetime peak per process [K]: the maximum of max(mem[1], mem[-1]) over the process's runs, in GB of 10^9 bytes, from
results/card/es_cdw24_100136_<order><N>_runs.json. "P" is the longest prompt in the process, in tokens.

| Process | Sets | P | Peak | Process | Sets | P | Peak |
|---|---|---|---|---|---|---|---|
| fwd 1 | GPQA | 919 | 24.813 | rev 1 | LCB | 1,120 | 24.976 |
| fwd 2 | GPQA | 339 | 23.818 | rev 2 | LCB | 1,360 | 24.874 |
| fwd 3 | GPQA | 807 | 24.606 | rev 3 | LCB | 1,465 | 24.984 |
| fwd 4 | GPQA | 350 | 23.920 | rev 4 | LCB | 1,144 | 24.957 |
| fwd 5 | GPQA | 2,801 | 25.045 | rev 5 | LCB | 921 | 24.748 |
| fwd 6 | IFBench | 147 | 23.811 | rev 6 | LCB | 739 | 24.395 |
| fwd 7 | IFBench | 168 | 23.765 | rev 7 | IFBench + LCB | 748 | 24.345 |
| fwd 8 | IFBench | 143 | 23.746 | rev 8 | IFBench | 190 | 23.717 |
| fwd 9 | IFBench | 148 | 23.638 | rev 9 | IFBench | 206 | 23.728 |
| fwd 10 | IFBench | 190 | 23.813 | rev 10 | IFBench | 190 | 23.724 |
| fwd 11 | IFBench | 206 | 23.748 | rev 11 | IFBench | 138 | 23.705 |
| fwd 12 | IFBench | 190 | 23.723 | rev 12 | IFBench | 148 | 23.721 |
| fwd 13 | IFBench + LCB | 748 | 24.423 | rev 13 | IFBench | 168 | 23.760 |
| fwd 14 | LCB | 739 | 24.444 | rev 14 | IFBench | 147 | 23.804 |
| fwd 15 | LCB | 921 | 24.720 | rev 15 | GPQA + IFBench | 2,801 | **25.112** |
| fwd 16 | LCB | 1,144 | 24.858 | rev 16 | GPQA | 350 | 23.735 |
| fwd 17 | LCB | 1,465 | 24.846 | rev 17 | GPQA | 807 | 24.610 |
| fwd 18 | LCB | 1,360 | 25.001 | rev 18 | GPQA | 807 | 24.411 |
| fwd 19 | LCB | 1,120 | 25.010 | rev 19 | GPQA | 384 | 23.858 |
| fwd 20 | LCB | 1,016 | 24.712 | rev 20 | GPQA | 919 | 24.747 |

- **Over the line:** 22 of 40 processes went over 24.0 GB. The maximum, 25.112 GB (reverse process 15), was set during
  the prefill of gpqad/recnTTKdBzfuoZ7w7, the gate's longest prompt at 2,801 tokens (MLX prefill peak 24.118 GB).
- **Where the line holds:** every process whose prompts stayed at or below 384 tokens stayed under 24.0 GB (at most
  23.920). That includes all 14 IFBench-only processes (prompts of at most 206 tokens: 23.638-23.813 GB). Every process
  with a prompt of 739 tokens or more went over (at least 24.345). Process 1's row with a 481-token prompt left the
  footprint at 23.910 GB, so the crossing lies between 481 and 739 tokens.
- **Per row [K]:** the lifetime peak is a running maximum, so a row can be read on its own only until its process first
  crosses 24.0 GB. Row order is checked: the running maximum never falls within a file. This per-row recount is the
  Editor's, rechecked here from the same files.
  - The 22 rows that first took a process over had prompts of 561 tokens or more (15 LCB, 7 GPQA, no IFBench). The
    shortest is lcb6/abc397_e: 561 tokens, 24.075 GB (reverse process 3).
  - The longest prompt in a row that left its process under 24.0 GB is lcb6/abc397_f: 592 tokens, 23.954 GB (reverse
    process 3, just before abc397_e). The same prompt went over in forward process 18 (24.112 GB).
  - **The limit is not exact:** that 592-token prompt was the first row of a fresh process both times. It peaked at
    23.954 GB once (MLX prefill peak 23.116) and at 24.112 GB once (23.339). That is run-to-run variation, not the
    process's state. (Corrected 10:4x after NewIdeas' number check via the Editor; the earlier text said "process
    state".)
  - **The evidence for "about 560":** no process crossed 24.0 GB on a prompt of 560 tokens or fewer. The shortest
    prompt at a crossing was 561 tokens (lcb6/abc397_e, the second row of reverse process 3).
  - Rows before a process's first crossing are under by construction, because the lifetime peak only rises. Rows after
    it cannot be read on their own; 185 of them had 560 tokens or fewer. (Corrected 10:4x: the earlier text cited the
    424 rows that stayed under as evidence.)
- **MLX prefill peak (mem[0]) by prompt length, all 764 runs:** the lifetime footprint runs about 1.0 GB above it.

  | Prompt tokens | Runs | Median (GB) | Max (GB) |
  |---|---|---|---|
  | up to 481 | 565 | 22.759 | 23.088 |
  | 482-919 | 165 | 23.328 | 23.832 |
  | 920-1,024 | 18 | 23.700 | 23.873 |
  | over 1,024 | 16 | 23.949 | 24.119 |

- **What it means:** with this config, the model card can say "under 24 GB" for prompts of up to about 560 tokens. That
  is the per-row reading above: 561 tokens is the shortest prompt that took a process over. Longer prompts reach up to
  25.1 GB. BigMoE's W24 memory checks ran HumanEval, MBPP and IFEval (its
  commit 86f0d13). The cause and any fix belong to BigMoE, the config's owner.
- This supersedes the interim note docs/RESULT_card_w24_peak_2026-11-10.md (017c82b, after process 1).

## Speed (information; rough, not a speed claim)
Whole decode over the uncut, unpaused runs (sum of n over sum of n / tok/s), forward / reverse:

| Set | Forward | Reverse |
|---|---|---|
| IFBench | 38.0 tok/s | 37.0 tok/s |
| GPQA Diamond | 36.5 tok/s | 36.4 tok/s |
| LCB v6 | 68.8 tok/s | 69.7 tok/s |

These are rough. They come from the card sets' long outputs at the night fan cap, with heat pauses at pass boundaries,
and with one neighbour for about 20 minutes. They are not comparable with W24d's HumanEval whole decode, which used
another set.

## Known limits
- **Caps:** non-thinking greedy at the caps above. Arm and reference run under the same caps, so the gate asks whether
  the arm keeps the reference's results within that budget.
- **Power:** as in RESULT_card_lcoh_2026-11-09.md. A PASS bounds the loss at these N; it does not show there is none.
  IFBench's -3.33 pt (strict also -3.3 pt) and LCB forward's -2.29 pt are inside the rule, not zero.
- **Memory:** as above. The memory claim and the score claim are separate: the scores hold for this config at its
  measured peaks, up to 25.112 GB.

## Data
- Rows: results/card_w24_fwd and results/card_w24_rev.
- Per process: results/card/es_cdw24_100136_* (runs, tokens, manifests and logs; the checks file for forward process 1
  only).
- The lines: results/card_w24_<set>_<order>_d4.txt, and results/CARD_w24_DONE.
- Logs: results/day61_w24_runner.log and results/card_w24_start.log.
- Inputs: results/card_w24_env_run.txt (the frozen env) and results/w24c_verdict.txt (the GO file).
- results/w24_gate_eta.txt: the last ETA.
- Reference: results/ref2tnaxn.r6.
