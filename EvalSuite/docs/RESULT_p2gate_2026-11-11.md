# RESULT: the P2 same-scores gate on the 24 GB card (Wed 2026-11-11 14:41-19:51) — PASS: IFBench and GPQA Diamond hold the card's scores, and the 5 controls are token-identical

Pre-registered in docs/PREREG_p2gate_2026-11-11.md (12007aa, 14:33) before any row, with amendment 1 (41935ca) and 1a
(6005b66). Runner 62 ran it (tools/day_runner62.sh, dbdc426d...). It was scored with tools/set_d4.py against the card as
gated, results/card_w24_fwd.

**Verdict [K]** (results/CARD_p2_DONE):

| Line | P2 | Card | Delta | McNemar (arm only / card only) | Verdict |
|---|---|---|---|---|---|
| IFBench (n 150, loose) | 39.3 % | 40.7 % | -1.33 pt | p 0.8036 (7 / 9) | PASS |
| GPQA Diamond (n 67, prompts <= 256) | 55.2 % | 58.2 % | -2.99 pt | p 0.6250 (1 / 3) | PASS |
| Controls (5 GPQA prompts of 258-269 tokens) | token-identical 5 / 5 | | | | holds |
| **P2 gate** | | | | | **PASS** |

The rule is the card gate's: per set, a paired delta above -5 points and an exact McNemar p of at least 0.05. Both sets
pass, so **P2 keeps the card's scores on every prompt it touches**. LCB v6 has no prompt of 256 tokens or fewer, so P2
never runs there.

## Validity [K]
- **The arm:**
  - env BigMoE data/stack/card_w24p2_env.txt, blob 6ff0296f: w13b plus R57_P2=256, R57_P2_ROWS=13 and R57_P2_VERIFY=0;
  - the w24ft screen 0e2cc385e83e, the .so ff69fa24a745 and code pin 2d66c4634f77;
  - bm_pin OK at every process.
- **The code stayed the same:** in every process, the 38 outside modules equal BigMoE HEAD's, and the 40 manifest
  modules equal the first process's.
  - BigMoE's HEAD moved from 4fa9149 to 4684cb8 between the hold and the resume, but no file the gate loads changed.
- **P2 engaged on every kept run:** p2 > 0 on all 217 short keys (median 6 chunks on IFBench, 16 on GPQA) and p2 = 0 on
  all 5 controls. QOM was not used, as configured.
- **Rows:** every key has exactly one uncut run: 217 arm rows and 5 controls, with no duplicates.

## How it ran [K]
- **Start:** 14:41:56, after amendment 1a. The first start at 14:35 waited without end (BSD date's seconds); it was
  stopped before any GPU step and fixed.
- **fwd processes 1-10:** 14:41-16:51, 195 rows.
- **The hold:** 16:51:05-19:39:12, on the orchestrator's word, while Kamen used Claude Code on the Mac.
  - The runner was stopped by pid right after fwd 10's rows were appended, between processes.
  - It resumed with CARD_RESUME=1, against the first process's manifest.
- **After the resume:** fwd processes 1-2 (22 rows), then the control process, ended 19:51:04.
- **Heat:** fanwatch froze the gate in 12 heat episodes of 121-263 s each. Each was a PAUSE, then a freeze after 40 s,
  so the processes were stopped for about 81-223 s.
  - 11 fell between 15:08 and 16:46 (the day's heat), and one, 19:48, in the control process.
  - Shorter PAUSE waits also came between them.
  - Tokens are unaffected; only times are.

## Information (no verdict)
- **Token identity with the card:** IFBench 15 / 150, GPQA 0 / 67. P2 is lossy, as BigMoE said, yet its scores hold.
- **First token, P2's purpose** (runs.json pfs, the prompt phase, against the card's own runs on the same keys):

  | Set | P2 median | Card median | P2 p90 | Card p90 |
  |---|---|---|---|---|
  | IFBench (prompts 28-206 tokens) | 1.60 s | 3.31 s | 2.60 s | 3.60 s |
  | GPQA Diamond (112-256 tokens) | 3.12 s | 3.75 s | 3.89 s | 4.06 s |

  - That is 2.1x faster on IFBench's short prompts and 1.2x faster on GPQA's longer ones.
  - Two outliers are heat, not P2. GPQA's 145.9 s (fwd 3) falls in the 15:33 freeze. IFBench's 42.2 s (fwd 7) falls in a
    40 s PAUSE at that process's start.
  - The card's times come from its night run, which had heat pauses of its own.
  - On the controls, which take the prefill, P2's 4.1-4.7 s matches the card's 3.9-5.3 s.
- **Decode:** the median whole decode is 27.9 tok/s on IFBench (card 29.3) and 34.9 on GPQA (card 36.2), about 4-5 %
  lower. These are day and night runs, so the comparison is information only.
- **Memory:** P2's lifetime peak (mem[6]) is at most 23.25 GB on these prompts, and no row goes over 24.0 GB. The card
  reached 25.05 GB on the same GPQA keys. The controls peak at 23.69 GB.

## What it means
- **The verdict:** on the card's own benchmarks, P2's lossy first-token path keeps the 24 GB card's scores. IFBench is
  -1.3 points and GPQA -3.0 points, both well inside the -5 point bar, with p far above 0.05.
- **The speed:** the first token on short prompts falls from about 3.3 s to about 1.6 s, and the peak stays under 24 GB
  on these prompts.
- **Limits:**
  - GPQA's 67 pairs give the test little power.
  - Only one order was run; this rests on the card's order identity on these sets.
  - IFEval and other chat use are not gated here.
  - The decode is about 4-5 % slower in this day run.

## Data
- results/CARD_p2_DONE;
- results/card_p2_fwd/decode.jsonl (217 rows) and results/card_p2_ctl/decode.jsonl (5);
- results/card_p2_ifbench_d4.txt and card_p2_gpqad_d4.txt;
- per process: results/card/es_cdp2_* (log, modules, runs, tokens);
- results/card/p2_keys.txt and p2_ctl_keys.txt;
- results/day62_p2_runner.log.
