# RESULT: the card gate on L + COH (lcoh) vs NAX6-N (Mon 2026-11-09 10:07) — PASS

Pre-registered in docs/PREREG_card_gates_2026-11-08.md (Sun 12:0x, before any row; amendments 1-9). The sets are the
model card's IFBench, GPQA Diamond and LiveCodeBench v6, non-thinking greedy, caps 1,536 / 2,048 / 2,048, 381 gate keys.

**The config:** BigMoE's data/stack/card_lcoh_env.txt, 34 keys.
- L + COH = S5' + REF2_NAX_COH=1. QOM runs on the code sets (humaneval, mbpp) plus lcb6. IFBench and GPQA read exact
  experts (L's QOM sets hold no chat set).
- Sat 256, MEM 47 GB. Card snapshot data/r57/snap/card (BigMoE e057abf; frozen code == the FU snapshot, ref2 d05063fa),
  screen daabc51562c5. BigMoE b37f36b, ext build_qom .so ff69fa24a745.
- Code pin 5680b8a9731b from 03:42; 8ecc1604bd56 after amendment 9 (of the pinned files only tools/r57rows.py changed;
  the runner's engagement check now skips cut runs).
- In every process, the 38 outside modules equal b37f36b's, and the manifests equal the 03:42 one. QOM was engaged on
  every kept LCB row (pf.qom_opt > 0 on 131 of 131 in each order); none of the 250 IFBench and GPQA rows per order used
  it, as configured. Every kept key has exactly one uncut run.

**Process notes:**
- **Forward:** 03:42-06:26, 20 processes, no yields.
- **The cut rows (amendment 9):** the screen stops a row mid-decode when its process's 600 s budget runs out, and flags
  it. r57rows kept 10 such rows in the forward order (5 GPQA, 5 LCB), ending in "length" below the cap. At 06:37 the
  reverse order's first process cut a 1-token LCB row with no QOM use, so the engagement check voided the arm and the
  chain stopped. Nothing from that process was kept.
  - The fix: cut rows are left out and their keys re-run. The 10 forward keys were re-decoded 06:43-06:49 (none cut).
  - At 06:3x I had scored the forward order for information with the cut rows in it. That look is void.
- **Reverse:** 06:49-10:07 (21 processes). The fix left out 10 cut runs (5 LCB, 5 GPQA), each re-run in the next
  process. The arm yielded at four process boundaries to other sessions' quiet runs, 22 min in all: 07:00-07:02 and
  07:12-07:16 (a package smoke test and its re-run), 07:28-07:33 (a third smoke run), 09:46-09:57 (the release rows
  run). Decode time 176 min.
- **The cost of one arm:** forward 164 min, the 10 re-decoded keys 6 min, reverse 176 min: 5.8 h of decoding for 381
  keys in two orders (process starts included); 6 h 25 min of wall time with the yields and the void process.
  - Kept tokens: 867,659 (forward 433,011, reverse 434,648), 1,139 per row, about 42 tok/s over the 5.8 h.
  - The PREREG priced ~450K tokens (380-600K) and ~3.6 h (3-4.6 h) for both orders, about 590 tokens per row, at 38-45
    tok/s plus the process starts. The rate held; the rows were about twice as long as priced.
- **Scoring:** LCB verdicts come from the grader cache results/lcb6_grades.jsonl, keyed by (question, code sha): 226
  gradings for the reference and both orders, 52 s in all, no timeout codes. A row with no fenced code is a fail.

## The lines (vs NAX6-N)
| Line | Result | Verdict |
|---|---|---|
| IFBench forward (n 150) | 40.7 % vs 44.0 %, delta -3.33 pt, McNemar p 0.3018 (arm only 5 / reference only 10) | PASS |
| IFBench reverse | 40.7 % vs 44.0 %, delta -3.33 pt, McNemar p 0.3018 (5 / 10) | PASS |
| GPQA Diamond forward (n 100) | 47.0 % vs 44.0 %, delta +3.00 pt, McNemar p 0.5078 (6 / 3) | PASS |
| GPQA Diamond reverse | 47.0 % vs 44.0 %, delta +3.00 pt, McNemar p 0.5078 (6 / 3) | PASS |
| LCB v6 forward (n 131) | 30.5 % vs 32.1 %, delta -1.53 pt, McNemar p 0.6875 (2 / 4) | PASS |
| LCB v6 reverse | 29.0 % vs 32.1 %, delta -3.05 pt, McNemar p 0.2891 (2 / 6) | PASS |
| **Card gate** | | **PASS** |

The rule: a paired delta above -5 points and an exact McNemar p >= 0.05, per benchmark and order; a benchmark passes
only in both orders; the card gate passes when all three do. Every p above was recomputed by hand from its discordant
counts (two-sided exact binomial).

## Information (no verdict; tools/card_info.py lcoh)
- **IFBench (n 150):** loose 40.7 % in both orders vs 44.0 %; strict 36.7 % vs 40.0 %. 22 rows token-identical to the
  reference; 10 at the cap (reference 10). Forward and reverse are token-identical on 150 / 150 rows.
- **GPQA Diamond (n 100):** Biology (n 7) 57.1 % vs 57.1 %; Chemistry (n 47) 23.4 % vs 21.3 %; Physics (n 46) 69.6 % vs
  65.2 %. No row token-identical to the reference; 51 at the cap (reference 54). Forward and reverse are token-identical
  on 100 / 100 rows.
- **LCB v6 (n 131):** forward / reverse vs reference: easy (n 31) 87.1 / 87.1 vs 93.5 %; medium (n 39) 25.6 / 23.1 vs
  28.2 %; hard (n 61) 4.9 / 3.3 vs 3.3 %; AtCoder (n 82) 30.5 / 28.0 vs 30.5 %; LeetCode (n 49) 30.6 / 30.6 vs 34.7 %.
  Token-identical to the reference: 0 (forward), 1 (reverse); at the cap 88 / 89 (reference 88). Forward vs reverse:
  token-identical 4 / 131, same verdict 127 / 131.
- So the order changed rows only where QOM runs (LCB): on IFBench and GPQA the two orders gave the same tokens.

## Known limits
- Non-thinking greedy at the caps above. The reference stops at its cap on 10 IFBench, 54 GPQA and 88 LCB rows; arm and
  reference run under the same caps, so the gate asks whether the arm keeps the reference's results within that budget.
- Power at these N: an arm with no real loss whose rows differ from NAX6-N on 10-15 % of keys fails somewhere with a real chance
  (~25-50 % over 3 benchmarks x 2 orders; PREREG "Known limits"). A PASS bounds the loss at these N; it does not show
  the arm has none: IFBench's -3.33 pt (strict also -3.3 pt) and LCB's -1.5 / -3.1 pt are inside the rule, not zero.

## Data
results/card_lcoh_fwd, results/card_lcoh_rev (rows), results/card/es_cdlcoh_* (per-process runs, tokens, manifests),
results/card_lcoh_<set>_<order>_d4.txt (the lines), results/CARD_lcoh_DONE, results/day61_lcoh_runner.log,
results/card_chain.log, results/lcb6_grades.jsonl. Reference: results/ref2tnaxn.r6.
