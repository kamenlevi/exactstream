# RESULT: the card gate on M1-B + COH (m1b) vs NAX6-N (Mon 2026-11-09 16:06) — PASS

Pre-registered in docs/PREREG_card_gates_2026-11-08.md (Sun 12:0x, before any row; amendments 1-11). The sets are the
model card's IFBench, GPQA Diamond and LiveCodeBench v6, non-thinking greedy, caps 1,536 / 2,048 / 2,048, 381 gate keys.

**The config:** BigMoE's data/stack/card_m1b_env.txt, 35 keys.
- M1-B + COH = card_lcoh_env.txt + QOM on every gate set + R57_QOM_MTP=2 (QOM-gated MTP drafts). Key by key against
  lcoh: R57_QOM_SETS humaneval,mbpp,lcb6 -> humaneval,mbpp,ifeval,multi,lcb6,ifbench,gpqad, and R57_QOM_MTP=2 added;
  nothing else differs. The shipped 45 GB chat configuration adds SB1 and PF-D v4 (EVALUATION Table 1b).
- Sat 256, MEM 47 GB. Card snapshot data/r57/snap/card (BigMoE e057abf), screen daabc51562c5, ext build_qom .so
  ff69fa24a745. BigMoE HEAD was e29df3d at the start and 82ac64b at the 15:26 resume. bigmoe/, ext/, snap/card and
  snap/frozen_card are identical at b37f36b (lcoh's start), e29df3d and 82ac64b (git diff --quiet), so both arms and
  all 42 of m1b's processes ran the same card code.
- Code pin 8ecc1604bd56 (amendment 9's).
- In every process the 38 outside modules equal BigMoE HEAD's (e29df3d in 38 processes, 82ac64b in the 4 resumed
  ones), and the manifests of the 41 later processes equal the first one's (results/card/es_cdm1b_091007_fwd1_modules.json;
  40 compared, 40 equal). QOM was engaged on every kept row: pf.qom_opt > 0 on 100 / 150 / 131 of the 100 GPQA, 150
  IFBench and 131 LCB rows in each order. Every kept key has exactly one uncut run.

**Process notes:**
- **Forward:** 10:07-13:05 (21 processes), 174 min of decoding. The arm yielded once at a process boundary,
  12:09:52-12:13:52, to the package's clean-clone run (4 min). The 600 s budget cut 10 runs (4 GPQA, 6 LCB); each was
  left out and its key re-run in the next process (amendment 9). Kept tokens 439,386 (lcoh's forward: 433,011).
- **Reverse:** 13:05-16:06 (21 processes: 17 under the first runner, 4 after the resume), 181 min of decoding, no
  yields. The budget cut 11 runs (4 GPQA, 7 LCB), each re-run. Kept tokens 433,181 (lcoh's reverse: 434,648).
- **Cutoff (amendments 10-11):** at 15:26:00 the first runner (pid 91678) stopped on the arms' 15:30 cutoff before
  reverse process 18 ("does not fit before the cutoff"; rc 6, no DONE file). tools/card_extend_cutoff.sh resumed m1b at
  15:26:12 with CARD_RESUME=1 and CARD_CUTOFF 16:20 (runner pid 98272; reference manifest = the first process's). Its
  four processes ran the 67 GPQA keys left (17 + 18 + 18 + 14 rows); the last started at 15:59:58, inside the 16:05
  limit, and ended at 16:06:19. No process was stopped. zs32's runner stopped at its first gate at 15:26:01 (15:30
  cutoff) and ran nothing: zs32 is not gated.
- **The cost of one arm:** forward 174 min, reverse 181 min: 5.9 h of decoding for 381 keys in two orders (process
  starts included); 5 h 59 min of wall time (10:07:26-16:06:50) with the yield and the resume.
  - Kept tokens: 872,567 (forward 439,386, reverse 433,181), 1,145 per row, about 41 tok/s over the 5.9 h. lcoh:
    867,659 tokens in 5.8 h, about 42 tok/s.
- **Scoring:** LCB verdicts come from the grader cache results/lcb6_grades.jsonl, keyed by (question, code sha). m1b's
  rows brought 79 codes not graded before (27 s; the other rows' codes were cached). No global timeout. One code hit the
  6 s per-test limit (question 3764, code -3) and fails, as in the official grader; it is an arm row, so as a pass it
  could only help the arm. A row with no fenced code is a fail. Verdict scoring 16:06:20-16:06:50.

## The lines (vs NAX6-N)
| Line | Result | Verdict |
|---|---|---|
| IFBench forward (n 150) | 44.0 % vs 44.0 %, delta +0.00 pt, McNemar p 1.0000 (arm only 9 / reference only 9) | PASS |
| IFBench reverse | 41.3 % vs 44.0 %, delta -2.67 pt, McNemar p 0.5034 (8 / 12) | PASS |
| GPQA Diamond forward (n 100) | 42.0 % vs 44.0 %, delta -2.00 pt, McNemar p 0.7539 (4 / 6) | PASS |
| GPQA Diamond reverse | 42.0 % vs 44.0 %, delta -2.00 pt, McNemar p 0.7905 (6 / 8) | PASS |
| LCB v6 forward (n 131) | 29.0 % vs 32.1 %, delta -3.05 pt, McNemar p 0.3877 (4 / 8) | PASS |
| LCB v6 reverse | 32.1 % vs 32.1 %, delta +0.00 pt, McNemar p 1.0000 (3 / 3) | PASS |
| **Card gate** | | **PASS** |

The rule: a paired delta above -5 points and an exact McNemar p >= 0.05, per benchmark and order; a benchmark passes
only in both orders; the card gate passes when all three do. Every p above was recomputed by hand from its discordant
counts (two-sided exact binomial). The queue owner (NewIdeas, 16:07) re-graded the GPQA and LCB lines independently and
got the same counts and p's.

## Information (no verdict; tools/card_info.py m1b)
- **IFBench (n 150):** loose 44.0 % forward, 41.3 % reverse vs 44.0 %; strict 37.3 / 38.7 % vs 40.0 %. Token-identical
  to the reference: 18 (forward), 17 (reverse); at the cap 15 / 14 (reference 10). Forward vs reverse: token-identical
  17 / 150, same score 132 / 150.
- **GPQA Diamond (n 100):** forward / reverse vs reference: Biology (n 7) 57.1 / 42.9 vs 57.1 %; Chemistry (n 47) 19.1 /
  23.4 vs 21.3 %; Physics (n 46) 63.0 / 60.9 vs 65.2 %. No row token-identical to the reference; 56 at the cap in each
  order (reference 54). Forward vs reverse: token-identical 0 / 100, same score 86 / 100.
- **LCB v6 (n 131):** forward / reverse vs reference: easy (n 31) 87.1 / 87.1 vs 93.5 %; medium (n 39) 25.6 / 35.9 vs
  28.2 %; hard (n 61) 1.6 / 1.6 vs 3.3 %; AtCoder (n 82) 26.8 / 30.5 vs 30.5 %; LeetCode (n 49) 32.7 / 34.7 vs 34.7 %.
  Token-identical to the reference: 2 (forward), 1 (reverse); at the cap 90 / 87 (reference 88). Forward vs reverse:
  token-identical 7 / 131, same score 121 / 131.
- With QOM on every set, the order changed rows on every set; in lcoh (no QOM on IFBench and GPQA) the two orders gave
  the same tokens there (150 / 150, 100 / 100).
- Against lcoh (L + COH: the same code and config without QOM on the chat sets and without R57_QOM_MTP): IFBench 44.0 /
  41.3 vs lcoh's 40.7 / 40.7 %; GPQA Diamond 42.0 / 42.0 vs 47.0 / 47.0 %; LCB v6 29.0 / 32.1 vs 30.5 / 29.0 %. Both
  arms are inside the rule on every line. This is not an A/B: one run per order, and QOM's rows depend on the order.

## Known limits
- Non-thinking greedy at the caps above. The reference stops at its cap on 10 IFBench, 54 GPQA and 88 LCB rows; arm and
  reference run under the same caps, so the gate asks whether the arm keeps the reference's results within that budget.
- Power at these N: an arm with no real loss whose rows differ from NAX6-N on 10-15 % of keys fails somewhere with a
  real chance (~25-50 % over 3 benchmarks x 2 orders; PREREG "Known limits"). A PASS bounds the loss at these N; it does
  not show the arm has none: LCB's forward -3.05 pt, IFBench's reverse -2.67 pt (strict -2.7 / -1.3 pt) and GPQA's
  -2.00 pt are inside the rule, not zero.
- QOM depends on what is cached, so m1b's rows depend on the order on every set (lcoh's only on LCB).
- The gate covers M1-B + COH at C 256 only. SB1 and PF-D v4 (in the shipped chat configuration) and the 32 GB point
  (zs32, C 160) have no card gate.

## Data
results/card_m1b_fwd, results/card_m1b_rev (rows), results/card/es_cdm1b_091007_* and es_cdm1b_091526_* (per-process
runs, tokens, manifests; 091526 = the resumed runner), results/card_m1b_<set>_<order>_d4.txt (the lines),
results/CARD_m1b_DONE, results/day61_m1b_runner.log, results/day61_zs32_runner.log, results/card_chain.log,
results/card_extend.log, results/lcb6_grades.jsonl. Reference: results/ref2tnaxn.r6.
