# K scoreboard by workload (MTP Head, Fri 2026-11-06 11:3x, engine section Sat 11-07 08:2x; K_PLAN_1106 item 2)

K = tokens kept per 16-row check. [K] = measured, [P] = predicted (replay). The drafter is DFlash2 L1-20 throughout
(the engine runs it as D7B, 4-bit layers, code kept x1.00).

## Engine [K], the levers measured Fri day + Sat night (BigMoE data/fday, data/sat; FAST-NAX, C 224)

**Bottom line.** The shipped code profile (v1 = stackb) keeps **11.3 per check on the 12 HumanEval**, x1.08 over the
stack, all of it from MATCH g2@16. The two big K levers are not shippable for code: CD16 failed quality, relax v2 is
speed only. The guarded T16 is BETWEEN. Safe K on code is ~11.2-11.3 on HumanEval and ~6-6.7 on MBPP12.

12 HumanEval, 2-3 passes per arm. "geo" = BigMoE's figure (geo mean over prompts); "pooled" = all tokens / all checks,
the same definition as the walks below.

| arm | adds | geo | pooled | vs stack | status |
|---|---|---|---|---|---|
| qall | FAST-NAX base | 10.45 | 9.95 | | control |
| stack | + DN-KS, GDN lazy, D7BNAX_32, FASTTAX (no CD16) | 10.48 | 10.01 | x1.00 | |
| stackb (v1's code profile) | + MATCH g2@16 (+ span B: 0 relaxed accepts) | 11.33 | 11.16 | x1.08 / x1.115 | SHIPPED (v1, v1'); C 256: 11.34 / 11.12 |
| stackr | + relax v2 (M-44: DELTA + SPAN) | 11.64 | 11.39 | x1.11 / x1.14 | speed only (relaxes code tokens; never in code_env) |
| stackrm | + relax v2 + MATCH | 11.74 | 11.54 | x1.12 / x1.15 | speed only |
| stack + CD16 (Fri day run) | + CD16 | 12.58 | 12.21 | x1.20 / x1.22 (across runs) | withdrawn (MBPP quality KILL) |

12 MBPP, CAP 384, 2 passes per arm (FAST output differs pass to pass: Fri's two qall passes pool to 5.61 and 6.27).

| pair | control | arm | ratio (median per task / pooled) | tests passed, 2 passes | verdict |
|---|---|---|---|---|---|
| CD16 (Fri day, c966c0d) | qall 6.48 geo / 5.89 pooled | 9.38 / 9.76 | x1.449 / x1.66 | 22/24 -> 16/24 | speed PASS, quality KILL (comment loops to the cap) |
| T16 (Sat, GOAL's guard: a comment / docstring row kept only if the 125B ranks it top 3 at >= 0.1 p1) | qall 6.09 pooled | 6.73 pooled | x1.035 / x1.105 | 22/24 -> 21/24 | BETWEEN (K line PASS >= 1.08, KILL < 1.03) |

[P] vs [K]:
- CD16, HumanEval: [P] x1.24-1.26 (GOAL) vs [K] x1.20 / x1.22 (across runs; GOAL's same-block read x1.19-1.27). Held.
- CD16, MBPP: [P] x1.8-1.9 vs [K] x1.45 / x1.66. The replay holds the 125B's text fixed. In the engine the drafter's
  comment text ran away and changed the text.
- T16, MBPP12: [P] x1.15 (x1.08 with the gain halved) vs [K] x1.035 / x1.105.
- T16, HumanEval (Sat 14:31-14:37, on v1' + wired at C 256, REPORT 'T16-HE'): kept 10.56 -> 10.89 geo (x1.031), 10.75
  -> 11.11 pooled (x1.033), the low end of [P] x1.03-1.08. Speed net 0 (173.2 vs 173.2 tok/s whole decode): the tagger
  adds +1.9 ms per cycle. KILL.
- Relax v2, HumanEval: [P] x1.10-1.23 (MoonShots) vs [K] x1.11 / x1.14, the low end.
- MATCH g2@16, HumanEval: walk [K] x1.038 (24 HE+) vs engine x1.08 / x1.115 (12 HE). The engine arms' texts differ (238
  vs 250 tokens per task), so the engine figure carries text noise.

## Walks and replays (Fri 11:3x; the CD16 column is withdrawn, see above)

| workload | n | exact, official loop [K] | exact, every anchor [K] | under CD16 [P, GOAL] | sources |
|---|---|---|---|---|---|
| 12 HumanEval, engine FAST-NAX, C 224 | 12 prompts | 10.36 (5.7-14.2 per prompt) | | tok/s x1.15 | BigMoE fnx_report |
| HumanEval+ walk | 24 answers | 9.57 | 11.64 (6,351 anchors) | fresh 12: 10.44 -> 12.91-13.16 | d02b_walk.txt, d23h_dfl |
| MBPP walk (held out from L1-20) | 100 answers | 5.94 | 7.07 (15,541 anchors) | 5.61 -> 10.69-10.87 | MATCH RESULT, d23h_dfl, GOAL |
| docstring-free: checks after the docstring | MBPP / HE | MBPP 5.5, HE 6.2-7.7 [P] | | MBPP 9.6-9.8, HE 10.7-10.8 | GOAL K_WALL_1106 |
| GSM8K math | 60 answers | 7.00 | 8.23 (14,260 anchors) | | d02b_walk.txt, d23h_dfl |
| agent | 12 windows | 3.01 | | | MATCH RESULT |

How to read it:
- The every-anchor mean is higher than the official loop because it weights every position equally. The official loop
  lands its anchors after misses, which are the harder spots. GOAL's 5.61 for MBPP uses the engine's 0.3 cut, my 5.94
  the uncut official loop.
- Exact K: HumanEval is ~1.6x MBPP. Under CD16 the two converge (~10.7-13), because CD16 accepts comment and docstring
  text wholesale. After CD16 the gap is the drafter's depth on code rows (GOAL: MBPP pick 0.60 at depths 12-15).
- Noise for retrain comparisons: one reseed moves code by 0.16 and math by 0.01 (SEED).

Data for CPU replays: notes/K_WALL_DATA.md. The every-anchor files are data/d02b_walk_d23h_dfl_{code,math,mbpp,multi,ifeval}.npz
(per anchor: dtok, dcf, ttok, tcf). Top-16 candidates at every anchor exist only for the 36 decision-map windows
(d02b_walk_tree3_*), not for MBPP or math.
