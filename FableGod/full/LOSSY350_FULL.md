# LOSSY350-FULL: the cross-wall bet (Fable FULL, Thu 5 Nov 2026, v2.2 16:5x EET)

Mission (the orchestrator 16:00, from Kamen): the single bet most likely to reach 350 tok/s LOSSY on code at 45 GB within about a
week, never tried or an assumption broken, with a <= 10-min first test and its PASS / KILL lines.
Rules kept: `~/TRIED/TRIED.md` grepped first; CPU replays only (numpy, `python -I`, one process, < 0.5 GB, under 1 s each; swap
flat at 8.94-8.96 GB throughout); no GPU; writes only in `~/FableGod/full/`. Marks: [K] measured (source named), [P] replay or
arithmetic on [K], [G] judgement. Scripts and outputs beside this file: `copyrun_replay`, `copyrun_groups`, `copyrun_tiles`,
`copyrun_engine12`, `nocomment_replay`, `nocomment_bands`, `nocomment_chain`, `nocomment_alt`, `nocomment_rest`, and for section 8
`price_b_vs_c`, `price_c_livecut`, `price_split_nocom`, `price_copy_rule`, `price_all` (`.py` + `.out` each).
History: v1 / v1.1 (16:2x) named the comment-free mode alone. Classifying the misses it leaves showed what the answers are
made of, and that changed the order: the copy run is the larger part and needs no product decision.

## 0. The bet in seven lines
1. **What a HumanEval answer is [K `copyrun_replay.out`, the HPD-0 walk, 24 answers, 6,351 generated tokens]: 61.1 % of the
   generated tokens are ONE verbatim copy run of the prompt** (the re-emitted signature + docstring; exactly 1.0 run per answer,
   median 140 tokens, max 397). Comments are 14.1 %. **Real code is 28 %.**
2. **Where the checks go [K]:** 273 of the walk's 666 checks start inside that copy run and keep 13.4 tokens each; DFlash2
   still misses 2.5 % of those rows (95 misses) where a lookup cannot. The other 393 checks keep 6.9. First misses by region:
   comments 52 %, the copied docstring 24 %, real code 24 %.
3. **Assumption broken: "a check is 16 rows of the drafter's block, and every token is a decision."** Three quarters of the
   tokens are not decisions. THE BET: **spend a check only where a decision is needed.**
   - **A. Copy runs go through wide strict passes with no drafter** (64 rows; the k-gram lookup supplies the tokens; every
     copied row is a strict row of the split form). Output unchanged; no product call; works for EXACT with exact rows.
   - **B. Comments leave the path** (a comment-free FAST code mode: the 125B never opens a `#` comment in a code block).
     Output changes; Kamen's call; decision 4 judges it.
   - What is left for the drafter and the 16-row check is the real code.
4. **Size [P]:** tokens per check 9.54 -> 13.7 with A at width 64 (x1.44; 12.2 at width 32, 14.6 at 128) -> 21-24 with A + B.
   A 64-row strict copy pass computes 2,153 expert groups, the same as one of today's FAST checks, and 89 % of its experts
   were already used by the prompt's own rows [K `copyrun_groups.out`, recorded routing]. tok/s: A alone x1.3-1.5; A + B
   x1.9-2.4 (the low end of A + B and the high end of A are the engine's own 12 prompts at CAP 256).
5. **On the line [P]:** today's 95 ms engine 110-117 -> ~150-170 (A) -> ~215-270 (A + B). The all-in + split engine
   (~62 ms, this week's build, unproven) -> ~230-260 (A) -> **~335-415 (A + B)**; the low end is the engine's own 12 prompts
   at CAP 256, where comments are only 7.4 % of the measured tokens (section 1). With A alone, 350 sits at a ~45 ms engine.
   So 350 moves from "under the floor" to "this week's engine plus two modes", at the edge, not with room.
6. **Never tried in this form:** MATCH 2.5 (a 32-row block inside long copies) went BETWEEN at x1.204 kept and was priced
   x1.06 per check-ms, because rows were priced exact (~3 ms each) [K TRIED 782, copygate/GATE.md 71, 156]; the engine's general
   32-row check was KILLed at -62 % (exact rows, DFlash2-32 drafts) [K NIGHT_1104 76]. Strict copy rows, width 64-128 and no
   drafter were never priced or run. A comment-free mode: 0 grep hits anywhere.
   Priced beside the split form and rule C, in the engine's own live-cut convention, in section 8: on the 12 map windows
   split 0.9 x1.15, rule C x1.01-1.03, B x1.23, A x1.38, A + B + split 0.9 x2.49.
7. **First test (<= 10 min GPU, Fri after 00:15):** time a 32-row strict copy check on the 12 HumanEval prompts. PASS <= 3.0 ms
   per copied token and tok/s >= +15 %; KILL >= 4.5 ms per copied token or tok/s < +5 % (section 5).

## 1. Part A: the copy run [K unless marked]

| HPD-0 walk (DFlash2 L1-20, block 16, exact acceptance) | HumanEval+ | MBPP |
|---|---|---|
| answers, generated tokens | 24, 6,351 | 100, 15,541 |
| verbatim prompt-copy runs >= 8 tokens: per answer, median / mean / max length | **1.0, 140 / 162 / 397** | 0.5, 11 / 12 / 26 |
| generated tokens inside such a run | **61.1 %** (44.8 % in runs >= 128) | 3.7 % |
| usable copy tokens (after an 8-token trigger) | 3,691 = **58.1 %** | 1.2 % |
| regions of the generated text: docstring / real code / comments / prose / other strings | 56.7 (100 % copied) / 28.0 (16 % copied, the signature) / 14.1 / 0.8 / 0.3 % | 37.9 (8 % copied) / 39.2 / 21.1 / 1.3 / 0.5 % |
| DFlash2's first-miss hazard per row: copy tokens / the rest | 0.0251 (95 misses) / 0.0942 (303) | 0.039 / 0.116 |
| the walk's checks: starting on a usable copy token (tokens kept each) / the others | **273 (13.42)** / 393 (6.90) | 13 / 2,617 (5.93) |
| checks a copy block of width W needs for the same copy tokens: W 16 / 32 / 64 / 128 | 258 / **129 / 71 / 42** | - |

What a wide copy pass computes [K `copyrun_groups.out`: the decision map's recorded routing, 12 code windows]:

| block of W copied rows | rows per block (runs end) | groups, ranks 0-1 (a strict row's loads) | groups, ranks 0-9 (exact rows) | ranks 0-1 experts already used by the prompt's rows |
|---|---|---|---|---|
| W 16 | 15.8 | 949 (59.9 per row) | 3,870 (244 per row) | 90.2 % |
| W 32 | 30.9 | 1,498 (48.5 per row) | 5,599 | 89.7 % |
| **W 64** | 56.1 | **2,153 (38.4 per row)** | 7,352 (131 per row) | 89.1 % |
| W 128 | 98.8 | 3,010 (30.5 per row) | 9,229 (93 per row) | 88.3 % |
| for scale: 16 non-copy answer rows | 16 | 840 | 3,312 | |

Readings.
- **A copied row is the cheapest row there is [P on K].** It needs no draft call (the lookup is on the CPU), no top-10 (in
  FAST it is a strict row: ranks 0-1 + the block's members, the split form's rule, map PASS on code at 0.92 / 1.31 % [K
  Breakthrough 10:36]), and its experts are warm (the prompt's rows used 89-90 % of them). At width 64 it costs 38 groups; an
  exact row at width 16 costs 244.
- **Price of a 64-row strict copy pass on today's engine [P]:** dense x1.70 of the 16-row pass (D32's two points, 17.96 ms
  at 16 rows and 22.19 at 32 [K TRIED 478], read as 13.7 ms fixed + 0.264 ms per row; 64 rows unmeasured) = 48-61 ms on
  today's 28-36; experts 2,380 padded 16-row tiles (2,153 groups x1.11 [K `copyrun_tiles.out`]) x 8.5-9.2 us [K KERNEL90 per
  group] = 20-22 ms; draft 0; glue 5; SSD 12-20 (the wait law's floor + a few cold groups [K FAR 7b]). **85-108 ms for 56-64
  tokens = 1.4-1.9 ms per copied token**, against 5.9-7.1 today (95 ms for 13.4-16 tokens). At width 32: 66-79 ms for 31
  tokens = 2.1-2.6 ms (1,537 tiles).
- **A wide strict row is nearer to exact than a 16-row one [K `copyrun_tiles.out`]:** the block's union grows with its width,
  so each copied row keeps more of its own ranks as members: 3.96 of 10 at width 16, 4.70 at 32, 5.45 at 64, 6.20 at 128.
  The split form's map PASS was measured at the 16-row share; the wide pass computes more per row, not less.
- **Tokens per check [P]:** 6,351 / 666 = 9.54 today; with copy blocks of 32 / 64 / 128 rows the 273 copy checks become
  129 / 71 / 42: 12.2 / 13.7 / 14.6 tokens per check.
- **tok/s [P]:** on today's FAST engine take copy checks at ~80 ms and the others at ~105 (average 95): 63.1 s for the 6,351
  tokens. With 71 wide passes at 90 ms: 47.7 s = **x1.32**; if today's copy checks cost the average, x1.45.
- **EXACT too [P]:** with exact rows a 64-row copy pass is 7,352 groups (~68 ms) + dense ~51 + SSD + glue = ~135-150 ms for 56
  tokens = 2.4-2.7 ms per copied token against 9.4 (126 ms / 13.4): 83 -> ~115 tok/s EXACT, if wide exact rows are bitwise
  (NAX6 is certified at 1-16 rows; Tuesday's 32-row check differed on 3 / 11 prompts).
- **Why the old price missed it:** MATCH 2.5 and M-18c / U-32 priced a row at the exact check's ~3 ms (480 pairs on the
  per-pair kernel) and the width at 32. The strict rule (2 ranks), the per-group kernels and the missing draft call change
  the row's price about fivefold.
- **Inside the benchmark's own window [K / P]:** the engine's line is 12 HumanEval prompts at CAP 256 [K BigMoE
  `scripts/allin_wed.sh`]. Every one of the walk's 24 answers starts its copy run at token 2, and within the first 256
  generated tokens the copy share is **67.8 %**. Cross-check of the model: 67.8 copied tokens at ~6.5 ms + 32.2 others at
  13.8 ms (95 ms / 6.9 kept) = 885 ms per 100 tokens = 113 tok/s, which is the engine's own 110-117 [K]. So today's line is
  already two speeds: ~155 tok/s on the copy, ~72 on the rest. With A at 1.6 ms per copied token: 552 ms per 100 = 181
  tok/s (x1.6); with A + B ~318 ms per 88 tokens = ~278 (x2.5). The table in section 3 uses the full answers (58 %
  usable copy), the more careful of the two. (The walk's 24 tasks are not the engine's 12; the shape is the same in all 24.)
- **The engine's own 12 prompts, checked [K `copyrun_engine12.out`: EvalSuite `results/ref2t/decode.jsonl`, the stored
  REF2-T answers, CodeScan regions]:** all 12 answers re-emit the docstring. In their first 256 tokens: docstring 57.0 %, real code 32.3 %, comments
  7.4 %, strings 2.0 %, prose 1.4 % (full answers: 50.5 / 36.3 / 9.7 %; all 60 HumanEval+ rows: 49.9 / 35.5 / 12.2 %). With
  the signature the copy is ~60 % of the measured tokens. The cap cuts most comments off (they come later, in the body), so
  **on the line as it is measured A is worth more and B less than in the walk's full answers** [P]: per 100 measured
  tokens, today 60 x 6.5 + 40 x 12.4 ms = 884 ms (113 tok/s); with A 60 x 1.6 + 494 = 590 ms (**x1.5**); with A + B
  96 + 32 x 10.2 = 422 ms for 92 tokens (**x1.9**).
- **It is the HumanEval shape, and I say so:** the run exists because the prompt contains the function's docstring and the
  model re-emits it. MBPP has no such run (1.2 % usable). File-re-emitting edits in agent work should have longer ones (not
  measured here; MATCH read agent x1.03-1.05 at 16 rows [K TRIED 782]). The headline has to name the workload.

## 2. Part B: the comment-free mode [K unless marked; `nocomment_*.out`]
- Comment tokens are 13.3 % of HumanEval+'s generated tokens (14.3 % of the answer goes with them, counting one newline per
  comment-only line; MBPP 21.2 %) and carry **52 % of all first misses** (207 of 398): hazard 0.169 per comment row, 0.139 on
  the newline that ends a comment, 0.085 on code right after a comment line, 0.030 on other code.
- Comment rows are 55 % unsure (cf < 0.9: bands 19 / 36 / 22 / 23 %); code rows 12 % (2 / 10 / 14 / 74 %). Without comments the
  split form's strict share goes from 80.3 % to 87.3 % of rows.
- Kept per check, one convention (rows to the first miss on full-width checks, + the bonus) [P]: base 10.12; the chain on the
  code rows' own hazard by depth x1.23 (working); pessimistic (the "which step next" choice moves onto the next code line at
  the comment rows' hazard) x1.17; comment-free stretches as recorded x1.31; MBPP x1.11. Halved for planning x1.115.
- **The mask is not a nudge** [K `nocomment_alt.out`, the 125B's top-8 on the 12 map windows]: at the 43 comment openers the
  opener is the 125B's top-1 on 43 / 43. Where code follows inside the window (24 openers), its best non-comment token is
  its second choice on 24 / 24, a median 2.44 nats below (17 % below 1 % of p1), and equals the token the recorded text
  resumes with on 14 / 24. So ~4 lines in 10 would take a different next statement: a different valid path, which is why
  decision 4 judges this part and why the prompt form (the model plans without comments) is tested first.
- What the misses on non-comment rows are [K `nocomment_rest.out`]: 97 of 191 inside the docstring (the copy run: part A
  removes them), 35 punctuation, 26 layout, 12 identifiers seen before, 1 new name at a binding site. So after A and B the
  remaining misses are the real code's (94 on 2,044 rows, hazard 0.046), and there is no third free slice (binding-site
  names are 0.5 %).
- Forms: P = one sentence in the prompt ("Do not write comments in the code."), zero build. M = `R57_NOCOM`: on rows whose
  pre-draft CodeScan state is code, tokens that would open a comment get -inf before the argmax; drafted comment spans are
  stripped from the block; reference = the AR decode with the same mask (the mode is exact against its own reference).

## 3. A + B together, and the line [P]
Non-copy part after B: 6,351 - 3,691 copied - 907 comment tokens = 1,753 tokens at 9.2-11.5 kept per check (a flat chain at
the real code's hazard 0.046-0.055, less a discount for the hazard's rise with depth) = 152-190 checks, against 393 today.
Total 71 + 152..190 = 223-261 checks for 5,444 tokens: **21-24 tokens per check (x2.2-2.5 of 9.54)**.
Time on today's engine: 71 x 90 + (152..190) x 105 ms = 22.4-26.4 s for 5,444 tokens against 63.1 s for 6,351: **tok/s
x2.0-2.4, seconds per answer x0.35-0.42.**

| engine (average ms per check) | today's path | + A (copy passes, W 64) | + A + B |
|---|---|---|---|
| today, 95 ms [K: 110-117 tok/s] | 113 | 149-164 | **232-271** |
| the all-in + split engine, ~62 ms [P FULL section 1 + audit F4; unproven] | 173 | 228-251 (260 on the engine's 12) | **355-415** (334 on the engine's 12) |
| the realistic engine + split, ~45 ms [G FULL] | 239 | 315-346 | 489-573 |
| the FAST floor + split, 33-35 ms [P FULL] | 307-325 | 405-470 | 630-780 |

- The table scales both kinds of check with the engine, which flatters the copy pass slightly where the dense term shrinks
  less than the rest; halve the gains (Kamen's rule): A x1.16-1.22, A + B x1.5-1.7, i.e. 350 at the ~45 ms engine with both.
- **Odds [G]:** the wide strict pass costs <= 100 ms at W 64 in the real engine ~55 % (the one unmeasured number; W 32 exists
  and ran Tuesday, slowly, with exact rows); B passes its engine test ~60 % and decision 4 ~70 % of that; the engine reaches
  ~60 ms within a week ~35 %. 350 within about a week: **~8-10 %** with the bet (A + B + a ~58-62 ms engine), ~1 % without;
  250+ within a week ~50 % against ~15 %.
- What the number means: tok/s counts the copied docstring, today as well (about 60 % of the measured tokens). A body-only
  answer (no re-emitted docstring) would be ~2x shorter for the user and slower in tok/s, because the easy tokens leave.
  Kamen's line is tok/s; I price that and flag this.

## 4. Why this is the cross-wall bet
- It does not lower any wall's ms; it removes the tokens from the walls. A copied token pays no drafter, shares one dense
  pass with 55-63 others, computes 38 groups instead of 244, and reads no cold expert; a removed comment pays nothing.
  Every wall session's gain (A cost per group, B dense, C / Lunatic groups, FAR waits) multiplies it.
- Size against the table: the audit's best halved gains are +7-9 % (split) and +4-6 % (rule C) [K AUDIT_1105]; A alone is
  x1.16-1.22 halved with the output unchanged, A + B x1.5-1.7 halved.
- Against GOAL's wall (kept per check): the drafter is not asked to get better; 72 % of its rows are taken away from it.
- Cost: A = the k-gram lookup (MATCH, built) + a wide check (32 rows ran Tuesday; 64 needs the dense kernels at 64 rows, D32
  is the template) + the split rule (built 10:48); B = a sentence, then ~40 lines.

## 5. The tests (lines written now)
**T-A1 (first; <= 10 min GPU, Fri after 00:15, BigMoE): the 32-row strict copy check.** 12 HumanEval prompts, C 224. Arm S =
qall + split (16 rows). Arm W32 = the same, and on a k-gram hit (>= 8 tokens matched, >= 31 source tokens left) the block is
the next 31 copied tokens with no draft call, every copied row strict. S-W32-S-W32. Per check: rows, kept, ms, groups,
misses.
- PASS: <= 3.0 ms per kept token on the copy blocks AND >= 28 of 31 copied rows kept on average AND tok/s on the 12 prompts
  >= +15 % over S. Prediction [P]: 2.1-2.6 ms per copied token, tok/s +18-25 %.
- KILL: >= 4.5 ms per copied token, or tok/s < +5 %. Then the wide row is not cheap in this engine and the old x1.06 stands.
- BETWEEN: the group count per block is as replayed (1,500 +- 15 %) but the ms are not: the cost is in the wide kernels,
  not the rows; hand the per-term log to Leap.
**T-A2 (after a PASS; Leap + BigMoE): width 64.** PASS <= 1.8 ms per copied token; then the engine A/B on the 12 prompts,
PASS tok/s >= +28 % over S.
**T-B1 (<= 10 min GPU, zero build): the prompt form.** Arm A = the prompts as they are; arm P = the same + "Do not write
comments in the code."; A-P-A-P; per task tokens, comment tokens, kept per check, tok/s, seconds; then the 12 tasks'
HumanEval+ tests on the CPU. Inside CAP 256 comments are only 7.4 % of these prompts' tokens, so the lines are set for that
window [P: B alone there = 722 ms per 92.6 tokens against 884 per 100, tok/s x1.13]. PASS: comment tokens <= 2 % AND kept
on the checks that start outside the copy run >= x1.15 AND tok/s >= +8 % AND no task that passes in A fails in P. KILL:
that kept < x1.05, or tok/s < +3 %, or >= 2 new failures. BETWEEN: the sentence is ignored -> form M decides. Prediction
[P]: kept outside the copy run x1.2 (8.5 -> 10.3), tok/s +10-13 %, tokens -7-8 % (more on uncapped answers: -14 %).
**T-B2 (after T-B1; EvalSuite, ~1.5 h GPU): decision 4 for the comment-free mode** on HumanEval+ 60 + math 50 + IFEval 100
against NAX6; bars read against the masked reference. Part A needs no new quality test in FAST beyond the split form's own
(the per-row rule is the same; a wider block only adds members).

## 6. What can kill it
1. **The wide pass is slow in the real engine.** Tuesday's general 32-row check ran at -62 % with exact rows and DFlash2-32
   drafts; nobody has timed strict copy rows. T-A1 is exactly this number.
2. **Run ends:** one partly wasted block per run (counted in the 71); a wrong copy source wastes one pass (COPYGATE's duel
   exists for this [K copygate/GATE.md]).
3. **B is a product change** (no inline comments in FAST code answers) and the 125B wants its comments; T-B2 decides.
4. **The path after B may be less predictable than the recorded comment-free stretches** (pessimistic x1.17 instead of
   x1.23); T-B1 reads it.
5. **Workload shape:** A is worth nothing on MBPP-like prompts (and on math or chat: 0.9 / 2.9 % copy). B helps there
   (comments are 21 % of MBPP's tokens and 33 % of its first misses); its model-written docstrings (38 % of tokens, 42 % of
   first misses, hazard 0.125) are prose too, and leaving them out would be a second product decision, not part of this bet.

## 7. What I did and did not do
Done: nine CPU replays on recorded data (the walks, the 125B's top-8 and the routing capture of the 12 map windows, EvalSuite's
stored REF2-T answers). Not done:
any timing (the wide pass's ms are arithmetic on [K] pieces), any generation (B's real kept and quality), agent-turn copy
runs, a web check for prior art (MoonShots: copy acceleration is published as prompt-lookup decoding, LLMA "inference with
reference" and predicted outputs [G from memory]; what I do not know published is the MoE pricing, a copied row as a strict
2-rank row in one wide pass, or removing a token class from the target to raise acceptance). The replays were run before
these lines were written; v1's first kept figure (x1.234) used the wrong convention and was corrected by
`nocomment_chain.py` (the working number stayed x1.23, from the chain).

## 8. Priced beside the split form and rule C (the orchestrator's 16:4x ask) [P replays on K; `price_*.out`]
Two conventions, kept apart. The no-cut walk (rule C's own: MoonShots' `span_relax_c` machinery, kept(16) from the per-depth
acceptance) and the ENGINE's (Fable C's `bytes10_replay`: live-cut blocks at t 0.3, the 45 GB pools, one session, the audit's
rules; my copy reproduces the audit's 320 / 291.5 / 175.3 / 136.7 tiles per kept token exactly).

**8.1 B beside rule C, same rows, rule C's convention [`price_b_vs_c.out`; reproduces M-41's x1.189 / x1.074 / x1.137 / x1.125].**

| top-8 source | exact kept(16) | rule C | comment-free (B) | B + rule C | rule C forgives, of the comment-row misses | B over rule C |
|---|---|---|---|---|---|---|
| the 12 map windows | 7.78 | x1.189 | x1.464 | x1.464 | 66 / 160 (41 %) | x1.232 |
| the fresh 12 | 10.15 | x1.074 | x1.176 | x1.176 | 21 / 64 (33 %) | x1.095 |
| pooled 24 | 8.75 | x1.137 | **x1.331** | x1.331 | 87 / 224 (39 %) | **x1.170** |
| MBPP 100 | 5.01 | x1.125 | x1.187 | x1.187 | 294 / 830 (35 %) | x1.055 |

Overlap: complete and one-sided. Rule C forgives a third to two fifths of the misses that B removes; with B in place rule C
has no row left to act on (B + C = B on every set); B adds x1.06-1.23 on top of rule C.

**8.2 The same two in the engine's convention [`price_c_livecut.out`, `price_split_nocom.out`; the 12 map windows].**
- Today's 427 live checks keep 3,378 tokens (7.91 each). **149 of them (35 %) keep only comment tokens, 2.62 per check**;
  167 (39 %) sit in the copy run (12.47 per check); about 110 (26 %) are real-code checks.
- **Rule C under the live cut reaches x1.013-1.030 kept per check**, against x1.189 in the no-cut walk on the same windows:
  198 blocks end at the cut with no miss to forgive, 229 end at a miss, 109 of those on a comment row, 43 forgiven. The cut
  ends the block before the comment row is verified. RELAX measured the comment rule at +1.8 % kept in the engine [K
  RESULT_relax]: inside this bracket. So rule C's engine value is ~+2 %, not its profile's +14-19 %.
- **B under the live cut, lower bound** (comment rows and tokens dropped, blocks not re-formed): 277 checks, 10.20 kept
  each = **x1.29 kept per check, 35 % fewer checks**; on comment-free blocks as recorded 11.49 (x1.45, biased up). B is not
  shrunk by the cut the way rule C is, because it removes the rows that trigger the cut instead of forgiving them.

**8.3 Everything together, engine convention [`price_all.out`].** Today's terms per check [K FULL section 1: dense 36,
experts 23, draft 10, glue 5, SSD 21 = 95 ms]: a live check = 72 ms + 9.97 us x its tiles; a wide copy pass = dense x1.24
(W 32) or x1.70 (W 64) + glue + SSD x1.2 / x1.45 [G] + 9.97 us x its padded tiles, no draft. tok/s against today's preset:

| arm (12 map windows: 61.7 % of kept tokens in the copy run, 16.3 % comments) | QOM (today's rule) | + split 0.99 | + split 0.9 |
|---|---|---|---|
| today | 1.000 | 1.107 | **1.148** |
| + rule C (engine reach; +1.8 % measured) | 1.01-1.03 | 1.12-1.14 | 1.16-1.18 |
| + B, lower bound | **1.231** | 1.416 | **1.475** |
| + A, W 32 | 1.276 | 1.433 | 1.471 |
| + A, W 64 | **1.378** | 1.519 | **1.562** |
| + A (W 32) + B | 1.806 | 2.176 | 2.230 |
| + A (W 64) + B | 2.065 | 2.425 | **2.492** |

- **B x split compounds above the product** (1.231 x 1.148 = 1.413 against 1.475 measured in the replay): without comments
  the rows are 88 % sure instead of 81 %, so SPLIT90's tiles per check stay ~1,100 (1,081 -> 1,104) while the check keeps
  10.2 tokens instead of 7.9. Tiles per kept token: QOM 291.5 -> 270.4 with B; SPLIT90 136.7 -> 108.3.
- **A leans on the split rule** [`price_copy_rule.out`]: a 64-row copy pass keeps 5,976 tiles under today's QOM rule (its
  resident keep lets a copied row compute nearly everything) against 2,153 strict. A alone is x1.38; A over split 0.9 is
  x1.36 (1.562 / 1.148); split over A is x1.13. Section 1's price of the copy pass assumed strict rows, i.e. it contains
  that part of the split form.
- **A x B x split is far above the product** (1.378 x 1.231 x 1.148 = 1.95 against 2.49): once the copy run is in 40 wide
  passes, the 149 comment-only checks are 57 % of the 260 checks that remain.
- **Rule C** is inside B and worth ~+2 % in the engine where B is not taken.
- SSD: loads that miss per kept token are ~2.1-2.25 in every arm (QOM's loads are the rows' ranks 0-1 in all of them); B
  and A change the checks, not the bytes per token.

**8.4 On the benchmark's own window** (the engine's 12 at CAP 256: copy ~60 %, comments 7.4 %, code 32.6 % [K section 1])
[P: 8.3's per-token costs re-weighted: copy token 8.4 ms today, 6.8 with split 0.9, 2.3 in a 64-row strict pass, 3.0 in a
64-row QOM pass; comment token ~23 ms; code token 13.6 ms, 12.1 with split, 11.9 with B + split]:

| | QOM | + split 0.9 |
|---|---|---|
| today | 1.00 | x1.16 |
| + rule C | ~x1.01 | ~x1.17 |
| + B (lower bound) | x1.09 | x1.30 |
| + A (W 64) | x1.40 | x1.60 |
| + A (W 64) + B | x1.65 | **x1.97** |

So on the line as it is measured: the split form first (x1.16, already built), then A (x1.38 more), then B (x1.23 more,
x1.09 alone); rule C adds nothing once B is in. Against the engine's 110-117 tok/s that is ~128-136 with split 0.9, ~176-187
with A, ~217-230 with A + B, before any of the owned engine work; at the all-in terms (dense 28, draft 6, SSD 12, 8.5 us per
tile [P FULL, unproven]) the same stack is ~300-335 [P]. This agrees with section 3's low end and replaces its wider range
for the measured line.

**8.5 What this changes in the tests.** T-B1's prediction is now in the engine's convention: kept per check outside the copy
run x1.3 or more on these prompts' full answers (every comment-only check goes), x1.10-1.16 over all checks inside CAP 256;
its lines stand. Rule C's engine arm (Fri ~01:40) should read ~+2 % kept: if it reads more than +5 %, 8.2's cut argument is
wrong and B's lower bound should be re-derived. A new free read for the same ticket: the share of checks that keep only
comment tokens (prediction ~15-20 % of checks inside CAP 256, 35 % on full answers).
