# LOSSY350_GOAL: the kept-tokens wall (GOAL session, Thu 5 Nov 2026; v1 16:27, v2 16:4x EET)

> **Superseded Fri 6 Nov 2026.** CD16 failed MBPP12 on quality in the engine (tests 11 / 12 -> 8 / 12; speed held): the
> drafter's wholesale comment / docstring text runs away. CD16, C16, G16 and D16 are withdrawn. The guarded arm T16 and
> its lines are in `build_1106/GUARD.md`; the account is in `K_WALL_1106.md` (update at the top).

Mission: LOSSY350-GOAL (the orchestrator, ~15:55). Wall: tokens kept per check, lossy track, 45 GB, code first.
Read first: `~/FableGod/fablec/AUDIT_1105.md` (its corrections to GOAL.md are adopted below), `~/TRIED/TRIED.md` (the whole
drafter section, lines 713-924, and the kept / acceptance lines 1226, 1256, 1269, 1274-1276), `~/EvalSuite/docs/RESULT_relax_2026-11-05.md`
and its PREREG. Method: CPU replays on disk data (numpy, `python -I`, one process, < 0.1 GB, swap 9.03 -> 8.93 GB, no GPU).
Marks: **[K]** measured (source named), **[P]** my replay or arithmetic on [K], **[G]** judgement.
v2 adds the stress tests (3.3), the exact half (3.2), the graded menu (3.4) and the test arms that follow from them.

## 0. The answer in nine lines

1. **One lever of the size this wall needs was found: stop paying full checks for text that cannot change the result.**
   On code answers 57-70 % of the answer tokens are comments + docstrings, 62-67 % of the drafter's first misses fall
   inside them, and 68-72 % of today's decode time goes to checks that start there [P].
2. **It has an exact half.** On HumanEval+ the docstring (56.5 % of answer tokens) is a 100 % verbatim copy of the prompt
   (mean run 144 tokens) [P]. A lookup-drafted copy check of 64 / 128 rows is exact and worth **x1.24 / x1.28** at today's
   row cost. MATCH 2.5 measured the 32-row form when a row cost 7.82 ms (x1.06 per check-ms [K TRIED]); at 2.8 ms it is x1.18.
3. **And a lossy half: INERT mode.** Inside comments and docstrings accept the draft wholesale (the span's end stays exact,
   code tokens stay exact) and let the check run to the span's end. Replay: tok/s **x1.27-1.30 / x1.47-1.53** on HumanEval+
   and **x1.63-1.65 / x1.79-1.87** on MBPP at 16 / 32-row checks; kept 9.24 -> 13.1 / 17.3 and 5.62 -> 10.7 / 13.0 [P].
   Fresh HumanEval+ windows: x1.18 / x1.35. Halved for the engine (Kamen's rule): +9 to +26 % HumanEval+, +32 to +43 % MBPP.
4. Why it is not rule C again: rule C and COMMENT-LOOPHOLE are likelihood rules (top-3 at >= 0.1 p1) on '#' comment rows at
   16 rows; they gave x1.07-1.13 and 0 accepts in docstrings [K TRIED 1226, 1269]. INERT drops the likelihood condition,
   adds docstrings, and removes the cut and the 16-row end inside a span.
5. The lead survived four CPU stress tests (3.3): the drafter's own span ends give the same gain; answers get 2-7 %
   shorter, not longer; a guard that keeps long comment blocks out still gives x1.07-1.26 / x1.41-1.55; the graded forms
   order as expected.
6. Everything else on this wall is small, with numbers: continuing a check past the first miss KILL (ceiling +4-8 %); every
   likelihood rule at today's block <= x1.06 at passing quality, <= x1.3 at any quality; chained blocks x1.12 at best.
7. Kept alone cannot reach 350: a check is ~62 ms + ~2.8 ms per row [P], so infinite kept caps at 357 tok/s and kept 20
   gives 165. **The path (3.6): the other walls' floor engines alone give 218-258 on fresh HumanEval+; INERT 32 on top
   gives 292-357; 350 appears at 25 ms + 1.3 ms per row.** Without the inert-text lever they stop ~100 tok/s short.
8. The open risk is quality and it is a GPU question: what the 125B writes after reading drafter-written comments. The
   exact half and the docstring half carry little of that risk; long comment blocks carry most of it.
9. Chat has no such spans. Nothing new for chat from this wall beyond the kills.

## 1. The cost model, and the audit's corrections adopted

- One FAST check on code = 90-95 ms at ~9.7 verified rows [K audit, DAY9 875]. Experts 22-27 ms for those rows (F2) = ~2.8 ms
  per row [P]; the rest (~62 ms: dense ~32-36, SSD floor ~12, drafter ~5.8, commit, glue) does not depend on rows [P].
  So **check(rows) = 62 + 2.8 x rows** and tok/s = kept / check. The replays reproduce today's line: 100-113 tok/s on the
  HumanEval+ walks vs 110-117 measured [K].
- What kept can buy: the limit of kept -> infinity is 1 / 2.8 ms = 357 tok/s. So this wall multiplies with the fixed-cost
  and per-row walls; it cannot replace them.
- GOAL.md's wrong numbers (dense 6 ms, SSD wait 55-60 ms, "latency not bandwidth") are dropped; nothing below uses them.
- Chat sensitivity (F3): every lossy rule on chat rows already sits near its line (R1(0.5) passes at -3.0 points, R2 fails
  at -5.0 [K RESULT_relax]). I propose nothing that touches chat rows or unsure code rows.

## 2. Ranked ideas, each with its number

Replay base (the official loop, 0.3 cut, DFlash2 L1-20 every-position walks): HumanEval+ 24 answers kept 9.24 per check,
MBPP 100 answers kept 5.62 [P]. "tok/s x" = against that base under 62 + 2.8. Two bookkeepings of rows differ by ~2 %; ranges
are given where they do.

| # | idea | tried? | number [P unless marked] | verdict |
|---|---|---|---|---|
| 1 | **COPYRUN: the HumanEval+ docstring as lookup-drafted copy checks of 64 / 128 rows** | T-A at 32 rows (MATCH 2.5, COPYGATE: kept x1.20-1.21, BETWEEN at 7.82 ms per row); U at 64+ rows and at today's row cost | docstring = 100 % verbatim prompt copy, 98 % in runs >= 64 tokens. tok/s x1.06 / **x1.18 / x1.24 / x1.28** at 16 / 32 / 64 / 128 rows (x1.13 / 1.16 / 1.18 at the old row cost). MBPP docstrings are 9 % verbatim: nothing there | **exact, no quality question**; first arm |
| 2 | **INERT mode: wholesale acceptance inside comments + docstrings, exact span ends, check runs to the span's end** | U (rule C = the likelihood corner, comments only, 16 rows) | HumanEval+ 24: kept 13.1 / 17.3 / 19.7 at 16 / 32 / 64 rows, tok/s **x1.27-1.30 / x1.47-1.53 / x1.50**, checks x0.71 / 0.53 / 0.47. Fresh 12: x1.18 / x1.30-1.35. MBPP 100: kept 10.7 / 13.0 / 14.1, tok/s **x1.63-1.65 / x1.79-1.87 / x1.78**. Drafter-written text 14-40 % (HE) / 33-45 % (MBPP) of tokens, 0 code tokens | **the bet** (section 3) |
| 2a | ... comments only | U | x1.20-1.21 HE (fresh x1.11), x1.21 MBPP at 16 rows; longer blocks add nothing. Rule C's replay: x1.074 fresh HE, x1.125 MBPP [K M-41] | the riskier half |
| 2b | ... docstrings only | U (likelihood rules get 0 there on HumanEval+ [K 1269]) | HE x1.05-1.06 / x1.13-1.18 (16 / 32 rows) = row 1 without the lookup; MBPP **x1.27 / x1.31-1.36** | the safer half; on HumanEval+ use row 1 instead |
| 2c | ... with the scratchpad guard (no wholesale in blocks of >= 3 consecutive comment lines) | U | HE x1.12 / x1.26 (fresh x1.07 / x1.19); MBPP x1.41 / x1.55 at 16 / 32 rows | the middle arm |
| 2d | ... graded instead of wholesale (profile, 16 rows, both spans) | rule C's form is T-A on comments | HE fresh: rule C form x1.04, p >= 0.01 x1.06, the 125B's top-8 x1.10, wholesale x1.16. MBPP: x1.17 / x1.24 / x1.35 / x1.63. Rule C's own form extended to docstrings: MBPP x1.07 -> x1.17 | **+9 % on MBPP in the shipped risk class** |
| 2e | ... + in-span rows on the cheapest gate (their outputs are unused; only their state matters) | U | if an in-span row costs 1.5 ms [G]: HE x1.31 / x1.59, MBPP x1.72 / x1.93 at 16 / 32 rows | for the expert-group and cost-per-group walls; F3's downstream-flip floor applies |
| 3 | Chained blocks: block 2 drafted from the end of block 1 inside one check (an ideal re-anchoring drafter), exact | U as a build; DEPTH32 KILL is the measured cousin | HE fresh: kept 13.8, tok/s x1.12; with R1(0.5) x1.15. MBPP: x0.98 exact, x1.05 with R1(0.5). 4 blocks worse than 2 | BETWEEN at best; such a drafter does not exist (EARLYTAP KILL) |
| 4 | Any likelihood rule at today's block (R1(delta), R2, "inside a KL budget", typical acceptance) | T-A / T-K (R1 0.5 on code fails decision 4, passes on chat) | profile ceilings, HE fresh / MBPP: R1(0.5) kept x1.033 / x1.060 at 0.7 / 1.8 % forgiven; R1(2) x1.084 / x1.163 at 1.7 / 5.0 %; "any token in the 125B's top-8" x1.183 / x1.301 at 4.0 / 10.5 % | **KILL as a 350 lever** |
| 5 | BRIDGE: after a miss commit the 125B's token and keep going through the draft's tail in the same check | U (M-06 RECYCLE is the exact cousin, KILL) | the draft's next token after its first miss is right 21.0 % code, 17.4 % MBPP, 21.8 % math, 14.4 % IFEval, 12.0 % multi. Ceiling with a perfect acceptor: tok/s x1.04 / 1.06 / 1.08 / 1.07 / 1.06 | **KILL** |
| 6 | A drafter with no errors on tokens the 125B is sure of (p1 >= 0.99), cut unchanged | measurement | HE x1.055, MBPP x1.037 kept. First misses by the 125B's p1: HE 48 % at >= 0.99, 38 % at 0.5-0.9; MBPP 18 % / 49 % | on HumanEval+ those sure-token misses are mostly docstring miscopies (row 1 removes them exactly) |
| 7 | New names at binding sites; blank-line count | U | 0.9 / 0.8 % and 1.7 / 1.3 % of first misses (HE / MBPP); kept +0.2-0.5 % | **KILL** (too rare) |
| 8 | Lower cut (0.1) or no cut, exact | T-K (S45d the other way) | kept x1.03-1.05, tok/s x0.97-1.03 | KILL |

Reading the table: at passing quality the acceptance family gives 3-6 %, the block-length family 12 %, and one family gives
18-87 %: stop verifying text that cannot change the result, exactly where it is a copy and wholesale where it is not.

## 3. The bet: inert text gets one cheap pass, not a check per miss

### 3.1 What the replay found [P]

| | HumanEval+ (24 answers, 6,375 answer tokens) | MBPP (100 answers, 15,641) |
|---|---|---|
| answer tokens: docstring / comment / code | 56.5 % / 13.1 % / 29.4 % | 37.7 % / 19.6 % / 41.0 % |
| first misses: in a comment / in a docstring / at a span's end / on real code tokens | 34.1 % / 28.0 % / 8.4 % / 26.6 % | 26.8 % / 40.5 % / 4.9 % / 24.8 % |
| DFlash2's per-token accuracy: docstring / comment / code | 0.928 / 0.465 / 0.860 | 0.632 / 0.453 / 0.778 |
| today's checks by where they start: comment / docstring / code (share of time; kept) | 28 % (4.1) / 44 % (13.3) / 29 % (9.5) | 28 % (3.9) / 40 % (5.3) / 32 % (7.9) |
| docstring tokens that are a verbatim copy of the prompt | 100.0 % (mean run 144 tokens) | 9.1 % |

The 125B re-writes the prompt's docstring in every HumanEval+ answer and writes its own docstrings and comments on MBPP.
None of that text is executed by the tests. Today's check treats a wrong word in a docstring exactly like a wrong operator:
the check ends, 62 ms are paid, the next check starts. A check that starts in a comment keeps 4 tokens.

### 3.2 The exact half: COPYRUN (HumanEval+-style answers)

- When the answer enters a span the prompt holds verbatim (the k-gram match MATCH / COPYGATE already compute), the draft for
  the rest of that span comes from the prompt and one check verifies 64-128 rows of it. The 125B's verdicts are used as
  today, so the output is unchanged; a wrong lookup token is an ordinary miss.
- Replay: every docstring row kept, check runs to the docstring's end: x1.18 / x1.24 / x1.28 at 32 / 64 / 128 rows [P].
  The engine needs checks longer than 32 rows on the copy path (kernels: FORM_ROWS 16..64 [K R-78b]) and the copy gate on
  those rows.
- It removes the docstring half of the lossy question on HumanEval+: what stays lossy there is comments only.

### 3.3 The lossy half: INERT, the rule

- A causal tagger gives every row a state: code, comment (after '#' on the line), docstring (inside triple quotes). This is
  R57_RELAX_SPAN's tagger [K built, bit-equal to the replay on 21,504 rows] plus one state for triple-quoted strings.
- On a row whose draft sits inside a comment or docstring: **the draft is kept whatever the 125B's argmax is.** The 125B's
  rows still run (they build its state); their verdict is not used.
- **Exact at the edges:** a draft that would end the span (a newline in a comment, a quote in a docstring) is verified as
  today; the row must already be inside the span before its draft (a drafted '#' or opening quote stays exact); rule C's
  tool-prefix guard (`# type:` and friends) stays; lines starting `>>>` stay exact.
- **Inside a span the cut and the 16-row end do not apply:** the check runs as long as the drafter's block lasts (16 or 32
  rows; DFlash2 L1-20 is the block-32 drafter [K TRIED L1-20]).

Stress tests on CPU (`inert_v2.out`) [P]:

| test | HumanEval+ 24 | fresh 12 | MBPP 100 |
|---|---|---|---|
| A. span ends where the DRAFT's own chain ends it, 16 rows (v1 used the 125B's span ends) | x1.30 (v1 x1.30) | x1.18 (x1.18) | x1.65 (x1.66) |
| A. answer length ratio T (emitted / the 125B's own) and time per answer | 0.975, x0.75 | 0.980, x0.83 | 0.931, x0.57 |
| B. comment tokens in blocks of >= 3 consecutive comment lines (the scratchpad suspects) | 61 % | 73 % | 58 % |
| B. guarded INERT (docstrings + comments in blocks <= 2 lines), 16 / 32 rows | x1.12 / x1.26 | x1.07 / x1.19 | x1.41 / x1.55 |
| C. after INERT 32: share of time in checks that start in code; their kept | 56 %; 13.2 | 58 %; 13.0 | 77 %; 10.6 |

So the gain does not depend on my span-end assumption, the drafter does not inflate answers (it ends spans a little early;
the 125B still has the last word at every span end), and after INERT the remaining time is real code.

### 3.4 Why it should hold quality, where it may not, and the graded menu [G unless marked]

- Code tokens deviate 0 % by construction; the tests run the same code unless the 125B writes different code after reading
  different comments. That is the one risk.
- Evidence so far: rule C's arm B deviated on 1.37 % of positions, all in comments, and HumanEval+ went 95.0 -> 93.3 (one
  task of 60, inside decision 4) [K RESULT_relax]. INERT deviates on 14-45 % of tokens; that evidence does not carry over.
- Docstrings: on HumanEval+ use COPYRUN (no deviation at all). On MBPP the drafter is right on 0.63 of docstring tokens;
  the text is a summary written before the code (examples in `inert_examples.out`). Medium-low risk.
- Comments: the drafter is right on 0.45-0.59 of tokens, and 58-73 % of comment tokens sit in blocks of three or more
  lines, some of them the 125B's scratchpad ("# Let me reconsider ..."). This is the half that can fail.
- The menu, from safest to largest (profile numbers, 16 rows, both spans, fresh HumanEval+ / MBPP) [P]:
  rule C's own form extended to docstrings x1.04 / **x1.17**; p(draft) >= 0.01 x1.06 / x1.24; the 125B's top-8 x1.10 / x1.35;
  wholesale with the scratchpad guard x1.07 / x1.41; wholesale x1.16-1.18 / x1.63-1.65. Each step up accepts text further
  from what the 125B would have written.

### 3.5 The tests (<= 10 min each, GPU, first slots after Fri 00:15) and their lines, written now

Engine arms on BigMoE's r57 relaxed copy, C 224, A-B-A-B against qall, 12 HumanEval (the fresh windows) + 12 MBPP, tests
executed per task. Per task: S = wall seconds, T = tokens to EOS, K = kept per check, V = test verdict, W = share of tokens
kept wholesale.

1. **COPYRUN64** (exact; HumanEval only): lookup-drafted docstring, checks up to 64 rows on the copy path.
   PASS: every token equal to qall's AND median S ratio <= 0.90 (replay x1.24). KILL: S > 0.96 (long checks cost more than
   2.8 ms per row), or any token differs (a build bug).
2. **CD16** then **CD32**: comments + docstrings wholesale (on HumanEval the docstring via COPYRUN).
   PASS (CD32): median S ratio <= 0.87 on HumanEval and <= 0.75 on MBPP AND V equal to the control's on >= 23 of 24 tasks
   with no net loss above 1 AND median T ratio <= 1.10. BETWEEN: speed passes and exactly 2 tasks flip to fail, or T ratio
   1.10-1.25: fall back to arm 3. KILL: >= 3 of 24 tasks flip to fail, or T ratio > 1.25, or S > 0.95 on both sets.
3. **G32** (the guarded arm): docstrings + comments in blocks of <= 2 lines wholesale; long comment blocks on the 125B's
   top-8. PASS: S <= 0.93 on HumanEval and <= 0.80 on MBPP, V equal on 24 of 24.
4. Any relaxed accept on a code row voids an arm (a tagger bug, not a result).

On a PASS: EvalSuite decision 4 on HumanEval+ 60 with the passing arm (math rows are untouched by the rule), then the stack
with the split form and the dense wall, then in-span rows on the cheapest gate (row 2e) as its own arm.

CPU work before the slot (no GPU, minutes): MoonShots runs the wholesale flag + the docstring state through the engine
patch's own tagger on the 21,504 walk rows. PASS = my numbers within +-0.03 (x1.18 fresh HumanEval at 16 rows, x1.65 MBPP).
The MBPP +9 % of row 2d (rule C's form on docstrings) needs only the docstring state in the same patch.

### 3.6 What it is worth on the 350 line: the path table [P replay tok/s; the engines right of "today" are the other walls' targets, G]

Engine = fixed ms per check + ms per row. "in-span 1.0" = rows inside comments / docstrings cost 1.0 ms (row 2e).
Replay numbers, not halved (`path350.out`).

| HumanEval+ fresh 12 | today 62 + 2.8 | 45 + 2.2 | 30 + 1.5 | 25 + 1.3 | 30 + 1.5, in-span 1.0 | 25 + 1.3, in-span 1.0 |
|---|---|---|---|---|---|---|
| today's rule | 110 | 147 | 218 | 258 | 218 | 258 |
| COPYRUN 64 (exact) | 133 | 178 | 264 | 311 | 276 | 321 |
| INERT 16 | 130 | 173 | 257 | 303 | 261 | 307 |
| guarded INERT 32 | 131 | 174 | 259 | 306 | 269 | 313 |
| **INERT 32** | 148 | 197 | 292 | **344** | 307 | **357** |
| INERT, one pass per span (<= 128 rows) | 162 | 214 | 318 | **374** | **344** | **395** |

| | today | 30 + 1.5 | 25 + 1.3 | 25 + 1.3, in-span 1.0 |
|---|---|---|---|---|
| HumanEval+ all 24: today's rule / INERT 32 / one pass per span | 100 / 153 / 170 | 200 / 301 / 333 | 236 / 355 / 391 | 236 / 370 / 417 |
| MBPP 100: today's rule / INERT 32 / one pass per span | 67 / 126 / 135 | 136 / 249 / 266 | 161 / 294 / 314 | 161 / 306 / 330 |

Reading: the other walls alone put fresh HumanEval+ at 218-258 with their floor engines (FULL's measured floor engine is
210-280 at this kept [K audit]). **350 appears when INERT 32 sits on an engine of 25 ms + 1.3 ms per row, or when one pass per
span and 1.0 ms in-span rows sit on 30 + 1.5.** Without the inert-text lever the same engines stop about 100 tok/s short.
MBPP-style answers reach 250-330. On today's engine: COPYRUN64 ~133 (exact), INERT 32 ~148; halved, 120-130.

## 4. Kills and ceilings, for the record (TRIED lines for the recorder; I write only here)

- **BRIDGE** (non-prefix acceptance; U -> KILL on CPU): after its first miss the draft's own chain is right on the next
  token 12-21 % of the time; ceiling tok/s +4 % code, +6 % MBPP, +8 % math, +7 % IFEval, +6 % multi with a perfect acceptor.
  Also closes "whole-span acceptance after a miss" and any carry of the old tail into the next check.
- **ACCEPT-CEILING** (likelihood rules, profile on the 12 fresh HumanEval+ + 100 MBPP windows with the 125B's top-8):
  R1(0.5) x1.033 / x1.060; R1(1) x1.052 / x1.103; R1(2) x1.084 / x1.163; R2 x1.052 / x1.089; top-8 x1.183 / x1.301.
  A KL budget is a re-ordering of the same accepts; it cannot exceed the top-8 line.
- **CHAIN2 / CHAIN4** (chained re-anchored blocks, ideal drafter): exact x1.12 / x1.08 on HumanEval+, x0.98 / x0.87 on MBPP.
- **MISS-ANATOMY** (measurement): first misses by the 125B's own p1, HumanEval+ fresh: 48.1 % at >= 0.99, 5.9 % at
  0.9-0.99, 37.8 % at 0.5-0.9, 8.1 % below 0.5; MBPP 18.0 / 16.0 / 48.8 / 17.1 %. The draft token is in the 125B's top-8
  at 78-95 % of misses and is its rank 2 at 38-65 %.
- **BINDNAME / BLANK** (U -> KILL): 0.8-0.9 % and 1.3-1.7 % of first misses.
- **INERT-CENSUS** (measurement): the token shares, miss shares and time shares of table 3.1.
- **SPAN-SWITCH** (U, exact, not replayed: no MTP-chain picks by span on disk) [G]: inside comment spans draft with the MTP
  chain, chat's better drafter (D-17: 1.08-1.32x on chat), instead of DFlash2 (0.45-0.59 per token there). Comment-start
  checks are 28 % of time at kept 3.9-4.1; +30 % kept there is ~+5 % tok/s net of the chain's steps. Only worth it where
  comments must stay verified; INERT makes it moot.

## 5. Notes for the other walls

- **The code line is mostly not code.** 57-70 % of what the 125B emits on HumanEval+ / MBPP is docstring and comment text,
  and 68-72 % of the decode time is spent on checks that start there. A cost-per-row or expert-group lever priced "on code"
  is priced mostly on that text. In-span rows are the natural place for the most aggressive gate (row 2e), with F3's caveat.
- **After INERT the wall is real code:** 56-77 % of the time in checks that start in code, kept 10.6-13.2. That is where
  the per-row and fixed-cost levers then act.
- **Publishing:** a tok/s figure on code should be reported with this split (code tokens / inert tokens) and with the
  wholesale share W, so the lossy mode is described exactly.
- **MTP Head:** a drafter with perfect sure-token accuracy gains x1.04-1.06 at today's block and cut (row 6); the capacity
  drafter's value is in comments and model-written docstrings (0.45-0.63 per token), or INERT makes those rows moot.
- **Chat:** no inert spans by syntax. The only chat numbers here are kills (BRIDGE +6-7 %).

## 6. Files (all in ~/FableGod/goal/)

- `inert_classes.py` -> `inert_classes.out` (token shares, first misses by class, profile forgiveness), `inert_mode.out`
  (INERT at 16 / 32 / 64 rows), `inert_mode_split.out` (fresh / seen windows, accuracy by span), `inert_examples.out`
  (the 125B's text vs the draft's inside spans).
- `inert_v2.py` -> `inert_v2.out` (stress tests A-D); `copyrun_check.out` (verbatim share, exact copy-run pricing);
  `path350.out` (absolute tok/s by engine and rule).
- `accept_ceiling.py` -> `accept_ceiling.out` (miss anatomy by p1, likelihood-rule ceilings), `accept_ceiling2.out`
  (lower cut, chained blocks).
- `bridge_replay.py` -> `bridge_replay.out` (tail anatomy, bridge ceilings; all five domains).
- Data read: `~/TrainLab/data/d02b_walk_d23h_dfl_{code,mbpp,math,ifeval,multi}.npz`,
  `~/Breakthrough/data/rulec/rulec_{he12,mbpp}.json`, the cached Qwen3.6 `tokenizer.json` (decodes the 125B's ids; checked
  on text).

Caveats of the replays [P]: teacher-forced along the 125B's own answers (the convention of M-41 and the HPD walks); each
INERT check stops at the span's end (conservative); 32 / 64-row in-span blocks assume the drafter supplies that many drafts
(DFlash2 block 32 exists; 64 is [G]); test A assumes the 125B agrees where the draft ends a span; the cost model is two
numbers; my lexer is a small state machine checked by eye on three answers, not the engine's tagger; COPYRUN's numbers
assume the lookup is right wherever the copy is verbatim and that a long copy check costs 62 + 2.8 ms per row.
