# K wall: kept tokens per 16-row check on average code (GOAL, Fri 6 Nov 2026, first list 10:50 EET)

> (a quote from the project owner, kept private)
docstring-free HumanEval). Read first: `~/TrainLab/K_PLAN_1106.md`. CPU replays only; nothing outside this folder written.
Marks: [K] measured by its owner, [P] my replay / arithmetic on stored walks, [G] judgement. K = kept per check with the
engine's 0.3 confidence cut (MTP Head's 9.57 / 5.94 are the same walks without the cut; mine reproduce them: 9.58 / 5.93).

## Update Fri 21:2x: the engine's numbers on CD16, and what replaces it

- **CD16's speed held in the engine, its quality did not [K, BigMoE c966c0d, data/fday].** MBPP12: kept 6.48 -> 9.38
  (x1.449; my line >= 1.30), tok/s +23.7 %, tokens per task x1.02, but tests 11 / 12 -> 8 / 12 in both passes: my KILL
  line. HumanEval, inside the stack: kept in all / kept exact on the same blocks x1.19-1.27 (replay x1.24-1.26).
- **Why it failed:** runaway text. The drafter's free comment or docstring text leaves the 125B's distribution at its
  first or second deviating token (ranks 11 to 6,395, margins -6 to -20), and then the 125B follows the babble with its
  own argmax until the cap. The risk I had named for MBPP (drafter text replacing the 125B's plan) is what happened; the
  HumanEval pair could not show it, since its docstrings are copies.
- **So item 2 below changes:** CD16 is still the only lever of that size on K, but it is not shippable. Every arm that
  keeps text wholesale (CD16, C16, G16, D16's docstrings) is withdrawn.
- **What replaces it is small: T16** (`build_1106/GUARD.md`): a comment or docstring row is kept only if the 125B ranks
  the draft in its top 3 at p >= 0.1 p1. Replay: kept x1.08 on HumanEval, x1.15 on MBPP12 (x1.04 / x1.08 with the gain
  halved) [P]. Built as a module-only change, self-tests pass, sent to BigMoE for tonight's ticket with an MBPP12 line.
- **For the 300 / 200 gate:** the kept-tokens wall has no safe lever of CD16's size. On HumanEval the safe pieces are
  T16 (comments) and the exact copy path for the copied docstring (MATCH: x1.04 kept at 16 rows, x1.20 at 32, the latter
  blocked by the lossy preset's 16-row assert).
- Still owed: one read of tonight's T16 / relax v2 numbers against `build_1106/GUARD.md` sections 5-6, then stop.

## 0. The answer (first list, 10:50; item 2 is amended by the update above)

1. **K 14 on MBPP-like code is not reachable with any drafter in sight [P].** With comments and docstrings accepted and an
   autoregressive drafter that had the 125B's own features at every step, MBPP would still keep 13.3-13.6. It needs code
   rows right 99 % of the time at every depth. HumanEval-like code can reach ~13.2 with what is already built.
2. **One correction to the plan: CD16 is not copy drafting, and on MBPP it is not ~x1.0.** It accepts comment and
   docstring text wholesale. Replay: MBPP 5.6 -> 10.3-10.9 kept (x1.8-1.9; x1.66-1.69 on the 90 answers that finish, see
   section 1), docstring-free HumanEval 6.2-7.7 -> 10.7-10.8, whole HumanEval 10.4 -> 13.2 [P]. It is the only lever of
   this size. MBPP has no engine number yet; the 12-task arm is specified in `mbpp12_1106/MBPP12.md`.
3. **Why exact K is stuck on MBPP:** 57 % of its tokens are comment / docstring text the 125B writes fresh. With a perfect
   code drafter, exact K on MBPP would still be only 6.8-7.1 [P].
4. **What is left after CD16 is a depth wall, not a picking problem.** On code tokens the 125B is sure of, DFlash2 is right
   99 % of the time at depths 1-3 and 60 % at depths 12-15 (MBPP). At deep code misses the 125B's token is absent from the
   16 candidates or below rank 3 in 56 % of cases (HumanEval+, the only candidate walk), so a picker cannot fix it. Only
   fresher 125B features can.
5. **The one mechanism that gives fresh features is an autoregressive tail (the trained MTP chain past depth 4).** Its
   ceiling is +24 % K on MBPP and +12 % on HumanEval on top of CD16 [P], but the chain costs more time than that today:
   it pays only if it adds under ~13 ms per check at the ceiling, and it costs ~20 ms [K, K_PLAN]. Ranked second on K,
   priced as no gain.
6. **Everything small is measured and small:** fresh variable names +2 %, span-boundary rules <= +4.5 % all together,
   prompt-name lookup +1 %, a looser cut +1-3 % K for +10-27 % rows.
7. **So the 300 tok/s gate on average code moves to the cycle:** at K 10.3-10.9 (MBPP) it needs a 34-36 ms cycle, at
   K 13.2 (HumanEval) 44 ms. At the priced 48-55 ms the replay's K gives MBPP ~185-225 tok/s and HumanEval ~240-275; with
   the gain halved (Kamen's rule: K ~8.1 and ~11.7) ~150-170 and ~215-245 [P, at a full check's cycle].
8. **Asks:** an MBPP prompt set in BigMoE's screen (section 5); every-position walks from MTP Head (section 5).

## 1. Where K stands and what ends a check [P]

`kwall_replay.py` on MTP Head's every-position DFlash2 walks (the 125B's own answers), the engine's own span tagger.
CD16-B = CD16 where the 125B's code after a drafter-written comment is the code it wrote after its own comment (the
convention under which CD16 keeps a task's result at all; convention A, used in the pre-check, stops at the span end).

| set | exact | CD16 (A / B) | checks that start in the body: exact -> CD16-B | rows per check, exact -> CD16-B |
|---|---|---|---|---|
| MBPP 100 | 5.61 | 10.69 / 10.87 | 5.47 -> 9.84 | 7.6 -> 12.5 |
| MBPP 94 (see note) | 5.69 | - / 10.28 | 5.61 -> 9.59 | 7.7 -> 12.1 |
| HumanEval+ fresh 12 | 10.44 | 12.91 / 13.16 | 7.72 -> 10.82 | 11.9 -> 14.2 |
| HumanEval+ all 24 | 9.24 | 12.98 / 13.20 | 6.19 -> 10.66 | 10.9 -> 14.2 |

- "Body" = after the signature and the first docstring: this is the docstring-free number. The engine's 10.36 on its
  12 HumanEval prompts matches the replay's 10.44 on other prompts.
- Note on MBPP: 6 of the 100 answers run to the token cap inside one docstring (18 % of all MBPP tokens). MBPP 94 leaves
  them out; quote both. Added 10:5x, the finer split: MBPP has three kinds of answers. 35 are bare code under 60 tokens
  (7 % of tokens; 7.38 -> 7.38, CD16 gives nothing). 55 are commented answers that finish (60 % of tokens; 5.8 -> 10.5).
  10 never finish and sit at the walk's token cap (33 % of tokens, 88 % of it comment / docstring text; 5.02 -> 13.3).
  **The 90 that finish: 5.94 -> 9.84-10.06 (x1.66-1.69).** The all-100 x1.9 leans on the ten runaways.
- What ends a check under CD16-B (MBPP 100 / HumanEval+ fresh): every draft kept 45 % / 60 %; a code token 30 % / 21 %;
  the confidence cut 15 % / 10 %; a span opening, a span ending or a guarded row 10 % / 9 %.
- The code stops, by the 125B's own confidence on that row: p1 >= 0.99 in 41-43 %, 0.9-0.99 in 15-19 %, below 0.9 in
  40-43 %. By class (MBPP, share of all checks): operators 7.3 %, new names 6.8 %, names seen in the answer 3.5 %, names
  from the prompt 2.5 %, whitespace 4.1 %, keywords 2.2 %, builtins, strings and numbers 3.2 %.

**The depth wall.** DFlash2's pick on the right prefix (teacher-forced), code tokens only:

| | depth 1-3 | 4-7 | 8-11 | 12-15 |
|---|---|---|---|---|
| MBPP, rows the 125B is sure of (p1 >= 0.99; 82 % of code tokens) | 0.993 | 0.927 | 0.776 | 0.596 |
| HumanEval+ fresh, the same (88 %) | 0.998 | 0.978 | 0.889 | 0.712 |
| MBPP, rows with p1 0.5-0.99 (17 %), depth 1 / 8 / 15 | 0.89 | | 0.51 | 0.30 |

The "no depth decay = +3 %" kill (SD-AR, on paper) was read on copy-heavy HumanEval. On MBPP it does not hold: an
autoregressive tail past depth 4 with the 125B's features is worth x1.21 kept even on exact acceptance (section 2, idea 2).

**Ceilings on top of CD16-B** (an oracle on one thing; everything else as the drafter is):

| what is made perfect | MBPP 94 | HumanEval+ fresh 12 |
|---|---|---|
| CD16-B itself | 10.28 | 13.16 |
| every stop at a span opening / ending | x1.045 (MBPP 100) | x1.019 |
| every code stop, the drafter's cut kept | 11.82 (x1.150) | 13.98 (x1.062) |
| every code stop, no cut | 13.84 | 15.14 |
| autoregressive tail past depth 4, the 125B's own features | 12.77 (x1.242) | 14.74 (x1.120) |
| the same at every depth | 13.34 (x1.298) | 14.89 (x1.132) |
| code rows right with probability 0.97 / 0.99 at every depth, no cut | 12.93 / 13.94 | 13.92 / 14.84 |

## 2. Ranked ideas for MBPP and docstring-free HumanEval

### 1. INERT CD16: keep comment and docstring text wholesale (lossy; built)
- **Mechanism:** rows inside a comment or a docstring of a python fence are kept whatever the draft says; span ends, code
  tokens, tool comments and doctest lines stay exact. Build: `build_1105/INSTALL.md`.
- **Tag:** adapted. Learned lossy judges on code are published (AutoJudge arXiv 2504.20039, Judge Decoding 2501.19309,
  FLy 2511.22972); a rule on comment / docstring spans was not found (`prior_art_1106.md`, B.1).
- **Predicted K [P]:** the table in section 1. MBPP x1.8-1.9, HumanEval body x1.4-1.7. Halved for the engine (Kamen's
  rule): MBPP ~8.0-8.3, HumanEval ~11.7.
- **Replay and line:** done Thu (pre-check PASS at x1.155 / x1.614 tok/s); today's census gives the K above.
- **GPU step:** tonight's arm on the 12 HumanEval (BigMoE). **Missing: MBPP.** The arm is specified in
  `mbpp12_1106/MBPP12.md`: 12 tasks chosen by a fixed rule, EvalSuite's existing `dec_mbpp` items (same prompts as the
  walk), qall with and without `R57_INERT=CD16`. Replay for these 12: 6.13 -> 9.64-9.83 (x1.57-1.60). Line: PASS = K ratio
  >= 1.30 and tokens per task <= 1.05 and verdicts equal on >= 11 of 12; KILL = ratio < 1.15, or tokens >= 1.15, or >= 2
  tasks flip to fail. Then EvalSuite decision 4 on MBPP. (The first version of this line said >= 1.40, from all-MBPP's
  x1.8-1.9 halved; it moved at 10:5x, before any engine run, to half the gain of the 12 tasks that will actually run.)
- **Cost:** one `SETOF` entry and `R57_QOM_SETS` in BigMoE's screen, ~10 min of GPU. The patch is ready.
- **Risk, stated:** MBPP docstrings are where the 125B plans (6 answers think inside one until the cap). There the
  drafter's text replaces the 125B's reasoning, which HumanEval's copied docstrings do not test. Fallbacks if MBPP's
  quality fails: G16 / D16, or a top-8 guard on the span text (Thursday's replay: x1.35 tok/s on MBPP against x1.63).

### 2. An autoregressive tail for the rows CD16 leaves exact (exact picks; priced as no gain today)
- **Mechanism:** DFlash2 keeps depths 1-4 and all inert text; from depth 5 the trained MTP chain (b5), which carries its
  own state token by token, re-picks the code rows. Under CD16 the comment rows need no head call.
- **Tag:** adapted (EAGLE-3-style head, arXiv 2503.01840; block + autoregressive tail hybrids are published).
- **Predicted K:** ceiling 12.77 on MBPP 94 (x1.242), 14.74 on HumanEval (x1.120) [P, R7]. A real chain reaches part of
  it: [G] MBPP 11.3-11.8, HumanEval 13.8-14.2. On exact acceptance the ceiling is x1.21 / x1.20.
- **Replay and line (R7, written before the run): PASS** (line: tail past depth 4 >= x1.12 on MBPP 94; got x1.242).
  A tail that starts at depth 9 is worth only x1.08-1.09: it must start early, so nearly the whole chain runs.
- **GPU step (MTP Head):** one every-position walk that stores the b5 chain's picks and confidences on MBPP 100 and
  HumanEval+ 24; I re-score it under CD16-B. Line: PASS >= 11.5 on MBPP 94; KILL < 10.9.
- **Cost, and why it ranks as no gain:** the tail pays only if its added time per check is below (K gain) x cycle. At the
  ceiling and a 55 ms cycle that is 13 ms for the 15 chain steps (16 ms at 68 ms): under ~0.9 ms per step, and about half
  that for a real chain, which reaches only part of the ceiling. A chain step is 1.35 ms today with Leap's screened head
  [K, MTP Head 10:5x], ~20 ms for the chain, and nothing under 0.5 ms is in sight: MTP Head's answer is "not now". Reopen
  only if a step reaches ~0.5 ms. This agrees with the MTPPICK kill; the new part is the size of the prize under CD16.

### 3. Judge and train the drafter on CD16's K, not on exact K (exact picks; slow lever)
- **Mechanism:** under CD16 only code rows and span ends decide a check, yet every retrain so far (SEED, P-CAP, P-OFF,
  data x2 / x4, 2-step) was judged on exact K, where 62-67 % of first misses are comment / docstring text. (a) Re-score
  the stored variants under CD16-B. (b) Retrain L1-20's recipe with no loss on comment / docstring tokens and flat depth
  weights (checks now reach depth 15 in 45-60 % of cases; DFlash's published loss down-weights deep rows, and whether
  L1-20's does is MTP Head's to check).
- **Tag:** adapted (DFlash's depth-decayed loss; DFlare arXiv 2606.02091, whose gain is mostly 3x data from a new
  source: +0.47 per doubling against our +0.13 on same-source data).
- **Predicted K [G]:** +0.2-0.6. The deep misses are missing information (R6), which a loss cannot add.
- **Replay and line:** (a) needs each variant's picks at every position (only the shipping drafter has them). Line: a
  variant counts only if its CD16-B K differs from L1-20's by more than twice the seed-1 delta, on MBPP and HumanEval.
- **GPU step (MTP Head):** the walks for (a), minutes each; (b) one retrain night, judged the same way.
- **Cost:** (a) ~15 min of GPU; (b) ~1 h of GPU.

### 4. FLy's window on code choice points (lossy; published)
- **Mechanism:** keep a mismatch on a code row where the 125B is unsure, if the 125B then agrees with the draft's next
  4-6 rows. Those rows are already computed in the same check.
- **Tag:** published (FLy, arXiv 2511.22972, training-free: HumanEval tau 12.6 with an 8B drafter, score unchanged). Two
  2026 audits find code loses the most pass rate under lossy rules (arXiv 2610.23818, 2607.08690).
- **Predicted K:** bound x1.08-1.09 on MBPP, x1.02-1.05 on HumanEval (every code stop with p1 < 0.99 kept) [P, R5]; a
  real window keeps part of it: [G] +3-5 % on MBPP.
- **Replay:** not possible from disk. It needs the 125B's picks on the draft's own path after a miss; MoonShots' M-06
  capture stored only summaries.
- **GPU step:** MoonShots' `recycle_capture.py` phase B storing each row's pick and p1, ~300 first-miss checks on MBPP
  and HumanEval (~5 min). Line: PASS >= x1.04 on MBPP with >= 70 % of rescued tokens being names, whitespace or strings;
  KILL < x1.02, or > 30 % of rescued tokens operators / keywords / numbers.
- **Cost:** the capture, then a tagger change and a decision-4 run. Low rank: small K, the riskiest class.

### 5. ALPHA: let the drafter name new variables (lossy, inert; BETWEEN)
- **Mechanism:** where the 125B starts a new identifier and the draft proposes another new name, keep the draft's; the
  125B goes on with that name.
- **Tag:** adapted (AutoJudge notes that renames can be skipped; no rule published).
- **K [P]:** x1.022 on MBPP 94 for the realistic subset, x1.000-1.003 on HumanEval (R8; ceiling over all new names
  x1.043 on MBPP 94, x1.005-1.009 on HumanEval).
- **Line (written before): BETWEEN** (PASS >= x1.03, KILL < x1.015). Best folded into idea 4 as its safe subset.

## 3. Killed today, with numbers [P]

| idea | line | result |
|---|---|---|
| ALIGN: the decay comes from comment text between anchor and row (a drafter trained to skip it would fix it) | gap >= 0.10 PASS, < 0.03 KILL | **KILL**: sure-code accuracy at depths 8-15 is 0.69 with none between and 0.71 with >= 5 between |
| A picker over the 16 candidates on deep code rows | rank 2-3 in >= 70 % PASS; absent or rank > 3 in >= 50 % KILL | **KILL**: rank 2-3 in 44.6 %, absent 37.7 %, rank 4-16 17.8 % (HumanEval+, depth >= 8) |
| PROMPT-NAME: complete identifiers that occur in the prompt by lookup (exact) | >= x1.03 PASS, < x1.015 KILL | **KILL**: x1.007-1.014 |
| Span-boundary rules: the drafter may skip, open or end a comment / docstring where the 125B would not | (census) | **KILL**: all of them together x1.045 on MBPP, x1.019-1.027 on HumanEval, at oracle level |
| A looser confidence cut under CD16 | (sweep) | **KILL**: 0.3 -> 0 gives +0.1-0.3 K for +1.4-3.3 rows; tok/s x0.91-0.99 under three cost models |
| REFRESH: fresh 125B features every D rows (the ceiling behind idea 2) | K(D = 4) >= x1.15 on MBPP PASS; K(D = 1) < x1.08 KILL | **BETWEEN**: D = 4 x1.109-1.131 on MBPP, x1.08-1.09 on HumanEval; D = 2 x1.18-1.21 on MBPP; D = 1 x1.25-1.30 on MBPP, x1.13 on HumanEval |

Carried from Thursday (not re-run): continuing past the first miss, likelihood rules on code, chained blocks, blank lines.

## 4. What tonight's CD16 numbers should show (written before they exist)

On the engine's 12 HumanEval, qall against qall + CD16 [P from other prompts of the same kind]:
- kept per check 10.4 -> 12.9-13.2 (the report's K column, arm / control: x1.24-1.26; with the gain halved x1.12);
- share of checks with all 15 kept 34 % -> 57-60 %; rows per check 11.9 -> 14.0-14.2; tokens per task 0.97-0.98.
- If K rises but tok/s does not, the rows explain it (+2.3 per check): read seconds per task, not only K.
- K and tok/s count drafter-written comment tokens. On MBPP a comment or docstring token costs about 1/15 of a check
  under CD16 and a body token about 1/10, so a comment-free answer (FULL's arm) has lower K and still finishes sooner.
  Report seconds per task beside K.

## 5. Asks

- **BigMoE / the orchestrator:** an MBPP prompt set in the screen (12 of the walk's 100 tasks), so CD16's K and its
  quality are measured where the lever is largest. Until then "K on average code" is a replay number.
- **MTP Head** (answered 10:5x): (1) yes, Saturday's MBPP / math walks will be stored at every position; (2) the MTP
  chain walk not now (a step is 1.35 ms); (3) the variant walks noted, lowest priority.
- **From the desk read:** z-lab's 0.8B DFlash card for Qwen3.5-122B-A10B reports MBPP 7.24 and HumanEval 8.70 (greedy,
  thinking on), a ratio of 0.83; ours is 0.57-0.62. MTP Head: that target is a sibling, not our 125B, so the checkpoint
  cannot read our taps and is not walkable; the ratio still points at training-data coverage, which fits DFlare's gain
  being data. Not pursued (the orchestrator). Test-Time Speculation (arXiv 2605.09329) adapts a DFlash drafter inside
  one request (LiveCodeBench Hard 2.4 -> 4.0 on that sibling); it pays on long sessions (the agent wall), not on MBPP's
  short answers.

## 6. Files and limits

- `kwall_replay.py` (census | split | ideas | tail), `kwall_depth.py`, `kwall_examples.py`; outputs `kwall_*.out`.
  Lines, with the time each was written: `K_WALL_lines_1106.md`. Desk read: `prior_art_1106.md` (a sub-agent's web read;
  its numbers carry their sources and were not re-checked by me).
- The replay cannot know what the 125B writes after drafter-written text. Convention B assumes the same code and that it
  ends a comment where the drafter does. Tonight's engine arm replaces that assumption.
- The candidate walk exists for HumanEval+ only, so R6 is not an MBPP number. Top-8 rows exist for 12 of the 24
  HumanEval+ answers.
- Ceilings use the drafter's own picks on the right prefix after a fixed row (the first-order selector re-walked).
- The 12 engine prompts and the walk's 12 fresh HumanEval+ answers are different tasks; only averages compare.

## 7. TRIED lines for the recorder (I write only here)

- CD16-MBPP12 (engine, BigMoE c966c0d): qall vs qall + R57_INERT=CD16 on 12 MBPP | wall: kept | exact: no | verdict:
  speed PASS (K x1.449, tok/s +23.7 %, tokens x1.02), quality KILL (tests 11 / 12 -> 8 / 12, both passes): the drafter's
  wholesale comment / docstring text runs away | 11-06 | BigMoE docs/WED_TESTS.md, data/fday.
- INERT wholesale arms G16 / D16 / C16: withdrawn without a run (each keeps one of the two failing kinds of text) | 11-06.
- INERT-T16 (guarded: top-3 at 0.1 p1 on comment + docstring rows, cut floor 0.8): BUILT, CPU self-tests pass; replay kept
  x1.08 HumanEval, x1.15 MBPP12; engine pending (Sat 00:30 ticket) | 11-06 | ~/FableGod/goal/build_1106/GUARD.md.
- K-WALL kills (CPU replays, `K_WALL_lines_1106.md`): ALIGN (comment text between anchor and row as the cause of the
  depth decay), a picker on deep code rows, PROMPT-NAME lookup, span-boundary rules, a looser cut under CD16; BETWEEN:
  ALPHA (fresh names, x1.02), REFRESH at D = 4; PASS as a ceiling only: an autoregressive tail past depth 4 (x1.24 MBPP /
  x1.12 HumanEval on top of CD16, priced as no gain at 1.35 ms per chain step) | 11-06.
