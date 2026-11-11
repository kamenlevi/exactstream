# COPYGATE: a precision gate for the session match override (FableGod copygate, Mon 2 Nov 2026, 18:03-18:3x EET)

Tags: **[M]** measured on stored data (exact replay or the drafter's stored blocks), **[S]** a simulation on a model that was checked against the real walks (section 2), **[A]** arithmetic on those, **[J]** judgement. CPU only; TrainLab's files were only read.

## 0. Verdict
- **Precision was never the problem where it counts.** At reached positions (every earlier draft in the block right, the only positions that decide acceptance), MATCH's own walks replay to a match at **57.5 % of positions, right 99.1 %** on HumanEval+ at 16 rows. The probe said 58.2 % / 99.2 %. MBPP is 4.3 % / 76.6 % and agent 24.0 % / 91.0 % [M]. The walk's reported 39.7 % / 93.6 % (and MBPP 32 %, agent 51 %) also count positions after the block's first miss. There the suffix carries the block's own wrong drafts, and those positions never change the kept count.
- **The probe was too optimistic about the gain, not about precision.** Measured per check at the base walk's anchors with the drafter's own blocks, the override is worth x1.082 on code at 16 rows. The walk measures checks saved and gets x1.038 [M]. Three things hold the gain down:
  - 27.6 % of code checks are already full.
  - A fix only extends a check into the rows that are left.
  - Just after its first miss, the drafter's own next pick is right only 20.5 % of the time (code; 13-21 % elsewhere) [M].
- **MBPP's loss.** At the positions that decide acceptance, the copy is wrong where the drafter was right 110 times and right where the drafter missed 35 times [M]. Two thirds of the harm comes from the answer copying itself (the 125B's reasoning, its test lists). The rest comes at the end of a restated prompt sentence, where the copy proposes the prompt's next line ("Your code should pass these tests", "assert ...") [M].
- **The chosen gate, COPYGATE**, has three rules:
  1. Copy only from the context before the answer.
  2. A per-session duel, fed by each check's own labels: keep copying while the copy wins at least 60 % of its disagreements with the drafter. It starts from a Beta(3, 1) prior, so it begins on.
  3. A short line-boundary copy (aligned run < 24) needs its own boundary duel of at least 60 %. It starts from Beta(1, 1), so it stays off until it has won once.

  The gate runs on the CPU, needs no new weights, and keeps the output exact (the check still decides).
- **Estimate** [S, A], given as raw simulation / shifted by that set's real-minus-simulated gap at G = all:

  | | 16 rows | 32 rows |
  |---|---|---|
  | HumanEval+ | x1.033 / x1.047 | x1.197 / x1.217 |
  | MBPP | x0.999 / x1.007 | x1.000 / x1.013 |
  | agent | x1.032 / x1.040 | x1.031 / x1.041 |

  The likely miss is agent's line (≥ x1.04); agent has 12 sessions and is noisy [J].
- **Walk for MTP Head:** `copygate/gate_walk.sh`, about 10 min under a shared lock, 6 arms on the same three sets. The pass and kill lines are in section 5. Prediction for G2: PASS 25 %, BETWEEN 60 %, KILL 15 %. Counting the pre-registered G1 fallback, P(some PASS) is about 30 % [J].

## 1. Where real precision is lost

### 1.1 Method [M]
`gate_sim.py` replays the walk's override teacher-forced at every position each real walk checked. The code is the `Matcher` of `~/TrainLab/sticky/d02b_walk_match.py`. Its own class, run verbatim, agrees with `copygate.py` on 642 random blocks, wrong in-block drafts included (0 differ).

At reached positions the walk's suffix is the true text, so the replay is exact there: every copy recomputed at an accepted position equals the true token (0 replay errors in 6 walks).

### 1.2 Reached positions vs all positions [M]
| walk | reached positions | with a match | right | the walk's figure (all positions) |
|---|---|---|---|---|
| HumanEval+ 16 | 6,088 | 57.5 % | 99.1 % | 39.7 % / 93.6 % |
| HumanEval+ 32 | 6,239 | 58.1 % | 99.1 % | 28.2 % / 86.2 % |
| MBPP 16 | 15,450 | 4.3 % | 76.6 % | 5.3 % / 32.1 % |
| MBPP 32 | 15,541 | 4.4 % | 76.8 % | 6.3 % / 17.1 % |
| agent 16 | 1,893 | 24.0 % | 91.0 % | 10.5 % / 51.3 % |
| agent 32 | 1,905 | 24.5 % | 91.2 % | 10.4 % / 29.0 % |

**(a) In-block drafts entering the suffix.** This causes the drop in the reported numbers, but only at unreached positions. After the block's first miss, the suffix holds a wrong token. Either nothing matches, so coverage falls. Or the copy keeps following its source past the point where the 125B left it, so precision falls. Its cost in kept per check is zero, because acceptance has already stopped there [M]. SAFARI H2's 99.7 % "125B sure inside ≥ 8 copies" and the 99.1 % here describe the same copies.

**(b) Short or ambiguous sources.** First-order effects at the base walk's anchors are exact. A *fix* is the copy right at the drafter's first miss. A *harm* is the copy wrong where the drafter was right (`gate_sim.out`, LOSS) [M]:
- HumanEval+: 3,598 reached matches, 137 fixes, 28 harms. 3,406 matches are k = 16 and 2,280 have an aligned run ≥ 64 tokens.
- MBPP: 674 matches, 35 fixes, 110 harms, 46 where both are wrong. **Every k bucket is a net harm** (k 8-9: 13 / 47; k 16: 3 / 14), so match length does not separate MBPP.
- agent: 467 matches, 57 fixes, 23 harms.
- Ambiguous k-grams (more than one continuation seen) are bad but rare: 1 / 5 / 14 matches, 0 fixes and 9 harms among them. Gating them moves nothing on MBPP (section 3).

**(c) Source and domain.** This is the separating feature [M]:

| source of the copy | HumanEval+ (fix / harm) | MBPP | agent |
|---|---|---|---|
| the prompt / context | 3,572 hits: 137 / 22 | 219: 14 / 35 | 375: 50 / 20 |
| the answer itself | 26: 0 / 6 | 455: 21 / 75 | 92: 7 / 3 |

Typical MBPP harms (`gate_sim.out`, EXAMPLES):
- `"...the sum of common divisors of two given numbers.\n"`: the copy says `Your`, the 125B says `   ` (it is writing a docstring).
- `"smartNumber(1) == 30"`: the copy says `\n` (the assert line ends), the 125B says `,`.
- `"273 = 3 * "`: the copy says `9`, the 125B says `7` (its own arithmetic, copied from earlier in the answer).

HumanEval+ restates the signature and docstring verbatim and keeps going. MBPP restates one sentence, then the prompt's scaffolding begins. Harm is concentrated: 33 of 100 MBPP sessions have any, and the top 10 hold 73 of the 110 [M].

### 1.3 Why the probe's gain was about 2x the walk's
- **Per check vs per trajectory** [M, A]. Measured at base anchors with HPD-0's stored blocks (HPD-0 = base16, checks identical), G = all is worth x1.082 on code and x1.030 on IFEval. The real walk gives x1.038 on code. A fix merges two checks, and kept per check counts checks saved, not tokens gained. Copies sit where the drafter already runs long, so a saved check there is a long check; per-check and per-depth profiles overstate the gain. On IFEval, 99-100 % of the hybrid trajectory's checks are base-anchor checks (real blocks), and the gain falls from x1.030 per check to x1.002.
- **After a fix** [M]. The drafter's own pick just after its first miss is right 20.5 % on code, 20.8 % math, 14.1 % IFEval, 12.9 % multi. Re-running the chain with the copy as predecessor (HPD-0's teacher-forced picks, exact at reached positions) adds only +0.0 to +0.2 points (hybrid). The drafter's later depths are lost after a miss, not just its chain, so a fix pays only where the copy keeps covering the next positions.
- **The 16-row cap** [M]. 27.6 % of code's base checks are full. At 32 rows the same copies give x1.203.

## 2. The models behind the estimates, and their checks
- **Token-hardness model** (the trajectory simulation for code, MBPP and agent at 16 and 32 rows). The drafter is right at a token iff the base walk with the same rows drafted it right. It is wrong at that walk's first misses. Tokens that walk never drafted (bonus tokens) are taken from the other row count's base walk, else counted right.
  - Checked per check on the real match walks [M]: model k = real k at 92.5 % (code 16), 91.1 % (code 32), 96.2 % (MBPP 16), 96.5 % (MBPP 32), 97.7 % (agent 16) and 98.7 % (agent 32) of checks.
  - The model is pessimistic. G = all simulates x1.023 / 1.183 / 0.972 / 0.972 / 1.037 / 1.048 against real 1.038 / 1.203 / 0.980 / 0.986 / 1.045 / 1.058.
  - "Corrected" = simulated + (real all − simulated all) per set and rows [A].
- **Hybrid** (16 rows; code, math, IFEval, multi). HPD-0's stored blocks (own picks, cf, teacher-forced picks) at base anchors, which carry 68 % of the gated checks on code and 92-100 % on the others. The model is used elsewhere [M at base anchors, S elsewhere].
- **Per-anchor exact** (cf gates). The drafter's real block and cf at every base anchor, anchors held fixed. Good for comparing gates, but it overstates gains (1.3).
- `copygate.py` (the walk's gate) reproduces the simulator's G0 / G1 / G2 / all to 4 decimals on all six set × rows cells [M].

## 3. Gates measured (selected; full table in `gate_sim.out`)
Trajectory simulation [S], raw / corrected [A], kept per check vs base:

| gate | HE+ 16 | HE+ 32 | MBPP 16 | MBPP 32 | agent 16 | agent 32 |
|---|---|---|---|---|---|---|
| real MATCH walk (all) | 1.038 | 1.203 | 0.980 | 0.986 | 1.045 | 1.058 |
| all | 1.023 / 1.038 | 1.183 / 1.203 | 0.972 / 0.980 | 0.972 / 0.986 | 1.037 / 1.045 | 1.048 / 1.058 |
| k ≥ 12 | 1.031 / 1.045 | 1.194 / 1.215 | 0.990 / 0.997 | 0.991 / 1.004 | **1.022** / 1.031 | 1.038 / 1.048 |
| k = 16 | 1.031 / 1.045 | 1.185 / 1.206 | 0.995 / 1.003 | 0.997 / 1.010 | **1.014** / 1.023 | 1.021 / 1.031 |
| aligned run ≥ 24 | 1.031 / 1.045 | 1.181 / 1.201 | 0.999 / 1.006 | 0.999 / 1.012 | **1.005** / 1.013 | 1.008 / 1.018 |
| unambiguous 8-gram | 1.023 / 1.038 | 1.178 / 1.199 | **0.974** / 0.982 | 0.975 / 0.988 | 1.041 / 1.049 | 1.055 / 1.065 |
| shadow precision tracker ≥ 0.93 (Beta 19:1) | 1.023 / 1.038 | 1.183 / 1.203 | 0.991 / 0.999 | 0.991 / 1.004 | **1.022** / 1.031 | 1.027 / 1.038 |
| check-label precision tracker ≥ 0.9 (Beta 9:1) | 1.023 / 1.038 | 1.183 / 1.203 | 0.991 / 0.999 | 0.991 / 1.004 | 1.027 / 1.035 | 1.034 / 1.045 |
| session duel alone (Beta 3:1, ≥ 0.6) | 1.023 / 1.038 | 1.183 / 1.203 | **0.984** / 0.992 | 0.984 / 0.997 | 1.040 / 1.049 | 1.044 / 1.055 |
| **G0** source in the prompt | 1.033 / 1.047 | 1.199 / 1.219 | 0.992 / 0.999 | 0.992 / 1.005 | 1.034 / 1.042 | 1.043 / 1.053 |
| **G1** G0 + session duel | 1.033 / 1.047 | 1.199 / 1.219 | 0.993 / 1.001 | 0.993 / 1.006 | 1.037 / 1.045 | 1.039 / 1.050 |
| G0 + boundary guard (no short line-boundary copy) | 1.033 / 1.047 | 1.197 / 1.217 | 0.998 / 1.006 | 1.000 / 1.013 | **1.022** / 1.030 | 1.029 / 1.040 |
| **G2 COPYGATE** (G1 + boundary duel) | 1.033 / 1.047 | 1.197 / 1.217 | **0.999 / 1.007** | 1.000 / 1.013 | 1.032 / 1.040 | 1.031 / 1.041 |

**The drafter's confidence (cf)** can be measured only where blocks are stored (per-anchor exact, 16 rows; MBPP and agent have none). Per anchor, code / math / IFEval [M]:

| gate | code | math | IFEval |
|---|---|---|---|
| all | 1.082 | 0.944 | 1.030 |
| cf < 0.99 | 1.067 | 0.985 | 1.026 |
| cf < 0.9 | 1.043 | 0.996 | 1.015 |
| copy when the session's shadow precision > cf | 1.062 | 0.995 | 1.022 |
| G0 | 1.086 | 0.990 | 1.009 |
| G0 + k ≥ 12 | 1.085 | 0.999 | 1.010 |

**Killed at the desk:**
- *cf gates.* 36 % of code's fixes come where the drafter was ≥ 99 % sure and wrong (cf < 0.99 keeps 331 of 517 fixes) [M]. On code the copy pays exactly where the drafter is confidently wrong, so cf cannot gate it. This fits K-HEAD's finding that cf is already the drafter's best self-signal.
- *k ≥ 12, k = 16, run ≥ 24-32, precision trackers alone.* MBPP is fixed, but agent loses a third to all of its gain [S].
- *Ambiguity gates.* They do nothing on MBPP [S].

**Hybrid at 16 rows, G2** (real blocks at base anchors) [M + S]:
- code: x1.034 (all: x1.026)
- math, a harm domain kept off by the type rule: x0.987 (all: x0.953)
- IFEval: x1.001 (all: x1.002)
- multi: x1.000

## 4. The gate (`copygate.py`, mode `copygate`)
Per session, on the host, around each check:
1. **Propose.** This is the MATCH lookup, unchanged: the table of k-grams (k 8-16) of the committed window, longest suffix ≥ 8 of window + this block's chosen tokens. Each hit also yields:
   - the source index;
   - the aligned run (how far back the current text equals the text before the source, capped at 256);
   - a boundary flag (the copy's token, or the source token it follows, holds a newline).
2. **Gate.** Use the copy only if all of these hold:
   - (i) its source token lies before the answer's start;
   - (ii) (W + 3) / (W + L + 4) ≥ 0.6, where W / L = checked positions in this session where only the copy / only the drafter was right;
   - (iii) the run is ≥ 24, or it is not a boundary copy, or the same duel restricted to short boundary copies, (Wb + 1) / (Wb + Lb + 2), is ≥ 0.6.
3. **Learn.** After the check, positions p ≤ k label the duel. The host holds both tokens and the committed truth. The duel keeps learning while the gate is closed.

Cost [J]: one hash lookup per position per k, as MATCH. A rolling hash in the engine is microseconds and a few MB for a 30K-token context. Exact: the check still decides every token.

## 5. Pre-registered real walk for MTP Head (written before any run, Mon 11-02 18:3x EET)
- **What:** `sh ~/FableGod/copygate/gate_walk.sh` inside `peer lock share`, the same guards as `turns/match_walk.sh` (NO-SWAP ≥ 13 GB, swap +2 GB stop).
  - It runs `~/FableGod/copygate/d02b_walk_gate.py`: `sticky/d02b_walk_match.py` plus `TL_WALK_GATE=all|prompt|pduel|copygate` (implies the override) and reached-position stats. The diff is `d02b_walk_gate.patch`, 80 lines; with no `TL_WALK_GATE` the copy is unchanged.
  - Same three sets: HumanEval+ 24, MBPP 100, agent 12. DFlash2 L1-20, official loop.
  - Arms: `gall16` (sanity), **`g2_16`, `g2_32`** (decision), `g1_16`, `g1_32`, `g0_16` (report).
  - Base arms are reused (`m_base16` / `m_base32`; the walk is deterministic).
  - About 10 min at MATCH's pace (5 arms in 7 min) [A]. Then `gate_read.py` writes `~/TrainLab/data/gate_read.txt`.
- **VOID** if `gall16` ≠ `m_match16`, bit for bit (every check), on any set.
- **LINE (G2):**
  - **PASS** if HumanEval+ kept(16) ≥ x1.035 **and** kept(32) ≥ x1.18 **and** MBPP kept(16) ≥ x1.00 **and** agent kept(16) ≥ x1.04.
  - **KILL** if HumanEval+ kept(16) < x1.03 **or** kept(32) < x1.15 **or** MBPP kept(16) < x0.99.
  - **BETWEEN** otherwise.
  - All ratios are kept per check vs the same-rows base walk, as in MATCH.
- **Fallback (pre-registered):** if G2 misses only agent's line, G1 (`pduel`) decides on the same four lines.
- **Beside (report only):**
  - G0 / G1 / G2 per set;
  - MBPP and agent at 32 rows;
  - the COPYGATE line per arm (reached positions with a match and its precision, the share used and its precision).
- **Prediction** (G2, raw / corrected): HumanEval+ x1.033 / x1.047 and x1.197 / x1.217; MBPP x0.999 / x1.007; agent x1.032 / x1.040. **PASS 25 %, BETWEEN 60 % (agent below x1.04 most likely), KILL 15 %** [J].
- **After a PASS** [J]: BigMoE's host hook between the drafter and the check (MATCH's hook) plus the duel update after each check. 32 rows inside long copies (MATCH 2.5, BETWEEN on its own) is the next walk on top. The gate leaves code's copies on, so 2.5 should read as in MATCH (x1.063 per check-ms) [J, untested].

## 6. TRIED check (`~/TRIED/TRIED.md`)
- **LK1** (spec125 prompt lookup, +1.2 / −0.1 %), **SUFFIX** (too few long matches in Kamen's past turns) and **M-05** (datastore + lookup on math, oracle +0.08 kept): killed for few, imprecise matches. **The reason applies** to MBPP (4.3 % of reached positions at 76.6 %), math and multi, which is why the gate exists: it closes the copy per session there. It **does not apply** to HumanEval+ (57.5 % at 99.1 %) or agent's context copies (prompt-sourced: 50 fixes / 20 harms).
- **D-06** (the 125B's heads as a copy pointer: 4.5 % of agent first misses; prompt lookup 10.6 %). Consistent: here copies fix 57 of agent's first misses in 637 checks [M]. The gate is a filter on the same signal, not a new pointer.
- **L1-10** (verify long copies in one wide pass, BETWEEN; 5.8-6.5 % of agentic output). A different mechanism (verification width). COPYGATE leaves long aligned copies (run ≥ 24) free of the boundary rule; no overlap.
- **MATCH** (TrainLab REPORT 'MATCH RESULT', BETWEEN on code, today; not yet in TRIED.md). This is the step its reading named ("a per-session precision tracker or k ≥ 12"). Both were measured here: each fixes MBPP but costs agent a third or more of its gain. The source rule plus the duel is what keeps agent.

## 7. Files (all in `~/FableGod/copygate/`)
- `gate_sim.py`: replay, models, gates. `~/BigMoE/.venv/bin/python -I gate_sim.py valid loss ex gates cf hyb` takes about 40 s on the CPU; its output is `gate_sim.out`.
- `copygate.py`: the gate (pure Python).
- `d02b_walk_gate.py` and `d02b_walk_gate.patch`: the walk.
- `gate_walk.sh`: the run.
- `gate_read.py`: the verdict. Dry-run on the ungated walks it prints x1.038 / x1.203 / x0.980 / x1.045 → KILL (MBPP), and the sanity check passes 3 of 3.
- Inputs (read-only):
  - `~/TrainLab/data/d02b_walk_m_{base16,base32,match16,match32}_{code,mbpp,agent}.npz`
  - `~/TrainLab/data/d02b_walk_hpd0_{code,math,ifeval,multi}.npz`
  - `~/BigMoE/models/bare/vocab.json`

Weekly meter at the end: 84-85 % (stop line 93 %). Nothing left undone in scope.

## 8. RESULT: BETWEEN, agent short (MTP Head ran it 19:14-19:21 shared; ~/TrainLab/data/gate_read.txt; TrainLab REPORT 'COPYGATE RESULT', 9d9ae1e; read here 19:2x)
- **Sanity passes** [M]: `gall16` equals `m_match16` bit for bit on 3 of 3 sets. The walk's own reached-position counts for 'all' (57.5 % / 99.1 %, 4.3 % / 76.6 %, 24.0 % / 91.0 %) equal this session's CPU replay (section 1.2).
- **Kept per check vs base** [M], with this file's prediction (raw / corrected) in brackets:

  | arm | HumanEval+ | MBPP | agent |
  |---|---|---|---|
  | G2 copygate, 16 rows | **x1.041** [1.033 / 1.047] | **x1.000** [0.999 / 1.007] | **x1.032** [1.032 / 1.040] |
  | G2 copygate, 32 rows | x1.211 [1.197 / 1.217] | x1.001 [1.000 / 1.013] | x1.036 [1.031 / 1.041] |
  | G1 pduel, 16 rows | x1.041 [1.033 / 1.047] | x0.995 [0.993 / 1.001] | x1.044 [1.037 / 1.045] |
  | G1 pduel, 32 rows | x1.211 [1.199 / 1.219] | x0.997 [0.993 / 1.006] | x1.048 [1.039 / 1.050] |
  | G0 prompt, 16 rows | x1.041 [1.033 / 1.047] | x0.994 [0.992 / 0.999] | x1.041 [1.034 / 1.042] |

  All 15 numbers fall inside the predicted raw-to-corrected ranges. Agent's G2 equals the raw simulation exactly.
- **LINE:** x1.041 / x1.211 / x1.000 / x1.032 → **BETWEEN**: agent kept(16) is below x1.04, the other three lines pass. The fallback G1 misses MBPP (x0.995), so it is BETWEEN too. The prediction was BETWEEN 60 %, with agent the likely miss.
- **Reading** [J]:
  - No single arm passes all four lines. The MBPP / agent trade-off that the simulation showed is real: the rules that stop MBPP's loss (the boundary duel) also close some of agent's right line-boundary copies.
  - G2 is the only arm that loses nowhere: every set ≥ x1.000 at both row counts. It keeps code's gain (x1.041 / x1.211, up from MATCH's x1.038 / x1.203) and removes MBPP's loss (x0.980 → x1.000). On agent it gives up 1.3 points of the ungated x1.045.
  - Agent's walk set is 12 windows of about 160 answer tokens. A boundary duel that starts closed has little time to open there; a real agent session carries the duel across turns. That argues the walk understates G2 on agent, but it is untested.
  - Wiring G2 is BigMoE's call via the orchestrator; MTP Head's reading says the override is now "safe to wire everywhere".
