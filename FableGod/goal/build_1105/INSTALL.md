# R57_INERT build (GOAL session, Thu 5 Nov 2026, 18:2x-18:5x EET): install, arms, lines

> **Superseded Fri 6 Nov 2026.** CD16 failed MBPP12 on quality in the engine (tests 11 / 12 -> 8 / 12; speed held): the
> drafter's wholesale comment / docstring text runs away. CD16, C16, G16 and D16 are withdrawn. The guarded arm T16 and
> its lines are in `build_1106/GUARD.md`; the account is in `K_WALL_1106.md` (update at the top).

> (a quote from the project owner, kept private)
engine's own tagger, and a ready-to-apply patch for BigMoE's relaxed screen. Design and replay numbers:
`~/FableGod/goal/LOSSY350_GOAL.md`. Nothing outside `~/FableGod/goal/` was edited; no GPU was used.
Marks: [K] measured, [P] replay / arithmetic, [G] judgement.

## 1. What is here

| file | what it is |
|---|---|
| `inert_span.py` | the module r57 imports: `InertScan` (the engine's `relax_core.CodeScan` + fence language, bracket depth, docstring-is-a-statement, doctest lines, comment-block index), `classify`, `pre_block`, `inert_k`, `copyrun_extend`, `Log` |
| `apply_inert.py` | 13 asserted string edits to `r57_screen_rlx.py` (MoonShots' `apply_span.py` form: every anchor once, or nothing is written); `--check` writes nothing |
| `test_inert.py` -> `test_inert.out` | the CPU self-test: **ALL OK, 38 checks** |
| `precheck_inert.py` -> `precheck_inert.out` | the INERT replay through the engine path: **PRE-CHECK PASS** |
| `inert_report.py` | reads one engine arm against its control (S, T, K, W, code-equality, the lines, the log audit) |
| `r57_inert.diff`, `r57_screen_rlx.patched.py` | the same patch as a unified diff and as a patched copy of BigMoE's script (reference only; written here, not in `~/BigMoE`) |

Written against `~/BigMoE/scripts/r57_screen_rlx.py` at fef5586, sha1 `df5d6efea608`.

## 2. What the CPU checks say

**Pre-check (task 1): PASS.** The same functions r57 will call, the engine's own tagger, BigMoE's tokenizer with every
token decoded alone, the scan fed the way r57 feeds it, on MTP Head's every-position DFlash2 walks [P]:

| arm, 16 rows | HumanEval+ fresh 12 | HumanEval+ all 24 | MBPP 100 |
|---|---|---|---|
| exact (0.3 cut): kept per check | 10.44 | 9.24 | 5.61 |
| **CD16** tok/s (kept; answer length) | **x1.155** (12.91; 0.969) | x1.269 (12.98; 0.976) | **x1.614** (10.69; 0.951) |
| the line: the lexer replay +- 0.03 | x1.17 -> PASS | | x1.63 -> PASS |
| C16 | x1.098 | x1.192 | x1.195 |
| G16 | x1.089 | x1.146 | x1.505 |
| D16 | x1.050 | x1.056 | x1.306 |
| CD16 with the doctest guard off | x1.163 | x1.273 | x1.614 |

tok/s under 62 ms per check + 2.8 ms per row. The engine's stricter conditions (the fence's third line, tool comments,
doctest lines, docstring-is-a-statement, python fences only) cost 0.01-0.02 against my lexer.

**Self-test: ALL OK.** Module cases; the patch applies once per anchor, compiles, every inserted block is behind
`INERT` / `iscan` / `CRUN`, the four replaced expressions equal the old ones for `R57_ROWS` 16 and 32; and the safety
property against Python's own parser: on every answer of the walks, **every token the tagger opens to wholesale acceptance
lies inside a COMMENT token (tokenize) or a string statement (ast): 3,231 of 3,231 on HumanEval+ 24, 5,647 of 5,647 on
MBPP 100, 0 outside** (22 / 24 and 94 / 100 answers parse; the open tokens cover 91-97 % of that text).

## 3. Install (BigMoE runs it)

```bash
cp ~/FableGod/goal/build_1105/inert_span.py ~/BigMoE/scripts/
```

```bash
~/BigMoE/.venv/bin/python ~/FableGod/goal/build_1105/apply_inert.py --check
```

```bash
~/BigMoE/.venv/bin/python ~/FableGod/goal/build_1105/apply_inert.py
```

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/goal/build_1105/test_inert.py
```

- A run from a snapshot (as `fri_night.sh` does) needs `inert_span.py` copied beside the snapshot, like `relax_span.py`:
  the screen imports it from its own folder.
- **Exactness first:** run the patched copy with every flag off against REF2-T (the existing `frx0` step). The patch adds
  no statement that runs with `R57_INERT` and `R57_COPYRUN` unset, except `INERT, CRUN = None, 0`, a no-op assert, three
  `None` assignments and an unused counter.
- Not with `R57_RELAX_SPAN` or `R57_RELAX_DELTA` (asserted), nor with a FAST arm whose comment loop is on (B, C).

## 4. The arms

Environment on top of the arm's own (`QA` = `fri_night.sh`'s qall string):

| arm | env added | what it does | runs on |
|---|---|---|---|
| **CD16** | `R57_INERT=CD16` | comments + docstrings kept wholesale, 16 rows | qall (R57_FAST=QOM, D17) |
| C16 | `R57_INERT=C16` | comments only | qall |
| G16 | `R57_INERT=G16` | docstrings + comment lines 1-2 of a block wholesale; later comment lines on the 125B's top-8 | qall |
| D16 | `R57_INERT=D16` | docstrings wholesale; comments on rule C (top-3 at 0.1 p1) | qall |
| CD32 / G32 / D32 | `R57_INERT=CD32,R57_ROWS=32` | the same, a block keeps its drafts past the 15th while they sit in a span | **the exact engine only** |
| COPYRUN64 | `R57_COPY=g2,R57_ROWS=64,R57_COPYRUN=64` | exact: a long verbatim copy run goes on past the drafter's 31 picks, up to 63 drafts | **the exact engine only** |

Optional: `R57_INERT_LOG=<path>` (one JSON line per relaxed accept), `R57_INERT_DOCTEST=0` (lifts the doctest-line guard).
In the runner's arm syntax: `C45_ARMS="fcd16_$CC:$CC:$QA,R57_INERT=CD16,R57_INERT_LOG=$PWD/data/fri/inert_cd16.jsonl"`.

**Two engine facts that decide what Friday can measure [K, the script itself]:**
1. `R57_FAST` asserts `R == 16` ("16-row blocks only", its row gate and `_fast_call` are written for <= 16 rows). So the
   32-row INERT arms cannot ride on qall today. On the exact engine a check over 16 rows leaves the sync-free gate
   ("slow either way", the script's own comment): CD32 / G32 / D32 give **rule numbers** there (kept per check, answer
   length, verdicts), not a tok/s claim. **The honest speed number Friday is CD16.**
2. The screen's prompt sets are humaneval, math, ifeval, multi, agent; **there is no MBPP set**. The engine arms run on the
   12 HumanEval keys; MBPP stays a replay number until the set exists.

The 32-row copy check (MATCH 2.5) is already in the engine: `R57_COPY=g2,R57_ROWS=32`. COPYRUN64 only lifts its ceiling
from 31 to 63 drafts. It has never run on a GPU: checks over 32 rows are new to the engine, so the token check decides.

## 5. The read-out and the lines (written before any run)

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/goal/build_1105/inert_report.py data/r57/fqall_224_runs.json data/r57/fcd16_224_runs.json --log data/fri/inert_cd16.jsonl
```

Per prompt: S = seconds per task, T = tokens per task, K = kept per check (arm / control), W = share of tokens kept by
INERT, and **code** = whether the answer's code is the same in both arms once comments and string statements are removed
(ast). "same" settles a task without running its tests; "differs" means the 125B wrote other code after the drafter's
text, and that task's tests decide.

Control = qall on the same patched copy, same C, A-B-A-B or the palindrome. On the 12 HumanEval:

| arm | PASS | KILL | replay [P] |
|---|---|---|---|
| **CD16** | median S <= 0.90 AND median T <= 1.05 AND verdicts equal on >= 11 of 12 with no net loss above 1 | S >= 0.97, or T >= 1.15, or >= 2 tasks flip to fail | S 0.84 (tok/s x1.155, T 0.969) |
| C16 | S <= 0.93, T <= 1.05, verdicts as above | S >= 0.98, or T >= 1.15, or >= 2 flips | S 0.88 |
| G16 | S <= 0.95, T <= 1.05, verdicts equal 12 of 12 | S >= 0.99, or >= 1 flip | S 0.91 |
| D16 | S <= 0.97, verdicts equal 12 of 12 (the quality control: docstrings only deviate freely) | S >= 1.00, or >= 1 flip | S 0.95 |
| COPYRUN64 (vs the exact default with `R57_COPY=g2,R57_ROWS=32`) | every token equal to the reference's AND S <= 0.95 | any token differs (void), or S > 1.00 | docstring rows x1.24 at 2.8 ms per row |
| CD32 / G32 / D32 on the exact engine | no speed line. Read K against the replay (CD32 kept 16.5-17.3), T <= 1.05, verdicts | T >= 1.15 or >= 2 flips | LOSSY350_GOAL.md 3.6 |

Always: the log audit must show every relaxed accept in a comment or a docstring (anything else voids the arm), and the
arm's counters must show `rows open` > 0 (else the tagger never fired: a fence or tokenizer problem, not a result).
BETWEEN = neither line. On a CD16 PASS: EvalSuite decision 4 on HumanEval+ 60. On a CD16 quality KILL: G16, then D16.

Suggested order (each <= 10 min): the flag-off check; qall vs qall + CD16; then G16 / C16 by what CD16 showed; D16 as the
control; COPYRUN64 and one 32-row arm on the exact engine if a slot is left.

## 6. What the patch changes in r57_screen_rlx.py (13 edits; `r57_inert.diff`)

1. `rows`: `R57_ROWS=64` allowed only with `R57_COPYRUN=64`.
2. `flags`: `INERT`, `CRUN`, the module import, the arm's asserts, relax_core when no FAST arm loaded it.
3. `dwin masks`, 6. `propose masks`: the drafter never drafts more than 31 positions (`min(R, 32) - 1`).
4. `block rows`: a 32-row INERT arm drafts 31 positions on every check.
5. `run state`: `iscan` (the tagger), counters, the log; the first generated token is fed.
7. `match rule`, 8. `copyrun`: MATCH 2.5 unchanged for 16 / 32 rows; a 32-row INERT arm hands the block's length to
   `pre_block`; `R57_COPYRUN=64` extends a full copy block.
9. `pre block` (before the row gate and the cut): rows in a span are sure in the running-confidence cut; a longer block
   keeps its drafts past the 15th only inside a span or a copy.
10. `accept` (after rule C's block, before the commit): `classify` + `inert_k`; counters; the log.
11. `tagger feed`: the committed tokens join the tagger's text. 12. `result`, 13. `runs record`: `inert` in the run's
    result and in `{TAG}_runs.json` (None when unset).

## 7. Limits, stated

- The 125B's code after drafter-written comments is the open question; nothing here tests it. The report's `code` column
  and the tests do.
- A program that reads its own docstrings (`__doc__`, `argparse(description=__doc__)`, a doctest runner with the guard
  lifted) makes docstring text functional. The tagger cannot see that.
- Only fences that say `python` / `py` / `python3` are open. An unlabeled fence or another language is always exact.
- `copyrun_extend` and the over-32-row path were exercised on CPU only (BigMoE's own `CopyGate`, a synthetic 63-draft run).
- The replay's convention for what the 125B says after a deviation is in `precheck_inert.py`'s header; the engine arm is
  what replaces it.

TRIED lines for the recorder (I write only here): INERT-PRECHECK (CD16 through the engine path: x1.155 fresh HumanEval+,
x1.614 MBPP; PASS within 0.03 of the lexer replay; 8,878 open tokens, 0 outside Python's comments / string statements) and
INERT-BUILD (patch + module + self-test ready for BigMoE; 16-row arms on qall, 32-row arms and COPYRUN64 exact-engine only).
