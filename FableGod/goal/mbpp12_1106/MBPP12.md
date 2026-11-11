# MBPP 12 for BigMoE's screen: CD16 where it is largest (GOAL, Fri 6 Nov 2026, 10:5x EET)

Asked by the orchestrator (10:5x): the 12 task ids, the prompt set, and tonight's predictions. Nothing here was written
outside `~/FableGod/goal/`; nothing ran on the GPU. Marks: [K] measured, [P] replay on stored walks, [G] judgement.

## 1. What to run

**No new prompt file is needed.** EvalSuite already has the set: `~/EvalSuite/data/items/dec_mbpp.jsonl`, 100 tasks,
keys `mbpp/<id>`, with `test_list` and `ref_code`. Its `prompt_ids` equal the prompts of MTP Head's MBPP walk for all 100
tasks (checked token by token), so the replay below predicts these exact prompts.

In `scripts/r57_screen_fast.py` (sha1 d30f6b3cdd30 when I read it):
1. `SETOF` (line 116): add `"mbpp": "mbpp"`, the way `agentlong` was added.
2. `R57_QOM_SETS=humaneval,ifeval,multi,mbpp`. The lossy preset is per prompt set (line 239), so without `mbpp` in the
   list these prompts run exact acceptance under `R57_FAST=QOM`.
3. The keys (also in `mbpp12_keys.txt`):

```
R57_KEYS=mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319
```

4. Cap 384 (`C45_CAP=384`). The walk's answers for these 12 are 62-282 tokens. A run cut at the cap is dropped by the
   read-out.
5. Arms, same palindrome as the HumanEval pair: control = qall; arm = qall + `R57_INERT=CD16` (+ `R57_INERT_LOG=<path>`).

Checked on CPU: the screen's own type rule returns "code" for all 100 MBPP prompts (the D7B=code drafter and the code
preset apply). Not checked: an MBPP run itself. `R57_NOCOM` only touches the humaneval set, so it does not apply here.

## 2. The 12, and how they were chosen

Rule, fixed before any per-task gain was looked at: the walk's answers of 60-320 tokens (49 of the 100: they finish
under a 384 cap and hold comment / docstring text), sorted by task id, 12 at even steps.

| key | answer tokens | comment + docstring share | exact K (checks) | CD16 K, A / B (checks) | tokens per task, CD16 / exact |
|---|---|---|---|---|---|
| mbpp/16 | 111 | 0.41 | 4.83 (23) | 8.17 / 9.00 (11) | 0.89 |
| mbpp/100 | 87 | 0.34 | 7.25 (12) | 12.29 / 12.29 (7) | 0.99 |
| mbpp/146 | 123 | 0.73 | 7.69 (16) | 12.50 / 12.50 (10) | 1.02 |
| mbpp/161 | 106 | 0.59 | 8.15 (13) | 11.78 / 11.78 (9) | 1.00 |
| mbpp/183 | 161 | 0.59 | 4.44 (36) | 10.31 / 11.33 (12) | 0.85 |
| mbpp/208 | 65 | 0.37 | 7.22 (9) | 10.00 / 10.00 (7) | 1.08 |
| mbpp/224 | 62 | 0.29 | 7.75 (8) | 8.00 / 8.00 (6) | 0.77 |
| mbpp/246 | 123 | 0.06 | 7.24 (17) | 7.29 / 7.29 (17) | 1.01 |
| mbpp/268 | 70 | 0.56 | 6.36 (11) | 10.00 / 10.00 (7) | 1.00 |
| mbpp/296 | 282 | 0.22 | 8.06 (35) | 9.40 / 9.40 (30) | 1.00 |
| mbpp/305 | 119 | 0.38 | 6.26 (19) | 8.50 / 8.50 (14) | 1.00 |
| mbpp/319 | 195 | 0.72 | 4.22 (46) | 10.36 / 10.85 (13) | 0.73 |

A / B are the replay's two conventions for what the 125B does after a drafter-written comment (A stops the check at the
span end, B lets it run on). Per-task numbers are in `mbpp12_replay.json`. A single task has 6-46 checks, so read the set,
not one row.

## 3. The replay's prediction for these 12 [P]

| | control (exact acceptance in the replay) | CD16 |
|---|---|---|
| kept per check | 6.13 | 9.64-9.83 (x1.57-1.60) |
| rows per check | 8.2 | 11.9 |
| checks for the 12 answers | 245 | 143 |
| tokens per task | 1 | 0.935 (range 0.73-1.08) |
| seconds per task (62 + 2.8 ms per row / 30 + 2.4 ms per row) | 1 | 0.65 / 0.69 |

The engine's control is qall, not exact acceptance; on HumanEval the two agree (10.36 engine against 10.44 replay).

## 4. Lines (written before any MBPP engine number exists)

- **PASS:** K (arm / control) >= 1.30 AND tokens per task <= 1.05 AND test verdicts equal on >= 11 of 12 with no net loss
  above 1.
- **KILL:** K ratio < 1.15, or tokens per task >= 1.15, or >= 2 tasks flip to fail.
- BETWEEN otherwise. On a PASS: EvalSuite decision 4 on MBPP. On a quality KILL: G16, then D16 (`build_1105/INSTALL.md`).
- The line moved once, at 10:5x, before any run: the first list said >= 1.40, set from all-MBPP's x1.8-1.9 halved. These
  12 must finish under the screen's cap, and answers that finish have a replay ratio of x1.57-1.60. Half of that gain is
  x1.29-1.30 (Kamen's rule: halve the replay's gain).
- Always: the arm's counters must show rows open > 0, and the log audit must show every relaxed accept inside a comment
  or a docstring.

Read-out (S, T, K, W and the code column work for `mbpp/` keys; verdicts come from EvalSuite's `test_list`):

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/goal/build_1105/inert_report.py <control>_runs.json <arm>_runs.json --log <inert log> --arm CD16
```

The report's own S line for CD16 (median S <= 0.90) was written for HumanEval; for this set use the K line above.

## 5. What this arm tests that the HumanEval arm cannot

- On HumanEval the docstring is a copy of the prompt; the drafter reproduces it. On MBPP the 125B writes its docstring
  and comments fresh, and they read like its plan. Under CD16 that text is the drafter's. Whether the 125B's code stays
  the same after it is the open question, and the code column answers it per task.
- Look first at the four tasks where the replay shortens the answer most: mbpp/319 (0.73), mbpp/224 (0.77), mbpp/183
  (0.85), mbpp/16 (0.89).

## 6. How to read an MBPP number: the set has three kinds of answers [P]

| the walk's answers | answers | share of tokens | comment + docstring share | exact K | CD16 K (A / B) |
|---|---|---|---|---|---|
| under 60 tokens: bare code | 35 | 7 % | 0.00 | 7.38 | 7.38 / 7.38 (x1.00) |
| 60-320 tokens (the pool of the 12) | 49 | 44 % | 0.38-0.47 | 6.16 | 10.04 / 10.24 (x1.63-1.66) |
| 321-499 tokens | 6 | 16 % | 0.63 | 5.00 | - / 11.45 |
| at the walk's token cap (513-519 tokens), never finished | 10 | 33 % | 0.88 | 5.02 | 13.23 / 13.31 (x2.63-2.65) |
| all that finish (under 500 tokens) | 90 | 67 % | | 5.94 | 9.84 / 10.06 (x1.66-1.69) |
| all 100 | 100 | 100 % | | 5.61 | 10.69 / 10.87 (x1.91-1.94) |

So "MBPP x1.9" leans on ten answers that ramble in a docstring or comments until the cap. For answers that finish the
replay says x1.66-1.69, and a third of the answers (bare code) gain nothing.

## 7. Tonight's HumanEval pair: what the replay expects (K_WALL_1106.md section 4)

qall against qall + CD16 on the engine's 12 HumanEval [P, from 12 other HumanEval+ answers of the same kind]:
- kept per check 10.4 -> 12.9-13.2 (the report's K column x1.24-1.26; with the gain halved x1.12);
- share of checks where every draft is kept 34 % -> 57-60 %;
- rows per check 11.9 -> 14.0-14.2 (so tok/s rises less than K: read seconds per task);
- tokens per task 0.97-0.98;
- after CD16, a check ends on a code token in ~21 % of checks and on the confidence cut in ~10 %.
The speed line for that pair stays as in `build_1105/INSTALL.md` section 5 (median S <= 0.90, T <= 1.05, verdicts equal
on >= 11 of 12).
