# R57_INERT=T16: the guarded form after CD16's MBPP quality KILL (GOAL, Fri 6 Nov 2026, 21:2x EET)

Asked by the orchestrator (Kamen wants >= 200 tok/s lossy by Saturday; CD16 failed MBPP12 on quality, 8 / 12 vs 11 / 12).
Nothing outside `~/FableGod/goal/` was written; no GPU was used. Marks: [K] measured, [P] replay, [G] judgement.

## 1. The short answer

- **G16, D16 and C16 are built but not safe. Do not run them.** Each keeps some text wholesale, and wholesale text is
  what failed.
- **T16 is the safe form and is ready:** a new `inert_span.py`, no script patch. Flag `R57_INERT=T16`.
- **It is small.** Replay: kept per check x1.08 on HumanEval, x1.15 on MBPP12 [P]; with the gain halved x1.04 / x1.08.
  It does not replace CD16's +21 %.

## 2. Why CD16 failed, from BigMoE's own logs [K, data/fday]

- The failure is runaway text. In every run that flipped, the drafter's free text leaves the 125B's distribution at its
  first or second deviating token (mbpp/16: rank 11-13, margin -8.5; mbpp/268 docstring: rank 60-71, margin -14.4).
- After that the 125B follows the babble with its own argmax until the cap: mbpp/16 had 356 open rows and only 78 that
  deviated. So a rule must stop the text at its first far-off token; a later guard is too late.
- The same thing shortened or changed answers that still passed (mbpp/146, 161, 305: 115-162 tokens -> 45-81).
- G16 keeps the first two comment lines of a block and all docstring text wholesale, and its later-line guard (top-8, any
  p) would pass 71-79 % of the tokens CD16 kept. D16 keeps all docstring text wholesale (mbpp/268 was a docstring loop).
  C16 keeps all comment text wholesale (mbpp/16, 296, 319).

## 3. What T16 does

- A row inside a comment or a docstring is kept only if the 125B itself ranks the draft in its top 3 with p >= 0.1 p1
  at that row. This is rule C's form, the one that passed decision 4 on comments, now on every comment and docstring row.
- Nothing is kept wholesale. Span ends, tool comments, doctest lines and all code tokens stay exact, as in CD16. When the
  125B's own token ends the comment or the docstring, the row is exact, so it can always end the span.
- Before the check, a guarded row's confidence is floored at 0.8 in the running-confidence cut, so a block is cut later
  inside a span. Rule C as shipped keeps the drafter's low confidence there; in this replay its comment rule alone gains
  x1.03-1.04 kept (MoonShots' own replay of it: x1.07 HumanEval / x1.13 MBPP, so read my T16 numbers as the low side).
- TC16 is the same on comments only. U16 (top-8 at 0.01 p1) is looser and is not the safe pick.

**Why it should avoid both failures:**
- Comment loops: read through T16's rule, the failed run's text is refused within its first 4 deviating tokens in all
  six flipped runs (at deviating token #0-#2), and T16 keeps 41-42 % of the tokens CD16 kept against the 125B.
- Code that drifts after the drafter's comments: every kept token is one the 125B gave at least a tenth of its top
  probability, so the comment is a wording the 125B itself could have chosen. This is the risk class of rule C, which
  EvalSuite passed on HumanEval+ [K]; it is not zero.

## 4. Install (BigMoE runs it)

```bash
cp ~/FableGod/goal/build_1106/inert_span.py ~/BigMoE/scripts/inert_span.py
```

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/goal/build_1106/test_inert.py
```

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/goal/build_1106/test_t16.py
```

- Module sha1 `0ca24e0d8db5`. The script needs no change: its accept edit already passes rank and margin for every row,
  and it touches the arm only through `INERT["rows"]`, `INERT["name"]` and the module's functions (read at
  `r57_screen_fast.py` d30f6b3cdd30).
- Self-tests on CPU: `test_inert.py` ALL OK, 38 checks (the Thursday arms behave as before, on the new module);
  `test_t16.py` ALL OK, 26 checks (the guarded arms; T16 opens exactly the 12,555 rows CD16 opens on the walks, so the
  Thursday safety check carries over; the failed run read through the rule).
- Not run: any GPU arm with T16.
- With `R57_COPY=g2` at 16 rows: compatible as written. The copy gate learns from the committed tokens
  (`cg.learn(d0_, ci_, dr[:k] + [am[k]], k)`), so a kept deviation enters its history as what was written.
- `R57_ROWS=32` cannot ride the lossy preset: line 246 asserts `R == 16` under `R57_FAST`.
- Not with `R57_RELAX_SPAN` / `R57_RELAX_DELTA` (asserted): in one arm it is T16 or relax v2.

## 5. Predicted K [P]

Replay on the walks with the engine's tagger. The first deviation of a block is judged with the 125B's real top-8; after a
kept deviation the 125B's view is not on disk, so later guarded rows are kept at the odds measured by depth on its own
text (comments 0.96 at depth 1 falling to 0.22 at depth 15 on MBPP; `guard_pi.py`).

| set | exact | T16 kept | rows per check | tok/s (62 + 2.8 ms per row / 51 + 1.1) | tokens per task | CD16, for scale |
|---|---|---|---|---|---|---|
| HumanEval+ fresh 12 | 10.44 | 11.25 (x1.08) | 11.9 -> 13.0 | x1.04 / x1.06 | 1.00 | 13.16 (x1.26) |
| MBPP12 | 6.13 | 7.06 (x1.15) | 8.2 -> 9.7 | x1.10 / x1.12 | 0.98 | 9.83 (x1.60) |
| MBPP, the 90 answers that finish | 5.94 | 7.03 (x1.18) | 8.1 -> 9.5 | x1.13 / x1.15 | 1.00 | 10.06 (x1.69) |

- With the gain halved (Kamen's rule): K x1.04 on HumanEval, x1.08 on MBPP12.
- TC16: x1.07 on HumanEval, x1.09 on MBPP12. The docstring part of T16 is worth nothing on HumanEval (its docstrings are
  copies; a mis-copy never passes the guard) and about half of T16's gain on MBPP.
- How the replay did on CD16: HumanEval x1.26 against the engine's k / kx x1.19-1.27; MBPP12 x1.60 against x1.449.

## 6. Lines, written before any T16 run

**MBPP12 quality (the gate), pass pairs as in the CD16 run (qall against qall + `R57_INERT=T16`, cap 384):**
- PASS: test verdicts equal to the control's on >= 11 of 12 in each pass pair, with no net loss above 1; AND no arm run
  ends at the cap where its control ended by itself; AND tokens per task <= 1.05.
- KILL: >= 2 tasks flip to fail in either pass pair; or any arm run at the cap where its control ended; or tokens per
  task >= 1.15.
- Void: the arm's counters show rows open = 0, or kept whole > 0, or the guard audit finds an accept outside the rule.
  (`build_1106/inert_report.py ... --arm T16 --log <log>` prints the audit.)

**MBPP12 speed (recorded, not the gate):** PASS K ratio >= 1.08; KILL < 1.03.

**HumanEval pair (stack against stack + T16):** PASS median S <= 0.975 and tokens per task <= 1.05 and verdicts equal on
>= 11 of 12; KILL S >= 1.00 or >= 2 flips. Expect K x1.04-1.08 and +1.1 rows per check (the replay's tok/s is +4-6 %,
halved +2-3 %).

## 7. MATCH and T16 in one arm

- On HumanEval the docstring is a copy of the prompt. MATCH's copy override removes those misses exactly; T16 gains
  nothing there. About half of CD16's HumanEval accepts were docstring rows (121 of 275; 118 of 213 [K]).
- T16's gain on HumanEval is in comments, which are not copies. So the two stack: MATCH x1.04 kept at 16 rows [K, walk]
  on top of T16's x1.08 [P] is about x1.12 kept, x1.06 with the gain halved.
- At 32 rows MATCH measured x1.20 kept on the walk, but the lossy preset is 16-row only today.

## 8. Limits

- No T16 arm has run. The odds after a kept deviation come from the 125B's own text, not from the drafter's path.
- T16 keeps the 125B's second or third choice in comment and docstring text. That is lossy by design; its quality
  evidence is rule C's pass on HumanEval+, and tonight's MBPP12 pair is its first test on fresh text.
- On HumanEval the size is +2-4 % tok/s. It will not close the gap to 200 by itself.
