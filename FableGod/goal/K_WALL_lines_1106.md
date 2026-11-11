# K wall replays: lines written BEFORE each run (GOAL session, Fri 6 Nov 2026)

Data for every replay: MTP Head's every-position DFlash2 walks (`~/TrainLab/data/d02b_walk_d23h_dfl_{mbpp,code}.npz`),
Breakthrough's top-8 rows (`~/Breakthrough/data/rulec`), MTP Head's candidate walk (`d02b_walk_kw_code.npz`).
Base = CD16 under convention B (the 125B's code after a drafter-written comment is the code it wrote after its own), 0.3 cut.
Script: `kwall_replay.py`. CPU, python -I, one process.

## Written 10:38, before the runs R2-R6

**R2 ALIGN.** Question: is the block drafter's decay on code rows caused by the comment / docstring text that sits between
the anchor and the row? Measure: teacher-forced accuracy on code tokens where the 125B's p1 >= 0.99, depths 8-15, with 0
inert tokens between anchor and row vs >= 5.
- PASS (inert text in between is the cause; a drafter that does not have to place code behind text of unknown length is
  the lever): the gap is >= 0.10 on MBPP.
- KILL: gap < 0.03 (the decay is depth alone).

**R3 REFRESH ceiling.** K under CD16-B if every code row deeper than D were picked with the accuracy the same drafter has
at depth D on that same position (oracle: the every-position walk's teacher-forced pick from the anchor D tokens back).
The ceiling of any mechanism that gives the drafter fresh 125B features every D rows (an autoregressive tail is D = 1).
- PASS (worth a GPU probe of a refresh mechanism): K(D = 4) >= x1.15 on MBPP and >= x1.05 on HumanEval+ fresh.
- KILL: K(D = 1) < x1.08 on MBPP (even an autoregressive tail with the 125B's own features would not move K).

**R5a PROMPT-NAME (exact).** Identifiers that occur in the prompt (MBPP's function name in the tests) completed by lookup.
Ceiling: an oracle on the stops whose token belongs to an identifier present in the prompt.
- PASS >= x1.03 on MBPP 94; KILL < x1.015.

**R5b ALPHA (lossy, inert).** The drafter names fresh parameters / locals; the 125B goes on with that name. Ceiling: an
oracle on the stops at a new identifier.
- PASS >= x1.03 on MBPP 94; KILL < x1.015.

**R6 CANDIDATES (HumanEval+ only, the candidate walk).** At code rows of depth >= 8 where the teacher-forced pick is wrong:
where is the 125B's token among the 16 candidates?
- PASS for a picker: rank 2-3 in >= 70 % of those misses. KILL: absent or rank > 3 in >= 50 %.

R4 (no line, a pricing table): K under CD16-B if code rows were right with probability r at every depth (r = 0.90-0.99).

## Written 10:43, appended in the same command as the run, seconds before R7-R8 started

**R7 AR-TAIL ceiling.** Rows deeper than D0 get the pick a fresh anchor one token back would make (an autoregressive tail
with the 125B's own features: the upper bound of the MTP chain used only past depth D0); rows up to D0 stay DFlash2's.
- PASS (worth one GPU walk that stores the trained MTP chain's picks at every position): D0 = 4 gives >= x1.12 on MBPP 94.
- KILL: < x1.06.

**R8 ALPHA, the realistic subset.** As R5b, but only stops where the 125B starts a NEW identifier and the draft's token is
also name-like (not a keyword, not an existing name's first token).
- PASS >= x1.03 on MBPP 94; KILL < x1.015.

## Written 10:5x, before any MBPP engine run: the line for BigMoE's 12-task MBPP arm (`mbpp12_1106/MBPP12.md`)

The first list (10:50) set PASS at a K ratio >= 1.40, from all-MBPP's replay x1.8-1.9 halved. The 12 tasks that will run
(a fixed rule: walk answers of 60-320 tokens, sorted by task id, 12 at even steps) have a replay ratio of x1.57-1.60, since
the all-MBPP figure leans on ten answers that never finish. Kamen's rule (halve the replay's gain) on those 12 gives:
- PASS: K (arm / control) >= 1.30 AND tokens per task <= 1.05 AND verdicts equal on >= 11 of 12.
- KILL: K ratio < 1.15, or tokens per task >= 1.15, or >= 2 tasks flip to fail.
