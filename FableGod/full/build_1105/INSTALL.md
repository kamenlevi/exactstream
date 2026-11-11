# build_1105: Fable FULL's two tests, ready to apply (Thu 5 Nov 2026, 18:5x EET)

For BigMoE. Built on CPU against BigMoE `6adad93` (`scripts/r57_screen.py` md5 `9a633d1c…cacda`, `scripts/fri_night.sh` md5
`5a92d239…e687f`, `scripts/c45_runner.sh` unchanged). Nothing in `~/BigMoE` was edited: every patch here writes a COPY.
Background and arithmetic: `~/FableGod/full/LOSSY350_FULL.md` (sections 1, 2, 5, 8).

- **T-A1, the copy probe**: what a wide strict copy pass costs. Rolled-back timing passes over the copied rows of the 12
  HumanEval answers (the signature + docstring every answer re-emits: ~60 % of the measured tokens).
- **T-B1, the comment-free prompt arm**: qall with one sentence added to the user turn ("Do not write comments in the
  code."), A-P against qall, with the comment-token counter.

## 1. Files

| file | what it is |
|---|---|
| `apply_screen.py` | writes a copy of `r57_screen.py` with two flags, both off by default: `R57_NOCOM=1`, `R57_COPYPROBE=g+x+s` (19 lines added, 4 anchors) |
| `copyprobe.py` | the probe module the patched screen loads by path when `R57_COPYPROBE` is set |
| `copyprobe_report.py` | reads `{TAG}_copyprobe.jsonl`, prints the table and T-A1's verdict |
| `apply_fri.py` | writes a copy of `fri_night.sh` with the P arms in the palindrome, the probe arm after `fexr`, and the two reports (16 lines added, 5 anchors) |
| `full_arms.sh` | the two tests on their own (A-P-P-A + the probe), for a ticket that is not `fri_night.sh` |
| `nocom_report.py` | T-B1's read and verdict from the arms' `runs / checks / tokens` files |
| `nocom_tests.py` | HumanEval+ tests of the A and P answers with EvalSuite's scorer (EvalSuite's venv) |
| `dryrun_copyprobe.py`, `dryrun_nocom.py` | the CPU dry-runs (no mlx, no model) |
| `out/r57_screen.patch`, `out/fri_night.patch` | the same changes as `git apply`-able diffs (`git apply --check` passes at 6adad93), for review |
| `out/r57_screen_full.py`, `out/fri_night_full.sh` | the patched copies as generated here, for review |
| `out/dryrun_*.log`, `out/dry_copyprobe_report.txt` | the dry-runs' output |

## 2. Install

Check first (CPU, seconds, writes nothing):

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/full/build_1105/apply_screen.py ~/BigMoE/scripts/r57_screen.py /dev/null --check
```

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/full/build_1105/apply_fri.py ~/BigMoE/scripts/fri_night.sh /dev/null --check
```

Both must print `OK`. If an anchor moved they say which, and nothing is written.

**Option A, inside Friday's ticket (the orchestrator's placement).** Generate the patched script and point the ticket at it:

```bash
cd ~/BigMoE && .venv/bin/python -I ~/FableGod/full/build_1105/apply_fri.py scripts/fri_night.sh scripts/fri_night_full.sh && bash -n scripts/fri_night_full.sh
```

Then run `scripts/fri_night_full.sh` where the ticket runs `scripts/fri_night.sh` (same usage line, same env). At run time
the script itself patches its frozen screen snapshot into `data/r57/snap/r57_screen_frix.py`; `scripts/r57_screen.py` is
not touched. Arm order (new arms in capitals; they run on the patched snapshot, every other arm on the frozen screen as
before):

`frx0 | u3c qall QALLP s99 s90 qallr span | span qallr s90 s99 QALLP qall u3c | fexr | FCPROBE [FCPROBET]`

- Time: each P arm is one qall arm (~105 s on Thursday's C 208 run); the probe arm is one qall arm plus at most 240 s of
  probing. Raise `FRI_DEADLINE` by ~5 min for the two P arms, or the mirrored tail (`fqallb`, `fu3cb`) can be cut. The probe
  arm has its own 8-minute deadline after `fexr` (`FRI_DEADLINE3`).
- Switches: `FRI_FULL=0` skips all added arms; `FRI_CPROBE_T=1` adds the experimental `fcprobet` arm (section 4);
  `FRI_B1105=<dir>` if this folder moves.
- If the applier fails at run time (an anchor moved after this build), the added arms are skipped by `arm()`'s own
  "no switch" rule and the rest of the ticket is as it was.

**Option B, on their own** (if Friday's ticket has no slack), inside a quiet GPU turn exactly as `fri_night.sh`:

```bash
NOTBEFORE=00:32 NOTAFTER=09:00 WAITMEM=40 GAP0=0 GAP=0 MARGIN=4 bash ~/BigMoE/scripts/turn45.sh 37 30 "BigMoE: FULL's T-B1 + T-A1" /bin/bash ~/FableGod/full/build_1105/full_arms.sh
```

Arms `fqall fqallp fqallpb fqallb fcprobe`, ~12 min; `FULL_TB1=0` or `FULL_TA1=0` runs one test only; `FULL_CPROBE_T=1`
adds `fcprobet`. Its qall env list is fri_night.sh's at 6adad93; pass `FULL_QA=...` if qall changed since.

**Recommended pre-flight** (one prompt, like Thursday's): the patched snapshot with `R57_KEYS=humaneval/HumanEval/91`,
CAP 96, C 160, arms `<qall env>,R57_QOM_SPLIT=0.9,R57_COPYPROBE=g+x+s,R57_COPYPROBE_MAX=1` and `<qall env>,R57_NOCOM=1`.
Expect in `data/r57/<tag>_copyprobe.jsonl`: one `point` line and its `pass` lines with `"err": null`, `"wc": 48` on the
`s` passes and `"wc": 0` on `g` / `x`; and no `#` comment in the NOCOM arm's decoded answer.

**After the run (CPU):** the tests, then the report again so the verdict includes them (from `~/BigMoE`; `<C>` = the run's C):

```bash
~/EvalSuite/.venv/bin/python ~/FableGod/full/build_1105/nocom_tests.py <C>
```

```bash
cd ~/BigMoE && .venv/bin/python -I ~/FableGod/full/build_1105/nocom_report.py <C> data/fri/nocom_report.txt
```

`nocom_tests.py` runs the model's generated code inside evalplus's checker, as EvalSuite's reports do; it writes
`~/BigMoE/data/fri/nocom_tests.json` (use `--out` / `--dir` for Option B's `data/full`).

Uninstall: delete `scripts/fri_night_full.sh` and `data/r57/snap/r57_screen_frix.py`. Nothing else was created.

## 3. T-B1: the comment-free prompt arm

- **What runs:** `R57_NOCOM=1` on the humaneval set inserts the 9 tokens of `"\n\nDo not write comments in the code."`
  before the user turn's `<|im_end|>`; everything else is qall. The stored REF2 row is not compared (another prompt).
- **Read:** `data/fri/nocom_report.txt`. Per prompt, pair means over the two passes of each arm, as fri_night's report:
  comment tokens (CodeScan on the decoded answers), kept per check over all checks and over the checks that start outside
  the copy run, steady tok/s, tokens and seconds per task, and the share of checks that keep only comment tokens.
- **Lines (pre-registered in LOSSY350_FULL.md section 5, for the CAP 256 window):**
  - PASS: comment tokens <= 2 % of P's tokens AND kept per check outside the copy run >= x1.15 AND tok/s >= +8 % AND no
    task that passes in A fails in P (tasks that ended before the cap in both arms).
  - KILL: kept outside the copy run < x1.05, OR tok/s < +3 %, OR >= 2 new failures.
  - BETWEEN: anything else. With comment tokens > 2 % the sentence was ignored and the mask form decides.
  - Prediction [P]: kept outside the copy run x1.2, tok/s +10-13 %, tokens per task -7-8 % inside the cap; comment-only
    checks 15-20 % of qall's checks.
- A PASS here is the kept gain only. Adoption still needs Kamen's yes (the answers carry no inline comments) and decision 4.

## 4. T-A1: the copy probe

- **What runs:** the arm is qall + the 0.9 split switch + `R57_COPYPROBE=g+x+s`. At up to 3 non-overlapping points per
  prompt where the last 8 committed tokens occur in the prompt with >= 31 prompt tokens after them, the module times these
  passes over `[tok + the copied tokens]`, each twice (as the pools are, then warm), **every one rolled back** (the
  warm-split's own pattern), then the real check runs as always:
  - `g4 / g8 / g16`: the sync-free gate path as the arm runs it; copied rows are sure rows at cf 1.0 (strict under the
    split), the last row exact. No drafter call. Their warm ms against rows = the fast path's cost per strict copied row.
  - `x32 / x64`: wide exact passes. Above 16 rows a call leaves the gate for EvalSuite's block and `tier.prefill_call`
    (exact 4-bit, every rank), as `R57_ROWS=32` did on Tuesday.
  - `s32 / s64`: wide strict passes on that same path: ranks 0-1 + the block's members, the other ranks routed to the row's
    rank-0 expert at weight 0 (no new read). `copyprobe.py` wraps the MoE block for these calls only.
  - `t32 / t64` (only in `fcprobet`): top-2 only as a narrow `[W, 2]` call, which fits the sync-free gate (W x 2 ids <=
    MAXK 160) and the tier call. Never run at HEAD. A Python error drops it and the run goes on; a hard abort would end
    that arm only (it is the last arm, its own process, and every finished pass is already on disk).
- **The arm's own tok/s is not a result** (the probe warms the pools); the screen's clocks are shifted by the probe's time.
- **Read:** `data/fri/copyprobe_report.txt` (`copyprobe_report.py data/r57/fcprobe_<C>_copyprobe.jsonl`).
- **Lines (written before any run; per copied token = pass ms / rows):**
  - confirmed = the share of copied rows the exact wide passes (`x`) confirm (argmax == the copied token); needs >= 0.90.
  - DIRECT = the best wide strict pass that really ran strict (`s` or `t`, `wc` > 0) and itself confirms >= 0.90 of its
    copied rows (a wrong kernel is fast too), first-touch median.
  - SLOPE = the fast path extended by its own row cost: intercept + W x slope from the warm `g` passes, plus W x the
    extra a first-touch `g16` pass costs per row.
  - PASS: confirmed >= 0.90 AND (DIRECT or SLOPE) <= 3.0 ms per copied token at W 32 or <= 1.8 at W 64.
  - KILL: confirmed < 0.80, OR every one of DIRECT and SLOPE at both widths >= 4.5 ms per copied token.
  - BETWEEN: anything else.
  - Prediction [P]: SLOPE 2.1-2.6 (W 32) and 1.4-1.9 (W 64); DIRECT by `s` above the line (it runs the prefill kernels);
    today's 16-row gate pass on the same rows ~5-6.5.
- **What this build cannot give:** the tok/s clause of LOSSY350_FULL's T-A1 (>= +15 % on the 12 prompts) needs a LIVE wide
  copy check. At HEAD the FAST gate is 16 rows (`assert FAST in FASTARMS and R == 16`; MAXK 160), so that arm needs Leap's
  wide gate first. A "PASS by SLOPE only" says exactly this: the rows are cheap, the wide fast path is still to be built.
  If `t64` runs and is fast, a live top-2 copy check is a small step from it.

## 5. What the CPU dry-run checked, and what it could not

Checked (`out/dryrun_copyprobe.log`, `out/dryrun_nocom.log`; rerun with the two `dryrun_*.py`):
- both appliers on HEAD's files: anchors unique, the patched screen compiles, the patched script passes `bash -n`, and
  the diffs pass `git apply --check`;
- the block wrapper's math (`s` and `t`) against an independent numpy reference at 32 and 64 rows; every call of 16 rows or
  fewer, or outside a probe pass, reaches the screen's own block;
- the hook in a fake engine loop on the 12 prompts: 24 probe points (1-3 per prompt), 396 passes, every pass rolled back,
  the gate and the wrapper switched off after each, the run's tokens untouched; an engine that fails inside `t64`'s graph
  build is undone and the run goes on;
- **on real data** (the 125B's stored REF2-T answers): the copied rows a pass would confirm are 1.000 at 16 rows, 0.990
  at 32 and 0.965 at 64 (the misses are the last tokens before the prompt's closing fence);
- the 9-token splice on the 12 prompts; `nocom_report.py` and `copyprobe_report.py` on synthetic files; `nocom_tests.py`
  under EvalSuite's venv on two finished tasks (2 of 2 pass, both arms);
- the patched `fri_night.sh` and `full_arms.sh` in a scratch root with a stub runner: arm order, the skip rules
  (`FRI_FULL=0`, a missing applier, no split switch), the report calls.

Not checked (needs the GPU; the pre-flight above covers the first three):
- that the patched screen runs inside the engine process (the module's import, `CS.rollback_after_verify` after 32- and
  64-row passes, EvalSuite's block at 64 rows in decode mode);
- the route-mode strict call on the sync path (`s`) and the narrow gate call (`t`);
- the sentence's effect on the answers; every timing. The ms in `out/dry_copyprobe_report.txt` come from a synthetic
  cost model and are not measurements.

Known limits, stated once:
- `s32 / s64` run the prefill kernels on exact 4-bit pairs, so DIRECT by `s` is an upper bound on a real wide strict kernel.
- SLOPE extrapolates the gate path past its 16-row limit; it prices the rows, not a path that exists.
- The probe points sit where at least 31 copied tokens remain, i.e. inside the docstring run; the run's first 8 tokens and
  its tail are not probed.
