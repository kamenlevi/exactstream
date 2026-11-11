# NEVER-WAIT build: install, run, read (Fable FAR, Thu 5 Nov 2026 19:1x EET)

The ready-to-apply pieces of `~/FableGod/far/NEVERWAIT_TEST.md` v2, section 2 (the map arms) and 4 (the fallback).
Everything is in `~/FableGod/far/build_1105/`. Nothing outside `~/FableGod/far/` was edited; the installer below is
what touches the decision map, and it is run by the map's owner (Breakthrough) or the orchestrator.
Marks: [K] measured, [P] replay or arithmetic on [K], [G] judgement.

**State: built and checked on CPU; nothing has run on the GPU.** The patch equals an independent reference forward
to 3e-16 on a 48-layer model made of the real layer classes, and leaves every old arm bit-equal. What only the GPU can
show (the bf16 Metal kernels, the streamed 125B) is covered by the self-check in step 2, about 2 GPU minutes.

## 1. Install (CPU, about 15 s, no GPU, no model load)

```bash
bash ~/FableGod/far/build_1105/install_nw.sh
```

- Needs no map process running (`run_map.py`); exit 0 = in (or already in), 1 = failed, 2 = a map process is running.
  On 1 or 2 the map's `.py` files are unchanged.
- It checks the build folder against `MD5SUMS`, patches scratch copies of `dm_common.py` and `dm_ablations.py`
  (`nw_patch.py`: anchored edits, each anchor must occur exactly once), runs the CPU self-test on them (`test_nw.py`,
  44 checks), snapshots the masks into `~/FableGod/decision_map/out/far/<arm>/` with their md5s, loads every arm's 36
  masks through the patched loader (`check_install.py`), then moves the two files in atomically. The old files stay
  beside them as `dm_common.py.pre_nw` and `dm_ablations.py.pre_nw`. Scratch: `~/Breakthrough/data/abl_scratch/nw/`.
- The patch was made against the map as of Thu 19:00: `dm_common.py` md5 `78f789c9dd45f742581f9403205a4132`,
  `dm_ablations.py` md5 `dae4bc7a91b57fd4b22d59c7f2f95c5d` (it already holds Fable C's six arms of 18:56). If nothing
  else changes before the install, the installed files are `patched/dm_common.py` (`dabd653eaae8a0ea26567d859663279a`)
  and `patched/dm_ablations.py` (`c94b21e6c410bcc15c3d121499efbe54`). If other arms are appended first, the installer
  still applies: it patches the live files, not these copies. `run_map.py` is not changed.
- An arm without `_nw` is untouched: baseline, a topk arm and the mask twin come out of the patched harness bit for bit
  as from the unpatched one [K CPU, `test_nw.log` A]. So installing early costs nothing.
- Undo: copy the two `.pre_nw` files back (only if nothing else was installed since), or delete the `NEVER-WAIT` lines.

Dry run done here: the installer ran end to end against a scratch copy of the map (`scratch/dm_copy`): "installed",
then "already in" on a second run.

## 2. Self-check on the real model (GPU, about 2 min; before the arms)

```bash
bash ~/Breakthrough/scripts/dm_s7/gslot.sh nw_selfcheck 4 18 env BIGMOE_EXT_BUILD=build_nat ~/BigMoE/.venv/bin/python ~/FableGod/far/build_1105/nw_selfcheck.py
```

One map process, the two shortest code windows, five runs (the twin twice, `nwz`, `nw0`, `nw1`), scratch output in
`~/Breakthrough/data/nw_selfcheck/` (the map's `out/` is not written). Log: `~/Breakthrough/data/gslot/nw_selfcheck.log`;
the verdict is its last line and the file `~/Breakthrough/data/nw_selfcheck/VERDICT`.

| check | what | PASS | FAIL |
|---|---|---|---|
| (i) | `nwz`: the twin's masks through the never-wait path, late flags off | top-8 ids and log-probs equal the twin's exactly (if the twin's second run is not exactly its first, no farther from the twin than that) | anything else |
| (ii) | `nw0`: lateness 0, every late term written straight after its own layer's write | top-1 differs on <= max(2, 0.4 %) of rows and mean abs d log p1 <= 0.02 | > 3 % of rows or mean > 0.2; between: REPORT, the arms still run |
| (iii) | rows and flags, on `nwz`, `nw0`, `nw1` | no WARNING, every late flag used exactly once, late writes > 0 (0 for `nwz`) | anything else |

SELF-CHECK PASS or REPORT: run the arms. FAIL: do not; send me the log. Only (i) and (iii) are exact gates. (ii) has
little power on two windows [G]; the lateness-0 path is checked to 5e-16 on CPU.

No separate slot? Then keep the `nwz` line in the arm list (step 3): `nwz` and `nw0` run on the 12 code windows and
`nw_read.py` prints the same three checks before the verdicts. A failed check makes every arm line INVALID.

Keep `nw0` on the 12 windows in either case. It is also the control: it splits each late term off and writes it
straight after its own layer's write, so its D is the split write's bf16 re-rounding alone, and nw1 against nw0 is
lateness itself. In the CPU toy at bf16 that re-rounding is as large as the late terms' whole effect (2.7 % of the
stream; a random 48-layer toy is far more sensitive than the 125B) [K CPU], so the size on the real model is worth
its 2 minutes. The engine form has the same split write, so the lines stay on nw1 against the twin.

## 3. Arms

The lines are in `fri_extra_nw.txt` (format of `~/Breakthrough/data/fri_extra.txt`: arm, est min, start by, twins).
Append them to `fri_extra.txt` before `fri_c3.sh` starts its runner (Fri 00:30), or to `s67_extra.txt` while a runner
is up. The runner leaves an arm waiting until `install_nw.sh` has run. All on the 12 code windows (`+code`), about
2 min each [K Thu's `+code` arms].

| tier 1, in this order | twin | what |
|---|---|---|
| `qom_0_live_ball_c208+code` | (`qom_0_live_ball+code`, on record 0.51 %) | the twin: today's FAST code preset on 208 uniform slots |
| `qom_0_live_ball_c208_nwz+code` | `qom_0_live_ball_c208+code` | self-check (i); drop if step 2 passed |
| `qom_0_live_ball_c208_nw0+code` | `qom_0_live_ball_c208+code` | the control (and self-check ii): lateness 0, the split write alone; keep |
| **`qom_0_live_ball_c208_nw1+code`** | `qom_0_live_ball_c208+code` | decisive: every demand miss one layer late |
| `qom_0_live_ball_c208_nw1r1+code` | `qom_0_live_ball_c208+code` | fallback: rank-0 experts wait, the rest one layer late |
| `qom_0_live_ball_nw1+code` | `qom_0_live_ball+code` (on record) | nw1 at the published pool (low exposure) |
| `qom_0_split0.99_live_ball_nw1+code` | `qom_0_split0.99_live_ball+code` (on record 0.92 %) | the stack: split 0.99 + nw1 |

Five arms and the control, about 12 GPU minutes; 14 with `nwz`.

**Tier 1b, recommended if 4 more GPU minutes exist** (commented out in the file): `qom_0_live_ball_c266+code` and
`qom_0_live_ball_c266_nw1+code`, the same pair on 266 uniform slots. Fable ZERO's reading of the engine's log
(`~/FableGod/zero/LOSSY350_ZERO.md` 5d) puts the engine's by-day base pool at the replay's 266-290 slots: its
"C 208" run holds 226 base slots plus a 64-slot victim tier, and about 3.0 of its 6.6 reads per kept token are base
records [P]. So 208 slots is about twice the engine's late-term exposure and the published pool about 70 % of it; the
266-slot pair sits on it (3.54 misses per kept token, 5.36 late terms per row) and settles a "pass only at the low
pool" outcome the same night.

Tier 2 is built too and commented out in the file:
`_c208_nw2` (two layers late), `_c208_nw1s` (late only on rows whose draft cf >= 0.9), the stack at 208 slots
(`qom_0_split0.99_live_ball_c208` and its `_nw1`), `qom_0_split0.9_live_ball_nw1`, and chat = the tier-1 names without
`+code` (36 windows, about 6 min each). Not built: `nwh` (half a layer late); it needs a second write place in the hook.

## 4. Read-out and lines

```bash
~/BigMoE/.venv/bin/python -I -B ~/FableGod/far/build_1105/nw_read.py
```

CPU, seconds, reads the map's JSONs, writes nothing. It prints the self-checks (if their arms ran), each arm against
its own twin (flips, D in points, McNemar counts, D by the draft's cf band), the control (nw0 against the twin, nw1
against nw0) and `VERDICT` lines. Any other pair:
`nw_read.py --pair <arm>+code <twin>+code`. The runner's own `code_read.py` lines also appear, from the twin fields.

All-row top-1 flips against the baseline argmax, 12 code windows, 3,366 rows; T = the twin, D = arm - T in points.

| arm | PASS | BETWEEN | KILL |
|---|---|---|---|
| nw1 at 208 slots | <= 1.34 % and D <= +0.40 ("clean" if D <= +0.15) | flips in (1.34, 2.0] or D in (+0.40, +0.80]: nw1r1 decides | > 2.0 % or D > +0.80, and not clean at the published pool |
| nw1 at the published pool | clean = D <= +0.15 against 0.51 % | | |
| nw1 at 266 slots (tier 1b) | nw1's lines, against its own twin | as nw1 | as nw1 |
| pool-conditional PASS | nw1 not PASS at 208 slots but clean at the published pool: ships only where the engine's base-pool misses are <= ~3 per kept token (BigMoE's answer to the spec's 8.1) | | |
| nw1r1 (rank 0 waits) | <= 1.34 % and D <= +0.25; ships only if nw1 does not PASS | | D > +0.60 |
| the stack | <= 1.34 % | | above: the two levers do not ship together |

If the twin itself is over 1.34 % at 208 slots, that is reported first and every "+" is read against it.
A map PASS is a screen. Adoption is EvalSuite same-scores on code, and the speed needs the engine test of the spec's
section 3 (BigMoE's ticket; not part of this build).

Registered predictions [G/P, NEVERWAIT_TEST.md section 7]: twin at 208 slots 0.6-0.9 %; nw1 D +0.1 to +0.4, PASS 60 %;
nw1 at the published pool D +0.03 to +0.15, clean 80 %; nw1r1 D +0.05 to +0.2, PASS 80 %; the stack 0.95-1.15 %,
PASS 65 %.

## 5. What was built

**The hook** (`neverwait.patch`, 199 lines; `nw_patch.py` applies it; `patched/` holds the result):
- `_pert_scores`: the late flags are read at the row offset the keep flags are read at (the row counter advances once).
  A call with late terms returns scores x (keep and not late) and hands scores x (keep and late) to its caller.
- `_combine_p` (chunks of 32 rows and more) and the narrow path stash delta_L, the late ranks' weighted sum of this
  call's rows: layer L's experts, layer L's input, the model's own scores.
- The read wrapper keeps layer L's MLP-side inject gates, the read picked by identity.
- A post-wrap on `DecoderLayer.__call__` writes delta_(M-d) on the stream layer M returns, through layer (M-d)'s own
  gates. At the last layer nothing may be pending (assert). d = 1 for nw1, 2 for nw2, 0 for the self-check.
- `measure()`: loads `late` beside `keep`, prints a "never-wait" note per window, stores `res[window]["nw"]`, and
  warns if the late terms used differ from the mask's count. Never with `_t3`, a gate, reuse or topk (refused).

**The masks** (`masks/`, 13 folders x 36 windows; `nw_masks.py` = NewIdeas' `dm_oea_masks.py` plus the lines in
`nw_masks.diff`, generated by `mk_builder.py`). `keep` as the twin's but for the tail's drops, plus `late`
[rows, 48, 10], rank-sorted. Code, per realized row, before the tail rule [P `build_masks.log`]:

| folder | pool | demand misses per kept token | late terms per row (top-2 + members) | late MoE mass | rows with a late term | folder digest |
|---|---|---|---|---|---|---|
| `qom_0_live_ball_c208` (twin, keep only) | 208 | 6.70 | | | | `2a690a7e...` |
| `qom_0_live_ball_c208_nw1` | 208 | 6.70 | 10.07 (6.11 + 3.97) | 3.58 % | 93 % | `19d5684b...` |
| `qom_0_live_ball_c208_nw1r1` | 208 | 6.70 | 5.43 (3.19 + 2.24) | 1.61 % | 90 % | `32123298...` |
| `qom_0_live_ball_c208_nw0` (self-check) | 208 | 6.70 | 10.07 | 3.58 % | 93 % | `94fb72e9...` |
| `qom_0_live_ball_nw1` | published (`laysz45_256`) | 2.17 | 3.25 (1.97 + 1.28) | 1.14 % | 77 % | `61578850...` |
| `qom_0_live_ball_c266` (tier 1b twin) | 266 | 3.54 | | | | `f5e3a0e4...` |
| `qom_0_live_ball_c266_nw1` (tier 1b) | 266 | 3.54 | 5.36 (3.22 + 2.14) | 1.88 % | 87 % | `b0f47773...` |
| `qom_0_split0.99_live_ball_nw1` | published | 2.19 | 3.30 (1.99 + 1.31) | 1.15 % | 78 % | `38dce31e...` |
| `qom_0_live_ball_c208_nw2` (tier 2) | 208 | 6.70 | 10.07 | 3.58 % | 93 % | `5abe92f7...` |
| `qom_0_live_ball_c208_nw1s` (tier 2) | 208 | 6.70 | 8.45 (5.01 + 3.44) | 2.96 % | 78 % | `b676ec9c...` |
| `qom_0_split0.99_live_ball_c208` (tier 2 twin) | 208 | 6.72 | | | | `84cb5ae6...` |
| `qom_0_split0.99_live_ball_c208_nw1` (tier 2) | 208 | 6.72 | 10.18 (6.14 + 4.04) | 3.60 % | 93 % | `b66c2d8c...` |
| `qom_0_split0.9_live_ball_nw1` (tier 2) | published | 2.19 | 3.33 (2.00 + 1.33) | 1.16 % | 78 % | `e419d7de...` |

Every file's md5 is in `MD5SUMS` (516 files: the scripts, the patch, `base/`, `patched/`, the masks). The folder digest
is `md5 -r masks/<folder>/*.npz | md5 -q`. The twins at the published pool are the snapshots Breakthrough ran on Thu
(`out/ni/`); they are not shipped again. Chat at 208 slots: IFEval 3.80 late terms per row, 1.49 %; multi 2.81, 1.12 %.
The tail rule at layer 47 (nw1, code, 208 slots): 36 late rank-0 terms stay in place, 78 other late terms are dropped.

## 6. What was checked on CPU

`test_nw.py` (`test_nw.log`, 44 checks, 4 s, 230 MB, MLX's CPU device): a 48-layer float64 model built from the real
`Q.DecoderLayer.__call__`, `Q.GatedResidual`, `Q.SparseMoeBlock` and `Q._hyper_residual_write`, run through the
patched `install()` wrappers in chunks of 40 / 33 / 7 rows. Stand-ins: the mixer and the routed experts only.

| | result [K CPU] |
|---|---|
| old arms after the patch (baseline, topk 4, the mask twin) | bit-equal to the unpatched harness |
| self-check (i): late flags off | bit-equal to the twin |
| self-check (ii): lateness 0 | relative error 5e-16 against the twin |
| nw1, nw2 against a reference forward written from the spec's 2.4 | relative error 3e-16 |
| the same reference with layer L+1's gates / late rows shifted by one / another lateness | 2e-3 to 7e-3 away: 2 to 6 million times the match |
| every late flag used exactly once; late writes = (chunk, layer) pairs with a flag | yes; 88 stashes from the wide path, 27 from the narrow one |
| guards: a tail flag, a flag outside keep, topk, a term that reaches the last layer | all refused |
| the real `measure()` and `run_window()` end to end (fake prefill), two windows | `nwz` JSON equals the twin's; nw1's top-1 equals the reference run's; a prefill that skips rows trips both WARNINGs |
| the same test in bfloat16, the 125B's dtype (`test_nw_bf16.log`) | every exact check passes: the bit-equalities, the counts, the guards, `measure()`; the reference comparisons are notes there (8-bit mantissas over 48 layers allow no tolerance) |

`check_masks.py` (`check_masks.log`): the twins equal NewIdeas' own builder and Thursday's snapshots bit for bit;
each never-wait `keep` equals its twin's below the tail; `late` lies within `keep`, never at the tail, never on a
prompt row; nw1r1 has no late rank 0; the nw0 / nw1 / nw2 / nw1r1 / nw1s families are consistent. The counts match
Fable ZERO's independent replay (`~/FableGod/zero/late_terms.out`: 6.64 misses per kept, 9.98 late terms, 3.55 %).

Not checkable on CPU: the bf16 Metal combine kernel on the late path (the same call the map's reuse arms already make)
and the streamed model's chunking. The streamer evaluates the expert outputs before it returns them
(`bigmoe/stream.py`: "before the next group can evict these experts"), so a stashed term depends only on concrete
arrays. Step 2 covers the rest.

## 7. Files

| file | what |
|---|---|
| `install_nw.sh` | the installer (step 1) |
| `nw_patch.py`, `neverwait.patch`, `base/`, `patched/` | the hook: patcher, diff, the files it was made against, the result |
| `test_nw.py`, `test_nw.log`, `test_nw_bf16.log` | the CPU self-test, its last run in float64 and in bfloat16 |
| `check_install.py` | arms and masks through the patched loader (run by the installer) |
| `nw_masks.py`, `mk_builder.py`, `nw_masks.diff` | the mask builder, how it is generated, its diff against NewIdeas' builder |
| `masks/`, `build_masks.log`, `build_masks_tier2.log`, `build_masks_c266.log` | the masks, their stats (`*.stats.json`), the builder's output |
| `check_masks.py`, `check_masks.log`, `masks_orig/`, `masks_check/` | the mask checks and their reference builds |
| `nw_selfcheck.py` | the GPU self-check (step 2) |
| `fri_extra_nw.txt` | the arm lines (step 3) |
| `nw_read.py` | the read-out (step 4) |
| `MD5SUMS`, `mk_md5.sh` | md5 of every shipped file |
| `scratch/` | test outputs and the installer's dry run; not needed |

Rebuild the masks (CPU, 13 s per arm, 50 MB): `python -I -B nw_masks.py <out dir> <arm> ...` in the BigMoE venv.
A rebuilt `.npz` has another md5 (the zip stores the time); `check_masks.py <out dir>` compares the arrays.

## 8. TRIED line (for the recorder; I write only in `~/FableGod/far/`)

- NW-BUILD (Fable FAR; the build of NEVERWAIT_TEST.md v2; adapted, RE-OPENED: KTransformers Expert Deferral, TRIED 29 / 33 / 183 / 608 / 631): the never-wait map hook (a demand-miss term written d layers late through its own layer's inject gates), the mask builder, 13 mask folders at 208 / 266 / published pools, installer, GPU self-check, read-out | wall: SSD | line: 45 GB FAST | exact: no | verdict: BUILT, not run (CPU: the hook equals an independent reference forward to 3e-16 on a 48-layer toy of the real layer classes; old arms bit-equal to the unpatched harness; bf16 exact checks pass; twins equal NewIdeas' builder bit for bit) | code, late terms per realized row: 10.07 at 208 slots / 5.36 at 266 / 3.25 at the published pool (3.58 / 1.88 / 1.14 % of the MoE mass) | 2026-11-05 | ~/FableGod/far/build_1105/INSTALL.md

## 9. Status at stop (Thu 5 Nov, wind-down)

Done: everything in sections 1-8, CPU-checked (test_nw.log, test_nw_bf16.log, check_masks.log; installer dry run OK).
Not done: nothing installed in the real map; nothing run on the GPU (steps 1-4 are for Breakthrough / the orchestrator);
Fable ZERO's one review of `neverwait.patch` was requested 18:5x and had not arrived (if it comes, it is not folded in);
`nwh` and the spec's section 3 (BigMoE's speed ceiling) are not built.
