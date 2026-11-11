# INSTALL: Friday's code arms and the non-tie read-out (Fable C, Thu 5 Nov 2026, 18:3x EET)

Built for Breakthrough to install when it is back. Everything is in `~/FableGod/fablec/build_1105/`. CPU only, `python -I`,
one process at a time; nothing outside `~/FableGod/fablec/` was written, nothing is installed, no GPU run was made.
Background and the reasons for each arm: `~/FableGod/fablec/WALL_1105.md` (section 4).

## 1. What is here

| file | what it is |
|---|---|
| `masks/<arm>/<window>.npz` | the six arms' keep masks, 36 windows each, `keep` [rows, 48, 10] bool, rank-sorted: the format of `qom_0_split0.99_live_ball` |
| `masks/MANIFEST.md5`, `masks/MANIFEST.sha256` | one line per file (216 lines each) |
| `build_wall_masks.py` | the builder: a COPY of NewIdeas' `~/NewIdeas/scripts/dm_oea_masks.py` (md5 c6b88322a4b571c222a062a3519f68ac), every changed or added line marked `WALL` |
| `rebuild_check/` | the builder's rebuilds of `qom_0_split0.99_live_ball`, `qom_0_split0.9_live_ball`, `qom_0_live_ball` (for the bit-check) |
| `check_build.py` -> `check_build.out` | the bit-check, the structural checks, the tile table, the manifests |
| `code_read_nontie.patch`, `dm_screen_read_nontie.patch` | the non-tie read-out as patches (`patch -p1`) |
| `code_read_nontie.py`, `dm_screen_read_nontie.py` | the same two scripts already patched (they can be run as they are) |
| `install_fc.sh`, `test_fc_arms.py` | COPIES of Breakthrough's `install_c3.sh` / `test_c3_arms.py` (changed lines marked `FC`), needed only for arm B |
| `selftest/` | outputs of the self-tests in section 5 |
| `build_rebuild.log`, `build_arms.log` | the builder's own prints (unions, tiles, pool statistics) |

## 2. The arms

All six are `qom_0_split0.99_live_ball` (QOM at cf 0, the 45 GB pool `laysz45_256` C_m, live checks, DRAFT17's last row,
strict rows = cf >= 0.99: ranks 0-1 + the block's union members, no resident keep) plus one or two rules. Loads are unchanged
except in B. Tiles = the builder's own "tiles per kept token" divided by the twin `qom_0_live_ball`'s (291.4 / 490.1 / 477.0);
with this count the installed split 0.99 is x0.628 / 0.921 / 0.942 and split 0.9 x0.508 / 0.825 / 0.865.

| tag | arm = mask directory | rule added to the split | tiles vs the twin, code / IFEval / multi | load misses vs the twin | md5 of `cat <arm>/*.npz` |
|---|---|---|---|---|---|
| K8 | `qom_0_split0.99_k8_live_ball` | a non-strict gated row keeps ranks 0-1, union members, and its resident experts at ranks 2-7 only | x0.539 / 0.768 / 0.782 | x1.00 | 0d00a9cf4fde056f411cff5e782b56c7 |
| K6 | `qom_0_split0.99_k6_live_ball` | the same with resident ranks 2-5 only | x0.449 / 0.605 / 0.613 | x1.00 | 360fc0b6e236a5e24a5f78c4b1c1e37c |
| A | `qom_0_split0.99_r0.7_live_ball` | a drafted row with cf < 0.99 whose reach (the product of the cf of the drafts before it in its live segment; 1 for the anchor row) is < 0.7 is strict too | x0.491 / 0.646 / 0.710 | x1.00 | 3c669a3be9a1f4963f0a0d7e5396b362 |
| A' | `qom_0_split0.99_m0.9_live_ball` | a row with cf in [0.9, 0.99) is strict in layers 12-35 only | x0.572 / 0.876 / 0.903 | x1.04 / 1.00 / 1.00 | 499f0e36dd2318b839dbac0c51b702eb |
| B | `qom_0_split0.99_r0.7_z0.999_live_ball` | A, and a row whose real draft cf is >= 0.999 keeps NO routed expert in layers 8-35 and loads nothing there (2,215 rows) | x0.449 / 0.641 / 0.706 | x0.72 / 0.97 / 0.98 | 6958af37009952d9f41cbc2e31466f6f |
| K4 (probe, last) | `qom_0_split0.99_k4_live_ball` | resident ranks 2-3 only | x0.356 / 0.431 / 0.435 | x1.00 / 1.06 / 1.00 | c7748a84f5723806e520c6d342b9522c |

Verify the copies:
```bash
cd ~/FableGod/fablec/build_1105/masks && shasum -a 256 -c MANIFEST.sha256 | grep -c ': OK'
```
(216 expected.) One arm at a time:
```bash
cd ~/FableGod/fablec/build_1105/masks && cat qom_0_split0.99_k6_live_ball/*.npz | md5
```

Details that are easy to get wrong, as built:
- The cap (K arms) limits the RESIDENT keep only: a union member is kept at any rank, as in the split. A capped row touches
  the pool only for what it keeps.
- A's reach is dm_oea_masks' own REACH-CUT quantity (`np.r_[1.0, np.cumprod(dc)]` over the live segment's drafts); in the
  split form it does not change the loads (every drafted row already loads ranks 0-1), it only removes the resident keep.
- B's skip uses the row's REAL draft cf: a last row that `_ball` gates with the synthesized cf 1.0 (no next draft) is strict as
  in the split, but it is not skipped.
- Rows that no live segment realizes (prompt rows, rows outside the walk) keep all 10 in every arm.

## 3. The bit-check of the builder (required: 36 / 36)

`check_build.out`, section 1 [K]:
- `qom_0_split0.99_live_ball`: 36 windows rebuilt; 36 / 36 equal to the installed masks (`~/FableGod/decision_map/out/ni/`),
  36 / 36 equal to NewIdeas' source (`~/NewIdeas/data/dm_masks/`).
- `qom_0_split0.9_live_ball`: 36 / 36 and 36 / 36. `qom_0_live_ball` (the twin): 36 / 36 and 36 / 36.

Structure of the new arms (section 2 of the same file): 36 windows each; bool with the routing capture's shape; every
unrealized row untouched; ranks 0-1 kept on every row-layer (B: except its 2,215 skipped rows, which keep nothing in all of
layers 8-35 and ranks 0-1 elsewhere). An arm keeps 0.02-0.10 % of positions that the split does not: the pool's history
changes when rows touch fewer experts, so the arms are not strictly nested (K6 keeps 1,066 positions K8 does not).
To repeat both:
```bash
cd ~/FableGod/fablec/build_1105 && ~/BigMoE/.venv/bin/python -I build_wall_masks.py rebuild_check qom_0_split0.99_live_ball qom_0_split0.9_live_ball qom_0_live_ball
```
```bash
cd ~/FableGod/fablec/build_1105 && ~/BigMoE/.venv/bin/python -I check_build.py
```

## 4. Install and queue

**K8, K6, A, A' (and K4): Breakthrough's own installer, unmodified.** It snapshots each arm into
`~/FableGod/decision_map/out/fc/<arm>` with a sha256 manifest, appends `ABLATIONS["<arm>"] = q1_union("<snapshot>")` to a
scratch copy of `dm_ablations.py`, runs `test_c3_arms.py`, and moves it in with no map process running.
```bash
C3_SRC=~/FableGod/fablec/build_1105/masks C3_SNAP=~/FableGod/decision_map/out/fc TEST_R01=1 bash ~/Breakthrough/scripts/dm_s7/install_c3.sh qom_0_split0.99_k8_live_ball qom_0_split0.99_k6_live_ball qom_0_split0.99_r0.7_live_ball qom_0_split0.99_m0.9_live_ball qom_0_split0.99_k4_live_ball
```
Dry run of that test here, on a scratch copy of `dm_ablations.py` inside `selftest/` (the map's files untouched): 20 / 20
checks ok, ALL OK (`selftest/install_dryrun_c3test.txt`).

**B needs one exception.** `test_c3_arms.py` requires rank 0 on every row and B's skipped rows keep nothing in layers 8-35,
so the unmodified test refuses it at that line only (dry run: the other three checks ok). The harness itself multiplies the
routing weights by the mask with `renorm` False (`dm_common._pert_scores`), so a row-layer with nothing kept posts a zero
routed output, the same numbers as the gate path's k 0 in the measured `gate_top0_cf0999_L08_35`; that is read from the code
and not yet run. Two ways:
```bash
bash ~/FableGod/fablec/build_1105/install_fc.sh qom_0_split0.99_r0.7_z0.999_live_ball
```
(`install_fc.sh` is `install_c3.sh` with this folder as the default source, `out/fc` as the snapshot, and `test_fc_arms.py`,
which allows skipped rows only for an arm with `_z` in its name; dry run on all six: 24 / 24 ok, `selftest/install_dryrun_fctest.txt`),
or leave B out: the other arms do not depend on it.

**Queue.** The arms are code-decisive, so the cheap form is the code-only run, as today's splits: one line per arm in the s67
runner's extra file, `<arm>+code <est min> <start by> <primary twin> <info twins>`, ~4 min each:
```
qom_0_split0.99_k8_live_ball+code 4 <HH:MM> qom_0_live_ball+code qom_0_split0.99_live_ball+code qom_0_split0.9_live_ball+code
qom_0_split0.99_k6_live_ball+code 4 <HH:MM> qom_0_live_ball+code qom_0_split0.99_live_ball+code qom_0_split0.9_live_ball+code
qom_0_split0.99_r0.7_live_ball+code 4 <HH:MM> qom_0_live_ball+code qom_0_split0.99_live_ball+code qom_0_split0.9_live_ball+code
qom_0_split0.99_m0.9_live_ball+code 4 <HH:MM> qom_0_live_ball+code qom_0_split0.99_live_ball+code qom_0_split0.9_live_ball+code
qom_0_split0.99_r0.7_z0.999_live_ball+code 4 <HH:MM> qom_0_live_ball+code qom_0_split0.99_r0.7_live_ball+code
qom_0_split0.99_k4_live_ball+code 4 <HH:MM> qom_0_live_ball+code qom_0_split0.99_k6_live_ball+code
```
The twin and both splits were measured today on the converted store (`out/qom_0_live_ball+code.json` 0.51 %, split 0.99
0.92 %, split 0.9 1.31 %); they need a rerun only if the store or the build changed. The masks cover all 36 windows, so a
full run of any arm (without `+code`) is possible if chat is wanted; the chat windows are not decisive for these arms.

## 5. The non-tie read-out

Why: the 125B's tie rows (`~/FableGod/floor_rows.json`'s `cls`, p1 - p2 <= 0.1: 50 / 290 / 249 rows) flip 20-45 % under every
arm, so the all-row rate mixes real damage with coin flips. The patches add, to every arm line, its non-tie and tie flips,
repeat the McNemar on the non-tie rows, and add a second verdict; every existing line, the all-row VERDICT included, is
unchanged. Level in non-tie terms: `gate_top2_cf099`'s own non-tie flips, 0.87 / 2.22 / 1.63 % (the script recomputes it).

Apply (either or both):
```bash
cd ~/Breakthrough/scripts/dm_s7 && patch -p1 < ~/FableGod/fablec/build_1105/code_read_nontie.patch
```
```bash
cd ~/FableGod/decision_map && patch -p1 < ~/FableGod/fablec/build_1105/dm_screen_read_nontie.patch
```
or run the patched copy without installing anything (it only prints):
```bash
~/BigMoE/.venv/bin/python ~/FableGod/fablec/build_1105/code_read_nontie.py qom_0_split0.99_k6_live_ball+code qom_0_live_ball+code qom_0_split0.99_live_ball+code
```
`dm_screen_read.py` gains one optional variable, `SCREEN_OUT`, a directory for its `SCREEN_read_<arm>.txt` (default: `out/`, as
before); `FLOOR_ROWS` names another class file in both scripts.

Self-tests [K], all in `selftest/`:
- Both patches apply cleanly to copies of the originals and give exactly the patched files here.
- `code_read` on today's split results (`code_read_thu.txt`): twin 0.51 % (17 / 3366), non-tie 0.21 % (7 / 3316), tie 20 % (10 /
  50); split 0.99 0.92 % (31 / 3366), non-tie 0.51 % (17 / 3316), McNemar vs the twin 23 / 9 z +2.47 (non-tie 14 / 4 z +2.36);
  split 0.9 1.31 % (44 / 3366), non-tie 0.78 % (26 / 3316), 37 / 10 z +3.94 (non-tie 23 / 4 z +3.66). The all-row numbers and
  McNemars are SCOREBOARD's; VERDICT2 is PASS for all three.
- `dm_screen_read` on `qom_0.5_live_bonus5g qom_0.5_live 0.1 --null`: Breakthrough's stored read-out
  (`results_wed/SCREEN_read_qom_0.5_live_bonus5g.txt`) reappears line for line (10 / 10 lines in order) with 6 new lines; the
  map's own `out/SCREEN_read_*.txt` was not written.
- `dm_screen_read` on `qom_0.5_live_c147_bonus5g qom_0.5_live_bonus5g`: all-row 0.71 / 4.30 / 2.73 PASS, non-tie 0.33 / 0.93 /
  0.85 PASS, VERDICT2 PASS x3; the recomputed non-tie level prints 0.87 / 2.22 / 1.63.

## 6. The lines (written before any run)

For each arm, on the 12 code windows (3,366 rows, 3,316 non-tie), from `code_read_nontie.py`'s `VERDICT2`:
- **PASS: all-row flips <= 1.34 % AND non-tie flips <= 0.87 %.**
- **KILL: over either.** (BETWEEN, for the record: all-row <= 1.34 % with non-tie 0.87-1.0 %.)
- The code preset is the passing arm with the fewest tiles. Order of interest: K8, K6, A, A', B; K4 is a probe and runs last.
- Measured anchors in the same convention: twin 0.51 % / 0.21 %; split 0.99 0.92 % / 0.51 % at x0.628; split 0.9 1.31 % /
  0.78 % at x0.508.
- Predictions [G, sums of measured parts; no CPU method ranks these better than +- 0.2]: K8 and K6 non-tie ~0.7 (all-row
  ~1.15-1.2); A 0.6-0.9; A' 0.6-0.7; B = A + ~0.1; K4 0.9-1.1 (at or over the bar).
- What the result decides: if K6 passes, the code preset computes x0.45 of today's tiles at today's bytes; if only K8 and
  split 0.9 pass, the code frontier at the bar is x0.51-0.54 and the 2x is what was measured today; if A or B pass under
  K6's flips, the reach / skip shapes are cheaper than the cap and B also reads x0.72 of the bytes.

## 6b. State on Thursday evening (18:5x): what is already in place, and what is on hold

- **Done (CPU, no map file changed):** the six arms are snapshotted in `~/FableGod/decision_map/out/fc/<arm>` with a
  `<arm>.sha256` beside each, exactly as `install_c3.sh` step 1 does; the 216 sha256 equal `masks/MANIFEST.sha256`. The map's
  `dm_ablations.py` is NOT changed, so the arms are not registered there yet. `install_c3.sh` / `install_fc.sh` can be run as in
  section 4 all the same: they rename the existing snapshot aside, copy again, and register.
- **Ready, not run:** `run/run_1105.sh`, a runner on Breakthrough's gslot / s67 pattern that works from copies in `run/` (its own
  copy of `run_map.py`, of `scap_pass.py`, and of `dm_ablations.py` with the six arms registered against `out/fc`; `dm_common`
  and the rest resolve to the map's own files). Sequence: the capture smoke, a one-window control of the twin that must equal
  the twin on record row for row, then K8, K6, A, A', B, K4 on the code windows, each read at once with `code_read_run.py`
  (the non-tie read-out). Guards: one map process, a share lock "MEM 18 GB" per run, clear = no map process, no other
  session's quiet window or > 5 GB shared run, lid open, no PAUSE, NO-SWAP >= 24 GB; swap +1 GB ends the batch; last start
  21:15, hard stop 21:30 (both overridable with `RUN_LASTGO` / `RUN_DEADLINE`). Log and outputs: `run/run.log`, `run/out/`.
  Checked on CPU: module resolution, `<arm>+code`, the install test, the read-out and the control comparison.
- **ON HOLD:** the GPU part. The Fable sessions' standing rule is no GPU and it is Kamen's to lift; the runner now refuses to
  start without `run/KAMEN_YES`. It was armed from 18:45 to 18:47 on the orchestrator's first instruction, waited on the
  memory guard (NO-SWAP 20 GB < 24 GB) and was stopped before anything started: **no GPU run was made**.
- If the arms go to Breakthrough instead, nothing in `run/` is needed: section 4's install and queue lines are the path.

## 7. Checksums of the other deliverables (md5)
```
9c199fecaab9f3dd2fbea98abf578209  build_wall_masks.py
39444c4fd39a0f034b8074fe3ffc09b0  check_build.py
386dfd3071641b7b84f4320fd5cda279  code_read_nontie.patch
e73de8532cf2332e233e7c08668b0701  dm_screen_read_nontie.patch
bf413b5aaad4e1d04993c888d47304f2  code_read_nontie.py
f5e75a4351a2db743c304650e8833348  dm_screen_read_nontie.py
9f407f41063d3acf5ae084a35915ee8c  install_fc.sh
646856dac338e24a80f93f741ad60c7a  test_fc_arms.py
7b354a70446fccbbd5b5f7a3002d884e  masks/MANIFEST.md5
8217892a5bca905a03e3edf315dd5e22  masks/MANIFEST.sha256
```
