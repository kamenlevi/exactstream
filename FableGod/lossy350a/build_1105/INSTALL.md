# T1 build: input-sparsity arms for the decision map (Fable A, Thu 5 Nov 2026, 18:4x EET)

For whoever holds the GPU slot. Source of the test: `~/FableGod/LOSSY350_A.md` section 1, T1. Nothing here has run on the
GPU; the patch is not applied. Everything in this folder was built and tested on the CPU.

## What it is
- Arms `xs48`, `xs40`, `xs32` (and `xs64`, the identity check): in every MoE block the routed experts read x with all but
  the c largest-|x| members of each 64-wide input group zeroed. The block's router, its shared expert and the
  shared-expert gate keep the full x. It applies to every row that passes a MoE block, like `top_k8` and `topk5`.
- Off unless `DM_XS=1`: without it the harness imports nothing new, registers no xs arm, and runs as today.
- They stack with the harness's `+`: `qom_0_live_ball+xs48`, `xs40+code`.

## Why the change is not at dm_common.py line 310
The map prefills each window in 512-row chunks, so nearly every row takes the >= 32-row path, which never reaches line
310. That path hands x to the block's `switch_mlp` through `sorted_experts(x, inds)` or, in BigMoE's streamed class,
through its own `__call__`. A mask at line 310 alone would cover only the narrow tail chunks and report a near-zero
effect. The patch wraps those two entry points on the switch's runtime class, once, so every path is covered and the
model's own code runs unchanged behind the wrapper. Line 310 goes through the same wrapper.

## Files
| file | what |
|---|---|
| `t1_xs.patch` | unified diff: `dm_common.py` (+12 lines, 3 hunks), `dm_ablations.py` (+8 lines) |
| `dm_xs.py` | the new module: `mask(x, c)`, the wrapper, the row counter |
| `apply.sh`, `revert.sh` | apply with an md5 guard and `*.pre_xs` backups; undo |
| `read_xs.py` | reads the new arms by row class, in `fablec/wall_class_flips.py`'s convention, with the lines below |
| `predict.py`, `PREDICT.out` | the predictions, computed from the map's measured neighbour arms |
| `selftest_xs.py`, `selftest/`, `SELFTEST.out` | the CPU self-test and its last result (68 checks, all PASS) |
| `base/`, `applied/` | today's two harness files, and the three files as they are after the patch |

## Apply
Cut against `dm_common.py` md5 `78f789c9dd45f742581f9403205a4132` and `dm_ablations.py` md5
`e5c608ed0b572ad6e2f419e03da5b48f` (the live files at 18:24 today). `apply.sh` refuses anything else and refuses while
a `run_map.py` is running.
```bash
sh ~/FableGod/lossy350a/build_1105/apply.sh
```
After it, the three files have md5 `eb64579813dd7aa83f9e18241471bb1b` (dm_common.py), `4e792bf1de636751c12687cf3017e0f6`
(dm_ablations.py) and `fa38fd9c4c0dc9510789ef695626b446` (dm_xs.py); the originals sit beside them as `*.pre_xs`.
Undo:
```bash
sh ~/FableGod/lossy350a/build_1105/revert.sh
```
**If HEAD moved:** try `cd ~/FableGod/decision_map && patch -p1 --dry-run -i ~/FableGod/lossy350a/build_1105/t1_xs.patch`.
The four hunks are independent; by hand they are: (1) after the `PC` import, `_XS = None` and `import dm_xs as _XS` under
`DM_XS=1`; (2) in `moe_call`, after `CUR["L"] = li`, `if _XS is not None: _XS.wrap(self.switch_mlp)`; (3) in `measure`,
the row count around `run_window` (`x0`, the `xs` record, the WARNING); (4) at the end of `dm_ablations.py`, the four arms.
Then copy `dm_xs.py` beside them.

## Run
BigMoE's venv, from the map's folder, under the usual share lock (`map_slot.sh` uses `peer lock share 40m --cpu 200 --gpu 50`,
MEM 17 GB). A code-only arm took ~2.5 min this morning; a 36-window arm is ~3x that [P].
All three domains, the four arms (about 30 min):
```bash
cd ~/FableGod/decision_map && DM_XS=1 ~/BigMoE/.venv/bin/python run_map.py xs64 xs48 xs40 xs32 --max 12 --skip-done
```
Shorter first pass, code only (about 10 min), then chat for the mild arm:
```bash
cd ~/FableGod/decision_map && DM_XS=1 DM_SETS=code ~/BigMoE/.venv/bin/python run_map.py xs64+code xs48+code xs40+code xs32+code --max 12 --skip-done
```
Stacked on today's twin, code (about 8 min; the frame the lever would ship in):
```bash
cd ~/FableGod/decision_map && DM_XS=1 DM_SETS=code ~/BigMoE/.venv/bin/python run_map.py qom_0_live_ball+xs48+code qom_0_live_ball+xs40+code qom_0_live_ball+xs32+code --max 12 --skip-done
```
Without `DM_XS=1` an xs name stops the runner before the model loads (unknown name).

## Check in the log before trusting an arm
1. Every window line of an xs arm ends with `xs<c>: <n> masked switch rows` and n = 48 x (ids - 1) or 48 x ids. A
   `WARNING xs masked ...` means some path bypassed the mask (for example a fused one-row decode path): the arm is void.
2. `xs64` shows 0 flips on every row (as `topk10` does). If not, stop: the wrapper is not inert.
3. The gate and baseline are unchanged by the patch: no need to re-run them.

## Read
```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/lossy350a/build_1105/read_xs.py xs64 xs48 xs40 xs32
```
Give it the names as run (`xs48+code`, `qom_0_live_ball+xs48+code`). It prints all-row, non-tie, tie, sure and mid flips
per domain, the parent arm and the ADDED flips for a stacked arm, and the verdict lines below. It reproduces
`fablec/wall_class_flips.out` on the arms already on disk (self-test 5).

## Predictions, written before any run [P on K]
From the map's own uniform arms, read by row class (`predict.py` -> `PREDICT.out`). Non-tie flips are close to linear in
the routed block's relative error energy: slope per unit, code / IFEval / multi: 4.6 / 49.5 / 43.3 (every expert at MSB-3,
`t3_all - t4_all`), 6.6 / 59.3 / 48.2 (`top_k8`), 7.3 / 54.3 / 58.4 (`topk5`). An xs arm's energy is its measured
whole-expert error (`lossy350a/arms_plain.out`, real weights, held-out routed rows).

| arm | error x MSB-3 | predicted non-tie flips %, code / IFEval / multi | standalone all-row, with the tie floor |
|---|---|---|---|
| xs48 | 0.17 | 0.05-0.07 / 0.55-0.66 / 0.48-0.54 | 0.3-0.6 / 2.5-4.7 / 1.8-3.5 |
| xs40 | 0.56 | 0.17-0.24 / 1.8-2.2 / 1.6-1.8 | 0.4-0.8 / 3.6-6.1 / 2.8-4.6 |
| xs32 | 1.38 | 0.42-0.60 / 4.5-5.4 / 3.9-4.4 | 0.7-1.2 / 6.1-9.0 / 5.0-7.0 |
| room above today's preset, non-tie | | 0.66 / 1.03 / 0.72 (bars 0.87 / 2.22 / 1.63 minus 0.21 / 1.19 / 0.91 [K WALL_1105]) | |
- Standalone arms carry the tie floor in their all-row number (tie rows are 1.5 / 8.5 / 7.0 % of rows and flip 18-48 %
  under any arm), so they are read on non-tie rows.
- Stacked on the twin the added all-row flips should be close to the non-tie increments; SWEEP v1.1's KL model gives
  +0.11 / +0.30 / +0.61 on code for xs48 / xs40 / xs32 (LOSSY350_A.md section 1).
- Expected outcome: all three fit code; only xs48 fits chat, thinly on multi; xs40 and xs32 are over both chat bars.

## PASS / KILL lines (pre-registered)
- **Identity:** `xs64` has 0 flips. Otherwise nothing else counts.
- **Standalone arm, FITS:** non-tie flips within the room: <= 0.66 % code, <= 1.03 % IFEval, <= 0.72 % multi.
- **Stacked arm, PASS** (WALL_1105's convention): all-row <= 1.34 / 5.42 / 3.69 % AND non-tie <= 0.87 / 2.22 / 1.63 %.
- **Lever dead** (sparsity error is not priced like the errors the predictions rest on): standalone `xs32` non-tie code
  flips above 1.03 % (`topk5`'s own cost, at 1.6x this arm's error), or standalone `xs48` above 0.21 % (`top_k8`'s own
  cost, at 3x this arm's error); stacked: `xs32` adds more than +1.2 all-row code points, or `xs48` more than +0.25.
- **What a pass buys** [P]: xs48 / xs40 / xs32 leave 17 / 25 / 33 % of the expert weights untouched on gate and up
  (x-only arms); with the down half the lever is 0.59-0.51 of the weights, 5-7 ms of the 22-27 ms expert term on code.
  A build (store re-layout, two kernels) only if code holds beyond what the split form already uses.

## The self-test (CPU, about a minute)
```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/lossy350a/build_1105/selftest_xs.py
```
MLX runs on its CPU device; no model is loaded. Last result in `SELFTEST.out`: 68 checks, all PASS.
1. The mask equals an independent numpy definition on random values, exact ties, zero groups and real map inputs (three
   windows of `out/xcap`, 48 layers), for bfloat16 / float16 / float32; c = 64 returns the same array object. On real
   inputs it keeps 48.04 / 40.06 / 32.08 members per group (ties) and 99.2 / 97.3 / 93.4 % of the energy.
2. The harness's own code, today's and the patched, driven with a small fake model through both MoE paths, in three
   class layouts (BigMoE's streamed subclass, the fused base class, both mixed):
   - `DM_XS` unset: patched == today's, bit for bit; no xs arm exists;
   - `DM_XS=1`: baseline, a rank cut and `xs64` == today's, bit for bit; under `xs48` / `xs32` the switch reads exactly
     the masked x while the router, the shared expert and its gate read x; outputs equal a reference whose switch masks
     its own input, bit for bit; each row is masked and counted once; `clear()` switches the arm off.
3. `measure()` records 48 x rows per window and raises its WARNING when one layer's experts escape the mask.
4. The patch applies to today's files with no fuzz, `apply.sh` gives exactly `applied/`, refuses a second run, and
   `revert.sh` restores the originals. The live files still had the recorded md5 at 18:42.
5. `read_xs.py` reproduces `fablec/wall_class_flips.out`, and its verdict paths run on stand-in files.

What the self-test cannot cover: the GPU device (same MLX ops: abs, sort, compare, where) and the real model's class
names. The two log checks above cover both at run time.

## Limits
- `*_t3` arms compute their record-served pairs in `dm_t3`, outside the switch: do not stack xs on them.
- BigMoE's mid-token look-ahead routers read the switch's x, so in a streamed decode they would see the masked x. That
  affects prefetch only, and the map's prefill path does not use them.
- Inputs only: the down-side mask (the largest units of each 64) needs the switch's three projections called
  separately; it is the second screen, after this one reads.
