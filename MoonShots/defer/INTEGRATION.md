# dh_w4 fast mode on BigMoE's v3 engine: integration notes

**RESULT 11:36 (defer/RESULTS.md): NOT recommended at 24 GB on v3.** It passes quality and the build is bit-exact, but
it gives no speed on top of the deployed JIT prefetch (code 34.8 -> 34.9 tok/s, prose 30.4 -> 30.3, edit 44.6 -> 45.6).
Deferral and JIT prefetch use the same SSD windows: without JIT, dh gives +23 %. Keep these notes for a
no-prefetch setting (the small-RAM B12/B8 line).

MoonShots, 2026-10-22. Authorized by NewIdeas on Kamen's 07:40 decision: an **opt-in, non-exact** fast mode. The exact
mode stays unchanged, and making it the default is Kamen's call. BigMoE integrates on its side. Everything here lives
in ~/MoonShots/defer; nothing in ~/BigMoE or ~/Breakthrough was edited.

## What it does (the gated semantics, "dh_w4")
Per decode/verify row of every layer but the last:
- The router's top-10 by score are used, with v3's tiers and weights: plan 4433333333, weight p / sum(top-10 p).
- Ranks 0-3 are read and waited for, as today.
- A rank 4-9 expert that is resident is used now.
- A cold rank 4-9 expert is read **after** the GPU is released (phase A). Its gate-weighted output is computed on
  layer L's MoE input and written into the hyper-connection streams with layer L's inject weights after layer L+1's
  attention, before layer L+1's MLP read ("half a layer late").
- The last layer and the MTP layer wait for everything, as v3.

Quality (Breakthrough's real v3f decoder, the Eval suite's step scorer, C_m 128, every 8th sequence):
- v3 control: gsm 0.00298, code 0.00129, ifeval 0.01763, multi 0.00989.
- v3 + dh_w4: 0.00334, 0.00101, 0.01807, 0.01108.
- All four pass the 1.10 x ar4 targets. IFEval is tight: paired upper bound 0.02071 vs 0.0204.
- Denser IFEval re-check (every 4): PENDING (defer/tf_dense.py).

## Pieces to port (diff against BigMoE 7b8c7b4 = defer/v3d/UPSTREAM)
1. **ext, from Breakthrough's v3f** (its diff against 7b8c7b4, unchanged here):
   - Flag-mode routing rows: GPU bit 20 = read and wait, bit 21 = late-eligible. `post_ids(..., wk)` packs
     `nw = 255 | nk << 8`.
   - The post kernel's fast path requires only the flagged ids to be resident.
   - Loader: in flag mode with `BIGMOE_DEFER > 0`, it releases phase A (waited + resident, late ids = -1), reads the
     late ids, then publishes phase B in a second region buffer (`set_regions_b`). `spin_wait(..., phase=1)` reads it.
   - `ensure(..., protect)`, so phase A's resident ids are not evicted by the late reads.
   - `defer_stats`.
2. **ext, MoonShots v3d** (`BIGMOE_DEFER_FULL=1`): phase B publishes every position's slot (n ints, like a normal
   release) instead of (position, slot) pairs.
   - Needed for verify rows: S = 8 x 6 late-eligible ranks = 48 positions, more than the 46 pairs a 3-line region holds.
   - It also makes the GPU side trivial: phase B is just a second slot vector.
   - Trace kind 5 = the phase-A release time.
3. **ext, optional** (`-DDEFER_PIPE=1`, `BIGMOE_DEFER_PIPE_US=d`): the late reads are issued d µs after the waited
   ones, instead of after they land (`ensure_reserve` + `load_jobs_async`). This saves one SSD request latency per
   layer that has both kinds of read. Value: PENDING (ssd_batch microbench + decode A/B).
4. **gated.py** (v3f): `post(..., wk)`, `spin(..., phase)`, and the phase-B region buffer when `BIGMOE_DEFER > 0`.
5. **stream.py**: the tier `post` closure passes `**kw` (wk) through. One line.
6. **tier.py** (`BIGMOE_DH=W`): `_dh_call` plus a branch at the top of `tier_call`, and `install_dh(model)`, which
   wraps `DecoderLayer.__call__` to add the late HC write. About 100 lines. Per layer, OoO as v3:
   - post: bit 20 on ranks < W, bit 21 on the rest. The sidecar (ranks 0-1) uses a plain row.
   - Phase 1 GLU: pairs resident at post (residency table).
   - Spin A.
   - Phase 2 GLU: pairs released in phase A.
   - `combine_dh(which=0)`: phases 1-2, late pairs skipped.
   - After layer L+1's attention: spin B (full slots), then a phase 3 GLU over the late pairs only (every other
     threadgroup returns at once), then `combine_dh(which=1)`, then the HC write with layer L's inject.
   - `BIGMOE_DH_D=1` (d1, ungated on v3): the late part is written at the end of layer L+1 instead, so layer L+1's
     routing does not wait for L's late reads.
6b. **Optional** `BIGMOE_DH_SUB=1`: the phase 3 call runs the subset kernels (`tier_glu_pairs_phase`) over the fixed
   pair list (m, r >= W) instead of the whole pair grid. That is 40 % fewer empty threadgroups at W=4.
   Value: PENDING (all-hits overhead dh vs dh_sub).
7. **tierglu.py**: `combine_dh`, the combine kernel with a phase-3 selector, in MLX's LY=8 order. With no late pair
   it is bitwise `combine_phase`.

## Checks (defer/run.py, defer/tf_compare.py)
- `DH_ARM=dh10` (flag mode, nothing late) must equal v3 bit for bit (teacher-forced logits, S=4 and S=8).
- `DH_ARM=force` (every rank >= 4 late, residency-independent) must equal `DH_ARM=ref` (the same split computed from
  v3's synchronous per-rank 3c outputs).
- RESULT: PENDING.

## Speed (24 GB, BigMoE b13's deployed v3 env, spec125 adaptive MTP; defer/run_gpu1.sh)
- Model (defer/sim2.py, v3 traces): realistic +12-17 % for dh, about +16-22 % for d1. +22-28 % only without the second
  SSD batch latency and the GPU tail before layer L+1's post.
- All-hits overhead per pass at S=4 / S=8 (v3 vs dh vs force): PENDING.
- Decode A/B, code / prose / edit (tok/s; verify ms at matched S): PENDING.

## Constraints
- Verify rows S <= 8 need `BIGMOE_MAXK=80` (80 ids in a routing row; already in the deployed env).
- W >= the number of 4-bit ranks (2): the sidecar is never late.
- Flag mode with JIT prefetch (`BIGMOE_PF_JIT=1`) runs in the A/B, but has no separate check.
