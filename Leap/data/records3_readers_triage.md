# records3 readers: triage before the in-place conversion

Read-only triage of every program that reads `~/BigMoE/models/bare/records3/` (L00..L47.bin + index.json), ahead of the
in-place conversion planned for Thu 2026-11-05 ~10:05-10:45 EET. Written Wed 2026-11-04 18:34-19:05 EET. Nothing was edited,
moved or run, apart from read-only grep / stat / lsof / pgrep / launchctl print.

**What changes on disk.** Every record becomes "native+p11+pad":
- pieces in native order, s3 / b3 / b packed to 11 bits, pieces 4 KB aligned;
- record 2,932,736 → 2,785,280 B (MSB part 2,064,384 + LSB part 720,896).

index.json gets a per-layer "layout" key plus the new sizes and offsets, and keeps the old entry under "rowmajor".
Sources: `~/BigMoE/docs/PLAN_records3_convert.md` §1; `~/BigMoE/scripts/convert_records3.sh:48-62`.

**Engine state now: the layout patch is live.**
- `~/BigMoE/bigmoe/tier.py`:
  - LAYOUTS (tier.py:27);
  - load_index raises on a converted layer unless the caller passes layout_ok (:203-212);
  - tier_call refuses a converted layer without Leap's router (:171-172).
- stream.py passes layout_ok=True (:496); BIGMOE_RECORDS=3 asserts row-major (:518).
- The ext's read_expert, read_rebuilt_layout and record_rowmajor turn layout 1-2 records back into row-major
  (bigmoe_ext.cpp:392-443, 2063-2112).

**Verdicts.**
- **BREAKS-ACTIVE**: raw and layout-blind, AND any of: running now, armed or scheduled Wed night to Fri, named in a Thu / Fri
  plan item, or touched since Tue 11-03 12:00 as current work.
- **BREAKS-STALE**: raw and layout-blind, none of the above. Backups and frozen copies that nothing imports count here.
- **SAFE**: layout-aware, or reads through the engine / ext path.
- **NOT-A-READER**: only mentions records3 (sizes as constants, docstrings), or reads another store.

"Raw" includes the engine forks (Breakthrough / Fit12 / MoonShots / TrainLab copies of BigMoE's engine): none of the 24 has
the guard.

## (a) Summary

Files = the 238 listed + 138 new since Tue 15:44 + 9 found by import tracing / EvalSuite snapshots (see (e)).

| Owner | BREAKS-ACTIVE | BREAKS-STALE | SAFE | NOT-A-READER | Files | list / new / extra |
|---|---|---|---|---|---|---|
| BigMoE | 0 | 4 | 67 | 11 | 82 | 50 / 32 / 0 |
| Breakthrough | 0 | 46 | 25 | 7 | 78 | 51 / 27 / 0 |
| MoonShots | 12 | 35 | 1 | 11 | 59 | 45 / 14 / 0 |
| Fit12 | 2 | 37 | 0 | 9 | 48 | 39 / 5 / 4 |
| Leap | 18 | 32 | 13 | 1 | 64 | 33 / 30 / 1 |
| TrainLab | 0 | 10 | 1 | 0 | 11 | 11 / 0 / 0 |
| FableGod | 0 | 0 | 6 | 7 | 13 | 8 / 5 / 0 |
| NewIdeas | 0 | 0 | 0 | 5 | 5 | 0 / 5 / 0 |
| EvalSuite | 2 (guarded) | 8 | 7 | 7 | 24 | 0 / 20 / 4 |
| SpecDecode | 0 | 0 | 1 | 0 | 1 | 1 / 0 / 0 |
| **All** | **34** | **172** | **121** | **58** | **385** | 238 / 138 / 9 |

Not counted above: **2 data traps** (overlay stores, see (b) T1-T2): BigMoE `scratch/model_mix` and FableGod
`decision_map/r3_overlay`.

**Headlines.**
1. **Running now:** one layout-blind value reader, Fit12's TRL-store encoder (`fit_trl_store.py` → `fit_q.read`). It ends by
   22:45 Wed at the latest, before the conversion. It must not be restarted after the conversion without a layout check.
2. **Leap holds most of the active risk:** 18 current check / probe scripts read records3 as row-major. Its own Thu / Fri
   TODO is the risky case: "the QOM form on 3c-N before QOM runs on Friday's converted store" (Leap HANDOVER:54). In
   self-consistency checks, both sides would come from the same misread bytes, so such a check could pass on garbage.
3. **MoonShots:** 11 SSD timing-harness files with row-major part sizes hard-coded, plus `lossless45.py` (values).
4. **EvalSuite needs no pause.** RELAX's frozen engine (snap/bigmoe-6c64148) is layout-blind, but runner 40 exits on a converted
   index, and the conversion waits for RELAX_DONE. Runners 42-44 and r3check.sh use the patched snapshot. T0 reads records3m.
5. **Two loud blockers (not silent misreads; not in the counts):**
   - **Breakthrough's map:** `dm_t3.py` refuses "native+p11+pad" (its LAYOUT map, dm_t3.py:21, lacks the name), so Thu's
     map leftovers and FLOOR night 2 on the converted store will stop with an error.
   - **The conversion itself:** `convert_records3.sh:34` dies without `~/EvalSuite/results/T0_DONE`. EvalSuite's 18:05 order
     runs T0 after the conversion (Thu 15:00 → Fri 00:15; EvalSuite HANDOVER:383,392). Neither RELAX_DONE nor T0_DONE
     exists at 18:5x. As written, the Thu ~10:05 run would refuse.

## (b) BREAKS-ACTIVE (34 files) + 2 data traps

### B1. Value readers: wrong numbers, no error (21 files)

| # | File (mtime) | How it reads records3 | Why active (evidence) | Who |
|---|---|---|---|---|
| 1 | `~/Fit12/scripts/fit_trl_store.py` (11-04 14:29) | through `fit_q.read(L, e)` per expert (fit_trl_store.py:23,104). The file never says "records3", so the list missed it. | **RUNNING**: trl_chunks2.sh pid 84909 (stop 21:30), and trl_relay.sh pid 11163 may re-exec it with STOP_AT 2241 (Fit12 HANDOVER:917-934). It encodes the TRL store for map screen 7. | Fit12: let it end tonight as armed; no restart after the conversion until #2 is guarded |
| 2 | `~/Fit12/scripts/fit_q.py` (10-24) | os.pread of recsz at msb_piece_off / lsb_piece_off; s3 / b3 / b decoded as bf16 (fit_q.py:15-16,33-50); no layout check | the reader inside #1 | Fit12: assert the layer's "layout" is row-major in read(), or read through the ext's record_rowmajor |
| 3-7 | `~/Leap/scripts/ooo1c_check.py`, `ooo1c_native_check.py`, `ooo1c_qom_check.py`, `ooo1c_qom_native_check.py`, `ooo1c_wire_check.py` (11-04 15:11-17:15) | np.memmap of L20 at the index recsz; row-major / native / p11 pools are built in RAM; no layout check | Today's OOO-1C / QOM-OOO checks. Leap's TODO "Thu/Fri: the QOM form on 3c-N before QOM runs on Friday's converted store" (Leap HANDOVER:54); native one-call forms "for after Friday" (:186) | Leap: before any re-run after the conversion, add k90.py's assert (k90.py:28-29) or read via record_rowmajor |
| 8-11 | `~/Leap/scripts/nax_micro.py`, `nax_ablate.py`, `nax_occ_probe.py`, `merge_check.py` (11-04 15:52-16:00) | np.fromfile of a whole layer, reshape(E, recsz) | the current NAX6-N microbench / ablation line (Leap HANDOVER:109-112) | Leap: same |
| 12-15 | `~/Leap/scripts/p11pad_check.py` (11-04), `p11_check.py`, `coh_check.py`, `p11.py` self_test (11-03) | np.memmap, then p11.to_p11 on bytes it assumes are row-major; to_p11 does not check the source layout (p11.py:148-232, self_test :234-262) | this conversion's p11 / pad format checks | Leap: same. p11.py's pack / unpack (used by tierglu_3cn / nax6n / ooo1c) is unaffected |
| 16-18 | `~/Leap/scripts/native3c_check.py`, `native3c_screen.py`, `native_sample.py` (11-03 15:36-15:42) | np.memmap / np.fromfile; native_sample.py writes ~/Leap/data/native_sample from records3 read as row-major | NATIVE-3C checks around the conversion (in the 238 list) | Leap: same |
| 19-20 | `~/Leap/scripts/wc_screen.py` (11-03 17:57), `h1p_screen.py` (11-03 15:13) | np.memmap / np.fromfile | wc_screen = the WHOLE-CHIP screen (Leap HANDOVER:230). h1p_screen is active by the date rule only (UNSURE, see (f)). | Leap: same, or mark them row-major only |
| 21 | `~/MoonShots/scripts/lossless45.py` (11-03 17:04) | fixed piece offsets, no layout check | M-27, this week. MoonShots' own rule: after the conversion, route through load_index (MoonShots HANDOVER:560-561). | MoonShots |

### B2. Timing harnesses: row-major sizes, bytes not decoded (11 files)

| # | File | How | Why active | Who |
|---|---|---|---|---|
| 22-25 | `~/MoonShots/scripts/ssdmicro.c` (BASE 2162688, :38), `ssdmicro2.m` (:105,:295), `ssdmicro3.c` (:165), `ssdmicro4.c` (:138) | C / ObjC pread of records3 L*.bin at row-major part sizes | M-28..M-31, Tue-Wed; the batched SSD ticket ran Wed 09:01 (MoonShots HANDOVER:636-637). Nothing armed now. | MoonShots: take part sizes and offsets from index.json before re-measuring on the converted store. New numbers taken with old sizes would describe the old layout. |
| 26-32 | `~/MoonShots/scripts/run_ssdmicro.sh`, `run_ssdmicro2.sh`, `run_ssdmicro2b.sh`, `run_ssdmicro3.sh`, `run_ssdmicro4.sh`, `~/MoonShots/scratch/run_wed_ticket.sh`, `scratch/run_m29b_z1.sh` | launchers of #22-25 | same | MoonShots |

### B3. Already guarded: no action (2 files)

| # | File | How | Why active | Guard |
|---|---|---|---|---|
| 33-34 | `~/EvalSuite/snap/bigmoe-6c64148/bigmoe/tier.py`, `stream.py` | frozen engine without the layout patch | RELAX on runner 40, Wed ~23:50 → Thu ~07:10 (EvalSuite HANDOVER:386) | Runner 40 exits 8 if index.json has "layout" or a recsz other than 2,932,736 (day_runner40.sh:51-53,89,104). The conversion dies without RELAX_DONE (convert_records3.sh:33). |

### T. Data traps: silent after the conversion (stores, not programs; not in the counts)

Each overlay directory symlinks every layer file but keeps its **own copy** of index.json:

| # | Overlay | Index copy | Who reads it | Who: action |
|---|---|---|---|---|
| T1 | `~/BigMoE/scratch/model_mix/records3/` | 11-03 22:53; 3 layers "native+p11+pad" (scratch/r3p11pad) and **45 row-major entries whose L*.bin link into models/bare/records3** | runs with R57_MODEL_DIR=scratch/model_mix: mix_e2e.sh, test_fc_tier.py, test_layout_engine.py | BigMoE: delete or re-index right after the conversion. The e2e already ran Wed 07:56-08:05 (data/e2e/e2e.log). |
| T2 | `~/FableGod/decision_map/r3_overlay/` | 11-03 17:36; 3 unpadded native+p11 layers + 45 row-major entries into records3 | dm_t3 with DM_R3=r3_overlay (Breakthrough's layout check, done) | Breakthrough: delete after the conversion |

After the conversion, those 45 files are native+p11+pad while the copied entry still says row-major. A layout-aware reader
trusts the entry, so it decodes converted bytes as row-major with no error.

## (c) Sessions to notify

**Fit12** (2 active, 37 stale)
- Active: `scripts/fit_q.py` (the reader) and `scripts/fit_trl_store.py` (the running TRL encoder).
  - Let tonight's encode end as armed (stop 21:30, relay to 22:41).
  - After the conversion, restart only once fit_q.read checks the layer's "layout", or reads through record_rowmajor.
- Stale:
  - fit_b3 / fit_q2 / fit_q3 (via fit_q.read);
  - b4bench.swift + b4.sh, ssd_burst.py;
  - the bt_b12 copies in btc/ and pinned/: btc/ and pinned/bt_b12.py default to records3; the others read it only with
    B12_M01=0 or engine ≤ v3m;
  - the forks v3g / j / k / n53 / o53 / p / q / r.

**Leap** (18 active, 32 stale)
- Active: B1 #3-20.
  - The fix is the 2-line assert k90.py:28-29 already uses, or a read through record_rowmajor.
  - The k90 / MISSION 33 family is already guarded (k90.py:28).
- Stale: the N7 / H1 / K-62 screens of Sun 11-01 to Tue 11-03 02:00, ane_probe.py's load(), and the old
  p11_v1 / v2 / before_pad copies.
- WATCH: b10_trellis.py (and s11_grouped.py through it) may come back after map screen 7 (Leap HANDOVER:122-124).

**MoonShots** (12 active, 35 stale)
- Active: B1 #21 and B2.
- Stale:
  - SSD harnesses: defer/ssd_batch|prefix|qd2.c, the b8mr / defer launchers, engine/bench/latency.swift;
  - planes_engine converters and tests (make_records3.layout() hard-codes row-major offsets);
  - msb_entropy, pom24, sb_audit, sbx, sbx_confirm, scale_exp, xc_entropy, lsrefit2, ssdwake, w24sp/screen,
    wide/microbench;
  - w24ec.py (also behind BigMoE's r16_bg.py);
  - the forks defer/v3d, v3d2, v3h.

**Breakthrough** (0 active, 46 stale; also owns ~/FableGod/decision_map)
- Blocker (loud): dm_t3.py's LAYOUT map (dm_t3.py:21) has no "native+p11+pad", so every t3 arm raises at dm_t3.py:42
  after the conversion. That hits:
  - the 20:00 batch's Thu leftovers (s67_slot.sh:10-11; Breakthrough HANDOVER:800);
  - FLOOR night 2 on the converted store (HANDOVER:970).
- Fix:
  - map "native+p11+pad" to 2 (BigMoE's ext layout code for it, tier.py:27);
  - re-run test_t3_layout.py on a padded copy (~/BigMoE/scratch/r3p11pad). It was only run on the unpadded r3p11
    (test_t3_layout.py:1-5).
- Trap T2: delete `~/FableGod/decision_map/r3_overlay/` after the conversion.
- Stale: bt_b12.py and the _bt_b12_* copies read records3 only with B12_M01=0 or B12_ENGINE ≤ v3m (bt_b12.py:147-150),
  through layout-blind forks.
  - A guard there must wait for T0_DONE. EvalSuite's runner 41 pins bt_b12.py and ~/Breakthrough/v3p, and stops on any
    change (day_runner41.sh:8-9).
  - Also stale: la_advise_test, lossless_screen, m01_ab / m01_arms / runner_m01, test_v3n_m01, tf_b12, v3o_creep, and the
    forks v3f..v3q (27 files).
  - v3p is in use this week, but on records3m only: bt_b12 sets BIGMOE_M01=1 for T0 and Fit12's T3.

**BigMoE** (0 active, 4 stale)
- Trap T1: `scratch/model_mix`.
- Blocker (loud): convert_records3.sh:34 requires `~/EvalSuite/results/T0_DONE`, but T0 now runs after the conversion
  and reads records3m only (EvalSuite HANDOVER:383,392). Drop that condition or agree a new order before Thu ~10:05.
- Stale:
  - m01_fields.py;
  - r16_bg.py (via MoonShots' w24ec.read_codes);
  - scratch/stage_layout/pre_live/{stream,tier}.py (pre-patch backups).
- Everything else is SAFE:
  - engine, ext;
  - r57_screen.py and its frozen copies (r57_screen.py:281-286);
  - the b*_runner.sh (live engine);
  - the tests that now assert row-major (BigMoE HANDOVER:1157);
  - tonight's armed 20:20 / 23:00 / Thu-morning tickets: r57_screen + c45_runner only.

**EvalSuite** (2 active but guarded, 8 stale): nothing to pause (B3).
- Runners 42-44 and r3check.sh use the patched snap/bigmoe-da8e7a6 with store checks.
- Runner 41 (T0) reads records3m only (day_runner41.sh:13-14).
- After the conversion, never point a runner at snap/bigmoe-6c64148 or bigmoe-d0daa68. The older runners 25 and 30-39 and
  relax_smoke.sh pin them without a store guard.
- cli.py's B12 variants on v3f / v3l / v3m (cli.py:95-103, 121) read records3 through layout-blind forks.
- The six snapshot copies of p11.py are stale: their self_test is never run.

**TrainLab** (10 stale): the d1i / main24 / main24min / main24min2 forks and the D1i test of 10-22: row-major only.

**FableGod**: see Breakthrough (dm_t3, r3_overlay).

**NewIdeas, SpecDecode**: nothing (byte pricing; spec125 uses the live engine).

## (d) BREAKS-STALE (172 files): mark row-major only

Raw and layout-blind, and nothing armed or current runs them. Per-file evidence is in (g).

**Engine forks and old engine snapshots: 24 forks + 1 EvalSuite snapshot (70 files), no layout guard.** load_index →
records3, or records3m with BIGMOE_M01 on the M-01 forks.

| Owner | Fork dirs (each: bigmoe_v3*/stream.py, tier.py + ext/bigmoe_ext.cpp) | Files |
|---|---|---|
| Breakthrough | `~/Breakthrough/v3{f,g,i,l,m,n,o,p,q}/`. v3p is used this week on records3m only (T0, Fit12 T3). | 27 |
| Fit12 | `~/Fit12/v3{g,j,k,n53,o53,p,q,r}/` | 24 |
| MoonShots | `~/MoonShots/defer/v3d`, `v3d2`, `v3h` | 9 |
| TrainLab | `~/TrainLab/d1i/v3f`, `main24`, `main24min`, `main24min2` (stream.py + tier.py only) | 8 |
| EvalSuite | snapshots `~/EvalSuite/snap/bigmoe-d0daa68/bigmoe/` (tier.py, stream.py), last run by day_runner25.sh on 10-28 | 2 |

**Scripts and copies (102 files).**

**BigMoE (4):**
- `~/BigMoE/scripts/`: m01_fields.py, r16_bg.py
- `~/BigMoE/scratch/stage_layout/pre_live/`: stream.py, tier.py (pre-patch backups)

**Breakthrough (19 + 27 fork files):**
- `~/Breakthrough/scripts/`:
  - _bt_b12_bis.py, _bt_b12_gc.py (default engine reads records3);
  - _bt_b12_ff.py, _bt_b12_m01.py, _bt_b12_pfrenv.py, _bt_b12_pfrmin.py, _bt_b12_r60.py, _bt_b12_refill.py, _bt_b12_v3o.py,
    _bt_b12_v3q.py, bt_b12.py (records3 only with B12_M01=0 / engine ≤ v3m);
  - la_advise_test.py (timing), lossless_screen.py, m01_ab.py, m01_arms.sh, runner_m01.sh, test_v3n_m01.py, tf_b12.py,
    v3o_creep.py.

**MoonShots (26 + 9 fork files):**
- `~/MoonShots/b8mr/`: run_3b.sh, run_ssd_b10.sh, run_ssd_qd2.sh (timing launchers)
- `~/MoonShots/defer/`: run_gpu1.sh, ssd_batch.c, ssd_prefix.c, ssd_qd2.c (timing)
- `~/MoonShots/engine/bench/`: latency.swift (timing)
- `~/MoonShots/patches/planes_engine/`: make_records3.py, make_records4.py, test_rebuild.py, test_split.py,
  test_tierglu.py, test_tierglu4.py
- `~/MoonShots/scripts/`: lsrefit2.py, msb_entropy.py, pom24.py, sb_audit.py, sbx.py, sbx_confirm.py, scale_exp.py,
  xc_entropy.py, w24ec.py, ssdwake.py (timing)
- `~/MoonShots/w24sp/`: screen.py
- `~/MoonShots/wide/`: microbench.py

**Fit12 (13 + 24 fork files):**
- `~/Fit12/b4/`: b4bench.swift (timing)
- `~/Fit12/scripts/`: b4.sh (timing), ssd_burst.py (timing), fit_b3.py, fit_q2.py, fit_q3.py (all three via fit_q.read)
- `~/Fit12/btc/bt_b12.py` and `~/Fit12/pinned/bt_b12.py` (default engine reads records3)
- `~/Fit12/pinned/{31e4,8db7,b23,b677,cf8b}/bt_b12.py` (records3 only in non-default modes; 8db7 is T3's pin)

**Leap (32):**
- `~/Leap/data/`: p11_v1.py, p11_v2.py, p11_before_pad.py (old p11 versions, not imported)
- `~/Leap/scripts/`:
  - n7_repack.py, n7_repack_check.py, nax_hybrid_check.py, n7_stack.py, n7_native.py, n7_trim.py, h1i_screen.py, n7_ks.py,
    naxc_screen.py, n7f_screen.py, n7_gap.py, h1_hybrid.py, n7_regs.py, n7_resid.py, n7_screen.py, n7_occ.py,
    n7_knock.py, n7_trace_driver.py, n7_breakdown.py;
  - ane_probe.py (its load() only);
  - k62p_floor.py, k62o_nax_ooo.py, k62j_nax6.py, k62i_nax5.py, k62h_gpugrp.py, k62g_acc_real.py, k62f_real.py;
  - b10_trellis.py (WATCH: may be revived), s11_grouped.py (via b10_trellis).

**TrainLab (2 + 8 fork files):**
- `~/TrainLab/d1i/run_d1i_h.sh`, `~/TrainLab/d1i/tests/test_planes.py`

**EvalSuite (6 + 2 snapshot files):**
- `~/EvalSuite/snap/leap-{c805fd0,52cf2c9,2b3fc63,90456ce,0002e2e,7f43a3f}/p11.py`: frozen libraries whose self_test is
  never run.

## (e) New readers found after Tue 11-03 15:44

**138 files** mention records3 (excluding .md / .txt / .log / .json) and changed after the list was made:

| Verdict | Count | Files |
|---|---|---|
| BREAKS-ACTIVE | 26 | Leap 14: the ooo1c_* checks (5), nax_micro / nax_ablate / nax_occ_probe, merge_check, p11pad_check, p11_check, coh_check, p11.py, wc_screen.py. MoonShots 12: ssdmicro.c / 2.m / 3.c / 4.c, run_ssdmicro{,2,2b,3,4}.sh, scratch/run_wed_ticket.sh, scratch/run_m29b_z1.sh, lossless45.py |
| BREAKS-STALE | 11 | BigMoE pre_live backups (2); Leap data/p11_v1 / v2 / before_pad (3); the EvalSuite snapshot p11.py copies (6) |
| SAFE | 72 | see below |
| NOT-A-READER | 29 | docs backups, byte-pricing scripts, mask builders, provenance notes |

The 72 SAFE files:
- BigMoE: r57_screen.py, r57_screen_rlx.py, 14 frozen r57_screen copies, the conversion / A/B / e2e scripts, the layout
  tests;
- Breakthrough: the abl_scratch copies of the map (dm_t3 copies keep the layout map), the dm_s7 patch and tests,
  night_prefix.sh, wed_day.sh;
- EvalSuite: day_runner40 / 42 / 43 / 44, r3check.sh, snap/bigmoe-da8e7a6;
- FableGod: mk_r3_overlay.py, test_t3_layout.py;
- Leap: the k90 family (k90.py:28 guard), nat_engine_check.py, convert_records_p11.py, the w1_pickup ext source copies.

**9 more readers** are not covered by that grep. They read records3 through an import (no "records3" string in the file),
or they predate the list but were missing from it:

| File | Verdict | Reads through |
|---|---|---|
| `~/Fit12/scripts/fit_trl_store.py` (11-04) | **BREAKS-ACTIVE, running** | fit_q.read (B1 #1) |
| `~/Fit12/scripts/fit_b3.py`, `fit_q2.py`, `fit_q3.py` (10-24/25) | BREAKS-STALE | fit_q.read |
| `~/Leap/scripts/s11_grouped.py` (10-25) | BREAKS-STALE | b10_trellis's 3c pools |
| `~/EvalSuite/snap/bigmoe-6c64148/bigmoe/{tier,stream}.py` | BREAKS-ACTIVE, guarded (B3) | layout-blind engine snapshot used by RELAX |
| `~/EvalSuite/snap/bigmoe-d0daa68/bigmoe/{tier,stream}.py` | BREAKS-STALE | layout-blind engine snapshot |

**The 238-path list itself:** 5 BREAKS-ACTIVE, 155 BREAKS-STALE, 49 SAFE, 29 NOT-A-READER.
- 29 entries only use records3's part sizes or name it in text.
- The list predates the guards that now protect BigMoE's own readers: m01 / m03_convert, test_records3*.py,
  test_bounce_pool.py (BigMoE HANDOVER:1157).

## (f) Method, and what I could not decide

**Method**
1. **Inputs.** All 238 listed paths exist.
   - `find -newermt '2026-11-03 15:44'` over the ten session dirs (pruning .git, .venv, models, __pycache__, files > 3 MB)
     gave 7,447 files.
   - `grep -lI records3` on those gave 205; dropping .md / .txt / .log / .json and the listed paths left 138.
   - ~/SpecDecode had no new hit.
2. **Per file.** I grepped for:
   - the store name (records3, not records3m / records3p);
   - raw reads: np.fromfile / memmap / os.pread / preadv / open 'rb' / C pread / fopen;
   - engine imports (bigmoe, load_index) and ext readers (read_expert, read_rebuilt_layout, record_rowmajor);
   - a "layout" check;
   - offset keys and hard-coded sizes.

   Where the read path was not obvious, I read the code (cited file:line in (g)).
3. **Indirect readers.** I grepped the imports of all 124 modules that do raw reads, across every session.
   - Hits: importers of fit_q, b10_trellis, bt_b12, laysz45, p11 and ane_probe.
   - laysz45 reads only safetensors (laysz45.py:51-64); p11's importers use only pack / unpack; ane_probe's importers take
     only drafter helpers. None of these is a records3 reader.
   - fit_q and b10_trellis importers are records3 readers (in (e)).
4. **Forks.** I checked each fork's load_index (no guard in any of the 24) and its store (records3, or records3m when
   BIGMOE_M01 is set on v3n+). bt_b12 loads `~/Breakthrough/<engine>` (bt_b12.py:109,280).
5. **Use: plans.** I read the current sections of each session's HANDOVER / PLAN, plus the armed scripts:
   - BigMoE bd160n_wed.sh, eve_2300.sh → allin_wed.sh / l2_wed.sh, and night_thu.sh → agent_loo.sh / d256_wed.sh: all are
     r57_screen + c45_runner;
   - Breakthrough dm_s7/s67_slot.sh;
   - Fit12 trl_chunks2.sh / trl_relay.sh;
   - EvalSuite day_runner40-44.
6. **Use: scheduler and processes.**
   - launchd: 13 com.kamen agents + com.kamen.nightfiller, which is still loaded though its plist is gone. Its
     night_filler.sh:6 exits unless the date is 2026-10-20.
   - cron: no crontab.
   - None of them starts a records3 reader.
   - At 18:46: run_map.py s7_cal (pid 25518, via dm_t3) held records3 L00-L47 open, and fit_trl_store.py (pid 32225) was
     running.
7. **EvalSuite.**
   - evalsuite/env.py takes the engine from EVALSUITE_BIGMOE_CODE (env.py:11; default the live ~/BigMoE).
   - I checked every snapshot dir for the patch: only bigmoe-da8e7a6 has it.
   - I checked every recent runner's pin and store guard.
8. **Overlays.** I searched for directories of symlinked L*.bin. There are exactly two (T1, T2), and both carry a copied
   index.json.

**Open points (UNSURE)**
- **bt_b12.py and its copies** are stale only because their records3 modes are chosen by env and unused this week.
  - A run with B12_M01=0, or an engine ≤ v3m, after the conversion would misread silently.
  - I did not classify the ~35 older scripts that import bt_b12 (Breakthrough / Fit12, 10-22..11-02) one by one. They
    inherit this rule.
- **h1p_screen.py** is active by the date rule only; no open HANDOVER item names it.
- **b10_trellis.py** is stale but may be revived after map screen 7 (Leap HANDOVER:122-124).
- **The B2 timing harnesses** count as BREAKS although they decode no values. Their risk is measuring old part sizes on
  the new files, not wrong numbers in a model.
- **The pad format through dm_t3 + build_nat's record_rowmajor is untested.** Breakthrough's test covered unpadded
  native+p11 only. BigMoE's ext maps all three p11 names to layout 2 (tier.py:27), so the fix should be small; this is
  not verified.
- **Not scanned:**
  - compiled .so files and binaries;
  - the publish mirrors under ~/.local/share/github-mirrors (copies, not run);
  - directories outside the ten sessions, except ~/.local/share/pacer (clean).
- **Byte-pricing scripts** with row-major constants (2,162,688 / 770,048) read no store. Examples: laysz45.py and its ~20
  importers, ten_x_ledger.py, s10c_ab.py, draft17*.py. Their MB-per-token figures for the converted store need the packed
  sizes (2,064,384 / 720,896).

## (g) Per-file table (385 files)

"src": list = the 238-path list; new = changed since Tue 15:44; extra = found by import tracing / snapshot check.
mtime is as of 18:55 Wed.

| Owner | File | mtime | src | Verdict | How it reads records3 | Evidence / use |
|---|---|---|---|---|---|---|
| BigMoE | `~/BigMoE/scratch/stage_layout/pre_live/stream.py` | 11-04 07:50 | new | BREAKS-STALE | engine copy (pre-patch backup) | layout-blind backup taken at item 0 (07:50); not on any sys.path |
| BigMoE | `~/BigMoE/scratch/stage_layout/pre_live/tier.py` | 11-04 07:50 | new | BREAKS-STALE | engine copy (pre-patch backup) | layout-blind backup taken at item 0 (07:50); not on any sys.path |
| BigMoE | `~/BigMoE/scripts/m01_fields.py` | 10-29 15:46 | list | BREAKS-STALE | own raw reads (os.pread at msb_piece_off / lsb_piece_off) | no layout check (m01_fields.py:19-27); R-44 field study, 10-29 |
| BigMoE | `~/BigMoE/scripts/r16_bg.py` | 10-28 16:04 | list | BREAKS-STALE | own raw reads via MoonShots w24ec.read_codes | w24ec.py:39-47 preads w3 code bytes at index offsets, no layout check; 10-28 |
| BigMoE | `~/BigMoE/bigmoe/stream.py` | 11-04 07:50 | list | SAFE | engine | load_index(layout_ok=True) + per-layer layout to the ext (stream.py:496,591); BIGMOE_RECORDS=3 asserts row-major (stream.py:518) |
| BigMoE | `~/BigMoE/bigmoe/tier.py` | 11-04 07:51 | list | SAFE | engine | LAYOUTS (tier.py:27); load_index raises on converted layers unless layout_ok (tier.py:203-212); tier_call refuses a converted layer without Leap's router (tier.py:171-172) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_aloo.py` | 11-04 13:49 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_bd96.py` | 11-04 15:14 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_bdq.py` | 11-04 15:34 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_d256.py` | 11-04 09:49 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_d45.py` | 11-04 08:05 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_e2e.py` | 11-04 07:56 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_pr2.py` | 11-04 07:28 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_s160.py` | 11-03 19:16 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_sc.py` | 11-03 22:44 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_smk.py` | 11-04 15:50 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_smk3.py` | 11-04 17:10 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_u3.py` | 11-04 07:31 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_u3c.py` | 11-04 08:48 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/data/r57/snap/r57_screen_w1.py` | 11-04 07:42 | new | SAFE | engine (frozen r57_screen copy) | live engine + asserts the layout patch when converted layers exist (_CONV block) |
| BigMoE | `~/BigMoE/ext/bigmoe_ext.cpp` | 11-04 15:24 | list | SAFE | ext reader | set_layout_ / read_expert turn native+p11 back to row-major (bigmoe_ext.cpp:392-443); read_rebuilt_layout, record_rowmajor, layout_max (:2063-2112) |
| BigMoE | `~/BigMoE/scratch/stage_layout/bigmoe/stream.py` | 11-03 17:32 | new | SAFE | engine copy (staged, == live) | not on any sys.path |
| BigMoE | `~/BigMoE/scratch/stage_layout/bigmoe/tier.py` | 11-03 22:48 | new | SAFE | engine copy (staged, == live) | not on any sys.path |
| BigMoE | `~/BigMoE/scripts/ab_records3.sh` | 11-03 23:03 | new | SAFE | other: A/B driver, layout-aware | checks store state; Fri 00:30 A/B (BigMoE HANDOVER:1189-1201) |
| BigMoE | `~/BigMoE/scripts/b10_runner.sh` | 10-21 14:34 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b11_runner.sh` | 10-21 17:48 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b12_runner.sh` | 10-21 18:35 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b13_runner.sh` | 10-21 21:46 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b14_runner.sh` | 10-22 07:17 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b15_runner.sh` | 10-22 08:22 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b16_runner.sh` | 10-22 17:08 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b17_runner.sh` | 10-22 18:32 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b18_runner.sh` | 10-22 18:31 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b19_runner.sh` | 10-22 19:33 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b20_runner.sh` | 10-22 20:59 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b21_runner.sh` | 10-22 22:55 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b22_runner.sh` | 10-22 23:27 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b23_runner.sh` | 10-23 08:04 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b24_runner.sh` | 10-23 09:35 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b25_runner.sh` | 10-25 06:14 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b26_runner.sh` | 10-25 08:30 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b27_runner.sh` | 10-25 10:38 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b28_runner.sh` | 10-25 13:17 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b29_runner.sh` | 10-25 14:08 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b3_runner.sh` | 10-20 17:19 | list | SAFE | other: launcher (engine runs + test_records3.py) | engine on the live bigmoe; test_records3.py asserts row-major (test_records3.py:19); last used 10-20/21 |
| BigMoE | `~/BigMoE/scripts/b44_runner.sh` | 10-29 15:58 | list | SAFE | engine (launcher) | A = live engine on records3, S = records3m; 10-29 |
| BigMoE | `~/BigMoE/scripts/b46_runner.sh` | 10-29 20:06 | list | SAFE | other: launcher (test_records3*.py + engine) | the tests now assert row-major (test_records3.py:19 etc.); 10-29 |
| BigMoE | `~/BigMoE/scripts/b4_runner.sh` | 10-20 18:42 | list | SAFE | other: launcher (make_records3.py + engine runs) | conversion step skipped once records3/index.json holds layer 47 (b4_runner.sh:44); rest = live engine; 10-20 |
| BigMoE | `~/BigMoE/scripts/b5_runner.sh` | 10-20 20:34 | list | SAFE | other: launcher (engine runs + test_records3.py) | engine on the live bigmoe; test_records3.py asserts row-major (test_records3.py:19); last used 10-20/21 |
| BigMoE | `~/BigMoE/scripts/b6_runner.sh` | 10-21 08:55 | list | SAFE | other: launcher (engine runs + test_records3.py) | engine on the live bigmoe; test_records3.py asserts row-major (test_records3.py:19); last used 10-20/21 |
| BigMoE | `~/BigMoE/scripts/b7_runner.sh` | 10-21 09:57 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b8_runner.sh` | 10-21 11:09 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/b9_runner.sh` | 10-21 12:49 | list | SAFE | engine (launcher) | spec125 / gsm_v3 with BIGMOE_TIER or BIGMOE_RECORDS=3 on the live engine; records3 only in header comments; last used 10-21..10-25 |
| BigMoE | `~/BigMoE/scripts/convert_records3.sh` | 11-03 23:03 | new | SAFE | other: the conversion wrapper | Leap's tool --pad; checks index after (convert_records3.sh:48-62) |
| BigMoE | `~/BigMoE/scripts/fast24.py` | 10-23 08:05 | list | SAFE | engine | live bigmoe package (fast24.py:6,31); 10-23 |
| BigMoE | `~/BigMoE/scripts/m01_build_test.sh` | 10-29 15:57 | list | SAFE | other: launcher | runs test_records3.py (asserts row-major) |
| BigMoE | `~/BigMoE/scripts/m01_convert.py` | 11-03 17:40 | list | SAFE | own raw reads + layout check | asserts every layer row-major (m01_convert.py:30) |
| BigMoE | `~/BigMoE/scripts/m03_build_test.sh` | 10-29 17:41 | list | SAFE | other: launcher | runs test_records3.py (asserts row-major) |
| BigMoE | `~/BigMoE/scripts/m03_convert.py` | 11-03 17:40 | list | SAFE | own raw reads + layout check | asserts every layer row-major (m03_convert.py:31) |
| BigMoE | `~/BigMoE/scripts/mix_e2e.sh` | 11-04 08:05 | new | SAFE | engine (on scratch/model_mix) | asserts patch + ext layout_max (mix_e2e.sh:29-31); ran Wed 07:56-08:05 (data/e2e/e2e.log). See TRAP: scratch/model_mix |
| BigMoE | `~/BigMoE/scripts/r3layout_ref.py` | 11-03 22:46 | new | SAFE | own raw reads (reference converter, scratch only) | writes layout entries (r3layout_ref.py:61-65) |
| BigMoE | `~/BigMoE/scripts/r57_screen.py` | 11-04 17:09 | new | SAFE | engine | live ~/BigMoE + Leap ref2; asserts the layout patch when converted layers exist (r57_screen.py:281-286) |
| BigMoE | `~/BigMoE/scripts/r57_screen_rlx.py` | 11-04 18:00 | new | SAFE | engine (r57_screen copy) | same _CONV block; armed tonight 20:20 (BigMoE HANDOVER:850-853) |
| BigMoE | `~/BigMoE/scripts/test_bounce_pool.py` | 11-03 17:40 | list | SAFE | own raw reads + layout check | asserts row-major (test_bounce_pool.py:46) |
| BigMoE | `~/BigMoE/scripts/test_fc_tier.py` | 11-04 15:03 | new | SAFE | engine | load_index(model_mix, layout_ok=True) / load_index(bare) (test_fc_tier.py:26-28) |
| BigMoE | `~/BigMoE/scripts/test_layout.py` | 11-03 22:52 | new | SAFE | ext reader (layout test) | tests read_rebuilt_layout / record_rowmajor |
| BigMoE | `~/BigMoE/scripts/test_layout_engine.py` | 11-03 22:53 | new | SAFE | engine (layout test) | load_index refusal + layout_ok paths |
| BigMoE | `~/BigMoE/scripts/test_qom_ext.py` | 11-04 15:25 | new | SAFE | engine | load_index(models/bare) without layout_ok raises on a converted store (test_qom_ext.py:17) |
| BigMoE | `~/BigMoE/scripts/test_records3.py` | 11-03 17:40 | list | SAFE | own raw reads + layout check | asserts row-major (test_records3.py:19) |
| BigMoE | `~/BigMoE/scripts/test_records3m.py` | 11-03 17:40 | list | SAFE | own raw reads + layout check | asserts row-major (test_records3m.py:25) |
| BigMoE | `~/BigMoE/scripts/test_records3p.py` | 11-03 17:40 | list | SAFE | own raw reads + layout check | asserts row-major (test_records3p.py:24) |
| BigMoE | `~/BigMoE/patches/records3_layout.diff` | 11-03 22:48 | new | NOT-A-READER | patch file | the engine's layout patch (already live in tier.py / stream.py) |
| BigMoE | `~/BigMoE/scripts/ab_records3_report.py` | 11-03 23:03 | new | NOT-A-READER | reads result files | report only |
| BigMoE | `~/BigMoE/scripts/b45_runner.sh` | 10-29 18:16 | list | NOT-A-READER | other: launcher, records3m only | BIGMOE_M01=1 arms (records3m); no records3 path |
| BigMoE | `~/BigMoE/scripts/bytes45_study.py` | 11-01 06:08 | list | NOT-A-READER | mentions only | records3 part sizes in a docstring |
| BigMoE | `~/BigMoE/scripts/item0_live.sh` | 11-04 07:46 | new | NOT-A-READER | other: patch installer | applies records3_layout.diff + runs test_layout_engine.py |
| BigMoE | `~/BigMoE/scripts/m03_suite_check.py` | 10-29 17:30 | list | NOT-A-READER | reads records3p + safetensors | records3p index (m03_suite_check.py:19,30) and safetensors (:41); no records3 read |
| BigMoE | `~/BigMoE/scripts/r27_ab.py` | 10-29 15:58 | list | NOT-A-READER | reads miss.bin traces | r27_ab.py:17 |
| BigMoE | `~/BigMoE/scripts/s10c_ab.py` | 10-27 20:06 | list | NOT-A-READER | reads miss.bin traces; part sizes as constants | MSB, LSB = 2162688, 770048 (s10c_ab.py:16) = row-major byte accounting |
| BigMoE | `~/BigMoE/scripts/test_tier.py` | 10-20 19:57 | list | NOT-A-READER | reads safetensors | get_rec reads the safetensors shards (test_tier.py:18-29) |
| BigMoE | `~/BigMoE/scripts/test_vtier.py` | 11-03 15:46 | list | NOT-A-READER | synthetic pieces | records3 only as a size comment (test_vtier.py:14) |
| BigMoE | `~/BigMoE/scripts/w1b_24.sh` | 11-03 22:33 | new | NOT-A-READER | other: launcher, records3m only | records3m mention only |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_bis.py` | 10-28 16:08 | list | BREAKS-STALE | engine fork (bt_b12 copy, default v3l -> records3) | layout-blind fork tier.py (Breakthrough/v3l/bigmoe_v3f/tier.py:168); 10-28 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_ff.py` | 10-31 11:20 | list | BREAKS-STALE | engine fork (bt_b12 copy; default records3m, records3 with B12_M01=0 or B12_ENGINE<=v3m) | records3 modes only, layout-blind forks; 10-29..11-01 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_gc.py` | 10-29 01:48 | list | BREAKS-STALE | engine fork (bt_b12 copy, default v3m -> records3) | layout-blind fork; 10-29 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_m01.py` | 10-29 18:33 | list | BREAKS-STALE | engine fork (bt_b12 copy; default records3m, records3 with B12_M01=0 or B12_ENGINE<=v3m) | records3 modes only, layout-blind forks; 10-29..11-01 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_pfrenv.py` | 10-30 18:46 | list | BREAKS-STALE | engine fork (bt_b12 copy; default records3m, records3 with B12_M01=0 or B12_ENGINE<=v3m) | records3 modes only, layout-blind forks; 10-29..11-01 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_pfrmin.py` | 10-30 17:20 | list | BREAKS-STALE | engine fork (bt_b12 copy; default records3m, records3 with B12_M01=0 or B12_ENGINE<=v3m) | records3 modes only, layout-blind forks; 10-29..11-01 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_r60.py` | 10-31 17:26 | list | BREAKS-STALE | engine fork (bt_b12 copy; default records3m, records3 with B12_M01=0 or B12_ENGINE<=v3m) | records3 modes only, layout-blind forks; 10-29..11-01 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_refill.py` | 10-30 17:49 | list | BREAKS-STALE | engine fork (bt_b12 copy; default records3m, records3 with B12_M01=0 or B12_ENGINE<=v3m) | records3 modes only, layout-blind forks; 10-29..11-01 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_v3o.py` | 10-30 16:23 | list | BREAKS-STALE | engine fork (bt_b12 copy; default records3m, records3 with B12_M01=0 or B12_ENGINE<=v3m) | records3 modes only, layout-blind forks; 10-29..11-01 |
| Breakthrough | `~/Breakthrough/scripts/_bt_b12_v3q.py` | 11-01 04:41 | list | BREAKS-STALE | engine fork (bt_b12 copy; default records3m, records3 with B12_M01=0 or B12_ENGINE<=v3m) | records3 modes only, layout-blind forks; 10-29..11-01 |
| Breakthrough | `~/Breakthrough/scripts/bt_b12.py` | 11-01 12:40 | list | BREAKS-STALE | engine fork (live 12 GB loader; default v3p + BIGMOE_M01=1 -> records3m) | reads records3 only with B12_M01=0 or B12_ENGINE<=v3m (bt_b12.py:147-150), through layout-blind forks. This week's user T0 (EvalSuite runner 41) runs the records3m default (day_runner41.sh:13-14,24) |
| Breakthrough | `~/Breakthrough/scripts/la_advise_test.py` | 10-22 05:14 | list | BREAKS-STALE | own raw reads (preadv, F_NOCACHE) - timing | read-ahead timing on records3 records (la_advise_test.py:10-25); 10-22 |
| Breakthrough | `~/Breakthrough/scripts/lossless_screen.py` | 11-01 04:08 | list | BREAKS-STALE | own raw reads (open + read) - values | compression ratios of records3 parts, no layout check (lossless_screen.py:12,27); 11-01 |
| Breakthrough | `~/Breakthrough/scripts/m01_ab.py` | 10-29 17:31 | list | BREAKS-STALE | engine fork via bt_b12 (A arm = v3m on records3) | m01_ab.py:2,23; 10-29 |
| Breakthrough | `~/Breakthrough/scripts/m01_arms.sh` | 10-29 17:33 | list | BREAKS-STALE | other: launcher (m01_ab.py + test_v3n_m01.py --r3) | records3 arms through layout-blind forks; 10-29 |
| Breakthrough | `~/Breakthrough/scripts/runner_m01.sh` | 10-29 17:32 | list | BREAKS-STALE | other: launcher (A = v3m on records3) | runner_m01.sh:4; 10-29 |
| Breakthrough | `~/Breakthrough/scripts/test_v3n_m01.py` | 10-29 17:28 | list | BREAKS-STALE | own raw reads + fork (--r3 mode) | reads records3 index + part bytes (test_v3n_m01.py:27,33,63), no layout check; 10-29 |
| Breakthrough | `~/Breakthrough/scripts/tf_b12.py` | 10-22 07:18 | list | BREAKS-STALE | engine fork via bt_b12 (v3f, records3) | tf_b12.py:1,16-17; 10-22 |
| Breakthrough | `~/Breakthrough/scripts/v3o_creep.py` | 11-01 04:41 | list | BREAKS-STALE | engine fork via bt_b12 (ENV default B12_ENGINE=v3m -> records3) | v3o_creep.py:19,34; 11-01 |
| Breakthrough | `~/Breakthrough/v3f/bigmoe_v3f/stream.py` | 10-21 16:04 | list | BREAKS-STALE | engine fork (v3f) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3f/bigmoe_v3f/tier.py` | 10-22 06:30 | list | BREAKS-STALE | engine fork (v3f) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3f/ext/bigmoe_ext.cpp` | 10-22 00:15 | list | BREAKS-STALE | engine fork (v3f) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3g/bigmoe_v3f/stream.py` | 10-24 12:06 | list | BREAKS-STALE | engine fork (v3g) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3g/bigmoe_v3f/tier.py` | 10-22 06:30 | list | BREAKS-STALE | engine fork (v3g) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3g/ext/bigmoe_ext.cpp` | 10-24 12:06 | list | BREAKS-STALE | engine fork (v3g) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3i/bigmoe_v3f/stream.py` | 10-24 12:06 | list | BREAKS-STALE | engine fork (v3i) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3i/bigmoe_v3f/tier.py` | 10-26 08:37 | list | BREAKS-STALE | engine fork (v3i) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3i/ext/bigmoe_ext.cpp` | 10-25 05:56 | list | BREAKS-STALE | engine fork (v3i) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3l/bigmoe_v3f/stream.py` | 10-26 13:41 | list | BREAKS-STALE | engine fork (v3l) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3l/bigmoe_v3f/tier.py` | 10-26 13:41 | list | BREAKS-STALE | engine fork (v3l) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3l/ext/bigmoe_ext.cpp` | 10-26 13:41 | list | BREAKS-STALE | engine fork (v3l) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3m/bigmoe_v3f/stream.py` | 10-27 20:43 | list | BREAKS-STALE | engine fork (v3m) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3m/bigmoe_v3f/tier.py` | 10-27 20:43 | list | BREAKS-STALE | engine fork (v3m) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3m/ext/bigmoe_ext.cpp` | 10-27 20:44 | list | BREAKS-STALE | engine fork (v3m) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3n/bigmoe_v3f/stream.py` | 10-29 17:27 | list | BREAKS-STALE | engine fork (v3n) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3n/bigmoe_v3f/tier.py` | 10-29 17:27 | list | BREAKS-STALE | engine fork (v3n) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3n/ext/bigmoe_ext.cpp` | 10-29 17:27 | list | BREAKS-STALE | engine fork (v3n) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3o/bigmoe_v3f/stream.py` | 10-29 19:19 | list | BREAKS-STALE | engine fork (v3o) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3o/bigmoe_v3f/tier.py` | 10-29 19:19 | list | BREAKS-STALE | engine fork (v3o) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3o/ext/bigmoe_ext.cpp` | 10-29 19:19 | list | BREAKS-STALE | engine fork (v3o) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3p/bigmoe_v3f/stream.py` | 10-30 19:07 | list | BREAKS-STALE | engine fork (v3p) | no layout guard (v3p/bigmoe_v3f/tier.py:169-171). USED THIS WEEK but on records3m only: bt_b12 sets BIGMOE_M01=1 for T0 (runner 41, Thu 15:00-Fri 00:15) and Fit12 T3; records3 path unused |
| Breakthrough | `~/Breakthrough/v3p/bigmoe_v3f/tier.py` | 10-30 19:07 | list | BREAKS-STALE | engine fork (v3p) | no layout guard (v3p/bigmoe_v3f/tier.py:169-171). USED THIS WEEK but on records3m only: bt_b12 sets BIGMOE_M01=1 for T0 (runner 41, Thu 15:00-Fri 00:15) and Fit12 T3; records3 path unused |
| Breakthrough | `~/Breakthrough/v3p/ext/bigmoe_ext.cpp` | 10-29 19:19 | list | BREAKS-STALE | engine fork (v3p) | no layout guard (v3p/bigmoe_v3f/tier.py:169-171). USED THIS WEEK but on records3m only: bt_b12 sets BIGMOE_M01=1 for T0 (runner 41, Thu 15:00-Fri 00:15) and Fit12 T3; records3 path unused |
| Breakthrough | `~/Breakthrough/v3q/bigmoe_v3f/stream.py` | 10-30 19:07 | list | BREAKS-STALE | engine fork (v3q) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3q/bigmoe_v3f/tier.py` | 10-30 19:07 | list | BREAKS-STALE | engine fork (v3q) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/v3q/ext/bigmoe_ext.cpp` | 11-01 04:41 | list | BREAKS-STALE | engine fork (v3q) | no layout guard; load_index -> records3 (or records3m with BIGMOE_M01 on v3n+); last change 10-22..10-30 |
| Breakthrough | `~/Breakthrough/data/abl_scratch/dense3/base.py` | 11-04 16:59 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/dense3/test/dm_ablations.py` | 11-04 16:59 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/dm_ablations.py.qomt3` | 11-04 03:32 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/dm_ablations.py.screens` | 11-04 04:10 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/dm_ablations.py.slot` | 11-04 08:33 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/dm_t3.py.base` | 11-04 07:06 | new | SAFE | own raw reads + layout-aware (dm_t3 copy) | scratch copy of dm_t3 (same LAYOUT map, no native+p11+pad) |
| Breakthrough | `~/Breakthrough/data/abl_scratch/late/base.py` | 11-04 14:57 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/late/test/dm_ablations.py` | 11-04 14:57 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s1bc/dm_ablations.py.base` | 11-04 14:55 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s1bc/test/dm_ablations.py` | 11-04 14:55 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/base/dm_ablations.py` | 11-04 14:47 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/base/dm_common.py` | 11-04 14:47 | new | SAFE | other: dm_common copy (reads via dm_t3) | scratch copy |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/base/dm_t3.py` | 11-04 14:47 | new | SAFE | own raw reads + layout-aware (dm_t3 copy) | scratch copy of dm_t3 (same LAYOUT map, no native+p11+pad) |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/orig/dm_ablations.py` | 11-04 14:47 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/orig/dm_common.py` | 11-04 14:47 | new | SAFE | other: dm_common copy (reads via dm_t3) | scratch copy |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/orig/dm_t3.py` | 11-04 14:47 | new | SAFE | own raw reads + layout-aware (dm_t3 copy) | scratch copy of dm_t3 (same LAYOUT map, no native+p11+pad) |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/patched/dm_ablations.py` | 11-04 14:47 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/patched/dm_common.py` | 11-04 14:47 | new | SAFE | other: dm_common copy (reads via dm_t3) | scratch copy |
| Breakthrough | `~/Breakthrough/data/abl_scratch/s7/patched/dm_t3.py` | 11-04 14:47 | new | SAFE | own raw reads + layout-aware (dm_t3 copy) | scratch copy of dm_t3 (same LAYOUT map, no native+p11+pad) |
| Breakthrough | `~/Breakthrough/data/abl_scratch/t3test/dm_t3.py` | 11-04 07:06 | new | SAFE | own raw reads + layout-aware (dm_t3 copy) | scratch copy of dm_t3 (same LAYOUT map, no native+p11+pad) |
| Breakthrough | `~/Breakthrough/data/abl_scratch/test/dm_ablations.py` | 11-04 08:33 | new | SAFE | other: decision-map arm table copy (reads via dm_t3) | scratch / backup copy of dm_ablations.py; not run |
| Breakthrough | `~/Breakthrough/scripts/dm_s7/patch_s7.py` | 11-04 14:42 | new | SAFE | other: patch to dm_t3 (TRL store + dm_t3.read) | TRL files opened at patch_s7.py:58; records3 only via dm_t3 |
| Breakthrough | `~/Breakthrough/scripts/dm_s7/test_dm_s7.py` | 11-04 14:45 | new | SAFE | other: tests via dm_t3 | test_dm_s7.py:10,184 |
| Breakthrough | `~/Breakthrough/scripts/night_prefix.sh` | 11-03 22:29 | new | SAFE | other: launcher (map arms + layout check via dm_t3) | ran Wed 00:30 (done) |
| Breakthrough | `~/Breakthrough/scripts/wed_day.sh` | 11-03 17:37 | new | SAFE | other: launcher (map arms + layout check via dm_t3) | Wed 15:00 slot (done) |
| Breakthrough | `~/Breakthrough/keepset/qom_keep.py` | 11-04 15:15 | new | NOT-A-READER | mentions only | docstring (qom_keep.py:13) |
| Breakthrough | `~/Breakthrough/scripts/cand_check.py` | 10-30 17:49 | list | NOT-A-READER | engine fork via bt_b12 defaults (records3m) | pops B12_*/BIGMOE_M01 to the candidate's defaults (cand_check.py:23) |
| Breakthrough | `~/Breakthrough/scripts/dm_s7/install_s7.sh` | 11-04 14:47 | new | NOT-A-READER | other: installer | copies the s7 patch into decision_map |
| Breakthrough | `~/Breakthrough/scripts/eco12.py` | 10-30 16:20 | list | NOT-A-READER | mentions only | records3m part sizes (eco12.py:21) |
| Breakthrough | `~/Breakthrough/scripts/pom12.py` | 10-29 19:36 | list | NOT-A-READER | mentions only | records3m / records3p byte pricing (pom12.py:2-8) |
| Breakthrough | `~/Breakthrough/scripts/ubc_check.py` | 10-29 17:31 | list | NOT-A-READER | other: page-cache residency check | opens files to test residency (ubc_check.py:25); decodes nothing |
| Breakthrough | `~/Breakthrough/scripts/v3o_arms.sh` | 10-30 16:25 | list | NOT-A-READER | other: launcher, bt_b12 defaults (records3m) | unsets B12_*/BIGMOE_M01 (v3o_arms.sh:26) |
| MoonShots | `~/MoonShots/scratch/run_m29b_z1.sh` | 11-03 22:46 | new | BREAKS-ACTIVE | other: launcher (M-29b) - timing | 11-03 22:46 |
| MoonShots | `~/MoonShots/scratch/run_wed_ticket.sh` | 11-04 09:01 | new | BREAKS-ACTIVE | other: launcher (Wed SSD ticket) - timing | ran Wed 09:01 (MoonShots HANDOVER:636-637) |
| MoonShots | `~/MoonShots/scripts/lossless45.py` | 11-03 17:04 | new | BREAKS-ACTIVE | own raw reads (fixed piece offsets) - values | M-27, no layout check; MoonShots' own rule: route through load_index after the conversion (MoonShots HANDOVER:560-561); 11-03 |
| MoonShots | `~/MoonShots/scripts/run_ssdmicro.sh` | 11-03 18:49 | new | BREAKS-ACTIVE | other: launcher of the ssdmicro harness - timing | quiet-lock wrapper; 11-03 |
| MoonShots | `~/MoonShots/scripts/run_ssdmicro2.sh` | 11-03 22:32 | new | BREAKS-ACTIVE | other: launcher of the ssdmicro harness - timing | quiet-lock wrapper; 11-03 |
| MoonShots | `~/MoonShots/scripts/run_ssdmicro2b.sh` | 11-03 22:44 | new | BREAKS-ACTIVE | other: launcher of the ssdmicro harness - timing | quiet-lock wrapper; 11-03 |
| MoonShots | `~/MoonShots/scripts/run_ssdmicro3.sh` | 11-03 22:39 | new | BREAKS-ACTIVE | other: launcher of the ssdmicro harness - timing | quiet-lock wrapper; 11-03 |
| MoonShots | `~/MoonShots/scripts/run_ssdmicro4.sh` | 11-03 22:43 | new | BREAKS-ACTIVE | other: launcher of the ssdmicro harness - timing | quiet-lock wrapper; 11-03 |
| MoonShots | `~/MoonShots/scripts/ssdmicro.c` | 11-03 18:50 | new | BREAKS-ACTIVE | own raw reads (C pread) - timing | BASE 2162688 hard-coded (ssdmicro.c:38); M-28, 11-03 |
| MoonShots | `~/MoonShots/scripts/ssdmicro2.m` | 11-04 09:00 | new | BREAKS-ACTIVE | own raw reads (pread into Metal slots) - timing | part sizes as row-major records3 (ssdmicro2.m:105,295); M-29, 11-04 09:00 |
| MoonShots | `~/MoonShots/scripts/ssdmicro3.c` | 11-03 22:38 | new | BREAKS-ACTIVE | own raw reads (C pread) - timing | ssdmicro3.c:165; M-30, 11-03 |
| MoonShots | `~/MoonShots/scripts/ssdmicro4.c` | 11-03 22:42 | new | BREAKS-ACTIVE | own raw reads (C pread) - timing | ssdmicro4.c:138; M-31, 11-03 |
| MoonShots | `~/MoonShots/b8mr/run_3b.sh` | 10-25 06:57 | list | BREAKS-STALE | other: launcher of defer/ssd_prefix on records3 - timing | run_3b.sh:9; 10-25 |
| MoonShots | `~/MoonShots/b8mr/run_ssd_b10.sh` | 10-25 07:02 | list | BREAKS-STALE | other: launcher of defer/ssd_prefix - timing | run_ssd_b10.sh:8; 10-25 |
| MoonShots | `~/MoonShots/b8mr/run_ssd_qd2.sh` | 10-25 09:59 | list | BREAKS-STALE | other: launcher of defer/ssd_qd2 - timing | run_ssd_qd2.sh:8; 10-25 |
| MoonShots | `~/MoonShots/defer/run_gpu1.sh` | 10-22 11:20 | list | BREAKS-STALE | other: launcher of defer/ssd_batch - timing | run_gpu1.sh:28; 10-22 |
| MoonShots | `~/MoonShots/defer/ssd_batch.c` | 10-22 08:23 | list | BREAKS-STALE | own raw reads (C pread, F_NOCACHE) - timing | row-major record offsets; 10-22..10-25 |
| MoonShots | `~/MoonShots/defer/ssd_prefix.c` | 10-25 09:49 | list | BREAKS-STALE | own raw reads (C pread, F_NOCACHE) - timing | row-major record offsets; 10-22..10-25 |
| MoonShots | `~/MoonShots/defer/ssd_qd2.c` | 10-25 09:59 | list | BREAKS-STALE | own raw reads (C pread, F_NOCACHE) - timing | row-major record offsets; 10-22..10-25 |
| MoonShots | `~/MoonShots/defer/v3d/bigmoe_v3d/stream.py` | 10-22 08:15 | list | BREAKS-STALE | engine fork (v3d) | no layout guard; load_index -> records3 (tier.py); 10-22 |
| MoonShots | `~/MoonShots/defer/v3d/bigmoe_v3d/tier.py` | 10-22 08:34 | list | BREAKS-STALE | engine fork (v3d) | no layout guard; load_index -> records3 (tier.py); 10-22 |
| MoonShots | `~/MoonShots/defer/v3d/ext/bigmoe_ext.cpp` | 10-22 11:02 | list | BREAKS-STALE | engine fork ext (v3d) | no layout code; 10-22 |
| MoonShots | `~/MoonShots/defer/v3d2/bigmoe_v3d/stream.py` | 10-22 08:15 | list | BREAKS-STALE | engine fork (v3d2) | no layout guard; load_index -> records3 (tier.py); 10-22 |
| MoonShots | `~/MoonShots/defer/v3d2/bigmoe_v3d/tier.py` | 10-22 08:34 | list | BREAKS-STALE | engine fork (v3d2) | no layout guard; load_index -> records3 (tier.py); 10-22 |
| MoonShots | `~/MoonShots/defer/v3d2/ext/bigmoe_ext.cpp` | 10-25 05:18 | list | BREAKS-STALE | engine fork ext (v3d2) | no layout code; 10-22 |
| MoonShots | `~/MoonShots/defer/v3h/bigmoe_v3f/stream.py` | 10-24 12:06 | list | BREAKS-STALE | engine fork (v3h) | no layout guard; load_index -> records3 (tier.py); 10-22 |
| MoonShots | `~/MoonShots/defer/v3h/bigmoe_v3f/tier.py` | 10-22 06:30 | list | BREAKS-STALE | engine fork (v3h) | no layout guard; load_index -> records3 (tier.py); 10-22 |
| MoonShots | `~/MoonShots/defer/v3h/ext/bigmoe_ext.cpp` | 10-25 05:17 | list | BREAKS-STALE | engine fork ext (v3h) | no layout code; 10-22 |
| MoonShots | `~/MoonShots/engine/bench/latency.swift` | 10-21 16:15 | list | BREAKS-STALE | own raw reads (pread of L07 MSB parts) - timing | latency.swift:7,93; 10-21 |
| MoonShots | `~/MoonShots/patches/planes_engine/make_records3.py` | 10-20 17:18 | list | BREAKS-STALE | other: converter records/ -> records3 (+ --verify-only reads records3) | hard-coded row-major layout() (make_records3.py:26); its source records/ is gone; 10-20 |
| MoonShots | `~/MoonShots/patches/planes_engine/make_records4.py` | 10-21 10:28 | list | BREAKS-STALE | own raw reads (records3 -> records4) | make_records4.py:1-2; 10-21 |
| MoonShots | `~/MoonShots/patches/planes_engine/test_rebuild.py` | 10-20 17:40 | list | BREAKS-STALE | own raw reads via make_records3.layout() (row-major offsets) | planes_engine unit test; 10-20/21 |
| MoonShots | `~/MoonShots/patches/planes_engine/test_split.py` | 10-20 23:57 | list | BREAKS-STALE | own raw reads via make_records3.layout() (row-major offsets) | planes_engine unit test; 10-20/21 |
| MoonShots | `~/MoonShots/patches/planes_engine/test_tierglu.py` | 10-21 00:05 | list | BREAKS-STALE | own raw reads via make_records3.layout() (row-major offsets) | planes_engine unit test; 10-20/21 |
| MoonShots | `~/MoonShots/patches/planes_engine/test_tierglu4.py` | 10-21 15:17 | list | BREAKS-STALE | own raw reads via make_records3.layout() (row-major offsets) | planes_engine unit test; 10-20/21 |
| MoonShots | `~/MoonShots/scripts/lsrefit2.py` | 10-21 15:15 | list | BREAKS-STALE | own raw reads via make_records3.layout() | lsrefit2.py:5,9,13; 10-21 |
| MoonShots | `~/MoonShots/scripts/msb_entropy.py` | 10-26 03:17 | list | BREAKS-STALE | own raw reads (fixed piece offsets) - values | records3 path at msb_entropy.py:10, no layout check; 10-26..10-28 |
| MoonShots | `~/MoonShots/scripts/pom24.py` | 10-28 16:58 | list | BREAKS-STALE | own raw reads (fixed piece offsets) - values | records3 path at pom24.py:34, no layout check; 10-26..10-28 |
| MoonShots | `~/MoonShots/scripts/sb_audit.py` | 10-26 08:22 | list | BREAKS-STALE | own raw reads (fixed piece offsets) - values | records3 path at sb_audit.py:15, no layout check; 10-26..10-28 |
| MoonShots | `~/MoonShots/scripts/sbx.py` | 10-28 15:38 | list | BREAKS-STALE | own raw reads (fixed piece offsets) - values | records3 path at sbx.py:15, no layout check; 10-26..10-28 |
| MoonShots | `~/MoonShots/scripts/sbx_confirm.py` | 10-28 15:40 | list | BREAKS-STALE | own raw reads (fixed piece offsets) - values | records3 path at sbx_confirm.py:12, no layout check; 10-26..10-28 |
| MoonShots | `~/MoonShots/scripts/scale_exp.py` | 10-26 03:27 | list | BREAKS-STALE | own raw reads (fixed piece offsets) - values | records3 path at scale_exp.py:8, no layout check; 10-26..10-28 |
| MoonShots | `~/MoonShots/scripts/ssdwake.py` | 11-02 22:08 | list | BREAKS-STALE | own raw reads (F_NOCACHE preads) - timing | ssdwake.py:8,21; 11-02 |
| MoonShots | `~/MoonShots/scripts/w24ec.py` | 10-27 19:50 | list | BREAKS-STALE | own raw reads (read_codes: pread at msb_piece_off) - values | w24ec.py:16,39-47; 10-27; also imported by BigMoE r16_bg.py |
| MoonShots | `~/MoonShots/scripts/xc_entropy.py` | 10-27 20:03 | list | BREAKS-STALE | own raw reads (fixed piece offsets) - values | records3 path at xc_entropy.py:16, no layout check; 10-26..10-28 |
| MoonShots | `~/MoonShots/w24sp/screen.py` | 10-27 19:47 | list | BREAKS-STALE | own raw reads (fixed piece offsets) - values | screen.py:23; 10-27 |
| MoonShots | `~/MoonShots/wide/microbench.py` | 10-22 14:04 | list | BREAKS-STALE | own raw reads (v3 pools from records3) - values | microbench.py:40; 10-22 |
| MoonShots | `~/MoonShots/engine/test_m1b_speed.py` | 10-22 05:02 | list | SAFE | engine | live ~/BigMoE on sys.path (test_m1b_speed.py:26); 10-22 |
| MoonShots | `~/MoonShots/patches/planes_engine/bench_tierglu4.py` | 10-25 06:06 | list | NOT-A-READER | synthetic data | records3 named only as the comparison format |
| MoonShots | `~/MoonShots/patches/planes_engine/test_make_records3.py` | 10-20 17:18 | list | NOT-A-READER | synthetic records | test_make_records3.py:1-3 |
| MoonShots | `~/MoonShots/patches/planes_engine/tierglu4.py` | 10-21 15:17 | list | NOT-A-READER | kernel (pools in memory) | records3 named in a docstring (tierglu4.py:6) |
| MoonShots | `~/MoonShots/scripts/budget24.py` | 10-25 06:01 | list | NOT-A-READER | mentions only (part-size constants) | budget24.py:18 |
| MoonShots | `~/MoonShots/scripts/draft17_check.py` | 11-04 14:10 | new | NOT-A-READER | mentions only (part-size constants) | draft17_check.py:15 |
| MoonShots | `~/MoonShots/scripts/laysz45.py` | 11-01 22:50 | list | NOT-A-READER | reads safetensors + capture npz; part sizes as constants | laysz45.py:35,51-64,86; imported by ~20 pricing scripts, none read records3 |
| MoonShots | `~/MoonShots/scripts/path80_125.py` | 10-21 10:26 | list | NOT-A-READER | mentions only (model) | path80_125.py:3,7 |
| MoonShots | `~/MoonShots/scripts/planes4_sim.py` | 10-21 11:22 | list | NOT-A-READER | mentions only (simulation) | planes4_sim.py:1,6 |
| MoonShots | `~/MoonShots/scripts/pomvs24.py` | 10-29 16:03 | list | NOT-A-READER | mentions only (RAM pricing) | pomvs24.py:4 |
| MoonShots | `~/MoonShots/scripts/r51_busy.py` | 10-29 21:14 | list | NOT-A-READER | reads engine traces; sizes as constants | r51_busy.py:4,7 |
| MoonShots | `~/MoonShots/scripts/to2bit.py` | 11-04 05:53 | new | NOT-A-READER | library (2-bit rebuild on arrays dm_t3 hands it) | to2bit.py:2-7 |
| Fit12 | `~/Fit12/scripts/fit_q.py` | 10-24 02:04 | list | BREAKS-ACTIVE | own raw reads (os.pread at recsz / msb_piece_off / lsb_piece_off; s3/b3/b decoded as bf16) | fit_q.py:15-16,33-50, no layout check. Its read() feeds the TRL store encoder fit_trl_store.py (fit_trl_store.py:23,104), RUNNING now |
| Fit12 | `~/Fit12/scripts/fit_trl_store.py` | 11-04 14:29 | extra | BREAKS-ACTIVE | own raw reads via fit_q.read (indirect: no 'records3' string in the file) | RUNNING (trl_chunks2.sh pid 84909, stop 21:30; trl_relay.sh may restart it to 22:41 / 22:45: Fit12 HANDOVER:917-934). Encodes the TRL store for map screen 7 |
| Fit12 | `~/Fit12/b4/b4bench.swift` | 10-25 07:10 | list | BREAKS-STALE | own raw reads (pread, Metal slots) - timing | b4bench.swift:2,22; 10-25 |
| Fit12 | `~/Fit12/btc/bt_b12.py` | 10-26 18:00 | list | BREAKS-STALE | engine fork via bt_b12 copy (default v3l -> records3) | btc/bt_b12.py:49; 10-26 |
| Fit12 | `~/Fit12/pinned/31e4/bt_b12.py` | 11-01 11:43 | list | BREAKS-STALE | engine fork via bt_b12 copy (default records3m) | records3 only with B12_M01=0 / engine<=v3m; 10-30..11-01 |
| Fit12 | `~/Fit12/pinned/8db7/bt_b12.py` | 11-01 12:41 | list | BREAKS-STALE | engine fork via bt_b12 copy (default v3p + M01 -> records3m) | pinned for T3 (fit_t3.py:30-33); this week's runs use records3m; records3 only with B12_M01=0 / engine<=v3m |
| Fit12 | `~/Fit12/pinned/b23/bt_b12.py` | 10-30 01:22 | list | BREAKS-STALE | engine fork via bt_b12 copy (default records3m) | records3 only with B12_M01=0 / engine<=v3m; 10-30..11-01 |
| Fit12 | `~/Fit12/pinned/b677/bt_b12.py` | 10-30 19:07 | list | BREAKS-STALE | engine fork via bt_b12 copy (default records3m) | records3 only with B12_M01=0 / engine<=v3m; 10-30..11-01 |
| Fit12 | `~/Fit12/pinned/bt_b12.py` | 10-29 19:12 | list | BREAKS-STALE | engine fork via bt_b12 copy (default v3m, no M01 -> records3) | pinned/bt_b12.py ENGINE default v3m; 10-29 |
| Fit12 | `~/Fit12/pinned/cf8b/bt_b12.py` | 11-01 01:05 | list | BREAKS-STALE | engine fork via bt_b12 copy (default records3m) | records3 only with B12_M01=0 / engine<=v3m; 10-30..11-01 |
| Fit12 | `~/Fit12/scripts/b4.sh` | 10-25 05:31 | list | BREAKS-STALE | other: launcher of b4bench - timing | b4.sh:3; 10-25 |
| Fit12 | `~/Fit12/scripts/fit_b3.py` | 10-25 16:34 | extra | BREAKS-STALE | own raw reads via fit_q.read (indirect) | fit_b3.py:131,199,306,486; 10-25 |
| Fit12 | `~/Fit12/scripts/fit_q2.py` | 10-24 14:25 | extra | BREAKS-STALE | own raw reads via fit_q.read (indirect) | fit_q2.py:77; 10-24 |
| Fit12 | `~/Fit12/scripts/fit_q3.py` | 10-24 15:10 | extra | BREAKS-STALE | own raw reads via fit_q.read (indirect) | fit_q3.py:60; 10-24 |
| Fit12 | `~/Fit12/scripts/ssd_burst.py` | 10-24 12:09 | list | BREAKS-STALE | own raw reads (ctypes pread of MSB parts) - timing | ssd_burst.py:10-16; 10-24 |
| Fit12 | `~/Fit12/v3g/bigmoe_v3f/stream.py` | 10-21 16:04 | list | BREAKS-STALE | engine fork (v3g) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3g/bigmoe_v3f/tier.py` | 10-22 06:30 | list | BREAKS-STALE | engine fork (v3g) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3g/ext/bigmoe_ext.cpp` | 10-23 17:39 | list | BREAKS-STALE | engine fork (v3g) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3j/bigmoe_v3f/stream.py` | 10-24 03:50 | list | BREAKS-STALE | engine fork (v3j) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3j/bigmoe_v3f/tier.py` | 10-22 06:30 | list | BREAKS-STALE | engine fork (v3j) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3j/ext/bigmoe_ext.cpp` | 10-24 07:57 | list | BREAKS-STALE | engine fork (v3j) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3k/bigmoe_v3f/stream.py` | 10-24 08:43 | list | BREAKS-STALE | engine fork (v3k) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3k/bigmoe_v3f/tier.py` | 10-22 06:30 | list | BREAKS-STALE | engine fork (v3k) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3k/ext/bigmoe_ext.cpp` | 10-24 11:23 | list | BREAKS-STALE | engine fork (v3k) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3n53/bigmoe_v3f/stream.py` | 10-30 01:48 | list | BREAKS-STALE | engine fork (v3n53) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3n53/bigmoe_v3f/tier.py` | 10-30 01:47 | list | BREAKS-STALE | engine fork (v3n53) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3n53/ext/bigmoe_ext.cpp` | 10-30 01:14 | list | BREAKS-STALE | engine fork (v3n53) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3o53/bigmoe_v3f/stream.py` | 10-30 19:07 | list | BREAKS-STALE | engine fork (v3o53) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3o53/bigmoe_v3f/tier.py` | 10-30 19:07 | list | BREAKS-STALE | engine fork (v3o53) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3o53/ext/bigmoe_ext.cpp` | 10-30 19:07 | list | BREAKS-STALE | engine fork (v3o53) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3p/bigmoe_v3f/stream.py` | 10-24 08:43 | list | BREAKS-STALE | engine fork (v3p) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3p/bigmoe_v3f/tier.py` | 10-22 06:30 | list | BREAKS-STALE | engine fork (v3p) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3p/ext/bigmoe_ext.cpp` | 10-25 22:09 | list | BREAKS-STALE | engine fork (v3p) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3q/bigmoe_v3f/stream.py` | 11-01 15:08 | list | BREAKS-STALE | engine fork (v3q) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3q/bigmoe_v3f/tier.py` | 11-01 15:08 | list | BREAKS-STALE | engine fork (v3q) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3q/ext/bigmoe_ext.cpp` | 11-01 15:08 | list | BREAKS-STALE | engine fork (v3q) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3r/bigmoe_v3f/stream.py` | 11-02 21:21 | list | BREAKS-STALE | engine fork (v3r) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3r/bigmoe_v3f/tier.py` | 11-02 21:42 | list | BREAKS-STALE | engine fork (v3r) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/v3r/ext/bigmoe_ext.cpp` | 11-02 21:21 | list | BREAKS-STALE | engine fork (v3r) | no layout guard; records3 (or records3m with BIGMOE_M01 on n53/o53/q/r); not used this week (bt_b12 loads ~/Breakthrough/<engine>) |
| Fit12 | `~/Fit12/HANDOVER.md.bak_t3r2b` | 11-03 19:28 | new | NOT-A-READER | doc backup | text |
| Fit12 | `~/Fit12/HANDOVER.md.bak_t3r2res` | 11-04 13:22 | new | NOT-A-READER | doc backup | text |
| Fit12 | `~/Fit12/PLAN.md.bak_t3r2b` | 11-03 19:28 | new | NOT-A-READER | doc backup | text |
| Fit12 | `~/Fit12/PLAN.md.bak_t3r2res` | 11-04 13:22 | new | NOT-A-READER | doc backup | text |
| Fit12 | `~/Fit12/scripts/fit_prn.py` | 10-26 17:49 | list | NOT-A-READER | mentions only (byte accounting) | fit_prn.py:6 |
| Fit12 | `~/Fit12/scripts/fit_scr53.py` | 11-02 21:21 | list | NOT-A-READER | engine fork via pinned bt_b12 (v3n + records3m) | fit_scr53.py:1-5; records3m arms |
| Fit12 | `~/Fit12/scripts/fit_t3r_k1.py` | 10-25 20:05 | list | NOT-A-READER | mentions only (part sizes) | fit_t3r_k1.py:7 |
| Fit12 | `~/Fit12/scripts/fit_v3replay.py` | 10-25 16:40 | list | NOT-A-READER | mentions only (bytes per miss) | fit_v3replay.py:10 |
| Fit12 | `~/Fit12/scripts/trl_view.py` | 11-04 14:30 | new | NOT-A-READER | reads the TRL store (fallback via dm_t3) | trl_view.py:5-10 |
| Leap | `~/Leap/scripts/coh_check.py` | 11-03 22:42 | new | BREAKS-ACTIVE | own raw reads (np.memmap) - values | no layout check (coh_check.py:37-38); 11-03 22:42 |
| Leap | `~/Leap/scripts/h1p_screen.py` | 11-03 15:13 | list | BREAKS-ACTIVE | own raw reads (np.fromfile) - values | h1p_screen.py:37; 11-03 15:13 (current-week rule; not in the HANDOVER's open items) |
| Leap | `~/Leap/scripts/merge_check.py` | 11-04 15:52 | new | BREAKS-ACTIVE | own raw reads (np.fromfile whole layer, reshape(E, recsz)) - values | no layout check; NAX6-N microbench / ablation (Leap HANDOVER:109-112), 11-04 |
| Leap | `~/Leap/scripts/native3c_check.py` | 11-03 15:42 | list | BREAKS-ACTIVE | own raw reads (np.memmap) - values | no layout check (native3c_check.py:37); NATIVE-3C, 11-03 15:42 |
| Leap | `~/Leap/scripts/native3c_screen.py` | 11-03 15:37 | list | BREAKS-ACTIVE | own raw reads (np.memmap) - values | native3c_screen.py:35; 11-03 15:37 |
| Leap | `~/Leap/scripts/native_sample.py` | 11-03 15:36 | list | BREAKS-ACTIVE | own raw reads (np.fromfile) - values | writes data/native_sample from records3 assumed row-major (native_sample.py:19-23); 11-03 15:36 |
| Leap | `~/Leap/scripts/nax_ablate.py` | 11-04 15:58 | new | BREAKS-ACTIVE | own raw reads (np.fromfile whole layer, reshape(E, recsz)) - values | no layout check; NAX6-N microbench / ablation (Leap HANDOVER:109-112), 11-04 |
| Leap | `~/Leap/scripts/nax_micro.py` | 11-04 16:00 | new | BREAKS-ACTIVE | own raw reads (np.fromfile whole layer, reshape(E, recsz)) - values | no layout check; NAX6-N microbench / ablation (Leap HANDOVER:109-112), 11-04 |
| Leap | `~/Leap/scripts/nax_occ_probe.py` | 11-04 15:55 | new | BREAKS-ACTIVE | own raw reads (np.fromfile whole layer, reshape(E, recsz)) - values | no layout check; NAX6-N microbench / ablation (Leap HANDOVER:109-112), 11-04 |
| Leap | `~/Leap/scripts/ooo1c_check.py` | 11-04 15:11 | new | BREAKS-ACTIVE | own raw reads (np.memmap of L20 at recsz) - values | no layout check; builds row-major/native/p11 pools in RAM from records3. Leap TODO Thu/Fri: QOM form on 3c-N before Friday's converted store (Leap HANDOVER:54, 186) |
| Leap | `~/Leap/scripts/ooo1c_native_check.py` | 11-04 15:24 | new | BREAKS-ACTIVE | own raw reads (np.memmap of L20 at recsz) - values | no layout check; builds row-major/native/p11 pools in RAM from records3. Leap TODO Thu/Fri: QOM form on 3c-N before Friday's converted store (Leap HANDOVER:54, 186) |
| Leap | `~/Leap/scripts/ooo1c_qom_check.py` | 11-04 17:07 | new | BREAKS-ACTIVE | own raw reads (np.memmap of L20 at recsz) - values | no layout check; builds row-major/native/p11 pools in RAM from records3. Leap TODO Thu/Fri: QOM form on 3c-N before Friday's converted store (Leap HANDOVER:54, 186) |
| Leap | `~/Leap/scripts/ooo1c_qom_native_check.py` | 11-04 17:14 | new | BREAKS-ACTIVE | own raw reads (np.memmap of L20 at recsz) - values | no layout check; builds row-major/native/p11 pools in RAM from records3. Leap TODO Thu/Fri: QOM form on 3c-N before Friday's converted store (Leap HANDOVER:54, 186) |
| Leap | `~/Leap/scripts/ooo1c_wire_check.py` | 11-04 17:15 | new | BREAKS-ACTIVE | own raw reads (np.memmap of L20 at recsz) - values | no layout check; builds row-major/native/p11 pools in RAM from records3. Leap TODO Thu/Fri: QOM form on 3c-N before Friday's converted store (Leap HANDOVER:54, 186) |
| Leap | `~/Leap/scripts/p11.py` | 11-03 22:49 | new | BREAKS-ACTIVE | library (pack/unpack used by the kernels) + self_test raw read | self_test memmaps records3 L20 as row-major and runs to_p11 without a layout check (p11.py:234-262); kernels' pack/unpack are fine |
| Leap | `~/Leap/scripts/p11_check.py` | 11-03 17:29 | new | BREAKS-ACTIVE | own raw reads (np.memmap) - values | no layout check (p11_check.py:46-47); 11-03 17:29 |
| Leap | `~/Leap/scripts/p11pad_check.py` | 11-04 14:55 | new | BREAKS-ACTIVE | own raw reads (np.memmap) - values | no layout check; converts row-major -> p11 pad in RAM (p11pad_check.py:57-58); 11-04 |
| Leap | `~/Leap/scripts/wc_screen.py` | 11-03 17:57 | new | BREAKS-ACTIVE | own raw reads (records3 + native sample) - values | wc_screen.py:116; WHOLE-CHIP screen (Leap HANDOVER:229-232), 11-03 |
| Leap | `~/Leap/data/p11_before_pad.py` | 11-03 22:49 | new | BREAKS-STALE | old p11.py version (self_test raw read) | backup copy, not imported; 11-03 |
| Leap | `~/Leap/data/p11_v1.py` | 11-03 17:31 | new | BREAKS-STALE | old p11.py version (self_test raw read) | backup copy, not imported; 11-03 |
| Leap | `~/Leap/data/p11_v2.py` | 11-03 17:32 | new | BREAKS-STALE | old p11.py version (self_test raw read) | backup copy, not imported; 11-03 |
| Leap | `~/Leap/scripts/ane_probe.py` | 11-02 17:09 | list | BREAKS-STALE | own raw reads in load() (np.fromfile) - values | ane_probe.py:606-612; importers (frspec_walk, ane_v2) take only drafter helpers; 11-02 |
| Leap | `~/Leap/scripts/b10_trellis.py` | 10-25 10:32 | list | BREAKS-STALE | own raw reads (np.fromfile) - values | b10_trellis.py:365-368; 10-25. WATCH: may be revived after map screen 7 (Leap HANDOVER:122-124) |
| Leap | `~/Leap/scripts/h1_hybrid.py` | 11-02 22:03 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/h1i_screen.py` | 11-02 23:06 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/k62f_real.py` | 11-01 06:49 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; K-62 screens 11-01/11-02 |
| Leap | `~/Leap/scripts/k62g_acc_real.py` | 11-01 06:50 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; K-62 screens 11-01/11-02 |
| Leap | `~/Leap/scripts/k62h_gpugrp.py` | 11-01 06:53 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; K-62 screens 11-01/11-02 |
| Leap | `~/Leap/scripts/k62i_nax5.py` | 11-01 07:04 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; K-62 screens 11-01/11-02 |
| Leap | `~/Leap/scripts/k62j_nax6.py` | 11-01 07:08 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; K-62 screens 11-01/11-02 |
| Leap | `~/Leap/scripts/k62o_nax_ooo.py` | 11-02 10:49 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; K-62 screens 11-01/11-02 |
| Leap | `~/Leap/scripts/k62p_floor.py` | 11-02 10:55 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; K-62 screens 11-01/11-02 |
| Leap | `~/Leap/scripts/n7_breakdown.py` | 11-02 21:34 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_gap.py` | 11-02 22:08 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_knock.py` | 11-02 21:39 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_ks.py` | 11-02 22:54 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_native.py` | 11-02 23:28 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_occ.py` | 11-02 21:45 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_regs.py` | 11-02 21:57 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_repack.py` | 11-03 01:52 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_repack_check.py` | 11-03 00:01 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_resid.py` | 11-02 21:51 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_screen.py` | 11-02 21:49 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_stack.py` | 11-02 23:31 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_trace_driver.py` | 11-02 21:35 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7_trim.py` | 11-02 23:10 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/n7f_screen.py` | 11-02 22:37 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/nax_hybrid_check.py` | 11-02 23:33 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/naxc_screen.py` | 11-02 22:49 | list | BREAKS-STALE | own raw reads (np.fromfile whole layer) - values | no layout check; N7 / H1 screens of Mon 11-02 21:34 - Tue 01:52 |
| Leap | `~/Leap/scripts/s11_grouped.py` | 10-25 23:37 | extra | BREAKS-STALE | own raw reads via b10_trellis (indirect) | imports b10_trellis for its 3c pools; 10-25 |
| Leap | `~/Leap/patches/w1_pickup/apply_check/ext/bigmoe_ext.cpp` | 11-03 19:50 | new | SAFE | ext source copy (has the layout code) | patch work copy of BigMoE's ext; not built / loaded |
| Leap | `~/Leap/patches/w1_pickup/base/ext/bigmoe_ext.cpp` | 11-03 19:48 | new | SAFE | ext source copy (has the layout code) | patch work copy of BigMoE's ext; not built / loaded |
| Leap | `~/Leap/patches/w1_pickup/new/ext/bigmoe_ext.cpp` | 11-03 19:50 | new | SAFE | ext source copy (has the layout code) | patch work copy of BigMoE's ext; not built / loaded |
| Leap | `~/Leap/scripts/convert_records_native.py` | 11-03 15:44 | list | SAFE | conversion tool (layout-aware) | todo by layout (convert_records_native.py:29) |
| Leap | `~/Leap/scripts/convert_records_p11.py` | 11-03 22:50 | new | SAFE | the conversion tool (layout-aware) | convert_records_p11.py:35-38 (refuses unknown source layouts), keeps 'rowmajor' (:53-55,103) |
| Leap | `~/Leap/scripts/k90.py` | 11-04 18:35 | new | SAFE | own raw reads + layout check | asserts the layer is row-major (k90.py:28-29, added 18:35); MISSION 33 harness |
| Leap | `~/Leap/scripts/k90_arm.py` | 11-04 18:32 | new | SAFE | own raw reads via k90.load (guarded) | k90.py:28 asserts row-major |
| Leap | `~/Leap/scripts/k90_arm012.py` | 11-04 18:09 | new | SAFE | own raw reads via k90.load (guarded) | k90.py:28 asserts row-major |
| Leap | `~/Leap/scripts/k90_arms.py` | 11-04 18:11 | new | SAFE | own raw reads via k90.load (guarded) | k90.py:28 asserts row-major |
| Leap | `~/Leap/scripts/k90_base.py` | 11-04 17:50 | new | SAFE | own raw reads via k90.load (guarded) | k90.py:28 asserts row-major |
| Leap | `~/Leap/scripts/k90_p1.py` | 11-04 18:14 | new | SAFE | own raw reads via k90.load (guarded) | k90.py:28 asserts row-major |
| Leap | `~/Leap/scripts/k90_probe0.py` | 11-04 18:18 | new | SAFE | own raw reads via k90.load (guarded) | k90.py:28 asserts row-major |
| Leap | `~/Leap/scripts/nat_engine_check.py` | 11-04 14:48 | new | SAFE | engine | load_index(SCR, layout_ok=True) (nat_engine_check.py:41) |
| Leap | `~/Leap/scripts/dense3.py` | 11-04 16:51 | new | NOT-A-READER | no store read | MLX 3-bit layout helpers (dense3.py:7,42) |
| TrainLab | `~/TrainLab/d1i/run_d1i_h.sh` | 10-22 04:28 | list | BREAKS-STALE | other: launcher (D1i stream v3 = raw records3 + GPU rebuild, fork v3f) | run_d1i_h.sh:43; 10-22 |
| TrainLab | `~/TrainLab/d1i/tests/test_planes.py` | 10-22 03:34 | list | BREAKS-STALE | own raw reads + fork (bigmoe_v3f) | reads records3 index + parts (test_planes.py:8-12); 10-22 |
| TrainLab | `~/TrainLab/d1i/v3f/bigmoe_v3f/stream.py` | 10-22 03:28 | list | BREAKS-STALE | engine fork (d1i/v3f) | no layout guard; load_index -> records3 (tier.py:168); 10-22 |
| TrainLab | `~/TrainLab/d1i/v3f/bigmoe_v3f/tier.py` | 10-22 04:48 | list | BREAKS-STALE | engine fork (d1i/v3f) | no layout guard; load_index -> records3 (tier.py:168); 10-22 |
| TrainLab | `~/TrainLab/main24/bigmoe/stream.py` | 10-22 05:48 | list | BREAKS-STALE | engine fork (main24) | no layout guard; load_index -> records3 (tier.py:168); 10-22 |
| TrainLab | `~/TrainLab/main24/bigmoe/tier.py` | 10-22 06:34 | list | BREAKS-STALE | engine fork (main24) | no layout guard; load_index -> records3 (tier.py:168); 10-22 |
| TrainLab | `~/TrainLab/main24min/bigmoe/stream.py` | 10-22 06:06 | list | BREAKS-STALE | engine fork (main24min) | no layout guard; load_index -> records3 (tier.py:168); 10-22 |
| TrainLab | `~/TrainLab/main24min/bigmoe/tier.py` | 10-22 06:06 | list | BREAKS-STALE | engine fork (main24min) | no layout guard; load_index -> records3 (tier.py:168); 10-22 |
| TrainLab | `~/TrainLab/main24min2/bigmoe/stream.py` | 10-22 06:45 | list | BREAKS-STALE | engine fork (main24min2) | no layout guard; load_index -> records3 (tier.py:168); 10-22 |
| TrainLab | `~/TrainLab/main24min2/bigmoe/tier.py` | 10-22 07:21 | list | BREAKS-STALE | engine fork (main24min2) | no layout guard; load_index -> records3 (tier.py:168); 10-22 |
| TrainLab | `~/TrainLab/sticky/budget125_tf.py` | 10-27 21:01 | list | SAFE | engine (live bigmoe.loader) | part sizes as constants (budget125_tf.py:59), loads via load_streamed (:209) |
| FableGod | `~/FableGod/decision_map/dm_ablations.py` | 11-04 16:59 | list | SAFE | other: arm table (records3 via dm_t3) | dm_ablations.py:139-260; arms run by Breakthrough's s67_slot.sh tonight 20:00-23:30, leftovers Thu (s67_slot.sh:10-11) |
| FableGod | `~/FableGod/decision_map/dm_common.py` | 11-04 14:47 | list | SAFE | other: map harness (stock path via live bigmoe.loader; records3 via dm_t3) | dm_common.py:68,324-338 |
| FableGod | `~/FableGod/decision_map/dm_t3.py` | 11-04 14:47 | list | SAFE | own raw reads (preadv) + layout-aware (record_rowmajor via build_nat) | dm_t3.py:11-13,33-48,61-80. CAVEAT: LAYOUT map (dm_t3.py:21) lacks 'native+p11+pad', so it REFUSES the converted store (loud, not silent) |
| FableGod | `~/FableGod/decision_map/mk_r3_overlay.py` | 11-03 17:36 | new | SAFE | other: builds r3_overlay (symlinks + index copy) | asserts records3 is row-major (mk_r3_overlay.py:10). See TRAP: r3_overlay |
| FableGod | `~/FableGod/decision_map/test_t3.py` | 11-03 14:54 | list | SAFE | other: unit test via dm_t3 + live bigmoe | test_t3.py:10,88 |
| FableGod | `~/FableGod/decision_map/test_t3_layout.py` | 11-03 17:36 | new | SAFE | other: layout test via dm_t3 + build_nat | tested on unpadded scratch/r3p11 only (test_t3_layout.py:1-5) |
| FableGod | `~/FableGod/decision_map/dm_cold_masks.py` | 11-03 22:31 | new | NOT-A-READER | mask builder (no store read) | records3 named as the t3 source |
| FableGod | `~/FableGod/decision_map/dm_layout_check.py` | 11-03 17:36 | new | NOT-A-READER | compares map outputs | dm_layout_check.py:2-4 |
| FableGod | `~/FableGod/decision_map/dm_q1u_masks.py` | 11-03 11:01 | list | NOT-A-READER | mask builder (no store read) | records3 named as the t3 source |
| FableGod | `~/FableGod/decision_map/dm_t3all_read.py` | 11-03 01:46 | list | NOT-A-READER | read-out of result JSONs | dm_t3all_read.py:2-5 |
| FableGod | `~/FableGod/decision_map/to2bit.py` | 11-04 06:20 | new | NOT-A-READER | library on dm_t3's arrays | to2bit.py:2-7 |
| FableGod | `~/FableGod/expert_kernel.py` | 11-02 20:56 | list | NOT-A-READER | mentions only | expert_kernel.py:124 (comment); weights via live bigmoe.loader (:209) |
| FableGod | `~/FableGod/fablec/replay_c1.py` | 11-02 22:53 | list | NOT-A-READER | mentions only (part-size constants) | replay_c1.py:29 |
| NewIdeas | `~/NewIdeas/scripts/capacity_11bit.py` | 11-04 13:36 | new | NOT-A-READER | mentions only (byte pricing / masks) | row-major part sizes as constants; no store read |
| NewIdeas | `~/NewIdeas/scripts/dm_oea_masks.py` | 11-04 13:54 | new | NOT-A-READER | mentions only (byte pricing / masks) | row-major part sizes as constants; no store read |
| NewIdeas | `~/NewIdeas/scripts/draft17.py` | 11-04 13:53 | new | NOT-A-READER | mentions only (byte pricing / masks) | row-major part sizes as constants; no store read |
| NewIdeas | `~/NewIdeas/scripts/pool_split.py` | 11-04 13:37 | new | NOT-A-READER | mentions only (byte pricing / masks) | row-major part sizes as constants; no store read |
| NewIdeas | `~/NewIdeas/scripts/ten_x_ledger.py` | 11-03 23:45 | new | NOT-A-READER | mentions only (byte pricing / masks) | row-major part sizes as constants; no store read |
| EvalSuite | `~/EvalSuite/snap/bigmoe-6c64148/bigmoe/stream.py` | 10-30 16:17 | extra | BREAKS-ACTIVE | engine snapshot WITHOUT the layout patch | GUARDED (as tier.py) |
| EvalSuite | `~/EvalSuite/snap/bigmoe-6c64148/bigmoe/tier.py` | 10-29 21:17 | extra | BREAKS-ACTIVE | engine snapshot WITHOUT the layout patch | GUARDED: its only scheduled user, RELAX (runner 40, Wed ~23:50 -> Thu ~07:10), exits 8 on a converted index, and the conversion waits for RELAX_DONE (convert_records3.sh). Older runners 30-39 + relax_smoke.sh also pin it, without that guard |
| EvalSuite | `~/EvalSuite/snap/bigmoe-d0daa68/bigmoe/stream.py` | 10-28 16:37 | extra | BREAKS-STALE | engine snapshot WITHOUT the layout patch | day_runner25.sh (10-28) only |
| EvalSuite | `~/EvalSuite/snap/bigmoe-d0daa68/bigmoe/tier.py` | 10-28 16:37 | extra | BREAKS-STALE | engine snapshot WITHOUT the layout patch | day_runner25.sh (10-28) only |
| EvalSuite | `~/EvalSuite/snap/leap-0002e2e/p11.py` | 11-03 22:29 | new | BREAKS-STALE | frozen p11 library; self_test raw read never run from a snapshot | kernels import pack/unpack only |
| EvalSuite | `~/EvalSuite/snap/leap-2b3fc63/p11.py` | 11-03 17:38 | new | BREAKS-STALE | frozen p11 library; self_test raw read never run from a snapshot | kernels import pack/unpack only |
| EvalSuite | `~/EvalSuite/snap/leap-52cf2c9/p11.py` | 11-04 15:03 | new | BREAKS-STALE | frozen p11 library; self_test raw read never run from a snapshot | kernels import pack/unpack only |
| EvalSuite | `~/EvalSuite/snap/leap-7f43a3f/p11.py` | 11-03 23:00 | new | BREAKS-STALE | frozen p11 library; self_test raw read never run from a snapshot | kernels import pack/unpack only |
| EvalSuite | `~/EvalSuite/snap/leap-90456ce/p11.py` | 11-03 17:40 | new | BREAKS-STALE | frozen p11 library; self_test raw read never run from a snapshot | kernels import pack/unpack only |
| EvalSuite | `~/EvalSuite/snap/leap-c805fd0/p11.py` | 11-04 14:33 | new | BREAKS-STALE | frozen p11 library; self_test raw read never run from a snapshot | kernels import pack/unpack only |
| EvalSuite | `~/EvalSuite/snap/bigmoe-da8e7a6/bigmoe/stream.py` | 11-04 07:54 | new | SAFE | engine snapshot (has the layout patch) | as above |
| EvalSuite | `~/EvalSuite/snap/bigmoe-da8e7a6/bigmoe/tier.py` | 11-04 07:54 | new | SAFE | engine snapshot (has the layout patch) | LAYOUTS / layout_ok in the snapshot; runners 42-44 + r3check.sh use it |
| EvalSuite | `~/EvalSuite/tools/day_runner40.sh` | 11-03 17:23 | new | SAFE | other: RELAX runner on snap/bigmoe-6c64148 (layout-blind) WITH a store guard | converted() = layout key or recsz != 2932736 -> exit 8 before each chunk (day_runner40.sh:51-53,89,104); RELAX Wed ~23:50 -> Thu ~07:10 |
| EvalSuite | `~/EvalSuite/tools/day_runner42.sh` | 11-04 18:00 | new | SAFE | other: runner on snap/bigmoe-da8e7a6 with store guards | converted() / full_converted() checks; exit 8 on a store change (day_runner42.sh) |
| EvalSuite | `~/EvalSuite/tools/day_runner43.sh` | 11-04 18:01 | new | SAFE | other: runner on snap/bigmoe-da8e7a6 with store guards | converted() / full_converted() checks; exit 8 on a store change (day_runner43.sh) |
| EvalSuite | `~/EvalSuite/tools/day_runner44.sh` | 11-04 18:04 | new | SAFE | other: runner on snap/bigmoe-da8e7a6 with store guards | converted() / full_converted() checks; exit 8 on a store change (day_runner44.sh) |
| EvalSuite | `~/EvalSuite/tools/r3check.sh` | 11-04 15:17 | new | SAFE | other: conversion check, store-state aware | refuses unless the store is as wanted (r3check.sh:27,35) |
| EvalSuite | `~/EvalSuite/snap/bigmoe-da8e7a6/SOURCE` | 11-04 07:54 | new | NOT-A-READER | provenance note | text |
| EvalSuite | `~/EvalSuite/snap/leap-0002e2e/SOURCE` | 11-03 22:29 | new | NOT-A-READER | provenance note | text |
| EvalSuite | `~/EvalSuite/snap/leap-2b3fc63/SOURCE` | 11-03 17:38 | new | NOT-A-READER | provenance note | text |
| EvalSuite | `~/EvalSuite/snap/leap-7f43a3f/SOURCE` | 11-03 23:00 | new | NOT-A-READER | provenance note | text |
| EvalSuite | `~/EvalSuite/snap/leap-90456ce/SOURCE` | 11-03 17:40 | new | NOT-A-READER | provenance note | text |
| EvalSuite | `~/EvalSuite/tools/day_runner41.sh` | 11-04 18:04 | new | NOT-A-READER | other: T0 runner, records3m only | T0 reads records3m (bt_b12 v3p + BIGMOE_M01=1); guard watches records3m (day_runner41.sh:13-14,30,43,81) |
| EvalSuite | `~/EvalSuite/tools/snap_leap.py` | 11-03 17:38 | new | NOT-A-READER | snapshot tool | copies Leap files |
| SpecDecode | `~/SpecDecode/spec125.py` | 10-27 21:02 | list | SAFE | engine (live bigmoe) | spec125.py:239,284,423 |
