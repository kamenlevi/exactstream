# records3 in-place conversion: run-ready checklist (Leap, Wed 2026-11-04 18:4x)

**What:** `models/bare/records3` goes from row-major to native+p11+pad, in place (~3 min + ~1.5 min of md5s), with
`--inverse` ready.

**When:** Thu ~10:05-10:45, right after EvalSuite's NAX6 Part 1 PASS (the orchestrator 17:5x / 18:0x). The orchestrator
gives the 1 h heads-up.

**Who runs it:** BigMoE, by hand:

    CLAUDE_PID=<BigMoE pid> CONV_HEADSUP=<epoch> bash ~/BigMoE/scripts/convert_records3.sh

Plan: ~/BigMoE/docs/PLAN_records3_convert.md. Leap prepares and verifies.

## 1. Tools and engine (verified 18:4x)
- **Converter:** ~/Leap/scripts/convert_records_p11.py at 7624eb0 (P11PAD).
  - BigMoE's `--pad` comparison: its output == BigMoE's reference byte for byte, and index entries match field for field.
  - The script checks that ~/Leap is at 7624eb0 or later.
- **Wrapper:** ~/BigMoE/scripts/convert_records3.sh, tested Tue 17:45 in test mode (3 layers):
  - output == the reference;
  - `--inverse --md5`: files == records3 and the index entries are restored exactly;
  - re-convert: md5 == the first conversion.
- **Engine:** the layout patch is live in bigmoe/ (BigMoE d0030a1); ext/build was rebuilt Nov 4 07:51.
  - bigmoe.tier.LAYOUTS includes native+p11+pad.
  - load_index refuses a converted layer without layout_ok.
  - Zero-copy reads for the padded format: 172eba5.
- **Kernels:** 3c-N and NAX6-N read p11 / pad pools (bitwise on full layers 20 / 44).
  - ref2 routes 3c per layer.
  - NAX6 needs CFG nax_native (store-level, all 48 layers); ref2 raises without it.

## 2. The script's own guards (it refuses otherwise)
- ~/EvalSuite/results/RELAX_DONE AND ~/EvalSuite/results/T0_DONE exist.
  - **At 18:4x neither exists.** BigMoE's notes expect RELAX_DONE ~06:40 Thu and T0's leftover after it.
  - **To confirm with EvalSuite:** its moved morning order finishes both before ~10:05; otherwise the script refuses.
- The 1 h heads-up went out at least 60 min earlier (CONV_HEADSUP epoch).
- BigMoE holds a quiet lock with >= 10 min left.
- Nothing is open in records3 (lsof).
- Free disk >= 112 GB.

## 3. Disk
- 136 GB free at 18:4x; the tool refuses under 112 GB.
- Peak extra during the run: one layer's temp, ~1.4 GB. The store shrinks by ~3.5 GB.
- **Keep >= ~115 GB free until the conversion.** Watch tonight's writers: Fit12's TRL store, MTP Head's P-CAP, the night
  runs' logs and traces.

## 4. md5 before and after
- Before: the script writes data/records3_md5_rowmajor.txt for all 48 row-major files.
- After: data/records3_md5_p11.txt. The A/B night's re-conversion must match it.
- Then the index check: 48 layers native+p11+pad, each with its "rowmajor" entry kept. **Never drop the rowmajor key:
  --inverse needs it.**

## 5. The --inverse path

    ~/BigMoE/.venv/bin/python ~/Leap/scripts/convert_records_p11.py ~/BigMoE/models/bare/records3 --inverse --md5 ~/BigMoE/data/records3_md5_rowmajor.txt --yes

- It verifies the md5s against the "before" list.
- It is resumable layer by layer; a mixed store is safe because the index is per layer.
- Tested on 3 scratch layers Tue (BigMoE).

## 6. After the conversion
- EvalSuite check: its 3-row bridge vs the row-major Part 1 rows, on snap/leap-52cf2c9 + tierglu_nax6n.py, with flags
  tier_nax_ooo + nax_native.
- Engine smoke: the residency check + a short REF2-T decode, token for token vs the pre-conversion reference.
- Tell every session (peer say + the orchestrator).

## STATUS 20:0x
- DECIDED (the orchestrator, 20:0x): option (a). "D4X: DIFFERS (HIGHER)" counts as GO, like PASS; Kamen keeps the veto
  tonight. BigMoE aligned its guard (8618a37): convert_records3.sh converts on PASS or DIFFERS (HIGHER); FAIL / VOID /
  none still refuse. That matches EvalSuite's runners 43 / 44. No open blockers on Leap's side.
## STATUS 19:2x (superseded)
- B1: FIXED by BigMoE 8e8bdc4, verified against EvalSuite's day_runner42.sh.
  - T0_DONE is gone. The script needs NAX6_DONE == "D4X: PASS" (d4x.py prints exactly that; runner 42 writes it),
    CHECK1_DONE starting "check 1 rc 0 " (runner 42 writes "check 1 rc $c1 (HH:MM)"), and RELAX_DONE.
  - The script quarantines model_mix's index (.rowmajor-stale) and lists foreign overlays as "OVERLAYS TO FIX".
- OPEN DECISION (the orchestrator, asked 19:2x): on "D4X: DIFFERS (HIGHER)" the script refuses, but EvalSuite's
  runners 43 / 44 treat DIFFERS as go. They would wait for a converted store until 14:00 / ~16:15. Align (a) or (b).
## STATUS 19:1x (superseded)
- B2 (dm_t3): FIXED by Breakthrough e4d2c82, tested 25/25 on BigMoE's padded copy. It installs after tonight's map batch.
- r3_overlay: moved aside at the same time (renamed .STALE-before-records3-conversion).
- Readers now guarded: Fit12 fit_q.read (refuses a non-row-major layout; covers fit_trl_store / fit_b3 / fit_q2 / fit_q3);
  MoonShots ca49546 (5 readers + 7 launchers); Leap 68d2346 (18).
- Open: model_mix (BigMoE) to delete or re-index after the conversion.

## 0. BLOCKERS (verified 18:5x against the files)
- **B1. The script would refuse Thu ~10:55.** convert_records3.sh:34 requires ~/EvalSuite/results/T0_DONE.
  - EvalSuite's order (its HANDOVER, Wed 18:05) now runs T0 AFTER the conversion: after runner 44, ~15:50 -> Fri 00:15.
  - T0 reads only records3m (bt_b12 on v3p, BIGMOE_M01=1), so the T0 guard is obsolete.
  - **BigMoE: drop the T0_DONE line, or replace it with runner 42's PASS flag.** RELAX_DONE stays: RELAX ends ~07:10.
- **B2. Breakthrough's map stops after the conversion.** ~/FableGod/decision_map/dm_t3.py:21 LAYOUT has row-major /
  native / native+p11 but not "native+p11+pad", so LAYOUT[lay] (line 43) raises KeyError on every t3 arm. That hits
  Thu's map leftovers and FLOOR night 2.
  - Fix: add the name, then test on a padded copy (DM_R3 points at another store). So far it was tested on unpadded
    copies only.

## 7. Readers (385 files: the 238 listed + 138 changed since Tue 15:44 + 9 traced; data/records3_readers_triage.md)
Counts: 34 BREAKS-ACTIVE, 172 BREAKS-STALE, 121 SAFE, 58 not readers. No launchd / cron job starts a reader.
- **Fit12 (running now):** fit_trl_store.py reads every expert through fit_q.read (scripts/fit_q.py:33-50: a raw pread at
  index offsets, no layout check). It stops by 22:45 tonight; **do not restart it after the conversion** until fit_q
  checks the layout or reads through the engine.
- **Leap (18 check scripts):** guarded 68d2346. They now refuse a converted store (k90.py since 2e759d5).
- **MoonShots (12, nothing armed):**
  - lossless45.py decodes values;
  - ssdmicro / ssdmicro2-4 have row-major part sizes hard-coded;
  - 7 launchers.
  - Do not run them on records3 after Thursday without a layout check.
- **EvalSuite:** no action. Its frozen bigmoe-6c64148 is layout-blind but runs RELAX only; runner 40 exits on a
  converted index; and the script needs RELAX_DONE.
- **Silent traps: two overlay stores with their own index.json** keep 45 symlinked layers marked row-major, pointing at
  the real records3 files:
  - ~/BigMoE/scratch/model_mix/records3 (BigMoE);
  - ~/FableGod/decision_map/r3_overlay (the decision map).
  After the conversion, anything pointed at them decodes converted bytes as row-major, with no error. **Delete or
  re-index both right after the conversion.**
- Unsure (triage f): bt_b12.py copies (stale modes); h1p_screen (active by date only, now guarded); b10_trellis
  (stale).

## 8. Leap's own raw readers
k90.py now refuses a converted layer (2e759d5). The older harnesses (merge_check, nax_*, n7_*, k62*, ...) read row-major
bytes: row-major only after Thursday (they predate the conversion; no live runner uses them).
