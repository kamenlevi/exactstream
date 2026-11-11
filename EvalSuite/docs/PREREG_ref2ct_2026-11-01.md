# REF2c-T: regen + same-scores vs REF2-T, then R-56e (live loop at full length) (pre-registered Sun 2026-11-01 11:5x)

Written before any REF2c-T row.
- **Request:** NewIdeas 07:4x: "tonight: the REF2c regen + one same-scores test (REF2c-T), when Leap hands over the frozen
  bundle (att_single + padded in_proj)". NewIdeas' night order: BigMoE C45 -> MTP Head D-23 -> EvalSuite REF2c-T.
- **Freeze:** Leap 11:5x, "REF2c FROZEN at Leap commit 91cd316": `ref2.CFG["att_single"] = True` and
  `ref2.CFG["gdn_inproj"] = True` on REF2-T as it is; FATT16 left out; MTPLX_FUSED_GDN_INPROJ kept unset.
> (a quote from the project owner, kept private)
  same. A PASS lets NewIdeas make REF2c-T the 45 GB exact mode. Leap expects ~-6 ms per check: att_single ~-5,
  in_proj ~-0.5 to -0.9.

## What runs
- **Variant `ref2ct`** (model.VARIANTS), decoded by `ref2.generate` (its own AR), greedy, at C 192.
  - Code: `snap/leap-91cd316`, made with tools/snap_leap.py from the commit (git show). It holds ref2, leapk, qmm16,
    qmmg25, gmr24 and ms_gdn/. Every module loads from the snapshot (model._SnapFinder; import check: none from ~/Leap).
  - Switches: ref2_cfg {att_single: True, gdn_inproj: True}, set before apply() and checked after self_test.
  - On REF2-T's tier engine: BIGMOE_TIER=4433333333, BIGMOE_TIER_GLU=3c, BIGMOE_TIER_CL=112, BIGMOE_TIER_COMBINE
    unset (asserted). Frozen BigMoE `snap/bigmoe-6c64148`. MoonShots' planes_engine (3c) pinned by sha.
- **What differs from REF2-T's code (snap/leap-faff8e5):**
  - leapk / qmm16 / qmmg25 / gmr24: byte-identical.
  - ref2.py: Leap's commits since faff8e5. The two switches; GDNX (CFG gdn_exact, adopted 129c86e, on by default here;
    Leap R-61a: bitwise == the eager GDN chain); the tier_nax4 switch (off); docstrings.
  - Expected bit changes (Leap): attention past 512 keys (att_single); the GDN b and a projections (0.1-0.8 % of their
    outputs, reaching ~45 % of a GDN layer's outputs in the last bits). Every 16-row AR step changes, so nothing is copied.
- **INV evidence (Leap):**
  - R-62b: att_single at SDPA level.
  - R-71: in_proj on real packs: 16-row == 1-row; a row unchanged with 15 other rows; switch off == the separate
    modules.
  - R-71b: a real GDN layer under REF2 + GDNX: one 16-row call == 16 one-row calls after a 37-token prefill, y and
    states bitwise.
- **Engagement:** decode stops a row if a REF2c switch never ran in it. Every token after the first is a 16-row AR step,
  so both counts must be > 0. Rows record ref2.counts(); the provenance records ref2.CFG as run and the ms_gdn shas.

## Part 1: same scores (the verdict)
- **Rows:** all 210 core rows: math 50, IFEval 100, HumanEval+ 60.
- **PASS (decision 4's rule):** in each of math, IFEval (strict, prompt level) and HumanEval+ (pass@1), the paired delta
  REF2c-T − REF2-T (`results/ref2t`) is above −5 points AND the exact two-sided McNemar p is at least 0.05.
  - FAIL if any set fails. A gain with p < 0.05 is "differs (higher)" (NewIdeas' call).
  - Command: `evalsuite report --variant ref2ct --decode --ref ref2t`; `tools/d4.py` applies the rule to it.
- **Information:**
  - the suite's full standard table (loops, truncations, length, hesitation);
  - the same report vs `ref2b` and vs `ref.r2`;
  - token-identical rows vs REF2-T;
  - AR tok/s and lifetime footprint.

## Part 2: R-56e, the live loop at full length (only after a Part 1 PASS)
- **Arm `ref2cto_gate16`:** REF2c-T on BigMoE's gate (MAXK 160, ext build_next) with BIGMOE_TIER_COMBINE=1 and BIGMOE_OOO=1
  (the 45 GB default's form since R-56d). R-56's 36 prompts, at C 192, against its own AR (computed in the same process,
  OOO on).
- **PASS (all four):**
  1. Loop = its own AR: on all 36 prompts every token and every kept 16-row check row is bitwise equal to the AR's.
  2. AR = results/ref2ct (Part 1's rows, made without OOO) on all 36 prompts; a prompt that differs is VOID.
  3. OOO ran every tier call of the AR and of the loop (tier calls == OOO calls > 0), every prompt.
  4. Both REF2c switches ran in the AR and in the loop of every prompt. Leap's asserts stop the process otherwise: at
     AR step 1 and at check 1.
- Verdict: `tools/r56b.py ref2cto_gate16` -> results/r56e.txt (FAIL / VOID / NOT ENGAGED / INCOMPLETE / PASS).
- **Not covered:** check widths 2-15 (this harness checks 16 rows; R-69 and Leap's INV cover them at tensor level).

## Runner 36 (tools/day_runner36.sh)
- Starts queueing at 23:05, behind the night's earlier tickets (FIFO).
- One quiet ticket per chunk (decode chunks ~12 min, live16b processes ~9 min), re-queued at once.
- NO-SWAP >= peak + 4 GB after the grant: `--need 34` (R-58's decode peak 32.3 GB, R-56d's live16b peak 33.6 GB; ~1 GB
  transient for the in_proj build). Swap guard, PAUSE and lid gates.
- Code pin: both snapshots incl. ms_gdn, planes_engine, evalsuite, Leap's drafter, MLX / mtplx. A change stops it.
- Deadline Mon 2026-11-02 10:00.
- Sequence: Part 1 -> reports + d4 -> results/REF2CT_DONE; if PASS, Part 2 -> results/r56e.txt + results/R56E_DONE.
- ~79k tokens (~2 h GPU) + ~1 h.
- **Reporting:** to NewIdeas (adoption is theirs), Leap and BigMoE. A Part 1 FAIL goes to NewIdeas with the discordant
  items.

## Amendment 1 (Sun 12:4x, before any REF2c-T row)
- Runner 36's first launch (11:59, pid 24879) was stopped by pid in its wait loop, before any lock or row.
  - Why: Breakthrough's PMIN12 request needed two 12 GB arms in evalsuite/ (model.py / cli.py, the fold family only), and
    the runner's pin covers evalsuite/.
  - The ref2ct path is unchanged. Runner 36 is relaunched with a new pin; no condition of this pre-registration changes.
- PMIN12 (runner 37, docs/PREREG_pmin12_2026-11-01.md) uses shared locks by day until 22:45 at the latest and goes on
  only after runner 36 has exited. That way no lock of this session moves while runner 36's ticket waits.

## Amendment 2 (Sun 21:2x, before any REF2c-T row)
- Runner 36 (pid 43408) was stopped by pid in its wait loop at 21:24, before any lock or row. It was relaunched at 21:25
  as pid 48937 with one change: the deadline moves from Mon 10:00 to **Mon 2026-11-02 18:00**. The code pin is
  unchanged (015eb41be555).
  - Why: Kamen 20:3x lets big runs continue through Monday with no day hold. Kamen 20:2x moved MTP Head's DLOG (up to
    2 h) ahead of runner 36. Runner 36 takes one FIFO ticket per chunk, so its chunks can interleave with Leap's runner
    62 and NewIdeas' after-queue items.
> (a quote from the project owner, kept private)
  only with non-timed runs if the memory gate passes). It starts only 5 min after a `decode ref2ct` grant and exits
  within 30 s of each unlock. It stops at +1 GB swap and stays off during Part 2.
  - Outputs (tokens) do not depend on another process. Part 1's AR tok/s is information only and is labelled as shared.
- No PASS condition changes.

## Amendment 3 (Mon 04:1x, after 15 of 210 Part 1 rows; no score of a REF2c-T row looked at)
- Runner 36 now runs **60-min chunks** for Part 1 and Part 2 (`--chunk-minutes 60 --minutes 80`, was 12 / 17 and 9 / 18).
  It stops at 90 min before the deadline (was 25), and the deadline stays Mon 18:00. One quiet ticket per chunk, as before.
  - Why: each chunk takes one FIFO ticket, so 12-min chunks interleaved with others' 30-60-min turns. Only 2 chunks
    (15 rows) ran from 23:05 to 04:07.
  - Memory: REF2-T's decode peak is flat over rows per process (R-58: 31.6-32.3 GB for 8-34 rows; tonight 31.8 / 32.3).
- The swap: runner 36 (pid 48937) was stopped by pid at 04:08, inside its own lock (granted 04:07:38). Its cli 38996
  keeps that chunk. The new runner (pid 99782, same pin 015eb41be555) starts after cli 38996 ends.
- No PASS condition changes; chunk length does not touch the tokens (each row is decoded alone, from a fresh state).
