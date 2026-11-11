# PREREG: plain FAST on long agent turns, against the agent bar (Fri 2026-11-06 08:09; before any row)

Written before any row of this check.
- **Request:** the orchestrator, Fri 08:0x-08:2x, after RELAX's agentlong finding (docs/RESULT_relax_drift_2026-11-06.md):
  - B (Q1-UNION gate + copy + comment rule + HC8) commits 20 blunders in 8,842 tokens of long agent writes (22.6 per
    10K, 16 inside the written code). Mission 7's agent bar is "none".
  - The orchestrator's call: B ships for plain code answers only. Agent turns run without the RELAX gate until they pass
    the agent bar.
  - This check decides how agent turns run: plain FAST, or exact.
- **The orchestrator accepted the line as written below (Fri 08:3x).** It goes before the code-only drift top-up.

## What runs
- **Mode:** today's plain FAST = QOM-KEEP + DRAFT17, no RELAX gate, on the exact base that ships:
  - NAX6-N if BigMoE's FAST-NAX A/B (Fri ~08:35) passes, else REF2-T.
  - It runs through BigMoE's `scripts/r57_screen_fast.py`.
  - BigMoE supplies the env, the commit and the exact base. They are added here, under "Recipe", before any row. The
    runner pins those files and stops if they change.
- **Rows:** dec_agentlong's 17 write / edit turns (data/items/dec_agentlong.jsonl, sha256 ea6211b4933c; built Thu by
  tools/drift_build.py from dec_agent):
  - real tool-call answers with 1.6-12K-token contexts;
  - max_tokens 1024, greedy.
- **Two prompt orders,** forward (key order) and reverse, each in its own process. QOM-KEEP's keep set depends on the
  cache state, so per docs/PREREG_TEMPLATE_cachedep.md the certificate needs two histories. 34 rows in all.

## Scoring: the mode's exact reference along FAST's own tokens
- A new stage replays each FAST row's committed tokens through the base's exact reference (`ref2tnaxn` = NAX6-N on the
  converted store, or `ref2tf0` = REF2-T):
  - the prompt is prefilled as replay.py does;
  - the committed tokens then go through in 16-row verify windows, all rows kept.
- This is the exact verify path. Its rows equal the reference's own one-token AR bit for bit: R-56c for REF2-T, NAX6
  Part 2 for NAX6-N.
- Per committed token it records:
  - the reference's probability p of that token;
  - the reference's argmax.
- **Self-check before any FAST row is scored:** the stage scores three of an exact run's own rows; every token must be
  the reference's argmax. The rows are results/ref2tnax's HumanEval/27 and ifeval/3409 for NAX6-N, or
  results/relax_control's agent rows for REF2-T. Else VOID.

## The line
- **PASS:** 0 blunders, i.e. no committed token with the reference's p < 0.01, over all 34 rows (both orders).
- **FAIL:** 1 or more blunders.
- **VOID:**
  - the self-check fails;
  - FAST's own counters show QOM-KEEP or DRAFT17 not engaged;
  - or a row is missing.
- **What it means:**
  - PASS: agent turns run plain FAST, without the RELAX gate.
  - FAIL: agent turns run exact.

## Information (no lines)
- The deviation rate (committed token != the reference's argmax), per order and per row.
- Where any blunder sits: the first tool-call parameter (the command / code) or later text (the description), as for
  B.
- Token agreement between the two orders. QOM-KEEP's cache dependence shows here.
- The first divergence from the exact control: the reference's own AR on the 17 turns, decoded when a slot allows.
- On the 11 edit turns: whether a copied span breaks off early.

## Runs and cost
- **GPU** (rough, from RELAX's measured agent rows; ~90 s prefill per turn):
  - FAST rows: 2 x 17 turns, ~50 min.
  - The scoring replays: 34 rows, ~85 min.
  - About 2.5 h on a shared lock, MEM ~36-45 GB, no timing line.
- **Slot:** today by day if the build is ready by ~12:00, ending by 15:00 (the orchestrator). Otherwise tonight, first.
- **Code:** the new scoring stage in evalsuite/, the runner, and BigMoE's script under a pin.

## Recipe (BigMoE Fri 08:3x; the orchestrator Fri 08:3x: the base is NAX6-N; added 08:17, before any row)
- **Screen:** ~/BigMoE/scripts/r57_screen_fast.py at commit 6c08636, which adds two things:
  - SETOF knows `agentlong`;
  - R57_QOM_SETS turns QOM's FAST rule on for agent turns, which before ran exact.
- **Pin:** BigMoE's tracked scripts/ and bigmoe/ must equal 6c08636 (`git diff --quiet 6c08636`) with nothing untracked
  there, and ext/build_qom's .so sha256 must be ff69fa24a745. This is checked before each FAST process.
  - A wrapper (tools/r57_wrap.py) records every module the process loaded, with its sha256, in
    results/agentfast/<tag>_modules.json.
- **Env, as BigMoE's c45 runner:**
  ```
  R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_REFTOK= BIGMOE_EXT_BUILD=build_qom R57_FAST=QOM R57_D17=1 R57_D7B=code
  R57_HC8=1 REF2_HC8_FAST=1 R57_DNR=1 R57_HCB=1 REF2_SMALL_NATIVE=1 R57_D7BNAX=1 REF2_OOO_1C=1
  R57_QOM_SETS=humaneval,ifeval,multi,agentlong R57_NAX=1 REF2_NAX_1CQ=1
  ```
  - C 224, CAP 1024.
  - R57_RELAX_*, R57_INERT, R57_NOCOM, R57_QOM_SPLIT and EVALSUITE_BIGMOE_CODE are unset. With EVALSUITE_BIGMOE_CODE
    set, evalsuite.env would put another bigmoe first on the path.
- **Rows:** tools/agentfast.py rows turns <tag>_tokens.json into results/agentfast_fwd/ and agentfast_rev/ decode rows.
  Each row is cut after its first EOS or at the item's 1024.
- **Engaged (else VOID):**
  - every agent row's run has QOM optional uses (pf.qom_opt > 0 in <tag>_runs.json);
  - draft17 is in the module manifest.
- **The scoring reference:** `ref2tnaxn` (NAX6-N on the converted store, snap/bigmoe-da8e7a6, the OOO env as runner
  44), through the new stage `tfwin`.
  - The self-check keys: results/ref2tnax's humaneval/HumanEval/27, ifeval/3409 and gsm8k/707.
- **Smoke first** (NewIdeas' GPU order: the first gap, after MTP Head's PORT when memory is short): FAST on
  agentlong/edit10 (the shortest prompt), then tfwin (the self-check plus that row).
  - A smoke failure stops the run.
  - Its row is information only; it is not one of the 34.
- **Then the check:** FAST forward (key order), FAST reverse, then both scorings, then tools/agentfast.py report.
  - It starts by 12:30 (NewIdeas), else Saturday by day.
  - No step starts unless it can end by 15:00.

## Status Fri 09:01: runner 47's partial run, BigMoE's fix, the re-pin (before the remaining rows)
- **Runner 47 (08:19-08:57):**
  - The smoke passed. tfwin's self-check held; edit10 gave 93 tokens, 0 deviations, 0 blunders. The smoke row is
    information only.
  - FAST forward's first process kept edit00-05. The screen stops a process at HARD = 600 s of running time.
  - FAST reverse's first process crashed. A run that started just before HARD got 0 checks, and the stats raised
    IndexError. The tokens are dumped at the end, so all 7 of its prompts were lost.
- **BigMoE's fix, 9c82d1d (Fri 09:0x):** r57_screen_fast.py gains 6 lines and nothing else. A 0-check run's stats read
  nan, and the tokens are written after every run. `git diff 6c08636 9c82d1d` verified it; the other two changed files
  are new shell scripts the screen does not run.
  - So the run path is 6c08636's, and edit00-05 stand.
  - The pin moves to 9c82d1d for the remaining rows.
- **The remaining rows (runner 48):** each order's prompts run in chunks of 2-3 per process (well inside 600 s), their
  rows appended. A prompt that a chunk misses runs alone. The two orders keep different histories:
  - forward: the first process's 6 prompts, then chunks in key order;
  - reverse: chunks from write16 down.

## Status Fri 09:28: a provenance gap, the fix (before any further row)
- **What happened:**
  - Runner 48 ran the forward order to 17/17, then stopped (exit 7) because BigMoE committed fce0e55.
  - fce0e55 adds NEVER-WAIT behind R57_NEVERWAIT=1. It is off here; every line is under `if NW:` except a `nw_ = []`
    stats field, so the run path is unchanged.
- **The gap, found in the processes' module manifests:** each FAST process also loads live code from Leap, TrainLab,
  Breakthrough and MoonShots, which BigMoE's pin did not cover.
  - Leap's ref2.py and d7b_nax.py changed between forward chunks 2 and 3 (Leap 03a63c5, 09:12: DN-KS v2 + D17-NAX v2,
    off by default).
  - So edit00-10 ran on Leap 7ebf2d7 + an uncommitted d7b_nax, and write11-16 ran on 03a63c5.
  - ref2.py's diff is behind REF2_DN_KS (unset here; the default path is unchanged).
  - d7b_nax's earlier version is not in git, so its default path cannot be diffed. QOM's keep decisions read the
    drafter's confidence, so drafter code can move FAST tokens.
- **The fix:**
  - edit00-10's forward rows are set aside as information (results/agentfast_fwd_v1/).
  - Both orders run entirely on one code version: the reference manifest results/agentfast/es_af_fwd5_modules.json
    (Leap 03a63c5, BigMoE 9c82d1d's modules = fce0e55's, TrainLab / Breakthrough / MoonShots as listed), plus
    fce0e55's screen.
  - Every external module of that manifest and the screen is pinned before each process (exit 7 on a change).
  - Each process's own manifest must equal the reference (else VOID).
  - Forward: edit00-10 rerun in chunks. write11-16 stand (run on 03a63c5). Then reverse, all 17.

## Status Fri 09:38: the outside code frozen (before any further row)
- **What happened:** the relaunched runner 48 voided its first chunk (09:34). Leap committed 586c892 (NEVER-WAIT in
  ref2's one-call) at 09:29, between the pin check and the process's import, and the manifest check caught it. Its
  tokens are not used.
- **The fix, a freeze:** tools/freeze_r57.py built snap/r57frz-af1 from the reference manifest's versions:
  - Leap 03a63c5 and BigMoE fce0e55 by `git archive`, plus the build_qom .so;
  - TrainLab sticky + mtp125.py and Breakthrough keepset, copied.
  - Code-path strings in the frozen screen and the frozen modules point at the copies; data paths stay live.
  - All 33 frozen modules equal the reference exactly, or after undoing the rewritten path strings.
  - Still live, and pinned: MoonShots' tierglu.py (BigMoE's loader adds its path), and EvalSuite's own code.
- **Every FAST process now runs the frozen screen,** with EVALSUITE_BIGMOE_CODE pointing at the frozen BigMoE.
  - Before each process: the frozen tree and the pinned live modules must be unchanged (exit 7).
  - After it: every outside module it loaded must come from the freeze with the recorded sha, with the same module set
    as the reference (else VOID).

