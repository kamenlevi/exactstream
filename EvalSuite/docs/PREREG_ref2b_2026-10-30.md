# REF2b reference rows + decision 4's same-scores test (pre-registered Fri 2026-10-30 17:5x)

Written before any REF2b row.

## Why
- **REF2b changes the reference past 512 keys.** Leap's REF2b (`~/Leap` c4b6be9) adds key buckets: every 16-row
  attention call with N ≥ 512 keys runs at the next multiple of 512.
  - It fixes R-56's break: R-40 STEP 4c PASS, with geometry/702 160/160 and HumanEval/68 220/220 rows bitwise against
    the bucketed AR.
  - Leap's merge check (KBTEST, runner 42, Fri 17:01, rc 0) passed: the merged ref2.py equals the 4c module at N
    100..1100.
  - REF2b also changes REF2's own AR bits past 512 keys, so `results/ref2` is not REF2b's reference there.
- **NewIdeas' routing** (Leap's HANDOVER; NewIdeas 07:52 and 16:54):
  1. EvalSuite regenerates the reference rows on REF2b (this document);
  2. re-runs decision 4's same-scores test, which is inside Kamen's approval (this document);
  3. then re-runs R-56 on REF2b (a separate pre-registration, two arms: REF2b, and REF2b + GATE16).

## Variant `ref2b`
- Model `ref` with REF2b applied at load. The code is copied into `snap/leap-faff8e5`, from Leap's faff8e5:
  ref2.py, leapk.py, qmm16.py, qmmg25.py, gmr24.py.
  - It contains c4b6be9's key buckets and 7da7f5d's R-58 tier branch, which stays inert without BIGMOE_TIER.
  - The copy is identical to Leap's working tree, which KBTEST tested. `SOURCE` in the folder names this.
- Snapshot file hashes (sha256[:12]; `snap/` is not in git, Leap's faff8e5 is): gmr24.py 09e2389db277; leapk.py 3297b61fb257; qmm16.py 4350e9932870; qmmg25.py 3637a47dafec; ref2.py ef90fc209069.
- At load, `model.load` asserts `CFG["key_buckets"]` is on and BIGMOE_TIER is unset, then runs ref2's self-test.
- Rows are decoded by `ref2.generate` (`decode._gen_ref2`), greedy, at C 192, with PF1 as `evalsuite.env` sets it, as
  R-40's rows were.

## Which rows are decoded
`results/ref2b_keys.txt` (sha256[:12] e405642d19cb), 68 rows:
- **65 rows that reach a bucketed step.** The AR step making completion token i has N = n_prompt + i + 15 keys, so it
  is bucketed (N ≥ 513) once n_prompt + i ≥ 498. Taken conservatively, these are the core rows with
  n_prompt + n ≥ 497: 33 math, 21 IFEval, 11 HumanEval+, with 50,669 reference tokens.
- **3 spot checks:** rows that end just short of the boundary: gsm8k/715 (ends at 496), HumanEval/95 (495),
  ifeval/2969 (490).

## Checks before anything is copied (`tools/ref2b_assemble.py`)
1. All 68 rows are decoded.
2. The 3 spot checks are token-identical to `results/ref2`.
3. No decoded row differs from `results/ref2` before its first bucketed step (n_prompt + i ≥ 498).

Any failure means the "REF2b = REF2 below 512 keys" construction is wrong. Then nothing is copied, the runner stops
(exit 5), and it is reported. If all pass, the other 142 core rows are copied from `results/ref2` (marked
`copied_from`), so `results/ref2b/decode.jsonl` holds all 210.

## Verdict: decision 4's condition, with R-40 step 2's rule unchanged
- **PASS (same scores):** in each of math, IFEval (strict, prompt level) and HumanEval+ (pass@1), the paired delta
  REF2b − ref.r2 is above −5 points AND the exact two-sided McNemar p is at least 0.05.
  - FAIL if any set fails.
  - A gain with p < 0.05 is "differs (higher)", left to NewIdeas / Kamen.
  - The noise yardstick is R-40's (the floor vs ref).
- Command: `evalsuite report --variant ref2b --decode --ref ref.r2`.

## Information (no pass line)
- REF2b vs REF2 (`--ref ref2`): rows changed per set, where each changed row first differs, score deltas, discordant
  items.
- REF2b AR tok/s (quiet mode, FAN_CAP 3500); lifetime footprint per chunk.

## Order, locks, guards
- **Runner 30** (`tools/day_runner30.sh`). Chunks start no new row after 12 min, and each takes one queued quiet lock
  through `cli.gpu`, which adds:
  - a memory gate: 38 GB free;
  - a swap guard: stop if swap grows by more than 2 GB;
  - a 17-min stop.
- **Gates:** no chunk while `soft_end.sh` ramps, PAUSE exists, the lid is closed with no external display, or less
  than 34 GB is free. Deadline Sat 08:00.
- **Expected:** ~73 min of GPU (≈51.8k tokens at ~11.65 tok/s), plus loads and queue waits.
- **Code pin** (a change stops the runner):
  - the snapshot;
  - `~/BigMoE/bigmoe` (.py, .so, .metal) and `ext/build`'s .so (replaced by the frozen copy: amendment 1);
  - `evalsuite/`;
  - the MLX, mlx_lm and mtplx versions.
  - BIGMOE_EXT_BUILD and BIGMOE_MAXK are unset, so the defaults apply.

## Amendment 1 (Fri 17:3x, before any REF2b row): BigMoE from a frozen copy
- **BigMoE runs from `snap/bigmoe-6c64148`** (`EVALSUITE_BIGMOE_CODE`, env.BIGMOE_CODE), as R-40's runner 25 did. It
  holds bigmoe/ plus the `ext/build` and `ext/build_next` .so files; its `SOURCE` names the commit and the .so hashes.
  - The pin covers that copy instead of `~/BigMoE`.
  - Why: BigMoE commits several times a day (6c64148 at 17:25), and a live pin would stop the ~73-min job at the next
    commit.
  - The REF2b arm doesn't use `ext/build_next`. It is there for R-56's GATE16 arm.
- **Runner 30's first launch (17:24) was stopped before any row.**
  - When PAUSE cleared, it had queued its first chunk. That chunk (pid 6718) and its lock ticket were stopped and
    released at 17:2x, before the lock was granted, so nothing ran.
  - Runner 30 is relaunched with this amendment.

## Reporting
- **To NewIdeas and Leap.**
- **On PASS:** `results/ref2b` becomes the 45 GB line's exact reference (NewIdeas' routing, within Kamen's decision
  4), and R-56's REF2b re-run follows.
- **On FAIL:** decision 4 is back with Kamen, so it also goes to the orchestrator (address from `orch_addr.sh`).
- **On a failed check:** NewIdeas and Leap first.

## Appendix: the 68 keys (= `results/ref2b_keys.txt`, sha256[:12] e405642d19cb; the last 3 are the spot checks)
```
math500/test/prealgebra/1640.json
math500/test/intermediate_algebra/1422.json
math500/test/algebra/694.json
math500/test/intermediate_algebra/101.json
math500/test/precalculus/323.json
math500/test/number_theory/427.json
math500/test/intermediate_algebra/607.json
math500/test/prealgebra/1973.json
math500/test/geometry/702.json
math500/test/number_theory/459.json
math500/test/number_theory/45.json
math500/test/number_theory/634.json
math500/test/algebra/509.json
math500/test/algebra/733.json
math500/test/algebra/1282.json
math500/test/intermediate_algebra/1111.json
math500/test/algebra/2700.json
math500/test/precalculus/1291.json
math500/test/prealgebra/1139.json
math500/test/prealgebra/1655.json
math500/test/algebra/634.json
math500/test/algebra/2102.json
math500/test/counting_and_probability/430.json
gsm8k/701
gsm8k/702
gsm8k/707
gsm8k/709
gsm8k/710
gsm8k/711
gsm8k/712
gsm8k/714
gsm8k/721
gsm8k/724
ifeval/1251
ifeval/2275
ifeval/3223
ifeval/3089
ifeval/3608
ifeval/2246
ifeval/3710
ifeval/3071
ifeval/2567
ifeval/3439
ifeval/2304
ifeval/3425
ifeval/1367
ifeval/3063
ifeval/2225
ifeval/3538
ifeval/2328
ifeval/1000
ifeval/1154
ifeval/3126
ifeval/3326
humaneval/HumanEval/68
humaneval/HumanEval/141
humaneval/HumanEval/77
humaneval/HumanEval/107
humaneval/HumanEval/93
humaneval/HumanEval/147
humaneval/HumanEval/117
humaneval/HumanEval/130
humaneval/HumanEval/127
humaneval/HumanEval/159
humaneval/HumanEval/134
gsm8k/715
humaneval/HumanEval/95
ifeval/2969
```

## Amendment 2 (Fri 19:4x): memory floor raised after a swap-guard stop
- **Runner 30's third chunk was stopped by the swap guard at 19:33:49.** Swap went from 13.6 to 20.1 GB while an iOS
  Simulator (15 GB, the user's own app) ran beside our ~35 GB decode.
  - 9 of 68 rows are saved; the row in flight was lost and will be redone.
  - The saved rows peaked at 34.2-35.6 GB and keep the same code pin (e7d0e3c042b8). They stand.
- **New floor, by Kamen's 18:45 rule** (start only with the expected lifetime peak plus >= 8 GB free): runner 30 waits
  for 43 GB free, and cli's memory gate is 43 GB (`--need 39`).
- Runner 31 stopped with it, as designed, because runner 30 ended without REF2B_DONE. Both are relaunched; runner 31's
  floor is in docs/PREREG_r56b_2026-10-30.md.

## Amendment 3 (Fri 20:5x): a shared lock and C 96, after a second swap-guard stop
- **The second swap-guard stop was at 20:43:07:** swap went 12.7 -> 16.9 GB with Kamen's own apps busy (a Windows game
  through Wine, the iOS Simulator at 9 GB, Chrome at 5.8 GB). 23 of 68 rows are saved.
- **The orchestrator (20:4x), applying Kamen's FAST INTEGRATION rule:** this regeneration is bits-only work with no
  timing, so it runs under `peer lock share`, not quiet. A quiet lock refuses every shared screen.
- **Changes for the remaining 45 rows** (no row changes its meaning):
  - `cli.gpu --share` takes `peer lock share` and asks again while a quiet lock runs. When a quiet ticket waits, the
    stage stops at the next row boundary (decode's yield flag; exit 75), and the runner asks again.
  - Chunks are 6 min, so a waiting quiet ticket waits at most one row.
  - **C 96 instead of 192:** about 22 GB instead of 35. The floor is 30 GB free and cli's gate 30 GB (`--need 26`).
    - The suite's streamer is exact at any C, and the AR's 1-row calls take the slot path at any C >= 12 (10 ids <=
      0.9 C).
    - So C changes only speed. The assemble checks (prefix vs `results/ref2`, spot checks) still apply to every row.
- **The code pin changes:** cli.py gets the share mode, and stages.py / live16b.py get R-56b's yield. No file the
  decode stage uses for numbers changes. Rows before 20:43 came from pin e7d0e3c042b8, and both pins are reported.

## Amendment 4 (Fri 22:5x): memory gate on vm_stat, and retries after a swap trip
- **The third swap-guard stop was at 22:50:32:** swap went 11.7 -> 14.2 GB beside MTP Head's shared job. 39 of 68 rows
  are saved; the C 96 rows peak at 21.4-22.3 GB.
- **Leap's finding (via NewIdeas, 21:1x), now the RULES.md gate:** other MLX jobs' wired Metal memory is missing from
  `memory_pressure`'s free %.
- **Changes:**
  - Runner 30 now waits for vm_stat available memory (free + inactive + speculative + purgeable) of at least 31 GB:
    the ~22 GB peak plus 8.
  - After a swap-guard trip, which still stops the stage at +2 GB, runner 30 waits 10 min and goes through its gates
    again, up to 6 trips, instead of stopping.
  - Nothing that makes a row changes.

## Amendment 5 (Sat 00:0x): the NO-SWAP memory gate (orchestrator 23:5x, RULES.md)
- **The new gate:** every MLX load of 10 GB or more needs NO-SWAP (free + speculative + purgeable + file-backed pages)
  of at least the lifetime peak + 8 GB under a shared lock. "Available" counts inactive anonymous pages that only swap
  can free.
- **Runner 30 now waits for 30 GB NO-SWAP** (~22 GB peak + 8), and cli.gpu's own memory gate (`free_gb`) uses
  NO-SWAP. The pause after a swap trip is 2 min instead of 10 (orchestrator 23:2x).
- **How the change went in:** runner 30 yielded to a quiet ticket at 23:56:32 (42 of 68 rows), then stopped itself
  between chunks when the cli.py edit changed its pin (exit 7, 23:58:32). No row was lost, and runner 31 stopped with
  it.
- Both are relaunched under a new code pin. The edit is cli.py's memory gate only, so nothing that makes a row changes.

## Amendment 6 (Sat 00:4x): back to quiet tickets (orchestrator 00:4x)
- **The orchestrator's call:** "until REF2B_DONE, take QUIET tickets (FIFO) for your REF2b chunks instead of shared
  locks."
  - Since 00:10 every shared request had lost to back-to-back quiet locks, and REF2b sat at 44 of 68.
  - With NO-SWAP around 36 GB and nothing running, memory serializes everything anyway.
- **Runner 30 now takes one quiet ticket per chunk**, re-queued right after each chunk.
  - It queues first and gates memory after the grant: cli.gpu waits for NO-SWAP >= peak + 4 GB, i.e. 26 GB
    (`--need 22`).
  - Still C 96 and 6-min chunks.
- **The switch happened at a yield,** with no row lost. The chunk from 00:38:58 stopped at a row boundary for a waiting
  quiet ticket at 00:47:50 (46 of 68 rows). Both runners were stopped in the 2-min pause that followed.
- Two earlier shared chunks failed (rc 1, 21:21 and 21:32) beside MTP Head's shared runs. A failed chunk saves no row
  in flight (decode's gate check), and the next chunk redoes it. Rows went 29 -> 31 -> 34.
