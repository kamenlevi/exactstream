# FLAT-2: install and run

Built by the zero session, Thu 5 Nov 2026, on the orchestrator's request ("build everything so we get actual numbers").
Spec: `~/FableGod/zero/LOSSY350_ZERO.md` section 4. **Not yet run on the GPU**: everything below the dry-run is
unproven on the engine.

## What it answers

Whether 350 tok/s lossy is possible at all on the 125B at 45 GB. With the split form and never-wait in place, the 350
inequality has one unknown left: **FLAT**, what a cycle costs before any expert row is paid and with no SSD wait
(dense pass + draft + commit).

| Line | FLAT | Meaning |
|---|---|---|
| PASS | <= 30 ms | The last 5-6.5 ms have named owners; 350 is live on code at kept >= 12 |
| BETWEEN | 30-36 ms | 350 needs kept >= 15-18: only the capacity drafter with RELAX at 1.39 nat |
| KILL | > 36 ms | No kept the chain allows closes it; state the code ceiling as ~300 |

350 itself needs FLAT <= 23.5-25 ms at kept 12. Today's estimate was ~57 ms with no waits (from FAR's read of the
"C 208" run), FULL's "tonight" ~39.

## Owner, time, memory

- **Owner: Leap or BigMoE** (whoever holds the next quiet slot). Nothing outside this folder is edited; all output
  lands in `build_1105/out/`.
- **GPU: ~2-3 min per arm (my estimate from the pre-flight's timings, not measured), two arms, under 10 min.** Each arm
  is one process with a 420 s alarm.
- **Memory**: the runner picks C by NO-SWAP, as `fri_night.sh` does.

| NO-SWAP at start | C | Lifetime peak (BigMoE's measured all-in) |
|---|---|---|
| >= 44.3 GB | 224 | ~39.7 GB |
| >= 42.2 GB | 208 | ~37.6 GB |
| >= 40.0 GB | 192 | ~35.5 GB |
| >= 36.0 GB | 160 (uniform pools, the pre-flight's config) | ~31.3 GB |
| below | nothing runs | |

FLAT does not depend on C: the dense pass, the draft and the commit are the same, and the expert rows are subtracted.

- **Lock**: quiet (exact timings). The runner takes it itself; `HAVE_LOCK=1` runs inside yours.
- Guards as BigMoE's runners: PAUSE / HOLD honoured, swap +2 GB during an arm stops it.

## Steps

1. **CPU pre-check, right before the slot (3 s, no GPU, no MLX):**

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/zero/build_1105/dryrun_cpu.py
```

   It must end `DRY-RUN OK`. It re-patches BigMoE's `scripts/r57_screen.py` as it is at that moment, so it is the check
   that the HEAD still fits. `warning (the run goes on)` lines are fine (see "If the HEAD moved").

2. **Optional smoke (~1 min GPU per arm): does each lever set run?**

```bash
FLAT2_SMOKE=1 bash ~/FableGod/zero/build_1105/run_flat2.sh
```

   One short pass of each kind. Its verdict line says INVALID (too few cycles): expected. Look for `rc=0` per arm.

3. **The run:**

```bash
bash ~/FableGod/zero/build_1105/run_flat2.sh
```

4. **Read** the last lines of `out/flat2_fri_<C>_screen.txt` (and `flat2_all_<C>_screen.txt`): `FLAT-2 VERDICT: ...`.
   The runner prints both at the end. Send the two verdict lines and the two files to the orchestrator.

## The two arms ("every owned lever on")

| Arm | Levers | Status at this HEAD |
|---|---|---|
| `fri` (**the verdict**) | Friday's all-in FAST arm, exactly as `fri_night.sh` sets it: `R57_PRESET=45` (chat switch, K-73, K-83, JOINT, copy split, coherent spin, volatile tier 64/32, keepalive, OOO), `BIGMOE_EXT_BUILD=build_qom`, `R57_FAST=QOM`, `R57_D17=1`, `R57_D7B=code`, `R57_D7BNAX=1`, `R57_HC8=1`, `REF2_HC8_FAST=1`, `R57_DNR=1`, `R57_HCB=1`, `REF2_SMALL_NATIVE=1`, `REF2_OOO_1C=1`, plus the split form `R57_QOM_SPLIT=0.9` | BigMoE's pre-flight ran `pqall` and `ps99` clean today (Thu 14:07) |
| `all` | `fri` + `R57_DKV=1 R57_DKV_ADD=1` (Leap's drafter cache, tokens PASS Thu morning on the exact default) + `R57_GDNB=1` (Leap's one-dispatch GDN body, bitwise, wired, never run live) | Not run in this combination. If it crashes, `fri` stands |

Not on, and why:
- **NAX6 / NAX6-NH** (`R57_NAX=1`, `R57_H1C=1`): not in Friday's arm, and I could not confirm it is wired with QOM and
  the one-call OOO. It changes the expert rows, which FLAT subtracts, so it can move FLAT only through per-call overhead.
  Add it to `EXTRA` in `run_flat2.sh` once BigMoE's preset carries it.
- **RELAX / rule C**: they change kept per check, not FLAT. They live in the relaxed screen copy, which this patcher
  refuses (it lacks the split's lines).
- `R57_HEADC` (never smoked, superseded by DSUB), `R57_D8NAX` (the code prompt drafts on D7b here).

## How the number is built

`flat2.py` runs BigMoE's own `run()` loop on one code prompt (HumanEval/68) with each check held to 2 rows, using
`run()`'s existing draft cap (`maxd=1`) and a cut t of ~0, so a check is always [last kept token, one draft]. The
drafter still proposes its full block: that is the steady draft cost.

| Pass | What it is for |
|---|---|
| 1 cold | Discarded: kernels, pools and drafter warm up |
| 5 measured | The engine's loop untouched (the per-check hooks are off). 50-100 cycles each (100 tokens, 1-2 kept per cycle) |
| stage | Each commit replay is waited for: draft, check, commit one after the other |
| probe | Leap's C0 probe at checks 8 / 16 / 24 / 32: warm repeats, the synced split, the row sweep 2 / 4 / 8 / 12 / 16 |
| real-block x3 + commit | The engine's own blocks (its t cut, up to 16 rows), all-resident. Not in the verdict |

- **All-resident**: the prefill adopts the prompt's experts over the pools (`ext.adopt`), so a repeated pass would start
  as cold as the first. Before each pass the hooks put back the residents the pass before ended with (`ext.lookup`,
  then `ext.ensure` per pool, the harness's own A6 warm call, capped at the pool size). This is FAR 4b's
  oracle-prewarmed form. A cycle counts only if its check had zero misses and no stalled layer call.
- **Cycle** = one cycle start to the next (two list appends in `run()`), so it holds whatever the loop does between a
  commit and the next draft.
- **FLAT = median cycle - 2 x the row sweep's warm slope.** The slope is the expert cost per row (every dense piece is
  flat from 2 to 16 rows, Leap's C0). The synced split's experts piece is the cross-check and the fallback.
- The report also sizes each flat term for the next cut (GDN, HC reads, attention, router + shared, MoE tail, HC
  write, PLE, head, draft, commit), and gives the measured no-wait tok/s of the engine's own blocks.

Validity, printed with the verdict:
- `INVALID` = fewer than 100 steady zero-miss cycles (the pools did not hold the pass), or the harness's gate check
  failed. Fix: `FLAT2_CAP=60` (a shorter pass), or a larger C.
- `INCOMPLETE` = the cycles are fine but no probe gave the expert term; FLAT then has only its upper bound, the cycle.
- "warm repeats bitwise == live: n of n" should be full. If not, the probe's re-run is not the loop's check (see
  below), and the slope is suspect while the cycle is still good.

## Files

| File | Role |
|---|---|
| `run_flat2.sh` | Runner: memory ladder, lock, guards, both arms |
| `flat2.py` | GPU entry: patches the harness in memory, runs it, hands over to the driver |
| `flat2_patch.py` | Reads BigMoE's screen, keeps everything up to `run()`, inserts six one-line calls, checks the copied rules |
| `flat2_probe.py` | The hooks: pool prewarm, stage timing, the C0 probe ported to today's gated loop |
| `flat2_driver.py` | The passes, the read-out, the files |
| `flat2_core.py` | The arithmetic and the verdict (numpy only) |
| `dryrun_cpu.py` | CPU dry-run: 43 checks |
| `out/` | `<tag>_screen.txt` (report), `<tag>_flat2.json` (every number), `<tag>_generated.py` (the harness as run), `<tag>.log` |

## What the dry-run proves, and what it cannot

Proves (CPU, 3 s): the patcher fits the HEAD and refuses a changed loop; BigMoE's real patched `run()` executes with
the hooks, the probe and the report on a stub engine with a virtual clock, where every piece has a set cost, and the
cycle, stages, slope, split, FLAT and the PASS / BETWEEN / KILL / INVALID outcomes come out exactly; the runner's ladder
and env.

Cannot prove: anything about the real engine. The three calls most likely to surprise on the GPU:
1. `ext.ensure` with a pool-sized list after a prefill (the prewarm). Read from the ext source: it aborts only past
   `C_main` distinct ids, and the list is capped there. Under LP1 the sidecar is only partly restored; QOM's sure rows
   never read it.
2. The probe's rollback and re-run under QOM with the one-call OOO. Same calls as the harness's own warm split and A6
   `rep`; C0 ran them Monday on the exact loop.
3. The `all` arm's lever combination.

A failed stage, probe or real-block pass does not cost the measured cycles: the report is still written.

## If the HEAD moved

BigMoE edited `r57_screen.py` in its working tree while this was being built (Thu 19:0x, per-check QOM counters). Two
things follow, both handled:
- The patcher compares the loop's rules the probe copies on the syntax tree. **Added** statements or arguments give a
  warning and the run goes on; a **changed** copied rule stops with the source line, before the model loads. Then
  either run on a frozen screen that still fits (`FLAT2_SCREEN=~/BigMoE/data/r57/snap/r57_screen_pf.py bash
  run_flat2.sh` is today's pre-flighted text), or update `masks_for` / `fwd` in `flat2_probe.py` with `G_CUT` / `G_FG`
  in `flat2_patch.py`.
- The new counters are read from the GPU right after the commit is submitted, in QOM arms only. That read waits for
  the commit replay, so the commit no longer overlaps the next draft there. FLAT-2 reports it as "the loop's tail".
  It also means QOM arms pay something the U3C and exact arms do not in Friday's A/B.

## Known limits

- One prompt, one domain (code). Chat drafts on the MTP head and is out of scope.
- Tokens are not judged: 2-row checks change what the FAST gate keeps, and this measures time only.
- A 2-row check's commit replays at most one row. The real-block passes give a real block's commit and check
  `FLAT + rows x slope + commit difference` against the measured cycle; that difference is what the 2-row form misses.
- All-resident is a ceiling: never-wait has to deliver it.
