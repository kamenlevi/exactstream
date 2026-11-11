# build_1105 (Fable B): the FAST-tax switch and the lazy GDN commit, ready to apply (Thu 5 Nov 2026, 18:2x-19:1x EET)

Asked by the orchestrator ("build everything so we get actual numbers"). CPU only; nothing outside this folder was edited;
nothing ran on the GPU. Patches are against **BigMoE 4a9a4f1** (cut at 6adad93, rebased at 19:05 after BigMoE's 18:55
commit changed other lines of the same file) and **Leap cccfd42** (`ref2.py` and `gdnbody.py` are byte-identical at Leap's
current 0405421). `git apply --check` passed for all three. Both repos are moving: rerun `check_fasttax.py` before applying.
Check outputs are saved beside the scripts (`*.out`). Background: `../LOSSY350_B.md` (its corrections block).

| part | files | state |
|---|---|---|
| 1. FAST-tax switch (BigMoE) | `fasttax_r57.patch`, `fasttax_aba.sh`, `fasttax_report.py`, `check_fasttax.py` | ready to apply and run; CPU check ALL PASS |
| 2. lazy GDN commit (Leap) | `lazy_commit/gdnbody_lazy.patch`, `lazy_commit/ref2_lazy.patch`, `lazy_commit/gdnb_lazy_check.py` + three CPU checks | a sketch that applies and compiles; protocol proven on CPU; the kernel's bits and ms need Leap's GPU check |

## 1. FAST-tax: `R57_FASTTAX` in r57_screen.py

**What it does.** One env-gated block after the build-day switches, one line in the end-of-run report. `R57_FASTTAX=1`
turns on ref2's `att_single` (R-62: each 16-row attention call as one fused SDPA instead of MTPLX's 12 head chunks;
Leap GDNB-2: -0.2 ms per call, 12 calls per check = -2.4 ms [K]). Lists work too: `att`, `inproj`, `att+inproj`.

**One change against the brief: `gdn_inproj` is skipped in the all-in arm.** ref2's `_InProjPad` concatenates the four GDN
projections' packs as stored, and DENSE-N-RAM (`R57_DNR=1`, part of the all-in arm) has already replaced two of them by 4-D
native copies, so the two cannot run together as ref2 stands [K ref2.py 1009-1045, 288-312]. With `R57_DNR=1` the switch
turns on `att_single` only and prints why; K-83's exact `gdn_inproj_x` keeps serving the layer. Without `R57_DNR` it turns
on both. `R57_FASTTAX=inproj` under `R57_DNR=1` stops on an assert. DENSE-N-RAM (-2.0 ms) is worth more than `gdn_inproj`
(at most -0.5, most of it already in `gdn_inproj_x`), so the all-in arm should keep DNR.

**Guards.** FAST only: it asserts `R57_FAST` is set and `R57_REFTOK` is empty (past 512 keys `att_single`'s bits differ
from REF2-T's; below 512 they are equal). Unset or empty, nothing runs.

**CPU check (`check_fasttax.py`, ALL PASS at 19:05 on BigMoE 4a9a4f1; it tests HEAD + the patch, so it stays valid as HEAD moves):**
- the patch applies to HEAD and the result compiles;
- the patched file's AST minus the two `if os.environ.get("R57_FASTTAX"):` blocks equals the unpatched file's, node for
  node (207 top-level nodes), and the blocks bind only two throwaway names that the old code never reads first: with the
  variable unset the program runs the old statements one for one, so EXACT arms cannot change;
- the two blocks run against ref2's real CFG literal under 8 arm environments without the variable (ref2.CFG identical,
  nothing printed), 6 FAST arms with it (only the intended switches move) and 5 misuse cases (each stops on an assert);
- Leap's ref2 has what the patch relies on; the A-B-A script's arm equals `fri_night.sh`'s QA.

```bash
~/BigMoE/.venv/bin/python -I ~/FableGod/fableb/build_1105/check_fasttax.py
```

**Install (BigMoE).**

```bash
cd ~/BigMoE && git apply --check ~/FableGod/fableb/build_1105/fasttax_r57.patch && git apply ~/FableGod/fableb/build_1105/fasttax_r57.patch
```

```bash
cp ~/FableGod/fableb/build_1105/fasttax_aba.sh ~/FableGod/fableb/build_1105/fasttax_report.py ~/BigMoE/scripts/
```

**Arms (A-B-A, same 12 HumanEval keys as `fri_night.sh`, FAST, one process per arm, about 2 min per arm).**

| arm | env |
|---|---|
| A `fta1_C`, `fta2_C` | `fri_night.sh`'s QA: `BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1` |
| B `ftb_C` | A + `R57_FASTTAX=1` |

```bash
NOTBEFORE=00:20 NOTAFTER=02:30 WAITMEM=40 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 37 20 "BigMoE: FAST-TAX A-B-A" /bin/bash scripts/fasttax_aba.sh
```

`FTA_PAL=1` runs the palindrome A B B A; `FTA_C` fixes C; `FTA_REPORT_ONLY=1 FTA_C=<C>` rebuilds the report. Output:
`data/fta/fta.log`, `data/fta/fta_report.txt`.

**The A/B line (pre-registered Thu 5 Nov ~18:4x; metric = median check ms over all checks but each prompt's first, B vs
the mean of the two A medians):**
- PASS: B is >= 2.0 ms faster, the two A medians are within 2.0 ms, kept per check within 1 %, and ref2 counted
  `att_single` calls in B and none in A.
- KILL: the saving is < 1.0 ms (A medians within 2.0 ms).
- VOID: the A medians differ by more than 2.0 ms (drift): rerun as the palindrome.
- BETWEEN: anything else.

The report also prints check minus SSD busy, the fit's intercept (the fixed cost per check, where the saving should show),
tok/s per prompt and how many prompts kept identical tokens. Quality is not judged by this run: a FAST arm needs the map
arm and EvalSuite's decision 4. Expected: -2.4 ms of about 90 ms by day, so the run is near its own noise; the palindrome
at night is the safer read.

## 2. Lazy GDN commit: a sketch for Leap

**The cost it removes [K, read from the code].** After a partly accepted check, MTPLX's `commit_verified_window` rebuilds each
pure-GDN layer's recurrent state by replaying the kept rows from the snapshot state: a second kernel per layer (mlx_lm's
`gated_delta_update` + its gate ops) that reads and writes the 3.15 MB fp32 state again, pulled by the next check's eval.
35 layers, 110 MB. A full accept pays nothing.

**The change.** A pure-GDN cache entry's state becomes `[conv, base, pack]`: the true state is the pack's rows (per kept
row q | k | v | a | b, 10,336 bf16) replayed on `base`. The next <= 16-row call's body (`gdnbody.gdn_body_lazy`) copies the
pack's rows in front of its own in threadgroup memory, runs phase B's loop body over them first (state only), writes that
committed state once, runs its own rows and emits their pack. MTPLX's commit still runs (conv state, stash); its replay
node is dropped unevaluated and the pack is cut to the kept rows (a view). The pack is a state leaf, so MTPLX's snapshots
(views) and restores (clones) carry it: no hook there and nothing leans on array identity. Calls the body declines flush
first (today's replay). The PLE layer is never lazy. Switch: `REF2_GDN_LAZY=1` or `ref2.CFG["gdn_lazy_commit"]`, default
off, needs `gdn_body`.

**Per 16-row check and layer [P, `lazy_sim_check.out`, 24 % full accepts]:**

| | state reads | state writes | replay kernels | recurrence steps |
|---|---|---|---|---|
| today | 1.75 | 1.75 | 0.76 | 22.1 |
| lazy | 1.00 | 1.00 | 0 | 26.0 |

Expected -1.4 ms per code check (-0.8 to -2.0) [P/G]: 165 MB less state traffic (0.85 ms at 195 GB/s) and about 240 fewer
small dispatches, against 3.9 more register steps. A full accept costs more than today (the next check replays 16 rows).

**What the CPU checks prove, and what they do not.**

| check | proves | result |
|---|---|---|
| `lazy_sim_check.py` (numpy model, `lazy_protocol.py`) | the bookkeeping: 240,000 random events (partial / full accepts, rollbacks, abandoned checks, AR steps, prefill, stock-path calls), snapshots as copies and rollbacks as clones | every output, conv state and true state equal to today's bit for bit |
| `ref2_lazy_cpu_check.py` | the same on the REAL patched ref2 functions, MTPLX's commit / snapshot / rollback, mlx_lm's cache and recurrence (MLX on its CPU device), 320 events; only the Metal kernel is a stand-in | equal bit for bit |
| `metal_syntax_check.py` (`xcrun metal -c`, no GPU) | the derived lazy kernel source is well-formed Metal (today's source is the control) | compiles, G3 off and on |
| not proven | the kernel's bits against today's body + MTPLX's replay, the 28.9 KB of threadgroup memory on the device, the ms | Leap's `gdnb_lazy_check.py` |

The fuzz caught one real bug in my first design (pending rows keyed by array identity were lost on a rollback); the code
reading caught the cause (snapshots are views, restores are clones). The shipped design is the second one.

**Install and test (Leap).**

```bash
cd ~/Leap && git apply --check ~/FableGod/fableb/build_1105/lazy_commit/gdnbody_lazy.patch ~/FableGod/fableb/build_1105/lazy_commit/ref2_lazy.patch
```

```bash
cd ~/Leap && git apply ~/FableGod/fableb/build_1105/lazy_commit/gdnbody_lazy.patch ~/FableGod/fableb/build_1105/lazy_commit/ref2_lazy.patch && cp ~/FableGod/fableb/build_1105/lazy_commit/gdnb_lazy_check.py scripts/
```

```bash
~/BigMoE/.venv/bin/python scripts/gdnb_lazy_check.py
```

**Its lines (pre-registered ~19:0x; in the script's header).** PASS: every comparison bitwise (off == today; nothing
pending == `gdn_body`; committed state == MTPLX's replay for keep 1, 2, 9, 15, 16; a 40-event protocol stream) AND >= 0.8 ms
saved per partial-accept cycle over 36 layers AND <= 0.8 ms more per full-accept cycle. KILL: < 0.4 ms saved. BITS: fast but
not bitwise, an exact-track KILL that FAST may still take.

**Risks, in order.** (1) The replay loop is phase B's body without the output store; if the compiler rounds it differently
the committed state is not bitwise (the check's step 3 shows it; fallback: keep the dot alive in B0). (2) Threadgroup
memory goes from 16.5 to 28.9 KB per head. (3) Anything that reads `entry.state[1]` directly sees the base, not the true
state: use `ref2.gdn_lazy_state(model, cache)` or flush. (4) `_gdn_body_call` and `_gdn_call`'s lazy branches were not run
on the real module (the CPU check uses a double for the forward); step 4 of the GPU check runs them.
