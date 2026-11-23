# Screen 3: what a check costs at 1 / 16 / 32 rows (WEEK98 route 3, "the cheap check")

The orchestrator 18:2x: "yes to screen 3, the real 32-row check at 45 GB split into dense / experts / misses". 18:0x:
"Report: per-layer ms before and after, at 1/16/32 rows, at 45 and 24 GB, and what it does to your break-even tables."
19:1x: High Power Mode since ~19:15; compare only within a run.

Tools:
- scripts/week98/cc3_runner.sh v5.
- cc3_probe.py v2, on the shipped loop's R57_COPYPROBE hook. The engine is unchanged; every pass is rolled back.
- cc3_report.py.
- flagcheck.py: the head test against the real loops.

Runs use the v121 clean clone on 4 HumanEval keys (68, 16, 91, 6), cap 256.

## Ticket 1, 20:58-21:02 (v4): no data
- lossy-24 died at start-up. The 24 GB loop asserts that R57_ROWCAP and R57_WARMSPLIT are exclusive (w24p2:159).
- exact-45 was skipped at its gate: 34 GB on both reads, gate 47.
- Outputs are in v4_2058/. The fixes are listed in v5 (commit 2fbfc3f).

## Ticket 2, 21:15:56 (v5): lossy-24 (b24/, b24_probe.jsonl, b24_report.txt)

The run: rc 0, 4 prompts, 807 tokens, lifetime peak 23.91 GB, power mode 2 (High Power) on AC. The probe triggered on its own
every 4th check (23 probe checks), and passes ran on REF2's continuation (10 strict + 10 loose).

| lossy-24 (C 100, checks capped at 13 rows) | ms | per layer (/ 48) |
|---|---:|---:|
| live 13-row check, in-run reference (29 checks, probed checks and the next left out) | 81.5 | 1.70 |
| live 13-row check right after a probe (10) | 76.1 | 1.59 |
| = dense only, same 13 rows (routed experts stubbed) | 23.9 | 0.50 |
| + experts and misses (130 misses, SSD wait 23.9 ms) | 53.7 | 1.12 |
| dense only, 1 row (an upper bound, see below) | 30.6 | 0.64 |
| dense only, 32 rows | 181.3 | 3.78 |
| real 32-row pass, today's slot path, fast gate off (3) | 2814.7 | 58.64 |

Where the time goes (medians; graph build vs wait, with thread CPU in each):

| pass | build | build thread CPU | wait | wait thread CPU |
|---|---:|---:|---:|---:|
| d1 | 8.7 | 8.4 | 21.8 | 15.1 |
| d16 | 18.3 | 17.7 | 5.0 | 0.7 |
| d32 | 14.7 | 11.3 | 168.1 | 59.5 |
| s32 | 2807.3 | 1271.0 | 7.5 | 2.4 |

Reading it:
- **13 rows, dense only, is CPU-bound.** 23 ms total, of which 18 ms is graph build on the main thread; the GPU work
  hides under it (SB).
- **The 1-row pass is the first pass after the loop's own work**, and it pays work deferred from the previous check:
  15 ms of thread CPU inside its wait, against 0.7 ms for the 13-row pass that follows it. 30.6 ms is therefore an upper
  bound. At 45 GB the loop's warm split runs before the probe and absorbs this.
- **32 rows, dense only, takes the slow path.** 181 ms, of which 168 ms is wait (60 ms of it on the CPU). The routed
  experts are stubbed, so this is the non-expert path above 16 rows.
  - **Cause, CORRECTED 21:3x: DENSE-N-RAM's unpack.** My first guess was the GDN / attention fallback.
  - lossy-24 sets R57_DNR=1, which converts every large 4-bit dense weight in place to the native layout used by calls
    of <= 16 rows (ref2.py dense_native_convert).
  - A call above 16 rows (ref2.py `_dnr_qmm`: `rows <= R`, R = 16) instead runs `_dn_unpack`, the MLX-op inverse repack
    of the whole weight, on every call, and then MLX's qmm.
  - The run's own counter confirms it: `dense_native_unpack 2592` = 24 passes above 16 rows (20 d32 + 3 s32 + 1) x 108
    converted weights.
  - exact-45 does not set R57_DNR, so its 32-row dense pass takes MLX's qmm directly.
  - Whatever the GDN / attention fallback above 16 rows costs is left over under the unpack; not measured.
  - **Fix candidate (Leap's ref2; not built).** In verify scope only, run 17-32 rows as two 16-row native calls: per
    row exact against the 16-row path, with no unpack. Or use a 32-row native kernel (D32's mr2 / mg2: one weight read).
  - The prefill must keep today's unpack + MLX qmm. A final prefill chunk of 17-32 rows would otherwise change its bits.
- **The real 32-row pass takes 2.8 s on the slot path.** It is almost all graph build, of which 1.27 s is thread CPU:
  the prefill's records3 reads and CPU rebuilds. Its miss / SSD counters read 0 because that path bypasses the pool's
  stats.

What it does to the break-even tables (24 GB):
- A deep drafter needs a 32-row check at <= ~1.35-1.4x a 16-row check (cheapcheck.md).
- At 24 GB the check is capped at 13 rows, and a 32-row check today costs 2.81 s, 34.5x the 13-row check.
- Even with a widened gate, the non-expert 32-row path alone (181 ms) is 2.2x the whole 13-row check.
- So at 24 GB nothing in the deep-drafter break-even moves until both are fixed:
  - DENSE-N-RAM's >16-row unpack (a ref2 fix in verify scope, above);
  - the gate (piece 1).
- The 24 GB line keeps its <= 13-row checks.

## exact-45 (a45): did not run Sat night; re-armed Sun 21:00 -> Mon 01:00
- Sat 21:17 -> Sun 00:49 it waited, holding nothing, for NO-SWAP >= 47 GB on two reads 30 s apart. It never got there:
  41 GB at the 21:15 grant, 12-15 GB while another session's routing log ran, 32-35 GB after midnight (00:49: free 9.9 +
  purgeable 0.8 + file-backed 21.6; anon 28.3, compressor 2.6). The gate was not lowered. No ticket was queued.
- v6 of the runner (CC3_ONLY=a45) runs exact-45 alone: no b24 ticket, no b24 flag check. Dry-tested on both paths
  (a45 only; both runs) and with a bad value (exit 2).
- Armed Sun 00:50 (pid 12468): from Sun 21:00, every run ends by Mon 01:00 (the orchestrator 01:0x). Its flag check
  PASSES against the real exact-45 loop. The orchestrator is asking Kamen about closing apps for the ~13 GB it needs.

## exact-45 (a45), Mon 16 Nov 03:26:10 (v7, start gate 40 on the orchestrator's exception): a45/, a45_probe.jsonl, a45_report.txt

The run: rc 0, 4 prompts, 809 tokens, C 256 (configs/exact-45.env), lifetime peak 42.71 GB, power mode 2 (High Power) on
AC, process wall 61 s. Tokens == REF2 on every prompt, errors 0. The loop's warm split triggered the probe on 17 checks
(ref_ok 11, loose 3). This is REAL decoding on the shipped v121 loop, and every probe pass is rolled back. Its 53.9 tok/s
is not a speed number, because the probe runs inside the decode.

How it started, and the bias check:
- NO-SWAP sat at 41-42 GB from 01:51 on an idle Mac and fell ~0.04 GB/min, so the normal gate (47 = peak + 4) could not
  open. The orchestrator gave two exceptions: 42, then 41 (03:16; it opened once, but the run's own check read 40 and
  skipped), then 40 (03:2x). The run started at NO-SWAP 40 GB on both reads.
- Memory through the run (a45_vm_1116.txt, every 10 s): swap 6,884 -> 6,884 MB (+0) and swapouts +0. The compressor
  took the shortfall: 3.0 -> 8.3 GB at its peak, compressing through the first ~40 s. NO-SWAP was 6.9-9.1 GB during the
  run and 50 GB at exit.
- The bias check was PRE-REGISTERED 03:1x: swapouts inside the run window would label the misses / SSD-wait part. There
  were none, so the split stands as measured. For information: the compressor's CPU work overlapped the first part of
  the run. The dense passes are GPU-bound at 32 rows and CPU-bound at 16.

| exact-45 (C 256, 16-row checks) | ms | per layer (/ 48) |
|---|---:|---:|
| live 16-row check, in-run reference (25 checks, probed checks and the next left out; misses 658, SSD wait 89.4 ms) | 139.1 | 2.90 |
| cold 16-row pass (the loop's warm split, 17; misses 469) | 110.7 | 2.31 |
| = dense only, 16 rows | 30.2 | 0.63 |
| + experts (warm 16 - dense 16) | 40.1 | 0.84 |
| + misses (cold 16 - warm 16) | 40.4 | 0.84 |
| warm 1 / 4 / 8 / 16 rows | 50.6 / 50.3 / 57.7 / 70.3 | |
| dense only, 1 row (an upper bound, see below) | 37.8 | 0.79 |
| dense only, 32 rows | 70.9 | 1.48 |
| real 32-row pass, today's slot path (3) | 2502.8 | 52.14 |

| pass | build | build thread CPU | wait | wait thread CPU |
|---|---:|---:|---:|---:|
| d1 | 7.3 | 7.3 | 30.5 | 11.1 |
| d16 | 19.9 | 15.1 | 10.3 | 0.7 |
| d32 | 11.2 | 8.5 | 59.3 | 14.0 |
| s32 | 2494.4 | 1031.4 | 8.4 | 2.0 |

Reading it:
- **At 16 rows the cold check splits about evenly:** dense 27 %, experts 36 %, misses 36 % (per layer 0.63 + 0.84 +
  0.84 ms). The live check is 28.4 ms above the cold pass: the loop's own per-check work, plus more misses (658 vs 469).
- **32 rows, dense only, is 2.35 x 16 rows and GPU-bound** (59.3 ms wait); 16 rows is CPU-bound (19.9 ms build).
  exact-45 sets no R57_DNR, so it takes MLX's qmm directly (compare lossy-24's 181 ms through the DENSE-N unpack).
- **The 1-row pass is again an upper bound.** d1 (37.8) is above d16 (30.2): it still pays deferred work, with 11.1 ms
  of thread CPU inside its wait. The loop's warm split did not absorb it, as the lossy-24 note had expected.
- **The slot path at 32 rows takes 2.5 s, almost all graph build** (1.03 s thread CPU), as at 24 GB. That is 18 x the
  live 16-row check. Its miss counters read 0 because it bypasses the pool's stats.
- **32 rows with a gate carrying 320 ids [est]: ~188.4 ms = d32 + E16 + the union growth + 1.5 x M16.**
  - The printed v2 verdict was 1.35 x the in-run 16-row check, "at the line". CORRECTED 03:4x (basis): it divided a
    forward pass built from probe parts by the whole live check, which carries 28.4 ms that the parts do not.
  - Like for like (cc3_report.py v3, a45_report.txt): **1.70 x** the cold 16-row pass, forward over forward.
  - It is 1.56 x if the live check's extra 28.4 ms is a fixed per-check cost added to both, and ~1.66 x if that extra
    is misses (x 1.5 at 32 rows).
  - So **~1.56-1.70 x [est]**. That is in line with the engine profile's REF2-T split path (Breakthrough's CHECK55, Mon
    00:42: 1.67 x) and above its one pass (1.48 x).

What it does to the break-even tables (45 GB):
- A deep drafter needs a 32-row check at <= ~1.35-1.4 x a 16-row check (cheapcheck.md).
- At 45 GB the like-for-like estimate is ~1.56-1.70 x on today's forms. So the 32-row check misses the line even with
  the gate built (piece 1), and today's slot path (18 x) is far off.
- The levers inside the estimate:
  - the 32-row dense pass, 70.9 ms = 2.35 x 16 rows (one-pass forms; v32one's step 1 took 11.6 ms off the engine's
    32-row check);
  - the misses at 32 rows, 1.5 x M16 = 60.6 ms (the 32-row union's extra experts read from SSD).
- This agrees with the engine correction of Mon 00:4x: on today's engine a 32-row tree does not pay
  (treek_costs_v2.json).
