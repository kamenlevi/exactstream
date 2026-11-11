# Wall 1, lever 1: zero per-call overhead (build plan; build Wed, measure Thu by day)

BigMoE, lead of the WALL 1 SPRINT. Tue 3 Nov 2026 ~18:5x, aligned 19:1x with Fable A's spec
(~/FableGod/push/WALL1_D1_SPEC.md: the call's timeline in the code, each step's bound and line). Plan only tonight.
Tonight's trace (c45t_224 / c45tj_224 -> stall_report) fills Fable A's pre-registered table (its section 4); its section 5
re-orders the steps below if the numbers say so.

## Where the cost sits (Fable A, [K] = measured)

- The ~0.28-0.30 ms is paid once per BURST (loader call with misses), whatever the number of experts in it [K Fit12 B4:
  0.300 ms at 1-24 experts]. It is not queue depth, and the drive's first byte is only ~0.08 ms of it [K Leap B1].
- Its parts:
  - first byte ~0.08;
  - the burst's TAIL ~0.15: the last read's CPU work inside pread (~0.10) + its 2.15 MB memcpy into the slot pieces (~0.055);
  - GCD wake / join ~0.06;
  - release ~0.01;
  - +0.05-0.065 when the SSD idled >= 0.25 ms before the burst [K M-20];
  - +0.035-0.04 GPU pickup, on exposed bursts only [K SC5].
- So splitting a read into pieces pays only if each piece is copied as it lands (the tail shrinks to one piece's). A deeper
  queue alone is worth ~0.03-0.06 per burst.

## M-28 SSDMICRO (MoonShots, Tue 18:59, quiet; ~/MoonShots/data/ssdmicro.txt, 6e3a693): the read path alone, measured

MoonShots' harness copies load_jobs' read path: page-aligned F_NOCACHE preads, one dispatch_apply at user-interactive QoS.

- Median C 256 call (2 base records, whole reads, 0.8 ms idle before): 527 us, of which 187 us is fixed (the engine fit
  says 283).
  - Parts: first byte ~119 us, idle wake +44 us, GCD start + return 8 us.
  - The other ~96 us of the engine's per-call cost is OUTSIDE the I/O path: the last record's copy into the slot pieces,
    the release and the GPU pickup. S0 confirms this.
- Pieces are SLOWER at our queue depth: one record in 8 pieces +39 us (2-32 pieces: +17..+93); 2 records in 4 / 8
  pieces +39 / +49 us. Pre-spun I/O threads gain nothing (GCD's start is only 5-23 us).
- Keepalive: the setting 50 really spaces reads ~180 us apart, and anything up to 270 us removes the whole +44 us wake
  (settings >= 200 leave half). No spacing fix is needed.
- JOINT (one call for base + sidecar): -120 us per layer (1 + 1 records), -155 us (2 + 2).
- Page cache: +49 us per call, so keep F_NOCACHE.
- Per check at C 256: keepalive -3.4 ms SSD busy, JOINT -4..-5 ms. The best I/O-only call (keepalive, whole reads) has a
  130 us fixed cost. Its floor is the first byte, which only overlapping calls (lever 2) can hide.

**Plan change:** S3's pieces, S5 (workers) and the keepalive spacing fix are DROPPED. S1 + S2 stay. S0 still gets built
Wed: it measures the ~96 us outside the I/O path. If the copy tail is the biggest part there, the candidate is reading
straight into aligned slot memory, which removes the copy (Fit12 B4: -20 % fixed; killed earlier by a 30 % line, to be
re-judged per burst), not pieces.

## W1-PICKUP (Leap, Tue ~19:5x, 5221761, TABLE_350 'W1-PICKUP'; Swift harness, quiet, 300 interleaved trials per cell)

The GPU <-> loader handshake, measured. The harness copies bigmoe_spin and the loader's release verbatim from a0a5fbe.

- Today: release 6.5 us, pickup 16.7 us even when the data is already there, plus half a poll gap. That is ~+29 us at a
  0.6 ms wait, ~+54 at 1 ms, ~+104 at 2 ms (matches SC5).
- One-row spin with volatile coherent(system) loads + system fences (MLX's FAST_SYNCH pattern, `#pragma METAL internals`):
  release 0.2 us, pickup 2.9 us flat from 50 us to 2 ms.
- [K, new] A device-scope GPU store (our routing post) reaches the CPU only when something flushes or evicts its lines.
  - That is ~20 us after the next real dispatch, and practically never while the GPU only spins: 293 / 300 unseen within
    5 ms.
  - So the loader's "see" is ~20 us, not ~5. On a layer with no resident hits (post -> spin directly) it can be
    milliseconds: part of the exposed time may be this, not the SSD.
  - A system-scope post: 7.4 us, complete.
- Flags (Leap writes the kernels as patch files; BigMoE integrates Wed into ext/build_w1 + gated.py, default off; today's
  path stays the fallback):
  - BIGMOE_SPIN_COHERENT=1: the spin polls ONE row, the release writes one row;
  - BIGMOE_POST_SYS=1: the post's row through a coherent(system) pointer + system fences;
  - S0's histogram gets pair -> time (p0 17 us, p6 22, p26 73, p41 236, p52 623, p58 1080, p65 2074).
- Line (traced A/B, Thu's lock): stall_report's cadence a minus burst a falls >= 0.03 ms AND tokens identical. Prize ~1.7-2
  ms per check at 45 GB (~+2 % bound). The bigger value: per-expert releases and lever 2's in-flight waits drop to ~3 us,
  and the all-miss layers lose their late "see".

## S7B-PROBE (Leap, Tue ~22:3x, 50de28c, TABLE_350 'S7B-PROBE'): a correctness rule for every lever that moves slot writes closer to their reads

Inside one command buffer, if the GPU has read a slot and the CPU then rewrites it, later dispatches can read the OLD
bytes from the GPU's caches.
- Leap's harness: 96-100 % of words stale, 20/20 trials. memoryBarrier(.buffers) still stale in 16/20. Read only after
  the rewrite: 0 stale. coherent(system) loads: 0 stale, at plain-load bandwidth (239 vs 220 GB/s).
- Today's engine is safe by its schedule:
  - under OOO, phase 1 reads only resident slots;
  - LRU victims are never in the current routing;
  - a layer's pool is touched once per pass, and MLX commits several command buffers per pass;
  - routing rows and tables carry seq stamps.
- Tonight's C 160 chain (the fastest slot reuse we run, every token checked, including the volatile tier's copy-back) is
  the live check.
- Rule: any new path that rewrites a slot the GPU read in the SAME command buffer needs coherent(system) slot reads
  (Leap builds default-off copies for 3c-N / NAX6-N) or a command-buffer boundary. That covers per-expert release (S7b),
  lever 2's forecast reads landing mid-buffer, and two-pass checks at 12 GB.

## The steps, in order (all exact: the same bytes land in the same slot bytes; flags default off)

| Step | What | State | Bound / halved, ms per check | Line |
|---|---|---|---|---|
| S0 | instrument the split: read events get t_issue / t_pread0 / t_pread1 / t_copy1; service events t_seen / t_dispatch / t_joined / t_rel; spin pickup histogram (status[32 + p]); stall_report prints plan / wake / first byte / tail / release / pickup per burst | build Wed (~60 lines ext + ~30 stall_report) | none (measurement) | the loader-side parts sum to within 20 % of the per-burst fit |
| S1 | JOINT: base + sidecar of a layer in ONE burst (83 -> ~46 bursts per check) | built (BIGMOE_JOINT=1) | -11 / -5.5 busy | tonight c45tj vs c45t: PASS busy <= -8 ms AND bursts <= 50 AND tokens identical; KILL > -3 |
| S2 | SSD keepalive: a 4 KB F_NOCACHE read while serving, so no burst pays the wake-up | built (BIGMOE_SSD_KEEPALIVE_US); spacing fix Wed | -1.8 / -0.9 | ka1 vs ka0: the reads-alone fit d -0.04 ms AND busy -1 ms = PASS; d unchanged = KILL |
| S3 | pieces WITH the copy pipelined under the reads: one dispatch_apply over (job, piece); P = clamp(ceil(8 / jobs), 1, 8) pieces per job, 16 KB-aligned page ranges (base 132 pages, sidecar 47); each block preads its piece into the bounce at its offset and copies that range into the slot pieces at once; done[job]++; the job is ready at done == P; no nested dispatch_apply (b5) | build Wed (BIGMOE_SPLIT=1, ext/build_w1) | -7.6 / -3.8 | engine: per-burst fit a <= 0.18 ms (from ~0.30) AND tokens identical = PASS; a > 0.25 = KILL |
| S5 | persistent I/O workers instead of GCD wake-ups | only if S0 shows wake median >= 0.03 or p90 >= 0.08 after S3 | -0.9..-1.4 / -0.45..-0.7 | S0's wake <= 0.01 median |
| S6 | the volatile tier's ghost copy off the read's critical path (issue the read first) | only if the tier is adopted | -1.6 / -0.8 | ghost jobs' t_pread0 - t_issue 0.035 -> <= 0.005 |
| S4 | the spin's back-off (pickup) | parked: tried twice, no gain | -0.9 / -0.45 | S0's histogram re-checks it for free |
| S7 | one routing row per layer (sidecar ids ride in the base row) | after S1-S3 are measured | -1..-2 / -0.5..-1 | row-format change; later |

- Budget [P, Fable A]: the fixed part per check goes 23.5 ms -> 13.8 (S1) -> 12.0 (+S2) -> 6.4 (+S3) -> ~5 (+S5/S6). At
  halved gains that is ~10-12.
- Exposed SSD in Fable A's replay: 22.3 -> ~9 ms at the bound, so steady about +16 % bound (~+8 % halved).
- What D1 cannot cut is the first byte and the wait for the GPU's routing. Lever 2's forecast reads hide those (D2).

## Wed build list (on main, after the move)

0. W1-PICKUP: `git apply ~/Leap/patches/w1_pickup/w1_pickup.patch` (Leap 77ada05, README beside it; ext only, against
   ec08cec). Apply it ONLY AFTER the conversion's ext/build rebuild (~15:27), so EvalSuite's snapshot .so carries 761cb3d's
   reader alone. Build into ext/build_w1.
   - Flags, default off: BIGMOE_SPIN_COHERENT=1 (one simdgroup polls pair 0 with coherent(system) loads; the release
     writes one pair), BIGMOE_POST_SYS=1 (post_ids through a coherent(system) pointer; off = today's exact statements),
     BIGMOE_SPIN_COH_CAP (polls before timeout, default 1e6), BIGMOE_SPIN_PAIRS=1 only with SPIN_COHERENT.
   - Rows: base and sidecar never share one (region keyed by (seq, L); JOINT's sidecar spins under side0 + L).
   - Leap's checks: applies cleanly; syntax rc 0; its kcheck PASS (400 trials, exact slots incl. -1 / -2, pickup 3.5-5.9 us,
     fast path / stale word / timeout as today; post_ids rows bitwise identical with the flag off and on).
   - S0 under SPIN_COHERENT: a log2 histogram of the poll count `it` (~0.25 us per poll) times the GPU-side wait directly.

1. S0 instrumentation (exact: timestamps only), with stall_report's new per-burst split.
2. S3 in ext/build_w1 behind BIGMOE_SPLIT=1, plus its unit test: a model-free random walk like scripts/test_vtier.py,
   split vs whole reads giving identical slot bytes, with JOINT and VTIER on, 1-24 experts per burst. Check MoonShots' S3
   screen first (SSD + CPU only, by day): (iii) pieces copied as they land <= (i) whole read + copy - 0.10 ms at 1 expert.
3. S2 spacing: MoonShots found BIGMOE_SSD_KEEPALIVE_US=50 reads every ~220 us (pread + usleep slop); in M-20, 200 us
   spacing removed only about half the wake penalty.
   - Fix: sleep to a deadline (usleep for the bulk, then spin on mach_absolute_time to the next tick) so the spacing holds
     at 50 us.
   - Check: the keepalive thread logs its achieved median spacing; M-28 (MoonShots, Wed ~15:01) measures it independently.
4. Rebuild of ext/build for the conversion stays separate (it carries 761cb3d only). The S0 / S3 / S2 builds go to
   ext/build_w1 until they pass.

## Measurement (Thu by day after T0_DONE, one quiet lock ~30 min, row-major store; docs/WALL1_BENCH.md)

1. The 24 GB baseline (W1B, the shipped s10b stack) first.
2. Then the 45 GB arms at C 224, all traced: default / + JOINT / + JOINT + keepalive / + SPIN_COHERENT / + POST_SYS (W1-PICKUP).
3. Then the winner untraced for tok/s and MB per kept token.

- Lever 1's overall line (pre-registered Tue 18:5x): every token exact AND the fixed part <= 10 ms per check AND steady
  >= +5 % vs the default = PASS; steady < +1 % = KILL; otherwise BETWEEN. Each step also gets its own line, in the table above.
- If RELAX or T0 run late: Fri 00:30, next to the conversion A/B.
