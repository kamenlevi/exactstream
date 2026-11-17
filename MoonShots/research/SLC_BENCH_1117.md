# M-45 SLCBENCH: the SLC microbench that decides "software etching" (MoonShots, Tue 17 Nov 2026, quiet lock 13:14)

Spec: ZERO's decisive test, ~/FableGod/zero/IDEAS_1117c_slc.md section 1.4. Pre-registered as NewIdeas QUEUE M-45 (b91b98c)
with ZERO's lines: PASS if CPU-touched ~= GPU-touched (~x2 faster than cold); KILL if CPU-touched == cold, or the knee <= 8 MB.
Code: scripts/slc_bench.m (Metal + C). Raw data: data/slc_bench.json. Printout: data/slc_bench.txt. Apple M5 Pro.
MB = 10^6 bytes. Every time is the GPU's own command-buffer time (GPUEndTime - GPUStartTime). Exact: scheduling only.

## Verdict: KILL (the knee is fine, the mechanism is not)
- **SLC size for GPU re-reads: knee ~24-28 MB.** Re-reads run ~500 GB/s (x1.70-1.74 of DRAM's 290.9 GB/s) flat from 10 to
  24 MB. Then x1.66 at 26 MB, x1.52 at 28, x1.30 at 32, x1.14 at 36, x1.05 at 40, and DRAM from 48 MB. 85 MB is x1.00.
  This clears ZERO's "knee <= 8 MB" kill line.
- **CPU-touched vs GPU-touched (21 MB read once, after a 100 MB GPU stream):**
  - cold: 76.8 us
  - GPU-touched: 42.9 us (x1.79)
  - CPU-touched: 147.2 us (x0.52). The CPU touch makes the GPU's read **2x slower** than cold.
  - CPU-touched, then pushed out of the CPU caches (another 16 or 48 MB read by the CPU): x0.99 / x1.00, i.e. exactly cold.
  - So the SLC keeps CPU-fetched lines for the GPU neither when they are filled nor when the CPU's cache evicts them.
- **Other variants:**
  - A PRFM PLDL3KEEP prefetch ("prefetch into L3, keep") instead of loads: x0.73.
  - A CPU touch during the GPU stream (the design's own timing): x0.54.
  - A CPU write (memcpy) instead of a read: x0.94.
- **ZERO's own numbers fall with it:**
  - check: -1..-4.5 ms per check becomes 0;
  - draft: the 85 MB per-pop re-read set never hits (x1.00);
  - GPU-touched lines do not survive a 100 MB stream in between (x1.00, the spec's literal ordering C).
- **What still holds:** any GPU re-read set <= ~24 MB, with less than ~24 MB of other traffic between its reads, comes back at
  ~x1.7. PLAN900's "<= 26 MB per level" sits on the knee's edge (26 MB: x1.66 in (i), with no other traffic between).

## (i) GPU re-read bandwidth vs buffer size (median of 7)
cold = one read after a 512 MB GPU flush. hot = P whole-buffer re-reads right after it, one dispatch each, so the reuse
distance is the whole buffer.

| MB | 2 | 4 | 6 | 8 | 10-24 | 26 | 28 | 32 | 36 | 40 | 48 | 64 | 85 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hot GB/s | 716 | 982 | 475 | 481 | 494-507 | 484 | 443 | 378 | 330 | 305 | 295 | 292 | 290 | 290 | 291 |
| hot / DRAM | 2.46 | 3.38 | 1.63 | 1.65 | 1.70-1.74 | 1.66 | 1.52 | 1.30 | 1.14 | 1.05 | 1.02 | 1.01 | 1.00 | 1.00 | 1.00 |

- At 2-4 MB a GPU-side cache level serves the re-reads, not the SLC.
- 6-8 MB read slightly low because each pass includes a ~1-2 us dispatch cost.
- Cold single reads converge to 285-290 GB/s from 26 MB up. The first ~2-3 us of every command buffer is fixed cost.

## (ii) 21 MB read once by the GPU (30 interleaved trials per condition)
Every trial starts with a 512 MB GPU flush and a 256 MB CPU flush. "stream" = a 100 MB GPU read (the expert phase).
Between the stream and the timed read the GPU runs an ALU-only spin kernel (~1.2 ms, no memory traffic) in every
condition, so the timed read never starts from an idle, down-clocked GPU: 0 of 330 trials idled. (A first smoke run with
a 2 ms idle gap gave cold reads of 235 us instead of 77: idle gaps must not be used for this kind of timing.)

| condition | median us | p25-p75 | GB/s | cold / x | touch us |
|---|---|---|---|---|---|
| hot: GPU read, spin, read (no stream) | 47.9 | 45.5-49.2 | 439 | 1.60 | |
| cold: stream, spin, read | 76.8 | 76.0-78.6 | 273 | 1.00 | |
| A_gpu: stream, GPU touch, spin, read | 42.9 | 42.4-43.8 | 489 | 1.79 | |
| A_cpu: stream, CPU touch (during the spin), read | 147.2 | 125.5-164.0 | 143 | **0.52** | 159 |
| A_ev16: A_cpu + the CPU reads 16 MB other | 77.6 | 74.4-88.6 | 271 | 0.99 | 270 |
| A_ev48: A_cpu + the CPU reads 48 MB other | 76.6 | 75.5-79.0 | 274 | 1.00 | 419 |
| A_pf3: A_cpu with PRFM PLDL3KEEP, no loads | 104.7 | 90.5-109.5 | 201 | 0.73 | 136 |
| A_cpuw: A_cpu with a CPU memcpy write | 82.0 | 77.2-84.3 | 256 | 0.94 | 221 |
| B_cpu: CPU touch during the stream | 143.0 | 117.9-156.8 | 147 | 0.54 | 150 |
| C_gpu: GPU touch, stream, spin, read | 76.8 | 75.0-78.8 | 274 | 1.00 | |
| C_cpu: CPU touch, stream, spin, read | 134.1 | 119.1-148.6 | 157 | 0.57 | 116 |

CPU touch speed, 21 MB from cold: one thread 225 us (93 GB/s), four threads 109 us (193 GB/s).

## Side finding for the engines (a check, not a lever)
A GPU read of bytes the CPU has just *read* runs at ~half speed while those lines sit in the CPU's caches (coherency).
Once the lines are pushed out, the read is normal. A CPU *write* (memcpy) costs only ~6 %.
- Engine paths worth checking: any CPU pass that reads weight or KV bytes shortly before the GPU reads them (page-touch
  loops, checksums, CPU-side reads of GPU inputs).
- The cost is bounded by the CPU cache size: ~16-21 MB at most per event, i.e. <= ~70 us per event.

Novelty (papers): [engineering] measurements of an unpublished chip parameter (the M5 Pro's GPU-visible SLC ~24-28 MB) and of
CPU -> GPU cache sharing on it. Not checked against literature in this run.

Prior records: TRIED "Keep weights in the SLC" (KILL 10-18, "SLC ~24 MB"; now measured: 24-28 MB); Rnd16 SLC warming (KILL);
HC-SLC-STAGING (KILL in substance 15 Nov; GPU touch kernel). This run adds the CPU-filled form, which is dead too.
