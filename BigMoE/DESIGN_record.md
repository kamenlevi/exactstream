# One-record expert pack + deep-queue loader (24 GB goal; exact: same bytes, same pools)

Prior art: one contiguous record per expert read with a single I/O is a published win (Strata +17 %, mlx-lm +32 %);
SlotStream measured an M5 Pro SSD at 9.5 / 17.3 / 17.3 GB/s for cold 2.76 MB F_NOCACHE preads at QD 1 / 8 / 32.

## File format (offline repack, ~70 GB; check df first, floor 100 GB)
`models/bare/records/L{L:02d}.bin`, one per layer: E records of the 9 pieces in loader order
(gate w/s/b, up w/s/b, down w/s/b), each record padded to a 16 KB multiple (F_NOCACHE alignment), record e at
offset e * RECSZ. Same bytes as the safetensors pieces (a pure relayout), so pools and results are unchanged.

## Loader path (BIGMOE_RECORDS=1, ext only)
- add_layer gets the record file fd + RECSZ + per-piece offsets inside the record.
- load_jobs: one `preadv(fd, iov[9], 9, e * RECSZ)` per miss; iov[i] = pool_i + slot * per_i. One syscall, one
  contiguous range, zero copy (no bounce buffer). Short reads: loop on the remaining iov.
- Queue depth: today dispatch_apply issues 9 x misses small preads; with records it is one 2.77 MB read per miss.
  Deep queue = issue every miss of a pass (and prefetches) concurrently on a dedicated concurrent queue sized
  BIGMOE_IO_QD (default 16), instead of GCD's core-count default. Prefetch and demand share the device; demand first.
- Staging (PF1) and prefetch use the same record read.

## Measure (A/B, exact by construction; check sha anyway)
ar2 / spec k=3 at 25 % and 50 %: io ms/token, stalled layers, tok/s; record path vs 9-piece path; QD 8 / 16 / 32.

## Split of work
BigMoE: repack script + ext record path. MoEPrefetch: QD / read-size policy in its fork on top (and prefetch
ordering), MoonShots: SSD QD curve (1..64) on our device, which decides the QD default.

## Hooks for MoEPrefetch's policy work (agreed 2026-10-19)
1. Every read is tagged demand or prefetch at submit; an atomic in-flight counter per tag.
2. Before a prefetch read is issued, check whether demand for that expert has already arrived; if so, promote
   (wait on the in-flight demand read / mark it) instead of issuing twice.
3. Per-pass stats: io wall ms, bytes, and max in-flight per tag (exposed via ext.stats()).
MoEPrefetch's first policy: reserve R of the QD slots for demand, cap prefetch at QD-R in flight, skip queued
prefetches whose target layer has already started.

## Measurements in hand (2026-10-19 midday)
- Scout loadbench (cold F_NOCACHE bursts): today's 9 unaligned pieces a = 313 µs + bytes/12.1 GB/s; one 16 KB-aligned
  read per expert a = 191 µs + bytes/13.1 GB/s (device floor) → burst latency −18…26 %, ≈ +10 % tok/s at a 17 GB cache.
  Every piece in today's pack is at an unaligned file offset; alignment alone is worth 6–10 %. Chunking a read: worse.
  GCD dispatch_apply is fine. Destination memory type doesn't matter.
- MoonShots: SSD 14.6 GB/s for whole reads at QD 32; at 17 GB the loader uses ~3.5 GB/s, ~40 % of each stall is
  loader overhead (9 pieces, ramp/drain). Orchestrator policy: demand reads in parallel chunks at QD 1, prefetch as
  whole-expert reads at QD 4–32 (MoEPrefetch tunes).
- Open: preadv into 9 pool destinations leaves the 51,200 B scale/bias slots unaligned every other slot (Scout: alignment
  matters). Plan: build preadv-9 first and measure against a single aligned read into a record-layout bounce; if the gap
  is real, move the pools to a record layout (piece arrays as strided views) or pad the scale/bias slots.
- Load peak counts toward 24 GB (orchestrator 13:40): load + warm-up exceed the 23–25 GB MLX peak (mmap'd dense
  weights, Python, n-gram cache) — separate item.
