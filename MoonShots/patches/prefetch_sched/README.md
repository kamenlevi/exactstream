# Verify-pass prefetch scheduler for BigMoE's loader (flag-gated patch)

Against BigMoE aa5dc96 (4-line routing row, 61 predictions; the diffs were rebased from 825c0a3 and apply cleanly). Files:
- `sched_ext.diff`: `ext/bigmoe_ext.cpp`. All new behaviour sits behind `BIGMOE_PF_JIT=1`; with it unset the loader is
  unchanged. The only exception is a harmless parse fix (`(np >> 8) & 255`) and the post_ids `tot` mask, both no-ops
  while bits 16+ of `np` are zero, which they are unless `BIGMOE_PREFETCH3` / `BIGMOE_PF_RANKED` are set.
- `sched_py.diff`: `bigmoe/stream.py` (distance-3 router reuse `BIGMOE_PREFETCH3`, ranked unique lists `BIGMOE_PF_RANKED`)
  and `bigmoe/gated.py` (posts `n1 | n2<<8 | n3<<16 | ranked<<24`). Defaults are unchanged.
- `ext/`, `py/`: the patched files in full (the `*.orig.*` files are BigMoE's, for the diff).
- `mk_tree.sh` + `test_sched35.py`: build a patched copy outside ~/BigMoE and check it on the streamed 35B.

## Why the current loader cannot keep the SSD busy
Victim prefetch (V2, +5 %) may only evict while the GPU is held at a gate, because every layer's table for the next
pass is published right after that layer is served. A layer that is not stalled therefore prefetches nothing in the
victim configuration (there are no spares), and nothing can be issued further ahead than the next stall.

## The protocol change: just-in-time publication (`BIGMOE_PF_JIT=1`)
Layer X's table for pass s is published right after the CPU serves (and releases) X-1 of pass s, instead of
right after X was served in pass s-1. Layer 0 is never published, so it always takes the gate (one round trip per
pass). The rest follows from that:
- Between X's use in pass s-1 and its publication in pass s, no table the GPU will read references X's slots. The
  GPU also reads no slot of X except those `ensure(X)` hands out. So a prefetch may evict X's LRU tail, or recycle
  X's finished prefetches, at any time in that window. It no longer needs a stall.
- Slots are only reserved for layers X greater than the layer being served, and only before X is published (the
  loader schedules L+1..L+3 and then publishes L+1). An `abort()` checks this. A published layer loses no slot until it
  is served, which keeps BigMoE's invariant: "a real demand miss ⇒ the GPU took the slow path and is held".
- The published table includes finished prefetches (ready = 1), so a prefetch that lands before publication still
  gives the fast path. One that lands after publication is added by `table_add`, as today.
- In a speculative read, `table_add` now runs before `ready = 1`. Recycling needs `ready = 1`, so a slot is never
  recycled under a pending `table_add`.
- If the CPU publishes late, the tag does not match, the GPU takes the slow path, and the result is still exact.

## The scheduler (on top of JIT)
- **Reserve when the prediction arrives, read in priority order.** When layer L is served, its row's predictions for
  L+1, L+2 and L+3 each reserve a slot (a free slot, else the LRU tail, skipping experts predicted for that layer this
  pass) and queue the read. `pump()` issues queued reads nearest layer first, then by the class's measured precision.
  It runs after the release and whenever the loader thread is idle.
- **Demand pre-empts (`BIGMOE_PF_QD=n`).** At most n speculative reads are in flight. The loader thread is the only
  issuer and is blocked during demand reads, so no speculative read starts while a demand read runs, and demand reads
  never wait in the queue. `ensure(X)` cancels X's still-queued reads; a needed one becomes an ordinary demand miss.
- **Cold-end insertion.** Prefetched experts stay in `pf` (outside the LRU) until used, as today. Recycling prefers the
  oldest finished prefetch not predicted for this pass. Per-layer cap: spare + `BIGMOE_PF_VICTIM` (default 12).
- **Confidence.** With `BIGMOE_PF_RANKED=1` the GPU posts one unique list per distance, ranked by the expert's best-row
  router log-prob (30 ids hold about 30 distinct experts instead of 4 rows × 7 with repeats). Without it, the CPU ranks
  by how many verify rows predicted the expert. There are 9 classes = distance × bucket (rank < 8 / < 16 / rest, or
  rows ≥ 3 / 2 / 1).
- **Feedback (`BIGMOE_PF_THETA=p`).** Each class's precision is measured as used when its layer is served, divided by
  reserved (EWMA by halving). A class below p is not reserved, except 1 in 16 to keep it measured.
  `BIGMOE_PF_BUDGET=n` caps reservations per pass.
- `BIGMOE_PF_EARLY=1` reserves and issues before the layer's own demand reads instead of after the release.
- **Bound: `BIGMOE_ORACLE_DIST=D`** with `BIGMOE_ORACLE_PLAY=file` (recorded with `BIGMOE_ORACLE_REC`). Serving L
  reserves the recorded true ids of L+1..L+D, through the same JIT and QD machinery. D = 1, 2, 3 measures the value of
  lead time with perfect knowledge, and D = 48 is the full-pass ceiling.
- `ext.stats()` gains `pf_reserved, pf_cancel, pf_qdemand, pf_gated_off, pf_nofit, pub_jit` and `pf_cls` (per class:
  reserved, scored, used, precision).

## Latent issues in today's loader, found while reading it (dormant in the shipped configs)
- `prefetch()` recycles "the oldest finished prefetch". When a layer gets two prefetch calls in one pass (d=1 from L-1
  and d=2 from L-2, i.e. `n2 > 0`), that can be a same-pass prefetch already added by `table_add` to the published
  table. The GPU may then take the fast path onto a slot being refilled. `n2` is 0 in every default config: verify rows
  use all 30 slots for d=1, and `BIGMOE_PREFETCH2` is off. The same applies to `ready = 1` being stored before
  `table_add`. JIT mode avoids both.

## Wide prediction line (`BIGMOE_PRED_LINES=k`, k = 0..4)
BigMoE found that 87–92 % of verify misses are unpredicted: 30 slots hold only top-7 per row at S=4. With k extra
128-byte lines per routing row ([seq, 31 ids] each, written by post_ids, each tag checked by the loader), a row carries
30 + 31k predictions. With k = 3 that is 123, e.g. top-12 × 4 rows for L+1 plus top-8 / top-6 × 4 for L+2 / L+3.
gated.py sizes `ROW = 96 + 32k`, and stream.py's budget `left = 30 + 31k`. It needs the spin gate (asserted).
With k = 0 the behaviour is unchanged.

## 35B results (streamed, 55 %, 64 tokens; `data/sched35_matrix.txt`; share lock, so timings are noisy and counts are not)
**Exact: every run gives base's tokens (K=3 sha 70904c380e, K=0 3cb379f49e).** Slow = loader time on layers with
misses, which is about the GPU's wait. All values are per pass.
| K | config | stalled layers | demand io | late-prefetch wait | slow |
|---|---|---|---|---|---|
| 3 | base (victim B=12) | 26.9 | 16.6 ms | 3.6 ms | 20.4 ms |
| 3 | JIT | 23.3 | 12.9 | 4.8 | 17.8 |
| 3 | JIT + QD 4 | 23.7 | 13.5 | 4.0 | 17.6 |
| 3 | JIT + QD 4 + ranked d=1/2/3 (16/9/5 unique) | 25.3 | 15.2 | 1.7 | 17.0 |
| 3 | same + EARLY | 25.3 | 19.0 | 0.7 | 19.9 (worse: speculative reads slow the demand reads) |
| 3 | **oracle D=1** (true ids, 1 layer ahead) | 4.5 | 3.0 | 7.3 | **10.4** |
| 3 | **oracle D=3** | 2.4 | 1.9 | 2.5 | **4.6** (D=48: 4.4) |
| 0 | base | 12.8 | 6.6 | 1.1 | 7.8 |
| 0 | JIT + QD 4, per-row d=1/2/3 (12/6/4) | 5.6 | 3.1 | 2.1 | **5.3** |
Reading: with perfect knowledge, a 3-layer lead removes ~78 % of the verify wait, and 1 layer removes ~49 %. With router
reuse in 30 slots, JIT recovers only 14–17 %, and the gap is recall (`pf_cls`: d=1 precision 54–100 % by class; an L+2/L+3
guess not confirmed at d=1 is right 11–13 % of the time). Plain decode gains most from JIT (−32 % wait), because
today a layer that does not stall prefetches nothing.

### Re-check on BigMoE's 61-slot row (aa5dc96 + this patch; `data/sched35_wide.txt`): exact again (same shas), wait per pass
| K | config | stalled | demand io | late-prefetch wait | slow |
|---|---|---|---|---|---|
| 3 | base W1 (61 slots, victim B=12) | 22.8 | 15.4 | 9.4 | 25.1 (30-slot base was 20.4: wider line, unbounded issue, contention) |
| 3 | JIT + QD 4 | 18.9 | 12.2 | 7.2 | 19.6 |
| 3 | **JIT + QD 4 + `BIGMOE_PF_VICTIM=24`** | 18.3 | 11.3 | 6.8 | **18.3 (−27 % vs W1)** |
| 3 | same + 123-id row, top-12/8/6 for L+1/L+2/L+3 | 18.2 | 12.7 | 6.0 | 19.0 (L+2/L+3 precision 6–13 %: no help) |
| 3 | same, ranked unique 40/30/20 | 22.0 | 15.3 | 5.8 | 21.4 (over-issues: 169 reads/pass) |
| 3 | 123-id + θ 0.25 | 18.7 | 11.6 | 6.5 | 18.4 (θ just switches L+2/L+3 off) |
| 3 | oracle D=3 | 2.5 | 1.9 | 2.4 | 4.5 |
| 0 | base W1 | 12.8 | 6.4 | 1.1 | 7.6 |
| 0 | JIT + QD 4, top-24/12/8 (1 row) | 8.0 | 5.5 | 2.7 | 8.3 (worse: d=1 precision 11 %; 12/6/4 on the 30-slot run gave 5.3) |

## Checks
1. Compile: `mk_tree.sh` (CPU only).
2. Exactness and mechanics on the streamed 35B (quality only, `peer lock share`): `test_sched35.py base 0.55 64 3`
   with `BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12`, then `sched` with the variants below. Every sha must equal base.
   Also check K=0 (plain decode: 1 row, fast path common).

## Hardware A/B for BigMoE (125B, spec k=3, 50 %, ab125-style interleaved, 3 rounds; tokens identical)
Base = W1: `BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12` on the 61-slot row. With all scheduler flags unset the patched loader
behaves exactly like aa5dc96 (it only masks the `np` fields).
| run | flags (on top of base) | question |
|---|---|---|
| J1 | `BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4` | JIT protocol + bounded queue: prefetch at every layer, not only stalls |
| (35B pick) | J2 below; J3/J3r did not help on the 35B | run J1, J2, ar, O-3 first |
| J2 | J1 + `BIGMOE_PF_VICTIM=24` | per-layer cap |
| J3 | J2 + `BIGMOE_PRED_LINES=3 BIGMOE_PREFETCH2=8 BIGMOE_PREFETCH3=6` | 123-id row: top-12/8/6 per row for L+1/L+2/L+3 |
| J3r | J2 + `BIGMOE_PRED_LINES=3 BIGMOE_PF_RANKED=1 BIGMOE_PREFETCH=40 BIGMOE_PREFETCH2=30 BIGMOE_PREFETCH3=20` | same, ranked unique lists |
| J4 | best J3 + `BIGMOE_PF_THETA=0.25` | feedback throttle |
| ar | `BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_PREFETCH2=6 BIGMOE_PREFETCH3=4 BIGMOE_PF_VICTIM=24` vs base, plain decode | the −32 % wait seen on the 35B (not 24/12/8: over-issues) |
| O-D | record with `BIGMOE_ORACLE_REC=f` (any run), then `BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=0 BIGMOE_ORACLE_PLAY=f BIGMOE_ORACLE_DIST=D`, D = 1, 3 | ceiling and the value of lead time |
Report tok/s, stalled layers and io ms per pass, and `pf_cls` precision per class.
Decision rule: if O-48 is within 10 % of base, drop the scheduler. If O-1→O-3 gains a lot but J3 does not, the
predictor is the limit (any better predictor plugs into the same slots, ranked).
