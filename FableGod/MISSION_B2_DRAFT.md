# MISSION B2 (Fable B): THE BOUNDARIES. Can the dense pass run as a few persistent kernels, exactly?

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~21:3x.)

**Why.** Your GDN_PASS.md found that the exact dense pass stops at ~24 ms expected and ~20 at every bound, above cloud's <= 17 ms line.
- **Recount with Leap's real D1:** Leap measured Fable A's HC kernel B at -0.74 ms, not -2.6 (~/Leap/TABLE_350.md "D1-B"; D1 total ~-1.3..-1.8, not -3.9). That makes it ~26 ms expected and ~21.5 at every bound.
- **What's left is not bytes:**
  - ~14 ms of floors
  - ~5 ms of structural remainder: dependent GEMM pairs, fills and drains, attention's per-key walk, GDN's row steps
  - ~1,000 kernel boundaries
- **The same pattern shows up elsewhere:**
  - Fable A found 4.7 ms of "chain tax" in the HC read alone (HC_READ.md).
  - Leap counts ~2,800 kernels per check at 1.5-6 us each, with no blocking host syncs.
  - CPU graph build is ~8.5 ms per check, ~3.4 of it exposed (B5).

The exact lever class nobody has priced is **fewer boundaries**: overlap independent work across ops and layers, persistent (mega)kernels, pre-encoded command buffers.

**Deliver**
1. **The dense pass as a dependency graph at 16 rows.** Cover each layer type: GDN, attention, the HC reads and writes, router / shared / tail. Give:
   - the kernels, with their bytes and ms
   - which edges are true data dependencies and which are only ordering
   - where the GPU idles between kernels (from Leap's timelines: C0, R-65b, R-83, the CA runs)
2. **The boundary cost, accounted:** fills and drains, launch gaps, dependent latency and exposed CPU build. Reconcile with ~34 ms of GPU against ~14 ms of floors.
3. **What Apple's GPU allows for persistent or dataflow kernels:**
   - threadgroup residency and forward progress
   - device-memory atomics and ordering
   - how many threadgroups can stay resident
   - Metal and MLX tools: indirect command buffers, events, concurrent dispatch, mx.compile

   Include prior art (persistent megakernels on CUDA: Mirage MPK, Hazy Research's "no bubbles" decode megakernel, FlashInfer's persistent kernels) and say why each does or doesn't transfer. Web reading for prior art is fine; no downloads.
4. **At least three exact designs** (bitwise = REF2-T: each op's arithmetic unchanged). Starting points, not limits:
   - a persistent per-layer kernel
   - cross-layer pipelining (layer L+1's weights prefetched while layer L computes)
   - concurrent dispatch of independent kernels
   - pre-recorded command buffers that remove the CPU build

   For each give:
   - ms per check (upper bound, then halved: our estimates run ~2x high)
   - why it stays bitwise
   - the risk (deadlock, occupancy, MLX limits)
   - what Leap builds
   - the cheapest test
5. **The bet:** does the exact dense pass reach <= 17 ms with it? If not, give the honest exact floor and what's left.

**Rules**
- CPU only; writes only in ~/FableGod/fableb/; no downloads. GPU timing is Leap's.
- Before writing, read Leap's entries and ~/TRIED/TRIED.md (compile, graph capture, megakernel, persistent, ICB). Leap's entries: R-55, R-65b, R-83, C0, the CA runs, R-10 (compile), R-11b, B5.
- Exact mode stays exact; reference changes are Kamen's call. Mark every number [K] / [P] / [G].
