# TRIED_SHORT: ideas an outside model would likely re-propose, and what happened (Sun 1 Nov 2026, updated Mon 2 Nov; full list: TRIED.md)
Setup: a 125B MoE (512 experts per layer, top-10, 48 layers, GDN + attention hybrid), 4-bit / 3-bit expert records streamed
from SSD on a 64 GB M5 Mac; lines at 45 / 24 / 12 GB of RAM. "Exact" = output tokens bit-identical to one-token decoding.
Speculative decoding verifies up to 16 drafted tokens per check. Measured check cost on the 45 GB default: 7.82 ms per
verified row plus a fixed 64-179 ms per check, so rows are cheap and the fixed part dominates.

## SSD reads (expert cache misses)
- Look-ahead prefetch of the next layer's experts (pre-gating, ProMoE, SlotStream): ADOPTED at 24 GB (V1 +9-10 %, JIT
  residency tables +6 %), KILL at 45 GB (live +3.4 %, R-57b: wrong guesses evict). Two layers ahead, deeper prefetch queues,
  score thresholds, ranked lists: KILL (no gain). On Apple Silicon SSD DMA and the GPU share the memory controller
  (F_RDADVISE slowed the GPU 73 %, Flash-MoE).
- Prefetch during the drafting window (SP-MoE): PARKED, upper bound 13-21 % of a check's misses (R-70D).
- mmap / OS page cache instead of pread: KILL, mmap ~5x slower; MTLIOCommandQueue +21-23 % slower (B4).
- Split one expert read into parallel chunks, or zero-copy preadv into the cache slots: KILL. Reads are latency-bound
  (512 KB chunks 0.39 -> 0.46 ms each), and unaligned F_NOCACHE destinations doubled read time. One 16 KB-aligned record
  per expert: ADOPTED (+7 %). Batched, sorted read submits (SlotStream claims +30 % bandwidth): PARKED.
- Pin the hottest experts permanently: KILL, worse than LRU in simulation (13.7-14.3 vs 13.61 stalls per token).
- Bigger expert cache: ADOPTED as RAM allows: misses -31 % from C 224 to 256 (replay); engine C 256
  with per-layer sizes 63.6 tok/s steady (41.7 GB, exact); C 300-320 need ~55 GB free.
- LFU instead of LRU: ADOPTED at 24 GB (decayed LFU 0.99: +13 % code); at 45 GB LRU has ~9 % fewer misses, so it stays LRU.
- Learned / Belady-trained eviction, popularity prior: BETWEEN, -9.6 to -10.5 % misses. Belady's 56 % headroom needs 5-20
  checks of foresight; a perfect next-check predictor buys only ~15 % (M-14).
- Per-layer cache sizes: ADOPTED at 45 GB (engine: -17.2 % SSD bytes per kept token, +8.3 % tok/s, exact; M-17); it died
  in the 24 GB engine (-1.5 %, R-27).
- Overlap SSD reads with resident compute: ADOPTED, +18.7 % (A-B-A), exact (OOO, R-56d).
- Runahead (compute the next block with misses skipped while they load, re-check exactly): BETWEEN. Under the 16-row cap it
  models -7 .. +9.6 %; it needs a re-check cost <= 3.4 ms per row (M-16).
- Cache-aware or sticky routing, substituting resident experts for missing ones: KILL, non-exact and small (reads -1-4 %;
  I6, SWITCHCAP; substitution not lossless). Waiting only for each token's top-2 experts: BETWEEN, +26 % but non-exact
  (GSM8K 19/20).
- Lossless entropy coding of expert records (DFloat11, ZipNN, Huffman): BETWEEN, ~8.5 % at the ideal bound, decoding needs
  4-7 CPU cores.
- RAID-0 striping onto an external NVMe: PARKED (hardware, Kamen's call).

## Bytes per token (precision, fewer experts)
- 3-bit experts by routing rank (ranks 0-1 at 4 bits, 2-9 at 3; HOBBIT / MC-MoE style): ADOPTED as the 45 GB default
> (a quote from the project owner, kept private)
- All experts at 3 bits (3333333333): KILL, KLD x1.24-1.83 over the strict gate; ranks 0-1 at 4 bits are load-bearing.
- 2-bit anything (tails 4433332222, 2-bit trellis codes, 2-bit rare experts): KILL: over the KLD gate; 2.49x MSB-3's error
  held out (B10); rare-expert 2-bit saves only 4.3 % of SSD bytes (M-12).
- Fewer experts per token (top-8, budgets; MoE-Spec, AcceptMoE): KILL, top-8 fails the strict gate with or without
  renormalization (M-09 / M-09b / M-09d).
- Fewer experts only on rows the model is sure of (decision map, lossy): top-2 changes 0.21 % of sure code tokens if the
  kept weights are not renormalised (2.33 % renormalised), top-5 0.00 %; unsure rows 9-44 %: lead, not exact.
- Reuse the previous position's routed-expert output (P-frame): one 12-layer group in 12-35 changes 0.10-0.14 % of sure
  code tokens, all of 12-35 at once 1.03 %, 2 / 4 rows old 2.46 / 4.76 %, chained 8.11 % (KILL); 0-7 / 36-47 19-23 %:
  lead only for 'at most 1 row old' on ~8-35. Reusing attention (29 %) or GDN (88 %) outputs: KILL.
- Expert pruning or merging (REAP etc.): not allowed (Kamen: all 512 experts stay); REAP-288 HumanEval 93.9 -> 91.5.
- Bias / low-rank correction of the 3-bit tier (DFQ, QERA): BETWEEN, block error -21.6 %: a quality knob, not bytes.
- Neuron-level sparse reads (Deja Vu style) or projected experts: KILL, activations not sparse enough; inputs keep only
  60-79 % of their energy in an expert's own basis (I3, I2).

## Tokens kept per check (drafter)
- Draft trees (EAGLE-2, Medusa, SpecInfer; Bole / TreeWY for GDN): KILL, best tree vs best chain -10 .. +2 % at equal cost
  (TREES, D-24); rows are cheap but extra branches are rarely kept.
- A bigger drafter (4-8x DFlash2): not trainable locally; a paid run (~$650-1,050) is estimated at +8-20 %: Kamen's call.
- Self-speculation, layer skip, early exit: KILL, an AR step reads the 3.44 GB dense part (~170 ms for 15 steps); exiting
  after layer 40 agrees with the full model on 5.7 % of code positions (D-07, L1-09).
- Suffix / n-gram / prompt-lookup drafting: KILL on Kamen's agent turns (SUFFIX).
- Relaxed, near-tie or typical acceptance: BETWEEN and non-exact, chat +23 % per pass at rho 0.25; exact-loop form +14 %
  (R-61N, M-15).
- Exact speculative sampling instead of greedy matching (accept with min(1, p/q)): KILL, per-token acceptance only
  +0.003 to +0.014 on IFEval at the official sampling settings, code / GSM within +-0.001 (A1, Oct 23).
- Fixing deep drafter drift: KILL on paper, no depth decay at all gives only +3 % at block 16; DFlash2 already holds
  ~0.93 per token at shallow depths (SD-AR).
- EAGLE-3 training-time test on the model's own MTP head: PASS, code 10.98 tokens kept per check vs DFlash2 10.44 (D-22).
- Acceptance-targeted losses (LK losses, Spec-AUF): +0-3 % at greedy (b5).
- Cut the verified chain by drafter confidence: ADOPTED (t 0.3), already optimal; calibrated stop rules add nothing.
- Drafter switch by text type: KILL on the new default as built (-2.7 %, D-23); with the MTP chain cut on chat only:
  PASS, chat +24.5 % exact on 23 fresh prompts (multilingual +27 %; D-23d); agent flat (more cold experts), stays on DFlash2.
- MTP head drafts position 1 (or 1-2), DFlash2 the rest: KILL (D-18), code -2.5 %, chat / agent below the MTP chain alone;
  the native MTP chain alone beats DFlash2 on chat (1.08x IFEval, 1.32x multilingual, D-17).
- Drafter trained on labels from past sessions: KILL, no held-out gain (REGENFREE).
- More drafter data where it is weak (the 125B's own de/ru/ja answers), or borrowing resident experts inside the drafter:
  KILL, sure-token hits +0.02 / +0.01 vs lines +0.05 / +0.1 (E1, E5).

## Check time (kernels, syncs)
- Batch-invariant kernels + fixed key buckets (Thinking Machines style): ADOPTED (REF2b), enables bit-exact 16-row checks.
- Sync-free expert gate (expert ids as GPU data): ADOPTED, +35-48 %.
- M5 matrix units for the expert matmul: KILL at the 45 GB point, +1.8 % (SSD-bound, short checks); parked for wide checks.
- Bit-identical custom expert / small-M kernels (dequant-once, register prefetch): KILL, 1.6-2x and 16-47 % slower than MLX.
- mx.compile of the drafter or the verify graph: KILL, -0.7 ms (graph build is only 0.46 ms).
- Exact-safe kernel fusion (router, MoE tail, RoPE, residual): ADOPTED, ~-1,100 of 3,944 kernels per check, +2.4 % (K-73).
- Fused or block-sparse attention: BETWEEN, fused ~ the current single call; sparse saves a few ms only at 10-20K keys.

## Memory, prefill, quality, other
- Fewer-expert ('frugal') decoding healed by LoRA, pre-routing or background repair: KILL, LoRA closes ~19 % of the gap;
  repair keeps KLD 0.107 vs 0.064 (LORAHEAL, I7, I5).
- KV quantization + sparse attention at long context: PARKED, lossy, pays only near ~100K tokens.
- Several streams in one pass: BETWEEN, -36 % bytes per token but only as total throughput (R-63S).
- Prefix checkpoints for repeated prompts: noted (first-token time only).
