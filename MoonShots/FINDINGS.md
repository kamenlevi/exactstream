# MoonShots FINDINGS (night 10-20 -> 10-21, account-1; details and data paths in IDEAS.md rounds 32-32i)

## Delivered (main track, 3-bit tier engine)
- records3 converter + verifier (BigMoE converted all 48 layers with it); one-dispatch 3b rebuild (bitwise, 1.3-1.7x).
- 3c fused tier GLU with the block's combine inside: bit-identical to BigMoE's 3b + MLX's combine (MLX sums the 10 ranks
  in bf16 as col_reduce_small with 8 row-threads; reproduced exactly). -22..-28 % per layer vs stock 4-bit.
- **Exact fused GDN verify prework** (patches/planes_engine/gdn_exact.py, 02:10): conv + silu + l2norm + scale + conv state
  in one dispatch, bitwise == MTPLX's eager chain (S=1..8); drop-in for MTPLX_FUSED_GDN_CONVNORM / _CONVNORM_VERIFY,
  whose own kernel (+9 % in LEDGER GDN1) was not exact. Plus an exact gated output norm (rms_norm + sigmoid gate, one
  dispatch, bitwise == SigmoidRMSNormGated). Together -13 chain kernels per GDN layer (~-3.5 ms/pass at ~7.6 us each).
  **Model-level: decode + 4-row verify logits bitwise == the eager reference under BigMoE's exact env** (Wed 09:00).
- Out-of-order experts (resident pairs during the loader spin, misses after, exact combine): bitwise for any hit mask,
  engine form with GPU phase flags (no host sync). Ceiling ~30 % of loader stalls hidden (~+7-10 %). Needs a partial-hit
  slot row from BigMoE's ext. Measured: post-spin path at M=4 0.314 -> 0.058-0.122 ms/layer (~9-12 ms/pass when layers
  stall); MoEPrefetch is integrating it.

## Research conclusions (exact-mode physics of the 125B on the M5 Pro)
1. **Ceiling ~94 tok/s at today's ~3.1 tokens/pass** with every exact lever (calibrated pass model: 21.9 vs 23.7 measured at
   17 GB, 32.6 vs 32.9 at 34 GB). Kernel-side levers give +30 %; the rest needs engine structure: compute at the read
   ceiling incl. the dependent-kernel floor (46 -> ~29 ms/pass), inter-pass 17 -> ~4 ms, all SSD traffic hidden.
2. **Tokens per pass is the lever past ~95 tok/s**, and draft depth is not (flat optimum at D = 3-5; MTP's acceptance caps
   ~3.5-3.9 tokens/pass). Needed: a better drafter (lookup + MTP) or wider trees (sibling-union probe tonight).
3. **Caching is near its practical limit.** Belady OPT would cut SSD bytes 37-49 %, but most of that needs foresight of
   8-16 passes (25-50 tokens): perfect 1-2-pass prediction gets only 7-13 %; every online policy tried (LIP, SRRIP,
   reuse distance, context matching) gets <= ~6 %.
4. **Wide verify amortizes weakly:** expert union over k consecutive tokens ~ k^0.67 (32 tokens: only 3.1x fewer bytes/token).
5. **Inter-pass time** is mostly the ~14 ms after the last (lazy) draft call: the 3 MTP draft steps' GPU work + verify
   launch + layer 0 (split Wednesday with DRAFT_SYNC).
6. **Small-memory goals:** at 6-12 GB the 125B is SSD-bound (12.7 GB: ~62 tok/s with every lever but OPT); a 3 GB/s
   Mac mini tops out ~9-14 tok/s at 4-bit-class bytes, so it needs ~2-bit-class experts that keep quality.

7. **The verify pass is latency-bound on its dependent kernel chain:** 2,707 of 4,902 kernels per pass lie on the
   critical path (~55 per layer: MoE 12, two hyper-connection blocks 10 each, GDN attention 20, router 3); 55 x ~7.6 us
   = the measured per-layer excess over bytes (0.42 of 0.89 ms). Exact chain fusion is the largest exact compute lever. The HC part already
   has an exact fused read (BIGMOE_HC_VREAD); next targets: GDN attention (~20/layer) and the MoE part (3c+combine).
   Re-measure the chain under the reference flags first (day_runner14 job 5).

8. **Tree width is not cheap:** 4 sibling candidates use 21.2 experts/layer vs 26.1 for a 4-token chain (one token 10),
   and the top-1 candidate usually holds p 0.8-1.0.
9. **Between passes:** without the reference flags ~18.5 ms = verify tail 2.9 + drafts 5.5 (1.9/step) + ~9.8 ms from
   the last draft to the GPU at layer 0. **Under the true reference (ASYNC_EVERY=1) that gap is already pipelined:** the
   verify call starts 0.9 ms after the previous gate and reaches layer 0's gate ~3 ms later (data/verify_start125.txt).
   Not a lever.
10. **Reference chain** (BigMoE's exact env, ASYNC_EVERY=0 export): 3,589 kernels/pass, critical chain 1,938 (~40/layer;
   HC already fused). Measured cost of the exact GDN prework's ~10 chain kernels: 0.048-0.055 -> 0.014-0.019 ms/layer,
   **~1.2 ms/pass** (36 GDN layers; ~3.5 us per dependent small kernel, not 7.6): the exact GDN pieces are worth ~1.5 %.

11. **Path to the main-track 80 tok/s at 24 GB** (scripts/path80_125.py; measured 24 GB pass ~110 ms = stalls ~70 +
   compute ~40; 765 MB/pass at ~11 GB/s effective): 3-bit tiers 33 tok/s, + out-of-order 38; 3-bit tiers + FULL loader/
   compute overlap 57 (+ online caching limit 61); **2-bit-heavy tiers (4222222222 on records4, -45 % bytes) + full
   overlap 78, + compute at its floor 81.** So 80 needs both full overlap and ~-45 % SSD bytes; only 2-bit-heavy tiers
   give the bytes (caching can't). records4 becomes the main-track path if 4222222222 (or similar) passes quality.
   **Prototyped (Wed 11:21):** make_records4.py (lossless from records3, layer 0 verified) + tierglu4.py (tiers 2/3/4 from
   the planes, bitwise to MLX per tier; v3 stays bitwise on records4). Missing: BigMoE's 3-pool engine side.

## Decisions
- **2+1+1 record format:** not worth it for 4433332222 (-6..-7 % bytes vs records3 + 4433333333, +46-49 % SSD reads);
  it IS the main-track path to ~80 tok/s at 24 GB with a 2-bit-heavy plan (4222222222 -41..-52 %, 4422222222 -28..-35 %
  bytes), so the quality runs of those plans decide it (item 11).
- Cache-policy research line closed (finding 3).
- 3c load tuning closed: up-front prefetch slower, ushort loads / geometry within noise; v1 stays (-23..-28 %).

## Method: exactness-preserving fusion against MLX (what made 3c, the combine, the GDN prework and the gated norm bitwise)
A fused kernel stays bit-identical to MLX's unfused ops if it reproduces, per output, (1) every rounding point (each op's
output dtype: bf16 intermediates round, fp32 ones don't), (2) every reduction's topology and order, (3) the exact
elementwise implementations. Pinned for MLX 0.32.2 on this GPU (probes in patches/planes_engine):
- **bf16 elementwise:** MLX's op structs on native `bfloat` (Sigmoid = 1/(1+exp(|x|)) form, Multiply x*y) round per op;
  copy them verbatim (TgSigmoid/TgMultiply). mlx_lm swiglu / nn.silu = Multiply(x, Sigmoid(x)) then Multiply(., up).
- **qmv (gather_qmm, M=1 per expert):** K % 512 == 0 -> qmv_fast lanes (16 values/lane, 512-value blocks), else qmv_impl
  (8 values/lane, 256 blocks + clamped tail); load_vector's x scaling and qdot's expressions verbatim, block-ordered
  accumulation, simd_sum, one bf16 write. Code bits may come from registers (rebuilt planes) without changing results.
- **Sum over a small middle axis (the MoE combine, 10 ranks):** col_reduce_small with 8 row-threads: ranks j, j+8, ...
  accumulated `val + total` in bf16 from 0, partials combined j = 1..7 (combine_probe.py).
- **Row sum over 128 fp32 values / rms_norm over 128:** lane l takes 4 contiguous values in order, then simd_sum
  (gdn_probe.py); rsqrt = metal::precise::rsqrt; rms_norm out = Multiply(w, bf16(x * inv)) in bf16.
- **Depthwise conv1d k=4 in bf16:** any fp32 accumulation order is exact (bf16 x bf16 products are exact in fp32), one
  bf16 rounding at the output.
- **Python scalars** become bf16 operands (weak typing): pass `mx.array(1.0, bf16) * s` to the kernel, not a C literal.
Tests compare with `.view(uint16)` bitwise on random data at several scales and S; then one model-level logits check
under the reference env (gdn_exact_model125.py pattern).
