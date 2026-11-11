# Patches for BigMoE's 125B ledger (MoonShots, 2026-10-18)

I checked the four rows I sent against MTPLX 2.12.2's `mtplx/models/qwen4_exp.py` (uv cache
`~/.cache/uv/archive-v0/dfadjselx9BIIozk/mtplx/`) before writing code. Two are already there:

| ledger row | status in MTPLX 2.12.2 | action |
|---|---|---|
| 16: skip the QSA indexer at ≤ 2048 context | **already done**: `QSAIndexer` returns early when `last_nb <= self.block_topk` ("dense == sparse in this regime", qwen4_exp.py ~L4064 and ~L4482) | none; drop the row |
| 15: hyper-connection fusion | **already done for the read**: `hyper_connection.py` fused read (bf16, exact path) and `hyper_connection_v3.py` (two dispatches, `MTPLX_FUSED_HC_V3=1`, **8-bit mixer weights, so not exact**, KLD 0.0003); compiled write `_hc_compiled_write` | see `hc_exp_pack/` below for an exact byte cut |
| 17: residency-aware draft depth | BigMoE's `costmodel.py` already compares MTP depth 1/2/3 per residency; the per-step version needs MTP confidence and live residency in the drafting loop | sim spec only (below) |
| 18: oMLX adaptive depth | read oMLX 0.6.3 "Lightning MTP" | not a patch |

## hc_exp_pack/: exact, fixed-rate bf16 pack for the hyper-connection mixers (new)
- **Why it matters:** MTPLX's own notes say the eager HC read is ~13.2 MB of bf16 weights × 96 reads/step =
  **1.27 GB/token**, the largest bf16 stream left, about ⅓ of the 3.78 GB a token reads. The only fused kernel that
  cuts it (v3) quantizes to 8 bits, which isn't exact.
- **Mechanism:** per 64 weights, one base exponent; per weight, sign+mantissa (8 bits) plus a 3–4-bit exponent offset.
  Groups that don't fit stay raw. Decoding is an add and a shift inside the existing fused read kernel. Bit-identical
  by construction, so the tokens are unchanged.
- **Expected:** if ~95 % of groups fit 4 bits, it's ~12.2 bits/weight → saves ~0.3 GB/token ≈ 8 % of bytes →
  ~+6–8 % decode on the exact path (more if 3 bits fit). Prior art: DFloat11 (Huffman, CUDA) and ZipNN (storage).
  I found no fixed-rate, in-kernel-decodable exponent pack on Metal.
- **Step 1 (CPU, one command, ~1 min, mmap only):**
  `~/MoEPrefetch/.venv/bin/python ~/MoonShots/patches/hc_exp_pack/check_hc_exponents.py`
  It prints the share of groups that fit and the GB saved per token. Kernel work only if it shows ≥ 0.2 GB/token.
- **Step 2 (if worth it):** extend `fused_hyper_read` (bf16 path) to read the packed planes; exactness test = output of
  the packed read equals the bf16 read bit for bit on 1,000 random hidden states per layer.

## Residency-aware draft depth (sim spec)
Per step, choose k ∈ {1,2,3} to maximise E[accepted + 1] / E[pass time], where E[accepted] comes from the MTP head's
chained probabilities and E[pass time] = base(S) + 0.5 ms × E[stalled layers | S], with stalled layers taken from
`sim.py`'s union-vs-S curve at the current residency. Exact: only the draft length changes. SpecDecode found the
confidence cut worth nothing with everything resident (§10); under streaming each extra verify row costs ~4–5 more
stalled layers (sim: 4.0 → 8.7 at S = 1 → 3), so a low-confidence second draft is the first thing to drop.

## Measured 2026-10-18 13:5x (read-only, ~/BigMoE/models/bare)
- HC mixers: 1.28 GB, 99.5 % of groups fit a 4-bit exponent offset → 12.16 bits/weight → **−0.30 GB/token, exact**.
- Expert scales/biases (bf16, 7.5 GB stored): 100 % fit a 3-bit offset → 11.1 bits → −2.3 GB RAM (more experts resident) and −3–4 % bytes per streamed expert.
- Kernel plan: planes `hi` (u8 sign+mantissa) and `off` (u4 exponent offset) per weight, plus `base` (u8) per 64 weights.
  Decode: bf16 = (sign<<15) | ((base+off)<<7) | mantissa. Escape groups (0.5 %) keep a raw bf16 copy and a flag bit.
  Exactness test: unpack(pack(W)) == W bit for bit for every tensor, then the fused read's output equal to the bf16 read's on random inputs.

## Exactness caveat for hc_pack.py (2026-10-18 14:0x)
`hc_pack.fused_hyper_read_packed` is MTPLX's `fused_hyper_read` (the `MTPLX_FUSED_HC=1` path) with the weight loads swapped for the
decoder. So it is **bit-identical to MTPLX's fused HC read**, but MTPLX's own docstring says that read is *not* bit-identical to the eager
chain (its dot-product lane order differs from MLX's GEMV tiling). BigMoE's exact baseline is the eager chain (`MTPLX_FUSED_HC` defaults to 0),
and LEDGER row A2 already saw other fused flags diverge from token 1.
Steps in order:
1. A/B `MTPLX_FUSED_HC=1` alone against the baseline, comparing token hashes. If the tokens stay identical on the test prompts, hc_pack is a drop-in on top of it.
2. If they diverge, the exact route is a packed kernel that copies MLX's bf16 GEMV reduction order (mlx/backend/metal/kernels/gemv.metal:
   block/thread split along K and the simd_sum order) for the down/up/inject matvecs, with the norm and activations kept as separate ops.
   That's more work. The 0.30 GB/token saving is the same.
