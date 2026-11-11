# 3-bit tier engine, plan 4433333333 (adopted by Kamen 2026-10-20 14:15)

Gate ranks 1–2 (by score) run the full 4-bit expert; ranks 3–10 run its MSB-3 tier. Exact mode (default) is
unchanged: it must stay bit-identical to today's reference (v1 shas). Plan mode = BIGMOE_TIER=4433333333, reference v3.

## Plane math (groups of 64 along K; q = 2m + l, m = q >> 1 in 0..7, l = q & 1; s, b = the 4-bit scale, bias)
- Exact 4-bit weight: s·q + b.
- MSB-3 tier: 2s·m + b3, b3 = bf16(b + s·off), off = mean over the group of l (fp32 math, then one bf16 round:
  the same values the Eval suite and GSM8K runs measured via planes125._tier_proj).
- Rebuild: q = 2m + l, bit-exact, with s = s3 / 2 (exact in bf16) and the original b from the LSB part.

## Record file: models/bare/records3/L{L:02d}.bin, record e at e·RECSZ, RECSZ = MSB_SZ + LSB_SZ
Per projection p in (gate, up, down), shapes from today's pieces (gate/up [640, 2560], down [2560, 640], 1,638,400
weights, 25,600 groups):
- MSB part (offset 0, padded to 16 KB): for p: w3 = MLX 3-bit packing of m (exactly what mx.quantize(bits=3) lays out,
  614,400 B), s3 = 2s (bf16, 51,200 B), b3 (bf16, 51,200 B). Body 2,150,400 B → MSB_SZ 2,162,688.
  The MSB part is a ready MLX bits=3 expert: the slot pools feed gather_qmm(bits=3) with no conversion.
- LSB part (offset MSB_SZ, padded to 16 KB): for p: l1 = 1 bit per weight, little-endian, bit j of byte i = weight
  8i + j in row-major [out, K] order (204,800 B); b (original bf16 bias, 51,200 B). Body 768,000 B → LSB_SZ 770,048.
- RECSZ 2,932,736 B; 48 × 512 records = 72.1 GB (records/ today 63 GB). index.json like records/ plus per-piece offsets
  for both parts.
- Converter check (every expert, every layer): rebuild(m, l) == original 4-bit words bitwise; s3/2 == s; b == b.
  Plus per layer: gather_qmm(bits=3) on the MSB part == planes125._tier_proj(n=3) on the 4-bit pieces (allclose; they
  differ only by kernel), one random batch.
- Disk: write layer by layer; delete records/L{L}.bin only after records3/L{L}.bin is verified (net +9 GB, ~120 GB
  free at the end). Floor 110 GB checked before each layer. records/ is a relayout of the safetensors (regenerable).

## Engine (BigMoE)
- Pools per layer: MSB pool (C_m slots, 3-bit pieces) for every routed miss; LSB sidecar (C_l slots, l1 + b) for
  ranks 1–2. Two LRUs, one slot table each (virtual layers 2L, 2L+1 in the ext; route_and_gate with 2 ids for 2L+1).
- Exact mode on records3: the loader reads MSB + LSB of each miss and rebuilds the 4-bit pieces on the CPU into
  today's 4-bit pools (NEON interleave in the loader thread). GPU code and pools unchanged → bit-identical by
  construction. Cost: +5.6 % bytes per miss + CPU rebuild (target hidden under the read).
- Gate: ranks 1–2 must be the two highest scores (argpartition is unordered): take top-10, then top-2 by score with a
  fixed tie rule (lower expert id wins). Near-tie test: count tokens whose 2nd/3rd scores differ < 1e-3.
- 3a: ranks 3–10 → gather_qmm(bits=3) over the MSB pool. 3b: ranks 1–2 → GPU rebuild (MSB slot + LSB slot → 4-bit
  scratch rows) + gather_qmm(bits=4). 3c (MoonShots prototype): one fused routed kernel reading 3-bit + 1-bit directly.
  3b and 3c must give the same tokens (v3 reference is defined by 3b; 3c is bit-identical to 3b or it is a v3' with its
  own GSM8K check).
- Prefill: plan mode prefill uses the same tiers (v3 = tiered prefill and decode); exact mode prefill unchanged.
- Repetition guard (all modes, decode-time, output-only): stop when the last N tokens repeat a period ≤ P for ≥ R
  cycles (MTPLX loop_guard defaults); emits finish_reason "repetition". Does not alter tokens before the stop → exact
  shas unaffected on the reference prompts (checked).
- Validation: v3 shas (prose/code/edit at 0.25/96 and 0.50/64), spec == own ar2, S≤8 band and dup-row rechecks,
  GSM8K-472 with the engine (expect ~448 ± noise), near-tie count.

## Engine decisions (2026-10-20 14:40)
- Evaluated plan = tiered DECODE, exact prefill (Eval suite caveat; planes125 tiers 1-row decode only). v3 = tiers on every
  decode/verify row (S <= 8), prefill exact 4-bit.
- Rank order: full argsort of the 10 scores, descending (as planes125's block), so the combine order matches the evaluated
  variant; ranks 1-2 = positions 0-1. Plan-mode block override for streamed layers at rows <= 8 (keeps sharedfirst's
  shared-expert dep edge and the verify-phase shared lowering); prefill rows -> the original block.
- ext: rec_fmt 31 = MSB part only (into a 9-piece 3-bit pool), 32 = LSB part only (6-piece sidecar), same bounce/preadv
  code as fmt 1 with a part offset/length. Sidecar = extra ext layers (index nl+1+L), route_and_gate with ids[:, :2];
  look-ahead predictions feed the MSB layer only at first.
- Plan-mode prefill: full-E scratch from records3 with the fmt-3 rebuild (resident 3-bit slots are not usable for the
  4-bit prefill), no pool adoption at first (decode starts cold); later: warm the MSB pool from the 4-bit scratch on the
  CPU (m = q >> 1, b3 = bf16(b + s*mean(l))).
- Exact mode on records3 (BIGMOE_RECORDS=3): DONE 20499ec, bitwise test scripts/test_records3.py (layers 0, 17).
- Repetition guard: MTPLX loop_guard on by default via bigmoe.loader (BIGMOE_LOOP_GUARD=0 off). To check: v1 shas
  unchanged; ifeval/1219 with the plan stops looping; armed steering works on the spec/verify path.

## MoonShots answers (2026-10-20 ~14:45)
- Its GSM8K runs: generate_ar, MTP off, tiered decode + exact prefill, b3 = bf16(fp32(b + s*mean(l))) once. Verify rows
  in any MTP-based eval ran exact (patch tiers 1-row calls only) -> v3 tiers verify rows too: GSM8K re-check is needed.
- Budget at 17 GB (362.7 MB/layer): C_m ~150, C_l ~47 (sidecar ~10 % of bytes; 20 % was worse). Sweep C_l 32/48/64.
- ETAs: converter + 2-layer verify ~16:45, GPU rebuild kernel ~18:45, fused 3/4 GLU ~1 day (3b = equivalence target).

## Split (proposed to MoonShots 2026-10-20)
- MoonShots (own repo ~/MoonShots/patches/tier3/): converter + verifier (run on 2 layers into its own scratch; BigMoE
  runs the full conversion), MLX bits=3 packing check, 3a micro-bench at our shapes, 3c fused kernel + equivalence test.
- BigMoE: ext record3 path (CPU rebuild for exact mode, split pools + sidecar), gate top-2, 3a/3b integration, repetition
  guard, v3 reference, validation runs, LEDGER.
