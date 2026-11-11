# records4 (2+1+1 planes) for 2-bit-heavy tier plans: design (MoonShots, 2026-10-21 10:30; build only if a plan passes)

Why: the main-track 80 tok/s at 24 GB needs ~-45 % SSD bytes on top of full overlap (FINDINGS item 11); only a
2-bit-heavy plan gives that (4222222222: -41..-52 % demand bytes vs records3 + 4433333333 at 10-24 GB; 4422222222:
-28..-35 %; data/planes4_sim*.txt). Gate: those plans' GSM8K + Eval suite runs.

## Record (lossless from records3: q = 2m + l, m = 2 m2 + c, so m2 = q >> 2, c = (q >> 1) & 1, l = q & 1)
Per expert, three parts, each padded to 16 KB, one cache unit each (base pool, B3 sidecar, B4 sidecar):
| part | per projection (gate, up, down) | bytes / proj | per expert |
|---|---|---|---|
| P2 | w2 = MLX bits=2 packing of m2 (16 values per u32, value i at bits 2i, row-major), s2 = 4s (bf16, exact), b2 = bf16(fp32(b) + fp32(s) * mean(q & 3)) | 409,600 + 51,200 + 51,200 | 1,536,000 -> 1,540,096 |
| B3 | c bits (bit j of byte i = weight 8i + j), b3 = bf16(fp32(b) + fp32(s) * mean(q & 1)) (records3's b3) | 204,800 + 51,200 | 768,000 -> 770,048 |
| B4 | l bits (records3's l1), b (original bias) | 204,800 + 51,200 | 768,000 -> 770,048 |
RECSZ 3,080,192 (records3 2,932,736: +5.0 %); 48 x 512 = 75.7 GB. Disk: 120 GB free today, so records4 must REPLACE
records3 layer by layer (records3 is regenerable from records4: m = 2 m2 + c, s3 = s2 / 2, b3 stored, l1 = B4 bits).

## Tiers and their exact references (what "bitwise" means per tier)
- tier 2 (P2): gather_qmm(bits=2) on (w2, s2, b2): MLX's 2-bit qmv arithmetic (load_vector bits=2: x, x/4, x/16, x/64;
  qdot bits=2; qmv_fast lanes for K=2560: 16 values/lane, 512 blocks; qmv for K=640: 16 values/lane, block 512 + tail at
  lanes 0-7). Same weights as planes125's tier 2 (s*(q & 0xC) + b2) -> the GSM8K numbers measured with _tier_proj(n=2)
  carry over up to arithmetic-order noise (like tier 3 vs _tier_proj(3)).
- tier 3 (P2 + B3): rebuild m = 2 m2 + c in registers into MLX's 3-bit stream layout -> gather_qmm(bits=3) arithmetic
  with s3 = s2 / 2 (exact) and b3 -> **bitwise identical to records3's tier 3 (today's v3)**.
- tier 4 (P2 + B3 + B4): rebuild q = 4 m2 + 2c + l -> bits=4 arithmetic with s = s2 / 4 (exact) and b -> bitwise
  identical to the 4-bit expert (exact mode).
So v3 (4433333333) stays bitwise on records4; a 2-bit-heavy plan defines its own reference (v4) through tier 2 above.

## Kernel (tierglu.py, extend): per-rank tier masks as today; code providers per tier from planes:
P2 lane chunk (qmv_fast: 16 values = one u32; qmv: 16 values = one u32 per 512 block) -> tier 2 uses it directly;
tier 3 interleaves c bits (2 bits -> 3-bit fields: a spread step like spread8); tier 4 adds l. Arithmetic per tier is
MLX's (tg_qdot2 to add, tg_qdot3 / tg_qdot4 as today). Exact-mode CPU rebuild (ext) gets a 2+1+1 variant (BigMoE).

## Converter (make_records4.py): from records3 (no records/ left), built-in verifier as make_records3's: rebuilt q ==
records3-rebuilt q bitwise, s2 / 4 == s and s2 / 2 == s3 bitwise, b3 == records3 b3, b2 == recomputed from q, padding
zero; MLX check: dequantize(bits=2) of w2 with unit scales == q >> 2.

## Engine (BigMoE): three pools per layer (P2 for every routed use, B3 sidecar for ranks with tier >= 3, B4 for tier 4),
three LRUs; the planes4_sim numbers assume exactly that.
