# planes_engine: converter and kernels for the 3-bit tier engine (MoonShots side)

Spec: `~/BigMoE/DESIGN_3bit.md` (plan 4433333333, records3 [MSB-3 | LSB-1]). BigMoE owns the engine (`bigmoe/tier.py`).

| file | what | status |
|---|---|---|
| `make_records3.py` | records/L{L}.bin -> records3/L{L:02d}.bin + index.json, built-in verifier | layers 0, 17: 0 mismatches on every check (~4.6 s/layer) |
| `test_make_records3.py` | synthetic layer: convert + verify, then 6 single-bit corruptions must fail | PASS |
| `rebuild.py` | 3b GPU rebuild: one dispatch for all pairs x 3 projections, reads the pools by slot | bitwise PASS; 1.3-1.7x BigMoE's 3b rebuild |
| `test_rebuild.py` | bitwise vs BigMoE's 3b rebuild and vs the original 4-bit pieces; ms vs BigMoE's path | PASS |
| `tierglu.py` | 3c fused routed GLU over rank tiers (4/3/2), two dispatches per layer | bit-identical to 3b; -18 to -25 % vs stock 4-bit |
| `test_tierglu.py` | activation exhaustive; h and per-rank y bitwise vs BigMoE's 3b path; 2-bit tier | PASS (layers 0, 17) |
| `bench_tierglu.py` | stock4 / all3 (3a) / 3b / 3b-rs / 3c, chained 48 layers, M = 1, 2, 4, 8 | see below |

## Converter (make_records3.py)
```
make_records3.py --layers all --out ~/BigMoE/models/bare/records3 [--delete-src] [--min-free-gb 110]
```
- Layout exactly as DESIGN_3bit.md: MSB part (w3, s3, b3) x gate/up/down back to back from 0, padded to 16 KB
  (msb_sz 2,162,688); LSB part (l1, b) x 3 back to back from msb_sz, padded (770,048); recsz 2,932,736.
- index.json: `{"align": 16384, "format": "records3", "layers": {"L": {"file", "E", "recsz", "msb_sz", "lsb_sz",
  "msb_piece_off", "msb_piece_size", "lsb_piece_off" (from the record start), "lsb_piece_size", "src_recsz",
  "verified", "mismatches", "halve_bf16_ok"}}}`.
- Bit math on packed words (no unpacking): w3 = 3 bytes per 4-bit word, l1 = 1 byte per 4-bit word, mean(l) =
  popcount of 8 l1 bytes (one u64 per group). b3 = bf16(fp32(b) + fp32(s) * popcount / 64), round-to-nearest-even once.
- Verifier (every expert, from the bytes on disk via F_NOCACHE): rebuilt words == originals; s3 / 2 == s (fp32 halving
  and BigMoE's exponent-decrement halve_bf16, reported separately: a nonzero halve count means subnormal scales); LSB
  b == b; b3 recomputed from the rebuilt words == b3; padding zero; s, b finite. Per layer: unpack/packbits reference
  records for 4 random experts, byte for byte; MLX dequantize(bits=3, unit scales) of w3 == q >> 1; b3 == planes125's
  bias computed by MLX; gather_qmm(bits=3) on the MSB part vs planes125._tier_proj(n=3) (allclose, max rel reported).
  Once per run: mx.quantize(bits=3) of a known matrix lays out its codes exactly as w3.
- `--delete-src` removes records/L{L}.bin only after its layer passed every check including halve_bf16.

## 3c kernel (tierglu.py)
- A: one threadgroup = one (row, rank) pair x 4 output rows; simdgroup 0 the gate rows, 1 the up rows; lane l holds
  values [16l + 512b, +16) (MLX qmv_fast); h = Multiply(Multiply(g, Sigmoid(g)), u) in bfloat16 (mlx_lm swiglu).
- B: one threadgroup = one pair x 8 down rows (2 simdgroups x 4); lane l holds [8l + 256b, +8) plus qmv_impl's clamped
  tail at 512..639 (MLX qmv, the kernel gather_qmm runs for K = 640).
- Tiers are compile-time masks over ranks: 4 = q = 2m + l rebuilt in registers (spread8), scale s3 * 0.5, LSB bias;
  3 = the MSB part as is; 2 = MSB codes with the low bit cleared (0xDB6DB6 per 24 bits) and a b2 bias pool.
- Output: per-rank y [M, R, D]; the block's combine (y * scores).sum(-2) stays MLX's, so 3c == 3b bitwise if A and B are.

## Results (2026-10-20 17:39-17:41, quiet lock; data/planes_engine/)
ms per MoE layer's routed GLU incl. the combine, 48 chained layers per command buffer, synthetic pools (C_m 96, C_l 48),
plan 4433333333 (GB/s of unique expert bytes):

| rows M | stock4 (today) | all3 (3a only) | 3b (BigMoE) | 3b + rebuild.py | **3c** | 3c vs stock4 |
|---|---|---|---|---|---|---|
| 1 | 0.132 (209) | 0.112 (191) | 0.225 (102) | 0.183 (126) | **0.108 (214)** | -18 % |
| 2 | 0.230 (230) | 0.196 (207) | 0.421 (104) | 0.318 (138) | **0.186 (238)** | -19 % |
| 4 | 0.408 (231) | 0.353 (207) | 0.773 (103) | 0.569 (139) | **0.329 (242)** | -19 % |
| 8 | 0.757 (206) | 0.657 (184) | 1.441 (92) | 1.047 (126) | **0.566 (232)** | -25 % |

3c moves 17 % fewer bytes than stock4 and runs at the same or higher GB/s, so the byte saving shows up as time; 3b loses
it to the rebuild traffic and its ~25 dispatches. Correctness: tierglu h and every rank's y == BigMoE's 3b bitwise
(layers 0 and 17, M = 1, 2, 4, 8, plans 4433333333 and 4433332222); activation == mlx_lm swiglu on all 65,536 gate values.

## Fused combine (20:33, quiet lock): dispatch B can also do the block's (y * scores[..., None]).sum(-2), bitwise
MLX's GPU sums the 10 ranks in bf16 as col_reduce_small with 8 row-threads (`combine_probe.py`: ranks j, j+8, ...
accumulated `val + total` from 0, partials combined j = 1..7; 0 mismatches at M = 1-8, every other order mismatches).
`tier_glu(..., scores=w)` (ly=8 default) reproduces it: one simdgroup per rank, bf16 outputs in threadgroup memory, then
the combine in that order. Bitwise == the block's combine on tier_glu's per-rank y (M = 1, 2, 4, 8; test_tierglu.py
section 4). It also passes from BigMoE's converted records3 (records/ was deleted by the full run).
ms per layer (same harness, this run's machine state): M=1/2/4/8 stock4 0.144/0.246/0.441/0.824, 3c 0.119/0.201/0.342/
0.594, **3c+comb 0.110/0.191/0.333/0.583** = -24/-22/-24/-29 % vs stock4.

## Out-of-order experts (~23:50, silent night: correctness only)
`tier_glu_split(x, msb, lsb, msb_slot, lsb_slot, hit, scores)`: the resident (hit) pairs run first (subset kernels A/B
read the full pair index from pidx), the missed pairs after, then a combine-only kernel in MLX's LY=8 order reading both
partial outputs. == `tier_glu(..., scores=)` bitwise for hit fractions 1.0 / 0.9 / 0.5 / 0.0, M = 1-8, plans
4433333333 and 4433332222 (test_split.py). Engine use: issue the hit call at t_seen (before the spin), the miss call +
combine after t_rel. Ceiling from pass_profile125: hides ~9.4-10.5 ms of 31-34 ms stalls per pass at frac 0.5
(data/ooo_bound125.txt). Timing of the extra dispatches: pending a daytime quiet lock.
Engine form (no host sync, bitwise too, test_split.py): `y1 = tier_glu_phase(x, ..., msb_slot_pre, lsb_slot, phase, 1)`
before the spin, `y2 = tier_glu_phase(x, ..., msb_slot, lsb_slot, phase, 2)` after it, `combine_phase(y1, y2, phase,
scores)`. phase [M*R] int32 on the device: 1 = the pair's experts were resident at routing time (4-bit ranks: MSB and
LSB slot), else 2; phase-1 slots may be -1 for non-resident pairs (never read). Threadgroups of the other phase exit at
once. **What BigMoE's ext would supply:** before the spin, a device array of per-id slots for experts already resident
and pinned for this pass (-1 otherwise): the `dep` fast-path row generalized from "all found" to partial hits, for the
base pool and the sidecar; phase = mx.where((pre_m >= 0) & (tier != 4 | pre_l >= 0), 1, 2), lazily.
Dispatches per layer: 5 (A1, B1 during the spin; A2, B2, combine after) vs 2 fused; cost pending (bench_split.py).

## Note (TrainLab, 2026-10-22): copy-free strided rebuild for prefill
TrainLab's decoupled B12 prefill found the per-layer router readback cheap (0.06 s per 2.6K-token prompt) and the real cost
in data movement before rebuild4 (byte-wise strided slicing of each 64-expert group, ~177 MB). A copy-free strided variant
of rebuild4 (reads the 3-bit / 1-bit words in place from a [G, recsz/4] uint32 ring; record stride and piece offsets as
template constants) cut the MoE phase 3.95 -> 3.3 s (prefill 426/474 tok/s at 11.8 GB, exact), bit-identical to rebuild4
by sha: ~/TrainLab/d1i/v3f/bigmoe_v3f/tier.py (_RB2_SRC, _rebuild_gpu2), ~/TrainLab/d1i/PATCH.md section 6.
