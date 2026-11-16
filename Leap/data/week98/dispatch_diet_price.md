# Dispatch diet: what fusion or a replayed command path would cut (desk price)

Leap, Mon 16 Nov 2026 09:2x, for NewIdeas (08:3x, 1c-tiers piece 2). The question: the 27.9 ms CPU build + encode + commit
per C32one check is a floor for any lossy check, and the CPU should take only about a third of a 15 ms check. What would
fused kernels, or a pre-encoded / replayed command path, cut, and at what build cost? This is not for this week's runs;
it is one of the pieces 900 needs.

Labels:
- **[M] measured.** Sources: Breakthrough's CHECK55 engine profile a082f8c (exact-45, C 160, 48 pairs per arm; pass K
  attribution); MoonShots p3_m012 (08:4x); Leap's op count (below; graph export on the CPU).
- **[E] estimate** from those numbers.
- **[D] desk:** design and build-cost judgement, not tried here.

## 1. Today's cost per check [M]

| arm | dispatches | CPU build | CPU encode | read + commit | CPU total | CPU per dispatch | GPU busy | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A16 | 3,607 | 11.75 | 12.43 | 1.47 | 25.65 | 6.7 us | 59.75 | 61.11 |
| C32one | 4,022 | 12.16 | 14.10 | 1.68 | 27.94 | 6.5 us | 89.28 | 90.46 |
| C32 (split) | 6,821 | 18.91 | 19.78 | 2.26 | 40.95 | 5.7 us | 98.57 | 102.09 |

- **Per dispatch:** build ~3.0 us + encode ~3.5 us, plus ~1.7 ms fixed per check.
- **C32one per check:** 434 command buffers (SB's flushes), 267 encoders, 1,871 barriers, 576 fence waits.
- **Today the CPU is hidden.** The checks are GPU-bound (GPU busy 0.98 / 0.97 of the wall). The CPU starts to bind once
  a check's GPU time falls below ~28 ms.
- **GPU-side fixed cost per dispatch:** ~1.1 us to ~2.1 us [E].
  - The low end is pass K's att_core: 1,080 dispatches, 1.18 ms of GPU.
  - The high end is MoonShots' sync-free chain at zero bytes.
  - So C32one's 4,022 dispatches carry ~4.4-8.4 ms of GPU time that is dispatch overhead, not bytes.
  - Only fewer dispatches remove it; a replayed command path does not.

## 2. Where the dispatches are [M]

- **By kernel (C32one, per check):**
  - MLX glue (elementwise / copy / gather / select / mx.compile'd): **2,093 (52 %, 43.6 per layer)**;
  - custom kernels ~1,430 (36 %);
  - MLX GEMM / qmv / conv 354 (9 %);
  - softmax 148 (4 %).
- **Top glue (per check):**
  - copies (bf16, various strides) ~590;
  - g3 Multiply bf16 197; g2 Select bf16 148; bool copies 147;
  - bigmoe_spin 98; int32 compares 98; uint32 -> int32 casts 98;
  - then a tail of ~50s: and / max / select / sigmoid-multiply.
- **By class (pass K, A16: dispatches; GPU ms; build + encode CPU ms):**

  | class | dispatches | GPU ms | CPU ms |
  |---|---:|---:|---:|
  | moe_experts | 1,200 | 33.7 | 8.8 |
  | att_core | 1,080 | 1.2 | 4.4 |
  | hc_read | 480 | 9.4 | 4.7 |
  | gdn_core | 252 | 2.2 | 2.9 |
  | moe_shared | 144 | 2.3 | 1.9 |
  | gdn_proj | 108 | 8.3 | 2.2 |
  | moe_router | 96 | 1.3 | 1.5 |
  | moe_tail | 96 | 0.2 | 0.9 |
  | att_proj | 60 | 3.0 | 1.2 |
  | hc_write | 48 | 0.2 | 0.8 |

  att_core is almost all glue: CPU-dominated, ~1.1 us of GPU per dispatch.

## 3. The levers

| lever | what | cut per C32one check | build cost [D] | exact? |
|---|---|---|---|---|
| **A. Glue fusion** | Fold the ~44 glue dispatches per layer into ~5. MoE id glue goes into router_fused / the one-call (the eprep_k precedent). HC glue goes into hc_verify_norm / accum_gate. att_core's copies / selects / multiplies go into S4 / the GDN body. | ~-1,800 dispatches: CPU -11.7 ms, GPU -2.0 to -3.8 ms [E] | ~1.5-2.5 weeks (6-8 groups x 1-1.5 days with bitwise tests) | Yes. Integer glue trivially (eprep_k: 0 / 400 differ). Float glue with the op order and the bf16 rounding points kept (K1M, HC8-F, D5 precedents). |
| **B. Python specialization** (+ mx.compile on the pure parts) | Engine Python is ~2 us of the ~3.0 us build per dispatch: MoonShots' pure op build is ~0.8-0.9 us per dispatching primitive, and compile saved 25-30 % of it. Build a per-config forward with prebound closures and no CFG / dict branching per layer. | -1.5 to -2 us per dispatch: -6 to -8 ms at 4,022 dispatches; -3.3 to -4.4 ms after A [E] | ~3-5 days | By construction (same ops, same order); a token A/B confirms. |
| **C. Fewer command buffers** inside MLX | 434 command buffers per check from SB's flushes. Their share of the 3.5 us encode is unknown. | unknown: needs a 1-day microbench (CPU per command buffer vs per dispatch) | 1 day | yes |
| **D. Pre-encoded replay** (Metal indirect command buffers; Metal 4's lower-overhead encoding on macOS 26) | Encode once per row count; per check, only update argument buffers and submit. MLX 0.32 has no capture / replay API, so this is a custom executor outside MLX. | CPU build + encode -> ~1-2 ms per check (about -25 ms) [D]. The GPU's per-dispatch cost is unchanged. | ~3-5 weeks [D], high risk. Needs a static memory plan; every per-check value in device buffers (positions, KV length, slot tables, gate seq); ICB-capable pipelines, including our own copies of MLX's built-ins (steel GEMM, qmv, softmax, copies); KV-length grids by indirect dispatch; the loader's posts / spins as GPU-polled regions (OOO-1C already does this). | Bitwise if every kernel is the same; the copies of MLX built-ins need bitwise A/Bs. |
| **E. Megakernels** | One kernel per sub-block: ~15 dispatches per layer, ~720 per check. | CPU ~5 ms per check (720 x ~4.7 us + 1.7); GPU fixed ~0.8-1.5 ms [E] | ~4-6 weeks [D] | Kernel work everywhere; harder next to GEMMs. |

## 4. What reaches the 900 budget (a ~15 ms check, CPU <= ~5 ms) [E]

- **A + B:** ~2,200 dispatches x ~4.75 us + 1.7 ms = **~12 ms CPU**. That fits under a 15 ms GPU-bound check, but not
  in a third of it.
- **A + B + E: ~5 ms.**
- **A + D: ~1-2 ms CPU**, with ~2.4-4.6 ms of GPU dispatch overhead left over.
- **Order:**
  1. **A first.** It pays at every check size: ~-2-4 ms of GPU per check today, plus the CPU that 900 needs.
  2. **B next** (cheap).
  3. **Then D or E for 900.**
     - D is the bigger bet: near-zero CPU and no kernel rewrites, but it lives outside MLX.
     - E stays inside MLX, but it is kernel work everywhere.
- **The microbench to run first** (C, 1 day): CPU per command buffer vs per dispatch. If command buffers carry a large
  share of the 3.5 us encode, B + C may take A + B from ~12 ms to well under 10 ms without D.

## 5. Side note: 1c-tiers adds glue [M count, E cost]

- **The count** (graph export on the CPU, scripts/week98/count1c_ops.py):
  - BigMoE's _t1c (tiers_mx + the counter) plus the km path: 28 non-view MLX ops per MoE layer;
  - Leap's 2-D dup pick (b43efc4): 6;
  - total **~34 per layer, ~1,630 per check** (+41 % over C32one's 4,022).
- **At today's costs:** +10.6 ms CPU (hidden while the check is GPU-bound) and +1.8-3.4 ms GPU [E].
- **Fit12's LOSSY32 price** (50-55 ms per 32-row check) should carry ~+2-3 ms for it unless it is fused.
- **The fix:** fuse the tiering, km, counter and dup pick into router_fused (one or two dispatches per layer). This is
  lever A's MoE group.
- **Not needed for this week's gate**, which judges quality, not speed.
