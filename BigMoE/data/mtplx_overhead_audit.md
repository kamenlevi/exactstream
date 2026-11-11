# MTPLX 2.12.2 AR decode overhead audit (qwen4_exp, S=1, all compile flags off)

Read-only source audit, nothing executed. Paths: `M=.venv/.../mtplx/models/qwen4_exp.py`, `G=.venv/.../mtplx/generation.py`.
Assumed env: COMPILED_VERIFY/QWEN4EXP_COMPILE/COMPILE_AR_FORWARD/COMPILED_TARGET_PREFIX=0, NGRAM_PREWARM=off, no FUSED_* set.
NOTE: `mtplx serve` (server/openai.py:1032-1060) stamps FUSED_HC_V3, GDN_INPROJ, GATE_UP, GDN_CONVNORM, GDN_STEP,
CONVNORM_VERIFY, QSA_GATHER, COMPILED_GDN=1 when unset. Calling `generate_ar` directly gets none of them, so we are on the
fully stock eager chain. MTPLX's own census (M:6501-6503): "~20.8ms of Python graph construction per token against <=14ms of GPU work".

## 1. Host syncs per token
| where | what | freq |
|---|---|---|
| G:9524 `_eval(logits_next)` (sync mode is the default, G:9413; `MTPLX_ASYNC_AR=1` moves it to G:9457) | full forward wait | 1/token |
| G:7487 `_eval(chosen, bad)` (argmax + isnan/isinf over 248K vocab) | 2nd GPU submit+wait | 1/token |
| G:7488 `bad.item()`, G:7491 `int(chosen.item())` | host read (already evaluated) | 2/token |
| M:6037 `np.asarray(input_ids)`, M:6040 `np.asarray(cache[3])` in PLE `stage` | host-born arrays, no GPU wait | 1/token |
| M:5407 `future.result()` in `_warm` (LRU misses only) | CPU waits on pread IO | ≤1/token |
| M:6047 `mx.array(new_hist)`, `mx.array(prev_np)`, M:5651 3x `mx.array(rows)` | small H2D creations | 5/token |
The model forward itself has **no per-layer sync** at S=1 on the default path (`.item()`/`mx.eval` hits are S>1/prefill only:
M:3573, 4701, 6491; M:3753 explicitly avoids one). Sync mode means the GPU sits idle for the whole ~20 ms graph build,
then the CPU sits idle for GPU time: build and execution are fully serialized. (If our streamed SwitchGLU reads `inds` on
the host, that adds 48 syncs/token of our own.) `MTPLX_ASYNC_AR=1` only overlaps host bookkeeping, not the build, because
step t+1's input token (and the PLE hash) needs the host token.

## 2. N-gram / PLE table path (layer index 1, `ple_layer_ids=[2]`)
- Called from `_forward` at M:6463, **before any graph is built** (`NGramEmbedding.stage`, M:6021-6055). Hashing in numpy
  (`_ngram_rows_np`, M:5701), 16 rows/token (ngram_size 3, 8 heads/ngram), row = 160 dims at 4-bit g32:
  80 B weight + 10 B scales + 10 B biases = **100 B/row, 1.6 KB/token** (table 32.0 GB = 320M rows x 100 B, matches files_sizes.txt).
- Read path `_rows_matrices` (M:5515): 16 <= 4096 rows, so it takes the hot-row LRU (OrderedDict, 1 GiB default, about 10.7M rows,
  `MTPLX_NGRAM_HOT_MB`). Hits are a Python loop plus 3x `np.stack`, about 50-100 us. On a miss, `_warm` (M:5400) submits one pool job per row
  (16 threads), each doing **3 preads** (planar layout: weight/scales/biases in separate tensors, so 3 random pages/row,
  up to 48 random SSD reads and roughly 768 KB of page IO for 1.6 KB of payload), waits synchronously, then memmap fancy-indexes.
  With PREWARM off, cold misses run about 0.1-0.5 ms/token, plus GIL contention.
- It is synchronous CPU work on the generation thread. It **does not stall the GPU mid-graph** (the staged design moved the old
  in-graph `np.asarray` sync, M:5668, out of the forward), but in sync mode it adds serially to every token. GPU work: one
  `mx.dequantize` [16,160] plus the PLE block (2 projections, 3 GroupedRMSNorm, conv1d over a 10-row window).
- Fixes: `MTPLX_NGRAM_ROW_FILE` (interleaved, 1 pread/row, M:5285), prewarm, or prefetching the rows of the top-k candidate tokens during GPU time.

## 3. Python per-layer work and op count (eager, S=1)
Branch and gate overhead per layer: about 10-20 `os.environ.get(...).strip().lower()` reads per layer. Examples: `_fused_hc_v3_enabled`/`_fused_hc_enabled`
x2 HC reads (M:2575-2582), `_fused_moe_decode_enabled` (M:1335), `_fused_gdn_step/convnorm/out` (M:1911-1923), `_prefill_dq_gemm_applies`
per `_linear` (M:1617), plus in QSA `_qsa_score_tile_rows`, `_qsa_flash_enabled`, `_qsa_gather_*`, `_verify_sdpa_head_chunk_enabled`,
`MTPLX_QSA_DENSE_BAND_SDPA`. That is roughly 700-900 env reads per token (about 0.3-0.5 ms). Also contextvar reads (`current_attention_phase`,
`_VERIFY_CAPTURE`, `_BOUNDARY_CAPTURE`, `in_compiled_step_body`) on every HC/MoE/GDN/attn call, module `"x" in self` lookups,
shape-check chains in every `_*_applies`, and `float32_operand`/`mx.array(-inf)`/`mx.array(True)` scalar creations in the selector (M:3664, 3715).
Approximate MLX ops issued per layer (counted from source):
- HC read (M:1289-1308): GroupedRMSNorm 3, down/div/silu(2)/up/sigmoid, 2 reshapes, mul, mean, inject linear/div/sigmoid/mul, about **16**, x2 per layer.
- HC residual write (M:596, opdiet off): about **5**, x2 per layer (one folded into the MLP read).
- GDN stock chain (M:771-984): 4 in_proj GEMVs, concat, slice+contiguous, conv1d, silu, split/3 reshapes, 2x l2norm (7 each),
  scale, sigmoid, compute_g (compiled, 1), gated_delta kernel, gated norm (6), out_proj: about **40**.
- QSA attn (M:4833-5144): q/k/v, split/reshapes, q/k norm, cos/sin table rebuilt (about 7) plus 2x partial rope (about 9 each),
  transposes, depends, KV update, SDPA, gate, o_proj: about **45**. Indexer: qk proj, norm, a second rope table plus rope, write_raw, every 4th token
  pooled extend (about 15). Once T > 2048 the eager selector `_select_eager` (M:3636-3824) adds about **35** (fp32 matmul, relu-sum, where,
  argpartition, zeros+put_along_axis, repeat to T, pad concat, 2 aranges, compares, and a [1,1,1,T] bool mask).
- MoE minus routed experts (mlx_lm qwen3_next.py:327): gate GEMV, softmax, argpartition, slice, take_along_axis, sum, div,
  weight-mul, sum, shared expert (6), shared gate GEMV, sigmoid, mul, add: about **20**.
- Totals: GDN layer about 100 ops, QSA layer about 135-170, PLE layer +25. Per token: 36x100 + 12x150 + head/mixer, **about 5.5K ops**.
  At about 3-4 us/op of Python+C++ encode this is the ~20 ms build MTPLX measured.

## 4. Fused decode kernels off by default (env flag, then exactness per MTPLX's own comments)
| flag | replaces | exact? |
|---|---|---|
| `MTPLX_QWEN4EXP_COMPILE`/`MTPLX_COMPILED_GDN=1` (M:6301) | mx.compile of contiguous non-PLE GDN runs (whole layers **incl. mlp**) | no claim; 42.6 to 51.9 t/s receipt (openai.py:1022). Traces our switch_mlp too. |
| `MTPLX_FUSED_GDN_INPROJ=1` (sanitize, M:1738) | 4 in_proj GEMVs to 1, 36 layers | **bit-exact** (row concat; openai.py:1025) |
| `MTPLX_FUSED_QSA_QKV=1` (sanitize, M:1808) | q/k/v(+indexer qk) GEMVs to 1 | **bit-exact per row** (M:4848-4851) |
| `MTPLX_QWEN4_OPDIET=1` (+`_ITEMS`, read at import) | `rope` shared half tables, `resid` HC write view-add | **value-identical** by design (M:446-454, runtime_options.py) |
| `MTPLX_FUSED_QSA_INDEXER=1` (+`MTPLX_COMPILED_QSA_INDEXER`) | Metal selector + query/pool prep | described "exact" (profiles.py:330) |
| `MTPLX_FUSED_GDN_STEP=1` (M:1064) | conv..delta..gated-norm, about 25 ops to 1 dispatch | recurrence and epilogue "exactly" mirrored; conv/l2norm part same class as CONVNORM (not bit-claimed) |
| `MTPLX_FUSED_GDN_CONVNORM=1` (M:986) | conv+silu+l2norm | **tolerance** (bf16 rounding after norm, M:992-993) |
| `MTPLX_FUSED_GDN_OUT=1` (M:1091) | norm+gate+out_proj (4-bit) | "exactly, up to accumulation order" |
| `MTPLX_FUSED_HC=1` (M:1138) | HC read, about 16 ops to 1 | **not bit-identical**, "within bf16 ulp" (kernels/hyper_connection.py:13) |
| `MTPLX_FUSED_HC_V3=1` (M:1156) | HC read rows=1, 2 dispatches | **lossy**: kernel-private 8-bit pack (KLD 0.0003) |
| `MTPLX_QSA_FLASH=1` (M:3724) | block-sparse flash over the selected set | same visible set; online softmax (not bit-claimed) |
| `MTPLX_FUSED_MOE_DECODE`, `MTPLX_FUSED_GATE_UP` | routed+shared MoE | need `_FusedGateUpSwitchGLU`, which conflicts with our streamed switch_mlp; MOE_DECODE falsified (-5%) |
| `MTPLX_ASYNC_AR=1` (G:9413) | double-buffer eval | exact; small gain only (see section 1) |

## 5. Extra copies on the per-token path
- M:6464 `mx.tile(h,(1,1,4))`: a 4x copy of the embedding row. Small.
- GDN stock: M:870 `concatenate([conv_state,qkv])` (4x10240), M:878 `contiguous` tail, M:894 fp32 casts in l2norm x2,
  SigmoidRMSNormGated fp32 round-trip (M:718-720). Per layer, 36x.
- PLE: M:6182 `concatenate([state(9 rows), x])` 10x10240 bf16 per token; M:6229 `astype`.
- QSA rope (opdiet off): `concatenate([-x2,x1])`, 2 fp32 casts, a cast back, `concatenate([x_rope,x_pass])` for attn q, k and indexer q; the cos/sin table
  rebuilt 24x/token (once per indexer and once per attn) where 1x would do. Selector: `q.astype(f32)` (M:3679), `zeros+put_along_axis` (M:3713),
  `repeat` to T plus pad concat (M:3808-3814): O(T) bool work per QSA layer per token.
- KV: `QSAKVCache` keeps stock mlx-lm step-256 growth (concat-copies the whole KV every 256 tokens, O(N^2) at long ctx);
  the doubling growth (M:3081) applies only to raw/pooled index buffers.

## Ranked: 5 most promising items to attack
1. **Serialized ~20 ms Python graph build (about 5.5K ops/token) with an idle GPU.** Compile per-layer segments *around* the streamed
   experts: [HC write, HC read, GDN/attn, HC read, router softmax/top-k, shared expert] as one `mx.compile`d fn, then our
   streamed experts, then [combine + HC write]. MTPLX's whole-layer compile (which can't host our streamer) took 42.6 to 51.9 t/s.
   Cheaper stopgap: `mx.async_eval(h, states)` every 4-8 layers (like prefill midloop, M:6489) so the GPU runs early layers while Python builds later ones.
2. **Turn on the exact/value-identical op cuts**: GDN_INPROJ (-108 GEMV dispatches/token), QSA_QKV, OPDIET rope+resid,
   FUSED_QSA_INDEXER at ctx>2K. Then the tolerance-class GDN_STEP (about -900 ops/token) and FUSED_HC (about -1500 ops/token), gated on our KLD check.
3. **Second sync per token in greedy sampling** (G:7480-7491): argmax, isnan and max run as a separate submit+wait after the
   forward eval. Build `argmax` and `bad` lazily on `logits_next` and eval them in the same `_eval` at G:9524. That saves one GPU round trip plus 3 tiny kernels per token.
4. **Per-layer Python gate tax**: about 800 env reads, contextvar reads and `_applies` shape chains per token (about 0.5-1 ms). Hoist the env flags to
   module constants (as runtime_options already does for OPDIET) and precompute a per-layer decode plan once.
5. **PLE n-gram gather on the critical path**: 1.6 KB/token, but 3 random preads per missed row (planar), a synchronous pool wait, and a
   Python LRU loop. Use `MTPLX_NGRAM_ROW_FILE` (interleaved, 1 read/row), keep PREWARM on, and/or prefetch top-k candidate rows
   during GPU time. Measure first with `ple_stage_seconds()` (M:5689). `MTPLX_AR_PROFILE_TOKENS=N` (G:3495) cProfiles forwards.
