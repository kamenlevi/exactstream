# Ledger candidates for BigMoE (125B Q4 → ~100 tok/s), pre-draft 2026-10-18

Drafted from BigMoE FINDINGS before its per-token split exists; estimates will be re-scored against the split.
Each row: item | trick | source | estimate | lossless.

| item | trick | source | est. | lossless |
|---|---|---|---|---|
| MTP verify width | pad verify passes of 6–15 rows to 16 (MLX 0.32.4 qmm: 12 rows 2.69x, 16 rows 1.37x a 1-row pass) | MoonShots skinny.py (measured) | only when drafts reach 6+ tokens; ~0 at k=2/3 | yes |
| MLX version | run qwen4_exp on MLX 0.32.4 instead of MTPLX's pinned 0.32.2 (SpecDecode measured +17 % decode, +25 % prefill on the 35B going 0.32.0 → 0.32.4) | SD §14 | +5–15 % (partly already in 0.32.2) | yes |
| full attention (12 layers, head_dim 256) | `MLX_SDPA_BLOCKS` tuning of the 2-pass SDPA kernel at long context | mlx 0.32.4 release notes | 0–3 % at > 16K context | yes |
| drafter | DFlash block-diffusion drafter instead of or alongside MTP, if one exists for Flash-Next | z-lab/dflash | +10–25 % tokens/pass on fresh text | yes |
| gate wait (CPU release, ~69 µs/layer) | put the loader/spin thread in a real-time `os_workgroup` (as audio threads do), user-interactive QoS, `beginActivity(.latencyCritical)` against App Nap and timer coalescing; pin to P-cores | Apple Audio Workgroups docs; game-engine job systems | 5–20 µs per gated layer → 1–3 % | yes |
| SSD stall (cold expert ~0.4–0.5 ms) | read the stalled expert's three matrices as separate parallel preads and start the gate/up matmul when its pieces land, so down's read overlaps compute (split-phase within an expert) | GPU streaming / virtual texturing practice | up to ~⅓ of each stall → 3–8 % | yes |
| SSD stall | stall-aware ordering: within a stalled layer compute the resident experts and the shared expert first (BigMoE next-exp 3) | BigMoE | ≤ 12 % of stall time | yes (same reduction order) |
| residency | pin layers 0–2 (least predictable, MP) and size per-layer budgets from LRU curves (BigMoE next-exp 4) | MoEPrefetch / BigMoE | 1–3 % | yes |
| page cache | `F_NOCACHE` everywhere so the page cache doesn't steal RAM from the expert pool | BigMoE already | — | yes |
| Python/launch overhead | `mx.compile` the per-layer non-expert graph; the gate loop in C++ | SD §10 (Python 2–3 % of spec time) | 1–3 % | yes |
| hyper-connections (4 residual streams) | fuse the HC mixers into one kernel per layer (MTPLX "trunk compile") | MTPLX | 1–3 % | yes |
| memory budget | raise `iogpu.wired_limit_mb` for 65 % resident (Kamen must run sudo) | BigMoE next-exp 4 | misses −20–30 % → 5–10 % | yes |

## Sweep A, SSD-streamed MoE (13:0x)
| item | trick | source | est. | exact | note |
|---|---|---|---|---|---|
| SSD stall | out-of-order expert compute: resident experts first while misses stream | ExpertFlow 2410.17954, DALI 2602.03495 | 2–3 % | yes | = BigMoE next-exp 3 |
| SSD stall | prefetch 2–3 layers ahead with residual-corrected prediction (DALI claims 93 % on Qwen) | DALI 2602.03495, PreScope | 1–3 % | yes | complements BigMoE v3 (depth-2) |
| residency | one slot pool shared by all 48 layers (busy layers borrow from quiet ones) | Slotstream (carloslfu): **same 125B model, ~12 tok/s on a 48 GB M5 Pro** | 1–2 % | yes | Slotstream is direct prior art: check its design |
| SSD stall | split read: gate/up first, start the matmul, down streams behind | LLM-in-a-Flash (inverted bundling) | 0.5–1.5 % | yes | |
| eviction | learned / workload-aware eviction instead of LRU | FlashMoE 2601.17063, HOBBIT | 1–2 % | yes | gap to OPT mostly unreachable online (2608.07911) |
| MTP | prefetch experts for the drafted tokens before verify | SP-MoE, DraftExpert, MoE-SpeQ | 2–4 % claimed | yes | **caveat:** needs the drafts' main-model routing, and MP measured router-on-draft-embedding recall at only ~0.2. Weak unless the MTP layer's own routing predicts it |
| MTP | choose draft length per step from the miss cost of the union | AcceptMoE, ECOSPEC | 1–3 % | yes | |
| lossy | 2-bit copy on miss (HOBBIT); buddy/low-rank substitutes (BuddyMoE, SPICE) | | 3–8 % | no | MP's substitution already beats these |

## Sweep B, Metal decode micro-opts
| item | trick | source | est. | exact | note |
|---|---|---|---|---|---|
| launch / sync | `METAL_FAST_SYNCH=1` (MLX opt-in faster CPU↔GPU sync) | mlx issue #3142 | 1–4 % on the gate's syncs | yes | **cheapest test for BigMoE's gate**; issue reports hangs with JACCL only |
| launch | sweep `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` | mlx issue #3142 | 1–3 % | yes | interacts with BigMoE's commit-after-router design |
| norms | fused add + RMSNorm (residual add folded into the norm kernel) | llama.cpp b11177; mlxcel #905 | 1–3 % | yes (check bit-equality) | |
| MoE layer | fuse norm→router→gather-qmv→act→down per layer | BaseRT 2607.00501 | 3–8 % dense; BaseRT got 0.90–1.01x on MoE | yes | skip: BaseRT's own MoE numbers are flat |
| geometry | per-chip launch geometry for small expert qmv on M5 | BaseRT 2607.19438 | 2–5 % on expert matvecs | yes | |
| megakernel | persistent per-token kernel with grid barriers | Gemma-4 Metal megakernel | 5–15 %, high effort | yes | incompatible with CPU-signalled gates unless gates live in-kernel |

## Sweep D, speculation under streaming
| item | trick | source | est. | exact | note |
|---|---|---|---|---|---|
| MTP depth | **residency-aware draft depth**: pick k ∈ {1,2,3} per step by P(accept) ÷ (expected non-resident experts in the union × 0.5 ms) | EVICT 2605.00342, Limits of Speculation 2609.22156, ECOSPEC | +5–12 % | yes | SD found this useless when all experts are resident (§5.3), but streaming makes each new expert ~10x costlier, so the trade flips. Testable in BigMoE's `sim.py` (has S) |
| verify prefetch | while layer l runs in a verify pass, run layer l+1's gate on all S rows and prefetch the union's misses | MoE-SpeQ 2511.14102, SP-MoE 2510.10302 | +5–15 % claimed | yes | = BigMoE's look-ahead extended to S rows; check it's already applied per row |
| lossy | cap verifier experts by routing mass or residency (MoE-Spec, AcceptMoE 2.06x) | | +10–40 % | **no** | out of scope |

## Sweep C, Qwen3.8-Flash-Next architecture (sources: lmsys blog 2026-08-26, HF card, oMLX 0.6.3, FP6-INT8 card)
| item | trick | source | est. | exact | note |
|---|---|---|---|---|---|
| HC (gated residual, 4 branches, low-rank 320) | **HC Mix as one GEMM with gating/SiLU/sigmoid/reduction fused, weights reordered offline: kernel 2.05x, +7.6 % end to end** | lmsys / FlashInfer #4266 | +5–8 % | yes | HC mixers are 1.19 GiB bf16, read every token; quantizing them is lossy |
| HC | HC Combine split along the hidden dim: 1.96x kernel, **+5.5 % end to end, bitwise identical** | lmsys | +4–6 % | yes | |
| fusion overall | unfused attention/GDN/norm/HC chain was ~10 of 30.6 ms per step on a GPU | FP6-INT8 card | up to 30 % of the step | yes | biggest exact lever after MTP |
| indexer | **skip the sparse indexer while context ≤ ~2048 tokens** (it keeps 512 blocks × 4 = 2048 tokens, so it selects everything); run dense attention | derived from lmsys + HF card | +3–6 % on short contexts | yes (derived; verify with a hash check) | |
| indexer | IndexShare: compute the indexer top-k once per MTP round and reuse it for every draft step | lmsys | +2–5 %, grows with context | yes for target output | |
| indexer | run the indexer on a second stream, overlapped with Q/K/V | lmsys | +1–3 % | yes | |
| MTP | depth 3: SGLang average accept length 3.3; oMLX "Lightning MTP" on M3 Ultra 97–98 % acceptance, 22 → 58 tok/s | lmsys, oMLX 0.6.3 | already BigMoE's main lever | yes | BigMoE should copy oMLX's adaptive-depth rule |
| PLE table | gather the 16 rows on a side stream overlapped with block 1; keep the table in its own mmap'd shard so Metal doesn't wire it (M5 Max wired 114 → 91 GiB) | lmsys, lilting, oMLX | frees ~20 GB for experts if it's wired today | yes | check BigMoE's wired memory |
| MTP draft path | the draft skips the PLE table | lmsys | small | yes | |
