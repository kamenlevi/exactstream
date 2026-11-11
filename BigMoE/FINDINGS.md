# BigMoE: Qwen3.8-Flash-Next (125B MoE) at 4-bit on a 64 GB Mac, lossless, ~40 tok/s?

MacBook Pro M5 Pro, 64 GB, 280 GB/s GPU read, APPLE SSD AP2048Z. 2026-10-17.

## Phase 3 (2026-10-18 morning): first end-to-end Qwen3.8-Flash-Next (125B, 4-bit) on this 64 GB Mac

Exact SSD expert streaming via MTPLX 2.12.2 + `bigmoe` (gate, LRU slot pools), greedy, 327-token coding
prompt, 64 tokens, page cache evicted before each run:

| residency (slots / 512) | plain decode | prefill + first token | misses / token | stalled layers / token | SSD ms / token | peak |
|---|---|---|---|---|---|---|
| 40 % (205) | 18.3 tok/s | 6.5 s | 50.2 | 27.3 | 20.6 | 33.6 GB |
| 50 % (256) | 21.1 tok/s | 6.7 s | 31.0 | 19.6 | 13.6 | 40.4 GB |
| **55 % (282)** | **22.1 tok/s** | 10.1 s | 22.7 (4.7 % of uses) | 15.3 | 10.3 | 43.8 GB |
| 60 % (307) | did not finish (memory) | | | | | |

- **Exactness:** tokens identical at 40 / 45 / 50 / 55 % (sha `c09a8163f9`): different slot layouts and miss
  patterns, same output. `check_layer.py`: streamed SwitchGLU vs full-tensor SwitchGLU is 6/6 bit-identical
  on layers 0, 24 and 47. MTPLX itself cannot run this model resident on 64 GB, so a token-level comparison
  against unstreamed MTPLX is not possible here.
- **The memory budget is the machine's, not the model's.** About 32 GB is in use by other processes, so
  above ~44 GB the GPU working set thrashes (60 % never finished; 55 % before the fixes ran at 0.2 tok/s).
- **Two fixes were needed:** MTPLX pre-reads up to 18 GiB of the n-gram table into page cache
  (`MTPLX_NGRAM_PREWARM=off`), and MLX's buffer cache kept the prefill scratch pools (up to 1.4 GB per
  layer). With `mx.set_cache_limit(1 GB)` and `clear_cache` after each prefill layer, decode went from
  0.2 to 22 tok/s.
- **MTP through MTPLX is slower here:** k=2 12.2 tok/s, k=3 15.0 tok/s at 55 %, with tokens that differ
  from plain decode (sha `eb5801b31c`; not checked against an unstreamed MTP run). MTPLX's verify path does
  not amortize the streaming the way SpecDecode's loop did on the 35B. Open: profile it, or drive MTP with
  our own verify loop.
- MTP profile (`scripts/mtp_diag.py`, 50 %): a verify pass costs ~64 ms in the trunk plus ~15 ms between
  passes (draft head, sampling), at only **~1.6 accepted tokens per pass** → ~54 ms/token, no better than
  plain. On the 35B, SpecDecode's MTP head reached 2.7 tokens per pass. Next: measure MTPLX's acceptance on
  this prompt with and without streaming at a tiny budget, to separate "low acceptance" from "streaming
  slows the verify pass".
- Residency sweep: 40–55 % measured above; 60–65 % is blocked by the machine's free memory, not by the
  method (it needs ~48 GB+ for the GPU, i.e. other apps closed).
- **Versus ~40 tok/s:** plain exact decode reaches 22 tok/s on this machine at its real memory budget,
  short of 40. The 35B results suggest that working MTP (×1.3–1.4) and prefetch (+8 %) would bring it to
  ~30–33, and more free RAM (60–65 % resident) would add a little more. 40 needs both and a
  quieter machine.

## Phase 2 (2026-10-17 evening): built and measured, on Qwen3.6-35B-A3B as testbed

The 125B download is paused at 8.5 / 105 GB: the home link tops out at ~9 MB/s, so it resumes
overnight (`scripts/night_download.sh`, starts 01:00, keeps ≥200 GB free). Until then, everything
is built and measured on **Qwen3.6-35B-A3B 4-bit**, which is on disk. It is the same family and
uses the same mlx-lm `SwitchGLU`, and it fits in RAM, so the streamed output can be checked
**bit-for-bit** against the fully resident model.

### Result: exact SSD expert streaming in MLX, measured

35B 4-bit, 60 % of experts resident (154 of 256 per layer), cold SSD (page cache evicted
before every run with `scripts/evict.py`), greedy, 200 tokens of a coding prompt:

| config | tok/s | tokens vs resident model | peak memory |
|---|---|---|---|
| resident model, stock mlx-lm | 102.2 | — | 20.0 GB |
| **streamed, plain decode** | **58.8** | **identical** (sha `4e0400f671`) | 13.4 GB |
| resident model + MTP k=2 (SpecDecode's head) | 137.0 | — | 22.7 GB |
| **streamed + MTP k=2** | **71.9** | **identical** to resident MTP (`20d21f271c`) | 15.6 GB |
| **streamed + MTP k=3** | **80.8** | identical | 15.6 GB |

Prompt-lookup-else-MTP drafting (SpecDecode's hybrid, lookup drafts ≤4) is *worse* on this
prompt: 2.06 tokens per pass, 51.6 tok/s streamed. The answer is mostly new code, so lookups
pre-empt better MTP drafts. MTP k=3 is the default.

Prefill and next-step logits hash identically to the resident model (`scripts/diag35.py`).
The streamed prefill uses an order-preserving scratch pool, because MLX's sorted gather
groups rows by index value. Correctness suite (`scripts/diag_gate.py`): 12,000 / 12,000
gathered experts byte-correct under concurrent GPU load. The 9-piece expert reads cost only
12 % more than a contiguous blob would (`data/piece_bench.txt`), so **no repack** (it would
cost 68 GB of disk on the target).

Per token at 60 % resident: 17.0 misses on 12.3 stalled layers (the trace simulator predicted
16.4: **validated**), 6.2 ms of SSD time on the critical path, about 0.5 ms per stalled layer.
That is now the dominant cost. Only earlier knowledge of the routing can hide it, i.e.
mid-token look-ahead prefetch, which was ruled out for this phase (see "decision" below).

### The gate: making the GPU wait for the CPU exactly, in MLX (new)

| version | real 35B cost per gated layer | what was wrong / learned |
|---|---|---|
| `mx.eval` per layer | ~230 µs | the known floor (MoEPrefetch) |
| MTLSharedEvent, CPU-signalled, C++ primitive | ~170–180 µs | a buffer with `encodeWait` is only *scheduled* after the signal, then ~70–110 µs to start (`scripts/trace35.py`); free in plain Swift, not inside MLX's stream |
| + correctness fixes | — | one shared event let out-of-order rows release early; consumers in a later command buffer are **not** held by an earlier buffer's wait; routing rows were torn (seq visible before ids) → per-layer events, GPU-side copy after the wait, line-atomic rows |
| **commit + fresh-line spin** (`spin_wait`) | ~69 µs | GPU writes reach the CPU only at command-buffer end inside MLX (not at kernel or encoder end, unlike Swift), so commit after the routing row; the CPU releases by writing into a fresh 32 KB region per (pass, layer), which the GPU polls line by line. A line never read before is never stale in the GPU's L2 |
| **+ residency published a pass ahead** | **~5 % total on hits** | after serving layer L, the CPU publishes L's residency table for the next pass; the routing-row kernel looks the ids up in parallel and the spin returns immediately if all are resident. Only real misses wait (C=255: 93.4 tok/s with misses skipped vs 102 resident) |

Every exactness failure along the way was caught by tests, not by luck. A spin that waits too
long sets a flag that is checked after the pass, so it can never pass silently.

### Projection for Qwen3.8-Flash-Next (to be measured once downloaded)

Same design, using the cost model fed with the 35B measurements: 60 % resident, misses
20.9 per token on 13.8 stalled layers, ~0.6 ms per stalled layer (2.77 MB experts), hit
overhead ~5 %. **Plain decode ~31–36 tok/s; MTP k=2/3 ~42–55 tok/s.** So ~40 is
reachable with MTP, and plain decode lands just under it. The 35B measurements put streamed
MTP k=3 at 0.79× the *resident plain* speed. Apply that to the target's modelled 53 tok/s
resident and the result is ~42.

### Decision for Kamen

The remaining 6 ms/token on the 35B (≈8 ms on the target) is SSD latency on the critical
path. **Mid-token look-ahead prefetch** (layer L+1's router on layer L's residual, 72–96 %
recall per MoEPrefetch) is the one known way to hide it. MoEPrefetch found it failing because
predictions reached the CPU ~150 µs late. With this gate, routing reaches the CPU at the
command-buffer commit right after the router, so the old reason may no longer hold. It is
not used here, because it was ruled out. Worth a test only if Kamen lifts that rule. Modelled gain on the
target: about +20–30 %.

## Phase 1 verdict (paper study, kept for reference)

- **Yes, ~40 tok/s is reachable on paper, bit-exact, with ~60 % of the experts in RAM.
  It needs two things together: an exact SSD expert cache, and MTP speculative decoding
  that spreads the per-layer cost over the tokens each pass verifies.** Without MTP, the
  exact streaming design tops out near **33 tok/s**.
- **Best realistic number: ~50–60 tok/s session average (model).** That assumes 60 %
  of experts resident (a 46–48 GB GPU budget), look-ahead prefetch, MTP depth 2, and the
  MTP acceptance measured on Qwen3.6 (91.8 % / 90.3 %, i.e. 2.66 tokens per pass). The
  pessimistic end, with acceptance 0.6 and 50 % resident, is ~40.
- **The hard floor is the ~150 µs GPU→CPU→GPU round trip per layer (measured).** Exactness
  means the GPU must not run layer L's experts until the CPU knows they are resident. It
  costs 48 × 0.155 = 7.4 ms per forward pass, about 40 % of the fully resident token time.
  In-kernel signalling, which would make it ~free, does **not** work on Metal today (measured, §4).
- **Nothing was downloaded.** All per-model numbers come from the checkpoint's safetensors
  headers (HTTP range reads), measured hardware numbers, and routing traces of
  Qwen3.6-35B-A3B used as a proxy (from ~/MoEPrefetch). An end-to-end run needs the 4-bit
  checkpoint (~70 GB of weights plus a ~29–32 GB n-gram table) and a runtime that supports
  `qwen4_exp`, which mlx-lm 0.32.2 does not.

## 1. The target model (confirmed)

`Qwen/Qwen3.8-Flash-Next`, `model_type: qwen4_exp_text`, the open-weight preview of the
Qwen4 architecture. Numbers come from `cfg/config.json` and the safetensors headers
(`scripts/shapes.py` → `cfg/shapes.json`):

| part | params | notes |
|---|---|---|
| routed experts | 120.8 B | 48 layers × **512 experts, top-10**, expert = gate_up [1280×2560] + down [2560×640] |
| n-gram ("PLE") tables | 51.2 B | layer 1 only: 128 shards × 2.5M rows × 160 dims; 16 rows (bigram/trigram hash heads) read per token |
| linear attention (Gated DeltaNet) | 2.09 B | 36 of 48 layers |
| full attention + indexer (sparse attn) | 0.62 B | 12 layers, 24 q heads / 2 kv heads, head_dim 256 |
| hyper-connections (4 residual streams) | 0.64 B | hc_count 4, low-rank mixers |
| embed / lm_head / shared expert / router | 0.64 / 0.64 / 0.24 / 0.06 B | 1 shared expert (640) per layer |
| MTP head | 2.61 B | 1 layer with its own 512 experts |
| vision tower | 0.45 B | |
| **total stored** | **180 B** (360 GB BF16) | "125B total / ~6B active" excludes the n-gram tables |

At MLX 4-bit (group 64 = 4.5 bits/weight): **one expert = 2.765 MB, all routed experts =
67.9 GB.** Per decode token the model reads **2.46 GB of dense weights + 1.33 GB of routed
experts = 3.78 GB.** Unlike Qwen3.6-35B-A3B, two thirds of the bytes per token are
*dense*, so streaming experts can never help more than the routed third.

Memory: dense + embed + MTP + KV (32k context, ~25 KB/token) + recurrent state + scratch ≈ 6.2 GB.
So a 46 GB GPU budget leaves 39.8 GB for experts (**59 %**), 48 GB → 62 %, and 52 GB → 67 %,
which means raising `iogpu.wired_limit_mb` and closing everything else. The n-gram table
lives on the SSD: 16 rows × 320 B per token, and the rows depend only on the token ids, so
they can be fetched while layer 0 runs. At the measured 55k 4 KB IO/s (QD16, p50 0.21 ms)
the table costs nothing. MTPLX and Atomic Chat already do this, so it is not new.

## 2. Hardware measurements

`scripts/ssd_bench.py` → `data/ssd.txt` (cold reads, F_NOCACHE, 95 GB pool of unused model files):

| read | QD1 | QD4 | QD16 | QD32 |
|---|---|---|---|---|
| 2.77 MB expert | 5.8 GB/s (p50 0.45 ms) | 12.3 | **14.0** | 13.9 |
| 4 KB page | 10k IO/s (0.09 ms) | — | **55k IO/s** (0.21 ms) | 46k |

These match ~/MoEPrefetch, which found 13.4 GB/s for 1.77 MB reads and a 1–8 % decode
slowdown from concurrent SSD streaming at 1–7 GB/s. GPU read bandwidth is 280 GB/s
(~/SpecDecode). If every expert came from the SSD, a token would need 1.33 GB, i.e. ≤10 tok/s,
so a large resident set is mandatory.

## 3. Lossless cache simulation (proxy traces)

`scripts/sim.py` → `data/sim_f60.json`, `data/sim_f50_f70.json`. These are the routing
traces of Qwen3.6-35B-A3B (40 layers, 256 experts, top-8; chat, code, Bulgarian, long
context, real code; ~5k tokens each) at the same *fraction* resident. Each layer gets its
own LRU cache. Every needed expert must be resident before its layer runs; there is no
substitution, so output is bit-exact. Look-ahead prefetch uses layer L−1's pre-MLP
residual through layer L's router (top-B), as in MoEPrefetch. S = tokens verified per pass:
each pass reads the union of their experts.

| resident | policy | experts/layer per pass | stall loads / pass (proxy, of 320·S) | stalled layers (of 40) |
|---|---|---|---|---|
| 60 % | LRU | 8.0 | 13.9 | 9.5 |
| 60 % | LRU + look-ahead 12 | 8.0 | 5.4 | 4.0 |
| 60 % | LRU, S=3 | 17.0 | 41.8 | 19.2 |
| 60 % | LRU + look-ahead 12, S=3 | 17.0 | 15.7 | 8.7 |
| 50 % | LRU + look-ahead 12 | 8.0 | 8.1 | 5.5 |
| 70 % | LRU + look-ahead 12 | 8.0 | 3.3 | 2.6 |

Per expert use, LRU at 60 % misses 4.4 %; look-ahead cuts the misses 2.6× for ~1 prefetch
per layer, which fits behind one layer of compute. Three verify tokens touch 2.1× the
experts of one token, not 3×.

## 4. Can the GPU wait for a missing expert without a CPU sync? (prototype, measured)

The idea that looked new was **GPU-side demand paging on unified memory**: a tiny kernel
before each layer's expert matmul checks a residency flag. On a hit it falls through. On a
miss it posts a request in shared memory and spins until a polling CPU thread has
`pread`'d the expert into its slot. There are no command-buffer breaks, and the cost is paid
only on misses.

- `scripts/gpuwait.py` (MLX custom kernel, 48 fake 20 MB layers): **the hit path is
  cheap, 3–5 µs per layer,** against **157 µs per layer for an `mx.eval` sync** (`data/gpuwait.txt`).
- **The miss path does not work.** `scripts/coherence.py` (`data/coherence*.txt`):
  - **MLX gotcha:** MLX binds small input arrays (< 4 KB) as `constant` data copied at encode
    time, so a kernel spins on a snapshot forever. Shared buffers must be > 4 KB.
  - With real device buffers and relaxed atomics, RMW atomics or volatile loads, the
    writes on each side are seen only after **0.5–1000 ms**, scattered at random, i.e. when
    the cache line happens to be evicted. Usually it is not seen until the kernel ends.
  - Fences: `thread_scope_system` does not exist in this MSL, and device-scope fences change
    nothing. Metal guarantees CPU/GPU visibility only at command-buffer boundaries.
- `swift/eventwait.swift` (`data/eventwait.txt`): an **`MTLSharedEvent` handshake** per
  layer. The GPU signals "router done", a CPU listener checks and loads, and the next
  layer's command buffer `encodeWait`s on the event. This costs **148–166 µs per layer**,
  the same as an `mx.eval` sync. The command-buffer split itself is free (≤3 µs). A stalled
  layer with one cold 2.77 MB read adds **~370 µs**.

So on Metal, every way of making the GPU wait on the CPU mid-token costs ~150 µs.
**Exact streaming pays ~7.4 ms per forward pass** no matter how good the cache is. This
is also the answer to MoEPrefetch's open item (a), a "mid-token GPU→CPU signal": it costs
~150 µs per signal.

## 5. Cost model (`scripts/costmodel.py` → `data/costmodel.txt`)

Per pass: t = launch overhead (4.0 ms, scaled from the 35B-A3B's 2.4 ms for 48 layers and
hyper-connections) + bytes / 258 GB/s + 48 × 0.155 ms handshake + stalled layers × 0.37 ms
+ extra misses at 13.4 GB/s + prefetch I/O not hidden behind compute, × (1 + 0.0076 per GB/s
of SSD traffic). Proxy misses are scaled to 48 layers × top-10 at the same miss rate per
use. MTP drafting costs 0.6 ms per drafted token. Tokens per pass with acceptance a per
position = (1 − a^(D+1)) / (1 − a).

Fully resident (needs ~74 GB, i.e. a 96 GB Mac): **53 tok/s** without MTP. This
cross-checks: the same model predicts ~100 tok/s on an M3 Ultra, where oMLX reports 91.

| config (all bit-exact) | tok/s, a = 0.6 / 0.75 / 0.9 |
|---|---|
| 60 % resident, per-layer sync, LRU, no MTP | **30** |
| 60 %, handshake + look-ahead 12, no MTP | **34** |
| 60 %, LRU, MTP depth 2 | 41 / 49 / 57 |
| **60 %, look-ahead 12 + MTP depth 2** | **47 / 56 / 65** |
| 60 %, look-ahead 12 + MTP depth 3 | 42 / 53 / 67 |
| 50 %, look-ahead 12 + MTP depth 1 | 38 / 41 / 45 |
| 70 %, look-ahead 12 + MTP depth 2 | 51 / 60 / 70 |
| *(hypothetical) in-kernel GPU wait, 60 %, look-ahead 12, MTP depth 2* | *57 / 67 / 79* |
| *(reference) fully resident, MTP depth 2* | *76 / 90 / 105* |

With the measured MTP acceptance (a ≈ 0.9; SpecDecode measured 2.66 tokens per pass for
k=2 on Qwen3.6, and MTPLX reports large MTP gains on this exact model), the
60 %-resident exact design models at **~60 tok/s**. Discounting ~15–25 % for what the model
leaves out (proxy routing, launch-overhead guesses, prefill and long-context attention)
gives **~50 tok/s realistic, ~40 conservative.**

## 6. Prior work and what is new

| work | what it does on this kind of problem | lossless? | numbers |
|---|---|---|---|
| [MTPLX](https://github.com/youssofal/MTPLX) | MLX + native MTP for this model; n-gram table on SSD; weights fully resident | yes | flat 4-bit needs a **96 GB** Mac (~74 GB resident); 63–126 tok/s on M5 Max 128 GB |
| [Atomic Chat guide](https://atomic.chat/blog/guides/how-to-run-qwen-3-8-flash-next-locally) | llama.cpp, n-gram table mmap'd from SSD | **no** for 64 GB: needs 3.84 bpw (KL 0.23, top-1 82.7 %); Q4_K_M needs ~80 GB | 36 tok/s on a 64 GB M5 Max |
| iQ-MLX 3.3 bpw pack (ddalcu) | fits 64 GB at 3.3 bpw, 52 GB resident | no | 52.6 tok/s on M4 Max |
| [oMLX](https://github.com/jundot/omlx/releases) | has "expert offload" for this model | ? | only prefill numbers (M3 Ultra); MTP head resident |
| ~/MoEPrefetch (this Mac, sibling study) | sync-free slot pool, cache-aware **substitution**, CPU one token behind | **no** (ppl −0.1…+0.2 % at 44 % resident) | 83 tok/s at 9.3 GB on 35B-A3B |
| [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) | per-layer slot cache, demand loads block the op, no prefetch; unmerged | yes | 1.8–2.2 tok/s GLM-5.2 on GB10 |
| [Flash-MoE](https://github.com/danveloper/flash-moe) | Metal, pread, OS page cache; spin-polling and prefetch hurt | yes | 4.4 tok/s, 397B on M3 Max |
| [ExpertFlow](https://github.com/jhammant/expertflow) | hot/cold experts, look-ahead `MADV_WILLNEED`, cold experts demoted to Q2 | no | projections only |
| LLM in a Flash, PowerInfer, MoE-Infinity, Eliseev & Mazur | flash-aware loading, hot/cold neurons, activation-aware prefetch, next-layer gate look-ahead | mostly yes | GPU/server or older Macs; the look-ahead gate idea is from Eliseev & Mazur |

What is new here, as far as a quick search shows:
1. **The first lossless budget for this model on a 64 GB Mac.** Everyone who runs it in
   64 GB today drops to 3.3–3.84 bpw. At 4-bit, the routed experts are only 35 % of the
   bytes per token, and ~60 % of them fit.
2. **Using MTP verification to make exact streaming affordable on a Mac.** The general idea
   is published: speculative decoding amortizes offloaded expert loads over a verify chunk
   ([SpecMoEOff](https://arxiv.org/pdf/2508.21706), [SP-MoE](https://arxiv.org/pdf/2510.10302),
   [DraftExpert](https://arxiv.org/pdf/2607.24434), [AcceptMoE](https://arxiv.org/pdf/2608.02989),
   which is residency-conditioned and so not exact). All of those target discrete GPUs with
   host-RAM offload. What is specific here: on Apple unified memory with SSD streaming, the
   amortized cost is mostly the 7.4 ms per-pass exactness handshake and the stall latency,
   not transfer bandwidth. Expert unions for S-token passes, measured on traces:
   8 → 12.9 → 17.0 → 20.8 per layer (proxy, top-8).
   Checked 2026-10-17: no Mac runtime does SSD *expert* streaming for this model.
   [ds4](https://dwarfstar.sh/blog/qwen38-flash-next-64gb-mac-ds4/) says it is not
   implemented and runs Q2 (41.7 GiB resident) on 64 GB. Its Q4 needs 69.7 GiB resident.
   The [SSD-Stream pack](https://huggingface.co/garnermccloud/Qwen3.8-Flash-Next-MLX-SSD-Stream)
   streams only the n-gram table.
3. **Measured on Metal (negative, but it was the open question):** in-kernel GPU↔CPU
   signalling is not usable (visibility only at command-buffer end or random eviction,
   0.5–1000 ms), and an `MTLSharedEvent` round trip costs ~150 µs, the same as an
   `mx.eval`. Plus the MLX `set_bytes` gotcha for small arrays. I found no published
   numbers for either.
4. Not new: the n-gram table on SSD, look-ahead gating, LRU per layer, F_NOCACHE pread
   into Metal-shared slots.

## 7. What it would take (not done; needs Kamen's go)

1. **Download** a 4-bit MLX MTP pack, e.g. `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`
   (MTPLX "Bare Speed" is 106 GB with the table). That is >20 GB, so it needs approval. There is
   396 GB free on disk.
2. A `qwen4_exp` model file for mlx-lm (or the module bundled with those repos). Repack
   experts into contiguous 2.77 MB blobs. Build the slot-pool cache, reusing MoEPrefetch's
   `slotmoe.py`, but in exact mode: split command buffers at every router and use a
   shared-event handshake with a CPU loader that does LRU plus look-ahead.
3. Measure it real (~1 day), with and without MTP. The first decision point is **without
   MTP, ≥30 tok/s at 60 % resident**; if lower, the model's launch/compute numbers are off.
4. A cheaper lossless lever worth a test: allocate per-layer cache budgets from each layer's
   LRU curve instead of a uniform 60 %, since layers 0–2 are the least local. This was not
   simulated yet.

If bit-exactness can be relaxed to "perplexity within noise", MoEPrefetch's substitution
removes the handshake and most stalls. That design would model at roughly the
hypothetical row (~65–80 tok/s with MTP), but it is not lossless, so it is out of scope here.

## Files

- `cfg/config.json`, `cfg/index.json`, `cfg/shapes.json`: target model config, weight map, tensor shapes
- `scripts/shapes.py`: parameter counts from remote safetensors headers
- `scripts/ssd_bench.py` → `data/ssd.txt`
- `scripts/sim.py` → `data/sim_f60.json`, `data/sim_f50_f70.{json,txt}` (reads ~/MoEPrefetch/traces)
- `scripts/costmodel.py` → `data/costmodel.txt`
- `scripts/gpuwait.py` → `data/gpuwait.txt`; `scripts/coherence.py` → `data/coherence_v012.txt`, `data/coherence.txt`
- `swift/eventwait.swift` → `data/eventwait.txt` (build: `xcrun swiftc -O swift/eventwait.swift -o swift/eventwait`)

Caveats: the routing is a proxy; this model's 512-expert top-10 routing may be more or less
local. The compute calibration is extrapolated from the 35B-A3B. The handshake cost was
measured with synthetic layers. The MTP acceptance is from Qwen3.6, not this model. GPU
timings were taken under quiet locks, with Chrome active.

## Next experiments (kept current)

1. **[done, v1; v2 running]** Mid-token look-ahead prefetch into spare slots (exact; Kamen OK'd one
   test). 35B, cold SSD, same memory as the 60 % runs (12 of 154 slots per layer become spares,
   top-12 look-ahead). Tokens identical to the resident model in all three runs:
   plain **58.8 → 63.7 tok/s (+8 %)**, MTP k=2 **71.9 → 82.3 (+14 %)**, MTP k=3 **80.8 → 85.8 (+6 %)**.
   Full-read stalled layers fall 12.3 → 4.1 per token and SSD time 6.2 → 2.0 ms. But a prefetched
   expert still takes the slow path (the GPU waits for the CPU to promote it), so the tok/s gain is
   smaller than the I/O gain. v2: when a prefetch read completes, the CPU adds the expert
   to this pass's published residency table. The update is additive, so it is race-free: read
   earlier means the slow path, read later means the fast path via the spare slot.
   **v2 measured: no gain** (62.8 / 80.4 / 86.4 tok/s). The one-layer head start (~250 µs) is
   shorter than a cold read (~330 µs), so the reads finish after the GPU has looked. v3 (next):
   also predict two layers ahead for a ~2× head start (`scripts/sim_pf2.py` first).
   **v3 sim** (`data/sim_pf2.txt`): 20 spares, top-10 at depth 1 and depth 2 → stall time −31 %
   (plain) / −29 % (S=3), against −12 % / −14 % for depth 1 only. Code is written (stream.py,
   ext; not built yet). GPU test: `scripts/series_pf2.sh 20 10 10`.
   First attempt aborted cleanly in prefill: the prefill row grouping still budgeted all 154 slots
   while 20 were spares (fixed: budget = C − spares).
   **v3 measured (01:20): no gain.** Plain 64.7, MTP k=2 80.6, MTP k=3 82.4 tok/s (v1: 63.7 / 82.3 /
   85.8), tokens identical. The longer head start is eaten by the 8 extra slots taken from the main
   cache. **Default is v1: 12 spares, depth 1, top-12** (+6–14 %, exact). What remains on the 35B
   (plain 63.7 vs 102 resident, ~5.9 ms/token) is now mostly slow-path round trips (~16 slow layers
   per token × ~0.24 ms), not SSD (~2 ms). That is the next thing to attack, after the 125B run.
1b. **Slow-path decomposition (01:25, 35B, prefetch v1, plain)** (`scripts/series_rt.sh`): exact 62.4 tok/s
   vs 91.2 when the GPU never waits. That is 5.0 ms/token over 14.4 slow calls ≈ **350 µs of GPU wait per
   slow layer, of which 305 µs is the CPU's own time** (demand reads + waits for prefetches still in
   flight). The pure round trip is only ~45 µs (~0.65 ms/token). **The slow path is SSD-latency-bound.**
   Lever: make one expert read faster (`scripts/chunk_bench.py`: split pieces into parallel chunks).
   **Measured: no.** A cold 1.77 MB expert is 0.39 ms p50 as 9 pieces; splitting into 512 / 256 / 128 /
   64 KB chunks gives 0.39 / 0.46 / 0.48 / 0.53 ms. It is bound by SSD latency, not by how the reads
   are issued. **Conclusion for "cut the slow-path round trip": the round trip itself is ~45 µs and
   not worth more engineering.** What remains is SSD latency on misses. Only fewer misses (more
   residency, a better budget split) or earlier *and* accurate prediction can remove it, and depth-2
   prediction costs more slots than it saves.
2. Prep for the 125B run (done, 01:10): `RUNBOOK.md` written. Dry runs on 35B pass:
   `check_layer.py` gives 6/6 bit-identical trials on layers 0, 20 and 39 (streamed SwitchGLU vs
   the full-tensor SwitchGLU). `bench.py` through MTPLX's loader decodes at 53.8 tok/s plain
   (prefetch off). The download started at 01:00 with 338 GB free.
   Sim (`data/sim_pf.txt`): 12 spares + top-12 cuts stall time 12 % (200 µs head start) to
   36 % (350 µs), plain and S=3 alike. Note for the target: qwen4_exp's MoE input is a
   hyper-connection mix, so the exact norm swap does not apply; recall must be measured
   (`scripts/recall_target.py`).
2. Qwen3.8-Flash-Next end-to-end once the overnight download finishes (01:00 →~05:30):
   `scripts/bench.py 0.6 ar,mtp2,mtp3`, with MTPLX trunk compile off. First check: tokens equal
   to MTPLX's own output on a short prompt.
3. Split-phase MoE on stalled layers (compute resident experts while misses load; exact if the
   per-expert rows are reassembled before the same reduction): ≤ ~12 % of stall time.
4. Residency budget sweep on the target: 55 / 60 / 65 % (needs the iogpu wired limit for 65 %).
8. **Full stack on the 125B, one layer at a time** (orchestrator, 2026-10-18): MLX 0.32.4 (only if
   tokens are identical), MoEPrefetch's fused routing kernel, mid-token prefetch, 60–65 % residency with
   apps closed, speculation, optional wait-top-2 → table of config / tok/s / exact? / peak GB.
   Caveats: `mx.compile` on the MoE blocks conflicts with the per-layer Python gate (a traced block runs
   it once), so it applies only around the expert call. SpecDecode's drafter and rollback are written for
   qwen3_5; on qwen4_exp it needs porting to MTPLX's caches.
6. MoEPrefetch's wait-top-2 as an **optional, clearly labelled non-exact fast mode** on the 125B (default
   stays exact). At 55 % it would cut the ~15 stalled layers/token to perhaps ~5 → roughly 28–30 tok/s
   plain (estimate). But Qwen3.8-Flash-Next routes top-10 of 512, so the quality cost must be measured on
   this model (GSM8K / perplexity), not carried over from the 35B. It needs MoEPrefetch's fork of the ext.
7. MTP on the 125B: acceptance is only ~1.6 tokens/pass through MTPLX; separate low acceptance from
   streaming overhead (tiny-budget run vs larger budget, same prompt).
5. After the first end-to-end 125B number: does MTPLX's qwen4_exp actually break on MLX 0.32.4
   (MTPLX pins 0.32.2)? Separate venv, extension rebuilt, outputs compared on a few prompts;
   SpecDecode saw +17 % plain decode on 35B Q8 from 0.32.4.
