# External community results: large-MoE decode with less fast memory than the model (2024 to 4 Nov 2026)

Written Wed 4 Nov 2026 (14:30 EET) for the "have we done something never achieved before?" check. Scope: open-source projects
and community reports (GitHub, HF, Reddit, HN), not papers. Budget used: 20 web searches and 35 page fetches. Reddit pages were read
through the Arctic Shift archive API (`arctic-shift.photon-reddit.com/api/posts/ids?ids=<id>`), because reddit.com blocks the fetcher.

**Memory accounting.** "Machine" = installed RAM (+ VRAM). "Process" = the engine's own budget or RSS. Every Mac streaming engine below
leaves the OS file cache uncontrolled, so a run with a 22 GB process target on a 48 GB Mac is *not* a 24 GB run. The buckets use the
machine figure. Runs with a small process budget on a big machine are flagged, because they matter for any claim phrased per budget.

**Exact / lossy.** Exact = the model as downloaded, all routed experts, verified speculative decoding allowed. Lossy-route = reduced
top-k, routing biased toward cached experts, or expert skipping. Lossy-quant = an aggressive quant (<= ~2.5 bpw). Several rows are both.

## Model facts needed to read the tables

- **Qwen3.8-Flash-Next** (the "~125B Qwen Flash MoE"): 125B total, ~6B active, plus ~51B of n-gram embedding tables held off the
  accelerator (r/LocalLLaMA posts mirrored on prismix.dev; unsure). It has 48 layers x 512 routed experts with 10 active per token
  ([ISTA-DASLab card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), which lists 177B params). Kapitsa calls it
  "180B-A7B". It is >= 80B under every count. Its *active* bytes per token are far below Qwen3.5-397B (17B active) or DeepSeek (37B).
- **ISTA-DASLab GSQ-RCO GGUF** (per-tensor bit allocation): the weights shard plus a 28.8 GB n-gram shard. The table is BF16 task average 93.12.

  | Quant | bpw | Weights | Total on disk | Task avg (vs BF16 93.12) |
  |---|---|---|---|---|
  | Q2_0 | 2.40 | 37.6 GB | 66.4 GB | 89.07 |
  | IQ2_XS | 2.50 | 39.2 GB | 68.0 GB | 89.16 |
  | IQ3_XXS | 3.00 | 47.0 GB | 75.8 GB | 92.57 |
  | IQ3_S | 3.50 | 54.8 GB | 83.6 GB | 93.26 |

- **Qwen3.5-397B-A17B** has 512 experts, "10 Routed + 1 Shared"
  ([model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)). flash-moe and its forks run **K=4**, so all of their rows are
  lossy-route, even at 4-bit.

## (1) >= 80B MoE with <= 24 GB fast memory (machine), rest streamed from SSD

| Project / poster | Model (total / active) | Precision + quality claim | Fast memory | Storage | Hardware | Decode tok/s | Date | Link | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| Splash 1.4.0 + ~800-line fork, u/Inside-Pen-8395 | Qwen3.8-Flash-Next (125B / ~6B) | ISTA GSQ-RCO **Q2_0** (2.40 bpw, lossy-quant). The fork's cache-aware routing (bias 2, top 4 protected) changes outputs (lossy-route). The overlap stage was token-identical on 11/11 fixed gate-set cases. LiveCodeBench, 20 problems: stock 17, fork 16 (one run each). DFlash2 draft model. | 24 GB unified. Expert store 6,800 of 24,576 slots (~8.8 GB). | Internal SSD (66.4 GB pack) | MacBook Air M5, 24 GB | **48.8** avg (code 48.9 / tool 57.3 / explain 40.3). Routing alone: 43.1. Real agent use (pi): ~45. Cold prefill: 522 tok/s at 32.6K, 485 at 63.2K. Context 256K. Draft acceptance ~0.70 code, ~0.80 tool. | 2026-10-25 | [r/LocalLLaMA 1xezdgs](https://www.reddit.com/r/LocalLLaMA/comments/1xezdgs/flash_next_q2_on_a_24gb_m5_macbook_air_49_ts/) | **sure** (original text via archive). Self-reported and not replicated. A second-hand write-up (remio.ai, 10-27) agrees. |
| Same post, **stock** Splash 1.4.0 | same | Q2_0 (lossy-quant). Routing is exact, with verified drafting. Splash notes that the default `--verify-rows auto` can change greedy output with timing. | same | same | same | **28.8** avg (31.6 / 29.5 / 25.4) | 2026-10-25 | same | **sure** (same caveats) |
| SlotStream, @davidcavazos | Qwen3.8-Flash-Next (125B) | MLX **4-bit original**, exact ("prefetch ... changes speed and never the output"). 0.2.25 turned on the draft head and decode lookahead. | 24 GB. Auto target 17.4 GB, peak 16.7 GB. | Internal SSD (~105 GB model) | MacBook Pro M4 Pro, 24 GB | **5.41** (5.58 / 4.92 in the same round). On 0.2.24: 3.57 (3.85-3.97 later). Prefill 86-93 tok/s (8K). | 2026-09-25/26 | [HARDWARE.md](https://github.com/carloslfu/slotstream/blob/main/docs/HARDWARE.md), [issue #41](https://github.com/carloslfu/slotstream/issues/41) | **sure** (project's hardware table; community run; issue page itself not read) |
| SlotStream, u/EggCess (comment) | Qwen3.8-Flash-Next | 4-bit, exact | 24 GB | SSD | Mac mini M4, 24 GB | "~2-3 tps" | ~Sep 2026 | comment in [r/LocalLLM 1w53inj](https://www.reddit.com/r/LocalLLM/comments/1w53inj/) | unsure (one-line comment, version unknown) |
| SlotStream, @flol | Qwen3.8-Flash-Next | 4-bit, exact | 16 GB machine. Process target 10.7 GB, 8.1 GB RSS. | SSD | Mac mini M2, 16 GB | **1.48** (0.2.3). 1.41 on 0.2.2 (10.2 GB target, 6.1 GB peak). Prefill 11 tok/s (8K). | 2026-09-02/03 | [issue #5](https://github.com/carloslfu/slotstream/issues/5) via HARDWARE.md | **sure** (project table) |
| SlotStream planning table | Qwen3.8-Flash-Next | 4-bit | 16 to <24 GB / 24 to <48 GB | SSD | any Mac | "~1-6" / "~5-16" (estimates, "rounded outward") | 2026-10 | [README](https://github.com/carloslfu/slotstream) | unsure (estimates, not measurements) |

**Best exact (original 4-bit): SlotStream, 5.41 tok/s on a 24 GB M4 Pro.** **Best lossy: the Splash fork, 48.8 tok/s (Q2_0 + cache-aware
routing).** The best run with exact routing but an aggressive quant is stock Splash 1.4.0 at 28.8 tok/s (Q2_0, 24 GB M5 Air). So ">= 10 tok/s
for a >= 80B MoE at 24 GB" was publicly claimed on 2026-10-25 for Qwen3.8-Flash-Next at 2.4 bpw.

**The r/macmini run was NOT FOUND.** The reported setup: Mac mini M4 Pro 24 GB, ISTA GSQ-RCO IQ3_XXS, ~76 GB on disk, ~11 tok/s short /
~8.5 long, 180 tok/s prefill, "9.5 GB peak runtime". Searches tried: 7 web searches, plus Arctic Shift title search of r/macmini. The query
"Qwen" since 2026-07-01 returned 2 unrelated posts. The queries "Flash" and selftext "IQ3_XXS" returned HTTP 422. Only the disk size checks
out: IQ3_XXS totals 75.8 GB (47.0 + 28.8 n-gram shard). The model name Qwen3.8-Flash-Next is real. Splash (since 1.4.0, 2026-10-20) and
Kapitsa stream GSQ-RCO GGUF experts on Macs; SlotStream is 4-bit MLX only. So the run is plausible but unconfirmed. Do not cite its numbers.

**No measured <= 24 GB report was found** for DeepSeek V3/R1, Kimi K2, Qwen3-235B, Qwen3-Next-80B, GPT-OSS-120B, GLM-4.5/4.5-Air,
MiniMax M2 or Llama 4. Sites like willitrunai.com publish formula estimates (e.g. "2.0 tok/s"); these are not measurements.
Unsloth's 1.58-bit R1 post (2025-01-27, [1ibbloy](https://www.reddit.com/r/LocalLLaMA/comments/1ibbloy/)) says a "24GB GPU like RTX
4090 ... should be able to get at least 1 to 3 tokens / s" (IQ1_S, 131 GB). That is a projection and states no RAM size: unsure.

## (2) >= 80B MoE with <= 12 GB fast memory

| Project / poster | Model (total / active) | Precision + quality claim | Fast memory | Storage | Hardware | Decode tok/s | Date | Link | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| ANEMLL's iOS port of flash-moe | Qwen3.5-397B-A17B (397B / 17B) | Quant not read. K=4 if the flash-moe default is kept (lossy-route). | 12 GB | Phone flash | iPhone 17 Pro | ~0.6 (GIGAZINE: 0.7). TTFT ~50 s. | ~2026-03-22/24 | [TweakTown](https://www.tweaktown.com/news/110610/the-iphone-17-pro-can-run-a-400b-parameter-large-language-model-on-device-by-streaming-weights-from-the-ssd/index.html), [GIGAZINE](https://gigazine.net/gsc_news/en/20260324-iphone-397b-ai/), [HN 47490070](https://news.ycombinator.com/item?id=47490070) | **unsure** (press only; the X post was not read; TweakTown returned 403) |
| SlotStream, @flol (also in table 1) | Qwen3.8-Flash-Next (125B) | 4-bit, exact | **Process** 10.7 GB target / 8.1 GB RSS, on a 16 GB machine | SSD | Mac mini M2, 16 GB | 1.48 | 2026-09-03 | [issue #5](https://github.com/carloslfu/slotstream/issues/5) | sure (only <= 12 GB by process budget) |

**Best exact:** none on a <= 12 GB machine. By process budget, SlotStream's 1.48 tok/s (16 GB machine) is the best.
**Best lossy:** the iPhone 17 Pro at ~0.6 tok/s (unsure). The Splash fork's ~8.8 GB expert store does not count: it ran on a 24 GB
machine and the total process size was not given.

## (2b) Context: SSD streaming with 25-128 GB (model still larger than RAM)

These runs sit outside both buckets. Several capped the *process* at <= 24 GB on a bigger machine, so they matter for any claim stated per budget.

| Project / poster | Model (total / active) | Precision + quality claim | Fast memory | Storage | Hardware | Decode tok/s | Date | Link | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| Kapitsa (KFlash spec. decoding) | Qwen3.8-Flash-Next ("180B-A7B") | Q2_0 experts streamed (lossy-quant). Claims exact: "changes no output: the big model checks every token". Same tokens as the release on 6 prompts x 22 runs. | 48 GB machine, `--expert-memory 16G` | SSD | MacBook Pro M5 Pro, 48 GB | **61.3** (53.7-64.7). Prefill 1,218. | 2026-10-30 | [nace-ai/kapitsa](https://github.com/nace-ai/kapitsa) | sure (README) |
| Splash 1.5.0, measured by Kapitsa | same | Q2_0 | same, 16G store | SSD | same | 56.5 (51.9-60.2). Prefill 1,178. | 2026-10-30 | same | sure (a competitor's measurement) |
| llama.cpp (c7474afc, PR #751), measured by Kapitsa | same | Q2_0 | 48 GB | SSD | same | 9. Prefill 37. | 2026-10-27 | same | sure (as reported) |
| SlotStream, author | Qwen3.8-Flash-Next 4-bit | Exact | 48 GB machine. Process ceiling 33 GB / **22 GB** / **20 GB**. | SSD | MacBook Pro M5 Pro, 48 GB | 20.91 / 23.69 / 21.59 (short / long / code) at 33 GB. **15.86** at 22 GB (0.2.19). 13.47 at 20 GB (0.2.16). | 10-05 / 09-16 / 09-13 | [README](https://github.com/carloslfu/slotstream), [HARDWARE.md](https://github.com/carloslfu/slotstream/blob/main/docs/HARDWARE.md) | sure ("filesystem cache is uncontrolled") |
| SlotStream, community | same | Exact | 32 GB (22 GB target) / 36 GB / 64 GB / 64 GB | SSD | M5 Air / M4 Max / M3 Max / M4 Max | 6.22 / 8.41 / 12.38 / 15.93. The 64 GB M4 Max on an external SSD: 2.98. | 2026-09-07 to 09-20 | HARDWARE.md, issues #12 / #26 / #20 / #22 / #23 | sure (project table) |
| SlotStream, @waterliu1981 | same | Exact, with spec. decoding | 128 GB machine. Target 34.6 / 48 / 73 GB. | SSD | M5 Max, 128 GB | ~21-22 / ~26.9 / ~31.5 | 2026-09-03 | issue #6 comment via HARDWARE.md | sure (warm summaries; never re-measured) |
| u/mmerken, llama.cpp Metal | Qwen3.8-Flash-Next | "AD-4.27bpw-Q4_K_M-M64", 33 shards | 64 GB (wired limit 60.5 GB) | SSD ("60 GB hit"; "15-20GB/s" GPU bandwidth blamed on SSD) | MacBook Pro M3 Max, 64 GB | ">20" at start, ~12 at 128K. Prefill ~300. | 2026-09-02 | [r/LocalLLM 1w53inj](https://www.reddit.com/r/LocalLLM/comments/1w53inj/) | unsure (model size vs RAM not stated) |
| flash-moe (danveloper), C/Metal | Qwen3.5-397B-A17B | 4-bit experts (209 GB), **K=4 of 10** (lossy-route). The 2-bit variant (120 GB) "breaks JSON/tool calling". | 48 GB. Process ~6 GB plus ~35 GB page cache (71 % hit). | SSD at 17.5 GB/s | MacBook Pro M3 Max, 48 GB | 4.36 (4-bit). 5.74 (2-bit). 7.05 2-bit warm peak. | ~2026-03 (README undated) | [danveloper/flash-moe](https://github.com/danveloper/flash-moe) | sure (numbers); date unsure |
| Anemll/flash-moe fork (WIP) | Qwen3.5-397B-A17B | K=4 (lossy-route). Q3 GGUF (163 GB) PPL 3.81. 4-bit MLX PPL 3.64. 2-bit PPL 5.71. | 128 GB (model 120-209 GB) | SSD | M5 Max, 128 GB | 12.9 (Q3) / 9.5 (4-bit) / 14.5 (2-bit). Full GGUF stack: 5.1. | undated (2026) | [Anemll/flash-moe](https://github.com/Anemll/flash-moe) | sure (README) |
| mlx-flash (szibis) | "Qwen3.5-397B" | Quant and K not stated | 64 GB | SSD | Mac, 64 GB | 4.4 | undated | [szibis/mlx-flash](https://github.com/szibis/mlx-flash) | unsure (may restate flash-moe) |
| u/VoidAlchemy, llama.cpp mmap | DeepSeek-R1 (671B / 37B) | UD-Q2_K_XL 2.51-bit (212 GB per Unsloth), lossy-quant | 96 GB, GPU off. Second run: 96 GB + 24 GB VRAM. | Gen5 x4 NVMe | AMD gaming rig | ~2.13 at 2k ctx. With the 3090 Ti: 1-2 at 2k-16k. | 2025-01-30 | [r/LocalLLaMA 1idseqb](https://www.reddit.com/r/LocalLLaMA/comments/1idseqb/) | sure (archive) |

anemll-flash-mlx ([repo](https://github.com/Anemll/anemll-flash-mlx)) has only Qwen3.5-35B-A3B rows (best SSD config 47.1 tok/s on an M5 Max
128 GB). It has nothing >= 80B, so it is out of scope.

**Best exact (4-bit):** SlotStream, 23.69 tok/s on a 48 GB machine (33 GB ceiling), and 15.86 tok/s at a 22 GB process target.
**Best lossy:** Kapitsa, 61.3 tok/s (Q2_0, 16 GB expert store, 48 GB machine). Its routing is exact and its drafting is verified.

## (3) Large-DRAM offload (the model fits in VRAM + DRAM)

| Project / poster | Model (total / active) | Precision + quality claim | Fast memory | Storage | Hardware | Decode tok/s | Date | Link | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| Strata (u/KnownAd4832, own CUDA engine) | Qwen3.8-Flash-Next | GSQ-RCO Q2_0 / IQ2_XS / IQ3_XXS | 12 GB VRAM + 64 GB DDR5-5600 | NVMe (fits in RAM + VRAM) | RTX 5070 + Ryzen 5 7600, Windows | 65.1 / 52.0 / **44.8** at 128K context. Prefill 543 / 472 / 414. | 2026-09-24 | [r/LocalLLaMA 1wp7zyb](https://www.reddit.com/r/LocalLLaMA/comments/1wp7zyb/), [Strata](https://github.com/Niko1221/Strata) | sure on the table (archive). The body says IQ3_XXS gets "~65 tok/s", which contradicts the table. |
| KTransformers V0.2 / V0.2.1 / V0.3 | DeepSeek-V3/R1 (671B / 37B) | Q4_K_M (int4). The "6 experts" rows use 6 of 8 routed experts (lossy-route). | 382 GB (1 socket) / 1 TB (2 sockets) DRAM + 4090(D) 24 GB (>= 14 GB used) | n/a | 2x Xeon Gold 6454S (server CPU, not consumer) | 8 experts: 8.73 (1 socket), 12.208 (2 sockets). V0.2.1: 13.4 / 13.2 / 12.4 at 2 / 4K / 8K. 6 experts: 10.303 / 13.69, and V0.2.1 up to **16.8**. llama.cpp: 4.51. Prefill up to 286.55 (V0.3, AMX). | 2025-02-10 / 02-15 | [tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md) | sure |
| KTransformers | Kimi-K2 (1T / 32B) | Q4_K_M, 384 experts | ~600 GB DRAM + 14 GB VRAM | n/a | "single-socket CPU + one consumer-grade GPU". Dual socket with NUMA. | ~10 (1 socket). ~14 (2 sockets). | support from 2025-07-11 | [Kimi-K2.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Kimi-K2.md) | unsure ("TPS" not labelled decode; hardware not named) |

**Best exact:** Strata IQ3_XXS, 44.8 tok/s (3.0 bpw, all experts). For DeepSeek, KTransformers' 13.4 tok/s with all 8 experts.
**Best lossy:** Strata Q2_0, 65.1 tok/s. For DeepSeek, KTransformers' 16.8 tok/s with 6 experts.

## (4) Context line: whole model in RAM

Kapitsa reports Qwen3.8-Flash-Next 4-bit (105.9 GB) at 108 tok/s decode and 1,701 tok/s prefill on an M5 Max with 128 GB
(2026-11-01). Splash's README reports Q2_0 at 95 tok/s on a 64 GB M5 Pro.

## Leads not verified (the page was not read or has no number)

- A llama.cpp expert-cache PR, from a post titled "Qwen3.8-Flash-Next on 2x3090 + DDR4: 17 -> 25-29 t/s decode with the expert cache PR"
  ([prismix mirror](https://prismix.dev/news/0713121e4e66)). It belongs in large-DRAM, with 2 GPUs.
- SwiftLM (SharpAI): SSD expert streaming for large Qwen MoEs. No numbers were read.
- Splash 1.4.0 release notes: "Flash-Next Q2_0 with a 12G expert store gained 13-38% decode throughput". The notes give no absolute tok/s
  and no machine size ([releases](https://github.com/incoai/splash/releases)).
- tayoun/flash-moe: Qwen3.5-35B-A3B at 11.5 tok/s on a 16 GB M4 Mac mini. That is 35B, out of scope.
