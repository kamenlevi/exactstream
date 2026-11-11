# Related-work check for the paper set (MoonShots; due Tue 10 Nov 12:00). Pass 1: Sun 8 Nov 17:00-17:1x EET

Every source below was re-opened today with `research/relwork/check.py` (all arXiv ids in one API query, every URL
fetched read-only) and `ctx.py` (the lines about our model). Raw extracts: `check_1108_1700.txt`. All competitor numbers
are SELF-REPORTED unless we reproduced them (only Strata, findings/2026-11-08_1630). Third-party text was read as data.

## 1. Citations (arXiv): 68 ids in MAIN.md + RESULTS_FIRST + EXTERNAL_* files
- 68 / 68 exist; titles and first authors are in `check_1108_1700.txt`.
- **Wrong name (fix):** 2511.14102 is "Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative
  Offloading". MAIN cited it as "MoE-SpeQ"; no arXiv paper of that name exists (arXiv search, 0 hits).
- **Name is the method, not the title (fine; cite with title):** 2605.00342 = EVICT ("Making Every Verified Token
  Count"); 2607.12696 = EcoSpec ("Less Experts, Faster Decoding"); 2609.02897 = AdaptiveSpec ("Margins, Not Windows").
- **Wording to check:** 2512.17073 ("Bandwidth-Efficient Adaptive MoE via Low-Rank Compensation") is cited as "token-level
  precision restoration". The abstract fits loosely; cite it by title.
- **Ambiguous label:** RESULTS_FIRST's "2606.10493 / KTransformers" row. 2606.10493 is "Achieving Cloud-Grade SLOs for
  Local MoE Inference through CPU-GPU Hybrid Design", a separate paper. KTransformers' Expert Deferral is the SOSP '25
  paper. Keep them apart.
- Still to open by hand: KTransformers (SOSP '25), Thinking Machines Lab "Defeating Nondeterminism" (Sep 2025),
  Leviathan et al. 2023, Pre-gated MoE (2023), QuaRot / SpinQuant, GPUDirect Storage / BaM, the patent and HotOS '15 items,
  and the benchmark papers (HumanEval / EvalPlus, MBPP, IFEval, GSM8K, MATH-500): no arXiv ids in the draft yet.

## 2. Competitors on OUR model (Qwen3.8-Flash-Next), re-opened
| who | what the source says today | our record (4 / 7 Nov) | status |
|---|---|---|---|
| Splash (github.com/incoai/splash, README "Performance" + release 1.4.0 notes) | Q2_0 (ISTA-DASLab GSQ-RCO, 2.4 bpw) on an **M5 Pro, 20-core GPU, 64 GB**: decode 95 tok/s on short prompts (1.3x Strata serving the same GGUF on that Mac), prefill 1,312 tok/s at 32K, cached TTFT 171 ms on a 32K replay, 171 tok/s aggregate with 4 concurrent requests; SPEED-Bench coding prompts, 1,024-token outputs; all replayed prompts hit its SSD cache. Releases now at 1.6.0 | 1.4.0: 28.8 on a 24 GB Air; 1.5.0: 56.5 (measured by Kapitsa) | **NEW and material**: same machine class, self-reported |
| Strata (github.com/Niko1221/Strata; issue #2844) | MIT. README lists Qwen3.8-Flash-Next packs (ISTA-DAS, Unsloth UD-IQ4_XS ... Q8_0, Gyro). Issue #2844: M1 Max 64 GB, Q2_0, decode 26.7 tok/s; cites an M5 Max 128 GB table at 119.5 / 108.2; engine RSS 40.3 GB | same | confirmed; our own run (IQ3_XXS, 44-47 tok/s) is the measured comparison; Splash puts Strata at ~73 on Q2_0 short prompts [P: 95 / 1.3] |
| Kapitsa (github.com/nace-ai/kapitsa) | "A 180B model on a 48 GB Mac, at 61 tok/s (M5 Pro)", experts streamed from the SSD (Q2_0); bench file 2026-10-28 (M5 Pro 48 GB, four engines); NEW 2026/11: "Qwen3.8-Flash-Next 180B writes 6% faster on an M5 Max, with the same tokens: greedy decode 107.3 -> 113.9 tok/s" | 61.3 (README) | confirmed; 113.9 on an M5 Max is new (memory setting to read) |
| SlotStream (github.com/carloslfu/slotstream) | original MLX 4-bit, exact; "Our 48 GB M5 Pro measured 20.91-23.69 tok/s ... at a 33 GB memory budget" (5 Oct 2026); latest release v0.2.30; issue #5 (Mac mini M2 16 GB, 10.2 GB target), issue #41 (M4 Pro 24 GB, 15.9 GB budget) | 23.69 / 15.86 / 5.41 / 1.48 | confirmed (README); issue figures to re-read for the exact tok/s lines |
| Evolving Agent Teams (arXiv 2610.36848) | "30.3 tokens/s decode for Qwen3.8-Flash-Next in MLX 4-bit on a 48 GB M4 Pro" (a Metal port of FreeToken streaming experts from SSD) | 30.3 | confirmed |
| Palm-Infra (github.com/TencentYoutuResearch/Palm-Infra) | NEW row: "Qwen3.8-Flash-Next NVFP4, 20 GiB: 16.2 t/s / 21.8 t/s (dagger: best observed real-prompt run, eight CPU threads, 20 GiB expert cache)"; CPU NVFP4 expert SSD offload, experimental CUDA | only the 122B row | new; CPU-side engine, different class |

## 3. Other >= 80B MoE rows, re-opened
| who | source says | status |
|---|---|---|
| Memory-Sovereign Inference (arXiv 2608.23805) | Qwen3-Next-80B-A3B in Q4_K_M; "a 512+64 mechanism diagnostic decodes at 5.037 tokens/s without reuse and 11.636 tokens/s with LRU64"; bit-identical logits and exact tokens for its held-consumer case | confirmed, but 11.6 is a mechanism diagnostic, "not a policy study": cite it as that |
| PagedMoE (arXiv 2611.03173) | Qwen3.5-397B-A17B, 24 GiB expert cache, 8.89 tokens/s (1.33x independent precision); PC with RTX 5090 32 GiB + 64 GB DDR5 | confirmed |
| flash-moe, Anemll fork, anemll-flash-mlx, mlx-flash, colibri | READMEs reachable (status 200) | lines to re-read if the papers cite them |

## 4. Not readable this way
- Reddit threads (1xezdgs "flash next q2 on a 24gb m5 macbook air 49 t/s", 1w53inj, 1wp7zyb, 1ibbloy, 1idseqb): the page is
  JS-rendered. The titles match the records; the bodies need a browser read or must be cited as "a user post".
- "Sushi" (named by the orchestrator): in no project file; asked for its link (Sun 17:1x).

## 5. Consequences for the papers (as of pass 1)
- [MAIN] section 8: replace the v1.1 rows with section 2 above; add Splash's M5 Pro 64 GB row; mark all self-reported.
- [MAIN] abstract, where we lose: a 2.4-bit engine reports 95 tok/s decode and a far faster prefill on this machine
  class (self-reported); our Strata run used Strata's 3-bit file.
- [MAIN] section 12: "MoE-SpeQ (2511.14102)" -> "Specter (2511.14102)".
- No "fastest" claim on chat, on exact decode, on prefill or on first token. A decode claim on code would need Splash run
  on our prompts (decision asked, Sun 17:2x).
- Pass 2 (Mon-Tue): the companion papers' reference lists, the hand-opened items in section 1, the issue figures.

## 6. Pass 2a (Sun 8 Nov 20:2x-20:4x EET): references and new engines
- **arXiv (check_1108_pass2a.txt):** 17 / 17 exist; titles and first authors match: HumanEval 2107.03374, EvalPlus
  2305.01210, MBPP 2108.07732, IFEval 2311.07911, GSM8K 2110.14168, MATH 2103.03874, MATH-500 (Let's Verify) 2305.20050,
  Leviathan 2211.17192, Chen (speculative sampling) 2302.01318, Pre-gated MoE 2308.12066, QuaRot 2404.00456, SpinQuant
  2405.16406, BaM 2203.04910, LLM in a flash 2312.11514, Fiddler 2402.07033, QTIP 2406.11235, and the architecture paper
  2608.30320 ("On the Design of Qwen3.8-Next Architecture", Qiu et al.).
- **Hand-opened:** Thinking Machines (H. He, Connectionism, 10 Sep 2025, doi 10.64434/tml.20250910); KTransformers SOSP
  '25 (Chen et al., doi 10.1145/3731569.3764843; "a novel Expert Deferral mechanism", madsys.cs.tsinghua.edu.cn page);
  US 11,789,658 B2 (SK Hynix; a PCIe / NVMe interface; exit latency of the L1.2 low-power state); HotOS '15 "Software
  Engagement with Sleeping CPUs" (Q. Zhu, M. Zhu, B. Wu, X. Shen, K. Shen, Z. Wang); GPUDirect Storage (NVIDIA page);
  prompt-lookup decoding (README); Apple Newsroom (Oct 2025): M5 GPU with "a Neural Accelerator in each core".
  developer.apple.com/metal has no "tensor" wording on its overview page: MAIN cites the Newsroom line only.
- **"Sushi" = github.com/beamivalice/sushi** (Zig; MIT; a detached fork of mlx-serve; releases 1.4.1 30 Oct, 1.5.0 4 Nov,
  1.5.1 6 Nov). README, self-reported: Qwen3.8-Flash-Next packs 2 bpw (48 GB+, or 32 GB streamed), STQ 2.75 bpw (64 GB+;
  weights 44.9 GiB), 4 bpw (96 GB+, or 64 GB streamed). Speed table (Sushi 1.4.0, M5 Max 128 GB): STQ-2.75 decode 168.6 /
  151.1 / 146.1 / 146.8 / 151.0 / 138.2 tok/s at 4K-128K context, prefill 1,122-2,049; 4 bpw decode 122.0-153.5, prefill
  2,079-2,379. MTP drafting for Qwen. Exactness not stated. A Reddit title quotes "~90 tk/s" on an "m5u 96GB": not
  opened (JS-rendered).
- **Mference (github.com/NeelM0906/Mference):** README: "Qwen3.8-Flash-Next, a 180B-parameter MoE: 24-210 prompt tok/s and
  18.2-21.0 decode tok/s resident on M3 Ultra". Self-reported.
- **No run of this model found** in the READMEs of SwiftLM (SharpAI; MIT; SSD streaming), mlx-flash (szibis; MIT), BaseRT
  (Apache-2.0), fiveangle/flash-moe. Edge0 (a blog) not opened.
- Applied to MAIN v3 (section 8 rows, 1.3, 4.1's 180 B note, section 12).

## 7. Pass 2b (Sun 8 Nov 21:1x-21:3x EET): KERNELS' source list
- MLX issue #4632: "[BUG] gather_qmm(sorted_indices=True) returns wrong results on M5 Max (macOS 27) when rows > 32768 and
  not a multiple of 64", opened 2026-10-05 by another user, closed. PR #3922 "Fix sorted gather_qmm NAX row overflow above
  32K", merged 2026-08-26. Our first hit: 18 Oct (KERNELS' BUG1). Wording: "reported upstream before we hit it".
- batch_invariant_ops (thinking-machines-lab; MIT; created 10 Sep 2025). vLLM docs "Batch Invariance" (page reachable).
  LMSYS "Towards Deterministic Inference in SGLang and Reproducible RL Training" (22 Sep 2025). DFlash: github.com/z-lab/dflash
  ("DFlash: Block Diffusion for Flash Speculative Decoding"; MIT).
- **MTPLX (github.com/youssofal/MTPLX; Apache-2.0)**, README, self-reported, M5 Max 128 GB: Qwen 3.8 Flash Next 125.8 tok/s
  (one OpenCode request, 1,301 tokens, 18,364 of 18,539 prompt tokens cached, depth 3, 2.11.3, 16 Sep 2026), 79.3 (9K-token
  code prompt, 1,500 tokens), 63.5 (65K prompt; prefill 1,094 tok/s), 61.8 / 50.3 (109K / 200K turns). Packs: Bare Speed
  flat 4-bit (96 GB+, peak 78 GiB), Optimized Speed (128 GB+, 87 GiB); n-gram table streamed from SSD. "Exact rejection
  sampling, correct at any temperature". First Apple Silicon backend for the family: 29 Aug 2026 (as stated).
  **Licence:** NOTICE requires "Powered by MTPLX" + URL displayed inside any product that includes or is built on it
  (CLI banner or shipped docs); research that builds on it should credit "MTPLX by Youssof Altoukhi" + link and cite
  CITATION.cff (v2.11.2). The package pip-installs MTPLX from a pinned commit (not vendored; Leap 21:4x) and prints the
  line in run.py's banner + README credits, with third_party/LICENSE-mtplx + NOTICE-mtplx (package a2eedbc).

## 8. Pass 2c (Mon 9 Nov 2026, 00:2x-00:5x)
- **MAIN section 12, arXiv ids:** the 33 ids not recorded in passes 1-2b were looked up through the arXiv API
  (check_1109_pass2c.txt: id, title, first author, first version). All 33 resolve, and MAIN's label matches the title for
  32. WhiFlash (2606.07710) was labelled "drafter by task". Its abstract describes token-level routing between an
  autoregressive and a diffusion drafter within one sequence, so MAIN now says that. All 58 section 12 ids are now
  checked against their titles.
- **DFlash** (2602.06036): "DFlash: Block Diffusion for Flash Speculative Decoding", Jian Chen, Yesheng Liang, Zhijian Liu;
  v1 5 Feb 2026, v2 28 May 2026; "Accepted at ICML 2026".
- **DFlash2 checkpoint** (SMALL-RAM TBD-26b; MAIN 5.1 / 12):
  - The HF API for incoai/Qwen3.8-Flash-Next-DFlash2 gives sha a22a1744e85c17364dcc54a6a420af7135d4c7f6, created and last
    modified 19 Oct 2026, licence apache-2.0 (tags + cardData), base_model Qwen/Qwen3.8-Flash-Next.
  - Our hub cache holds the same revision (refs/main = a22a174, config.json + model.safetensors, fetched 23 Oct). BigMoE's
    r57 screens and Fit12's t3_twopass.py both load it by that repo id.
  - The model card (Inco AI) describes a 0.6B BF16 block-diffusion drafter run inside a speculative-decoding server, with
    block size 4 in its own evaluation. It states no layer count. It asks for two citations: Inco AI, "DFlash 2: Keep
    Drafting Parallel" (blog, Aug 2026, inco.ai/blog/dflash2) and the DFlash paper above.
  - Strata's "tuned copy" is Strata's own, not ours.
- **Strata's file:** IQ3_XXS, two shards, 75,839,998,528 bytes = 75.8 GB (70.6 GiB) [K, SE:models/IQ3_XXS]. MAIN section 7
  no longer gives file size as the reason the 4-bit files went untested (the orchestrator's correction).
- **Strata's default-cache chat:** re-derived from SE:h2h/results/{B,D}{1,2}. First token median: D (exact read) 1.06 s,
  B (Q8_0) 0.98 s; MAIN section 7 had 0.98 s for both, now fixed. Whole request 28.2 / 29.5 confirmed.
- **Left for 12:00-16:00:** the companions' own reference lists, SlotStream's low-memory figures (issues #5 / #41) and
  Palm-Infra's 122B row (Fit12's RF rows), community posts, and Edge0.
- **Correction (01:0x):** both of our engines load the DFlash2 base and then our own fine-tuned weights (Leap's l120_ft).
  So we, like Strata, run a tuned copy, and the "not ours" line above means only that Strata's tuned copy is not ours.

## 9. Pass 2d (Mon 9 Nov 10:0x, editor): issue figures, Palm-Infra, Edge0, community posts
Fetched read-only with `gh api` (public issues and docs); raw copies in the session scratchpad only.
- **SlotStream issue #5** ("Measured: Mac14,3, 16GB", opened 2 Sep 2026, closed): a user's Mac mini M2 with 16 GB and base
  storage (about 1.5 GB/s). First run at a 10.2 GB target: decode 1.39-1.41 tok/s, peak 6.1 GB. Re-run of 3 Sep at a
  10.7 GB target: decode 1.42-1.48 tok/s, prefill 2.6 tok/s; 28-token prompt, 128 generated tokens; context check 8,192
  tokens at 8.1 GB peak RSS. SMALL_RAM's "1.48 tok/s at a 10.7 GB process target" is right; the machine (M2, base
  storage) and the user-report source were missing.
- **SlotStream issue #41** ("Measured: MacBook M4 Pro, 24 GB", 25 Sep 2026, closed 28 Sep): 3.52-3.61 tok/s at a 15.9 GB
  target (lifetime footprint peak 8.83 GB); on 0.2.25, 4.92-5.58 tok/s, 16.7 GB process peak in an 8,192-token
  context check. The maintainer's closing line: 5.41 tok/s on 0.2.25, up from 3.57.
- **Palm-Infra** (Apache-2.0; pushed 25 Oct 2026). README "Performance" and docs/ssd-offload.md:
  - The "122B" row is **Qwen3.5-122B-A10B** (a different model) with W4 weights on the CPU path. Cache sweep of 29 Jul
    2026: 16-token prompt, 256 generated, greedy, three processes per size. 1 GiB cache: 12.38 tok/s, peak RSS 5.90 GiB;
    10 GiB: 16.19 tok/s, 14.64 GiB; 16 GiB: 16.53 tok/s, 20.60 GiB. The doc names "a 48GB Apple Silicon Mac" for all four
    models and does not say "lossy" or "exact". SMALL_RAM's "a 122B model ... lossy" should name the model and say
    "4-bit (W4); exactness not stated".
  - **This model** (NVFP4 experts): 16.2 prefill / 21.8 decode tok/s are "best observed" with eight CPU threads and a 20 GiB
    cache, "not a strict five-process median". The controlled four-thread / 16 GiB run gives 18.70 / 15.74 (median of five
    processes). MAIN section 8's row put "16.2 / 21.8" in the decode column: fixed at 10:0x (decode 21.8 best observed,
    15.74 median; prefill in "other").
- **Edge0** (results_first/EXTERNAL_PAPERS.md row): a Qwen3.6-35B-A3B-derived model, lossy, 2.9 GiB. No draft cites it,
  and it is not this model nor >= 80B, so it stays out of the papers.
- **Community posts:** no draft cites a forum or social post (grep of all six drafts at 10:0x). Re-check the finals at 12:00.

## 10. Pass 2e (Mon 9 Nov 11:5x, editor): the companions' reference lists
Read-only: arXiv API, GitHub API (`gh api`), Hugging Face API, Crossref, two web pages; extracted facts only (raw copies in
the session scratchpad). Ids and titles: `check_1109_pass2e.txt`.
- **arXiv ids:** the companions cite 33 ids; 10 were not in MAIN's checked 58 (DRAFTER 3, EVALUATION 7). All 10 resolve,
  and each label matches its title and first author. DRAFTER section 6 named 14 works without ids. All 14 were found and
  their ids added: EAGLE 2401.15077, EAGLE-2 2406.16858, Medusa 2401.10774, SpecInfer 2305.09781, Sequoia 2402.12374,
  SpecExec 2406.02532, FR-Spec 2502.14856, SuffixDecoding 2411.04975, SpecDec++ 2405.19715, DISCO 2405.04304 (Mamou et
  al.), SP-MoE 2510.10302, Ghidorah 2505.23219, DuoDecoding 2503.00784, PEARL 2408.11850.
- **Fixed in DRAFTER:**
  - "MoE-SpeQ" -> Specter (2511.14102), as in MAIN since pass 1.
  - WhiFlash: "a drafter per task" -> switches between an autoregressive and a diffusion drafter token by token, as in
    MAIN since pass 2c.
  - "AngelSpec, Microsoft 2610.00888 [[RW: titles]]" -> AngelSpec (Liu et al., 2607.25852) and Badarayani et al.,
    2610.00888. The arXiv record shows no affiliation, so "Microsoft" was dropped.
  - RW-1: the DFlash2 base checkpoint, as in RELEASE 7 and pass 2c.
  - RW-3: EAGLE-3's training data is ShareGPT + UltraChat-200K, about 68K + 464K entries, with responses regenerated by
    the target, plus OpenThoughts-114k-math for its reasoning model. The paper states no token count (arxiv.org/html/
    2503.01840v3, Implementation).
  - RW-4: MLX documents MLX_ENABLE_TF32 ("The default is 1": docs/src/usage/environment_variables.rst; precision.rst).
    MLX issue #4595 (29 Sep 2026, closed 27 Oct) covers TF32 on fp32 conv on the M5 since 0.32.
- **Fixed in KERNELS 8:** "Our drafters are DFlash (z-lab, MIT)" -> our fine-tune of DFlash2 (Inco AI; Apache-2.0), a
  DFlash-method block drafter (2602.06036; code z-lab, MIT), as STYLE has said since Mon 01:0x.
- **Checked in KERNELS 8 (no change):**
  - LLM-42 2601.17768 (all four authors match).
  - Thinking Machines (Horace He, 10 Sep 2025; its table: vLLM default 26 s, deterministic 55 s, improved attention
    42 s).
  - MLX issue #4632 (gather_qmm sorted_indices on M5 Max, rows > 32768 and not a multiple of 64; opened 5 Oct 2026,
    closed).
- **Fixed in RELEASE 7:** MTPLX ran this model's MTP head on Apple Silicon "first" -> "before we did" (BRIEF: no
  unchecked "first").
- **Checked in RELEASE 7 (no change):**
  - The model's licence, from the HF API and the LICENSE file at sha 812d96e: Qwen Community License 1.0, "Copyright
    (c) 2026 Qwen". It covers the 100M monthly-users / US$20M monthly-revenue display clause and the separate licence
    for Model-as-a-Service / AI Work Assistant businesses.
  - The 4-bit pack (Youssofal/...-MTPLX-Bare-Speed, sha 4285a4a) is under the same licence.
  - z-lab/dflash is MIT.
- **EVALUATION 7:** every item checked; only the intro line changed.
  - lm-evaluation-harness and simple-evals are MIT.
  - Crossref matches McNemar 1947 (Psychometrika 12(2) 153-157), Dietterich 1998 (Neural Computation 10(7) 1895-1923)
    and Schuirmann 1987 (J. Pharmacokinet. Biopharm. 15(6) 657-680).
  - llama.cpp's perplexity tool has --kl-divergence (tools/perplexity/README.md).
- **SMALL_RAM 9 / App. C:**
  - Every id is in MAIN's checked set; the KTransformers DOI 10.1145/3731569.3764843 matches Crossref (SOSP '25,
    1014-1029).
  - Pass 2d's two fixes are applied at 11:5x by the editor. SlotStream line: a user's 16 GB Mac mini M2 with base
    storage (issue #5). Palm-Infra line: Qwen3.5-122B-A10B, W4, 1 GiB cache, exactness not stated.
- **Community posts:** none cited in the six drafts (grep at 11:5x).

## 11. Pass 2f (Mon 9 Nov 2026, 17:0x): MAIN section 8 novelty, SSD wake penalty and keepalive (Kamen via the orchestrator)
Web searches (Claude Code WebSearch; 9 Nov 2026), with what they returned:
- "NVMe autonomous power state transition exit latency first read after idle measured microseconds SSD": NVMe power
  states with entry / exit latency are documented (Microsoft Learn, "NVMe": StorNVMe picks the deepest state whose
  ENLAT + EXLAT fits a latency tolerance after an idle timeout of 50-200 ms; example EXLAT 5 us to 10 ms; opened). Also the
  Linux APST patch (openwall, 29 Aug 2016) and a Netdata ASPM / APST latency guide. No measured first-read-after-idle data.
- "keepalive periodic reads keep SSD out of low power state reduce read latency": patents on wake penalties (DEVSLP,
  self-refresh) and NVMe-oF keep-alive messages; no source evaluates periodic reads to keep an SSD awake for latency.
- "Apple silicon Mac internal SSD read latency after idle power state measurement" (extended): none. Eclectic Light's M1
  storage tests give regression latencies only; AnandTech notes Apple's non-standard power handling for an older drive.
- "LLM inference streaming weights from SSD ... idle power state wake latency keep drive awake MoE experts" (extended):
  SSD offloading energy (arXiv 2508.06978), SSD-LLaMA (2609.18110), HBF paths (2608.14333), llama.cpp MoE streaming
  (discussion 27149, PR 25294), HOBBIT, DALI. None addresses SSD power-state wake or keepalive.
- "utility keeps external hard drive awake by writing periodically": NoSleepHD, KeepAliveHD, Prevent Disk Sleep, Project
  Torpid (github.com/YorkshireDev/Torpid, GPL-3.0; writes an empty file every 2.5 s by default; opened). Keeping drives
  awake with periodic I/O is published practice (for HDD spin-down).
- "GPU kernel spin-wait polling flag written by CPU in host-mapped or unified memory": NVIDIA developer forum threads
  (2007 on), Stack Overflow persistent-kernel answers, Modular's CompletionFlag docs. Polling a host-written flag is
  common practice; no first publication identified.
Verdict for MAIN 8.1: not new: NVMe idle states and exit latencies; keeping drives awake with periodic I/O; waking ahead of
predicted I/O (HotOS '15, already cited). New (no prior publication found): the penalty's size on Apple silicon's internal
SSD at sub-millisecond gaps, that SSD-streamed MoE per-layer bursts pay it, and the read spacing that removes it.
