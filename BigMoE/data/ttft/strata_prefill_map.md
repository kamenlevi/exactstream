# Strata: prompt -> first token on the streamed 125B MoE (Qwen3.8-Flash-Next IQ3_XXS, 64 GB M5 Pro)

Source read: `~/StrataEval/src` at 146df3c (Strata 0.1.58-metal). Code paths below are relative to `~/StrataEval/src`;
`h2h/...` means `~/StrataEval/h2h/...`. "inferred" = not read directly in code or logs. Nothing was run.
Config of the measured runs (h2h/results_arena33/B1.summary.json): `strata --serve --pack ... --native <shard1> --ple-gguf <shard2>
--expert-profile data/expert-profile.bin --expert-cache 64 --prefill auto --max-context 32768 --mmap-experts --spec 2
--suffix-draft 0 --prompt-cache 0 --conversation-cache-mib 0 [--mtp rt]`, env `STRATA_METAL_STREAM=1 STRATA_METAL_ARENA_GB=33`.

## 0. Findings in brief

1. **The 0.83 s is a median over two different code paths.** In serve mode a prompt part of at most 64 tokens is read
   through the *decode verify windows*, S = `--spec` = 2 rows at a time; a longer part goes to the batched Metal
   *GEMM chunk* path (generate.cpp:964 `short_read = 64`; 16718-16724 `windows_ok`; 17926-17943; S = o.spec: 851, 13997, 16878).
   With `--prompt-cache 0` the part is the whole prompt minus its last token, so n_prompt <= 65 -> windows, >= 66 -> GEMM.
   IFEval (N45) at arena 33 GB, B1, wall TTFT: 37 tok 656 ms, 42: 760, 43: 930, 46: 909, 48: 978, 52: 1108, 54: 1123,
   57: 1275 (windows) | 66: 454, 74: 501, 78: 503, 101: 519 (GEMM); median 0.834 s (h2h/results_arena33/B1.requests.jsonl).
   **A ~50-token chat prompt costs Strata ~1.0-1.1 s, not 0.83 s**; Strata's own GEMM path does 66-101 tokens in ~0.45-0.52 s.
2. **Every routed expert missing from the arena is read** (both paths). Nothing is skipped or approximated for experts on
   Metal and no CPU expert lane runs. The GEMM path also streams **all** non-resident experts of layer l+1 behind layer l
   (queued reads not yet started and not routed are dropped at the next layer). The window path reads ahead the top 6 per token of
   layer l+1, predicted by l+1's router applied to layer l's FFN input. This is the same predictor as ours.
3. **Most of the speed comes from residency.** A 33 GB mlocked anonymous arena holds 17,723 of 24,576 experts (72%) as
   byte-exact GGUF bytes (~1.86 MB a slot). It is filled from a static shipped frequency ranking **lazily during the first
   request** (the discarded warm-up: TTFT 12.3 s). Strata's own fitted coverage curve gives ~97.8% of routed mass for 72% held
   (generate.cpp:6094-6104; inferred for prompt traffic). TTFT tracks arena size (IFEval median, B arm): 0.83 s at 33 GB,
   0.98 at ~26 GB (default), 1.65 at 16 GB, 2.68 at 4 GB, 2.78 at 3.37 GB.
4. **I/O:** `pread` on an fd with `F_NOCACHE` + `F_RDAHEAD 0`, 6 I/O threads (blocking, so queue depth <= 6), one
   request per expert *slice* (gate, up, down read separately; ~0.6 MB average, inferred). Urgent misses go before read-ahead.
   This path uses no preadv, mmap reads, madvise, F_RDADVISE, MTLIOCommandQueue or dispatch_io.
5. **The GPU stalls at every layer.** One command buffer per window/chunk carries 48 shared-event handshakes. The host
   busy-spins, reads that layer's misses synchronously while the GPU waits, queues the read-ahead, then releases the GPU
   (mtl_rt.mm:1051-1089).

## 1. Timeline (serve mode, h2h config, arena 33 GB)

| # | Phase | What the code does (file:line) | SSD reads: bytes; API; request size; threads/QD; order | Overlaps with | Measured / logged |
|---|---|---|---|---|---|
| 0a | Process start -> `READY` (not in TTFT) | Window::init (metal/src/engine/window.cpp:680-907): shard 1 mmapped read-only for the header (metal/src/gguf_lite.cpp:58-69); no WILLNEED when streaming (window.cpp:695-707 is the resident branch). ExpertStream::open (metal/src/engine/stream.cpp:60-166). Dense tensors into one buffer W_ (stream.cpp:180-220). HC matrices to Q8_0 (window.cpp:~840-875). Predictor buffer predL_ (window.cpp:882-885). | Dense 2.84 GB: pread on the F_NOCACHE fd, 4 threads, <= 64 MiB a call (stream.cpp:168-178, 203-214). The PLE table (shard 2) opens with a 16-thread F_NOCACHE pool (src/platform/direct_file.cpp:307-322). | 4 reader threads | B1.engine.log: "789 dense tensors read into memory (2.84 GB, 0.2 s)" (~14 GB/s: likely page-cache served, inferred); HC Q8 680 MB in 81 ms; "decode window ... ready in 312 ms"; READY at 1.27 s (B1.summary.json `ready_s`). |
| 0b | **First request only** (the discarded warm-up): arena build | Lazy: Window::run / prompt_gemm_init / prompt_chunk call stream_arena when the arena is not ready (window.cpp:1477, 2907, 3003; 1641-1735). The budget comes from arena_budget.h:112-118 (`STRATA_METAL_ARENA_GB` overrides). make_arena (stream.cpp:373-574): profile ranking (381-402); >= 576 slots per size class when the GEMM path is on (window.cpp:1714; stream.cpp:403-421); <= 4 MAP_ANON chunks wrapped no-copy (stream.cpp:422-465); smap 48x512 (470-471). Fill in 1.5 GB steps, each step sorted by file offset (484-517). LRU order = rank order (532-533). mlock (553-571). Prompt buffers placed in a 1.39 GB region of the arena (window.cpp:2905-2994; stream.cpp:721-746). | 33.0 GB = 17,723 experts: pread F_NOCACHE, 3 requests per expert (one per slice), 6 threads, urgent queue, file order within a step. | Fill runs on the 6 I/O threads while the host waits | "expert arena 33.00 GB, 17723 slots in 3 size classes ... in 6.0 s (5.16 GB/s)" (B1; D1 2.9 s); "33.0 GB ... wired (mlock) in 4280 ms"; warm-up TTFT 12.32 s (B1), 5.44 s (D1), 4.25 s (default ~26 GB arena). |
| 1 | `GEN` received | serve loop (src/program/generate.cpp:15699-15700); r0 at 15979. Token ids arrive on the line: the client applies the tokenizer and chat template (h2h ids from EvalSuite `prompt_ids`, h2h/make_prompts.py:3-9). | none | - | ~0 |
| 2 | Split and path choice | Cut points root/turn/message/reread are -1 with `--prompt-cache 0` and `ckpt=0` (generate.cpp:17837-17869), so there is one part [0, n-1) (17894). A part of <= 64 tokens goes through windows, otherwise batched (16718-16724, 17926-17943). | none | - | - |
| 3A | **Windows path** (n-1 <= 64; every ~50-token chat prompt) | read_windows (16862-16935): ceil((n-1)/2) windows of S = 2 rows, head off (16868-16877). Each window: Verifier::run -> window_run (src/core/verify.cpp:3245-3259; the host pool is ignored at 3247) -> Window::run (window.cpp:1406-1559). The CPU dequantizes the embedding rows into the shared buffer (1411-1419) and gathers the PLE rows (1420-1431). The cached graph for this window shape is patched (1447-1476) and run with run_graph_seg (1610-1613). At each of the 48 layer cuts (cut placed before the expert kernels, window.cpp:1315) the host runs at_layer(l, T=2) (stream.cpp:619-717). Then commit (lazy fold, window.cpp:2318-2361) and the drafter's K/V for the 2 rows (generate.cpp:16914 -> src/core/mtp.cpp:1564-1568 -> window.cpp:2742-2763: a separate synchronous command buffer). | Per layer: the misses among <= 20 routed ids, urgent, 3 preads each, QD <= 6, in first-seen order (stream.cpp:624-632, 653-691). Then read-ahead of layer l+1: per token the top 6 predicted logits, union, best logit first, non-resident only, into LRU-tail victims (stream.cpp:577-617, 713-715; top 6 because experts average > 1.5 MB, stream.cpp:153-162). PLE: 16 table rows a token (~90 B, IQ4_NL) from shard 2, F_NOCACHE pread on 16 threads, behind an 8-way row cache (src/ngram/ple_reader.cpp:35-60; direct_file.cpp:264-322). Bytes: not logged (bound in section 2). | The GPU waits during the host's turn: the host spins on the shared event (mtl_rt.mm:1074-1081) and the miss reads are serial with the GPU. The read-ahead overlaps the rest of layer l and layer l+1's attention. Embedding and PLE run on the CPU before each command buffer (serial). | Median per 2-row window: 38.2 ms at 33 GB (15 requests, median n 53, prompt_ms 992); 46.5 ms at ~26 GB; 85 ms at 16 GB; 126 ms at 4 GB (computed from h2h/*/B1.requests.jsonl). |
| 3B | **GEMM chunk path** (n-1 > 64) | read_part -> sp.run (generate.cpp:17130-17246) -> Prefill::run Metal branch (src/prefill/prefill.cpp:7926-7942) -> prompt_run_rows (window.cpp:3593-3617). hold_region(true) (stream.cpp:766-788) cancels queued read-aheads, waits for reads in flight, empties 911 slots and memsets 1.39 GB. Then prompt_chunk for each <= 1024 rows (window.cpp:2996-3339), with the CPU embedding and PLE gather inline (3013-3034). The 48-layer graph is encoded fresh for each chunk and run with run_graph_seg, one cut per layer (3243, 3306-3311). At each cut at_layer(T > 8) reads the urgent misses and waits, then queues **all** non-resident experts of l+1 in expert-index order (stream.cpp:692-711). hold_region(false) at the end (789-792). | Misses for the chunk's distinct routed experts (urgent, 3 preads each, QD <= 6). Then the speculative stream of up to all ~143 non-resident experts a layer (6,853 in total, ~12.7 GB at 1.86 MB each, inferred) into LRU-tail victims. Queued reads that have not started and are not routed are dropped at the next cut (stream.cpp:642-650). | The l+1 stream overlaps layer l's expert GEMMs and l+1's attention GEMM / GDN scan. The drafter's prompt rows run on a second queue, overlapping the next chunk (window.cpp:591-596, 2761). | Fit prompt_ms = 354 + 1.57 x rows at 33 GB (66 requests, B1 + D1). The fixed part grows as the arena shrinks: 537 ms (~26 GB), 1084 ms (16 GB). A 66-token prompt: 422 ms; a 520-token prompt: 1111 ms (B1.engine.log). |
| 4 | Prompt done | `PP` line; refill (nothing lent on Metal: inferred; the host cache is 84 slots, B1.engine.log "too few cache slots to borrow"). prompt_ms is measured here and then `REUSED` is printed (generate.cpp:18007-18008). | none | - | DONE's prompt_ms = r0 -> here. |
| 5 | First decode window -> first token | T = 1 for the last prompt token (generate.cpp:18891). ver.run (19141) -> Window::run with the head: output.weight GEMV, then argmax_part (120 parts) and argmax_final, both on the GPU (window.cpp:1533-1549). at_layer(T=1) at each cut. commit, then `T <id>` is printed and flushed (19247-19258). | Misses for that token's 10 routed experts a layer, plus the top-6 read-ahead | as 3A | Wall TTFT - prompt_ms: 31.3 ms median at 33 GB (B1). TTFT definition: GEN written -> first `T` line (h2h/run_arm.py:264-299). |

CLI mode differs (for reference): `prefill.run(tokens, n_prompt-1)` (generate.cpp:20073) goes straight to Metal `prompt_run`.
There, < 64 rows use the backend's own fallback with **8-row** windows (window.cpp:3618-3626, MAXT = 8 at window.cpp:100), not 2-row ones.

## 2. Experts not in the arena during prefill

- **(a) Every routed expert is read. No (b) subset and no (c) skip.** at_layer collects every distinct id of the T x 10
  routed ids (stream.cpp:624-632). It looks up each one: hits are pinned; queued read-aheads are promoted to urgent and waited
  for (653-671); misses take a class-LRU victim (258-272) and are read urgently (673-679). The GPU is released only after
  **all** of them have landed (682-691; mtl_rt.mm:1079-1080). A slot holds the expert's GGUF bytes unchanged, so the window path's
  answers are those of the resident engine (stream.h:15-16). The only skip is `STRATA_METAL_STREAM_FAKE=1` (timing only, wrong answers;
  stream.cpp:163-164, 633-638).
  The CUDA tricks are inactive on Metal. `STRATA_ROUTE_TAIL_SKIP` is printed in the log but lives in the host pool
  (src/core/expert_source.cpp:4963-5036), which Metal never calls (verify.cpp:3247 `(void) pool`). The "prefill CPU share"
  line is printed by Prefill::arm_cpu_share (prefill.cpp:2968-2975), but Prefill::run's Metal branch returns before run_impl
  (prefill.cpp:7926-7942).
- **Precision.** The window path is exact. The GEMM path differs numerically: fp16 operands on the matrix units ("Its numerics
  differ from the window's (fp16 operands)", window.cpp:2856-2862), XS/HS in half (2966-2967). The hyper-connection matrices are Q8_0
  by default in both paths (`STRATA_HC_Q8=0` restores BF16; window.cpp:3063). Top-k stays 10 in prefill (TOPK=10, window.cpp:100;
  route_topk at 3190).
- **Order and overlap.**
  - Window path: urgent FIFO in first-seen order. The 6 I/O threads take urgent work first, then read-ahead (stream.cpp:309-335). The read-ahead
    is ordered by best predicted logit (598). The "next layer" is predicted by a GEMV of layer l+1's `ffn_gate_inp` on layer l's FFN input XF_,
    in the same command buffer as layer l's router (window.cpp:1200-1210). Layer l+2 is opt-in (`STRATA_METAL_PREFETCH2`, default 0; stream.cpp:160).
    Read-ahead slots sit at the LRU tail (pre_mru_ = 0 for > 1.5 MB experts, stream.cpp:162, 610-611). A miss is moved to the LRU front
    (678), so later windows of the same prompt hit it.
  - GEMM path: no prediction. Every non-resident expert of l+1 is queued in index order 0..511 (stream.cpp:699). Experts streamed for a prompt stay
    at the LRU tail and are not promoted ("the decode-hot experts stay", 669, 692-694), so they act as recycled scratch slots.
  - Expert pieces: `STRATA_METAL_READ_SPLIT` (default 1) splits each slice into 4 KiB-aligned pieces spread over threads
    (stream.cpp:152, 274-307). The optional `<gguf>.experts` contiguous copy (one request per expert) was not present
    (RUN_PLAN.md:193-197; no "experts read from ... .experts" line in the logs).
- **Page cache / mmap.** Experts never go through the page cache by design: F_NOCACHE + no read-ahead on the stream fd
  (stream.cpp:61-64), and the GGUF mapping gets no WILLNEED in streaming mode. The host's `--mmap-experts` mapping
  serves only host-side code, which Metal decode/prefill never calls. Inferred OS caveat: F_NOCACHE does not bypass pages already
  in the unified buffer cache, so earlier cached reads could be served from RAM. The dense load's ~14 GB/s suggests that
  happened at startup. The arena fill's 11.2-11.6 GB/s (default, 16 GB and 4 GB arms) looks like cold SSD at QD 6 (inferred).
- **Persistent state.**
  - *Across processes:* only the static ranking `data/expert-profile.bin` (STRP v1, 48x512, all 24,576 pairs). It is shipped and
    built offline by tools/make_profile.py from routing traces; the engine only reads it (stream.cpp:381-398). `--expert-profile-save`
    (learned ranking written at QUIT) is opt-in and was off (generate.cpp:778-780, 13661-13671). Metal kernels are compiled from source on first use
    (RUN_PLAN.md:334-335). There is no KV/prompt cache on disk: `--prompt-cache 0 --conversation-cache-mib 0` and `ckpt=0` were set, and every DONE has reused = 0.
  - *Within the process:* the arena and its LRU carry over between requests. One discarded warm-up request (gsm8k/708, 53 tokens)
    builds the arena (row 0b). Serve on Metal ends with `std::_Exit(0)` (generate.cpp:19924-19929), so the atexit stream report
    (prompt hit/miss/streamed counts, window.cpp:1732-1733; stream.cpp:805-824) was never printed. That is why no bytes-per-prompt figure exists.
  - *SSD keep-awake:* while table rows are requested, one PLE table page is read after 100 ms without a read, until 60 s after the last request.
    The reason given: some SSDs stall first reads 50-150 ms after ~250 ms idle (generate.cpp:5356-5365; B1.engine.log).
- **Bytes per prompt (not logged; inferred bound).** A ~50-token prompt has 49 x 48 x 10 = 23,520 routings. At Strata's curve
  (2.2% miss at 72% held) that is <= ~520 first-time misses, about 1 GB at 1.86 MB. Repeats within the prompt hit after the first read, and
  mispredicted read-aheads add bytes. For a GEMM chunk the bytes lie between the chunk's distinct misses and min(12.7 GB, about
  11 GB/s x chunk time).

## 3. Prefill compute layout and the arena

- **Chunk:** 1024 rows on a Mac under 96 GB, 4096 rows at 96 GB or more (`STRATA_METAL_PROMPT_CHUNK`; window.cpp:2908-2917).
  The log says "chunks of 1024 (experts v3, QSA flash, matmul2d layout K4; 1.39 GB of buffers, in the expert arena while a prompt runs)".
- **GEMM-path layer** (window.cpp:3099-3298):
  - Hyper-connection read: each dense matrix is dequantized to fp16 per chunk, then gemm_f16 runs on the matrix units (3049-3088).
  - Attention: 36 GDN layers (`l % 4 != 3`) do fused-input GEMMs, gdn_pre, a prompt scan (gdn_scanr_4 on M5 Pro), gdn_post and out-proj GEMM (3102-3124).
    The 12 QSA layers do q/k/v/indexer GEMMs, qsa_prep, qsa_topk and qsa_flash (3125-3172).
  - MoE: router GEMM, then route_topk (3188-3190). Tokens are grouped per expert into 16-row tiles with mm_hist, mm_scan and mm_place
    (3191-3199), and mm_gather builds the half rows (3209-3217). The cut comes here (3243). Then **mm_gu3 / mm_down3** grouped GEMMs
    with matmul2d (metal/kernels/moe_mm.metal:885-899), mm_combine, the shared expert and moe_out (3225-3276). All on the GPU.
- **Window-path layer:** per-token GEMVs; moe_route then moe_gu_/moe_down_ GEMV kernels on the slots (window.cpp:1191-1331).
- **In place:** yes. Each slot is an anonymous page range wrapped no-copy, and pread writes straight into it
  (stream.cpp:460-469, 296-306). Kernels compute `arena_slot(smap[e], W, W1, W2, W3)` = chunk (top 4 bits) and 16 KiB page, with
  gate/up/down at offsets inside the slot (metal/kernels/moe.metal:45-57, 234, 298; moe_mm.metal:272-284, 303, 358, 896, 976;
  window.cpp:1223-1227, 1309-1311, 3204, 3219-3223). io_main publishes smap only after the last slice lands (stream.cpp:325-331).
- **Selection rule:**
  - Start state: the profile's top pairs fill the budget. Each size class first gets >= 576 slots ("a prompt chunk's layer needs up
    to 512 at once"). 9 layer sizes are merged into 3 classes with <= 15% waste (stream.cpp:124-150, 403-421).
  - Steady state: LRU per class. Hits and window misses move to the front, prompt-chunk streams stay at the tail. Victims come from
    the tail and skip slots pinned by this handshake or being read (258-272).
  - The 1.39 GB prompt region evicts 911 slots at every GEMM-path prompt (stream.cpp:776-786). Those are the end of the last
    buffer, i.e. the lowest-ranked fill of the last class (inferred from the layout at 416-447, 488-509).
  - Budget: the footprint cap is RAM - max(15% RAM, 40% RAM, file-cache floor + 12 GB) (arena_budget.h:80-118).
    The 33 GB arena put the planned footprint at 44.3 GB, above the 41.2 GB cap; the engine kept it as set. The measured peak was 40.76 GB (B1).

## 4. Non-MoE costs on the path

- **Tokenizer and chat template:** outside the engine (Python server or client); not in the h2h TTFT (ids are pre-made).
- **Embedding:** CPU dequant of IQ3_S `token_embd` rows into the shared buffer, broadcast to 4 HC streams. Per window
  (window.cpp:1411-1419) or per chunk (3013-3023).
- **PLE (n-gram table, 28.8 GB shard 2):** 16 rows a token, gathered on the CPU from the SSD (F_NOCACHE, 16 threads, row cache)
  before the command buffer. Projections and the PLE block run on the GPU after layer 0 (window.cpp:1420-1431, 1486-1530; 3025-3034,
  3090-3096, 3278-3297). Logged probe: "PLE direct read probe: 134888 rows/s".
- **Attention:** GDN state and conv history plus QSA int8 K/V. Selection is the identity while pos + T <= 2051 (window.cpp:441-443, 1433-1439).
- **lm_head:** none in the prompt. Prompt windows run with head = false (generate.cpp:16868-16877) and the GEMM chunk ends at
  the final HC norm with no head (window.cpp:3299-3305). The only head is the first decode window's output.weight GEMV (window.cpp:1534).
- **Sampling:** greedy argmax in the graph (window.cpp:1535-1549). sample.metal is used when a sampler is set (verify.cpp:3248-3258).
- **Emission:** after commit, `printf("T %d")` + fflush (generate.cpp:19249-19258). First decode window plus emission is ~31 ms.
- **Host encode:** decode-window graphs are cached per shape and only positions are patched (window.cpp:477-483, 1447-1461).
  GEMM chunks are re-encoded every chunk (local `ds`, 3035).

## 5. Sushi (https://github.com/beamivalice/sushi)

Read through WebFetch only (by a helper agent), `main` @ fa1fd4469c9b (2026-11-09 12:14 UTC). Nothing was cloned or saved. Paths are
under `https://raw.githubusercontent.com/beamivalice/sushi/fa1fd4469c9b04bcc4c2ac0d6ba69a3dfa6f5dd1/`. WebFetch gives no reliable
line numbers, so citations are function names plus quoted fragments. I have not re-checked them myself.

- **Relevant: yes, the same model.** It is a Zig engine and server on MLX (mlx-c) with its own Metal kernels for its EXL3/STQ quants
  (a detached fork of mlx-serve), for M1+ and macOS 26.2+.
  - Its Qwen3.8-Flash-Next has the same geometry: 48 layers, 512 experts, top-10, hidden 2560, expert FF 640, 36 GDN layers (docs/qwen4-arch.md).
  - `--ssd-budget-gb N` is "a TOTAL resident target of N GiB"; only routed experts stream (docs/engine-expert-streaming.md).
  - License: the README says MIT, but the repo also has a LICENSE-APACHE-2.0 file. Check before porting.
- **Prompt -> first token (streamed Qwen):**
  - Jinja template plus BPE tokenizer, then a tokenize LRU and a prefix-cache lookup.
  - `--prefill-chunk` defaults to 8192, capped at 4096 for MoE (src/prefill_plan.zig boundedPrefillChunk).
  - With SSM checkpointing, a 30-token tail (`SSM_SNAPSHOT_BACKOFF`) is held back and run with the final forward. A 50-token prompt therefore
    probably becomes two forwards (inferred).
  - PLE rows are gathered from `ngram_table.bin` through a PrefetchPool; `startWarm` preads the table at load.
  - Each MoE layer (src/trunk.zig streamedMoeResult / finishStreamedMoe):
    - the router is evaluated with a blocking `mlx_array_eval(ids_contiguous)`;
    - the host reads the ids and calls `engine.prepareHost(layer, occurrences)`, which fills the misses;
    - the GEMMs run over the cache memory. The doc: "A streamed layer is latency-bound: one GPU->host read of the router ids per layer."
  - A load-time warm-up (warmupTrunk) pre-faults pages and compiles kernels.
- **I/O** (src/expert_io.zig):
  - `fcntl F_NOCACHE` + `F_RDAHEAD 0`, then `pread` on 4 worker threads (`FILL_WORKERS = 4`, src/expert_stream.zig). runFill blocks on `pool.join()`.
  - Spans are sorted and coalesced when exactly adjacent, up to 64 MiB; reads are split across workers by a cost model (`READ_COST_BYTES = 1 << 20`).
  - Aligned spans land directly in the cache memory (anonymous mmap PageSlab, handed to MLX without a copy); others go through a 64 MiB bounce buffer.
  - One expert is 9 spans (trellis plus scales for gate, up and down).
  - Logged rate: "bound by the SSD at ~8.5 GB/s". On M1 Max, more than 4 workers or split reads were slower (docs/qwen4-perf.md).
- **Missing experts in prefill: (a), every distinct routed expert is read, exactly.**
  - The lossy expert pick is decode-only and off by default; `--drop-expert` is DeepSeek-only and off.
  - No cross-layer read-ahead for Qwen in prefill: the decode look-ahead needs `seq_len == 1`. The GPU likely waits on each layer's fill (inferred).
  - Inside a layer, rate-grouped packs overlap the reads with compute: fillStaged reads the scales first, then each rate group's trellis
    while the previous group's GEMMs run. Their measurement: +6-11% prefill at 32-64k on an STQ 1.75 bpw pack.
  - Prefill read-aheads they measured and dropped (docs/dsv41-arch.md, DeepSeek):
    - next layer's router on this layer's rows: "guessed ~28% of what it read and ~24% of the picks", 294 vs 296 tok/s;
    - a pool read-ahead: "cut a request's misses 9% and read 22% more bytes", no gain.
- **Cache:**
  - LRU per layer and rate group. In prefill a chunk's misses are admitted by how often the chunk uses them; the overflow goes to a temporary
    "union workspace", so a prompt does not flush the decode cache (GroupCache.resolve / prepareHostWide).
  - About 80% of slots are warmed at load from `expert_usage.json`, if the pack ships one (rankedWarm).
  - The per-layer split is greedy hits per byte from calibration (expert_cache_profile.json).
  - Kernels read the slots in place.
- **State kept between runs:** an SSD prefix cache (KV plus GDN checkpoints), on by default, that survives restart. Qwen restored 157,656
  tokens in 707 ms (docs/engine-prefix-cache.md). The expert cache is not saved.
- **Timings:**
  - M5 Max, Qwen 4 bpw streamed at 48 GiB: prefill 831 / 866 tok/s at 4k / 32k.
  - M1 Max: "~0.63 ms per missed expert".
  - No short-prompt streamed TTFT is published for Qwen.
- **Verdict:** relevant, but its streamed prefill has no TTFT mechanism beyond Strata's: blocking per-layer fills, 4 threads, no
  cross-layer prefetch. Worth noting:
  - its admission of prefill misses by chunk use count, with overflow to a scratch workspace;
  - the rate-group overlap inside a layer;
  - its negative result on next-layer router prediction in prefill (DeepSeek, cold 10k prompt).

## 6. The earlier notes, checked

- "whole GGUF experts in the arena": **partly right.** Whole experts are stored (gate|up|down of one (layer, expert) per slot) as
  byte-exact GGUF bytes (stream.h:15-16; stream.cpp:296-306). It is not the whole GGUF: 17,723 of 24,576 experts at 33 GB.
- "misses read into slots": **right** (stream.cpp:258-272, 337-348, 277-307).
- "kernels run on slots in place": **right** (moe.metal:45-57; moe_mm.metal:272-284, 896-899).
- "next-layer read-ahead of all non-arena experts in the prompt path": **right only for the GEMM chunk path** (n-1 > 64;
  stream.cpp:692-711). Short chat prompts (<= 65 tokens) never take it: they use 2-row windows with a top-6-per-token predicted
  read-ahead (stream.cpp:713-715).

## 7. Where Strata differs from our engine (contrast only, not an evaluation)

| Aspect | Strata | Ours (as briefed) |
|---|---|---|
| Expert record | Whole expert, GGUF bytes, IQ3_XXS mix ~1.75 MB avg (stream.cpp:155) | 4-bit, ~2.8 MB (3-bit MSB + 1-bit LSB), bases rebuilt on the GPU |
| Read call | 3 x pread per expert (slices), F_NOCACHE, 6 threads, QD <= 6 | preadv |
| Resident set at prompt time | 33 GB mlocked, 72% of pairs, static ranking then LRU, filled during warm-up | (not evaluated) |
| Short prompt (<= 64) | 2-row decode windows, predicted top-6 read-ahead, GPU stalls per layer | batched prefill with router-predicted read-ahead |
| Long prompt | 1024-row chunk; all non-resident l+1 experts streamed, unneeded ones dropped at the next cut | (not evaluated) |
| Expert skipping in prefill | none on Metal | (not evaluated) |
| Extras | SSD keep-awake; 1.39 GB prompt buffers borrowed from the arena | - |

Candidate methods to port, each with credit to Strata (MIT, github.com/Niko1221/Strata). All gains are inferred and untested here:

1. Stream all non-resident experts of l+1, then drop unstarted reads that did not get routed at l+1's cut. Strata uses this
   only for chunks; for short prompts its own data shows the chunk path is about 2x faster than its window path.
2. Keep prompt-streamed experts at the LRU tail so prefill does not evict decode-hot experts.
3. SSD keep-awake.
4. Fill the cache once from a frequency ranking, wire it, and keep it across requests.

## 8. Gaps

- Bytes per prompt, hit rates and the split between GPU time and wait time were never logged:
  - the stream report prints only at exit, and `_Exit(0)` skips it;
  - DONE's `file_mb` and `experts_streamed` count the host tier and are always 0 on Metal;
  - `--stats` and `STRATA_METAL_PROFILE` were not used.
- The model files were deleted (11-09 cleanup), so per-layer slice sizes could not be read.
