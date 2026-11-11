# Prior art: the GPU -> CPU -> GPU "gate" for streamed MoE experts (Apple Silicon, Metal via MLX)

Compiled 2026-11-09 from web searches; papers were read directly where the PDF or HTML was readable.
Numbers are as each source reports them. "(secondary)" marks a figure I saw only in a summary, forum post or slide excerpt, not in the primary text.

## Our design, for reference

- **Round trip:** the GPU writes a routing row into a shared buffer, and MLX commits the command buffer (CB) right after that kernel. A CPU thread reads the row, loads the missing experts into GPU-visible slots, and writes a release value into a fresh, never-before-read 32 KB region per (pass, layer). A later kernel in the same pass spins on that region line by line.
- **Residency table:** published one pass ahead, so the GPU skips the wait when every chosen expert is already resident.
- **Measured costs:** ~45 us for the pure round trip and ~69 us per gated layer. An MTLSharedEvent handshake or an mx.eval sync costs ~150-230 us.
- **Visibility observations:**
  - Plain in-kernel polling saw the other side's writes only at CB end or after random eviction (0.5-1000 ms).
  - MLX bound arrays under 4 KB as constant data copied at encode time.

---

## A. Apple Silicon / Metal: closest prior art

### A1. MLX "fast fence": a GPU kernel spins on a value the CPU writes (Apple, merged 2025-01-18)

- **Links:**
  - PR #1773 "Faster synchronization Fence primitive": https://github.com/ml-explore/mlx/pull/1773
  - Kernel source: https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal
  - Host side: https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/fence.cpp
- **What it does:**
  - A 4-byte counter in CPU-accessible memory.
  - `fence_wait` is a single-thread kernel: `while(1){ atomic_thread_fence(mem_device, seq_cst, thread_scope_system); if (timestamp[0] >= value) break; }` on a `volatile coherent(system) device uint*`.
  - The system scope is not public MSL. The file uses `#pragma METAL internals : enable` and defines `thread_scope_system` as private scope value 3.
  - For CPU streams, the CPU writes the counter, or spins on it with an empty loop.
  - An `input_coherent` kernel re-reads and re-writes each input word through `coherent(system)` so the CPU can read GPU outputs.
  - The path is gated by `MLX_METAL_FAST_SYNCH`, needs Metal 3.2 / macOS 15, and is off by default.
- **Numbers (PR #1773, M3 Max, 2 processes):**
  - All-reduce: 0.150 ms with shared events, 0.0165 ms with the fast fence. All-gather: 0.136 ms vs 0.0159 ms.
  - The author's summary: "15-20 microseconds depending on the machine, compared to 150-200 with shared events."
- **Status:**
  - MLX docs (https://ml-explore.github.io/mlx/build/html/usage/distributed.html) say it is "not reliable that can lead to deadlock and leave the GPU wedged … off by default and best left unset."
  - Issue #3142 (https://github.com/ml-explore/mlx/issues/3142, Feb 2026): the GPU gets stuck in `fence_wait` forever. A maintainer wrote on 2026-04-28 that "there is no support for any kind of atomic memory coherence between CPU/GPU … this flag would not be guaranteed to work."
  - Issue #3830 (https://github.com/ml-explore/mlx/issues/3830, opened 2026-07-10): deadlock over the TCP ring, and the GPU stays occupied until reboot. The maintainer: "not reliable and we don't see a way to fix it."
  - WWDC26 session 233 (MLX distributed) still tells users to set it: "enables faster GPU-to-CPU synchronization" (https://nonstrict.eu/wwdcindex/wwdc2026/233/).
- **Difference from ours:**
  - Same core mechanism: a GPU kernel spins on a value the CPU writes in shared memory. MLX re-polls one word under a private system-scope fence and `coherent(system)`. We poll fresh, never-read lines in public MSL.
  - MLX hands off a counter at stream boundaries, for collectives. We hand off a data-dependent routing row, the CPU loads experts, and the wait happens per layer inside one pass with a residency skip.
  - MLX's 15-20 us is lower than our 45 us. The measurements differ (theirs is an all-reduce sync between 2 processes, ours includes the CB commit and the row read), so they are not directly comparable.
- **Date:** the fast fence predates our 2026-10-17 gate by ~21 months. #3830 is a later bug report, not the origin.

### A2. MLX CUDA backend: the same pattern on NVIDIA, judged problematic (2025-2026)

- **Link:** issue #2391 (https://github.com/ml-explore/mlx/issues/2391).
- **What it does:** CPU->GPU sync with a `cuda::atomic` in unified memory, which a CPU task updates and a GPU kernel waits on.
- **Status:** the author calls its problems "fatal": a kernel occupies the GPU just to wait, and `cudaFree` deadlocks behind it. Closed in March 2026 with no final design.

### A3. Evaluating Apple Silicon for Data Processing (TUM, DaMoN '26, May 2026)

- **Link:** https://db.in.tum.de/~beischl/papers/Evaluating_Apple_Silicon_for_Data_Processing.pdf (M4 Pro and M3 Ultra).
- **Findings:**
  - "Atomic operations on Apple Silicon are only guaranteed at device level … not across devices." In a CPU+GPU `atomic_fetch_add` stress test, 0.3-0.5 % of buckets were incremented twice.
  - "GPU-wide synchronization can only be done between kernels on the CPU."
  - `coherent(device)`: "We have not observed a noticeable slowdown."
- **Numbers:**
  - Kernel start: the encode phase takes 28-79 us. The dispatch phase takes 14-17 us on M4 Pro and 65-75 us on M3 Ultra.
  - Newly allocated shared buffers trigger "wiring" (10-60 GB/s) plus a coherence pass at kernel launch (70-120 GB/s, M4 Pro). Managing one big region yourself cuts the coherence overhead to < 0.06 ms per run.
- **Relation to ours:** this independently confirms no CPU<->GPU atomic coherence and visibility at kernel/CB boundaries. It is consistent with our polling observation, and explains why our gate avoids cross-device read-modify-write atomics and uses one-way release values.

### A4. Apple documentation, feature tables and forums

- **MSL 3.2 (2024):**
  - Added `coherent(device)` and `atomic_thread_fence(…, thread_scope)`.
  - An Apple engineer: "use a `thread_scope_device` fence along with device atomics to synchronize coherent(device) memory across thread groups" (https://developer.apple.com/forums/thread/756663).
  - There is no public system (CPU) scope, which is why MLX uses the private one.
- **Metal Feature Set Tables (2026-10-23, https://developer.apple.com/metal/capabilities/):**
  - "Acquire and release memory ordering atomics": Apple7+ (M1+). A forum reply places the new orders in MSL 4.1 §6.16.1 (https://developer.apple.com/forums/thread/841598).
  - "Memory coherence for textures and buffers in shaders": Apple6+.
  - "Residency sets": Apple6+.
  - "Compute shader forward progress guarantees": Apple11 only, which is M6. The M5 Pro is Apple10, so it has no documented forward-progress guarantee.
  - I could not open the 383-page MSL 4.1 PDF, so I did not check whether 4.1 adds a CPU-visible scope.
- **Shared events:**
  - Apple says shareable events cost more than MTLEvent (https://developer.apple.com/documentation/metal/about-synchronization-events.md).
  - An Apple engineer (2022, https://developer.apple.com/forums/thread/701231): a CPU write made inside a shared-event handler is only "racily visible" to a kernel in an already-committed CB, unless the kernel waits for a value that the handler signals.
- **`setBytes`:** Apple's docs (https://developer.apple.com/documentation/metal/mtlrendercommandencoder/setvertexbytes(_:length:index:).md) describe it for data of 4 KB or less, copied at encode time. In MLX main `device.cpp`, `set_input_array` binds arrays with `setBuffer`; `set_bytes`/`set_vector_bytes` is used for shapes, strides and ndim in custom kernels. I found no MLX documentation of the "<4 KB array becomes constant" behaviour we saw. That path is worth re-checking (my view of the file was truncated).
- **Metal 4 (WWDC25, https://developer.apple.com/videos/play/wwdc2025/205/):**
  - Adds queue-level `MTL4CommandQueue waitForEvent/signalEvent`, residency sets and placement sparse resources.
  - I found no primitive that lets a running kernel wait for a CPU value. "Residency sets" mean OS residency of allocations, which is not the same concept as our per-expert table.
- **Fast resource loading (Metal 3, WWDC22, https://developer.apple.com/videos/play/wwdc2022/10104/):**
  - `MTLIOCommandQueue` gives CPU-issued, batched file-to-buffer/texture loads, synchronized with Metal events. It is Apple's analogue of DirectStorage.
  - There is no GPU-initiated I/O and no mid-kernel wait.
- **Asahi Linux AGX docs (https://asahilinux.org/docs/hw/soc/agx/):**
  - The GPU is run by a firmware coprocessor (ASC, RTKit). Submissions and completions travel as firmware messages and event indices.
  - This plausibly explains the cost of CB-level sync. That explanation is an inference; nobody measured it.

### A5. Anukari: real-time audio on Metal (2024-2025)

- **Links:**
  - "Huge macOS performance improvements": https://anukari.com/blog/devlog/huge-macos-performance-improvements
  - Apple forum thread: https://developer.apple.com/forums/thread/768160
- **Latency:** with MTLSharedEvent driving both kernel launch and wait (an Apple engineer's suggestion) plus double-buffered encoding, "scheduling and waiting overhead is now under 50 µs" on the author's machines. Encoding cost ~50 us per CB, overlapped.
- **Power state:**
  - macOS raises GPU clocks only when some CB looks saturated. Keeping one warp spinning "works perfectly, albeit wastefully."
  - Apple DTS called that strategy "wasteful and not future-proof" (Dec 2024) and had it "under investigation" (May 2025).
  - Anukari later moved to the CPU.
- **Relevance:** a gate's spinning GPU threads may also raise GPU clocks, which can colour A/B results. Shared-event latency of tens of us is achievable when the encode work is pipelined.

### A6. Apple Silicon expert-streaming engines (none documents a GPU-side wait on a CPU flag)

- **flash-moe** (danveloper, Mar 2026, https://github.com/danveloper/flash-moe):
  - Runs Qwen3.5-397B on an M3 Max with 48 GB at 4.36 tok/s.
  - Per-layer serial chain: CMD2 (routing) → CPU softmax and top-K → parallel `pread` of K=4 experts → CMD3, which is encoded and submitted without waiting.
  - Per layer: 4.28 ms total, of which I/O is 2.41 ms.
  - Rejected experiments:
    - "Spin-poll GPU wait −23 % (CPU thermal competes with GPU)." The mechanism is not detailed.
    - "F_RDADVISE prefetch net 0 % (SSD DMA slows GPU −73 %)."
    - "mmap −5x."
    - "Temporal expert prediction −18 %."
  - Its claim: "SSD DMA and GPU compute … cannot be profitably overlapped."
- **Edge0** (MLX, 2026-10-03, https://themenonlab.blog/blog/edge0-streaming-moe-ssd-expert-offload):
  - A trained "Prerouter" predicts experts one layer ahead and for later tokens. Predictions commit at generation-step boundaries.
  - 14.9-17.7 tok/s for a 35B model on an M4 Pro with 24 GB.
  - Sync internals are not described.
- **Strata:**
  - Experimental Metal engine in v0.1.44, 2026-10-13 (https://smartscope.blog/en/blog/strata-mac-metal-support-setup-guide-model-selection-2026/, https://github.com/Niko1221/Strata). It streams experts from the SSD.
  - Its docs describe next-layer prefetch over PCIe and CPU compute of uncached experts. They say nothing about Metal sync.
- **LLM in a flash** (Apple, 2023/24, https://arxiv.org/abs/2312.11514): FFN weights stream from flash using a sparsity predictor, windowing and row-column bundling. It reports up to 7x on Metal vs naive loading. The handoff mechanism is not described.

---

## B. Discrete GPUs: GPU waits on host-written values

### B1. CUDA stream memory operations

- **Link:** `cuStreamWaitValue32` / `cuStreamWriteValue32` (https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html). HIP has the same calls (`hipStreamWaitValue32`).
- **What it does:**
  - "Work ordered after the operation will block until the given condition on the memory is satisfied." The wait happens at the stream/front-end level, between kernels.
  - FLUSH needs "dedicated platform hardware features." Managed memory is not allowed. The docs warn: "Improper use … may deadlock."
  - The V1 API was disabled by default; on Linux it needed `NVreg_EnableStreamMemOPs=1` (CUDA 11.6 docs).
- **Numbers:** I found no published latency.
- **Difference from ours:** this is a driver-supported queue-level wait. Its Metal analogue is `encodeWaitForEvent` on a shared event, which is the path we measured at 150-230 us. Our wait runs inside an already-running pass.

### B2. GPUDirect Async (Agostini, Rossetti, Potluri, JPDC 2018; CCGrid 2017)

- **Link:** https://github.com/gpudirect/libgdsync
- **What it does:** the GPU triggers and polls InfiniBand operations with no CPU on the critical path. It has a stream model (stream memory ops) and a kernel-initiated model. It evolved into NVSHMEM IBGDA.
- **Difference from ours:** it removes the CPU, which needs NIC and driver support. Apple exposes no NVMe or NIC queues to the GPU.

### B3. Persistent kernels polling CPU-written flags

- **NVIDIA DPDK/gpudev blog** (2022-04-28, https://developer.nvidia.com/blog/optimizing-inline-packet-processing-using-dpdk-and-gpudev-with-gpus/):
  - A persistent kernel is "busy waiting for a notification from the CPU."
  - The flag goes in GPU memory, written by the CPU through GDRCopy. Polling CPU-pinned memory "would consume more PCIe bandwidth and increase the overall latency."
  - NVIDIA calls the pattern "risky and not supported by the CUDA programming model."
- **GPU-to-CPU Callbacks** (Stuart, Cox, Owens, Euro-Par 2010 workshops / LNCS 2011, https://escholarship.org/uc/item/58j4v6f8):
  - The GPU writes the request into zero-copy memory and issues a `threadfence()`. A CPU thread pool polls. The GPU thread "spins on a region of 0-copy memory until the CPU signals."
  - On a GTX 280, the memory-manager test averaged under 4 ms per callback (minimum under 2 ms). In the TCP test the GPU took 2.5 ms vs 1 ms on the CPU.
  - This is the earliest general form of the pattern I found.
- **Pagoda** (PPoPP 2017, https://engineering.purdue.edu/tgrogers/publication/yeh-ppopp-2017): a persistent "MasterKernel" schedules narrow tasks submitted by the CPU. It reports 1.51x over HyperQ.
- **Pitfall on NVIDIA:** `volatile` alone does not refresh caches. PTX offers the `.cv` ("cache-volatile") load for data that changes outside the program (https://nvidia.github.io/numba-cuda/reference/cache_hints.html; PTX ISA "Cache Operators"). That is a hardware/ISA answer to the stale-line problem we solved with fresh lines; Metal exposes no public equivalent beyond `coherent(...)`.

### B4. GPU -> host RPC built into toolchains and runtimes

- **LLVM libc GPU RPC** (https://libc.llvm.org/gpu/rpc.html): the GPU is the client and the host the server. Ports use a shared buffer plus inbox/outbox mailboxes in pinned unified memory, and the host server polls.
- **ROCm hostcall** (https://sources.debian.org/src/rocm-hipamd/6.4.3-5/rocclr/device/devhostcall.hpp): a work-item "submits a request as a packet … then blocks until it receives a response." A host "listener" thread services it. Device printf uses it.
- **NCCL / GIN proxy backend** (arXiv 2511.15076, https://arxiv.org/abs/2511.15076): GPU kernels push requests into a lock-free GPU->CPU queue, and a CPU proxy thread polls it and issues the network operations.
- **NCCL LL protocol** (https://arxiv.org/abs/2507.04786): data and flag go in one 8-byte store, and the receiver polls the data line itself with per-step flag values. One lecture puts it at ~1 us per hop (secondary).
- **Difference from ours:** these are discrete GPUs, or APUs with driver-supported coherent memory. The pattern itself (GPU posts a request, a host thread services it, the GPU waits on a flag) is standard.

### B5. GPU file systems and system calls

- **GPUfs** (ASPLOS 2013, https://infoscience.epfl.ch/record/212688): POSIX-like file API for GPU code, with a buffer cache extended into GPU memory. Per Genesys, it uses a CPU polling thread to service requests.
- **ActivePointers** (ISCA 2016, https://acsl.ece.technion.ac.il/?p=1135): software address translation and page faults over a GPU page cache. BaM puts it at ~823K IOPS.
- **GAIA** (ATC 2019, https://www.usenix.net/system/files/atc19-brokhman.pdf): lets the standard `mmap` call map files into the GPU address space.
- **Genesys, "GPU System Calls"** (ISCA 2018, arXiv 1705.06965, https://arxiv.org/abs/1705.06965):
  - Platform: AMD FX-9800P (Carrizo) APU, so unified memory with an integrated GCN3 GPU, on ROCm.
  - The syscall area uses 64-byte slots per work-item (1.25 MB total). The GPU signals the CPU with an `s_sendmsg` interrupt, and the work-item "waits for the availability of the result by polling on the state of the slot."
  - The paper studies invocation granularity (work-item, work-group, kernel) and blocking vs non-blocking calls.
  - The arXiv version reports no round-trip time in microseconds, only throughput (for example, the GPU reached 170 MB/s vs the CPU's 30 MB/s in wordcount).
- **Difference from ours:** the closest in spirit is the GPU waiting mid-kernel on a unified-memory APU for a CPU service. Genesys relies on hardware coherence (HSA) and interrupts; we have neither guaranteed on Apple.

### B6. GPU page faults: the hardware-assisted version of the same round trip

- **NVIDIA UVM far fault:**
  - "20µs to 50µs" (Kim et al., ASPLOS'20, https://ramyadhadidi.github.io/files/kim-asplos20.pdf, citing prior work).
  - 30-45 us (Allen & Ge, IPDPS'21, https://tallendev.github.io/assets/papers/ipdps21.pdf).
  - Up to ~50 us, involving several PCIe round trips plus host CPU work (Zheng et al., HPCA'16, https://www.cs.utexas.edu/~skeckler/pubs/HPCA_2016_Paged_Memory.pdf; secondary).
  - BaM measured the UVM fault handler saturating at ~500K IOPS.
- **AMD MI300A** (unified HBM; Wahlgren et al., arXiv 2508.12743, IISWC 2025, https://arxiv.org/abs/2508.12743): GPU minor faults average 16 us (20 us tail); major faults average 18 us (22 us tail).
- **GPUVM** (arXiv 2411.05309, https://arxiv.org/abs/2411.05309): the GPU itself drives paging through an RDMA NIC.
- **Relevance:** our ~45 us software round trip sits in the same range as hardware/driver GPU-fault servicing on NVIDIA, and about 2-3x MI300A's fault latency.

### B7. KTransformers (SOSP '25)

- **Link:** https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/
- **What it does:**
  - "A naive design must synchronize twice per MoE layer … These barriers break CUDA Graphs." KTransformers instead "encapsulates both submit and sync in cudaLaunchHostFunc", so "the entire decode path for one token now fits in a single CUDA Graph." That gives up to 1.23x.
  - Expert Deferral (3 of 8 routed experts deferred one layer) gives +33 % decode throughput.
- **Difference from ours:** the CPU computes the experts rather than loading them, and the wait is a stream-ordered host callback, so the stream stalls while it runs on a CUDA-internal thread. The goal is the same as ours: no host sync per layer.
- **Callback cost:** one forum measurement (anecdotal) found a host callback adds ~50 us vs ~1 us for a light kernel (https://forums.developer.nvidia.com/t/why-is-the-performance-of-cudalaunchhostfunc-low-and-what-are-the-optimization-strategies-or-alternative-solutions/306553).

---

## C. Coherence and round-trip numbers on unified / coherent memory

- **GH200** (Fusco et al., arXiv 2408.11556, 2024, https://arxiv.org/abs/2408.11556, Fig. 13):
  - Full ping-pong with a `cuda::std::atomic` CAS on a 1-byte flag. Grace<->Hopper is 750-752 ns with the flag in Grace DDR and 833-834 ns with it in Hopper HBM. Hopper-local is 315.5 ns.
  - NVLink-C2C is hardware coherent with "native atomics from both the CPU and GPU" (NVIDIA datasheet).
  - Our ~45 us is ~55x a hardware-coherent C2C round trip.
- **MI300A** (same paper as B6): HBM latency is 333-350 ns from the GPU and 236-241 ns from the CPU. "The Infinity Cache does not participate in coherency."
- **Apple M2 Pro GPU** (Chips and Cheese, 2023, https://chipsandcheese.com/2023/10/31/a-brief-look-at-apples-m2-pro-igpu/): global atomics take 58.57 ns, the SLC 234 ns, DRAM >342 ns. There are no CPU<->GPU figures.
- **Apple CPU<->GPU round trips actually published:**
  - MLX fast fence: 15-20 us. MTLSharedEvent: 150-200 us (A1).
  - Anukari: under 50 us of scheduling and waiting overhead with shared events plus pipelining (A5).
  - TUM: 14-17 us dispatch on M4 Pro, 65-75 us on M3 Ultra (A3).
  - Our 150-230 us shared-event figure agrees with MLX's 150-200 us.
- **Theory:**
  - Heterogeneous-race-free (HRF) memory models with scoped synchronization, including system scope (Hower et al., ASPLOS 2014, https://research.cs.wisc.edu/multifacet/papers/asplos14_hrf.pdf).
  - Chai benchmarks of CPU-GPU collaboration using system-wide atomics on integrated parts (ISPASS 2017, https://chai-benchmarks.github.io/assets/ispass17.pdf).

---

## D. Vulkan / D3D12 / WDDM

- **Vulkan timeline semaphores:**
  - The host can signal with `vkSignalSemaphore` to "kick" the GPU at queue level (https://docs.vulkan.org/samples/latest/samples/extensions/timeline_semaphore/README.html). I found no published latency numbers.
  - There is no forward-progress guarantee between workgroups (Vello shader comment, Dawn discussion, OOPSLA'21 "Specifying and Testing GPU Workgroup Progress Models": https://multicore.doc.ic.ac.uk/publications/gpu-oopsla-21.html).
- **VK_AMD_device_coherent_memory** (https://docs.vulkan.org/refpages/latest/refpages/source/VK_AMD_device_coherent_memory.html): device-coherent and device-uncached memory types, "lower performance … useful for debugging."
- **VK_AMD_buffer_marker** (https://github.khronos.org/Vulkan-Site/refpages/latest/refpages/source/VK_AMD_buffer_marker.html): writes a marker value when a command reaches a pipeline stage, intended for crash debugging. It is GPU->memory progress reporting and has no GPU-side wait.
- **llama.cpp Vulkan PR #12630** (2025-03/04, https://github.com/ggml-org/llama.cpp/pull/12630):
  - Fixes "a bubble waking up from waitForFences." The CPU waits on an "almost ready" fence at ~80 % of the graph, then spins on the final fence.
  - Up to 5 % on NVIDIA/Windows, and a "small positive effect on all vendors." The CPU side spins to cut wake-up latency.
- **WDDM monitored fences** (https://learn.microsoft.com/windows-hardware/drivers/display/native-gpu-fence-objects):
  - Before WDDM 3.2: "What wasn't supported was a native on-the-GPU wait for a monitored fence value. Instead, the OS held GPU work that depends on the waited value on the CPU."
  - WDDM 3.2 (Win 11 24H2) adds a GPU wait on native fences "without requiring a CPU roundtrip", at the engine level.
- **D3D12 Fence Barriers** (2026-02-26 preview, AgilitySDK 1.719, https://devblogs.microsoft.com/directx/fence-barriers-fine-grained-gpu-synchronization-in-direct3d-12/):
  - Tier-1 waits mid-command-list, but only on command-list-scoped fences signaled earlier in the same execution.
  - Tier-2 waits on CPU-signaled or cross-queue fences mid-command-list are planned. They need native fences plus user-mode submission.
  - So a mid-command-buffer wait on a CPU value, our gate's job, is on Microsoft's roadmap but had not shipped as of Feb 2026.
- **D3D12 Work Graphs** (2024-03-11, https://devblogs.microsoft.com/directx/d3d12-work-graphs/): the GPU generates its own work, which avoids CPU round trips and "previous-frame readback."

---

## E. GPU-initiated I/O and storage

- **BaM** (ASPLOS 2023, https://arxiv.org/abs/2203.04910):
  - GPU threads submit NVMe commands themselves, through queues in GPU memory plus a GPU-side software cache.
  - 45.8M random 512 B read IOPS with 10 Optane SSDs (22.9 GB/s).
  - Needs a custom Linux driver, GPUDirect RDMA, and GPUDirect Async to map the SSD doorbells into the CUDA address space.
  - Baselines: GDS reaches 23.6 % of PCIe at 4 KB; UVM saturates at ~500K IOPS.
  - Not possible on Apple, where the GPU has no NVMe queue access.
- **GPUDirect Storage / cuFile** (https://docs.nvidia.com/gpudirect-storage/overview-guide/):
  - A DMA path between storage and GPU memory, but the CPU initiates and controls it. For small I/O, the POSIX path can have lower latency.
  - NVIDIA open-sourced cuFile and announced "SCADA" for GPU-side control of small random reads (Aug 2026, trade press, https://www.storagereview.com/news/nvidia-scada-puts-storage-control-on-the-gpu-as-cufile-goes-open-source). Details unverified.
- **DirectStorage (Windows):** batched requests with GPU GDeflate decompression. Press coverage of the 3DMark DirectStorage test reports ~5.85 GB/s without GDeflate and up to ~13 GB/s with it (secondary; https://www.tomshardware.com/video-games/pc-gaming/3dmark-gets-new-directstorage-benchmark-gdeflate-measures-transfer-rates-from-nvme-ssd-to-gpu-vram). I found no per-request latency figures.
- **PS5 I/O complex** (Mark Cerny, "The Road to PS5", March 2020). I saw this only secondhand, quoted in forum copies of Digital Foundry's coverage; I did not reach the primary source. Forum thread: https://forum.beyond3d.com/goto/post?id=2165416 (the exact post is not verified).
  - "Coherency engines" tell the GPU which address ranges an SSD read overwrote, and "cache scrubbers" in the GPU caches do pinpoint evictions of just those ranges.
  - This is a hardware solution to the stale-GPU-line problem with streamed data in unified memory. It is the closest hardware analogue to our fresh-line workaround.

---

## F. MoE offloading engines: how they hand off between GPU and CPU

| System | Handoff / latency hiding | Host sync per layer? |
|---|---|---|
| Fiddler (ICLR 2025, https://arxiv.org/abs/2402.07033) | Non-resident experts run on the CPU; only activations cross PCIe | Yes, activations go to the CPU each layer |
| MoE-Infinity (2024, https://arxiv.org/abs/2401.14361) | Request-level expert activation matrix; prefetch priority by probability and layer distance | Host-driven, not described as GPU-side |
| Pre-gated MoE (ISCA 2024, https://www.microsoft.com/en-us/research/uploads/prodnew/2024/05/isca24_pregated_moe_camera_ready.pdf) | A trained pre-gate picks the next block's experts; migration overlaps the current block (AdapMoE reports accuracy loss without fine-tuning) | Prefetch hides it; host still issues copies |
| ProMoE (2024/25, https://arxiv.org/abs/2410.22134) | Learned sliding-window predictor; chunked prefetch, early preemption, reordered inference; in llama.cpp and transformers | Host-driven |
| HOBBIT (2024, https://arxiv.org/abs/2411.01433) | Low-precision copy on a miss; layer-level prefetch; on llama.cpp; up to 9.93x | Host-driven |
| AdapMoE (ICCAD 2024, https://arxiv.org/abs/2408.10284) | Adaptive gating, next-layer gate prefetch, DP cache sizing; 1.35x | Host-driven |
| MoE-Lightning (2024, https://arxiv.org/abs/2411.11217) | CGOPipe CPU-GPU-I/O pipeline with a hierarchical roofline; batch throughput; up to 10.3x on T4 | Pipelined, batch-oriented |
| eMoE (2025, https://arxiv.org/abs/2503.06823) | Async loads; "the corresponding MoE layer synchronizes with the event" | Yes (CUDA event) |
| KTransformers (SOSP '25) | Submit and sync inside `cudaLaunchHostFunc`; whole token in one CUDA Graph; Expert Deferral | No host-thread sync; stream-ordered callback |
| llama.cpp `--cpu-moe` / `-ot` | Graph splits between backends; one ik_llama profile put 1.88 s of 11.53 s in sync and copies, with thread start-up 25x per token (https://github.com/ikawrakow/ik_llama.cpp/pull/232) | Yes, at every split |
| FlashDMoE (2025, https://arxiv.org/abs/2506.04667) | Gate, dispatch, experts and combine fused in one persistent kernel with device-initiated communication (multi-GPU, not offload) | No host involvement |
| flash-moe / Edge0 / Strata (Apple, 2026) | See A6 | flash-moe: yes (serial CB chain) |

None of the engines I checked describes a GPU-side wait on a CPU-written flag for expert arrival. A targeted search for "device-side flag, no host sync per layer" in MoE offloading found nothing closer than FlashDMoE and eMoE.

---

## G. Games and real-time graphics: how GPU data reaches the CPU within a frame

- **Usual answer: it doesn't.** Readbacks are delayed by 1-3 frames.
  - Unity AsyncGPUReadback copies "without any stall … but adds a few frames of latency" (https://docs.unity3d.com/2021.1/Documentation/ScriptReference/Rendering.AsyncGPUReadback.html).
  - D3D12 guidance is to read queries a few frames late (GDC 2016, https://gpuopen.com/download/GDC_2016_D3D12_Right_On_Queue_final.pdf).
  - The Intel Galactic Civilizations 3 case removed Map stalls (https://www.intel.com/content/www/us/en/developer/articles/case-study/removing-cpu-gpu-sync-stalls-in-galactic-civilizations-3.html).
- **Virtual texturing feedback** (id Tech 5, van Waveren):
  - Feedback buffer → CPU analysis (~0.5 ms for an 80x60 buffer) → load and transcode → upload. Staging copies "force an extra frame of latency" (GTC 2010 slides, secondary excerpt: https://www.quaddicted.com/webarchive/mrelusive.com_20170131/publications/presentations/2010_gtc/GTC_2010_Virtual_Textures.pdf).
  - Coarser mips are shown until data arrives.
- **Sampler Feedback Streaming** (D3D12 Ultimate; Intel sample https://github.com/GameTechDev/SamplerFeedbackStreaming):
  - Feedback is resolved, read back and applied "for a future frame." Evictions are delayed "a frame or two."
  - A per-resource MinMip "residency map" (~4 KB for a 16k² BC7 texture) is read by the pixel shader so that "sampling … is clamped to the minimum mip resident."
  - About 200 MB of heap serves more than 350 GB of textures.
  - The MinMip map is the closest graphics analogue to our residency table: a shader-side check of what is resident. The difference is that it falls back instead of waiting.
- **Nanite** (Karis et al., SIGGRAPH 2021, https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf):
  - GPU-driven rendering. The CPU parses the GPU-written feedback and streaming requests at frame end (https://trickybitsblog.github.io/2024/04/20/nanite.html).
  - Coarser clusters are drawn until pages stream in.
- **Late latching** (Meta, https://developers.meta.com/horizon/blog/optimizing-vr-graphics-with-late-latching/):
  - The CPU keeps writing the newest head pose into a host-visible, coherent buffer after submission. The GPU latches it just before use, with no waiting.
  - This is the closest shipped graphics analogue of our release value, but one-way.
- **D3D12 Fence Barriers Tier-2 (planned):** would give games a native mid-command-list wait on a CPU-signaled value. That is a native gate, and it shows demand for one.
- **PS5 cache scrubbers:** see E.

---

## H. Assessment

### Known practice (well documented)

- **The pattern itself.** The GPU posts a request to host-visible memory, a CPU thread services it, and GPU threads spin on a flag:
  - GPU-to-CPU Callbacks (2010), GPUfs (2013), Genesys on an APU (2018), LLVM libc RPC, ROCm hostcall, NCCL/GIN proxy.
  - On Apple Silicon specifically: MLX's fast fence, merged 2025-01-18. **We should not claim the method as new.**
- **Queue-level GPU waits on a memory value:** `cuStreamWaitValue32`, HSA/AMD barrier-value packets, MTLSharedEvent `encodeWaitForEvent`, Vulkan timeline semaphores, WDDM 3.2 native fences.
- **The cost ladder** (a spinning fence is ~10x faster than shared events on Apple): MLX reported 15-20 us vs 150-200 us. Our 45 us vs 150-230 us is an independent confirmation on an M5 Pro, not a new finding.
- **Checking a residency/cache table on the GPU and waiting or falling back only on a miss:** BaM, GPUfs and ActivePointers GPU caches; virtual-texture page tables; SFS MinMip maps.
- **Predicting or prefetching a layer or pass ahead:** Pre-gated MoE, AdapMoE, ProMoE, HOBBIT, MoE-Infinity, Edge0.
- **Removing the per-layer host sync in an MoE engine:** KTransformers does it with `cudaLaunchHostFunc` inside one CUDA Graph. Same goal as ours, different mechanism (a stream-ordered callback, with experts computed on the CPU).

### Close analogues (different mechanism or platform)

- **Our visibility trick (fresh, never-before-read lines per (pass, layer)).** The analogues all use a different mechanism:
  - MLX: `coherent(system)` plus a private system-scope fence.
  - NVIDIA: PTX `ld.cv` cache-volatile loads.
  - NCCL LL: data+flag with per-step flag values.
  - PS5: hardware cache scrubbers.
- **A gate inside one pass, where the producer kernel ends a committed CB and the consumer kernel is already queued:** structurally the same as MLX `fence_update`/`fence_wait` across command buffers.
- **A residency table published one pass ahead and read by the GPU to skip the wait:** SFS MinMip maps (graphics) and BaM's GPU-side cache (storage).

### Found none (in this search)

- A published description of using never-before-read cache lines as the CPU->GPU visibility mechanism on Metal (or on any GPU without coherent loads).
- An MoE / expert-streaming engine, on any platform, where the GPU waits mid-pass on a CPU-written release and skips the wait using a GPU-read residency table. The engines I checked sync on the host, use host callbacks, or hide latency with prediction.
- A published per-layer CPU<->GPU handoff cost (~45-70 us) for expert streaming on Apple Silicon. flash-moe publishes phase times but no handoff figure.
- Any supported Metal API for a running kernel to observe CPU writes.
  - MLX maintainers say there is none, and TUM measured no cross-device atomic coherence.
  - Our correctness, like MLX's, rests on observed behaviour.
  - New in 2026: MSL 4.1 acquire/release atomics (Apple7+) and compute forward-progress guarantees on Apple11 (M6) only. Whether either makes a supported version possible is open.

### Risks the prior art flags for us

- GPU wedge or permanent spin (MLX #3142, #3830). The mitigations are a bounded spin with an error path, plus a watchdog fallback.
- Deadlock if the CPU thread stalls.
- Spinning warps raise GPU clocks (Anukari), and CPU spinning competes thermally (flash-moe −23 %). Either can bias A/Bs.
- Apple documents no forward-progress guarantee on the M5 (Apple10).

### Suggested calibration (small test, optional)

Run MLX's own `synchronize_bench.py` (PR #1773) with and without `MLX_METAL_FAST_SYNCH=1` on the M5 Pro. That would let us quote our 45 us next to MLX's mechanism measured on the same machine.

---

## I. Game / real-time uses that look plausible for a gate like ours: UNTESTED IDEAS

Budget: 45-70 us is 0.3-0.4 % of a 16.7 ms frame (60 fps) and 0.5-0.8 % at 120 fps. That is affordable a few times per frame if the CPU-side work is small, or if it was already started from a residency table one frame ahead. Disk reads take tens of us or more, so only data already in RAM (decompress, copy, small allocation) fits inside one gate. Each idea also needs a bounded spin with a fallback, because of the GPU-wedge risk.

1. **Same-frame miss service for virtual textures / sampler feedback.** For the few tiles that are already in RAM, the GPU writes the miss list, the CPU decompresses or copies the tiles into resident slots, and a later pass waits instead of sampling a lower mip. (UNTESTED)
2. **Nanite-style cluster page misses on camera cuts:** gate only the handful of critical misses and fall back for the rest. (UNTESTED)
3. **Same-frame answers to GPU queries for gameplay:** picking, raycasts against GPU-only data, visibility for AI or audio occlusion, instead of the usual 1-3 frame readback. (UNTESTED)
4. **GPU physics or particles reporting collision events to CPU audio or gameplay in the same frame.** (UNTESTED)
5. **GPU-driven passes with variable-size outputs** that ask the CPU for an exact allocation mid-frame, instead of worst-case over-allocation (the problem Work Graphs target). (UNTESTED)
6. **"Late latching with a guarantee":** the GPU waits briefly for a fresh pose or input sample when it has not arrived yet, instead of using a stale one. (UNTESTED)
7. **In-game ML with weight banks larger than memory** (neural shading or upscaling with many experts), streamed per frame through the same residency-table + gate scheme. (UNTESTED)
8. **Shader asserts and printf handled by the host immediately,** as in GPU-to-CPU callbacks, for debugging. (UNTESTED)

Open questions for all of these:
- Does a spinning kernel delay concurrent render work on the same GPU (TBDR scheduling)?
- How does the Apple GPU power-state heuristic behave?
- How does the macOS GPU watchdog treat long spins?
