# The GPU-CPU gate: source-backed fact sheet (records only)

Written Mon 2026-11-09 ~17:15 by a read-only research agent for ~/Papers/GPU_CPU_GATE.md (HANDOVER.md:850-860).
Nothing was run on the GPU and no file outside this scratchpad was changed. Every fact below has a source and a tag:

- **[measured]**: a recorded measurement (raw data file, run log or a measured result in a project record).
- **[modelled]**: derived by a model or by arithmetic on measured lines. "Derived here" = my own arithmetic, shown inline.
- **[assumed]**: an explanation, design intent or belief that no recorded test isolates.

Path shorthands: `BM` = ~/BigMoE, `EXT` = ~/BigMoE/ext/bigmoe_ext.cpp, `GATED` = ~/BigMoE/bigmoe/gated.py,
`STREAM` = ~/BigMoE/bigmoe/stream.py, `TIER` = ~/BigMoE/bigmoe/tier.py, `F` = ~/BigMoE/FINDINGS.md,
`WED` = ~/BigMoE/docs/WED_TESTS.md, `HO` = ~/BigMoE/HANDOVER.md, `LED` = ~/BigMoE/LEDGER.md, `TRIED` = ~/TRIED/TRIED.md,
`MAIN` = ~/Papers/MAIN.md (being edited live; its line numbers are as of 11-09 ~17:10 and may shift).
Code line numbers are for the current files (EXT = 2255 lines, last change 11-04 15:24). The released engine
(~/Leap/release/out/pkg/engine/) carries byte-identical gated.py and bigmoe_ext.cpp (diffed 11-09).

---

## 1. The problem

- **Exactness needs every routed expert resident before its layer's expert matmul runs; nothing is substituted.**
  The router's ids are computed on the GPU in the middle of the forward pass, and layer L+1's routing depends on layer
  L's output, so per layer the ids must reach the CPU loader, misses must be read from the SSD into GPU-visible slots,
  and the GPU must then learn which slots to read [assumed: design premise] (F:122-125; STREAM:1-12; EXT:1-10).
  - Look-ahead can only predict the ids: 72-96 % recall on the 35B (MoEPrefetch, F:105-107) [measured by MoEPrefetch];
    on the 125B, per-row recall@10 0.634 for the forecast the engine posts (L+1's router on L's MoE input, "R-57") and
    0.79 even for L+1's own MLP-side read taken at its start (WED:70-74; data/r57/pr2_160_preroute.txt) [measured]. So
    the exact path needs a per-layer handshake whatever the prediction quality [assumed: design reading].
- **Naive sync cost, as recorded:**
  - `mx.eval` forced at every layer (MoEPrefetch `scripts/sync_cost.py`, Qwen3.6-35B-A3B 4-bit, 40 layers, no SSD
    reads, M5 Pro, 2026-10-17 ~15:00): 107.8 -> 54.5 tok/s (-49 %); every 4 layers 80.3 (-26 %) [measured; raw repo
    ~/MoEPrefetch is no longer on disk; the table survives in the transcript
    ~/.claude/projects/-Users-kamen-MoEPrefetch/6954b3dc-d0d6-43f9-828d-7354cd4f0eb5.jsonl]. Per layer:
    (1/54.5 - 1/107.8) / 40 = 227 us [modelled, derived here] = F:84's "~230 us, the known floor (MoEPrefetch)".
  - Synthetic (BM/scripts/gpuwait.py, 48 fake 20 MB Q4 layers, 2026-10-17 19:16): `mx.eval` every layer
    +157.4 / 160.6 / 161.5 us per layer (BM/data/gpuwait.txt:4,14,24) [measured].
  - Plain Swift + Metal (BM/swift/eventwait.swift): a GPU-signalled MTLSharedEvent + CPU listener + `encodeWait` in the
    next layer's command buffer: +147.9 us/layer (20 MB layers), +166.2 (60 MB layers) (BM/data/eventwait.txt:4,10);
    131-148 us in reruns (eventwait_v2..v4.txt) [measured, 10-17 19:34-20:20].
  - Phase-1 cost model: 48 layers x 0.155 ms = 7.4 ms per forward pass, ~40 % of a fully resident token time
    (F:122-125; F:228-231) [modelled]. Note: BM/scripts/costmodel.py:23 now defaults the handshake to 24 us
    ("measured with bigmoe_ext (data/test_gate.txt); 155 us with MTLSharedEvent GPU signal"), so data/costmodel.txt as
    regenerated no longer uses 0.155 ms [code-read].
  - On the 125B (Leap R-50c, 2026-10-29 ~22:2x, warm exact 16-row check at C 192 on the synchronous slot path):
    "bare sync latency 0.256 ms", 97 syncs per check, CPU-side GPU-idle bubble ~59.7 ms of a ~185 ms check
    (~/Leap/data/r50c_screen.txt:2,4,5; HO:463-466) [measured].
  - The same 125B check, sync slot path vs the sync-free gate (R-55, C 192, 2026-10-30): warm check + commit
    232.6 -> 117.3 ms (HumanEval/68), 213.7 -> 106.2 ms (prealgebra/1640), logits bitwise equal on every window
    (BM/data/r55/r55_screen.txt; HO:480-487) [measured].
- **Why the obvious fix (GPU-side demand paging inside one kernel) was not usable on 10-17:** see step 2 in section 2.

---

## 2. The design steps, in order

Model shorthand: **35B** = Qwen3.6-35B-A3B MLX 4-bit (40 layers, 256 experts, top-8; fits in RAM, so outputs can be
compared bitwise with the resident model); **125B** = Qwen3.8-Flash-Next 4-bit (48 layers, 512 experts, top-10).

| # | step | date / commit | measured cost (what, where) | model / config | raw data | tag |
|---|---|---|---|---|---|---|
| 1 | `mx.eval` per layer | MoEPrefetch, 10-17 ~15:00 | 107.8 -> 54.5 tok/s, ~227 us per layer (derived) | 35B, no SSD reads | MoEPrefetch transcript above; F:84 | measured (+ derived) |
| 1b | same, synthetic | 10-17 19:16 | +157-162 us per layer | 48 fake 20 MB layers | BM/data/gpuwait.txt | measured |
| 2 | in-kernel demand paging (one kernel per layer checks a residency flag, posts a request, spins) | 10-17 19:16-19:33 | hit path 3.2-5.1 us/layer; miss path broken (below) | synthetic, MLX custom kernel | BM/data/gpuwait.txt, coherence_v012.txt, coherence.txt | measured |
| 3 | MTLSharedEvent handshake in plain Swift | 10-17 19:34-21:12 | GPU-signalled event: 127-166 us/layer; CPU-signalled wait after a memory flag: +-0.8 us/layer | Swift, 48 synthetic layers | BM/data/eventwait*.txt | measured |
| 4 | event gate inside MLX (CpuGate: id copy, forced commit, next buffer `encodeWait` on a per-layer CPU-signalled event, GPU-side copy of slot ids) | 7c9f049, 10-17 20:44 | F:85: ~170-180 us per gated layer; raw C=255 runs give 146-181 us (derived) | 35B, C=255 (all but one expert resident), 100 tokens | BM/data/diag35.txt, trace35.txt | measured (+ derived) |
| 5 | commit + fresh-line spin (`spin_wait`) | be479ce, 10-17 21:49 | F:87: ~69 us per gated layer; nearest saved raw line gives ~79 us (derived) | 35B, C=255 | BM/data/series35.txt:23; spinwait*.txt; test_gate.txt | measured (+ derived) |
| 6 | + residency table published a pass ahead (hits skip the wait) | be479ce, 10-17 21:49 | F:88: "~5 % total on hits"; synthetic all-hit gate 21.7 us/layer | 35B; synthetic 48 layers | F:88 (raw run not saved); BM/data/test_gate.txt | measured (see caveat) |
| 6b | slow-path decomposition | 10-18 01:25 (5573413) | round trip ~45 us of ~350 us GPU wait per slow layer; rest = SSD + CPU | 35B, 60 %, prefetch v1 | BM/data/series35.txt:80-82; F:356-366 | measured + derived |
| 7a | MAXK widening: 40 -> 61 (4-line row) -> 80/92 (G2: 5th id line, 3-line release entries, SpinWait stride fix) | 945cd21 / bc1537a / 46db7d3, 10-18 | G2: reference sha c09a8163f9 identical at MAXK 80 | 125B | LED:48, :52, :68 | measured |
| 7b | prediction lines in the routing row (W1, BIGMOE_PRED_LINES) | 10-18 | unpredicted misses 92 -> 71 % | 125B spec k=3 | LED:58 | measured |
| 7c | JIT table publication (MoonShots PS: L+1's table published right after L is served) | 6efbab5, 10-18 20:31 | spec k=3 37.1-37.3 -> 39.0-39.9 tok/s (+6 %), exact | 125B, 50 % | LED:56; ~/MoonShots/patches/prefetch_sched/README.md | measured |
| 7d | spin back-off knobs (BIGMOE_SPIN_G1/G2/KNEE/PAIRS) | 69e1c47, 10-19 | x1.1 back-off wakes +35/40/130 us late after 0.6/1/2 ms waits; finer back-off: no gain at 24 GB, +1.8 % at 12 GB (Fit12 SB1) | 125B | LED:87, :132; TRIED:505, :767, :773 | measured |
| 7e | out-of-order experts (post returns a partial-hit row; resident pairs run before the spin) | af1af44 / d0c2154, 10-21 | 24 GB era, spec k3: code 34.2 -> 36.0, prose 24.8 -> 26.3, edit 33.4 -> 34.8 tok/s (+4-6 %), bitwise; +18.7 % steady, +15.9 % whole at 45 GB C 224 (11-02) | 125B | LED:117; HO:1904-1905 | measured |
| 7f | JOINT (sidecar row's reads start while the base row is served) | 359c028, 10-25 | 24 GB +6.7-8 %; 45 GB C 160 KILL by its line | 125B | LED:131; BM/docs/NIGHT_1104.md:23-36 | measured |
| 7g | gate stamps (10-bit pass tag in every handoff word) | 41bb474 / fc20ff6, 10-25 | stamp waits 0 quiet; 9 torn calls caught under load, 0.02-0.03 ms | 125B, 24 GB | LED:128 | measured |
| 7h | GATE16: up to 185 ids per row (MAXK 93-185, 2-4 extra id lines) | 4be3b31, 10-30; default MAXK 160 since 8e2b22e, 11-01 | 16-row check 232.6 -> 117.3 ms; live steady 23.7 -> 32.0 tok/s (+35 %) | 125B, 45 GB line, C 192 | BM/data/r55/r55_screen.txt; HO:480-512 | measured |
| 7i | W1-PICKUP: coherent one-row spin (BIGMOE_SPIN_COHERENT) + system-scope post (BIGMOE_POST_SYS) | Leap 11-03 (patch 77ada05), built 5565a56, adopted 0365ada 11-04 | Swift: release 6.5 -> 0.2 us, pickup 16.7 us (+~10 % of the wait) -> 2.9 us flat; engine: +1.9 % / -0.6 % (traced single passes), stack +6.4 % | 125B, 45 GB, C 224 / C 160 | ~/Leap/data/w1_pickup*.txt; BM/data/w1/; BM/data/d45/d45_report.txt | measured |
| 7j | one-call OOO (the expert kernel itself polls the release region) | Leap, used from 155d5c7 (11-04) | nondeterministic until coherent pool reads (11-08) | 125B lossy configs | WED:942-990 | measured |

### Step details

**Step 2, in-kernel demand paging (the prototype that failed).** F:199-226; BM/scripts/gpuwait.py, coherence.py.
- Hit path: a tiny kernel per layer checks a residency flag and falls through: +5.1 / 3.6 / 3.2 us per layer
  (gpuwait.txt:3,13,23) [measured].
- Miss path: the GPU posts a request and spins; the CPU polls, `pread`s the expert, sets the flag.
  - Per stalled layer: 70,537-196,042 us with no I/O, i.e. 70-196 ms (gpuwait.txt:5,15,25) [measured].
  - The GPU's checksums of the slots the CPU wrote mid-command-buffer disagreed in 47 of 48 slots in all three
    variants (relaxed atomic load, RMW `fetch_add(0)`, volatile load) (gpuwait.txt:9,19,29) [measured]: stale data.
  - coherence.py (variants V0 relaxed atomics, V1 RMW / exchange, V2 volatile): the CPU saw the GPU's request after
    0.5-999 ms, GPU spins often hit the 20 M cap, kernels took 0.06-2.2 s (BM/data/coherence_v012.txt) [measured].
  - V3 (`thread_scope_system` fences): did not compile, "use of undeclared identifier 'thread_scope_system'"
    (BM/data/coherence.txt:1-10) [measured]. Device-scope fences changed nothing (F:215-216) [measured].
  - V4 (a 64 MB strided sweep before every load): the CPU saw the request after 0.6-0.9 ms (first trial 28-36 ms),
    but the GPU never saw the CPU's flag within its 4,000-poll cap (coherence.txt:11-12) [measured].
  - Conclusion recorded on 10-17: "Metal guarantees CPU/GPU visibility only at command-buffer boundaries" (F:216);
    "In-kernel signalling ... does not work on Metal today" (F:125). Partly overturned on 11-03 (step 7i): with
    `#pragma METAL internals : enable` and `coherent(system)` loads / system-scope fences, both directions work inside
    a running command buffer [measured, ~/Leap/data/w1_pickup.txt].
- MLX gotcha: small input arrays are bound as constant data copied at encode time, so a kernel spins on a snapshot
  forever; shared buffers must be > 4 KB (F:210-211) [measured per F; the failing run's output is not saved; MLX's exact
  rule was not checked in MLX source here (not on disk)]. Every gate buffer is sized past it (GATED:104: route
  `max(4096, nl*ROW)` int32 "> 4 KB: bound as device memory"; gpuwait.py:45; coherence.py:46).

**Step 3, MTLSharedEvent in plain Swift** (BM/swift/eventwait.swift; 48 layers, one bandwidth-bound kernel each).
- One command buffer per layer instead of one per token: +-0.5 to +-3.3 us per layer (eventwait.txt:3,9;
  eventwait_v2..v4) [measured]: "the command-buffer split itself is free (<= 3 us)" (F:220).
- GPU-signalled event -> CPU listener -> CPU signal -> next buffer's `encodeWait`: +147.9 / +166.2 us per layer
  (eventwait.txt:4,10); +131.2, +132.7, +139.6 (v2-v4) [measured].
- GPU-only signal -> wait chain: +127.1 / +142.3 us per layer (v2:5, v3:5, v4:5) [measured]: the GPU-signalled
  shared event itself is the cost.
- Pre-signalled shared-event waits: +-1.2 / +-1.0 us; MTLEvent (GPU-only) signal -> wait: +-1.2 / +-1.7 us (v3, v4)
  [measured].
- **Memory flag GPU -> CPU, CPU-signalled wait** (a 1-thread flag kernel at the end of layer L's buffer, the CPU
  busy-reads it, then sets the shared event layer L+1 waits on): +-0.8 us per layer (eventwait_v4.txt:8); with a cold
  2.77 MB `pread` on 4 / 12 layers: +150 / +170 us per stalled layer, against +468 / +384 with the listener handshake
  (v4:9-13) [measured]. Unretained references and 20 small kernels per layer: +0.7 to +1.0 us (eventwait_v5.txt)
  [measured, 10-17 21:12].
- A stalled layer with one cold 2.77 MB read adds ~370 us (eventwait.txt:11-12; F:220-221) [measured].

**Step 4, the event gate inside MLX** (7c9f049, 10-17 20:44; GATED at 7c9f049; EXT:1438-1478 still holds `CpuGate`).
- Mechanism: a tiny id-copy kernel writes [seq, n, ids] into a shared routing row; a 64 MB dummy input pushes the
  command buffer past MLX's per-buffer MB limit so MLX commits right after the copy; the next command buffer
  `encodeWait`s on layer L's MTLSharedEvent at value seq; the loader signals it after loading; a GPU kernel inside the
  waiting buffer then copies the CPU-written slot ids into the gate's own output (EXT:1446-1451) [code-read].
- Cost on the real 35B: F:85 "~170-180 us" per gated layer [measured per F]. Raw lines (BM/data/diag35.txt, C=255 =
  gate on every layer with almost no misses, 100 tokens, ~40 gated calls per token; C=256 = fully resident 102.1 tok/s):
  58.7 / 62.8 / 64.0 / 62.5 tok/s -> 181 / 153 / 146 / 155 us per gated layer [modelled, derived here as
  (1/tps - 1/102.1)/40]. The 64.0 and 62.5 runs set MLX_MAX_OPS_PER_BUFFER=1000 or 10 with MLX_MAX_MB_PER_BUFFER=60:
  no material change [measured]. NOWAIT (gate mechanics without the wait): 103.6 tok/s, i.e. the wait is the whole cost
  [measured, diag35.txt:5].
- Where the time went (BM/scripts/trace35.py -> BM/data/trace35.txt, event mode, 35B, n=1644 gated layers): the CPU
  signalled 107-112 us (median) BEFORE the gated buffer started on the GPU; the buffer ran 160-169 us after the
  signal; "buffer end -> CPU sees next row" 203-206 us; per-layer period 418-428 us; "scheduled-handler -> GPU start
  72 us" (trace35.txt:1,3) [measured]. Recorded reading: "a buffer with encodeWait is only scheduled after the signal,
  then ~70-110 us to start ... free in plain Swift, not inside MLX's stream" (F:85). Why MLX differs from the Swift
  harness is not explained by any test [assumed: unexplained]. trace35.txt:2 (13 / 14 / 182 / 197 us) is an
  unlabelled second configuration, consistent with the wait disabled [assumed].
- Synthetic (scripts/test_gate.py, committed with 7c9f049): all-hit gate +44.5 us/layer (on a 10.93 ms/token plain
  baseline) [measured, `git -C ~/BigMoE show 7c9f049:data/test_gate.txt`].

**Step 5, commit + fresh-line spin** (be479ce, 10-17 21:49; F:87).
- The GPU writes the routing row, MLX commits the command buffer (forced), and the next buffer starts with a 1-thread
  spin kernel that polls a fresh 32 KB region per (pass, layer) line pair by line pair; the CPU releases by writing
  [seq, slots] into every pair of that region. "A line never read before is never stale in the GPU's L2" (F:87).
- F:87 records ~69 us per gated layer on the real 35B [measured per F]. The derivation is not saved. The closest saved
  raw line (series35.txt:23, spin gate, C=255, 465 misses in 4036 gated calls) is 77.2 tok/s -> (1/77.2 - 1/102.1)/40 =
  79 us per gated layer [modelled, derived here]. Paper should quote 69 us only with this caveat or re-measure.
- Swift prototype (BM/swift/spinwait.swift, one command buffer per token, [work, row, spin] per layer, 1024-line
  region per (layer, pass)): +2.9 us/layer with no loader delay, CPU saw rows every 6 us, max 1 poll; +59.9 us/layer
  with a 100 us loader delay, max 184 polls; 0 timeouts (BM/data/spinwait.txt, 10-17 21:27) [measured]. Untracked /
  concurrent / one-encoder variants: +46.3 to +58.5 us at 100 us delay, 0 timeouts (spinwait_variants.txt) [measured].
- Synthetic ext test (scripts/test_gate.py, be479ce): all hits +21.7 us/layer; 101 us per stalled layer at 4 stalls
  per token, 133 us at 12; loader idle-wait 291 us per call; correctness 0 mismatches of 2,400 gathered experts with
  random ids and many misses (BM/data/test_gate.txt) [measured].

**Step 6, residency published a pass ahead** (be479ce; F:88).
- After serving layer L of pass s the loader publishes L's residency table for pass s+1; the routing-row kernel looks
  every id up in parallel; if all are resident the spin returns at once and the GPU never waits [code-read, EXT:1758-1766,
  EXT:1076-1093, EXT:1143].
- F:88: "~5 % total on hits"; same cell: "C=255: 93.4 tok/s with misses skipped vs 102 resident" [measured per F; the
  93.4 run is not in any saved data file; note 93.4/102.1 is -8.5 %, not -5 %].
- 35B with prefetch v1 at C=153: fast-path layer passes 2303 of 4037 gated calls (57 %) (series35.txt:40, :49)
  [measured].
- Prefetch v2 "additive table publish" (786c141, 10-17 22:34): a prefetch that lands after publication is added to the
  published table (`table_add`), race-free because additive ("read earlier = slow path, read later = fast path via the
  spare slot"); measured no gain: 62.8 / 80.4 / 86.4 vs 63.7 / 82.3 / 85.8 tok/s (F:340-344) [measured].

**Step 6b, slow-path decomposition** (10-18 01:25, scripts/series_rt.sh; F:356-366).
- 35B, 60 %, prefetch v1, plain decode, cold SSD: exact 62.4 tok/s with 14.4 slow calls/token at 305 us CPU each; the
  same with the GPU never waiting (BIGMOE_SKIPSPIN, wrong output) 91.2 tok/s (series35.txt:80-82) [measured].
- (1/62.4 - 1/91.2)/14.4 = 351 us GPU wait per slow layer; minus 305 us CPU = ~46 us round trip [modelled, derived
  here; F:357-359 says ~350 / 305 / ~45].
- One cold 1.77 MB expert: 0.39 ms p50 as 9 pieces; splitting into 512/256/128/64 KB chunks: 0.39 / 0.46 / 0.48 /
  0.53 ms (BM/data/chunk_bench.txt) [measured]. Verdict: "the round trip itself is ~45 us and not worth more
  engineering" (F:363-364). Loader order changed to "release first, bookkeeping after" (5573413; EXT:1756).

**Step 7i, W1-PICKUP** (Leap's Swift harness ~/Leap/native/w1_pickup.swift; 300 interleaved trials per cell under a
quiet lock, Tue 11-03 19:41; RS 192, n 160, 128 pairs = the MAXK 160 layout) [all measured, ~/Leap/data/w1_pickup.txt,
w1_pickup_P.txt, w1_pickup.log]:
- Arm A, today's fresh-line spin (bigmoe_spin verbatim): release (CPU writes 128 pairs) median 6.4-7.9 us; pickup
  15.7-17.4 us when the data is already there, 28.9 us at a 200 us wait, 73.5 us at 1 ms, 67.2 us at 2 ms. Pair index ->
  elapsed: p0 17 us, p6 22, p26 73, p41 236, p52 623, p58 1080, p65 2074 us.
- Arm B / B1, one simdgroup polling ONE row with `volatile coherent(system)` loads and seq_cst `thread_scope_system`
  fences (MLX's fast-synch pattern, `#pragma METAL internals : enable`): release 0.17-0.21 us; pickup 2.8-2.9 us flat
  from 50 us to 2 ms (3.6-3.9 us at w <= 0).
- Arm M, MLX's own compiled `fence_wait` kernel (from the .venv's mlx.metallib): pickup 1.2-2.0 us.
- Arm C, MTLSharedEvent wait: pickup 2.2 us if signalled before commit, 12.1 at 0, 76.6-96 us at 50 us-1 ms, 360 us
  at 2 ms.
- Post visibility (GPU -> CPU): a device-scope post followed only by the spin was not seen within 5 ms in 285/300
  (293/300 in the rerun); a following 64 MB / 8 MB read dispatch made it visible after 27.4 / 20.0 us (median); a
  system-scope post was seen in 7.4 us. Device-scope posts showed up to 129-145 unstamped words when the tag was seen;
  the system-scope post 0.
- Integrated in EXT:995-1047, 1059-1111, 1166-1168, 1738, 1851 (flags off by default in the ext; on in the 45 GB preset,
  BM/scripts/r57_screen.py:61-70).

---

## 3. The mechanism as implemented now

### 3.1 Buffers and sizes (shipped 45 GB preset: MAXK 160, PRED_LINES 1, tiers on, MTP stream off)

| object | what | size / indexing | source |
|---|---|---|---|
| gate rows `nl` | 48 base layers + 48 LSB sidecar layers (+1 if the MTP head streams) | `Gate(nl + nmtp + nl)` = 96 | STREAM:493 |
| routing rows (`route`) | one row per gate layer, GPU -> CPU | ROW = 96 + 32*PRED_LINES + 32*XLINES int32 = 256 ints = 1 KB = 8 lines at MAXK 160; buffer `max(4096, nl*ROW)` ints | GATED:25-32, :104; EXT:582-591 |
| slot-id ring (`ring`) | CPU's slot ids per (pass, layer) | 4M int32 = 16 MB; offset `((seq*nl + L) % (size/MAXK)) * MAXK` | GATED:105; EXT:1499-1500 |
| release regions (`region`) | CPU -> GPU release per (pass, layer) | 128 regions x PAIRS (128) entries x 32*(2+XLINES) ints: 32 KB per region at XLINES 0 (4 MB ring), 96 KB at XLINES 4 (12.6 MB ring); region `(seq*nl + L) % 128` | GATED:109-111; EXT:1736, 2209-2210 |
| residency tables (`tables`) | CPU -> GPU, published a pass ahead | 1024 tables x 17 lines x 32 ints (2.2 MB); 17 x 31 = 527 entries >= 512 experts; table `(seq*nl + L) % 1024` | GATED:91, :114-115; EXT:1221-1237 |
| `status` | GPU-written counters | 4096 int32: [0] timeouts, [16] fast-path hits, [17] stale-word re-polls | GATED:112, :183-192; EXT:1043-1045, 1158 |
| `bump` / `g_post_bump` | 64 MB unused input that forces MLX to commit after the post | 64 MB uint8 | EXT:1054, 1211, 2179 |
| events | one MTLSharedEvent per gate layer (event mode only) | value = pass seq | EXT:994, 2187 |

Reuse distances at nl = 96 [modelled from the formulas]: a release region every 128/96 = 1.33 passes; a table every
~10.7 passes; a ring row every ~273 passes. (At nl = 48 a region recurs every 2.67 passes, the "~3 tokens" of
GATED:109-110.)

### 3.2 One gated layer, step by step (default spin mode, decode / verify width)

1. **Python dispatch** (STREAM:76): a call takes the sync-free gate only if `indices.size <= GATE_N[0]` (= MAXK) and
   (`<= 92` ids or `<= C` slots); otherwise `_sync_call` (host sync on the ids, `ext.ensure`) [code-read]. Gate.new_pass
   (GATED:126-138) bumps `seq` per pass and waits at the pass boundary while ~/.local/share/quiet/PAUSE exists.
2. **GPU post** (`ext.post_ids` -> PostIds primitive, EXT:1186-1220; kernel EXT:1055-1118): 64 / 128 / 192 threads (one
   per id). Threads look their id up in the (seq, L) residency table and write `dep = [ok, slot_0..slot_{n-1}]` (-1 =
   not known resident). Thread 0 writes the routing row (3.3), then MLX commits because the 64 MB bump input pushes the
   buffer over its MB limit (3.5).
3. **CPU loader** (one native thread, `loader_main`, EXT:1652-1807, QoS user-interactive "keep the spinner on a P-core"):
   busy-polls word 0 of every row for a new seq (`__builtin_arm_yield()` when idle; optional 200 us naps after
   BIGMOE_IDLE_MS of idleness, EXT:1523-1527, 1803); waits until every line tag of the row equals seq; acquire fence;
   reads n (abort if n < 0 or > MAXK: "bad n", EXT:1677); re-reads the ids until every word carries this pass's stamp
   (warning after 250 ms, EXT:1681-1692); serves the call with `ensure()` (3.6); release fence; writes the release
   region (3.4); then, off the critical path, publishes residency tables and schedules prefetches (EXT:1756-1792).
4. **GPU spin** (`ext.spin_wait` -> SpinWait primitive, EXT:1176-1184, 1825-1853; kernels EXT:1123-1175 / 1014-1047):
   in the next command buffer. If `dep[0] == 1` (all hits) it copies the slots from `dep` and returns (fast path,
   status[16]++). Otherwise it polls the release region (3.4) and writes the decoded slot ids into its output.
5. **Consumers**: the expert kernels gather the Metal-backed pools at the spin's output slots; slot ids never leave GPU
   memory, so no CPU sync is needed. Under out-of-order execution the post comes first, resident pairs (`dep[1+i] >= 0`)
   run before the spin, the rest after it, and a fixed-order combine keeps the result bitwise (TIER:139-164) [code-read;
   bitwise equality measured, LED:117].

### 3.3 What the GPU writes: the routing row, line atomicity, stamps

- Layout (EXT:582-591, 603, 1071, 1094-1106; GATED:31-32), 128-byte lines of 32 int32, each with its own seq tag in
  word 0:
  - line 0: [seq, n, ids 0..29]; line 1: [seq, ids 30..60]; line 2: [seq, np, preds 0..29] (np = n1 | n2<<8 | n3<<16 |
    ranked<<24 for layers L+1..L+3); PRED_LINES lines [seq, 31 preds]; XLINES lines [seq, 31 ids] (ids 61..185).
  - Capacity: 30 + 31 + 31*XLINES ids (<= 185, asserted GATED:28); 30 + 31*PRED_LINES predictions.
- Write order (thread 0): np and predictions, n, ids (relaxed device-scope atomic stores; with POST_SYS plain stores
  through a `volatile coherent(system)` pointer), then a fence (device scope; system scope with POST_SYS), then the tags
  of lines 3+ (prediction and extra id lines), line 2, line 1, and line 0 last; with POST_SYS one more system-scope
  fence "publish now" (EXT:1096-1109).
- Line atomicity: "each line carries its own seq, written last, so a line seen with the new seq is complete"
  (GATED:31-32) [assumed in 10-17's design; shown insufficient under load on 10-25, see 5.13].
- Gate stamps (since 41bb474, 10-25): every id / prediction word carries `seq & 0x3FF` in bits 22-31; payload in bits
  0-21; -1 travels as 0x3FFFFF, -2 as 0x3FFFFE; bit 21 of an id = QOM's "optional" mark, bit 20 of a prediction = POM's
  flag; the table lookup uses `ids & 0xFFFFF` (EXT:592-601, 1082, 1095-1101). Readers check every word's stamp.

### 3.4 What the CPU writes back: the release region

- Region for (seq, L): `g_reg + ((seq*nl + L) % 128) * PAIRS * RS` with RS = 32*(2+XLINES) ints (64 / 96 / ... / 192)
  (EXT:589, 1736).
- Default: the loader writes the stamped slot words into all 128 entries ("pairs") of the region, release fence, then
  each entry's line tags from the highest line down to line 0 last (EXT:1735-1752). Per entry: line 0 = [seq, slots
  0..30], line 1 = [seq, slots 31..61], line 2 = [seq, slots 62..92], lines 3-5 for R-55 (EXT:1741-1744).
- With BIGMOE_SPIN_COHERENT=1: only entry ("pair") 0 is written (EXT:1738).
- Event mode (BIGMOE_GATE=event, not the default): `g_events[L]->setSignaledValue(seq)` instead (EXT:1753-1755).
- Release cost in the engine (S0 trace, 125B, C 224): median 0.004 ms with 128 pairs, 0.000 ms with the coherent one-row
  release (BM/data/r57/s0_report.txt) [measured]; Swift: 6.5 vs 0.2 us (3.1-arm data above) [measured].

### 3.5 When MLX commits

- MLX commits a command buffer after any primitive whose evaluation leaves `buffer_ops_ > max_ops` or
  `buffer_sizes_ >> 20 > max_mb` (BM/ref/mlx_device.cpp:511-514; BM/ref/mlx_eval.cpp:59-64; copies of MLX source made
  10-17) [code-read]. Input sizes count once per buffer (mlx_device.cpp:345-351).
- Limits by GPU class: phone 20 ops / 40 MB, base/pro ('g') 40 / 40, Max ('s') 50 / 50, Ultra ('d') 50 / 50, env
  overrides MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER (mlx_device.cpp:604-626) [code-read]. The 64 MB bump exceeds
  every class's MB limit, so the commit lands right after the post [modelled from the code; only M5 Pro measured].
- BIGMOE_NOPOSTCOMMIT drops the bump (EXT:2179); no recorded run uses it [code-read].
- Why the commit is needed (F:87): "GPU writes reach the CPU only at command-buffer end inside MLX (not at kernel or
  encoder end, unlike Swift)". PostIds also binds the row as a real output because "stores via const_cast of a read-only
  input only became CPU-visible at command-buffer end" (EXT:1186-1188). See section 4 for what is measured.

### 3.6 How misses are served (`ensure`, EXT:755-904)

- Deduplicates the call's ids; strips QOM's bit 21; aborts on an id outside [0, E) ("bad expert id", EXT:771) and when
  the call needs more distinct experts than the pool's main slots ("experts > C", EXT:777).
- Hits: LRU touch. Prefetched: waits for `ready`, then promotes into the LRU. Misses: `make_room` evicts LRU-tail experts
  not needed by this call, takes a free slot, and reads the record with F_NOCACHE `preadv` into the Metal-backed pool
  pieces (directly or through a bounce buffer), via GCD at user-interactive QoS (EXT:458-530, 845-891).
- Writes each id's slot into the ring (`-2` = not served: QOM optional miss or LP1) (EXT:899); the release copies these
  into the region (EXT:1737-1745).
- A layer's slots change only while that layer is served; victims are never in the current routing (PLAN_wall1_lever1
  S7B note, BM/docs/PLAN_wall1_lever1.md:74-78) [assumed as a schedule invariant; checked by the stale-slot test, 5.17].

### 3.7 Residency published a pass ahead, and the GPU-side lookup

- Non-JIT (default): right after releasing (seq, L) the loader writes L's table for pass seq+1: 17 lines x 31 stamped
  slot entries (-1 = absent), release fence, then each line's tag = seq+1 (EXT:1758-1766). "Valid until that pass
  serves L" (EXT:1758).
- JIT (BIGMOE_PF_JIT=1, MoonShots PS): L+1's table for pass seq is published right after serving L of the same pass,
  including finished prefetches; layer 0 is never published (always takes the gate) (EXT:1239-1253, 1771-1776;
  ~/MoonShots/patches/prefetch_sched/README.md:17-31).
- Additive updates: a prefetch that lands later is added with `table_add` (only if the table's line tag is this pass;
  `table_add` runs before `ready = 1`) (EXT:1221-1228, 1264-1265); victim prefetch removes entries with `table_remove`,
  "caller guarantees the GPU has not read (seq, L) yet" (EXT:1229-1234).
- GPU lookup (EXT:1076-1093): an entry counts only if its line tag == seq AND its word stamp == seq ("fresh"); any id not
  fresh and resident clears `ok`; `dep[1+t]` keeps the per-id slot or -1 (the partial-hit row that OOO uses). "If the CPU
  publishes late, the tag does not match, the GPU takes the slow path, and the result is still exact" (MoonShots README:31)
  [assumed by design; exactness of all runs measured].
- The loader still serves every posted row, fast path or not (it does not know which path the GPU took) [code-read,
  EXT:1658-1700].

### 3.8 What the GPU polls and how

- **Default fresh-line spin** (EXT:1123-1165): one thread. For p = 0..PAIRS-1: read entry p's line tags through a
  `volatile device` pointer; if all equal seq, check every word's stamp and decode (a fresh tag beside a stale word
  -> status[17]++ and move to the next entry); between polls a dependent FP loop of d iterations, d starting at 32 and
  growing x1.1 per poll (BIGMOE_SPIN_G1 / G2 / KNEE) up to 400,000. Each entry is read at most once per spin, so every
  poll reads lines the kernel has never read. After PAIRS (128) polls: timeout, status[0]++, every output slot = 0.
  "Today's 128 back-off pairs end after ~0.2 s" (Leap's README, ~/Leap/patches/w1_pickup/README.md:16-17) [measured in
  Swift: pair 65 = 2.07 ms; the 0.2 s total is extrapolated].
- **Coherent one-row spin** (BIGMOE_SPIN_COHERENT=1, EXT:1014-1047; 32 threads, EXT:1851): repeat up to COH_CAP
  (default 1,000,000 polls, "~0.25 s at ~0.25 us per poll", EXT:1000-1001): system-scope fence; `simd_all(tag of each
  line == seq)`; fence; lanes check the stamps of their words and decode; all good -> return; else status[17]++ and poll
  again. Timeout as above.
- `status` is bound as an INPUT: "registering a persistent buffer as an output every layer made MLX fence each spin
  against the previous one (~45 us/layer)" (EXT:1847-1849) [measured per the comment; raw run not saved].
- Timeout handling: the pass continues with slot 0 for every id (wrong data) and `Gate.check()` raises after the pass:
  "results of that pass are invalid" (GATED:187-192; F:90-91). Every screen calls it per generation (e.g.
  BM/scripts/r57_screen.py:1208, bench.py:57) [code-read].

### 3.9 Limits

- MAXK <= 185 (GATED:28); XLINES = 0 if MAXK <= 61 else min(4, (MAXK-61+30)//31) (GATED:30 = EXT:587); default MAXK 160
  since 11-01 (GATED:19-22); the 24 GB stack uses 80, the 12 GB stacks 70-92 (GATED:21) [code-read].
- A > 92-id call takes the gate only if the pool holds all its experts (`<= C`), else the sync path (STREAM:76-77);
  ensure() aborts above C_main (EXT:777). Consequence recorded at 24 GB: with 94-112 base slots, 16-row checks (160 ids)
  ran on the sync path: 231 MB SSD per kept token on HumanEval vs 18.1 at C 160 (HO:867-868) [measured]; "a 16-row check
  takes the sync-free path only when it fits the pool (C >= 160 per layer)" (WED:160-162) [measured].
- Expert ids must fit 20 bits (bits 20 / 21 are flags) [code-read, EXT:1082]; stamps carry 10 bits = 1024 passes, "far
  beyond how old a stale word can be" (EXT:596-597) [assumed].
- Ring sizes: 128 release regions, 1024 tables, 16 MB slot ring (3.1). Spin timeout ~0.2 s (default) / ~0.25 s
  (coherent); stamp wait warning at 250 ms.
- Spin polls are serial on one thread (default) or one simdgroup; MTLSharedEvents only in event mode.

### 3.10 What the shipped 45 GB exact default turns on

- R57_PRESET=45 with R57_D45=1 (default) and R57_COMBINE=1 sets BIGMOE_OOO=1, BIGMOE_TIER_COMBINE=1, BIGMOE_JOINT=1,
  BIGMOE_COPY_PAR=1, BIGMOE_SPIN_COHERENT=1, BIGMOE_POST_SYS=1, BIGMOE_SSD_KEEPALIVE_US=50; MAXK 160 from gated.py
  (BM/scripts/r57_screen.py:28-70; adopted 0365ada, 11-04 08:35) [code-read]. BIGMOE_SPIN_PAIRS is not set, so the
  12.6 MB region ring stays allocated although only entry 0 is used (Leap README:33-34) [code-read].

---

## 4. Why it works, as recorded (measured vs explained)

### 4.1 GPU -> CPU visibility: "command-buffer end inside MLX vs kernel / encoder end in Swift"

- Measured, Swift: in spinwait.swift (one command buffer per token, a separate encoder per kernel) the CPU saw each
  layer's row write mid-buffer, every 6 us at zero delay (BM/data/spinwait.txt:1) [measured]. In eventwait_v4 the
  flag written by the last kernel of each per-layer buffer was read by the CPU with +-0.8 us overhead [measured].
- Measured, Swift (Leap, 11-03, one encoder for all kernels): a device-scope store reaches the CPU only when something
  flushes or evicts its lines: ~20-27 us after a following 8-64 MB read dispatch; with only a spin following, not seen
  within 5 ms in 285/300 and 293/300 trials; a system-scope store + fence is seen in 7.4 us (~/Leap/data/w1_pickup.txt,
  w1_pickup_P.txt) [measured].
- Measured, MLX custom kernel (10-17): coherence.py V4 (a 64 MB sweep after the store) let the CPU see the request in
  0.6-0.9 ms; V0-V2 only after 0.5-999 ms or at kernel end (coherence*.txt; F:212-214) [measured].
- Recorded but without saved output: "inside MLX, GPU writes reach the CPU only at command-buffer end (not at kernel
  or encoder end)" (F:87); "stores via const_cast of a read-only input only became CPU-visible at command-buffer end"
  (EXT:1186-1188); BM/scripts/test_vis.py (10-17 21:33) measures exactly this but its output is not in data/; the
  `EndEncoder` "flush" primitive (EXT:1480-1495) has no caller. So the MLX-specific claim is [measured per notes, raw
  output not saved].
- Explanation for the difference (lines sit in a GPU cache until eviction or a flush, and a command-buffer end flushes)
  [assumed: consistent with Leap's P0 / P1 / PS arms and V4, but no test isolates the cache level or MLX's encoder
  behaviour]. Since 11-04 the shipped config adds POST_SYS (system-scope stores), which made the row visible in 7.4 us
  in Swift; whether the forced commit is still needed with POST_SYS is untested.

### 4.2 CPU -> GPU visibility: "a line never read before is never stale in the GPU's L2"

- Measured support:
  - Fresh-line polling never timed out: spinwait.swift 0 timeouts in 15 tokens x 48 layers per variant, max polls 1 at
    zero delay (spinwait.txt, spinwait_variants.txt) [measured]; engine tests 0 / 2,400 mismatches (test_gate.txt), 4,800
    and 12,000 / 12,000 gathered experts byte-correct under concurrent GPU load (diag_gate.txt; series35.txt:21 (12,000) and :39, :48, :58, :69 (4,800 each); F:70-71)
    [measured]; R-55: 48 loader-served gates per check, no timeouts (HO:485-486) [measured]; test_gate_wide.py: 300
    passes each at MAXK 40 / 80 / 160, every row right (HO:486-487) [measured].
  - Leap's S7B-PROBE (Swift, M5 Pro, 11-03 22:32-22:42, 20 trials per arm, 2 MB slot = 524,288 words, 4 runs):
    re-reading a slot the GPU already read in the same kernel (B1) or the next dispatch (B2): stale in 19-20 / 20
    trials; with `memoryBarrier(.buffers)` (B3) still stale in 16-20 / 20; a slot first read only after the CPU's write
    (B0): 0 / 20 stale; a slot read by an EARLIER command buffer (B5): 0 / 20; coherent(system) loads (B4, B6, B7):
    0 / 20 (~/Leap/data/s7b_probe*.txt; ~/Leap/native/s7b_probe.swift:4-7) [measured].
  - Same-line re-reads inside one kernel are stale: gpuwait / coherence V0-V2 (above) [measured].
- Not measured: which cache level holds the stale line (F says L2) [assumed]; that a region is safe to reuse "after ~3
  tokens, long after any GPU cache could still hold a line" (GATED:109-110) [assumed; partly supported by S7B B5:
  reads in an earlier command buffer leave nothing stale, and each spin runs in a later buffer than the last use of its
  region because MLX commits after every post]. Stale tags from an older pass carry another seq and older stamps, so a
  reused region cannot pass the checks (Leap README:30-31) [code-read].
- The coherent spin needs no fresh lines: `volatile coherent(system)` loads see the CPU's stores directly (Swift
  pickup 2.9 us; S7B B4) [measured]. It rests on `#pragma METAL internals : enable` and a self-defined
  `thread_scope_system` (EXT:1007-1013), which is undocumented; MLX uses it behind MLX_METAL_FAST_SYNCH (Leap README:
  "Risks") [assumed stable; not tested across OS or chip versions].

### 4.3 The > 4 KB constant-binding gotcha

- "MLX binds small input arrays (< 4 KB) as constant data copied at encode time, so a kernel spins on a snapshot
  forever" (F:210-211; MAIN F.5 at 1179-1184) [measured per F; the failing run's output is not saved]. The exact MLX
  rule (byte size vs element count) was not checked in MLX source in these records [assumed]. All gate buffers are
  >= 16 KB (3.1).

### 4.4 Why an event wait was slow inside MLX but free in Swift

- Measured: in Swift, CPU-signalled waits after a memory flag cost +-0.8 us/layer and pre-signalled waits +-1 us; the
  GPU-signalled shared event costs 127-166 us (eventwait*.txt). Inside MLX the CPU-signalled event gate still cost
  146-181 us per gated layer, with the waiting buffer starting ~107-112 us after the CPU's signal (trace35.txt;
  diag35.txt) [measured]. The reason MLX's stream adds this scheduling delay is not identified [assumed: unexplained;
  F:85 states only the observation].

### 4.5 Why consumers are ordered after the wait (event mode)

- "Consumers in a later command buffer are not held by an earlier buffer's wait" (F:86) [measured per F, found by
  diag_gate.py; raw failing counts not saved]. Fix: the slot ids are copied by a GPU kernel inside the waiting buffer,
  so consumers depend on a GPU-produced output and MLX's own fences order them (EXT:1446-1451) [code-read]. In spin mode
  the same holds by construction: consumers read the spin kernel's output [code-read].

---

## 5. Pitfalls and exactness failures, in date order

| # | date | what went wrong | how it was caught | fix | source | tag |
|---|---|---|---|---|---|---|
| 5.1 | 10-17 | small inputs bound as constants: a kernel spins on a snapshot forever | prototype (gpuwait / coherence) | shared buffers > 4 KB | F:210-211; GATED:104 | measured (output not saved) |
| 5.2 | 10-17 | in-kernel signalling with device-scope atomics / volatile: 0.5-1000 ms visibility, stale slot bytes (47/48), `thread_scope_system` did not compile | coherence.py, gpuwait.py checksums | abandoned for a commit + fresh-line design (revived 11-03 with the internals pragma) | data/gpuwait.txt, coherence*.txt; F:209-216 | measured |
| 5.3 | 10-17 | GPU-signalled MTLSharedEvent costs ~140 us | eventwait.swift | CPU-signalled waits | EXT:3-5; eventwait_v2..v4 | measured |
| 5.4 | 10-17 | one shared event for all layers let out-of-order rows release early | scripts/diag_gate.py ("pool bytes stale (other expert)" class) | one event per layer, value = pass seq | F:86; EXT:994; 7c9f049 message | measured per F |
| 5.5 | 10-17 | consumers in a later command buffer were not held by an earlier buffer's wait | diag_gate.py ("slot ids stale/wrong") | GPU-side copy of the slot ids after the wait | F:86; EXT:1446-1451 | measured per F |
| 5.6 | 10-17 | torn routing rows: seq visible before the ids | diag_gate.py ("loader saw wrong ids") | line-atomic rows (a seq per 128-byte line, written last after a fence; the loader checks every line's tag) | F:86; GATED:31-32; EXT:1666-1670 | measured per F |
| 5.7 | 10-17 | test_gate.py committed with the event gate shows "2397 mismatches of 2400 gathered experts"; diag_gate in the same commit shows 12,000 / 12,000 ok; the next test_gate run (be479ce) 0 / 2400 | the tests themselves | the fixes above (order of runs inside 7c9f049 not recorded) | `git show 7c9f049:data/test_gate.txt`, `...:data/diag_gate.txt`; data/test_gate.txt | measured |
| 5.8 | 10-17 | the event gate's scheduling delay inside MLX (~170-180 us per gated layer) | trace35.py, diag35.py | commit + fresh-line spin | F:85, :87 | measured |
| 5.9 | 10-17 | stores through a const_cast'd read-only input became CPU-visible only at command-buffer end | (recorded in a code comment) | the row is bound as a real output; 64 MB bump forces the commit | EXT:1186-1188, 1211 | measured per comment |
| 5.10 | 10-17 | `status` bound as an output made MLX fence each spin against the previous one (~45 us/layer) | (recorded in a code comment) | bound as an input | EXT:1847-1849 | measured per comment |
| 5.11 | 10-17 | a spin that waits too long must not pass silently | design | timeout -> status[0]++, slots 0, `Gate.check()` raises after the pass | F:90-91; GATED:187-192; RUNBOOK.md:56 ("never report its numbers") | code-read |
| 5.12 | 10-18 | the prefetch v3 attempt aborted in prefill: "138 experts > C=134 at layer 0" (prefill budget still counted spares) | ensure()'s capacity abort | budget = C - spares | series35.txt:59-61; F:349-350 | measured |
| 5.13 | 10-18 | MoEPrefetch's split-phase fork: phase-B entries shared the release-region ring with other (seq, L) entries -> fresh-line staleness, nondeterministic | MoEPrefetch's determinism check | separate phase-B regions (proposed) | LED v2-3 row (6b845b7) | measured (sibling project) |
| 5.14 | 10-18 | 80-id gate: the spin's per-entry stride was hard-coded to 64 ints | code review while widening | stride RS (64 / 96 ...) | bc1537a ("SpinWait stride fix pending test"), 46db7d3 ("80-id gate works") | code-read; no failing run recorded |
| 5.15 | 10-18 | latent: non-JIT `prefetch()` could recycle a same-pass prefetch already added by `table_add` to the published table (GPU fast path onto a slot being refilled); `ready = 1` stored before `table_add`. Dormant: n2 = 0 in every default config | MoonShots' code reading | JIT publication avoids both; `table_add` now runs before `ready` | MoonShots prefetch_sched README:57-62; EXT:1264-1265 | assumed (never observed) |
| 5.16 | 10-18 | invariant guard "reserve on published layer": a JIT prefetch may only reserve a slot in a layer whose table for this pass is not yet published | `abort()` in `reserve()` | (guard, part of the JIT protocol) | EXT:1295-1306; MoonShots README:24-26 | code-read; no record of it firing |
| 5.17 | 10-21 | spec runs on records3 exact mode hit spin-gate timeouts: nested `dispatch_apply` under wide verify miss lists starved the loader | the timeout flag | serial rebuild per job when >= 4 misses; b6 re-test "no spin-gate timeouts" | LED:113-114; 5b9842e; EXT read_expert `par` | measured |
| 5.18 | 10-25 | quiet windows SIGSTOP jobs mid-pass -> spin timeouts on resume | timeouts | `new_pass` waits while the PAUSE file exists, at a pass boundary | 5918b42; GATED:126-138 | measured (incident class) |
| 5.19 | 10-25 | torn 64-byte sectors under another process's GPU load: a line's tag fresh, its second sector stale; the loader computed with the previous post's experts, silently | MoonShots (7 torn calls in 6 loaded runs on v3g) | gate stamps: seq & 0x3FF in every word; readers re-read until stamped; spin rejects unstamped entries; stale table entries = unknown | EXT:592-601; LED:128 (GS1: quiet shas unchanged; 50 % foreign GPU load: 5 + 4 torn calls caught, 1,077 / 906 re-reads, 0.02-0.03 ms; loaded shas = quiet); TRIED:1250 | measured |
| 5.20 | 10-26 | Thermal Emergency Sleep 08:45:49-09:00:26 (lid closed): TrainLab's run lost a sequence, 2 spin gates timed out | the timeout flag | lid-closed hold rule | HO:171-176 | measured |
| 5.21 | 11-03 | inside one command buffer a CPU rewrite of a slot the GPU already read is not seen (S7B hazard) | Leap's S7B-PROBE | rule: any path that rewrites a slot read in the same command buffer needs coherent(system) reads or a buffer boundary | PLAN_wall1_lever1.md:68-83; ~/Leap/data/s7b_probe*.txt | measured |
| 5.22 | 11-03 | a device-scope post followed only by the spin can stay unseen for ms (zero-hit layers) | Leap's W1-PICKUP P0 arm | POST_SYS; at C 224 the engine had 0 zero-hit calls of 10,224 served base calls | w1_pickup*.txt; BM/data/r57/s0_report.txt | measured |
| 5.23 | 11-04 00:30 | check that today's engine is not exposed to 5.21 | stale-slot check: with vs without a command-buffer boundary before every MoE call, 3 configs x 3 prompts x 128 tokens | PASS: tokens and 127 logits hashes equal | BM/data/sc/sc.log; NIGHT_1104.md:20-22 | measured |
| 5.24 | 11-04 07:42 | "MAXK 80 crash": W1B 24 GB's 12-token warm-up died: tier.py's block (01f652c) took 9-16-row calls that MAXK 80 sends to the sync prefill path | crash | da8e7a6: such calls take the stock block | WED:81-86; HO:1640; TIER:480 | measured |
| 5.25 | 11-04 15:29 | QOM flagged optional ids with bit 21 in the graph's indices -> "a bad id at L5" | ext range check ("bad expert id", EXT:771) | flag only the base post's copy; the graph's ids stay clean | c387c97 (BM/scripts/r57_screen.py QOM block comment) | measured |
| 5.26 | 11-06 | price pitfall: "gate service" (post seen -> release on calls with misses) was priced as removable wait; never-wait raised it 14.0 -> 15.6 ms per check and lost 2.8 % | engine A/B | recorded as a spin bound that OOO already overlaps | WED:419-453; MAIN 10.1 | measured |
| 5.27 | 11-06 | metric pitfall: the per-check "stalled layer calls" counts any call that issued a read, not a wait | code reading (EXT:881) | correction note | WED:447-453 | code-read |
| 5.28 | 11-08 | one-call OOO (late slots resolved inside the expert kernel by polling the release region) was run-to-run nondeterministic: 3/12 and 4/12 IFEval prompts == AR, first diffs at different tokens | CHS3 forward bisect (12 prompts, two runs) | REF2_NAX_COH=1 coherent(system) pool reads: 12/12 == AR in both runs, 37.85 vs 37.85 tok/s | WED:942-990; data/chs/chs4_report.txt; 97e81b8; TRIED:1314 | measured |

Not found in the records: any run where the "reserve on published layer" abort fired (5.16); the pre-fix failure
counts for 5.4-5.6 (only the final 12,000 / 12,000).

---

## 6. Known (measured) vs assumed

| measured (with its source) | assumed / explained / not isolated |
|---|---|
| `mx.eval` per layer ~227 us on the 35B (MoEPrefetch sync_cost.py); 157-162 us synthetic (gpuwait.txt); 0.256 ms bare sync on the 125B (r50c_screen.txt:2) | that the 0.155 ms Phase-1 handshake models the real engine (costmodel.py now uses 24 us) |
| GPU-signalled MTLSharedEvent 127-166 us/layer; CPU-signalled waits +-0.8 us in Swift (eventwait*.txt) | why a CPU-signalled event wait inside MLX still cost 146-181 us (scheduling delay observed, cause not identified) |
| event gate 146-181 us per gated layer on the 35B (diag35.txt, derived) | |
| fresh-line spin ~69 us per gated layer (F:87, raw not saved); 79 us from the nearest saved raw line (derived) | that 69 us is reproducible from saved data |
| all-hit gate 21.7 us/layer synthetic; in-kernel hit check 3-5 us | F:88's "~5 % on hits" (raw run not saved; its own 93.4 vs 102 line is -8.5 %) |
| round trip ~46 us per slow layer (series35.txt, derived) | |
| in-kernel spinning on device memory: 0.5-1000 ms visibility, stale bytes (gpuwait, coherence) | the cache-level explanation (F: "GPU's L2") |
| fresh lines never stale in practice: 0 timeouts, 0 mismatches in 12,000 + 2,400 + R-55 + 900 wide passes | "regions reused after ~3 tokens are safe": supported by S7B B5, not tested on the engine's ring |
| S7B: re-reads in one buffer stale even with memoryBarrier; first reads / earlier-buffer reads / coherent loads not | that the hazard cannot arise in today's schedule (argued; the 11-04 stale-slot check passed) |
| device-scope post unseen for ms if only a spin follows; ~20-27 us after a real dispatch; 7.4 us at system scope (w1_pickup) | "inside MLX GPU writes reach the CPU only at command-buffer end" (notes and comments; test_vis.py output not saved) |
| coherent one-row spin: release 0.2 us, pickup 2.9 us flat (Swift); engine release 0.004 -> 0.000 ms (S0) | the engine-side pickup under SPIN_COHERENT (no histogram was built; Leap's harness only) |
| MLX's own fence_wait pickup 1.2-2.0 us (w1_pickup arm M) | that `#pragma METAL internals` / `thread_scope_system` stays available on other macOS or GPU generations |
| torn 64-byte sectors under foreign GPU load: 9 calls caught in GS1 (LED:128) | stamps' 10-bit wrap is safe ("far beyond how old a stale word can be") |
| small-array constant binding blocks spins (F:210) | the MLX threshold's exact rule |
| commit thresholds in MLX's code (ref/mlx_device.cpp) | that the 64 MB bump forces a commit on other chip classes (code says yes; not run) |
| bitwise exactness of every adopted gate step on the 125B (token equality on every run; R-55 logits bitwise) | whether the forced commit is still needed with POST_SYS (untested) |

---

## 7. Hardware and software

- **Machine** [measured]: MacBook Pro (Mac17,9), Apple M5 Pro, 18 CPU cores (6 Super + 12 Performance), 20-core GPU,
  Metal 4, 64 GB unified memory (system_profiler, 11-09); internal SSD "APPLE SSD AP2048Z" (F:3); GPU read ~280 GB/s
  (F:3, measured by ~/SpecDecode); SSD cold random 2.77 MB reads 5.8 GB/s at QD1, 14.0 GB/s at QD16 (BM/data/ssd.txt;
  F:164-169). Leap's harnesses report "Apple M5 Pro" (w1_pickup.txt:1, s7b_probe*.txt:1).
- **OS** [measured now / recorded]: macOS 26.5 (25F71) at the time of writing (sw_vers 11-09); ~/Papers/RELEASE.md:85
  states the release was "tested on macOS 26.5 (25F71) only". The OS version at each measurement date (10-17 to 11-08)
  is not logged in the gate data files; `softwareupdate --history` lists no macOS update (only Command Line Tools 26.6
  on 06.07.2026) [measured, inconclusive]. An ext build log of 10-21 warns "building for macOS-26.0, but linking with
  dylib '@rpath/libmlx.dylib' which was built for newer version 26.2" (BM/data/b5_gatefail/b5.log:2).
- **Software** [code-read]: MLX 0.32.2 + mlx-lm 0.31.3 in BM/.venv (the ext's default build `ext/build`, GATED:16);
  MLX 0.32.4 in BM/.venv324 (`ext/build324`) for some experiments; MTPLX 2.12.2 for the 125B's qwen4_exp model (F:7);
  Python 3.12; the ext uses MLX's internal C++ API (Primitive subclasses, `metal::get_command_encoder`, `set_input_array`
  / `set_output_array` / `set_bytes`, EXT:1186-1220, 1825-1853), which is not a stable public interface [assumed risk].
  Swift harnesses (eventwait.swift, spinwait.swift, Leap's w1_pickup.swift / s7b_probe.swift) use Metal directly.
- **Conditions**: "GPU timings were taken under quiet locks, with Chrome active" (F:330); "the handshake cost was
  measured with synthetic layers" (F:329); all paper timings "from one M5 Pro; 'quiet' means no other GPU job" (MAIN
  1158).
- **Other chips: nothing measured.** Records only mention: MLX's per-buffer limits differ by GPU class (code, 3.5);
  a future 24 GB target "the 36 GB machine is an M5 Max" (HO:857) [no gate measurement]; the GPU watchdog: MLX issue
  #3830 reports a ~5 s GPU watchdog (kIOGPUCommandBufferCallbackErrorTimeout) and a fast-synch fence deadlock in a
  distributed pipeline (MAIN reference list ~1418) [external report]; the engine's spin timeouts (~0.2-0.25 s) sit far
  below 5 s [modelled]. No record tests the gate on M1-M4, iPhone / iPad GPUs, or another macOS.

---

## 8. Production numbers on the 125B engine

Metric warning: until 11-07 most reports quote "steady" tok/s (checks >= 2); since 11-07 the headline metric is whole
decode, and the old steady metric overstated by 9-27 % (MAIN section 10; memory note). Numbers below keep their own
run's metric.

- **First end-to-end runs (10-18, 55 % resident, C 282, plain decode, 64 tokens, BM/data/split125.txt)** [measured]:
  per generated token 48 gated calls, 32.7 fast-path hits (68 %), 15.3 slow calls at 699-953 us each; trace
  (median over 37 tokens): loader CPU on slow calls 9.5 ms, GPU gaps after fast layers 17.0 ms, after slow layers
  8.9 ms, token-to-token 8.4 ms; CPU per call 228 us (split125.txt:9-10; LED row 0 at :162). SKIPSPIN (never wait) on
  the 125B gives non-finite logits (split125.txt:2), so the 35B's "nowait" decomposition cannot be repeated there.
- **24 GB line (10-27, b40, shipped v3, BM/data/b40/split.txt)** [measured]: decode 34.1 ms per token = gate wait
  19.2 (56 %) + verify GPU 10.9 (32 %) + ~4 outside; per code verify pass 76.8 ms = gate wait 45.6 + GPU 31.1, SSD busy
  69.6 ms. Each extra verify row ~12 ms gate wait + 5 ms GPU + 2 ms commit (c7bf39f) [measured]. "Gate wait" here is
  the GPU's total wait at gates (SSD reads included), not the handshake alone.
- **GATE16 at 45 GB (10-30, C 192, REF2 + SB, BM/data/r55/r55_screen.txt)** [measured]: warm check + commit
  232.6 -> 117.3 ms (HumanEval/68), 213.7 -> 106.2 ms (prealgebra/1640); 48 loader-served gates per check; no
  timeouts; logits equal across all 4 passes for every window. Live loop (R-55L, 6 prompts): steady 23.7 -> 32.0 tok/s
  (+35 %), whole 256-token runs 21.9 -> 31.0 (+41 %), median cycle -23 %, every token == REF2 (HO:508-512).
- **OOO at 45 GB (11-02, C 224, A-B-A)** [measured]: 45.2 / 45.6 -> 53.9 steady (+18.7 %, whole +15.9 %, all 6 prompts
  +16..+41 %) (HO:1904-1905).
- **Lever 1 at 45 GB (11-04 07:42-07:50, C 224, W1B's 6 prompts x 256 tokens, cumulative traced arms, one pass each,
  every token == REF2-T; BM/data/w1/*.txt, w1.log, stall_w1t_vs_w1tp.txt; BM/data/r57/s0_report.txt)** [measured]:

  | arm | adds | tok/s (steady), median of prompts | cycle ms | check + commit ms | demand SSD ms | gate service on misses ms |
  |---|---|---|---|---|---|---|
  | w1t | default (traced) | 55.5 (59.7) | 103.8 | 94.1 | 53.5 | 54.2 |
  | w1tj | + JOINT | 56.5 (59.9) | 105.8 | 95.0 | 53.7 | 55.7 |
  | w1tk | + keepalive 50 us | 56.9 (61.3) | 101.3 | 91.2 | 50.0 | 52.1 |
  | w1td | + copy split | 58.3 (62.2) | 99.8 | 90.6 | 48.7 | 50.7 |
  | w1tc | + coherent spin | 58.9 (63.1) | 98.9 | 89.1 | 49.0 | 50.7 |
  | w1tp | + system-scope post | 56.9 (61.5) | 99.9 | 88.2 | 47.0 | 48.8 |
  | w1u | default, untraced | 55.4 (60.6) | 112.2 | 99.9 | 58.9 | 59.6 |
  | w1uf | full stack, untraced | 57.8 (63.3) | 97.3 | 86.9 | 47.3 | 49.2 |

  - WED:87-91: geo steady w1u 60.3 -> w1uf 65.7 (+9.0 %); cumulative traced steps JOINT +1.6 %, keepalive +2.7 %, copy
    split +0.5 %, coherent spin +1.9 %, system post -0.6 %; lever 1 "BETWEEN" (fixed part per check 19.2 -> 19.5 ms
    against a <= 10 ms line).
  - S0 split per call with misses (8,540 calls, medians): plan + wake 0.005 ms, pread 0.71-0.77, copy 0.11-0.14, tail
    0.007-0.008, release 0.004 (0.000 with the coherent spin and after); demand reads per call 4. Served base calls:
    zero-hit 0, hits 8,540, no-miss 1,684; "L-1 release -> L seen" for hit calls 1.048 (w1t) -> 0.932 (w1td) -> 0.887
    (w1tc) -> 0.862 ms (w1tp); seen -> release on no-miss calls 0.007 ms (0.034-0.038 with JOINT's pairing wait).
  - Stall fit (w1t vs w1tp): stall per check 59.5 -> 55.4 ms; loader service 66.2 -> 62.8 ms; per stalled layer term
    0.137 -> 0.128 ms; demand-burst intercept 0.251 -> 0.255 ms per burst. The pre-registered pickup line ("cadence a
    minus burst a falls >= 0.03 ms") moves by ~0.013 ms across the whole cumulative stack [modelled, derived here from
    stall_w1t_vs_w1tp.txt; WED does not state the pickup step's own verdict].
- **Adoption A/B (11-04 08:05-08:32, C 160, 14 prompts, A-B-C-C-B-A, all arms 14/14 == REF2-T; BM/data/d45/d45_report.txt)**
  [measured]: lever 1's full stack (JOINT, copy split, coherent spin, system post) on top of the night's levers: +6.4 %
  geo steady (agent +3.2, chat +3.8, code +11.7, math +5.4 %); adopted (0365ada).
- **The stack at C 224 (Fri 11-06 15:40, 12 HumanEval, quiet)** [measured]: per check (medians, checks >= 2) check 68.8 ms,
  gate service 14.0 ms, misses 37, eval wait 7.5 ms (WED:419-421).
- **Coherent one-call reads (11-08, CHS4, C 256, 12 IFEval)** [measured]: E + one-call OOO 3/12 == AR in both runs; + COH
  12/12 in both runs; 37.7 / 38.0 vs 37.8 / 37.9 tok/s whole decode (WED:975-976).
- Not recorded on the engine: a per-call GPU-side pickup histogram under SPIN_COHERENT; a spin-wait distribution per
  layer; the event-vs-spin comparison on the 125B (the event gate predates the 125B runs).

---

## 9. The SSD keepalive (BIGMOE_SSD_KEEPALIVE_US) and related keep-awake

### 9.1 What it does (EXT:1518-1546; built 6c7cc5a, 11-02 22:13; HO:733-738)

- With BIGMOE_SSD_KEEPALIVE_US=n > 0, `ext.start` launches a thread (EXT:2189) that waits for the first served call,
  collects every layer's record file (fd, size in 4 KB blocks), and then loops: if the loader served a call in the last
  50 ms, issue ONE 4 KB `pread` (F_NOCACHE fds; 16 KB-aligned buffer) at a random 4 KB block of a random record file,
  then `usleep(n)`; otherwise sleep 2 ms. Reads only, so exact. Off by default in the ext; the 45 GB preset sets 50
  (r57_screen.py:65-66) [code-read].
- Real spacing at `usleep(n)` (loop = read + sleep slop): n = 25 / 50 / 100 / 200 / 500 -> 145 / 178 / 270 / 398 / 734 us
  (M-28, ~/MoonShots/data/ssdmicro.txt "Descriptive"); n = 50 / 100 / 125 / 150 / 175 / 200 -> 184 / 269 / 298 / 340 /
  361 / 401 us (M-29, ssdmicro2.txt K) [measured]. At n = 50 about 5,600 reads/s (~22 MB/s) (ssdmicro.txt S5) [measured].
  The engine logs ~16,100-16,900 keepalive reads during one prompt's decode (gsm8k/708, 198 tokens, ~3.6-3.7 s at
  53-55 tok/s; data/w1/w1tk..w1uf_224.txt, gsm8k line), i.e. ~4,500/s [measured; rate derived here].
- Related keep-awake on the CPU side: the loader thread itself busy-polls with `__builtin_arm_yield()` at
  user-interactive QoS, so no OS wake-up per layer (LED:47 "L0"); BIGMOE_IDLE_MS adds 200 us naps only after long idle
  (EXT:1523-1527). Pre-spun I/O threads instead of GCD: no gain, GCD's start is 5-23 us (TRIED:206) [measured]. System
  keep-awake (caffeinate / lid_guard) is unrelated to this effect.

### 9.2 Measured effect

- **The penalty, micro (MoonShots M-20 SSDWAKE, Mon 11-02 22:07, quiet lock; records3 bursts of 4 x 9 pieces of 245,760 B
  and single reads; 40 rounds per point; ~/MoonShots/data/ssdwake.txt)** [measured]:
  - Penalty vs back to back: bursts +5.8 % (0.25 ms gap) to +7.5 % (1 ms) to +14.0 % (20 ms); singles +16 % to +33 %.
    Recorded summary: "+50-65 us on the next burst / single read after 0.25-1 ms gaps, +100-120 us after 20 ms".
  - 4 KB every 200 us removes 44 % (pre-registered line 70 %: BETWEEN); 4 KB every 50 us removes all (burst -3 us,
    single -6 us at 1 ms; -11 / +3 at 10 ms); 64 KB every 200 us leaves +40 / +36 us.
  - A busy CPU core with no I/O ("cpuspin") changes nothing (+73 / +56 us at 1 ms).
- **M-28 SSDMICRO (Tue 11-03 18:59, ssdmicro.txt)** [measured]: R0 = 2 base records after an idle gap: +44 to +59 us
  for gaps 0.25-10 ms, no penalty at 0.10 ms (-6 us) (S4); keepalive usleep 25 / 50 / 100 us -> penalty -0.1 / -1.3 /
  -1.9 us, 200 -> +22.0, 500 -> +24.3 (S5); back-to-back calls ~2 % faster with it. Per-call decomposition: fixed 187 us
  of the engine's 283 us: first byte ~119 us, idle wake +44 us, GCD start + return 8 us, ~96 us outside the I/O path.
- **M-29 SSDMICRO2 (11-03 22:33, ssdmicro2.txt K and P)** [measured]: penalty without keepalive +47.9 us; <= 340 us real
  spacing holds it <= 10 us; 361-401 us leaves +12-18 us; a 16 KB read at 412 us +8.6, a 64 KB read at 450 us +4.6 ("bigger
  reads keep it awake longer"). Wake-ahead ping (one 4 KB read T us before the burst, no trickle): T 25 / 50 / 100 / 200 /
  400 us removes 8 / 21 / 22 / 97.5 / 78 %; the ping read itself takes ~142 us after idle.
- **M-29 follow-up (Wed 11-04 09:01, ssdmicro2.txt last block)** [measured]: at the engine's 1.2 / 2.0 ms gaps the penalty is
  +50.1 / +49.0 us; a ping 150 us ahead removes 87 / 94 %, 250 us ahead 91 / 87 %.
- **M-31 HEADPING (11-03 22:45, ssdmicro4.txt)** [measured]: with the keepalive on, pinging a record's head before its full
  read changes it by +0.6..+3.2 us (no map or page warm-up to win); with it off, a ping 100 us ahead saves 42 us (the wake),
  300 us ahead only 5 us (the drive dozes again).
- **M-30 QUEUEMIX (11-03 22:39, ssdmicro3.txt)** [measured]: forecast reads that end >= 0.25 ms before a demand call leave the
  drive dozing (F 2 at 800 us: -2 us), consistent with the same idle threshold.
- **In the engine (125B)**:
  - Gaps exist: per check 42.5 read bursts, SSD idle before them median 0.73 ms; 100 % after >= 0.25 ms, 17 % after >= 1 ms;
    "at ~60 us each, ~2.5 ms per check a keepalive could recover" (stall_w1t_vs_w1tp.txt, w1t block) [measured gaps,
    modelled prize]. The ext counts bursts after >= 0.25 / >= 1 ms idle (EXT:885-888; e.g. w1t gsm8k/708: 2,482 bursts,
    1,399 after >= 0.25 ms, 199 after >= 1 ms).
  - A-B-A-B at C 160, untraced (ka0 / ka1 / ka0r / ka1r, 11-04 00:36-00:41, build_la, W1B's 6 prompts; BM/data/s160/*.txt,
    c45.log:7-15): NIGHT_1104.md:37-42 reports "about -3 ms per check on clean prompts" (-2..-4 ms on the 4 clean prompts;
    the mean -10.9 ms came from one-off spikes in two A arms), tok/s geo +3.9 % with ~10 % noise [measured]. The arms'
    own median lines (median of prompts) read demand SSD 93.0 / 96.2 / 97.1 / 95.3 ms per check (ka0 / ka1 / ka0r / ka1r),
    i.e. no visible difference at that level [measured].
  - Traced cumulative arm (11-04, C 224): + keepalive on top of JOINT +2.7 % steady (single pass); S0 pread median per call
    0.768 -> 0.709 ms (w1tj -> w1tk) (WED:90-91; s0_report.txt) [measured].
  - Adopted in the exact default inside the +12.1 % bundle (chat switch, keepalive 50 us, volatile tier, K-83) (d45_report.txt;
    MAIN 8.1). "No isolated A/B of the keepalive exists on the final engine" (MAIN 8.1, line 640); the agent
    leave-one-out's "-keepalive" arm was cut (HO:1460; WED:118-124) [measured absence].
  - MAIN 8.1 and MAIN's lever table quote "about -3 ms per check" (BM/docs/NIGHT_1104.md:37).

### 9.3 What is recorded about WHY

- MAIN 8.1 "Why it works [G]" (a stated reading): "The penalty depends on the idle gap and not on CPU load, so it sits in the
  SSD path, consistent with an idle timer in the drive or its link. A read every 270 us or sooner keeps that timer from
  expiring; a single read about 200 us before a burst pays the wake early, off the critical path."
- What supports it [measured]: no penalty at 0.10 ms gaps, full penalty from 0.25 ms (M-28 S4); penalty grows with the gap
  (+50-65 us at <= 1 ms, +100-120 us at 20 ms, M-20); a busy CPU core does not help (M-20 cpuspin); larger keepalive reads
  stretch the safe spacing (M-29 K); a 4 KB ping must complete before the burst (T 100 us too late, M-29 P); head pings do
  not help when awake (M-31); forecasts ending >= 0.25 ms early let the drive doze again (M-30).
- What is NOT tested [assumed]: which component idles (the NVMe controller's power state, the PCIe link's low-power state,
  the SoC's storage controller, or the kernel's I/O path); the actual timer value (bounded only to (0.10, 0.25] ms of idle by
  M-28 S4, or ~200-265 us of idle between keepalive reads given ~75-140 us per read); whether a deeper state exists past
  20 ms; the energy cost of ~5,600 reads/s; dependence on power source, Low Power Mode, temperature or OS version; whether
  other Macs or external drives behave alike. The prior-art framing (NVMe idle power states with entry / exit latencies;
  keeping drives awake; anticipatory wake) is MAIN 8.1's literature reading, not a test here.

### 9.4 Cheap measurements that need no model (micro-benchmarks; the harness exists in ~/MoonShots ssdmicro*)

1. Fine idle-gap sweep 0.10-0.30 ms in 10 us steps, back-to-back control interleaved: locates the threshold and whether the
   penalty is a step or a ramp. (Existing S4 has only 0.10 and 0.25.)
2. Long-gap sweep 20 ms, 50 ms, 0.1 s, 1 s, 10 s: is there a second, deeper state (S4 / M-20 stop at 10-20 ms)?
3. Keepalive operation type at fixed spacing: 4 KB F_NOCACHE read (today) vs a page-cache HIT on the same block (should NOT
   work if the effect is in the drive path: a clean control) vs a 512 B / 16 KB read vs a read of the same LBA each time vs
   far LBAs. Separates "any host I/O syscall" from "a real device command".
4. Spacing sweep with deadline timing (spin to a target instead of usleep) at 150-400 us real spacing in 25 us steps: maps
   the hold-off curve exactly (today's points are 184 / 269 / 298 / 340 / 361 / 401 us with usleep slop).
5. Penalty vs queue depth: a 1-read vs 8- / 16-read burst after the same gap (does a deep burst pay it once?).
6. Energy and heat: package / SSD-related power with keepalive on vs off at idle-gap patterns like the engine's
   (`powermetrics` needs root; otherwise battery drain over a fixed duty cycle).
7. Conditions: AC vs battery, Low Power Mode, lid open vs closed with external power, after a thermal soak; one run each.
8. GPU interference: a GPU bandwidth kernel (e.g. the 64 MB stream kernel in Leap's w1_pickup.swift) with the keepalive on
   vs off, to bound its cost on memory bandwidth (~22 MB/s of reads; expected negligible, not measured).
9. The wake-ahead ping (one read ~200 us before a burst) vs the trickle in the same harness at the engine's measured gap
   distribution (median 0.73 ms), counting reads per second: the ping needs ~46 reads per check instead of ~300 (M-29 note).
10. Another Mac (the M5 Max named in HO:857) or an external NVMe enclosure: same harness, to see if the penalty is
   Apple-internal-SSD specific.

---

## Appendix: commits that carry the gate (from `git -C ~/BigMoE log --oneline`, dated by `git log --date`)

- 7c9f049 2026-10-17 20:44 Exact sync-free expert gate for MLX: native loader, per-layer CPU-signalled events
- be479ce 2026-10-17 21:49 Phase 2: exact SSD expert streaming measured on 35B (bit-identical), spin gate with pass-ahead residency
- 786c141 2026-10-17 22:34 Mid-token prefetch v1 measured (+6..14%, exact); v2 additive table publish
- 5573413 2026-10-18 01:30 Slow-path decomposition: round trip ~45 us, rest is SSD latency; chunked reads don't help; release-first loader order
- 945cd21 2026-10-18 16:48 PF1 staging, miss split counters, BIGMOE_MAXK, MLX 0.32.4 build
- 6efbab5 2026-10-18 20:31 PS: MoonShots JIT prefetch scheduler merged (+6% spec, exact)
- bc1537a 2026-10-18 21:31 ... 80-id gate (SpinWait stride fix pending test)
- 46db7d3 2026-10-18 21:37 G2: 80-id gate works
- 6b845b7 2026-10-18 22:40 v2-3: pair independence proven; prototype race (shared region ring)
- 69e1c47 2026-10-19 07:25 ... spin back-off knobs (BIGMOE_SPIN_G1/G2/KNEE/PAIRS)
- 5b9842e 2026-10-21 08:56 ... records3 rebuild serial per job for wide miss lists (spin-gate timeouts)
- af1af44 2026-10-21 12:49 Adopt out-of-order experts ... post_ids partial-hit row, Gate.post/spin
- d0c2154 2026-10-21 14:34 OOO1 adopted (bitwise, +4-6% spec)
- 41bb474 2026-10-25 06:15 Gate handoff stamps (MoonShots patches/gate_stamps ported to v3)
- fc20ff6 2026-10-25 09:58 GS1: gate stamps verified (quiet shas unchanged; under 50% GPU load 9 torn calls caught, shas = quiet)
- 5918b42 2026-10-25 10:02 gated.new_pass waits while ~/.local/share/quiet/PAUSE exists (... spin-gate timeouts)
- 359c028 2026-10-25 13:12 JOINT (port of Fit12 B1 v2 to v3 plain rows) ...; Gate.post(after=)
- 4be3b31 2026-10-30 16:21 R-55 GATE16: the sync-free gate carries up to 185 ids
- d7106c0 2026-10-30 16:42 R-55 PASS: the 16-row check on the sync-free gate halves the exact 45 GB check at C 192, bit-equal
- 6c64148 2026-10-30 17:25 R-57: R-55's gate in the live exact 45 GB loop (+48 % whole-run, cycle -21 %)
- eda7586 2026-10-30 18:00 R-58G PASS + R-55L: the gate is +35 % steady-state tok/s in the exact live loop
- 8e2b22e 2026-11-01 04:03 R-55 GATE16 ADOPTED: default BIGMOE_MAXK 160
- 6c7cc5a 2026-11-02 22:13 M-20 SSDWAKE: an SSD keepalive thread in the ext, its A/B, and burst gaps
- 42939ed 2026-11-03 19:45 Lever 1: Leap's W1-PICKUP (coherent one-row spin + system-scope post) joins Wed's build
- 5565a56 2026-11-03 22:27 Lever 1 build: Leap's W1-PICKUP patch + S0 instrumentation
- a001cd1 2026-11-03 22:34 Lever 1 plan: Leap's S7B-PROBE (stale GPU-cached slot reads inside one command buffer)
- 0ea5b8b 2026-11-03 22:45 The stale-slot check, first in tonight's chain
- da8e7a6 2026-11-04 07:52 tier: TIER_COMBINE calls on the sync path take the stock block (the MAXK 80 crash fix)
- 0365ada 2026-11-04 08:35 ADOPT the combined exact default as R57_PRESET=45 (item 5: ~+19 %)
- c387c97 2026-11-04 15:35 ... QOM flags only the loader's copy of the ids
- 155d5c7 2026-11-04 17:10 One-call OOO (QOM-aware, Leap 00961fc) in tonight's QOM arms
- 97e81b8 2026-11-08 07:49 chat_env.txt: M1-B + REF2_NAX_COH=1 (Leap: the one-call reads stale slots without it)
- 10fa1b4 2026-11-08 07:45 CHS4 doc: COH makes the one-call deterministic + exact (C4 solved)

Prior-art tags already in the records (no new search here): TRIED:507 and :501 tag the spin / R-55 gate "adapted
(anemll-flash-mlx / SlotStream sync-free slot gate)"; MAIN's T76 row (~line 600): "not new: MLX's own fast-sync fence has a
GPU kernel spin on a counter the CPU writes (MLX issue #3830); ours: fresh lines per pass, per-layer events and residency
published a pass ahead, between the SSD loader and the expert kernel".
