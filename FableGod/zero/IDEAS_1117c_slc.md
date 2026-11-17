# IDEAS_1117c: assumptions 7 + 8 on the M5 Pro: what the chip has that the frame does not use (fable-zero, Tue 17 Nov 11:1x-12:30, desk)

The question (BREAK98c, lens ZERO): the weights as they are, read at ~290 GB/s by the GPU (7); the experts computed the way
today's kernels do it, tiles 2/3 padding, compute-bound (8). What on this chip is unused or misused: bytes, units, caches,
the SLC, format? Numbers from measured parts; the calc is price_1117c_slc.py beside this file. Nothing ran on the GPU; one
one-core CPU sweep ran (1 min, beside EvalSuite's run: noisy, reported as such). [K] measured, [P] priced, [G] guess.

## 0. The inventory: what each resource does today and what it could add (the lens's answer in one table)
| resource | measured | used by the 32-row check (tue1, 56 ms) | what it can still give |
|---|---|---|---|
| DRAM | 280-300 GB/s distinct bytes [K]; a second unit beside the GPU slows the GPU (bw4 -22.8 %) [K] | 6.74 GB in 56 ms = 120 GB/s average; experts 4.60 GB in 26.3 = 175 GB/s (38 % idle = 9.9 ms); dense 2.13 GB in 14.8 = 144 GB/s | nothing beyond 280: the chip has no second bandwidth (SSD ~7 GB/s = 2.5 %; the ANE and CPU draw from the same bus) |
| the SLC (system-level cache; size unpublished) | a GPU re-read of 6.6 MB: 38.9 -> 13.0-18.7 us (x2.1-3.0) [K, hc_bw_probe / l3_slc]; at 8.85 MB x1.6, at 14.75 MB x1.45 [K] (the gain shrinks with size: the knee is somewhere in 10-40 MB; a one-core CPU memcpy sweep today showed 165 GB/s at 1 MB, ~106 at 20-40 MB, ~80 at 64-128 MB, noisy) | nothing: every check byte is read once; a GPU touch kernel beside the check was NOT hidden (+25.4 us per 6.6 MB window) [K] | the ONLY extra bandwidth on the chip: ~x2.5 on re-reads up to its size. Idea 1. |
| GPU matrix units | 14.0 T MAC/s bf16, x1.55-1.73 int8 [K] | dense at 32 rows ~10 % of peak (byte-bound); experts: 16-row tiles at 5.2 pairs = 2/3 padding; the MMA pass over every loaded weight is 16 slots per weight = ~11 ms per check at peak, 6.1 ms exposed [K Breakthrough's ablation: MMA 128 us of 572 per call] | nothing new: rows in the padding is BREAK98b (S2: up to ~64 rows) |
| GPU ALU / decode | decode 65 us + code loads 61 us per call [K]; K90 LUT decode slower than ALU decode [K] | 3 + 3 ms per check, serialized with the MMA and the bytes ("costs add") | the prize is the SERIALIZATION: bytes 16.4 + MMA 6.1 + decode 3.1 + loads 2.9 = 28.5 ~ the measured 26.3; perfect overlap = 16.4. Breakthrough's rewrite is at 0.82 (21.6 ms) [K]; no desk mechanism of mine beats their 6 arms and MoonShots' 13 forms |
| Neural Engine | 0.295 GB int8 package, 16 rows, 10 ms (~0.5 T MAC/s) [K]; GPU->ANE->GPU 78-147 us per layer [K KILL] | 0 | not a row engine (BREAK98b §4); no cache of its own worth a weight |
| CPU (6P + 12E, AMX / NEON) | 62 ms for the ANE's 12.4 ms call [K]; 70-150 us per mid-pass sync [K]; one core streams ~80-160 GB/s of memcpy [K today] | ~28 ms of command build + encode per check, hidden (GPU-bound 96-97 %) | a PREFETCHER: it can touch bytes into the SLC without a GPU dispatch and without a sync. Idea 1's enabling trick. |
| format (4-bit 2.784 MB / MSB-3 2.064 MB per expert, packed scales ~0.4-0.5 bit) | entropy of the codes 2.72 of 3 bits [K M-27]; GPU rANS decode 5.2 ms > the bytes it saves [K]; super-block scales ~-5 % [napkin]; 2-bit tails over the code KLD gate [K TAILS2]; shared bases / deltas hold 0.2-0.36 % [K] | | at a fixed bit width the bytes are at their floor (+-5 %); the lever left is WHICH rows' experts get which width. Idea 2. |

Read: assumption 7's two halves split. "Read at ~290 GB/s" is a wall (no unit, no compression, no cache changes DRAM's
rate); "by the GPU, once" is not: the SLC re-serves bytes at ~x2.5 and the CPU can fill it for free. Assumption 8's
"compute-bound" is the wrong word: the expert phase is SERIALIZED (bytes, MMA, decode, loads add up), each term alone is
under the byte floor; that prize (26.3 -> 16.4) is the kernel work already in flight (Breakthrough / Leap), not a desk idea.

## 1. Idea 1: THE SLC AS THE CHIP'S SECOND MEMORY, FILLED BY THE CPU (breaks 7: "read at 290 GB/s by the GPU")
1. **Assumption broken.** Every weight byte comes from DRAM at <= 280 GB/s. On this chip a byte re-read within the SLC's
   reach comes at ~x2.5 [K, 6.6 MB], and the SLC is memory-side: ANY agent's read allocates it. Today nothing is re-read
   inside the check, and the draft's re-reads (the head's ~85 MB per pop, 15 pops per tree-16) overflow it.
2. **Why the measured walls do not apply.** (i) "Keep weights in the SLC" (KILL 10-18) and "Rnd16 SLC warming" (KILL,
   Scout: "DRAM-bound decode leaves little idle bandwidth") were napkin KILLs under the premise that the expert phase
   saturates DRAM. It does not: tue1 measures 175 GB/s over 26.3 ms, 38 % idle = 9.9 ms = 58 MB of free reads per layer
   (30 MB after the rewrite's x0.82). (ii) HC-SLC-STAGING (KILL in substance 15 Nov) used a GPU touch kernel, which cost
   its own GPU time (+25.4 us per 6.6 MB window) on a 96-97 % busy GPU. The CPU is idle half the check and needs no
   dispatch and no sync to read memory: a CPU thread touching the next layer's dense weights (44 MB per layer; the GDN
   qkv|z projection alone is 21 MB) during the expert phase takes nothing from the GPU and only the DRAM slack.
3. **Number [P].** (a) Check: X MB per layer prefetched = X x 48 per check served at x2.5: X = 16 -> 0.77 GB, -1.6 ms at
   the dense byte floor (-0.8 at today's 144 GB/s dense); X = 24 -> -2.5 / -1.3; X = 44 (a whole layer's dense) ->
   -4.5 / -2.3. The window is the expert phase's slack (58 MB per layer today, 30 after the rewrite, 0 at the byte
   floor: this lever and the one-call's perfection are alternatives for the same DRAM slots, so it is worth most NOW and
   on the 16-row check). (b) Draft: the pop is 0.55 fixed + 0.56 bytes for ~116 MB (206 GB/s) [K MoonShots]; 85 MB of
   it (dense 73 + head 11.8 at Q8H32) is the SAME every pop; if the SLC holds F MB of it, bytes fall by F/116 x 0.6:
   F = 24 -> -1.0 ms per tree-16 block; F = 48 -> -2.1; F = 85 -> -3.7 (pop 0.86). No CPU needed here: pop 1 warms, pops
   2-15 hit; after each check (6.7 GB streamed through) the CPU re-touches the head during the check's last layers.
   (c) Together at kept 17: tue1 222 -> 227 (SLC ~16 MB) .. 251 (SLC >= 85 MB); Saturday's 292 -> 301 .. 345. The SLC's
   SIZE is the whole question, and it also decides a design: a drafter whose per-pop re-read set is <= the SLC gets its
   byte term x0.4; PLAN900's "<= 26 MB per level" target is then an SLC target, not only a DRAM one.
4. **The decisive test (<= 10 min, Metal + C, Leap's or MoonShots' slot; I cannot run the GPU).** One microbench:
   (i) GPU read bandwidth vs buffer size, 2 -> 256 MB, re-read 10x: the knee = the SLC's effective size for GPU reads
   (and whether 85 MB of head could ever hit); (ii) a 21 MB buffer read by the GPU cold / after a CPU thread touched it
   / after the GPU touched it, with a 100 MB GPU stream in between to mimic the expert phase: PASS if CPU-touched ~=
   GPU-touched (~x2 faster than cold) -> the CPU prefetcher works and the window math above holds; KILL if CPU-touched
   == cold (the SLC does not share CPU-allocated lines with the GPU, or a QoS partition keeps them apart). Exact: yes,
   scheduling only, bits equal.
5. **Quality risk.** None (bit-exact). Build risk: a CPU thread per layer timed to the GPU's progress (MLX's command
   buffer completion handlers, or a free-running thread one layer ahead of a GPU-written counter). Kills it: test (ii)
   KILL, or a knee <= 8 MB (then -0.4 ms per check and nothing for the draft).

## 2. Idea 2: EXPERT BYTES BY THE ROW'S SURVIVAL, NOT BY ITS RANK (breaks 7: "the weights are what they are")
1. **Assumption broken.** The check reads each expert in the union at the width its best (row, expert) pair asks for
   (C6: 15.27 experts at 4-bit + 30.73 at MSB-3 per layer, the tier by routing rank and gate mass, lambda 1.8e-2). Rows are
   treated as equal. In a tree-32 they are not: the head's path_p runs from ~0.9 at row 1 to < 0.05 in the deep bands,
   and the check's bytes for an expert used only by rows with path_p < 0.2 are > 80 % wasted in expectation (those
   rows die at the check or above it). The weight format is per expert; the right width is a function of the summed
   survival-weighted gate mass of the rows using it: sum_rows path_p(row) x gate(row, expert).
2. **Why the measured walls do not apply.** TAILS2 / B7-2bit / LEVER6 killed 2-bit widths applied to ALL rows (the
   code KLD gate 1.10x; sure rows x0.712 bytes at 1.5-10 % flips). R-30 UNIQ (PARKED, quality never measured) dropped
   single-use LOW-RANK experts regardless of the row. Neither weighted by the row's chance of being kept. Here the
   near-sure rows (path_p >= 0.5: the spine's first ~8-10 rows) keep C6's exact widths, so the expected quality cost is
   the sum over rows of path_p x (that row's flip rate at the lower width): a row at path_p 0.1 that flips 5 % costs
   0.5 % of one kept token. Kept per check falls by [G] 0.05-0.15 (0.3-0.9 %) at -0.5..-1.1 GB per check.
3. **Number [P/G].** If experts used only by rows with path_p < 0.2 hold 30-40 % of the union's bytes (the deep half of
   a 32-row tree; the real share is one CPU pass over Lunatic's 127-node dumps): MSB-2 for them -0.46..-0.61 GB = -1.6..
   -2.2 ms at the floor (-2.6..-3.5 at tue1's rate); dropping their SINGLE-USE experts (the row's gate renormalized,
   R-30's form) -0.83..-1.10 GB = -3.0..-3.9 ms at the floor (-4.7..-6.3 at tue1's). At Saturday's 48 ms check that is
   x1.03-1.08 tok/s before the kept loss; the net is x1.02-1.07.
4. **The decisive test.** Bytes (10 min, CPU, the t30b 127-node dumps + the check's route files): per check, each
   expert's width = f(max over its rows of path_p, gate); bytes per check vs C6 at thresholds 0.1 / 0.2 / 0.3. PASS >=
   -15 % bytes at 0.2. Quality is NOT a 10-min test: it is one EvalSuite lossy-gate pass with the width rule as a C7 arm
   (the same gate C6 passed), with the kept per check logged. KILL if kept per check falls > 1.5 % at -15 % bytes, or
   the gate fails.
5. **Quality risk.** Lossy-line only; medium (2-bit rebuilds are the weakest format we have: TAILS2's KLD); bounded by
   path_p: the rule never touches the rows likely to be kept. Kills it: the byte share at path_p < 0.2 is < 15 %
   (then the deep rows are too few to matter), or the lossy gate.

## 3. What I would not propose (priced, dead on this chip)
- Entropy-coded weights at DRAM rates: -9..-13 % bytes, the decode costs more than it saves on the GPU (5.2 > 2.1 ms)
  and needs ~100 CPU cores at 184 GB/s of output. 2-bit / Lloyd-Max codebooks: <= 8 % lower MSE than uniform at the same
  width, nothing in bytes. Texture (ASTC) decode: 3-bit-affine error, only helps an ALU-bound reader. Super-block
  scales: ~-5 %, worth a line in C7, not an idea.
- Table-lookup GEMM (T-MAC style) instead of the matrix unit: 2 lane-ops per weight per ROW (10 at 5.2 rows) vs 16
  matrix slots per weight at 14 T: ~0.53 vs 0.23 ms per call. The padded MMA still wins.
- Pre-decoded int8 experts (zero decode): x2 bytes, 33 ms of DRAM per check. A second GPU stream for overlap: kappa 0.93.
  The ANE / CPU as expert engines: 10-50x short and on the same bus.

## 4. One line
Assumption 7 splits: DRAM's 280 GB/s is a wall, "by the GPU, once" is not: the SLC re-serves bytes at ~x2.5 [K] and the CPU
can fill it with no dispatch and no sync inside the expert phase's measured DRAM slack (38 % idle = 58 MB per layer);
priced -1..-4.5 ms per check and -1..-3.7 ms per tree-16 draft block (x1.03-1.18), all on the SLC's unknown size; decisive
10-min Metal microbench: GPU re-read bandwidth vs buffer size, and a 21 MB buffer read cold / CPU-touched / GPU-touched.
Second: expert widths by the row's survival (path_p x gate), -0.5..-1.1 GB per check [P/G], bytes screen on the 127-node
dumps, quality by one EvalSuite arm.

> (a quote from the project owner, kept private)
The hardwired chip wins twice: zero weight bytes per token AND zero dispatch per layer. On this Mac the two are separate levers.
- **(i) On-chip capacity vs the drafter.** GPU-resident storage (20 cores' registers + threadgroup memory) is a few MB; the
  SLC's size is unpublished, its re-read gain is x2.5 at 6.6 MB, x1.6 at 8.85, x1.45 at 14.75 [K], so its usable reach is
  ~10-40 MB; DRAM 280 GB/s. The drafter's pop reads 163-178 MB (bf16 t30b) / 116 (Q8H32) / 95 (4-bit) [K]; of that the
  same bytes every pop are dense 73 + screen 11.8 = 85 MB at Q8, ~64 at 4-bit, ~38 at a 2-bit (TRL / codebook) dense; the 10
  routed experts (30.7 MB) change per pop and always come from DRAM. So a drafter SHAPED for residency (2-3-bit dense, <=
  ~40 MB re-read set) could hold its dense part in the SLC only if the knee is >= ~40 MB (the test in §1). Price [P]: bytes
  per pop 0.56 -> ~0.15-0.25 ms; the pop is still 0.55 ms of fixed dispatch cost, the second thing the hardwired chip has
  none of: that is the fused level kernel (PLAN900 bet 7, f 0.056 ms per level, multi-day). Both together: pop ~0.2-0.3 ms,
  tree-16 block ~3-4 ms vs 16.6 today / 8.8 in the Saturday ladder: cycle 58.3 -> ~53 ms at kept 17, 292 -> ~320 tok/s (+10 %).
  Verdict: real, exact, worth ~10 %; it does not move 550 because the check is 48 of the 58 ms.
- **(ii) Extreme compression with in-register decode on the CHECK.** Byte floor: experts 4.60 GB at C6's ~3.3-bit mix ->
  ~2.9 GB at ~2.1 bits (-5.9 ms), dense 2.13 -> ~1.1 GB (-3.6): floor 25.2 -> ~15.7 ms. Decode: the measured trellis decode
  (QTIP 3INST, no table) runs the expert call 1.15-1.21x SLOWER than today's 3-bit kernel [K B10(b)-3INST], i.e. +85-120 us
  per 572 us call against -120 us of bytes: net ~0 on today's serialized one-call, -9.5 ms only at a kernel that hides
  decode under bytes (the kernel nobody has yet). Quality is the kill: 2-bit trellis experts held-out error 2.49x MSB-3 (line
  1.5x) [K B10 / B8 KILL]; 2-bit tails fail the code KLD gate [K TAILS2]; 3-bit dense V4 FAILED. The one compression PASS
  is TRL-3 (3.07 bits, 0.5x MSB-3's error at -13 % bytes, PARKED: needs a GPU encoder, CPU ~55 h): on the MSB-3 two thirds
  of the union it is -0.4 GB = -1.4 ms at the floor, ~0 today. Verdict: at quality on this model, compression is spent;
  the next bit costs more quality than the lossy gate allows and more ALU than the bytes it saves. Does not move 550.
- **(iii) Kernels specialised to this exact model (a compiled check).** Today's check is 4,022 dispatches, 434 command
  buffers, ~28 ms of CPU build + encode (hidden) and 4.4-8.4 ms of GPU fixed cost [K Leap dispatch diet]; every class runs
  at 45-60 % of its byte rate because the kernels are generic and serialized. A compiled check (per-sub-block fused kernels
  in this model's exact shapes, weights stored in the tiles' native layout, one command buffer replayed) converges on the
  byte floor at 85-90 %: 25.2 -> ~28-30 ms at 32 rows; Leap prices the road at A + B (weeks) then D or E (4-6 weeks). At a
  29 ms check, a 3 ms draft and 1.5 overhead: kept 17 -> 507 tok/s; kept 19 -> 567; K64's 20.8 (BREAK98b) -> 620.
  Verdict: THIS is the lever that moves 550, and it is engineering time, not an idea: the hardwired chip's second win
  (no dispatch, no glue) is reachable in software to within ~15 % of the byte floor; its first win (no bytes) is not.
- **Plainly:** of the three, only (iii) moves 550, and only together with a 48-64-row tree (kept >= 19) and a <= 4 ms draft;
  (i) adds ~10 % on top and is exact; (ii) is dead at quality on this model. A software HC1 for the DRAFTER is a
  2-3-bit, SLC-sized, fused-kernel head (~3-4 ms per block); for the CHECK the 6.7 GB per block never fit on any chip's
  cache and the floor stays 25 ms at 32 rows.

## 6. Idea 2's BYTES screen, run (CPU, 11:4x; screen_1117c_pathp.py beside this file) [K on the proxies below]
Routing: Leap's r32_routes.npz (628 real 16-row checks of the 12-layer slice, expert ids + softmax gates), cut into
16- and 32-row windows as treek_union does; path_p per row: one real HE+ 127-node t30b tree per window, nodes 0..K-1 in
best-first order (node 15 median path_p 0.78, node 31 median 0.31: the head is confident on code, the deep tail is
thinner than my §2 guess). C6 = ranks 0-1 at 4-bit, 2-9 at MSB-3, gate < 1.8e-2 dropped; width per expert = max over
its uses. Rule: rows with path_p < tau are "demoted": their rank 0-1 uses count as MSB-3, rank 2-9 as MSB-2 (1.45 MB);
"drop" also skips an expert used ONLY by one demoted row at rank 2-9 (R-30 UNIQ's form, restricted). Slice x4 = 48 layers.
The proxy's C6 baseline is 5.36 GB at 32 rows vs Leap's 4.60 (x0.86: chain windows, slice layers); the shares carry.

| K | rule | tau | C6 -> new (proxy GB) | saving | scaled to Leap's C6 | rows demoted | experts touched |
|---|---|---|---|---|---|---|---|
| 32 | demote (MSB-2) | 0.2 | 5.36 -> 5.02 | -6.3 % | -0.29 GB (-1.0 ms at the floor) | 24.1 % | 22.4 % demoted |
| 32 | demote | 0.5 | 5.36 -> 4.81 | -10.2 % | -0.47 GB | 38.0 % | 35.6 % |
| 32 | drop single-use | 0.1 | 5.36 -> 4.72 | -11.8 % | -0.54 GB (-1.9 ms) | 14.6 % | 12.1 % dropped |
| 32 | drop single-use | 0.2 | 5.36 -> 4.30 | -19.8 % | **-0.91 GB (-3.3 ms at the floor, -5.2 at tue1's rate)** | 24.1 % | 20.3 % dropped |
| 32 | drop single-use | 0.3 | 5.36 -> 4.04 | -24.5 % | -1.13 GB (-4.0 ms) | 29.8 % | 25.0 % |
| 16 | demote | 0.2 | 2.89 -> 2.80 | -3.1 % | -0.08 GB | 11.6 % | 10.7 % |
| 16 | drop single-use | 0.2 | 2.89 -> 2.60 | -10.0 % | -0.25 GB (-0.9 ms) | 11.6 % | 10.1 % |
| 16 | drop single-use | 0.3 | 2.89 -> 2.47 | -14.4 % | -0.36 GB | 16.8 % | 14.6 % |

Read: the width-only form (MSB-2 for improbable rows) is under the 0.5 GB bar at any sane tau: a demoted row's experts
are mostly shared with sure rows, which hold the width up. The bytes are in the SINGLE-USE experts of improbable rows:
at tau 0.2 they are 20 % of the union at 32 rows (-0.91 GB on Leap's C6), 10 % at 16 rows (-0.25 GB). That is R-30 UNIQ
(PARKED, quality never measured) with the row's survival as the gate: the dropped expert serves only a row whose
chance of being kept is < 0.2, and the row then runs on 9 of its 10 experts with the gate renormalized (Fable B's
9-of-10 finding is the quality prior). Expected kept cost: 24 % of rows x mean path_p ~0.1 x the one-expert flip rate
(unknown; if 5-10 %: 0.04-0.08 kept of 17, 0.2-0.5 %). So the quality arm is worth asking at 32 rows (drop, tau 0.2);
at 16 rows the saving is too small to pay for a gate pass. Caveats: chain windows stand in for tree rows (siblings share
more routing than consecutive tokens, so single-use shares may be lower in real trees); the slice's 12 layers scaled x4.
