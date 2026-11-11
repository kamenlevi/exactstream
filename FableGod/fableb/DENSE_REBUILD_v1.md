# DENSE_REBUILD: fewer, larger dense kernels for the 45 GB FAST check on M5 (Fable B, v1, Fri 6 Nov 2026 07:50 EET)

Marks: [K] measured by its owner, [P] derived from measured numbers, [G] my estimate. Gains are already halved where
they are [G]. Nothing here ran on a GPU. Files: `kernels/` (drafts, CPU checks, pricing).

## 0. Verdict

- **The design: 6-9 dependent dispatches per layer instead of ~20.** A layer is re-cut along its independent axes (key
  head, channel slab), and every sum over heads or streams is deferred into the kernel that consumes it.
- **Expected dense time: 16-23 ms per check [G on P], centre ~19, against 35.8 today [K].** The byte floor is 10.7 ms at
  250 GB/s [P]. So the dense part falls by 13-20 ms.
- **12-15 ms is the top of the range, not the centre.** It needs a sync point at <= ~8 us AND >= 220 GB/s in flight in
  every kernel. Today's best quiet kernels reach 74-88 % of the floor [K DENSE-N, L1-29]. I price 12-15 at ~1 in 5.
- **First layer type drafted: the GDN layer (36 of 48 layers, ~20 of the 35.8 ms with its reads).** Two new kernels:
  `kernels/gdn_w.py` (W) and `kernels/hc_d.py` (D). Both compile offline; the weight stream of W is checked on CPU.
- **Step 0 part B decides two numbers** (section 6). `kernels/price.py --fit` re-prices everything from them.

## 1. What the records say [K], and the reading [P]

| piece | calls | MB per call | ms today | floor at 250 | GB/s today |
|---|---|---|---|---|---|
| GDN | 36 | 38.9 | 13.2 | 5.6 | 106 |
| HC reads (HC8) | 96 | 7.05 | 8.6 | 2.7 | 79 |
| attention | 12 | 32.2 | 6.4 | 1.55 | 60 |
| router + shared + tail + HC write | 48 | 4.2 | 7.6 | 0.8 | 26 |
| sum | | 2.66 GB | 35.8 | 10.65 | 74 |

- Model per dependent dispatch: `t = t_sync + bytes / B`. Fitting today's pieces with today's dispatch counts [P counts]:
  B 200-250 GB/s goes with t_sync 17-21 us for the wide kernels (GDN, HC) and 6-10 us for the small ones.
  B 110 goes with t_sync 1-9 us. Part B separates the two.
- Three records point at the first reading (a fixed cost per sync point, good streaming in between):
  - the HC write is one kernel, moves ~0.7 MB and costs 17 us [K];
  - an HC read is 3 dispatches for 7 MB and costs 90 us, a GDN layer ~10 for 39 MB and 367 us: both fit ~19 us each;
  - DENSE-N reached 74 % of the floor on independent calls [K], but the same GDN layer in a dependent chain stayed at
    0.37 ms [K GDNB-2]: the projections lose their gain when nothing overlaps their start and end.
- **The cost is per global sync point, not per dispatch.** MoonShots' M5 KILL shows it: with grid barriers in place of
  dispatches the op time stayed 81.7-84.7 vs 81.5 us [K]. A barrier and a dispatch both make every thread wait for the
  slowest one and then ramp up again. So removing dispatches while keeping the waits buys nothing. Removing the waits
  does.

## 2. Where a layer truly has to wait

A value is a sync point only if every thread of the next step needs it and every thread of the last step feeds it.

| step | needs | sync point? |
|---|---|---|
| HC down mix (K 10,240 -> 324 values per row) | all streams | yes: the 324-vector |
| HC up + mix (-> x, 2,560 per row) | the 324-vector | yes: x |
| GDN in_proj -> conv -> recurrence -> gated norm | x only, per key head | **no: 16 independent pipelines** |
| GDN out_proj (K 6,144 -> 2,560) | all heads | folds into the next down mix, per channel slab |
| pending write + next down mix | out_proj's output, per channel | same kernel as out_proj |
| router + shared gate/up + shared-expert gate | x only | no: one kernel, row-parallel |
| router top-10 | 512 logits | yes, tiny |
| shared down + expert sum | 640 GLU values, expert output | folds into the next down mix |

So a layer needs 6 waits (D, G, U on each side) plus the router pick, not ~20.

## 3. The kernel set

| kernel | what fuses into it | grid | MB | floor us | expected us [G] |
|---|---|---|---|---|---|
| **W_gdn** | in_proj qkv, z, a, b; gates; conv + silu + l2norm; recurrence; gated norm; conv and state writes | 16 x 1,024 | 30.0 | 120 | 150-190 |
| **D** (MLP side) | out_proj (or o_proj); pending write; A = h' w; sum h'^2; down mix + inject partials | 20 x 1,024 | 12.4 | 50 | 70-90 |
| G | partial sum, 1/rms, silu, inject sigmoid (HC8-1P K2, 80 partials) | as K2 | 1.8 | 7 | 10-20 |
| U | up mix + stream gate + mean (HC8-1P K3, unchanged) | as K3 | 3.5 | 14 | 25-35 |
| W_mlp | router logits (8-bit), shared gate/up with GLU, shared-expert gate | ~57 tiles | 3.2 | 13 | 25-35 |
| R | softmax + top-10 + ids to the expert post (R-70's chain) | tiny | - | - | 6-10 |
| **D** (attention side) | shared down + expert sum + gate; pending write; down mix partials | 20 x 1,024 | 4.5 | 18 | 35-45 |
| G, U (attention side) | as above | | 5.3 | 21 | 35-55 |
| W_att (12 layers) | q, k, v projections + RoPE + norms; then F1's fs4 SDPA as its own dispatch | - | 23.4 | 94 | 150-200 |

- Per GDN-type layer: 9 dispatches, ~61 MB, floor 243 us, expected 360-480 us [G] against 704 us today [K].
- Per check: 36 x 0.36-0.48 + 12 x 0.45-0.60 = **18-24.5 ms; 16-21 with G folded into U and R into the expert post.**
- `kernels/price.out` has the same sums under each (t_sync, B) pair.

### W_gdn (`kernels/gdn_w.py`, compiles)
- One threadgroup per key head: 1 key head feeds 3 value heads, so q, k, v, z, a, b of one head group are 1,030 outputs.
- Phase A: simdgroup i owns 32-column tile i of the head's stream and runs the full K (40 quant groups). The hot loop is
  Leap's `qmm16n` at KS 1, verbatim. The a | b tile (6 columns) is split over K by simdgroups 0-7.
- Phases A2, B, C are GDNB's code with 3 value heads per threadgroup: 12 state rows per simdgroup in registers.
- Phase changes are threadgroup barriers. Only the 1,024 threads of one head wait, never the grid.
- **Concatenated weight stream:** one buffer per layer, [16 heads][33 tiles][40 groups][32 lanes][8 words], in the order
  each simdgroup reads it (`pack_gdn`). Scales and biases follow the same order. 24.3 MB per layer, read once.
- Threadgroup memory 24.4 KB of 32. Raw q, k, v, z go to device buffers (they are needed there anyway).

### D (`kernels/hc_d.py`, compiles)
- 20 slabs of 128 channels. A slab computes its 128 columns of out_proj (K split 8 ways inside the threadgroup), applies
  the pending write for its channels of all 4 streams, and runs its K slice of the down mix.
- Out: `written`, sum h'^2 per (slab, row, stream), fp32 partials [slab, stream, 16, 352]. K2 then sums 80 partials.
- The MLP-side variant swaps step 1 for the shared down projection (K 640) plus the expert kernel's output.

## 4. Layout rules for M <= 16 rows on the M5 [P from Leap's records]

- **Keep the 16-row matrix-unit op and pad.** The dense set is 3.6 G weights = 58 G MAC at 16 rows = 4.1 ms at the 14.0 T
  MAC/s peak [K M27-P], 39 % of the unit at the byte floor. A scalar 2-row path is only ~1.5x better below 3 rows [K
  K-62j] and loses at 16. One form for every row count; the real row count is a template value (S_C).
- **One simdgroup = one 32-column tile; a threadgroup = one independent pipeline.** Use 1,024 threads where a pipeline
  has 32 tiles (W, D). Residency is not bound by threadgroup memory up to 20 KB [K NAX-OCC]; W sits at 24 KB, so try
  the 20 KB variant (v scratch in a device buffer) if W streams below 180 GB/s.
- **Split K inside a threadgroup only.** A K split whose partials cross threadgroups is a sync point. D's 8-way split and
  W's a | b tile stay inside one threadgroup.
- **Defer every cross-pipeline sum into its consumer.** out_proj and the shared down are computed per channel slab
  inside the next D, where only that slab needs them.
- **No persistent kernels, no grid barriers.** They keep the waits (section 1) and K90's persistent arm lost 5-56 % [K].
- **Weights in read order.** DENSE-N's fragment-native layout (x1.20 [K]) is kept and extended across the fused pieces.

## 5. Why this beats the M5 grid-barrier KILL and M6b

- **The KILL (MoonShots, 21 Oct):** persistent kernel, grid barrier 1.97-2.47 us vs 1.3-1.6 us per dependent dispatch,
  per-op time unchanged. It replaced each hop by a barrier and kept every op's fan-out and fan-in. This design has no
  grid barrier at all and 11-14 fewer fan-ins per layer. It does not depend on a hop being expensive.
- **M6b (PARKED, +3 %):** a persistent GDN-layer dispatch at 1 row, 0.419 vs 0.433 ms. It kept the op chain and its
  K order, because it had to stay exact, and at 1 row the layer was bound by the scalar matvec (~77-97 GB/s). Here FAST
  frees the K order, which is what allows the per-head and per-slab cuts, and the 16-row matrix-unit path streams at
  ~200 GB/s [K DENSE-N], so the fixed part is the larger share.
- **GDNB (KILL, small) and 'R3 more GDN fusion':** they fused the body, which is 0.08 of 0.37 ms. W removes the waits
  between the projections and the body instead, and D removes out_proj's.
- **What would kill this design too:** part B showing t_sync <= 6 us for wide kernels and B <= 130 GB/s in flight. Then
  the time is inside the hot loop, the rebuild gives ~-8 ms, not -13..-20, and the work moves to the loop itself.

## 6. What part B must give, and the rules

1. t_sync: the time of one wide dependent kernel with almost no bytes (HC write, K2), and of one tiny one.
2. B in flight per kernel class: 4-bit native, 8-bit HC, at their real sizes in a dependent chain.
3. Dispatch counts per piece (replaces TODAY_N in `price.py`).
- Run: `python3 -I kernels/price.py --fit <GDN> <HC> <ATT> <MLP>`.
- t_sync(wide) >= 12 us: build W, then D, as drafted. Each removed wait is worth its t_sync.
- t_sync(wide) 6-12 us: build W first; fold G into U only if a tiny kernel costs >= 6 us.
- t_sync(wide) < 6 us and B < 130: stop the rebuild at W; the lever is the hot loop (decode, epilogue), not the count.

## 7. The one-layer test for Leap (GPU, Leap's harness, layer 4 real weights, 36 chained, S 16 and S 2)

- A: today's FAST chain for the same span = R-83 in_proj + GDNB + out_proj + HC8-1P K1.
- B: W + D.
- **PASS: >= 0.10 ms saved per layer** (3.6 ms per check). **KILL: < 0.04 ms.** Prediction [G]: 0.09-0.16 ms.
- Bits: W's y and states against A. Expect equality in q, k, v, z when A runs in_proj at (ks, nt) = (1, 1); a and b
  differ in the last bit, so y is compared by max relative error (line 2e-3). D's `written` the same way.
- Then FAST quality as for HC8-1P: EvalSuite's same-scores gate, not bits.

## 8. Bitwise variants

- W: none as one dispatch. The exact arm keeps today's two (R-83 at the reference (ks, nt), then GDNB); both are bitwise
  == REF2-T [K]. A bitwise one-dispatch W would need 64 simdgroups per head for the reference's KS 2: not possible.
- D: none. HC8 is FAST already, and the exact bf16 S4 read fixes its own K order.
- W_mlp and R: bitwise is possible, since they are row-parallel with the reference's order per output (R-70 did it).

## 9. Not done, and risks

- Not run on a GPU. Compile and packing checks only: `metal_check.py` PASS (4 instances), `pack_check.py` PASS
  (every output written once, 4e-15 vs W x in fp64).
- Not drafted: G / U edits (small), W_mlp, R, the MLP-side D, W_att. Sections 2-3 specify them.
- Risk 1 [G]: 16 threadgroups of 1,024 threads may fill fewer GPU cores than 512 small ones. Fallback: W as two
  dispatches (in_proj on its usual grid, then the body per head), one more wait.
- Risk 2: W holds 48 state floats per thread (GDNB's best had 32). If phase B slows, run it as two passes of 6 state rows each (state rows are independent).
- Risk 3: D reads y 20 times (4 MB at 16 rows) and K2 reads 1.8 MB of partials. Slab 256 halves both.
- The 8-bit decode in D is the plain MLX layout. Leap's HC8-1P K1 decode should replace it.
