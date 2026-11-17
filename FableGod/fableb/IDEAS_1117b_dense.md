# IDEAS_1117b: the per-row DENSE compute of extra tree rows (Fable B, Tue 17 Nov 08:2x-09:0x)

Lens (BREAK98b, route a): ~96 extra rows for <= ~5 ms = <= 0.05 ms per row, on the dense pass: activation precision,
sharing between siblings and between rows with a common prefix. Labels [K] measured / [P] derived / [G] estimate.
Sources: the 125B config (BigMoE cfg), check25_plan (matrix peak 14.0 T MAC/s bf16 [K]; dn32q int8 x int4 at
1.55-1.73x bf16's rate, M27-C [K]), check25 README (dense 12.97 ms at 32 rows, 1.3-1.6x floors on the big shapes),
MoonShots HANDOVER (module 2: the expert one-call is compute-bound; 8-row tiles run padded), the 127-node tree dumps.

## 0. What a marginal row's dense compute IS, from the config [K] and the matrix peak [K]

MACs per row (hidden 2560; 36 GDN layers: qkv|z 16384 x 2560 + out 6144 x 2560 + small; 12 attention layers: q 12288 x
2560 + o 6144 x 2560 + k / v small; HC low-rank 320 x 4; shared expert 3 x 2560 x 640; lm_head 248,320 x 2560):

| piece | MAC per row | ms per row at 14.0 T MAC/s bf16 | at int8's 1.55-1.73x [K] |
|---|---:|---:|---:|
| dense layer matmuls, 48 layers | ~2.75 G | 0.197 | 0.114-0.127 |
| shared experts, 48 layers | 0.24 G | 0.017 | 0.010-0.011 |
| lm_head (full vocab) | 0.64 G | 0.045 | 0.026-0.029 |
| **dense total** | **3.6 G** | **0.26** | **0.15-0.17** |
| (routed experts, 10 x 48 x 4.9 M, for scale) | 2.4 G | 0.17 | the one-call pads to 16-row tiles, so rows 2-16 per expert are ~free; union-priced (0.08 in the frame) |

- **FAR's "int8 dense 0.15 ms per marginal row" is the matrix-unit floor itself, not an inefficiency.** The kernels run
  at 1.3-1.6x their floors today, so the engine's marginal dense row is ~0.2-0.25 ms; at a perfect kernel it is 0.15.
- **The frame (0.05 ms per row) is 3x BELOW the int8 floor and 5x below bf16.** No kernel tuning reaches it. It
  needs either 3-5x more matrix throughput than the measured 14 T MAC/s, or 3-5x fewer MACs per row.

## 1. Activation precision: at most 1.6-1.7x, already in the frame [K]

- int8 activations x int4 weights are measured at 1.55-1.73x bf16's rate (NAX-C / M27-C). That is the whole precision
  lever on the matrix units as the records stand; the frame's 0.15 already includes it.
- 4-bit activations: no matrix path on this GPU in any record; a 4-bit x 4-bit product would have to be decoded to
  int8 or fp16 first, so it buys bytes of activations (negligible at 128 rows x 2560) and no compute. KILL.
- fp8 / "fast half": same rate as bf16 on the simdgroup matrix units in every record we have. KILL.
- **The one open question is not precision but the UNIT:** is 14.0 T MAC/s the simdgroup-matrix rate, or the
  neural-accelerator rate (ZERO.md: "over 4x the peak GPU compute for AI" per GPU core on this generation)? If the
  measured peak is the simdgroup rate and the accelerators are reachable only through Metal 4 tensor ops that no
  kernel of ours uses yet, the dense floor drops up to 4x and the frame becomes reachable (0.15 / 4 = 0.04). This is
  ZERO / Leap's microbench, not a desk question: **one Metal 4 tensor-op matmul (int8 x int8 and fp16) on GDN qkv|z at
  M 128 vs dn32q; PASS >= 2.5x dn32q's rate, KILL < 1.3x.** 10 min on the GPU, no engine change. Everything in route
  (a) on the dense pass hangs on this one number.

## 2. Sharing between sibling rows and rows with a common prefix: zero on the dense pass (structural KILL)

- Two siblings differ in their last token. From layer 0's input on, every activation differs, and a matmul over
  different inputs shares nothing: the dense MACs are per row by construction. The prefix IS shared, but only
  through the cores (attention KV, GDN state), which the frame already prices at 0.04 ms per row.
- Lossy sharing (compute the parent row once, siblings as a low-rank delta): SIB's read-out says sibling states are as
  far from each other as from the truth (direction error 1.0-1.1), so there is no small delta to exploit. KILL.
- The "common prefix" lever that does exist is already the tree check itself: the prefix rows are computed once
  for all branches. Beyond it there is nothing left to share on the dense side.

## 3. Skipping dense work exactly: crumbs, priced

- **lm_head at internal nodes only.** Leaves' logits are never read (no children to verify). On the real 127-node
  trees [K, this morning: think1117b count]: internal nodes 88.6 % (HE+) / 78.2 % (MBPP) at 127 pops, 98 / 95 % at 63.
  Saving at 128 rows: 14-28 rows x 0.045 ms = **0.6-1.3 ms per check [P]**, exact, a gather before the head. Worth
  wiring, not a route.
- **Walk-driven lazy head (logits only along the accepted path, in waves): KILL by bytes.** Rows that need logits fall
  to 66 of 127 (HE+, waves every 8 depths) but each wave re-reads the 318 MB head: 5.1 waves x 1.1 ms > the 5.8 ms
  full-head compute. At 16-depth waves: 86 rows, 2.8 waves, 3.1 ms bytes + 3.9 ms compute > 5.8. [K rows / waves from
  the dumps; P ms]
- **Walk-driven lazy TRUNK (compute subtrees only when the walk reaches them):** each wave is a full 125B pass over a
  smaller batch and re-reads the dense + union bytes. That is staged checks, killed in WEEK98 (sibling reuse /
  staged checks can't pay). KILL.
- **int8 on the lm_head** (it is bf16 today in the FAST path as far as the README's 11 shapes show; the head is a
  separate 1-call shape): 1.6x on 5.8 ms at 128 rows = **-2 ms [P]**, lossy, under the same G1 int8 gate (EvalSuite
  int8 argmax 16 / 16, |diff| 0.22). Small, cheap, real; not a route.
- **Vocab-pruned exact argmax** (candidate set + a norm bound on the rest): the Cauchy-Schwarz bound on 2560-dim
  logits is far above the true margins, so nothing prunes. KILL at the desk.

## 4. Bottom line

1. The marginal dense row is 0.26 ms bf16 / 0.15-0.17 ms int8 AT THE MATRIX PEAK [K config, K peak]; the frame's 0.05
   is 3x below it. Precision gives 1.6x (already counted), sharing gives 0 on the dense pass, exact skipping gives
   0.6-1.3 ms per 128-row check (head at internal nodes) and int8 on the head another ~2 ms.
2. **Route (a) on the dense pass is closed at the desk UNLESS the GPU has matrix throughput the kernels do not use.**
   The one screen: a Metal 4 tensor-op matmul at M 128 on GDN qkv|z vs dn32q (PASS >= 2.5x). If it passes, the whole
   dense floor moves and 96 rows cost ~5 ms at the peak; if not, ~96 extra rows cost >= 14 ms of dense compute alone
   and route (a) must come from fewer rows per kept (the head's own edge gap, 31 % droppable) or from (b) / (c).
3. For the record: the expert one-call pads to 16-row tiles, so extra rows cost expert COMPUTE only when an expert
   exceeds 16 rows; at 128 rows (~7 rows per expert by the union law) it stays union-priced, as the frame has it.
