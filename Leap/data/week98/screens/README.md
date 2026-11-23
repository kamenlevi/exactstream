# WEEK98 night screens, Sun 15 Nov 2026 (Leap; the orchestrator 01:0x: "run your first screens TONIGHT")

## Summary (02:1x)
| item | result | verdict |
|---|---|---|
| C2 DENSE32 | D32's 32-row dense kernels == 2 x 16 per row; dense 18.30 -> 22.21 ms per check (R 1.21) | PASS, exact |
| P1 OVERLAP | the chain head on a 2nd GPU stream adds 93 % of its time to the check (kappa 0.93) | CLOSED: pipelining on one GPU loses |
| P2 CHAIN-BYTES | chain step = ~0.54 ms fixed + bytes at ~290 GB/s; best cut 0.864 ms at 95 MB | CLOSED: block drafting is the route |
| KNEE | cache re-read 2.4-3.2 TB/s at <= 8 MB, ~500 GB/s to 48 MB; a dependent chain 0.03 ms / step at 8-12 MB | a 2nd drafter route: cache-resident chain |
| C3 / C3b / C3c / C3d | REF2's 17-32-row verify path: splits, one-call experts, GDN at full width + mr2, HC in one pass | ALL BITWISE (4-layer slice) |
| C5 v1 | exact 32-row check ~1.6x a 16-row check (fair read; 12-layer slice; 2 x 16 = 2.0x) | 1.4-1.7: needs kernel work |
| a45 (cc3 screen 3, Mon 03:26, REAL decoding, shipped exact-45 C 256) | cold 16-row check 110.7 ms = dense 30.2 + experts 40.1 + misses 40.4; dense 32 rows 70.9 (2.35 x); gated 32-row check ~188.4 ms [est] = 1.56-1.70 x like for like (v2 printed 1.35 x on a mixed basis: CORRECTED); slot path 2.5 s | MISSES the <= 1.35-1.4 x line at 45 GB (details: ../cc3/README.md) |

- **Waiting:**
  - BigMoE's G32 test: their 32-row gate + v32 on the full streamed 125B, bits; it holds the next memory slot.
  - Then C6 ABLATE32: each build step off in turn + the graph-build share, with the engine's SB; it replaces C5 v2.
  - exact-45 (a45): RAN Mon 03:26:10 (start gate 40, the orchestrator's exception; peak 42.71 GB, swap +0); row above.
- **v32.py is frozen at 9e9f9ba** until BigMoE's test reports; the later builds are separate modules.

Each screen runs under its own short quiet ticket through scripts/week98/qscreen.sh:
- the two-read NO-SWAP gate first, holding nothing (the lower of two reads 30 s apart, parts logged);
- then the ticket, a 20 s settle, the gate again, the power mode logged, a swap guard and a time cap;
- the ticket is released at the end.

Log: screens.log. Outputs: <tag>.txt and <tag>_pmode.txt. The pre-registered lines are in data/week98/plan_1115.md
(written Sun 00:4x, before any run).

## C2 DENSE32, 01:04-01:05 (quiet ticket, power mode 2, NO-SWAP 32 GB): PASS, exact
scripts/week98/c2_d32.py is scripts/d32.py unchanged except its output paths, re-run on today's stack (MLX 0.32.2):
the 125B's 17 dense shapes at their per-check counts, random weights, cold. Fans were capped (quiet mode). Data:
c2_dense32.txt and data/week98/c2/d32_1115.{txt,json}.
- **Bits.**
  - D32's 32-row kernels (mr2: one simdgroup computes both 16-row tiles; mg2: adjacent threadgroups) are bitwise equal to
    two 16-row calls, per row, at every K split tried, for all 12 shapes they serve. Real rows are independent of the
    padding (17 / 24 / 31 real rows, zero or random padding).
  - MLX's own 32-row call equals 2 x 16 for bf16 and for the big 4-bit shapes. It differs for small-N 4-bit (shared
    gate / up, shared down, k / v, in_proj_a / b, shared_expert_gate) and for 8-bit (router, indexer). Those stay
    2 x 16 calls.
- **Time.** Dense per check: 18.30 ms at 16 rows, 22.21 at 32 rows, so **R = 1.21**. Two 16-row calls would be R 1.65;
  the floor is 12.3. D32 on 10-26 measured 17.96 / 22.19, R 1.24.
- **Pre-registered line** (all 32-row kernels still bitwise and R <= 1.3): met. Action: wire mr2 + 2 x 16 splits as
  REF2's 17-32-row verify path. That is a build (scripts/week98/v32.py, drafted), with no GPU timing until it is ready
  (the orchestrator 01:0x).

## P1 OVERLAP, 01:11 (quiet ticket, power mode 2, NO-SWAP 32 GB): CLOSED on one GPU, kappa 0.93
scripts/week98/p1_overlap.py. Data: p1_overlap.txt / .json.
- **Setup.**
  - Chain: the 125B's MTP head standalone (the chain drafter's shape), 32 steps, one sync per step, on GPU stream 2.
  - Check proxy: 48 chained layers, each with a share of one check's dense set (D32's 17 shapes x per-check counts,
    773 calls, 3.43 GB of distinct random weights; qmm16n at 16 rows, mr2 at 32) plus one NAX6 expert call.
  - The NAX6 calls run on random 512-expert pools in NAX6's tier layout, with real top-10 routing of layers 6 / 20 / 40.
    The records3 store changed layout since ane_probe's loader (01:07 smoke: a reshape error), so the pools are random;
    the bytes and the routing are what matter.
  - Arms rotated per round, 10 rounds after 3 warm-ups. The chain on stream 2 gives the default stream's tokens.
- **Results.**

  | rows | check alone | chain alone (per step) | together | chain inside "together" | kappa (median; rounds) |
  |---|---:|---:|---:|---:|---|
  | 16 | 84.6 ms | 37.6 ms (1.174) | 119.7 ms | 40.1 ms (x1.07) | 0.932 (0.91-1.06) |
  | 32 | 110.8 ms | 38.0 ms (1.186) | 145.3 ms | 41.6 ms (x1.10) | 0.926 (0.80-1.01) |

- **Reading.**
  - The GPU runs the two streams almost serially: the overlapped chain adds 93 % of its own time to the check.
  - The pre-registered line was kappa >= 0.52 (= q at a flat 98 %): pipelining on one GPU loses. Pipelined cycle =
    check + 0.93 x draft + 0.48 x draft, against check + draft sequentially.
  - Matches the priors (NAX-OCC two-stream 1-2 % at 32 rows; R-11 D) and the prediction (kappa ~0.8-1.0, KILL 65 %).
  - The chain step here is 1.17-1.19 ms (MTP-STEP-2: 1.145), so 32 steps = 38 ms per block.
- **Action** (pre-registered): closed on one GPU. Drafting must get cheaper itself: P2's byte diet for the chain step,
  or a one-pass block draft (R-10: 10.96 ms at 32 rows, 4-bit).
  - An ANE drafter remains the one overlap route measured to work: ANE-1 found the GPU check x1.07 beside back-to-back
    ANE drafts. The MTP chain's MoE step is not an ANE shape; a dense block drafter is.

## C3 CORES 32, 01:12 (share lock, bits only; no timing): ALL BITWISE
scripts/week98/c3_v32.py + scripts/week98/v32.py (the 17-32-row scaffold). Data: c3_v32.txt.
- **Setup.**
  - Slice: layers 0-3 of the 125B (G G G A: GDN, the PLE GDN layer 1, GDN, QSA attention), resident, 6.7 GB.
  - ref2 runs with exact-45's env and exact forms (att_rope, idx_prep, hc_write, hc_gate, gdn_inproj_x, gdn_body,
    hc_b), then v32.
  - MTPLX's own MoE block with resident experts (ref2 experts_nax at <= 16 rows). Real code tokens after a 1,024-token
    prefill.
- **T1 forward.**
  - One M-row verify through v32 (M = 32, 17, 24, 31) equals 16-row + (M - 16)-row verifies of the same rows, with a
    full-accept commit between: 32 / 32, 17 / 17, 24 / 24, 31 / 31 rows bitwise (max |diff| 0).
  - Wrappers fired per forward: the LM head's matmul (qmm 1), lin 2, the HC reads (gr 9), attention 1, GDN 3, and the
    MoE block 4 (split at the block).
- **T2 commit.** A 32-row verify, then commit keep k, then a 4-row probe, against k rows verified as <= 16-row checks
  with the same probe: bitwise at k = 1, 9, 16, 17, 24, 31.
  - The joined GDN stash makes MTPLX's commit replay one window of 32 rows. The PLE layer's 17-31-row replay runs
    through v32's splits.
- **Line** (pre-registered): bitwise. The non-expert 17-32-row path works with the experts as 2 x 16, so it goes to
  BigMoE with the gate.
- **Build steps left.**
  1. The experts as ONE 32-row call (the union read once). This needs BigMoE's wider gate, and a 32-row MoE block that
     keeps _verify_call's sigmoid lowering: MTPLX's block takes its prefill combine at >= 32 rows.
  2. mr2 for the dense projections at 32 rows (one weight read, R 1.21 instead of 1.65). Today they run inside the
     chunked GDN / attention / MoE calls.
  3. The exact-45 loop's own pieces at 17-32 rows: K-73's fused MoE block (v32 wraps whatever block is installed) and
     BigMoE's streamed tier path.

## P2 CHAIN-BYTES, 01:15-01:16 (quiet ticket, power mode 2, NO-SWAP 37 GB): CLOSED, no variant <= 0.5 ms and <= 60 MB
scripts/week98/p2_chainbytes.py. Data: p2_chainbytes.txt / .json.
- **Setup.** The 125B's MTP head standalone (the chain drafter's shape): its bf16 Linears re-quantized and its FR-Spec
  screened head cut. Per step: weight bytes (MoE experts at top-10 / 512) and time as the chain runs it (one sync per
  step, 12 x 32 steps per round, variants interleaved, 12 rounds after 2 warm-ups).

  | variant | MB per step | ms per step (p10-p90) |
  |---|---:|---:|
  | as stored | 177.6 | 1.147 (1.112-1.231) |
  | bf16 Linears -> 8-bit | 146.9 | 1.065 (1.041-1.143) |
  | bf16 Linears -> 4-bit | 130.5 | 1.008 (0.976-1.099) |
  | screened head 16K rows (was 32K) | 154.0 | 1.052 (1.026-1.148) |
  | screened head 8K rows | 142.2 | 1.009 (0.983-1.076) |
  | 4-bit + head 8K | 95.1 | 0.864 (0.845-0.932) |

- **Reading.**
  - The step = ~0.54 ms fixed + bytes at ~290 GB/s (a fit through the six: 3.4 us per MB). Even at zero bytes a
    32-step chain costs >= ~17 ms.
  - The cuts that leave the layer alone stop at 95 MB: attention 29.8 + MoE 34.9 MB are untouched.
  - The 400-exact budget is ~3-7 ms of drafting per cycle. A chain head of this design cannot meet it, at any bytes, in
    MLX's per-op form.
  - Agreement with the stored head (top-1 on synthetic hidden states) is info only. Random states make near-ties
    everywhere; the owners' bank replay measures acceptance.
- **Pre-registered line:** a step <= 0.5 ms AND <= 60 MB: not met. **Action: close the chain-step diet; block drafting
  is the route.** A one-pass block draft costs 10.96 ms at 32 rows, 4-bit (R-10); a smaller or fused block drafter goes
  lower. 4-bit + head 8K remains a cheap option for the chain if the owners keep it: 36.7 -> 27.6 ms per 32 steps.

## C3b: the experts as ONE 17-32-row call, 01:17-01:18 (share lock, bits only): ALL BITWISE
V32_MOE1=1 c3_v32.py. Data: c3b_v32_moe1.txt.
- **The path.** v32's _moeblock calls MTPLX's SparseMoeBlock._verify_call at 17-32 rows. That keeps the verify lowering
  of the shared-expert gate, where MTPLX's own block would switch to its prefill combine at >= 32 rows.
  - The router and the shared expert split 2 x 16 through the qmm wrapper.
  - The experts run as ONE call through ref2's own resident kernel (experts_split for the unfused SwitchGLU, as ref2's
    _split_call; experts_nax for the fused class). Each (row, expert) pair is computed in its own tile row.
- **The 01:17 run did not test this.** That option only covered the fused class, and the slice's experts are the unfused
  SwitchGLU: moe1 0, moe 4, so the experts still ran 2 x 16. Fixed at 01:18 and re-run.
- **The 01:18 run.** moe1 fired 4 / 4 per forward (every MoE layer's experts as one call).
  - Forward at M = 32 / 17 / 24 / 31: every row bitwise (max |diff| 0).
  - Commit at keep 1 / 9 / 16 / 17 / 24 / 31: the probe is bitwise.
  - The run's last line still prints "(experts 2 x 16)"; that text is fixed in the script, and the counts above are what ran.
- **Reading.** On the resident kernels, the 17-32-row MoE block with the union read once is exact. The router softmax,
  top-k, the k-axis combine and the gate lowering all match the 16-row path per row.
- **Next.** The same on the streamed path: BigMoE's widened gate (build_g32) feeding ref2's _ooo_nax6 at M = 32. BigMoE
  owns that test: one 32-row verify through the wide gate against 2 x 16, per-layer outputs plus the committed cache, on
  bank sequences.

## C3c: build step 2, the GDN projections at full width with mr2, 01:24-01:25 (share lock, bits only): ALL BITWISE
V32_MOE1=1 V32_MR2=1 V32_GDN32=1 REF2_DENSE_NATIVE=1 c3_v32.py, with scripts/week98/v32mr2.py. v32mr2 is a separate module
patched on top of v32, so v32.py stays as C3b tested it: BigMoE's g32 test imports v32. Data: c3c_v32_mr2.txt.
- **What changes.** At 17-32 rows, a GDN layer's projections run at full width: R-83's qkv | z concat and b | a, then
  out_proj. Only the recurrence core is chunked: ref2's one-dispatch gdn_body on rows [0, 16), then [16, n), carrying the
  conv and recurrent states; the capture stash is joined.
  - With DENSE-N's native copies the big projections (qkv | z, out_proj) and the LM head take D32's mr2 at 32 rows: one
    weight read.
  - Small-N b | a and the shared expert stay 2 x 16.
- **Result.**
  - T1 at M = 32 / 17 / 24 / 31: every row bitwise. mr2 fired 11 times per forward; gdn32 took all 3 GDN layers,
    including the PLE layer, with 0 declined.
  - T2, commit keep 1-31: bitwise.
  - One 16-row verify runs first so the native copies exist. In the first run (01:24) the M 32 case had none yet: mr2 0
    there, 2 x 16 instead, still bitwise.
- **Worth** (from C2's per-shape times, not measured in the engine). mr2 instead of 2 x 16 on qkv | z, out_proj and the
  LM head is ~-2.7 ms per 32-row check. DENSE-N's own 16-row gain (R-77: -1.46 ms) comes with it, for +~1.5 GB of
  native copies.
- **The 17-32-row path now has every piece but the gate.**
  - Row-local splits, GDN at full width with a chunked core, attention chunked.
  - The MoE block in its verify lowering, with the experts as one call.
  - The commit replay over the joined window.
  - What's left: BigMoE's g32 gate on the streamed path (their test) and timing once it's whole.

## KNEE, 01:35 (quiet ticket, power mode 2, NO-SWAP 40 GB): a cache knee at <= 8-12 MB, and a second level to ~48 MB
scripts/week98/knee_probe.py (the orchestrator 01:2x's optional probe). Data: knee_probe.txt / .json.
- **Re-read** is one dispatch, 40 x 1024 threads re-reading their share of W (uint4 loads, every pass used):

  | W (MB) | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 128 | 256 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | re-read GB/s | 2417 | 3165 | 2280 | 1118 | 517 | 498 | 508 | 483 | 358 | 294 | 277 | 274 |
  | MLX chain, ms per step | 0.011 | 0.013 | 0.026 | 0.035 | 0.044 | 0.062 | 0.111 | 0.177 | 0.249 | 0.368 | 0.471 | 0.950 |

  "MLX chain" = 32 dependent bf16 GEMVs (W square, the same W every step), one graph, one eval, no host round trip.
- **Reading.**
  - Up to ~8 MB the weights re-read from cache at 2.3-3.2 TB/s (~10x DRAM). 12 MB is 4x.
  - 16-48 MB holds a second plateau at ~500 GB/s (~1.8x DRAM).
  - From ~96 MB on it is DRAM at ~276 GB/s.
  - MLX's own chain of dependent steps costs ~10 us of fixed overhead per dispatch, plus bytes at 190-400 GB/s. Its
    GEMV does not reach the cache's bandwidth, but it never pays a host sync.
- **Pre-registered line** (a knee at W >= 4 MB re-reading at >= 870 GB/s): met, up to 12 MB.
- **Action: the drafter owners get a second route beside block drafting, a cache-resident chain.**
  - Budget: <= ~8-12 MB of weights per step at 1.1-3.2 TB/s, or <= ~48 MB at ~500 GB/s.
  - Measured per-step cost in MLX's per-dispatch form, no host sync: 0.026-0.035 ms at 8-12 MB, 0.11-0.18 ms at
    32-48 MB. So 30 steps cost ~1 ms at 8-12 MB, or ~3-5 ms at 32-48 MB, inside the 400-exact drafting budget.
  - A step must keep the token on the GPU (MTP-STEP-5's argmax -> take -> embedding) and use few dispatches. Today's
    MTP head reads 95-178 MB in ~40 kernels plus a host sync (P2: 0.54 ms fixed).
  - Example fit: ~24M parameters at 4-bit = 12 MB, the head's screened rows included (8K rows x 2560 x 4 bit is
    already 10.5 MB).

## C3d: build step 4, the HC read at 17-32 rows in one weight pass, 01:36-01:37 (share lock, bits only): ALL BITWISE
V32_MOE1=1 V32_MR2=1 V32_GDN32=1 V32_HC32=1 REF2_DENSE_NATIVE=1 c3_v32.py, with scripts/week98/hc32.py. Data:
c3d_v32_hc32.txt.
- **hc32.**
  - leapk.hc_read's fuse_gate chain at 32 rows. The split-K GEMM (S4x2) keeps a second 16-row NAX accumulator that
    shares every weight-tile load, so each row gets exactly the 16-row kernel's sequence of ops on the same tiles, and
    the weights are read once.
  - Then the partition sum with a 32-row stride, MLX's own up GEMM (== 2 x 16 per row, D32), and the row-generic norm
    and tail kernels. Rows 17-31 are zero-padded to 32.
- **Result.**
  - Per forward, 8 HC reads take the one-pass path; 1 declines and falls back to v32's 2 x 16 split. The declined read
    is not identified; most likely the read without a block_inject, which hc32 doesn't take.
  - Forward at M = 32 / 17 / 24 / 31: every row bitwise. Commit keep 1-31: bitwise.
  - The 16-row reference here ran kernel B (hc_b), so this also re-confirms D1-B: kernel B == the S4 chain, bitwise.
- **Worth** (C2's bf16 per-shape times: HC mix down 3.98 -> 2.75, up 3.64 -> 2.81, block_inject 0.60 -> 0.38 ms per
  check): ~-2.3 ms per 32-row check against 2 x 16; timing comes in C5.

## C5 TIME32 v1, 01:39 (quiet ticket, power mode 2, NO-SWAP 41 GB): R32 1.914 as registered, but the harness was biased
scripts/week98/c5_time32.py (v1). Data: c5_time32_v1.txt.
- **Setup.** 12-layer resident slice (GGGA x 3), 16 rounds.
  - A16: a 16-row check. B2x16: two 16-row checks. C32: one exact 32-row check through v32 + MOE1 + v32mr2 + hc32 +
    DENSE-N.
  - Each check = snapshot, verify, the acceptance read (sync) and a full-accept commit.
- **Result:** A16 24.03 ms, B2x16 58.81, C32 48.00. R32 = C32 / A16 = 1.914 (rounds 1.57-2.21), "look for a missed
  path" by the pre-registered line. C32 / B2x16 = 0.816.
- **Harness flaw.** A16 ran on its own cache advancing 16 tokens per round; B and C shared one advancing 64. So B / C
  checked at ~2.2K context against A's ~1.3K; B2x16 / A16 = 2.40, which two 16-row checks cannot be. On B's own cache
  the fair read is 2 x C32 / B2x16 = **1.63**: the exact 32-row check costs ~1.6x a 16-row check here, in the 1.4-1.7
  "pays only with the kernel work" band.
- **v2 (01:4x)** gives every arm its own cache advancing 32 per round. Its share-lock smoke (4 layers, noise) read 1.67.
  - The quiet run SKIPPED at 02:11: NO-SWAP stayed 19-27 GB, below its 27 GB gate, for 30 min. Nothing was queued.
  - It runs after BigMoE's G32 test, which has the next memory slot (the orchestrator 01:56).
- **Also next:** neither arm has the engine's SB (an async submit every 4 layers on 2-16-row checks: r57 ref_on), so
  Python's build and the GPU run partly serial in both. scripts/week98/c5_sb.py runs C5 with SB extended to 2-32 rows on
  every arm (scheduling only, bits unchanged) for the engine-like ratio.

## G32 drift (BigMoE 02:34) -> my v32 bug, fixed in b0c99d5 (02:4x)
- **What BigMoE saw.** g32_test on the full streamed 125B (C 160, MAXK 320, v32 9e9f9ba with V32_MOE1): one 32-row verify
  against 16 + 16 drifted at rounding level on every row from layers 0-5, with every argmax right.
- **Cause.** At 17-32 rows v32 called MTPLX's stock SparseMoeBlock._verify_call, not the exact-45 loop's own verify block
  (K-73 _exact_block / R-69 _fused_block). The loop's block does three things the stock one doesn't:
  - it sorts the top-k into gate-rank order, and REF2-T decodes ranks 0-1 / 2-9 by position at 4 / 3 bits;
  - it hands the scores and the shared expert to the tier call, whose OOO combine needs t['scores'] (bigmoe/tier.py:175);
  - it adds the shared expert in its tail.
  So the stock path decoded the right experts at the wrong bit widths and combined them with MLX afterwards.
- **Why C3b missed it:** the resident slice has no tiers.
- **Fix.** With a tier _bm on the switch, v32 replicates the installed block at 17-32 rows with one switch call.
  router_fused runs one threadgroup per row, moe_tail is elementwise, and shared_cat and the router GEMM split 2 x 16.
  - The 2 x 16 block split is refused there (BigMoE's B3: it breaks the sync-free gate's one call per layer).
  - The stock path is unchanged: the C3b re-run (c3b2_v32_moe1_fix.txt) is bitwise.
  - BigMoE re-runs g32_test.

## Pipelining at 16 rows, today's operating point (the orchestrator 02:4x)
- With today's drafter, kept/32 / kept/16 = 1.30 (EvalSuite 9480ba4), below C5's fair 1.63, so 16-row checks win.
- Pipelining pays iff kappa < q. P1 measured kappa 0.93 on one GPU (at 16 and at 32 rows).
- A 16-row block needs q = P(15 guesses kept and the bonus guessed) > 0.93, i.e. > 99.55 % per guess. A 32-row block
  needs > 99.77 %. No drafter is near that, so pipelining on one GPU is closed at every block size.
- An ANE drafter (ANE-1: the GPU check slowed x1.07, so kappa ~0.1-0.2) is the only overlap route; it pays at
  q > ~0.2, i.e. at 16 rows above ~90 % per guess.
- C6's grid now includes 20- and 24-row exact checks (C20, C24) beside A16, B2x16, C32 and the ablations.

## Qualification of P2 / KNEE by Sandbox98's CHAIN SIZE screen (d7d9319, relayed by the orchestrator 02:4x)
- **On the 2B, no chain head at <= 48 MB per step holds its accuracy.** The best <= 48 MB form (4-bit body + an 8K
  list, 43.7 MB) keeps 12.6 % (MBPP) / 23 % (HE) fewer tokens per check.
  - 8-bit is free.
  - 4-bit costs 6-11 % even after training.
  - An 8K static list costs 8-17 %.
- **P2's "4-bit + head 8K"** (0.864 ms, 95 MB on the 125B) is a speed option only. It needs QAT and a better list
  (static + the prompt's tokens) before it counts.
- **KNEE** stays a speed fact: a cache-resident chain would cost ~1-5 ms per 30 steps, but no drafter fits it yet.
  Block drafting stays the route.

## NAX6-ROWS, 02:44 (share lock, bits only): the exact-45 expert kernel is row-invariant at 17-32 rows
scripts/week98/nax6_rows.py. Data: nax6_rows.txt.
- **Setup.** NAX6 (REF2-T tiers 4433333333) on random 512-expert pools in its layout. Real-shaped routing: each row 10
  distinct experts in gate-rank order, the window's union ~84-103 (BigMoE step 0's mean at 32 is 108). Three windows
  at M = 32 / 17 / 24 / 31.
- **Result.** One M-row call against 16 + (M - 16): every (row, rank) pair bitwise (320 / 320 ... 170 / 170), and
  combine_phase's per-row k-sum bitwise on every row.
- **Reading.** The expert kernel and its combine are not where a 32-row drift on the exact-45 path would come from.
  Past b0c99d5's block fix, the parts left to suspect on the streamed path are OOO's phase split (the gate's spin) and
  the slot mapping of 9 id lines. BigMoE's loop test (R57_G32TEST, 378ad69) covers those.
- **Addendum 02:5x (BigMoE 6169f13):** the G32 ext also had a bug. The loader copied each call's ids into
  `int32_t ids_copy[256]` (ext:1682, sized for R-55's <= 185 ids), so a 320-id call wrote 64 ints past it.
  - The 02:34 drift may therefore have had two causes: v32's MoE path (b0c99d5, wrong for the tier path in any case) and
    this overflow.
  - BigMoE's D-23 bisect ("half the rows differ at L1-L5") and every 320-id result before 6169f13 are void; don't build on them.
  - R57_G32TEST reruns on the exact-45 loop with both fixes.

## G32 PASSES, 02:47 (BigMoE's test on the exact-45 loop): the exact single-pass 32-row check works on the full streamed 125B
- **Setup.** BigMoE r57_screen_fast R57_G32TEST (378ad69): exact-45 flags (R57_PRESET=45, NAX6, the K-73 block, OOO, the
  D45 extras), C 160. BIGMOE_MAXK 320 with 9 id lines, ext build_g32 + the ids_copy fix (6169f13). v32 b0c99d5,
  V32_MOE1=1. Windows dd__mc_27921 / 2753 / 28056 (gate_req 142 / 154 / 156). Peak 30.9 GB.
  Log: ~/BigMoE/data/deepdraft/bank/g32loop.log.
- **Result** (each window): the 32-row forward run twice is bitwise.
  - T1: one M-row verify == <= 16-row verifies with commits between: 32 / 32, 17 / 17, 24 / 24, 31 / 31 rows bitwise.
  - T2: commit keep k = 1 / 9 / 16 / 17 / 24 / 31 then a 4-row probe: 4 / 4.
  - Per 32-row forward: tierblk 48, moeblk 48 (one expert call per layer: the union read once), qmm 145, gr 97, att 12,
    gdn 36.
- **What it took:** REF2's 17-32-row path (v32), the loop's MoE block at 32 rows (b0c99d5), BigMoE's gate (488c428) and
  their loader-buffer fix (6169f13).
- **Open for production.**
  1. The overflow guard (BigMoE). The loop cannot know a window's distinct count before routing. Every window seen fits
     C 256 (max 242), but the guarantee needs the flag + 2 x 16 redo: slot 0 at weight 0, never -2.
  2. GATE_N = MAXK on the gate arm (BigMoE's loop).
  3. Timing on the engine (Leap: C6 on the slice, then an engine arm).

## G32T, the engine time mode (BigMoE runs R57_G32TEST=...:time; pre-registered by Leap Sun 03:0x, before any run)
- **Setup.** The exact-45 loop at C 160 (as G32: NAX6, K-73 block, OOO; MAXK 320, build_g32 + 6169f13, v32 b0c99d5,
  V32_MOE1=1). Quiet ticket, gate 35 GB (peak 30.9 + 4), power mode logged.
- **SB.** The loop's async submit every SBN = 4 layers, extended from 2-16 to 2-32 rows: in ref_on's lambda,
  SB_LO <= rows <= 32. Scheduling only, every bit unchanged. Without it, a 32-row check builds its whole graph before the
  GPU starts, and the 16-row checks it is compared with do not.
- **Windows:** dd__mc_27921 / 2753 / 28056, one prefill each. Drafts are the bank's REF2 continuation, so every guess
  is right.
- **Arms**, each on its own rollback from the prefill:
  - A16: one 16-row check;
  - B2x16: two 16-row checks with a commit between;
  - C24: one 24-row check;
  - C32: one 32-row check.
- **Rounds.** 2 warm-up + 12 measured per window. The arm order rotates per round (A B C D, B C D A, C D A B, D A B C),
  because the arms share the expert pool through the rollbacks and an earlier arm warms it.
- **Logged per arm:** wall ms per check (eval-synced), median + IQR; misses and SSD wait per check.
- **Metrics:** R32 = C32 / A16 and R24 = C24 / A16 (medians of the per-round paired ratios); C32 / B2x16.
- **Lines.**
  - R32 <= 1.40: the exact 32-row check pays at today's kernels for a ~98 % drafter (cheapcheck's 1.35-1.4).
  - 1.40 < R32 <= 1.70: it pays only with the kernel work.
  - R32 > 1.70: a missed path first (the 9-line gate's cost, misses, SB).
  - C32 / B2x16 <= 0.75: one pass saves >= 25 % over checking 32 rows as two 16-row checks.
- **Prediction:** R32 1.45-1.6 (the resident slice read ~1.6 without SB), C32 / B2x16 ~0.75-0.8, R24 within 0.1 of R32
  (the second 16-row chunk is padded either way).

## C6 ABLATE32, 02:50 (quiet ticket, power mode 2, NO-SWAP 45 GB): exact 32-row check 1.58x a 16-row check; the experts' single read pays, hc32 does not
scripts/week98/c6_ablate32.py. Data: c6_ablate32.txt.
- **Setup.** 12-layer resident slice, the engine's SB (every 4 layers, extended to 2-32 rows), each arm on its own cache
  (all advancing 32 per round, contexts 1024 -> 1472), 12 rounds after 2 warm-ups. "Build" = the forward call's Python
  time with SB's submits inside.

  | arm | ms per check | build (share) | x A16 | step worth (arm - C32) / A16 |
  |---|---:|---:|---:|---:|
  | A16 (16 rows) | 22.67 | 12.13 (0.54) | 1.000 | |
  | B2x16 | 45.50 | 24.87 (0.55) | 2.007 | |
  | C20 | 31.46 | 20.41 (0.65) | 1.387 | |
  | C24 | 33.13 | 21.32 (0.64) | 1.461 | |
  | C32 (everything on) | 35.81 | 22.25 (0.62) | **1.579** | |
  | noMOE1 (experts 2 x 16) | 41.69 | 27.61 (0.66) | 1.839 | **+0.260** |
  | noGDN32 (GDN chunked whole) | 35.78 | 21.95 (0.61) | 1.578 | -0.001 |
  | noHC32 (HC read 2 x 16) | 34.77 | 21.37 (0.61) | 1.534 | **-0.046** |
  | noMR2 | 37.18 | 21.52 (0.58) | 1.640 | +0.060 |

- **Reading.**
  - The harness fix holds: B2x16 / A16 = 2.007.
  - The pieces that pay at 32 rows: reading the expert union once (+0.26 x A16) and mr2 (+0.06).
  - GDN at full width is worth nothing. hc32 is SLOWER than two 16-row HC reads (-0.046). Its S4x2 kernel loses more
    than the second weight read costs, so it comes out, and C32 becomes ~1.53x.
  - **The cost of rows is a step:** 16 -> 20 rows costs +0.39 x A16, while 20 -> 32 costs +0.19. Crossing 16 rows starts
    a second, padded chunk of every op still split 2 x 16 (the GDN and attention cores, the 8-bit router / indexer, the
    shared expert, small-N projections). Those are latency-bound, so the second chunk costs about as much as the first.
  - A check of 17-31 rows therefore costs nearly a 32-row check: the useful lengths are 16 and 32.
- **Pre-registered lines.**
  - Steps worth < 0.03: GDN32 and hc32 are not paying; MOE1 and mr2 pay.
  - Build share >= 0.5 at 32 rows (0.62): by the line, CPU-bound, and the next lever is fewer dispatches, not kernels.
    The GPU still runs ~13.6 ms past the build's end at C32 (10.5 at A16), so the check is mixed rather than purely
    CPU-bound. Fewer dispatches serves both.
- **Next levers (fewer dispatches), in the step's order of size.**
  - One-launch 32-row forms for what is still 2 x 16. The GDN core is a sequential recurrence: a 32-row gdn_body is exact
    by construction. The 8-bit router / indexer and small-N shapes need 32-row replicas of MLX's 16-row qmm (leapk's D2
    machinery extracts MLX's qmm_t). Attention: projections at full width, core chunked.
  - Target: the 17-row step from ~0.37 toward ~0.1, which puts R32 at ~1.3.
- **Correction to C6's reading (03:0x).** The 32-row check pays iff kept32 / kept16 > its cost ratio. The registered
  "R32 <= 1.4" came from cheapcheck's estimate of the deep drafter's kept ratio; the measured ones replace it
  (EvalSuite 9480ba4 via the orchestrator):
  - today's t30b: 1.30 < ~1.53, so 16-row checks win;
  - the goal (98 % at depths 0-14): ~1.64 > ~1.53, so the exact 32-row check pays ~+7 % already, and ~+20 % if the
    second-chunk launches go (cost ratio toward ~1.35).
  - Why hc32 lost (KNEE): small weights (HC mixers ~6.6 MB a layer, shared expert, router) re-read from cache on the
    second 16-row chunk. One-pass forms of those save launches, not DRAM bytes. The remaining work needs single
    dispatches covering both row tiles (mg2-style), not two accumulators per simdgroup.

## G32T RESULT (BigMoE, Sun 02:5x-03:1x, under Leap's pre-registration 92f71ac): on the engine the exact 32-row check costs 1.625x a 16-row check
- **Setup.** Exact-45 loop at C 160, MAXK 320, build_g32, v32 b0c99d5 with V32_MOE1=1 (mr2 off: no DENSE-N copies on
  exact-45). SB on at 17-32 rows. Quiet ticket, NO-SWAP 45 GB, power mode 2. 2 warm-up + 12 rotated rounds per window.
  Warm pool: 0 misses and 0 SSD ms on every arm. Data: ~/BigMoE/data/r57/g32time/g32time_<key>.json, g32time.log.

  | window | A16 | B2x16 | C24 | C32 (ms per check incl. its commit; median) |
  |---|---:|---:|---:|---:|
  | dd__mc_27921 | 64.2 | 130.8 | 98.9 | 102.4 |
  | dd__mc_2753 | 61.2 | 124.2 | 95.6 | 99.4 |
  | dd__mc_28056 | 61.3 | 128.0 | 97.1 | 102.3 |

- **Pooled paired ratios** (3 windows x 12 rounds, median [IQR]):
  - R32 = **1.625** [1.597-1.676] (per window 1.593 / 1.627 / 1.676);
  - R24 = 1.560 [1.541-1.582];
  - C32 / B2x16 = 0.798 [0.784-0.801].
  - C32 == B2x16 logits bitwise on every window: SB at 17-32 rows is scheduling only.
- **Against the lines:**
  - R32 is in 1.40-1.70: needs kernel work.
  - C32 / B2x16 0.798 > 0.75: one pass saves ~20 %, short of the 25 % line.
  - Against the prediction (R32 1.45-1.6, C32 / B2x16 0.75-0.8, R24 within 0.1): R32 is at the top or just past;
    the other two are inside.
- **Shape (as C6 on the slice).** 16 -> 24 rows costs +34.7 ms, 24 -> 32 only +3.5-5 ms: a step at 17 rows, not a slope.
- **Economics on the engine.** It pays iff kept32 / kept16 > 1.625. The goal drafter (~1.64) is about break-even, so the
  second-chunk launches have to go before the 32-row check is worth using. Today's drafter (1.30) stays on 16 rows.
- **What the step is made of.** BigMoE's guess is v32's 145 qmm splits reading the dense / router / shared weights
  twice. C6 on the slice says otherwise:
  - mr2 + GDN projections at full width are worth only ~0.06 x A16;
  - the small weights re-read from cache (KNEE);
  - so the step is mostly the second, padded chunks' LAUNCHES of the latency-bound ops: attention and GDN cores,
    HC reads, router, shared expert, small-N projections.
  - mr2 on the engine needs REF2_DENSE_NATIVE=1 (+~1.5 GB) and v32mr2 installed. V32_MR2=1 alone does nothing on
    exact-45, which has no native copies.

## C32m on the engine (BigMoE, Sun 03:1x, paired with C32 within rounds): mr2 / full-width GDN projections save NOTHING; dropped
- **Setup.** G32T's setup + REF2_DENSE_NATIVE=1 (native copies), R57_GDNB=1 (gdn_body), R57_G32_MR2=1 (v32mr2 + mr2
  toggled per arm, paired within every round). 3 windows x 12 rounds, rotation over 5 arms, quiet.
- **Engaged on every C32m forward:** mr2 121 per forward, gdn32 36, declined 0. My ~73 missed the 48 shared-expert
  down_proj calls (N 2560 has a native copy). C32m == B2x16 bitwise on every window.
- **Result.** (C32 - C32m) / A16 = **-0.002** [IQR -0.023 to 0.025] pooled; per window -0.009 / +0.018 / 0.000. Under the
  0.03 keep line, so mr2 and v32mr2 are dropped from the 32-row path, as hc32 was.
  - R32 in this config 1.607 [1.572-1.665], C32 / B2x16 0.773: the same picture.
  - It confirms the reading: the step at 17 rows is launches, not second weight reads.
- **Side finding (BigMoE).** With REF2_DENSE_NATIVE=1 + gdn_body every arm ran ~35-40 % SLOWER than in the first G32T run
  (A16 82-90 ms vs 61-64; C32 136-142 vs 99-102), with wider IQRs.
  - No other load, warm pool; swap rose ~1 GB (to 8.2 GB) with the native copies' extra memory.
  - Not isolated yet: DENSE-N alone vs gdn_body alone, ~5 min each.
  - Neither is on in any shipped config (exact-45 sets neither; the lossy lines use DENSE-N-RAM in place, no extra memory).
    Isolate before anything turns DENSE-N on at 45 GB.
- **The 32-row engine path is now:** v32 (b0c99d5) with V32_MOE1, with no mr2, no v32mr2, no hc32. The next builds cut
  launches.
- **CORRECTION 03:2x (BigMoE ISO1 / ISO2): the 35-40 % slowdown was run-level drift, not DENSE-N or gdn_body.**
  - ISO1 ran config blocks in a fixed order on one window: A16 68.9 (neither) -> 78.9 (gdnb) -> 84.0 (dense) -> 93.0
    (both) -> 89.8 (neither again). Swap was flat at 8,208 MB; the blocks tracked a drift in time.
  - ISO2 rotated the 4 configs within each of 12 rounds: A16 medians neither 64.1, gdnb 64.3, dense 64.9, both
    65.0 ms. Paired deltas vs neither: +0.2 / +0.7 / +0.8 ms (<= ~1 %). Neither's own drift was +0.0.
  - GDNB's -0.71 ms (70cd998) did not show (+0.2), within noise on one window.
  - So absolute ms across tonight's runs are not comparable, only within-run paired rotations. G32T's and C32m's
    ratios are within-run pairings and stand.

## The 32-row check's overflow guard (BigMoE, Sun 03:1x; ext 0231f64, test 40b4a04, results 2a91737): built and tested OK
- **The guard.** BIGMOE_OVERFLOW_FLAG=1: a GATE call whose distinct experts exceed the pool gets slot 0 for every use,
  reads nothing, and bumps ext.overflow_count(). The sync route (prefill) still aborts.
  - Test hook: BIGMOE_FORCE_OVERFLOW=<layer>[:<min ids>] forces 17+-row calls only, so 16-row redos pass.
- **Test** (R57_G32TEST + R57_G32OVF=1, exact-45 loop, C 160, layer 5 forced, 3 windows):
  - every forced check counted +1 with finite logits (garbage when cold, by luck right when warm);
  - every 16 + 16 redo counted +0 and was bitwise to the reference, 9 / 9.
- **For the loop port** (the release loop, when 1.2.1 un-parks): after a 17-32-row check's read-back, compare
  overflow_count() with its value before. If it rose, roll back with the check's snapshot and redo as two 16-row checks
  (or take a 16-row check that cycle). The weight-0 pairs are moot, since the flagged check is discarded whole.
- **Still BigMoE's:** the redo in their decode loop, GATE_N = MAXK and the MAXK asserts. They matter once a loop runs
  32-row checks.
- **UPDATE 03:4x (Sandbox98 69c93b2, relayed by the orchestrator).**
  - A prompt-aware list (static 8K + the prompt's tokens + the answer's tokens so far) cuts the head's list loss from
    -8.5 / -18.1 % to -2.4 / -1.1 % kept on the 2B: ~9.7 MB per step at 4-bit, gathered once per check.
  - A QAT screen of the 4-bit body runs next. If it holds, the chain fits the 16-48 MB cache band at ~44 MB per step.
- **Pricing (Leap, from KNEE and P2):**
  - ~44 MB per step at the band's ~500 GB/s is ~0.09 ms of bytes. The step's launches then dominate.
  - Today's ~40-kernel MTP step carries ~0.54 ms fixed (P2), so 30 steps cost ~15-17 ms.
  - The same bytes in ~5 dispatches with the token kept on the GPU (MTP-STEP-5's form) cost ~0.15 ms per step, so
    ~4.5 ms per 30 steps, inside the 400-exact budget.
  - So the cache-resident chain needs both: <= ~48 MB per step (Sandbox98's track) and a step fused to a few
    dispatches (a kernel job, not a training one).

## TREE16 (desk pricing, Sun 10:3x; the orchestrator's ask from MTP Head's S9 TREE-DYN): data/week98/tree_verify_price.md
- **Verdict:** a 16-node tree check costs about chain-16 (61-64 ms) everywhere except exact attention.
  - FAST / d1: about chain-16 + 0.3-1 ms.
  - Exact: + 0.5-1.5 ms with (b) or (c); + 3-6 ms at 1-2K context with (a), + 10-20 ms at 8K.
- **GDN:** a DFS walk inside the one-dispatch body, the state saved at forks (+0.1-0.5 ms per check). Exact by
  construction.
- **Attention:** a plain tree mask is not bitwise vs REF2-T for side-branch nodes. MLX's unfused head-dim-256 SDPA
  groups the softmax sum and P·V by key column. Main-chain nodes, laid out first, are exact. Exact routes:
  - (a) side paths batched as chain layouts;
  - (b) MLX's scores + softmax with the tail permuted per node, plus a bitwise port of the P·V GEMM with a per-row V
    tail;
  - (c) a new reference with per-row key lists (Kamen's call).
- **Net vs S9's EAGLE-2-16 estimate (+17 % / +5 % kept):** ~+15 % / +3-4 % with (b) or (c). So BigMoE's top-3 dump is
  worth running.

## ATT32-INV (CHECK55 step 1, Sun 10:48; share lock, bits only): the 32-row attention core is exact as ONE call per class
scripts/week98/att32_inv.py, data: att32_inv.txt.
- **Test.**
  - REF = v32's two 16-row chunks through ref2's exact verify SDPA: rows 0-15 at N1 = L + 16 keys, rows 16-31 at N2.
  - CAND = one 32-row call at N2 with ref2's class logic per row. Row j's AR class comes from L + 16 + j; rows in a
    lower class (only rows 0-15 can be) run at chunk 1's length; higher ones are recomputed in their class.
  - Model shapes: 24 / 2 heads, head dim 256, bf16, bool causal mask, peaky scores.
  - 13 contexts from 60 to 4,500 keys (class 0, the 128-511 band, the 0/1 and band / bucket crossings, 512-key
    buckets), 3 seeds each.
- **Result.**

  | head plan at 32 rows | rows bitwise | max diff |
  |---|---|---|
  | MTPLX's own (24 chunks x 1 head x 32 rows) | 1,248 / 1,248 | 0 |
  | h2 (12 chunks x 2 heads x 32 rows, the 16-row call's count) | 1,248 / 1,248 | 0 |
  | one call, all 24 heads | 1,021 / 1,248 | 7.8e-3 (bitwise only below ~500 keys, as R-62 found) |

- **So** the attention core needs no second chunk at 17-32 rows: h2 keeps the dispatch count of one 16-row call.
- **Next:** the rest of the attention module at 32 rows (projections, RoPE, indexer, gate) and a C3-style slice test
  of the whole module.

## ONE32 matmuls (CHECK55 step 1, Sun 10:5x-11:0x; share locks, bits only): every quantized matmul of the exact check has a one-dispatch 17-32-row form, bitwise per row
scripts/week98/one32.py (pieces), one32_test.py, qmm32_inv.py. Data: one32_test.txt, qmm32_inv.txt.
- **REF2's certified 16-row forms** (ref2._qmm): 4-bit g64 with N >= 1024 and N % 64 == 0 run through qmm16c (Leap's
  fixed 16-row tile). Everything else runs through MLX's own 16-row call, padded.
- **QMM32-INV first.** No shape was bitwise for MLX's raw 17-32-row call. The reference was also wrong: its second
  chunk was not padded to 16 rows. Read with that fixed:
  - big N: MLX's 32-row call matches 16-row MLX calls, but REF2 does not use MLX there;
  - small N: MLX switches kernels at 25 rows (leapk routes_splitk's limit). So 17-24 rows match and 31-32 do not.
  - A [2, 16, K] batch is no help: MLX folds it into 32 rows.
- **MLX's 16-row kernel on the small shapes is qmv_wide at k_lanes 8.**
  - Its 16-row output does not match its own 1-row call (qmv; 0 / 16 rows at 4-bit).
  - A JIT build of qmv from the installed quantized.h does match MLX's 1-row call (16 / 16), so a JIT build is
    faithful to the shipped kernels.
  - Searching k_lanes against MLX's 16-row output: k_lanes 8 gives 16 / 16 on every small shape. vecs_per_tg does not
    matter: each vector's arithmetic depends only on k_lanes.
- **The one-dispatch forms** (17 / 20 / 24 / 31 / 32 rows x 3 seeds = 372 rows per shape, vs two padded 16-row calls):

  | form | shapes | rows bitwise |
  |---|---|---|
  | qmm16c2 (qmm16c with a 2nd row tile on the grid's y axis) | GDN qkv\|z 16384x2560, out / o_proj 2560x6144, q_proj 12288x2560, shared down 2560x640, PLE key / value, lm_head | 372 / 372 each |
  | qmvw32 (MLX's qmv_wide_impl, k_lanes 8, over 32 rows) | GDN b\|a 96, k / v_proj 512, indexer qk 640, router 512 (8-bit), K-73 shared gate\|up\|gate 1281 (K 2560) | 372 / 372 each |
  | MLX raw n-row call, for contrast | the small shapes | 183 / 372 |

- **So** no quantized matmul needs a second dispatch at 17-32 rows. Next: wire these into a one-pass v32 path
  (opt-in, so BigMoE's live import is unchanged), plus the GDN body and the HC read as two-tile kernels, then a C3 slice
  bit test and a C6-style timing.

## ONE32 GDN body (CHECK55 step 1, Sun 11:02; share lock, bits only): 17-32 GDN rows in one dispatch, bitwise
- **gdn_body32** (scripts/week98/one32.py): gdnbody's one-dispatch body, looping over 16-row tiles inside the dispatch.
  32 rows of q / k / v / out would pass the 32 KB threadgroup limit.
  - Each row's prework, recurrence step and gated norm are the 16-row body's expressions verbatim.
  - The fp32 state stays in registers across the tiles.
  - The conv window, gates and I/O are indexed by global row.
- **Test** (one32_gdn_test.py): vs gdn_body on rows 0-15, then the rest from chunk 1's conv and recurrent state.
  - **ALL BITWISE** on y, q, k, v, the conv state and the state at 17 / 20 / 24 / 31 / 32 rows x 3 seeds.
  - 9 and 16 rows (one tile) also match one gdn_body call.
- **Note:** the exact-45 preset runs the unfused REF2-T GDN chain at 16 rows, not GDNB. gdn_body is bitwise to that chain
  (GDNB's own tests), so the one-pass path can use gdn_body32 against either.

## ONE32 HC read (CHECK55 step 1, Sun 11:03; share lock, bits only): the HC read at 17-32 rows with one dispatch per kernel, bitwise
- **hc_read32b** (scripts/week98/one32.py): hc32's 32-row read with **s4_partials2** in place of S4x2.
  - s4_partials2 is leapk's own 16-row NAX split-K tile code with a second row tile on the grid's y axis (its own
    threadgroups; the weight tiles re-read from cache).
  - S4x2 used two accumulators per simdgroup and was slower (C6).
- **Test** (one32_hc_test.py): vs leapk.hc_read (fuse_gate) on rows 0-15 and rows 16-n zero-padded to 16, at the 125B's
  HC shapes (4 x 2560, low rank 320, block inject 4).
  - **ALL BITWISE** on mixed, written and inject at 17 / 20 / 24 / 31 / 32 rows x 3 seeds, with and without the
    pending carry.
- Its speed against two 16-row reads is timed with the wired path (a quiet ticket).

## C3-ONE (CHECK55 step 1, Sun 11:07; share lock, bits only): the one-dispatch 17-32-row check is exact on the 4-layer slice
scripts/week98/v32one.py (install(v32, ref2) after v32.install; no loop installs it), c3_v32.py with V32_MOE1=1 V32_ONE=1.
Data: c3_one.txt.
- **What runs once at 17-32 rows:**
  - quantized matmuls: qmm16c2 / qmvw32, mirroring ref2's routing;
  - HC reads with block inject: hc_read32b;
  - GDN layers: R-83's concats + gdn_body32 + the stash + out_proj;
  - attention at exactly 32 rows: the module once, with ref2's class logic at 32 rows on 2-head chunks;
  - the experts: one call (V32_MOE1).
  The head mixer, 17-31-row attention and unlisted matmul shapes keep v32's split.
- **Result, ALL BITWISE:**
  - T1 forward: 32 / 32, 17 / 17, 24 / 24 and 31 / 31 rows bitwise vs <= 16-row checks (max diff 0).
  - T2 commit: keep 1 / 9 / 16 / 17 / 24 / 31, then a 4-row probe: 4 / 4 each.
  - At M 32 the wrappers fired: qmm16c2 10, qmvw32 14, hc1 8 (head mixer split 1), gdn1 3, att1 1, vsdpa32 1.
    Still split: 5 small matmuls and 2 bf16 Linears.
- **Next:** time it, a C6-style paired run on the 12-layer slice under a quiet ticket: A16, B2x16, C32 split, C32 one.

## C7 ONE32 RESULT (Sun 11:13, quiet ticket, power mode 2, NO-SWAP 43 GB; pre-registered 91dd8a1): one-dispatch 32-row check 1.43 x A16 on the slice, PARTIAL
12-layer resident slice, SB every 4 layers, each arm on its own cache advancing 32 per round, 14 rotated rounds. Data:
c7_one32.txt, c7_one32_pmode.txt.

| arm | ms per check | build (share) | x A16 |
|---|---:|---:|---:|
| A16 | 23.00 | 12.25 (0.53) | 1.000 |
| B2x16 | 46.25 | 23.71 (0.51) | 2.011 |
| C32s (v32's split, experts one call: today's path) | 37.95 | 22.06 (0.58) | 1.650 |
| C32o (v32one: one dispatch per op) | 33.08 | 18.32 (0.55) | **1.438** |

- **Per-round medians [IQR]:** R32s 1.574 [1.515-1.718]; **R32o 1.430 [1.351-1.526]**; saving (C32s - C32o) / A16
  **0.172** [0.117-0.201].
- **Verdict (pre-registered):** 1.35 < R32o <= 1.50, **PARTIAL**: find what is still split.
- **Still split in C32o, per check (counters):**
  - the shared-expert gate, N 1 (12). It is slice-only: the engine's K-73 folds it into the N-1281 pack.
  - a 640 x 2560 8-bit matmul, 1 per attention layer (3);
  - the head mixer's HC read and its 2 bf16 Linears (1).
  - Everything else ran once: qmm16c2 28, qmvw32 42, HC reads 24, GDN layers 9, attention 3, the 32-row SDPA 3.
- **Reading:**
  - The one-dispatch forms cut the build by 3.7 ms and the check by 4.9 ms. That is ~0.17 of the ~0.57 x A16 step at
    17 rows.
  - Left above A16: build +6.1 ms (C32o 18.3 vs 12.3) and the GPU tail +4.0 ms (14.8 vs 10.8).
  - Part of that is inherent: the expert union grows ~+54 % from 16 to 32 rows, and the GDN recurrence and the
    row-tiled matmuls do twice the rows.
  - What the rest is (wrapper Python, graph size, GPU compute) needs Breakthrough's step-0 profile with C32o in it.
- **Next:**
  1. Close the remaining splits: the 8-bit 640 x 2560 (find MLX's k_lanes) and the head mixer.
  2. The engine arm: BigMoE, v32one on vs off.
  3. Breakthrough profiles C32o next to A16 / C32s.

## ONE32 one weight pass (CHECK55 step 1, Sun 11:20; share locks, bits only)
After Breakthrough's step 0 (the check is GPU-bound; 2 x 16 reads every dense weight twice): the two-tile forms now
read each weight once.
- **qmm16c2a:** qmm16c (KS 1, REF2's call) with both 16-row tiles fed by each decoded weight fragment. Each tile keeps
  its own A fragment, accumulator, row sums and output rows.
- **qmv_wide (qmvw32) at 32 vectors per threadgroup:** each weight row is decoded once for all 32 rows. Per-vector
  arithmetic depends only on k_lanes.
- **Bits** (one32_test.txt; 17 / 20 / 24 / 31 / 32 rows x 3 seeds vs two padded 16-row certified calls): qmm16c2a
  372 / 372 on all 7 qmm16c shapes; qmvw32 at 32 vectors 372 / 372 on all small shapes.
- **Four more small shapes** take qmv_wide at k_lanes 8, each 372 / 372: the 8-bit 640x2560, N 1 4-bit / 8-bit, N 48.
  - MLX's k_lanes was probed at 16 rows. For N 1 / 48 every k_lanes matched there (bf16 rounding hides the fp32
    differences), so the 17-32-row test is their check.
  - There, plain qmv got 73 / 372 on the N-1 4-bit shape and kl8 got 372.
- **v32one** now uses both by default (W1, QMV_VPT 32). C7b (pre-registered) times them: the weight pass and the
  seams to 32 attributed separately.

## C7b RESULT (Sun 11:24, quiet ticket, power mode 2; pre-registered 4476141): the weight pass as built LOSES; the seams give 0.05 x A16. Verdict: profile before E2
Same harness as C7. C3 first, on the same ticket, with the one weight pass + seams to 32: **ALL BITWISE** (c3_one_w1.txt).

| arm | ms | x A16 (per-round median [IQR]) |
|---|---:|---|
| A16 | 23.07 | 1 |
| C32s (v32 split, seams 16) | 37.70 | 1.628 [1.549-1.760] |
| C32o1 (v32one as in C7: qmm16c2, qmv_wide 4 vectors, seams 16) | 33.38 | **1.451** [1.371-1.521] (C7: 1.430) |
| C32w (+ one weight pass: qmm16c2a, qmv_wide 32 vectors) | 38.59 | 1.620 [1.579-1.726] |
| C32o (C32w + seams to 32) | 36.40 | 1.566 [1.520-1.641] |

- **Paired attribution / A16:**
  - the weight pass (C32o1 - C32w): **-0.219** [-0.240 to -0.108], so it COSTS ~5 ms;
  - seams to 32 (C32w - C32o): +0.053 [0.045-0.073], ~1.2 ms (Breakthrough's wrapper: 2.0);
  - all of step 1 as built (C32s - C32o): +0.020.
- **Verdict (pre-registered):** R32o 1.566 > 1.45, neither paid on the slice; profile the new kernels before E2.
- **Why the weight pass lost:**
  - qmm16c2a pairs both row tiles inside one simdgroup. That halves the threadgroups of qmm16c2: N/32 tiles of 32
    threads, only 80 for the N-2560 shapes.
  - qmv_wide at 32 vectors cuts its threadgroups 8x (12 for b|a).
  - These kernels are latency- and occupancy-bound at these sizes, so lost parallelism costs more than the second read.
- **Next:**
  - keep both tiles' threadgroups but make the row-tile index the fastest grid axis, so the two threadgroups of a
    column tile run together and the second weight read is a cache hit (qmm16c2 and the HC S4 currently run all of
    tile 0's grid, then all of tile 1's);
  - keep qmv_wide at 4-8 vectors;
  - a per-shape microbench on cache-cold rotating weights picks each form before any slice timing.

## ONE32-BENCH (Sun 11:28, quiet ticket): per-shape forms on cold weights; C7b's loss was the small kernels at 32 vectors
scripts/week98/one32_bench.py, data: one32_bench.txt. Each shape streams the engine's per-check set of distinct weights
(cold reads); ms per 32-row check for that shape's calls, 7 interleaved rounds.

| shape (calls / check) | 16-row | split | two tiles (c2) | interleaved (c2i) | paired one pass (c2a) | best (R32) |
|---|---:|---:|---:|---:|---:|---|
| GDN qkv\|z (36) | 4.45 | 9.05 | 5.69 | 5.66 | **5.25** | c2a (1.18) |
| out / o_proj (48) | 2.54 | 5.15 | 3.53 | 3.50 | **2.94** | c2a (1.16) |
| q_proj (12) | 1.57 | 2.72 | 1.77 | 1.77 | **1.66** | c2a (1.06) |
| shared down (48) | 0.72 | 1.34 | **0.72** | 0.72 | 0.78 | c2 (1.01) |
| lm_head (1) | 1.88 | 3.54 | 2.44 | 2.35 | **2.26** | c2a (1.20) |

| small shape (calls / check) | 16-row | split | 2 | 4 | 8 | 16 | 32 vectors |
|---|---:|---:|---:|---:|---:|---:|---:|
| router 8-bit (48) | 1.07 | 1.92 | 2.26 | 1.84 | **1.66** | 4.18 | 5.12 |
| shared gate\|up\|gate (48) | 1.73 | 3.34 | 3.85 | 3.23 | **2.91** | 8.76 | 9.96 |
| k / v_proj (24) | 0.68 | **0.94** | 1.21 | 1.16 | 1.34 | 2.58 | 3.20 |
| GDN b\|a (36) | 0.51 | 0.71 | **0.71** | 0.97 | 1.02 | 1.70 | 3.05 |

- **HC S4, the mixer GEMM (96 calls):** 16-row 3.20 | split 6.01 | tiles on y 3.84 | **interleaved 3.22**
  (R32 1.01: the second read is ~free).
- **Total for these calls per 32-row check:** 16-row 19.75 ms | split 36.81 | C7's forms ~27.1 | **best forms 24.12**.
- **Reading:** the paired one-weight-pass form wins every big shape. C7b's +0.22 x A16 came from the small kernels at
  32 vectors per threadgroup (3-5x slower). For k / v, MLX's own two 16-row calls stay fastest.
- **v32one now picks per shape** (W1 = the bench's best):
  - qmm16c2a for weights >= 4 MB, else qmm16c2;
  - qmv_wide at 2 / 8 vectors per the table, k / v on MLX's split;
  - the interleaved S4.
  W1 False = C7's forms. Slice timing (C7c) is next, after the PAUSE (Kamen's lid, 11:3x).

## TREE16 L1 (Sun 14:36, share lock, bits only): the GDN depth-first body is exact
scripts/week98/tree_gdn.py (gdn_body_tree), tree_gdn_test.py. Data: tree_gdn_test.txt.
- **The kernel:** gdnbody's 16-row body with
  - each node's conv taps from the tree's ancestor table;
  - the recurrence over the nodes in DFS order, the state restored from a device scratch slot before a non-first child
    and saved after a branching node (fork_tables; B1's fork_ops uses the same numbering);
  - phase C unchanged.
- **Test:** 10 trees (a chain, 2 bushy, a deep-narrow, 6 random; up to 5 slots and 8 paths) x 2 seeds. Every node's
  y / q / k / v vs gdnbody.gdn_body run on each root-to-leaf path as a chain: **ALL BITWISE**.

## ATT-HCHUNK (Sun 14:37, share lock, bits only): larger head chunks are not safely bitwise
scripts/week98/att_hchunk_inv.py. Data: att_hchunk_inv.txt. ref2's class logic with 2 / 4 / 6 / 12 query heads per
chunk, vs the certified forms; 13 contexts 60-4,500 x 3 seeds.

| chunk | rows per GEMM (16 / 32-row) | 16 rows | 32 rows |
|---|---|---|---|
| h2 | 32 / 64 | 624 / 624 | 1,248 / 1,248 |
| h4 | 64 / 128 | 624 / 624 | 1,247 / 1,248 |
| h6 | 96 / 192 | 624 / 624 | 1,247 / 1,248 |
| h12 | 192 / 384 | 623 / 624 | 1,247 / 1,248 |

- **Reading:**
  - Up to 64 rows per GEMM (h2 at 32 rows, h4 at 16 rows) every row is bitwise.
  - Above that a row occasionally differs, likely MLX's split-K partitioning changing with the size.
- **So:** the 32-row one-call attention stays on h2 (as v32one does). h4 at 16 rows (12 -> 6 calls per layer) is in
  the bitwise range, but it would change the exact reference's own code path. It needs a much larger real-data test
  first, so it is parked as a candidate lever.

## QMVW16 (Sun 14:38, share lock): 16-row small kernels with more threadgroups. KILL
scripts/week98/qmvw16_bench.py, data: qmvw16_bench.txt. MLX's qmv_wide (k_lanes 8) launched by us at 1-16 vectors per
threadgroup vs MLX's own 16-row call. Bits are equal (k_lanes alone sets each row's arithmetic).
- **Per-call GPU time:** MLX's launch is already at or near the best on every shape.
  - Ours at the best vector count: router 1.14x, shared pack 0.98x, k / v 0.84x, b|a 0.54x, indexer 0.86x, 8-bit 640
    0.91x (higher = faster than MLX).
  - Per 16-row check: MLX 4.12 ms vs our best 4.33 ms.
- **CHECK55 lever (a): KILL** (the router alone would save ~0.16 ms per check).

## TREE16 L2 step 1 (Sun 14:40, share lock, bits only): MLX's split-K P·V replicated; its partitioning pinned
scripts/week98/tree_att.py (pv_replica: NAX 16 x 32 x 32 k-steps per partition, fp32 partials summed in order),
tree_att_probe.py. Data: tree_att_probe.txt. Probe: probs [1, 1, 2, rows, K] @ v [1, 1, 1, K, 256] (the exact SDPA's
P·V) via mx.matmul vs the replica at candidate partitions, 2 seeds per K.

| K (keys, run length) | partitions MLX uses (bitwise match, 16 and 32 rows) |
|---|---|
| 96 / 128 / 384 / 512 | none (one partition = the fused GEMM's arithmetic) |
| 1,024 | 512 + 512 |
| 1,536 | 1,024 + 512 |
| 2,048 | 1,024 + 1,024 |
| 2,560 | 2,048 + 512 |
| 3,072 | 2,048 + 1,024 |
| 4,608 | 4,096 + 512 |

- **Rule:** K <= 512, no split. K > 512, two partitions [0, P) and [P, K), with P = next_pow2(K) / 2. Single extra
  matches (np8 at 1,536 / 32 rows; ps256 at 512 / 16 rows) are coincidences that no other K backs.
- **So the replica's per-step arithmetic is MLX's.** Step 2 adds per-row V tails in the k-steps from the cache end L.
  Those can straddle P (e.g. N 520 runs at K 1,024, P 512, L 504), so the per-row path covers any k-step >= L in either
  partition.

## TREE16 L2 (Sun 14:45, share lock, bits only): attention route (b) is exact. Every node == its path's chain row
scripts/week98/tree_att.py (tree_sdpa, pv_tree), tree_att_test.py. Data: tree_att_test.txt.
- **Route (b)** per ref2 AR class (keys L + depth + 16), at the class's run length:
  - MLX's own scores (q * scale @ k^T, 2-head chunks, queries padded to 16 rows as ref2 pads short checks);
  - each node's columns gathered into its chain order;
  - MLX's where (dtype min) + precise softmax at the exact run length;
  - pv_tree: MLX's split-K NAX nn GEMM replicated (partitions pinned in step 1). The k-steps from the cache end run
    once per row, with that row's chain-order V and the other rows of A zeroed.
- **Test:** vs ref2._verify_sdpa run on each node's path as a 16-row chain check (path rows + copies of the last; K / V
  = cache + path nodes + padding copies). 7 trees (chain, bushy, deep-narrow, 3 random at T 16, one T 7) x 10 contexts
  (60-2,030: class 0, the band and its edges, the 512 buckets) x 2 seeds.
  - **ALL BITWISE**, every node, every query head.
- **One fix on the way:** at T 7 the scores GEMM ran on 14 rows and gave different bits from 200 keys up (MLX's kernel
  depends on the rows). Padding the queries to 16 rows, as ref2 does for short checks, made it exact.
- **Limits:** contexts <= 2,048 tokens (every cache block selected). Above that the per-node QSA selection (B3) adds
  its mask; the per-row machinery is unchanged.

## TREE16 L3 (Sun 14:51, share lock, bits only): tree_verify end to end. Every node's logits + hidden_w == its path's chain verify
scripts/week98/tree16.py (tree_verify + install), tree16_test.py. Data: tree16_test.txt.
- **What tree_verify does:** inside the verify scopes, under a tree context:
  - attention: MTPLX's forward with RoPE at L + depth, the indexer's pre-RoPE raw keys written as usual, the K / V
    appended in node order, and the core through L2's route (b);
  - GDN: R-83's in-proj, L1's depth-first body, the stash per node;
  - PLE: B2's ple_tree once it lands (chain-shaped trees take the stock forward until then);
  - row-local ops as a T-row check.
  It returns (logits, hidden_w) via return_hidden.
- **Test:** a 4-layer slice WITHOUT the PLE layer (layers 4-7, GGGA; MTPLX's PLE stage off). 8 trees (chain, 2 bushy,
  deep-narrow, 3 random at T 16, and T 7) from fresh 1,024-token prefills, vs one chain verify per root-to-leaf path.
  - **ALL BITWISE:** logits and hidden_w of every node == its path's chain row.
  - B1's tree_meta (Breakthrough b78708b) built the metadata, imported read-only.
- **Leap's TREE16 pieces (L1-L3) are done on Sun.** Open:
  - B2 (the PLE layer; Breakthrough, Tue) plugs in through tree16.PLE_TREE[0];
  - B3 (contexts > ~2K);
  - BigMoE's M3 harness on slices with the PLE layer.

## TREE16-TIME (Sun 15:00, quiet ticket, power mode 2; pre-registered b34065c / 6fdfb78): a tree check costs +4 % engine-scaled with batched heads
scripts/week98/tree16_time.py, data: tree16_time.txt. 4-layer PLE-free slice (layers 4-7), SB 4, 24 rotated rounds,
context 1,024; each check = verify forward + acceptance read (rolled back to the root after).

| arm | ms per check | paired vs C16 [IQR] |
|---|---:|---|
| C16, a 16-row chain verify | 10.313 | |
| Tc, tree_verify on a chain-shaped tree (route (b) as 12 head chunks) | 11.218 | +0.769 ms [+0.587, +1.213] |
| Tb, a bushy tree (3 forks), chunked | 11.231 | +0.922 ms [+0.760, +1.351] |
| **Tbb, Tb with heads batched per kv head** | **10.412** | **+0.208 ms** [-0.002, +0.282] |

- **Verdict (pre-registered, on Tbb):** engine-scaled (x12: 48 / 4 layers, the head in both arms) +2.50 ms = +4.0 % of
  a ~62 ms chain-16. In the 2-5 % band: trim route (b) before M4.
- **Reading:**
  - The chunked route (b) ran ~10 ops per 2-head chunk x 12 chunks per attention layer: +0.77 ms per 4 layers.
  - Batching each kv head's 12 query heads into one scores matmul / gather / softmax / pv_tree (ALL BITWISE,
    tree_att_test_batch.txt) cuts the overhead 4.4x. It is the default now.
  - The forks cost ~nothing (Tb vs Tc).
- **Next trim (before M4):** pv_tree's tail k-steps run one MMA per row (16). Grouping rows by the root-to-leaf path
  that owns them (B1's owner) gives one MMA per path (~4 for an EAGLE-2-16 tree). Also share the V tiles across a kv
  head's 12 row tiles in one threadgroup.

## TREE16 trims + end to end (Sun 15:02)
- **pv_tree's tail k-steps** now run once per root-to-leaf path (the rows each path owns, its chain-order V) instead
  of once per row: ~4 steps for an EAGLE-2-16 tree, not 16.
  - ALL BITWISE in tree_att_test with chunked heads and with batched heads (tree_att_test_path*.txt).
  - Re-timed next (TREE16-TIME v2, same lines).
- **Breakthrough's end-to-end run with the PLE layer (8d37798): ALL BITWISE.** tree_verify + B1 + B2 on layers 0-3
  (GGGA, the PLE layer at 1), 10 trees incl. two with EOS tokens: logits + hidden_w of every node == its path's chain
  verify.
  - Every layer type of the 125B now runs exact in tree mode: GDN, PLE, attention below ~2K context, and the MoE with
    one expert call.
  - Above ~2K: B3 (Mon-Tue).

## TREE16-TIME v2 (Sun 15:06, quiet ticket; same pre-registration): +2.5 % engine-scaled with the per-path tail
Data: tree16_time_v2.txt. Same harness, 24 rotated rounds. C16 10.853 ms | Tc 11.393 | Tb 11.606 | **Tbb 10.984**.
- **Paired vs C16:** Tc +0.510 ms; Tb +0.749 ms; **Tbb +0.129 ms** [-0.008, +0.249].
- **Verdict:** engine-scaled +1.55 ms = **+2.5 %** of a ~62 ms chain-16 (v1: +4.0 %). Just inside the 2-5 % band, at
  the top of the TREE16 price (+0.5-1.5 ms).
- **Net vs MTP Head's S9c kept gains** (x1.136 MBPP / x1.060 HE+): about +11 % / +3.5 %.
- **Further trims are optional:** share V tiles across a kv head's 12 row tiles in one threadgroup, fuse
  gather + where + softmax.
- Absolute ms drift between runs (C16 10.31 in v1, 10.85 here); only the within-run pairs compare.
- **Sun 15:2x, BigMoE:** M3 re-run on the trimmed route (b) (8b1ff70: heads batched + per-path tail steps), live
  imports as M4 will use them. T1 ALL BITWISE, T2 73 / 73, on layers 4-7 (50 s, 8.1 GB) and layers 0-3 with B2
  (129 s, 8.2 GB).
- **Sun 17:0x, BigMoE M4 integration (BigMoE 7f4b4e0):** the tree pieces are frozen at de7fcc0 (tree16.py,
  tree_att.py, tree_gdn.py; route (b)) in snapshot t16b behind R57_TREE16=1 (draft -> tree_verify -> walk ->
  tree_commit -> t30b's MTP pairs; runs.json records each file's hash). Slice loop test: 200-token decodes over many
  checks with random trees; every kept row's logits and stream == REF2's 1-row AR bit for bit, on layers 0-3 (with B2)
  and 4-7. The engine A/B (chain-16 vs tree-16, 45 GB) opens Monday night (~23:05, hard end 00:25). Any change to the
  three files goes to BigMoE first (they re-freeze).

## CB1 (Sun 17:35, quiet ticket): the reference-contract bench (CHECK55 track B). PASS on both lines, ALL BITWISE
scripts/week98/cb1.py; pre-registered in check55_plan.md 'CB1' (8746000). Data: cb1.txt (smoke: cb1_smoke.txt).
- **Setup:** per real shape, a fixed K-chunk form vs REF2-T's form today.
  - Cold weights: pools of at least 128 MB, rotated each round.
  - A dependent chain of one check's calls; per call = (chain - its links) / n; 7 rounds, the forms' order rotated.
- **Contract forms:** qmm16c and the native-layout qmm16n at 16 rows; NEW qmm16c2a_ks at 32 rows (one weight pass with
  qmm16c's K split); S4 at a partition size PS.
- **Bits: ALL BITWISE**, 376 config-checks. Every config is bitwise across M = 1 (zero or random padding) / 16 / 32,
  across nt, and qmm16c == qmm16n at the same ks; S4 at every PS: 16 == 32 (2i) == M 1, and PS 4,096 == leapk's S4.

| shape (calls per check) | REF2-T 16 / 32 (us) | contract: ks | 16 rows (us, form) | 32 rows (us, form) |
|---|---|---|---|---|
| GDN qkv \| z (36) | 118.5 / 143.2 | 4 | 100.3 (native, nt 1) | 139.9 |
| GDN b \| a (36) | 7.9 / 9.8 | folded into the in-proj (ks 4: 101.8 / 140.4) | | |
| o_proj + out_proj (48) | 87.1 / 112.1 | 4 | 43.7 (native, nt 2) | 53.7 |
| QSA qkv_fused (12) | 108.4 / 133.5 | 4 | 90.4 (native, nt 2) | 124.6 |
| shared cat (48) | 32.0 / 54.3 | 5 | 13.6 | 18.7 |
| shared down (48) | 11.0 / 11.2 | 10 | 7.0 | 10.3 |
| lm_head (1) | 1,653.7 / 2,076.2 | 4 | 1,457.5 | 2,018.4 |
| HC down, S4 (97) | 49.1 / 49.5 (PS 4,096) | PS 2,048 | 33.5 | 36.9 |

- **Per check over this set:**
  - 16 rows: 18.52 -> 12.54 ms (**-5.98**; PASS line -4.0);
  - 32 rows: 22.52 -> 16.11 ms (**-6.41**; PASS line -6.0).
- **Where it comes from (16 rows):**
  - o_proj / out_proj at ks 4: -2.1 ms;
  - S4 at PS 2,048: -1.5;
  - the shared cat at ks 5: -0.9;
  - the in-proj at ks 4: -0.65;
  - QSA, shared down, lm_head, b | a: -0.2 each or less.
- **Against the prediction** (16 rows -5..-7, 32 rows -6..-10): 16 rows inside the range; 32 rows at its low end.
  - REF2-T's o_proj measured 87 us, not the 115 I assumed (that was a 2-row figure in the native layout).
  - The 32-row gain is smaller because v32one's one-pass forms already cut REF2-T's 32-row cost.
  - The 32-row contract forms are MLX-layout only. The native layout won at 16 rows on the big shapes, so a native
    one-pass 32-row twin is the next kernel to try.
- **M = 1** (AR, chat, the commit's 1-row calls): the contract pads to 16 rows, as REF2-T does today, so the 1-row
  path speeds up by the same amount.
- **Not in this set:** the 8-bit router and indexer (~1.7 ms), HC up, PLE, the experts, the attention core.
- **Transfer rule:** these are standalone numbers. DN-KS went -7.7 standalone -> -4.2 on the engine (2-row checks), so
  expect roughly half on the engine. Next: a slice A/B with an opt-in contract installer over ref2 (A16 and C32), then
  next week's re-certification plan.

## TREE16 B3 wired in (Sun 17:4x, share lock, bits): ALL BITWISE at 1K and 3K
Breakthrough's B3 (per-node QSA selection above ~2K tokens; qsa_tree bd0695c, its end to end 281470d) plugs into
tree16 through a new slot, like B2's PLE_TREE: `tree16.QSA_TREE[0] = qsa_tree`, set by the caller (BigMoE installs it
from its snapshot copy).
- **The slot set:** the stock indexer call runs in MTPLX's history-only mode (raw keys + pooled blocks, no selection).
  Then qsa_tree.tree_select gives each node's mask, and tree_att.tree_sdpa(sel=...) takes route (b)'s mask from it
  (route_b_mask inlined as _b3_mask).
- **The slot unset:** de7fcc0's code path (the stock call; a raise when it selects).
- **Tests:**
  - tree_att_test (route (b), sel None): ALL BITWISE (tree_att_test_b3w.txt).
  - tree16_test, layers 4-7, slot unset: ALL BITWISE (tree16_test_b3w.txt).
  - tree16_b3_test, layers 0-3 with B2 and B3 through the slots, no copies, P 1024 / 3000 / 3002: ALL BITWISE over 453
    nodes. 20 live selections (3K), 10 dense (1K: B3 returns None, and the history-only call keeps the bits); 179 s,
    9.4 GB (tree16_b3_test.txt).
- **Sun 17:4x, BigMoE (648826b):** with the QSA_TREE slot set, tree_commit composes with the history-only writes. The m4loop
  slice test was bitwise vs REF2 AR over 200-token decodes at P 1024 / 2030 / 3000, on layers 0-3 (B2 + B3) and 4-7, using
  live pieces (d6b5884 + qsa_tree d138471). B3 was dense on 60 / 60 checks at 1K, live on 58 / 60 at 2,030 and 60 / 60 at
  3K. Their commit gathers the raw keys in path order and caps pooled_len at L // 4, so the next indexer call rebuilds
  those blocks. Monday's M4 stays on t16b; t16c (slot + qsa_tree) is frozen after it.

## D8 (Sun 17:5x, share lock, bits): a correctness bug in v32's one-call experts at 17-32 rows (resident path), FIXED
C8's (b1) failed on the 12-layer slice: one 32-row check matched two 16-row checks on rows 0-15 only. REF2-T failed the
same way as the contract, so the contract wasn't the cause. scripts/week98/d8_bits.py localized it (d8_bits.txt,
d8_bits_v2.txt):
- **Ruled out:** seams (16 / 32), the mid-forward submits, and v32one on, off or with C7's forms. Every one of those
  variants failed identically.
- **Found:** with V32_MOE1 off (MoE block + experts as 2 x 16) it is 32 / 32. Per-call captures: the experts' output
  first differs at layer 8, rows 26 / 29 / 30; the shared expert only from layer 9.
- **Cause:** qmmg25's grouped expert GEMM runs each expert's (row, expert) pairs (gmr24.segments) as ONE 16-row tile
  and stores at most 16 of them. At <= 16 rows that always fits. At 17-32 rows an expert picked by more than 16 rows lost
  its 17th+ pairs: never computed, zeros. v32's V32_MOE1 resident branch (experts_nax / ref2.experts_split at 17-32 rows)
  had it.
- **Not affected:** the streamed exact-45 engine's 32-row block, which runs BigMoE's tier kernels (G32T: C32 == 2 x 16 on
  the engine), and every <= 16-row path, including TREE16.
- **Fix:** moe32.py cuts every expert's run into segments of <= 16 pairs (segments_cap; experts_nax_cap /
  experts_split_cap), and v32's moe1 branch uses them. ref2, qmmg25 and gmr24 are untouched.
- **Tests:**
  - moe32_test, expert 7 in 20 of 32 rows: old 316 / 320 pairs (the 4 past the tile lost); capped 320 / 320 for every
    config, fused and split; == experts_nax at <= 16 rows (moe32_test.txt).
  - 12-layer slice: b1 32 / 32 with V32_MOE1 on; all 12 expert / shared / block calls bitwise (d8_bits_fix.txt).
- **Impact:** the slice's 32-row timings (C7 / C7b / C7c, Breakthrough's slice profile) computed the same tiles and
  only skipped stores, so their times stand. Their 32-row bits at depth were wrong whenever an expert passed 16 rows.
  C3 / C3-ONE ran the experts as 2 x 16 on 4 layers and were not affected.
- **The engine, confirmed by BigMoE (18:0x):** exact-45's 32-row experts run NAX6 (tierglu_nax6.py, R57_NAX=1) through
  bigmoe.tier's OOO phases. Its grouping (_GRP / _GRPP, and the CPU fallback groups(cap=16)) starts a group at each new
  expert and every 16 pairs inside one, so 20 pairs run as 16 + 4. Without NAX, the tier glu runs one pair per
  threadgroup. So the engine never had this bug, by design; its 32-row limit is the slot pool (the G32 overflow guard).

## C8 REFC v2 (Sun 18:01, quiet ticket): the contract carries into a real check. PASS, ALL BITWISE
scripts/week98/c8_refc.py (pre-registered in check55_plan.md 'C8 REFC'), with the D8 fix. Data: c8_refc_v2.txt (v1, void
bits: c8_refc.txt).
- **Bits, every logits row, the 12-layer slice:**
  - under the contract: one 32-row check == two 16-row checks (32 / 32), and 16 one-row checks == one 16-row check
    (16 / 16). SD == AR end to end under the new reference.
  - REF2-T control: the same (32 / 32, 16 / 16).
- **Timing** (14 rotated rounds; wall per check with the Python build inside, about 12 of 23 ms at 16 rows):

| arm | ms per check | x A16 |
|---|---|---|
| A16 (REF2-T) | 23.37 | 1.000 |
| A16c (contract) | 22.12 | 0.947 |
| C32o (REF2-T, step 1 complete) | 32.12 | 1.375 |
| C32oc (contract) | 30.69 | 1.314 |

- **Paired:**
  - 16 rows -1.31 ms (IQR -3.70 / +4.27: the build's noise);
  - 32 rows -1.05 ms (IQR -1.71 / -0.86).
- **Transfer against CB1's standalone figures for the slice (upper bounds: MLX layout here):** T16 0.81, T32 0.67
  -> **PASS: the contract arms join Monday night's E2.**
  - Engine-scaled (per-layer parts x4, the head once) that is roughly -4..-4.5 ms per 16-row check and ~-4 per 32-row.
  - The engine A/B decides.

## BOF (Sun 18:08, quiet ticket): lever (b) as a blocked ordered fold. PASS, REF2-T's OWN BITS
scripts/week98/bof.py (qmm16bof) + bof_bench.py; pre-registered in check55_plan.md 'BOF' (4a4fc62). Data: bof.txt,
bof_bits.txt.
- **What it is:** qmm16c ks 1's K loop spread over bs simdgroups per 32-column tile.
  - Each block of bs x rg groups is computed in parallel: the same MMA, row sums and term expression as qmm16c.
  - Simdgroup 0 folds the block's terms in group order, which is qmm16c's fold.
- **Bits:** TERM 0 (the term written as qmm16c writes it) and TERM 1 (fma(s, P, b X)) are bitwise vs qmm16c ks 1 on all 5
  shapes x 4 configs, random padding included. TERM 2 / 3 are not. So qmm16c's compiled term is fma(s, P, b X).
- **Timing** (cold pools, dependent chains, 7 rotated rounds; per call):

| shape (per check) | REF2-T ks 1 | BOF best (bitwise) | contract ks (for reference) |
|---|---|---|---|
| GDN qkv \| z (36) | 115.6 us | 114.5 (bs 4) | 104.3 |
| o_proj / out_proj (48) | 85.9 | **48.6** (bs 4) | 46.0 |
| QSA qkv_fused (12) | 107.0 | 101.4 (bs 8) | 93.6 |
| shared down (48) | 10.8 | 7.4 (bs 4) | 7.2 (ks 10) |
| lm_head (1) | 1,612.7 | 1,454.0 (bs 4) | 1,425.2 |

- **Per 16-row check over these shapes:** 11.70 -> 9.48 ms (**-2.22**; PASS line -1.5). The contract's forms give 8.85
  (-2.85), so BOF keeps 78 % of that gain without changing the reference.
- **Against the prediction:** -1.8..-2.5 predicted, -2.22 measured. o_proj predicted 50-60 us, measured 48.6. The
  in-proj, predicted 105-110, barely moved: it already has 512 tiles.
- **Next:**
  - an opt-in installer (no ref2 edit) and a slice A/B (transfer rule), then an E1 arm Monday night;
  - a 32-row twin (one weight pass, both tiles) for the 32-row check.

## C9 BOF (Sun 18:14, quiet ticket): BOF in a real check is EXACT; transfer BETWEEN by the pre-registered line
scripts/week98/c9_bof.py with bofi.py; pre-registered in check55_plan.md 'C9 BOF' (0a8c085). Data: c9_bof.txt (smoke:
c9_bof_smoke.txt).
- **Bits:** A16b (BOF on o_proj / out_proj, QSA, shared down, lm_head) == A16 (REF2-T) on every logits row (32 / 32,
  two 16-row checks). BOF stays REF2-T's arithmetic inside the model.
- **Timing, 24 rotated rounds:**
  - A16 24.62 ms, A16b 23.38 ms.
  - Paired whole wall: **-0.30 ms** (IQR -5.60 / +4.30). The Python build is about half the wall and dominates the spread.
  - **Pre-registered verdict:** T = 0.30 / 0.66 = **0.46 -> BETWEEN** (PASS >= 0.5).
- **Not pre-registered, information only:** the paired time after the Python build is -0.49 ms (IQR -0.58 / -0.37),
  T ~0.74 on the GPU side. The engine's check is GPU-bound with its CPU hidden (Breakthrough's profile), which is
  where this matters. The verdict stays BETWEEN; the engine A/B decides.
- **Action:** BOF takes an E1 arm Monday night (the slot now runs to 03:00). It is exact, so the arm costs time only.

## D8 AUDIT (Sun 18:1x, share lock, bits): every slice 32-row bit verdict before 574060e, re-run or cleared
- **Re-run on the fixed path, ALL BITWISE.** T1 forward at M 32 / 17 / 24 / 31; T2 commit at keep 1 / 9 / 16 / 17 / 24 /
  31 plus a 4-row probe; all vs <= 16-row checks:
  - C3b (V32_MOE1 = 1, 01:18): 4 layers (c3b_fixed_4L.txt) and 12 layers (c3b_fixed_12L.txt; moe1 fired 87 times);
  - C3-ONE (V32_MOE1 = 1, V32_ONE = 1, 11:07): 4 layers (c3_one_fixed_4L.txt) and 12 layers (c3_one_fixed_12L.txt).
  - Their 01:18 / 11:07 verdicts were right for their inputs (no expert passed 16 of 32 rows in layers 0-3 there), but
    not proofs of the path. Now they are, with moe32_test forcing the case.
- **Unaffected, with the reason:**
  - Breakthrough's pass S (C32sb vs C32, 32 / 32): both arms are 32-row checks through the same expert path, so a drop
    would be identical in both; "the seams don't change bits" stands.
  - ATT32-INV, the ONE32 matmuls / GDN body / HC read / one weight pass: unit tests with no experts.
  - NAX6-ROWS, G32, G32T, C32m: the engine's NAX6 tier kernel, which groups at <= 16 pairs by design (BigMoE).
  - C6, C7, C7b, C7c: timing only (the dropped pairs were computed in full tiles and only not stored).
  - C8 v1: void, superseded by C8 v2 on the fixed path.
- **E2 (Monday, the engine) runs the tier path,** and its own bit check (C32 == 2 x 16 first) stays in the plan.

## BOF2 (Sun 18:20, quiet ticket): the 32-row twin is bitwise but KILLED by the pre-registered line
bof.qmm16bof2 + bof2_bench.py; pre-registered in check55_plan.md 'BOF2' (c317ed4). Data: bof2.txt, bof2_bits.txt.
- **Bits:** bitwise vs qmm16c ks 1 on each half and vs REF2-T's 32-row forms, 5 shapes x 4 configs, at M 32 and 24.
- **Timing at 32 rows** (cold, dependent chains, 7 rounds; per call):

| shape (per check) | REF2-T 32-row | BOF2 best | contract (ks) |
|---|---|---|---|
| GDN qkv \| z (36) | 143.7 us | 181.5 (bs 2) | 140.5 |
| o_proj / out_proj (48) | 113.1 | **73.6** (bs 4) | 54.2 |
| QSA qkv_fused (12) | 135.2 | 144.7 (bs 2) | 124.5 |
| shared down (48) | 10.7 | 10.0 (bs 4) | 10.0 |
| lm_head (1) | 2,079.6 | 2,486.0 (bs 2) | 2,085.8 |

- **Pre-registered verdict:** over the 5 shapes, 14.82 -> 14.77 ms (-0.04) -> **KILL**.
- **Reading:** carrying the second tile's terms through threadgroup memory costs more than the parallel K gains on the
  wide shapes. On the narrow o_proj / out_proj (80 tiles) the fold still wins: -1.9 ms per 32-row check standalone.
- **Not a verdict (picked after the data):** BOF2 on o_proj / out_proj alone would be a new test with its own line.

## C10 BOF2-O (Sun 18:25, quiet ticket): KILL by its whole-wall line; the 2-arm harness has a cold-read confound
c10_bof2o.py (pre-registered e43b757). Data: c10_bof2o.txt (smoke: c10_bof2o_smoke.txt).
- **Bits:** C32ob (BOF2 on o_proj / out_proj) == C32o, 32 / 32 logits rows.
- **Timing, 24 rounds:**
  - C32o 33.59 ms, C32ob 33.71 ms;
  - paired whole wall +0.025 ms (IQR **-8.33 / +8.08**) -> T -0.05 -> **KILL** as registered;
  - after the Python build -0.28 ms (IQR -0.33 / -0.21), information only.
- **The confound (in C9 too):** every arm checks the same new tokens each round, so whichever arm runs first reads those
  tokens' PLE n-gram rows cold from the SSD table. Breakthrough measured that at +8-12 ms per 32-row check
  (CHECK55_PROFILE 'cold PLE'). With 2 alternating arms it flips the sign of every pair, so the whole wall cannot resolve
  a sub-ms effect. C8 v2's 32-row pair stayed tight because its 4 arms put other arms first.
- **C9b / C10b** (pre-registered in check55_plan.md): the same lines, plus an untimed check of the round's tokens on a
  throwaway cache before the timed arms. The verdicts above stand as registered; the b runs are new tests.

## C9b + C10b (Sun 18:29 / 18:32, quiet tickets): with the confound removed, both exact levers PASS
The same harnesses and lines as C9 / C10 (pre-registered 2a7f0a8), plus an untimed check of each round's tokens on a
throwaway cache first. Data: c9b_bof.txt, c10b_bof2o.txt.

| test | bits | paired whole wall (IQR) | after the build | T (whole) | verdict |
|---|---|---|---|---|---|
| C9b: BOF, 16 rows (A16b - A16) | 32 / 32 | **-0.752 ms** (-0.846 / -0.647) | -0.527 | 1.14 | **PASS -> an E1 arm** |
| C10b: BOF2 on o_proj / out_proj, 32 rows (C32ob - C32o) | 32 / 32 | **-0.497 ms** (-0.576 / -0.420) | -0.264 | 1.06 | **PASS -> E2's C32 arm** |

- **The warm-up shrank the whole-wall IQR from +-5-8 ms to ~0.2 ms**, so the cold-read confound was the noise.
- **The whole-wall gains exceed the after-build ones.** The mid-forward submits (SB 4) wait on the GPU inside the build,
  so faster kernels shorten the "build" too. The whole wall is the right metric.
- **Engine-scaled (per-layer parts x4, the head once), standalone-to-slice transfer ~1.1:** BOF roughly -2.5 ms per
  16-row check, BOF2 on o_proj roughly -2 ms per 32-row check. Both are REF2-T's own bits; the engine A/B decides.

## NAX-SEP (Sun 18:33, share lock, bits): the NAX MMA's accumulation is NOT separable, so no exact BOF for S4 / P.V
scripts/week98/nax_sep.py. Data: nax_sep.txt.
- **Question:** does a chain of NAX k-steps, Dt = mma(Dt, A_s, B_s), equal the ordered fp32 fold of each step's product
  computed from zero? If yes, a BOF-style kernel could reproduce REF2-T's S4 (and MLX's P.V) exactly with better
  occupancy.
- **Test:** S4 at PS = K (k-steps chained in one partition) vs PS = 32 (one k-step per partition, then an ordered fp32
  fold), on the fp32 partials. Also REF2-T's 4,096-key partitions vs the folds of their k-steps. K 128 / 512 / 4,096 /
  10,240, 3 weight scales, 2 seeds.
- **Result: 0 / 24 bitwise,** even at K 128 (4 k-steps). The MMA adds its accumulator in with the products under its own
  internal rounding, not as acc + (a rounded term).
- **So:** exact parallel-K speedups stop at qmm16c. BOF works there because each 64-weight group's MMA starts from zero
  and the cross-group sum is an explicit fp32 add. S4's partitions and attention's P.V can only get faster through the
  contract's new reference (track B: S4 at PS 2,048 was -1.5 ms standalone in CB1).

## C7c (Sun 23:37, quiet ticket, power mode 2; pre-registered in c7c_one32.py, Sun 12:5x): step 1 MEETS its target on the slice, R32o 1.345
scripts/week98/c7c_one32.py, run by c7c_night.sh through qscreen.sh. Data: c7c_one32.txt.
- **Order swap (the orchestrator 23:34):** after the 23:30 app quit, NO-SWAP was 34-38 GB, so exact-45's a45 (gate 47 GB)
  could not run. The a45 runner was stopped (it held nothing), C7c ran first, and a45 is skipped tonight.
- **Setup:** the 12-layer resident slice (GGGAGGGAGGGA), SB every 4 layers, contexts 1,024 -> 1,536, 14 rotated rounds,
  own caches, peak 20.2 GB. Per check: snapshot, verify forward (the Python build timed inside), acceptance read, commit.

| arm | ms per check | x A16 | per-round median [IQR] |
|---|---|---|---|
| A16 (16 rows) | 22.95 | 1.000 | |
| C32s (v32's split: today's engine path) | 37.46 | 1.632 | 1.600 [1.537-1.726] |
| C32o1 (C7's forms, seams <= 16) | 33.08 | 1.441 | 1.420 [1.359-1.516] |
| C32b (per-shape best forms, seams <= 16) | 33.09 | 1.441 | 1.402 [1.352-1.480] |
| C32o (step 1 complete: + seams to 32) | 30.93 | 1.348 | **1.345 [1.286-1.364]** |

- **Attribution (paired, / A16):**

  | part | measured [IQR] | predicted |
  |---|---|---|
  | the best forms | 0.013 [0.003-0.019] | 0.03-0.05 |
  | the seams to 32 | 0.047 [0.037-0.063] | ~0.05 |
  | all of step 1 | 0.239 [0.187-0.281] | |
- **Verdict (pre-registered): R32o 1.345 <= 1.35: step 1 MEETS its target on the slice.** Monday's E2 runs as booked
  (E14's C32one arm).
  - The predictions held: R32o in 1.33-1.40, and C32o1 1.441 in 1.43-1.45.
  - The split path matches the engine: C32s 1.632 on the slice vs G32T's 1.625.
- **Caveats:**
  - **Cold reads.** The lockstep harness predates the cold-read lesson (C9 / C10): the first arm of a round reads that
    round's PLE n-gram rows cold. The arms rotate, so each is first in ~3 of 14 rounds, and the verdict is a median of
    per-round ratios. An untimed warm check per round (C9b) would remove the effect; E14 has a warm-up.
  - **Python build.** The build is inside each check: 0.52-0.59 of it on the slice, and the engine's share differs. E2
    gives the engine ratio.
  - **D8 fix.** The 32-row experts run through the D8-fixed path (moe32's capped segments, 574060e). Before the fix, the
    full tiles were computed too (only some stores were dropped), so the timing is comparable.
  - **One shape still splits.** A 4-bit 512 x 2,560 shape still runs as two 16-row calls (168 calls = 14 rounds x 12 layers).
- **ENGINE (Mon 00:42, Breakthrough's CHECK55 engine profile, quiet, 48 pairs, a082f8c): C32one / A16 = 90.46 / 61.11
  = 1.480, bitwise; C32 1.671.** The slice's 1.345 does not transfer. The slice compresses ratios (its check is ~half
  Python build; its experts are a smaller share). Quote R(K) from engine runs only (data/week98/treek_price.md
  'CORRECTION').
