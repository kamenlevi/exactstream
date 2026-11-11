# Roofline engine for the non-exact line — PLAN (MoonShots, started 2026-10-21 16:40; BIG BUILD approved by NewIdeas)

**STOPPED 2026-10-22 ~04:50 (NewIdeas, option a) on the true all-hits measurement: BigMoE 38.2 ms per 4-row verify pass =
engine 38.5 = gate-free 39.1 ms; the gap to today's ~100 ms streamed passes is cold-read stalls. The engine code stays as a
tested kernel library (fused GDN/HC kernels, M1a correctness at the numerics floor). M3 wide passes, frugal multi-row phase B
and in-RAM mode are PARKED. Next job: the stall anatomy (~/MoonShots/stall/).**

Contract: NOT bit-exact (fp32 accumulation, any reduction order), deterministic run to run; KLD vs BigMoE's exact engine at
the numerics floor (~0.004 per corpus part on the Eval suite's tf sets). Owners: MoonShots (engine + kernels), Breakthrough
(frugal / dh_w4 policies and their measurements), BigMoE keeps the exact engine. Design + decision: ~/NewIdeas/ENGINE_DESIGN.md.

## What is settled by measurement (patches/sol_layer/README.md, IDEAS 33a)
- Per GDN layer, fused bandwidth-shaped kernels: M=1 0.410 ms (224 GB/s), M=4 0.720 ms at 38 distinct experts (~0.63 at real
  routing overlap) vs 0.89-0.93 ms in today's engine; floor 0.44-0.52 ms.
- 4-bit GEMVs: MLX qmv-style row mapping (each row of x its own threadgroup row, the cache absorbs weight reuse across M):
  254 GB/s at M=4. My register-blocked fused RMSNorm+GEMV was ALU-bound (2.6x slower at M=4) -> norms are folded AFTER the
  GEMV into small epilogues (W (inv x) = inv (W x)).
- HC read: v2 kernels (group-split down with one-pass RMS; 4-lanes-per-row up), ~40 us per kernel at M=4.
- No megakernel / grid barriers: one persistent dispatch with a device-atomic barrier is 10-12 us SLOWER per merge than two
  dispatches on M5 (and the literature agrees: barrier 2.0-2.5 us vs dispatch boundary 1.3-1.6 us).
- CPU: Python graph build ~0.03 ms per layer, hidden by mx.compile (wall = GPU time within 2-3 %).
- Routed experts: tierglu 3c (fused gate-weighted combine; records3 tiers 4/3/2) and tierglu4 (records4 planes), bitwise to
  MLX per tier; out-of-order phase forms exist for both.

## Command buffers and the loader ring (the GPU-side gating)
Metal makes a GPU write visible to the CPU only when its command buffer completes (data/publish.txt), so:
1. Each layer's routing closes a command buffer (CB_a(L): HC read + attention + out_proj/HC write + MLP HC read + router +
   top-k + residency lookup -> per-(token, rank) tickets: resident slot or "cold").
2. CB_b(L) is encoded and committed right behind it, before the CPU knows the routing: resident experts + shared expert first,
   then kernels that wait on per-ticket ready flags (spin on a shared buffer; CPU -> GPU visibility mid-kernel works, as in
   BigMoE's spin gate) for cold experts the policy needs on time; dh_w4's late ranks go to CB_b(L+0.5) (before the next
   layer's MLP HC read).
3. The CPU learns the cold tickets in CB_a(L)'s completion handler (or by waiting on its output array), assigns pool slots,
   issues reads at QD >= 6-16 on a loader thread pool, and raises the flags as reads land. The GPU queue never runs dry while
   the CPU reacts; the CPU never blocks the GPU.
Measured costs (engine/bench/latency.swift, quiet lock): handler latency, flag latency, cold reads at QD 1/4/6/10, end to end
routing-end -> reads -> GPU resumes, measured 17:16 (data/day15/latency.txt, quiet lock):
- GPU end of a command buffer -> CPU completion handler: **50 us** median (p10 43, p90 57), same with the GPU busy.
- Cold reads of whole MSB-3 expert parts (2.16 MB, pread F_NOCACHE, concurrent): QD1 **0.30 ms**, QD4 0.80, QD6 **1.13**,
  QD10 1.81 ms (~12 GB/s; ~0.19 ms per extra read).
- CPU flag -> spinning kernel: my test re-polled one cache line and saw the store only after 7 ms-30 s (GPU caches don't
  see CPU writes to a line already cached); BigMoE's protocol polls a FRESH line each iteration (bigmoe_spin; their spin
  costs ~69 us per gate). Use BigMoE's region-line protocol; do not spin on one line.
- **dh_w4 window:** routing of layer L -> its late ranks must land before layer L+1's MLP HC read, i.e. one attention half
  (~0.3 ms at M=4, ~0.2 ms at M=1 in this engine). Handler 0.05 + one cold read 0.30 ms already exceeds it; 6 cold ranks at
  QD6 need ~1.2 ms (~2 layers). So half-layer deferral hides at most ~25 % of a 6-read batch (~70 % of a single read);
  hiding the rest needs reads issued earlier (look-ahead prefetch) or a longer deferral (d = 1-2 layers).

## Results so far
- M1a attention half (22:01, data/day15/m1a_attn.txt): the engine's GDN attention half (HC v2 read, MLX in_proj, conv/l2norm
  epilogue, recurrence + sigmoid-gated norm, out_proj, HC write + MLP read) vs MTPLX's modules on real layer-0 weights, M=4
  then M=1 on one cache: relative errors H1 0.9-1.5e-2, MLP mixed input 1.2-1.8e-2, inject 0.4-0.6e-2, conv state 2-3e-3,
  SSM state 6-7e-3: PASS at bf16 level (MTPLX runs bf16 intermediates; the engine accumulates in fp32).
- M1a model test: first try failed on setup (a 4-row chunk needs up to 40 experts resident; C=16): rerun with C=48.
- **M1a correctness: AT THE NUMERICS FLOOR (10-22 03:28, data/day15/m1a_floor.txt).** 64 real tokens (FINDINGS.md text):
  MTPLX M=1 vs MTPLX M=4 (its own decode vs verify kernels): KL mean 4.18e-2, median 1.9e-2, top-1 87.5 %; engine M=4 vs
  MTPLX M=4: KL mean 3.82e-2, median 1.2e-2, top-1 96.9 %; engine vs MTPLX M=1: 4.01e-2 / 2.5e-2 / 87.5 %. The engine is as
  close to MTPLX as MTPLX's own paths are to each other (EvalSuite's "~0.004" floor is on sharper tf text).
  Error budget on identical inputs (m1a_parts.txt): HC read 2.8-3.5e-3, GDN core 5-8e-3, HC write exact, MoE identical, but
  the MoE turns bf16-level (2^-9) input noise into 0.5-1.2 % output change without any routing swap: any non-bitwise engine
  pays this. bf16-faithful rounding (RLE_FAITH=1, default) took the first chunk's KL 8.7e-3 -> 4.0e-3.

- **M1b speed (10-22 03:50-04:03, data/day15/m1b_*.txt; 24 GB tier env, 4-row verify passes, fresh process each):**
  streamed (cold-ish): BigMoE 101.1 ms vs engine 104.6 ms per pass (same misses/stalls: stall-bound). All hits (the same chunk
  repeated): BigMoE 60.7 ms (graph build until the last eval 57.1 ms), engine + BigMoE's MoE block 59.5 ms (build 55.8 ms),
  **engine with the shared expert only (no routed experts, no gate) 22.8 ms (build 2.5 ms)**. => With BigMoE's MoE block the
  pass is CPU-BOUND: ~1.1 ms of Python/gate work per layer (post_ids + spin + tier GLU wrapper + look-ahead routers +
  per-layer async_eval). The engine's dense path is GPU-bound at 22.8 ms; + routed experts (~0.3 ms/layer at M=4) -> a
  gate-free target of ~37 ms per all-hits pass (~1.6x today's 60). NEXT: profile the MoE path's CPU cost, then a gate-free
  hit path (GPU-side residency lookup; the CPU only on misses; frugal waits only for ranks 0-1).
- **TRUE ALL HITS (10-22 04:34, fixed-state repeat: the same 4 positions re-run from a restored state, identical routing,
  0 misses; data/day15/m1b_fixed_*.txt): BigMoE's model 38.2 ms per 4-row verify pass (p10 38.0, p90 38.4), engine + BigMoE's
  gated MoE 38.5 ms, engine gate-free prototype 39.1 ms (0 of 36,864 lookups missed).** => (1) the gate costs ~nothing on hit
  passes: the gate-free hit path has NO prize (my 04:1x proposal is withdrawn); (2) BigMoE's engine under the reference flags
  is already near the speed-of-light layer at M=4 (~0.80 ms/layer incl. MoE vs SoL 0.65-0.72); the "0.89-0.93 ms today"
  baseline included stall effects; (3) the gap from ~100 ms streamed passes to ~38 ms is entirely cold-read stalls. The earlier
  "CPU-bound ~1.1 ms/layer" reading was wrong (CPU blocked inside per-layer async_eval; with it off, build is 5-6 ms).
- NewIdeas 04:1x (approved order: CPU profile -> gate-free hit path -> 10 % wait profile), two asks:
  (1) design the GPU-side residency lookup so BigMoE's EXACT engine can adopt it unchanged: on a hit layer the same
      residency/slot decisions, made on the GPU; the CPU wakes only for misses (24 GB and 0.50, where hit layers dominate).
      Bitwise there = the biggest exact lever (~1.6x on all-hit passes) -> main-track item for Kamen.
  (2) report the profile's CPU split per component (post_ids/spin, tier-GLU wrapper, look-ahead routers, async_eval) so
      BigMoE can cut cheap parts today (e.g. look-ahead routers on the GPU in the same dispatch).

## Milestones
- M0 (Phase 0, today): code study of BigMoE's engine + ext, Breakthrough's frugal / dh_w4, MTPLX's 125B blocks (subagents);
  latency microbench; this plan.
- M1a: dense path at roofline for the real 125B weights (resident dense 3.4 GB, experts from a resident pool): HC v2, GDN
  (MLX GEMVs + conv/l2norm epilogue + recurrence + gated norm), QSA (reuse MTPLX's sparse attention first), MoE (router
  epilogue + tierglu + shared + combine), LM head; compiled per layer; KLD vs the exact engine with ALL experts resident.
- M1b: SSD-streamed experts at 10 % residency: residency table on the GPU, tickets, loader ring, flags; frugal and dh_w4
  policies (Breakthrough's semantics); stalls/reads per token measured.
- M1c: 4-row MTP verify + drafts on the same blocks; KV / GDN-state rollback; Eval-suite tf KLD per part; tok/s.
- M2: in-RAM mode (no gates), near-tie re-route, 3-bit cache tiers, records4 tiers.

## Validation
- Speed vs BigMoE's spec stack on identical prompts: ~/SpecDecode/scripts/test_spec125.py FRAC N CONFIGS (configs
  k_mtp:max_draft:lookup[:p_min], PROMPT=code|edit|prose|long|long32|loop; runs ar2 first; tok/s and tokens/pass; its
  token-sha check does not apply to a non-exact engine). All b*/day runners use it.
- Per block: allclose vs MTPLX/BigMoE reference outputs on real weights (rel. err ~1e-2, bf16 level).
- Model: KLD vs the exact engine on the Eval suite's tf sets per part (code/gsm/ifeval/multi) at the numerics floor.
- Speed: per layer vs the byte roofline (target >= 80 %), fixed overhead per pass (target <= 1.5 ms).

## Phase 0 facts: policies and the eval interface (Breakthrough / EvalSuite / NewIdeas code, read 16:2x)
- All policies today are monkeypatches of `mtplx.models.qwen4_exp.SparseMoeBlock.__call__` (deferral also `DecoderLayer`);
  the expert GLU runs MTPLX's SwitchGLU on pool slots; the gate is BigMoE's ext (`Gate.route_and_gate`: `ext.post_ids`
  with nw | nk<<8, `ext.spin_wait`; flag mode nw=255 -> `ensure_wait_list`; LRU per layer, `ensure`/`make_room`).
- frugal w2f_r10: C = max(16, round(0.10 x 512)) = 51 slots/layer, no prefetch; router top-20; wait for ranks 0-1, LRU-touch
  resident ranks 2-9, keep the first 10 RESIDENT experts in rank order (ranks 10-19 stand in), renormalize; + shared.
  4-row MTP form: bt_frugal_spec.py (rank-major routing row, nw=2S, nk=10S). Latest: 39.2 tok/s code (3.20 tok/pass),
  GSM8K-150 146/150; official KLD 0.057 (5.97x ar4) FAILS.
- m85 (wait-by-mass, theta 0.85): only policy passing every part on the every-8 subsets (code 0.0008, gsm 0.0026, ifeval
  0.0110, multi 0.0074), 112 reads/token, 14.2 tok/s eager.
- dh_w4 (tf_defer125.py L55-125; bt_defer.py speed knobs BT_K=4, BT_M=6): ranks 0-3 waited; resident ranks 4-9 combined now;
  non-resident ranks 4-9 are read anyway, computed on layer L's MLP input, weighted p / sum(top-10), and added to the HC
  streams with layer L's inject weights AFTER layer L+1's attention and BEFORE its MLP HC read (last layer: same layer).
  No 125B speed numbers yet (prototype unbuilt); predicted KLD gsm ~0.0025 / ifeval ~0.014 (pass) at 41-50 waited reads/token.
- EvalSuite tf scoring: items = SG.tf_items() (tf_gsm 60, tf_ifeval 30, tf_code 24, tf_multi 24, tf_doc 4 x 64K); reference
  = cache/ref/tf_ref.npz (top-256 log-probs of the BigMoE bare 4-bit "ref"); head = EN.var_head(SG.load_ref_tf(items),
  SG._targets(items)); head(i, positions, logits [1, 248320]) -> {kld, agree, ...}. The engine only needs full-vocab logits
  per teacher-forced position; a 4-rows-per-call driver is ours to write. Official path: an MTPLX-style runtime object
  (restore_or_prefill_prompt_state, forward_ar, gate.check) registered in EvalSuite's model.py. evalsuite/env.py exits if
  BigMoE speed flags are set and puts ~/BigMoE first on sys.path (import the engine first).
- Numerics floor: exact probe 0.0041 (ifeval+multi); tfstep step-path noise 0.0018. Per-set screen targets (1.10 x ar4):
  code 0.0025, gsm 0.0045, ifeval 0.0204, multi 0.0206 (gate_table.py; the official verdict pools all own outputs).

## Phase 0 facts: BigMoE's gating already IS the loader ring -> reuse it (decision 16:4x)
- ~/BigMoE/ext/bigmoe_ext.cpp (nanobind, linked to MLX 0.32.2 internals; build in ext/build): `post_ids` kernel writes each
  layer's routing row [seq, n, ids, preds] (seq last per 128-B line) and a 64 MB dummy input forces MLX to commit the command
  buffer there; ONE loader thread (user-interactive QoS) busy-polls routing rows, runs `ensure` (per-layer LRU, make_room),
  reads with dispatch_apply + pread(F_NOCACHE) (records3: 16 KB-aligned, CPU rebuild in exact mode), then releases waiting
  kernels by writing slot ids + release fence + seq tags into per-(pass, layer) region lines; `spin_wait` kernel polls them
  (x1.1 back-off, timeout -> status). Dep fast path: residency table published a pass ahead -> all-hit layers don't wait.
  JIT prefetch reserves slots for L+1..L+3 (PF_QD caps speculative reads). Same design as my completion-handler plan
  (polling instead of a callback); no completion handlers in the default path.
- Breakthrough's forks extend it: fork125bf (flag mode nw=255, ensure_wait_list, protect-what-fits), fork125bd (deferral
  phase A/B for dh_w4: not built/tested yet).
- Measured gate costs (BigMoE FINDINGS L84-88): mx.eval ~230 us, shared event ~170 us, spin ~69 us per layer; residency
  published a pass ahead brought hit overhead to ~5 %. T1 (S=4, streaming removed): 48.7 ms per pass = experts+gate 20.4,
  GDN 11.7, QSA 5.3, router+shared 4.9, host graph build 4.7, spin on hits 1.9 (vs the SoL stack ~30-35 ms).
- The deployed MTP loop is ~/SpecDecode/spec125.py (k_mtp=1, max_draft=7, lookup, AdaptiveWidth), with
  commit_verified_window (QSA trim, GDN replay, PLE replay). Trunk runs eagerly (MTPLX_QWEN4EXP_COMPILE=0).
- DECISION: the engine imports BigMoE's ext (read-only use; never edit ~/BigMoE) for residency, loading, post_ids and
  spin_wait, and owns the compute: fused SoL kernels, compiled per-layer graphs, policy logic (frugal / m85 / dh_w4 via a
  phase-A/B release like fork125bd), MTP drafts on the same blocks. The latency microbench still runs to give the Metal-level
  floor under BigMoE's ~170-230 us per gate.

## M1b policy spec: frugal + late rank 1 (phase B), from Breakthrough 17:5x (fork125bd ext "Breakthrough bd"; bt_defer.py)
Measured on today's engine at 10 % eager: code 23.8 -> 25.4 tok/s, prose 22.6 -> 24.6, waited misses 27 -> 11 per token
(-60 %), ~45-50 % of each moved read hidden. Quality: 35B +3 % KLD; 125B screen base_l1t2 running.
1. Routing-row id flags (post kernel masks id & 0xFFFFF): bit 20 = wait (load before phase A); bit 21 = late-eligible (if
   cold: read after phase A, add half a layer late); unflagged = use only if resident (frugal substitutes from ranks 10-19),
   never read. Frugal + late rank 1: rank 0 bit 20, rank 1 bit 21, ranks 2-19 unflagged; the last layer waits for all.
2. Loader per row: read waited misses; protect EVERY posted id of the row; release phase A (slots for waited + resident,
   -1 for late); read late ids; publish phase B as (position, slot) pairs padded with (n, 0).
3. GPU layer L: experts = phase-A slots with frugal selection: rank 0 + rank 1 (counted even when late) + first residents of
   ranks 2-19 up to 10; weights renormalized over those 10; stash x_L (MoE input), weights, phase-B dep.
4. GPU layer L+1, after its attention and BEFORE mlp_hyper_connection: spin on L's phase B (the spin depends on L+1's
   attention output, so it is ordered after it); y_late = sum over pairs w[pos] * expert_slot(x_L) (padded rows weight 0);
   hyper = hyper_residual_write(hyper, y_late, inject_L) (layer L's MLP inject) - as NewIdeas' tf_defer125 "dh".
MTP verify needs a multi-row phase B (pairs index rank-major rows): not built in fork125bd -> ours to build in the engine.

## M3 (IN SCOPE, NewIdeas via Chat1000 10-22 00:0x): wide passes (16-256 rows), from Chat1000 18:0x (~/Chat1000/data/widepass_ops.txt, widepass.txt)
Stock MLX, 125B shapes, 48 distinct weight copies: whole stack 128 rows 237 ms (539 rows/s, 97 GB/s; roofline ~110 ms);
64 rows 242; 16 rows 86; 1 row 18.6 ms. Per 48 calls: dense 4-bit qmm (gdn_in) is compute-efficient from 64 rows
(K=128 21.7 ms); HC down bf16 ~5.4-6 ms at 16-128 rows (53-59 GB/s; 4 HC matmuls ~24 ms/pass); routed experts (3-bit,
sorted gather_qmm) K=128 / 193 distinct experts 184 ms (108 GB/s) vs a 66 ms byte floor - experts are 78 % of the pass
(~5-10 rows per expert, mostly empty tiles). Needed: (1) a grouped expert-major small-M kernel (each expert read once, its
5-10 rows computed; tierglu 3c lineage generalized from M <= 8 rows per pass to ~5-10 rows per expert) at >= 250 GB/s:
184 -> ~70-80 ms; (2) the v2 HC kernels tiled over K rows: 24 -> ~8 ms; (3) MLX qmm for dense. -> 128-row pass ~110 ms
(~1,200 rows/s, 2.1x); 2-3x at 16-64 rows (Route S configurations, prefill). Target (GAPS.md T_pass row of the
~1,000 tok/s table): <= ~0.9 ms per row at 128 rows (1.85 today). 64 rows is MLX's worst point. Also look at W4A8 for the
dense GEMMs (compute-bound from 64 rows). Order: after M1 (the 4-row verify engine) unless NewIdeas reprioritizes.

## M2 loader-ring design note (NewIdeas 10-22 ~01:3x; ~/NewIdeas/IDEAS.md I8, scripts/stage_time.py)
**CORRECTION (NewIdeas ~04:0x): FALSIFIED on hardware.** Breakthrough's A/B on the eager B12 engine: the corrected d=1 prefetch
issued after demand reads, 125 prefetches used per pass, 99 % on time -> per-pass time flat on code (83.9 -> 84.2 ms), -5 % on
prose. The model's assumption (a waited read = pure SSD time on the critical path; issue, bookkeeping, promotion free) was
wrong for that engine. RULE for M2: first PROFILE per-layer GPU wait vs compute in the new engine at 10 % residency (ms per
pass the GPU sits on cold reads); build prefetch / staging only if that number is large. Original note kept below:
Model on 125B traces, fast engine (0.63 ms/layer at M=4 + 3 ms/pass, 3 tokens/pass, frugal C=51): waits ~30 % of an MTP
pass (61 waited reads/pass); oracle forecast 57.8 -> 81.7 tok/s. SSD model from my QD latencies: 0.191 ms floor + 2.77 MB at
13.1 GB/s shared by the reads in flight -> prefetches issued WITH demand reads lose (-6 %, as Breakthrough measured; NVMe has
no priority). What wins (+10 % plain d=1 tap, +17 % at corrected-forecast recall ~0.69):
- DEMAND FIRST: issue layer l+1's prefetch hints (d=1 router on layer l's MoE input + MoEPrefetch's ridge correction
  BIGMOE_PF_CORR, 35 MiB, k=2 per row) only after layer l's demand reads landed; hints computed in the same dispatch as layer
  l's routing (no extra command-buffer boundary).
- Reads go to a small GLOBAL staging ring (8-16 slots), not the per-layer pools (no slots out of C). A staged expert routed at
  ranks 0-1 is promoted by a copy into the victim slot; at ranks 2-9 it is used free if landed; unused staged experts dropped.
