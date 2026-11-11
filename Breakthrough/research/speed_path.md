# Speed path: from 35-40 to 100-150 tok/s at 6-12 GB, and 4 GB (Breakthrough, 2026-10-20 night)

Inputs: SCOREBOARD section 2 (bytes), tonight's measurements (data/expS, data/tfvar, data/gmass), M5 Pro RAM
~300 GB/s, SSD 12-14 GB/s (Kamen; BigMoE's loader achieved ~4.7 GB/s). Expert 4.9152M weights: 2.765 MB at 4 bits,
2.15 MB as MSB-3, 1.54 MB as MSB-2. One verify pass = 4 rows (MTP 3:3), ~3.2 tokens.

## 1. Where frugal + MTP stands (measured 18:15, 10 % residency)
80 ms per pass, 3.2 tok/pass -> 39.8 tok/s at 14.3 GB (code); 5 %: 87 ms, 2.98 tok/pass -> 34.4 tok/s at 10.9 GB.
Measured RAM floor (no experts): ~7.5 GB (dense 3.4 + MTP head/stream, KV, n-gram hot cache 1 GB, scratch).
Quality: GSM8K 146/150 at 10 %, but teacher-forced KLD 0.057 (6x ar4); on ifeval+multi 0.107 vs exact 0.0043.

## 2. RAM roofline per pass (everything the pass touches, read once)
| part | GB per 4-row pass | lever |
|---|---|---|
| dense trunk (GDN, QSA, HC mixers bf16 1.27, shared, lm_head) | 3.4 | HC mixers 4-bit: -0.95 |
| routed experts used (~26 unique per layer x 48 x 2.765 MB) | 3.45 | 3-bit tier: -0.86; frugal computes ~70 % of slots |
| MTP drafts (3 x shared lm_head 0.36 + head) | 1.46 | FR-Spec subset lm_head (64K rows): ~-0.8 |
| **total** | **8.3 GB = 28 ms at 300 GB/s -> 115 tok/s** | with all three levers ~5.7 GB = 19 ms -> ~170 tok/s |
Today's 80 ms per pass is ~3x the roofline: per-layer kernels run at ~45 % of the read ceiling (MoonShots r26b) plus
48 per-layer gate waits and Python graph build. **Engine efficiency is the biggest single speed lever**, and it is
independent of residency.

## 3. SSD ceiling under the quality bar (the new constraint from tonight)
Quality needs ~85-95 % of the top-10 gate mass computed (KLD 0.0095 = ar4 at ~5 % missed, wait-by-mass 0.85,
with renormalisation). The routing is flat (rank 0 = 21 % of the mass), the missed experts are cold (a Belady
oracle cache recovers only 0.295 -> 0.266), and **MTP's 4-row passes do not share cold misses** (106 loads/token at
K=4 vs 111 at K=1, `pass_mass_sim.py`). Loads at the bar, 4-bit reads:
| residency | loads/token | MB/token | SSD-bound tok/s at 12 GB/s |
|---|---|---|---|
| 10 % (C=51, 14 GB total) | ~106 | ~295 | ~41 |
| 5 % (C=26, 11 GB total) | ~145 | ~400 | ~30 |
So exact-like quality from frugal at 10 % caps near 40 tok/s on this SSD before any engine cost: today's 39.8 tok/s is
bought with KLD 0.057. **The 100-150 tok/s goal at <= 12 GB needs ~3-4x fewer SSD bytes at equal coverage.**

## 4. What can cut SSD bytes at equal quality (ranked by expected yield per effort)
1. **Better estimate of the missing outputs (zero bytes).** Renormalising is harmful (drop 0.104 vs drop+renorm 0.141);
   SERE-style similar-resident substitution (2602.07616: Qwen3-30B top-2 14.3 -> 80.4 of 82.2) queued tonight; partial
   renorm beta in (0, 1) queued. If these cover half the missing error, the bar moves from 85 % to ~70 % coverage:
   ~72 loads/token (200 MB) at 10 %.
2. **Cheaper reads for waited tail ranks.** MSB-3 reads for ranks >= 2 cost 78 % of the bytes and ~nothing in KLD
   (plan4433333333 full-suite KLD 0.0064 < ar4 0.0096); MSB-2 = 55 % (quality unknown: 4422222222 never ran).
3. **More residency per GB.** A 3-bit cache holds 1.29x more experts; planes (MSB-2 tier) ~1.8x but only for 2-bit use.
   The miss rate falls slowly with residency (long tail): 10 % -> 20 % LRU missed mass 0.29 -> 0.23 at wait-top-2.
4. **Smaller floor -> more room for experts at 12 GB:** HC mixers 8-bit (-0.63 GB), MTP experts streamed (done), n-gram
   hot cache 1 GB -> 0.25 GB, prefill scratch released after prefill.
5. Not useful: smarter eviction (< 10 % gain even for an oracle), threshold-by-gate rules (cliff), entropy coding of
   4-bit codes (~6 %), longer passes for the cold tail (no sharing).
Combination estimate at 10 % (14 GB): 1 + 2 -> ~200 MB x 0.8 = ~160 MB/token -> ~75 tok/s SSD-bound; reaching 100-150
needs ~80-120 MB/token, i.e. the estimator (1) must cover ~2/3 of the missing error, or residency must double.

### 4b. Update 00:45 Oct 21: substitutes are dead, read the tail at low bits (rate-distortion)
The 125B's routed experts are mutually orthogonal (sim_table.py: best |cos| to any other expert median 0.024), so no
resident expert, shared expert or mean expert approximates a missing one (item 1 above is void). What remains is
coverage per byte. With orthogonal errors a missing expert costs w^2 ||y||^2 and an expert read at b bits costs NMSE_b
times that; KLD rises ~ (missed energy)^1.5 (energy_sim.py fit on the probes). Energy recovered per MB:
| tail read | MB | NMSE vs 4-bit | recovered / MB |
|---|---|---|---|
| 4-bit (today) | 2.77 | 0 | 0.36 |
| MSB-3 (records3 prefix) | 2.15 | ~0.02 | 0.46 |
| MSB-2 truncation (2+1+1 layout) | 1.54 | ~0.10 | 0.58 |
| MSB-1 truncation | 0.92 | ~0.44 | 0.61 |
| trellis 2-bit copy (D2) | ~1.3 | 0.071 | ~0.72 |
| trellis 1-bit copy (extrapolated) | ~0.6 | ~0.27 | ~1.2 |
So under an SSD budget the optimum reads MORE tail experts at FEWER bits (a trellis-coded 1-2-bit tail copy on disk,
~19-37 GB extra) instead of fewer at 4 bits. Example at 10 %: wait-to-85 % with 1-bit trellis tails ~111 MB/token,
predicted missed energy ~0.0088 (vs 0.0144 for wait-to-50 % at 4 bits for the same bytes). Night probes (chain 2 and 5)
test MSB-3 / MSB-2 / MSB-1 tails directly; a trellis tail copy needs ~1 h of all-core encoding (daytime, share lock).

### 4c. The base Mac mini (M6, 16 GB, SSD ~3.2 GB/s): estimate, nothing measured
~12 GB usable after macOS, measured floor ~7.5 GB -> ~4.5 GB of 4-bit experts (C ~ 34, 6.6 %).
| policy | loads/token | MB/token | SSD-bound tok/s at 3.2 GB/s (perfect overlap) | quality |
|---|---|---|---|---|
| frugal wait-top-2 | ~25 | ~70 | ~46 (realistic ~20-25 with today's unoverlapped waits) | frugal KLD (~0.1 on ifeval/multi) |
| wait-to-85 %, 4-bit reads | ~130 | ~360 | ~9 | ~ar4 |
| wait-to-85 %, 1-bit trellis tails | ~130 | ~120 | ~26 | to be measured (chain 5 tests MSB-1) |
The 50-75 tok/s Mac-mini goal at Q4-like quality is out of reach for inference-only policies on this model; it needs
fewer active experts per token (research/training_route.md) or a smaller model.

### 4d. If frugal is accepted (preliminary decode verdict 10-21): the path to ~100 tok/s at ~14 GB is engine work
Today frugal + MTP 3:3:0 at 10 %: ~80 ms per 4-row pass, 3.2 tok/pass -> 39.8 tok/s; SSD only ~56-82 MB/token (~200 tok/s
of SSD headroom at 14.6 GB/s). Estimated pass budget: dense trunk 3.4 GB + resident experts ~3.45 GB read at ~45 % of the
280 GB/s ceiling ~ 54 ms, waits for missing ranks 0-1 ~20 ms, drafts and host overhead ~6 ms.
| lever | owner | pass after | tok/s at 3.2 tok/pass |
|---|---|---|---|
| look-ahead prefetch of predicted ranks 0-1 into spare slots (waits 20 -> ~5 ms) | Breakthrough (BT_PF, testing 10-21) | ~65 ms | ~49 |
| 3-bit tier experts + MoonShots' 3c GLU (+21-23 % per BigMoE B5) | BigMoE / MoonShots (built) | ~53 ms | ~60 |
| HC mixers 4-bit, kernels toward ~80 % of the read ceiling (MoonShots' 3c load tuning ~230 -> ~310 GB/s) | MoonShots | ~30-35 ms | ~90-105 |
| load/compute overlap, out-of-order experts (MoEPrefetch) | MoEPrefetch | overlaps the rest | |
| more tokens per pass (draft depth / lookup on repetitive text) | SpecDecode | | x tok/pass |
So ~100 tok/s at ~14 GB is plausible with frugal once the main-track engine work lands and frugal is ported onto it
(IDEAS D1e); 150 needs ~4.5-5 tokens per pass or a further ~1.5x in kernels.

## 5. The 4 GB version
The measured floor without experts is ~7.5 GB (dense 3.4 GB at 4 bits is read every pass and cannot stream at 12 GB/s:
3.4 GB / 12 GB/s = 0.28 s per pass). A 4 GB build needs the dense trunk at ~2-2.5 bits (~2 GB; GDN is the most
quantization-sensitive class: noise35 Q4 KLD 0.057 alone), no MTP head, KV/state ~0.3 GB, and ~1 GB of experts
(C ~ 7 of 512, 1.4 %): frugal there misses ~45 % of the mass. Not reachable at Q4 quality with this 125B by inference
tricks alone; it needs distillation into fewer/denser experts or a smaller model. Recommend parking 4 GB until the
12 GB line works.

## 6. Plan (GPU turns are correctness-only tonight; timings from 07:30)
- Tonight: similarity tables (68 GB read once) -> SERE variants; beta / no-renorm mass variants; SERE + wait-by-mass.
- Then: fit KLD vs (missed mass, estimator) -> pick the cheapest policy at KLD <= 0.01; price it with pass_mass_sim +
  MSB-3 reads; build it into the fast path (GPU table lookup for SERE, mass-rule wait without CPU sync: the per-row
  prefix length can be computed on the GPU and sent in the routing row).
- Engine efficiency (the 3x roofline gap) is BigMoE/MoonShots/PrefillKernels' lane; flag it to the orchestrator.
