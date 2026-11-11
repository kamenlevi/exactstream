# Stall anatomy — first cut (MoonShots, 2026-10-22 05:45; NewIdeas' assignment)
Real streamed MTP decode (SpecDecode spec125, BigMoE b13 env + BIGMOE_GDN_EXACT=1; 160 tokens; per-pass ext counters via
stats["pass_hook"]), against true all-hits compute baselines per verify width S (fixed-state repeat, 0 misses).
Data: data/stall/ (anatomy_*.json, base_*_S*.txt, summary2.txt); code: stall/anatomy.py, anatomy_b12.py, report.py.
**Correction (10-22 11:45, after defer/RESULTS.md):** the 'demand I/O', 'strictly serial' and SSD-busy / idle-window
figures here count DEMAND reads only. v3's JIT prefetch (BIGMOE_PF_JIT, ~180-200 reads issued per code pass, ~90 used)
runs inside the 'idle' inter-layer windows, so they were partly prefetch time. Half-layer deferral bought nothing on top
of JIT (+23 % only with JIT off). Count prefetch traffic as busy before sizing any overlap lever.

All-hits compute per pass (ms) by S:  exact 29.0 / 40.0 / 56.0 / 66.5 (S 2/4/6/8); v3 tier 28.1 / 37.0 / 52.4 / 61.8;
tier at 10 % 28.1 / 32.8 / 37.1 (S 2/3/4, B12 proxy).

| mode | prompt | tok/s | tok/pass | S | pass ms | compute | exposed stall | demand I/O | MB/pass | GB/s | floor | perfect overlap | bytes x0.55, no overlap | bytes x0.55 + overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | code | 25.3 | 2.86 | 3.1 | 107.6 | 36.3 | 71.3 | 106.8* | 1447 | 13.6 | 106.7 | 110.0 | 75.5 | 66.4 |
| exact | prose | 22.7 | 1.86 | 2.5 | 75.8 | 32.1 | 43.7 | 43.4 | 598 | 13.8 | 49.0 | 50.0 | 56.0 | 38.4 |
| exact | edit | 30.4 | 6.96 | 7.1 | 224.5 | 60.7 | 163.8 | 274.6* | 3752 | 13.7 | 268.0 | 274.6 | 150.8 | 148.3 |
| **v3** | code | 34.5 | 3.14 | 3.4 | 85.1 | 35.4 | 49.6 | 45.0 | 637 | 14.1 | 51.4 | 50.2 | 60.2 | 37.8 |
| **v3** | prose | 29.7 | 1.72 | 2.1 | 54.0 | 28.6 | 25.4 | 23.3 | 280 | 12.0 | 30.9 | 31.6 | 41.4 | 28.7 |
| **v3** | edit | 42.2 | 6.96 | 7.1 | 160.7 | 56.5 | 104.2 | 102.7 | 1534 | 14.9 | 113.6 | 106.6 | 113.0 | 65.4 |
| B12 10 % | code | 31.1 | 2.91 | 3.9 | 84.0 | 36.9 | 47.1 | 41.9 | 459 | 11.0 | 40.3 | 45.4 | 60.0 | 37.1 |
| B12 10 % | prose | 22.6 | 2.25 | 4.0 | 89.0 | 37.0 | 52.0 | 45.8 | 521 | 11.4 | 42.3 | 47.4 | 62.2 | 37.1 |
(*) exact I/O exceeds the exposed stall: it includes draft-phase MTP-expert loads and the records3 CPU rebuild, not overlap.
Floor = max(compute, MB / 14 GB/s). Bytes-per-miss assumed 2.93 MB (exact records) / 2.16 MB (MSB-3 base part; sidecar
misses are 0.77 MB, so tier bytes and GB/s are slight over-estimates).

## Findings
1. **Reads and compute run strictly in series in every mode: the exposed stall equals the demand I/O time** (v3: 49.6 vs
   45.0, 25.4 vs 23.3, 104.2 vs 102.7 ms). v3's OoO achieves ~0 overlap: it can only hide the hit experts' compute
   (~0.3 ms per layer at S=4) behind ~0.8-1.2 ms of reads per stalled layer; the next layer's attention depends on this
   layer's MoE output, so nothing else can run. (Matches the main track's measured +1-6 % for OoO.)
2. **The SSD runs at its ceiling while reading** (13.6-14.9 GB/s exact / v3; B12 11 GB/s with smaller waited batches).
3. **Every pass is 1.5-2.1x above its floor** (v3: code 85 vs 51, prose 54 vs 31, edit 161 vs 114 ms; B12 84-89 vs 40-42).
4. NewIdeas' prior: **partly confirmed.** At the floor, v3 code and edit are SSD-bytes-bound (637 / 1534 MB per pass = 45 /
   110 ms at 14 GB/s vs 35 / 57 ms compute); prose is compute-bound (280 MB = 20 ms vs 29 ms compute). But today's passes are
   far from the floor because overlap is ~0, and overlap is not available exactly within a token (layer dependency).

## Lever 1's quality gate (06:05-06:14, scripts/kl125_screen.py; data/kl125/screen.txt): 2-bit tails FAIL
Per-part prefill KLD vs the 4-bit reference on the Eval suite's own-output tf sets (138 sequences, 34,457 positions; the suite's
machinery read-only; targets = 1.10 x ar4):
| plan | code | gsm | ifeval | multi | pooled |
|---|---|---|---|---|---|
| floor (same weights) | 0.00037 | 0.00068 | 0.00345 | 0.00220 | |
| ar4 (accepted) | 0.00230 | 0.00408 | 0.01856 | 0.01868 | |
| 4433333333 (v3, ships) | 0.00172 | 0.00275 | 0.01367 | 0.01097 | |
| 4433332222 | 0.00267 | 0.00433 | 0.02153 | 0.01862 | |
| 4422222222 | 0.00669 | 0.01105 | 0.05002 | 0.04950 | 0.0257 |
| 4322222222 | 0.00738 | 0.01234 | 0.05515 | 0.05532 | 0.0286 |
| 4222222222 | 0.00981 | 0.01722 | 0.07497 | 0.07680 | 0.0393 |
| target 1.10 x ar4 | 0.00253 | 0.00449 | 0.02041 | 0.02055 | |
Every 2-bit-heavy plan fails every part by 2.4-3.9x (GSM8K-50 accuracy was blind to it: 4422222222 scored 49/50); even
4433332222 is over on code and ifeval. Under the formal gate, bytes per miss via 2-bit tails is limited to the last ranks
(4433333322-class: ~5 % of expert bytes).

## Levers per pass (v3 code, today 85.1 ms)
- **Bytes per miss** (records4 2-bit-heavy tails, ~0.55x bytes): works WITHOUT any overlap because stall = I/O time:
  85.1 -> ~60 ms (1.4x); prose 54 -> 41 (1.3x); edit 161 -> 113 (1.4x). The cheapest exact-engine lever (kernels ready,
  quality gate pending: 4422222222 passes GSM8K; 4322222222 / 4222222222 screens pending).
- **Overlap**: the only way to overlap reads with compute is to issue them before the layer's routing is known (look-ahead
  prediction, MoEPrefetch recall@10 0.67-0.73) or to let the token proceed without them (deferral: non-exact). With good
  overlap + 0.55x bytes: v3 code -> ~38 ms (2.25x). I8's hardware null result says issuing predicted reads AFTER demand
  reads doesn't help; issuing them DURING the previous layer's compute (when the SSD is idle) is the untested form.
- **More residency / fewer misses**: misses scale with the miss rate; B12's 10 % runs at ~50 % misses on its waited ranks
  (188 waited reads per code pass). **Measured weak (TrainLab 10-22, B12 + MTP 3:3:0, C_l = 0.4 C_m; ~/TrainLab/d1i/out/):**
  C_m 51 -> 56 -> 61 -> 66 (11.2 -> 13.0 GB): waited reads per pass code 201 -> 193 -> 178 -> 180, long 256 -> 241 -> 242 ->
  217 (-6 % / -15 %); decode +7-14 % (on code partly changed text: frugal outputs depend on residency).

## SSD idle windows and the look-ahead decision (06:42, BIGMOE_TRACE per-layer gate events; stall/idle.py; v3 at 24 GB)
| prompt | verify ms/pass | SSD busy | idle between layers | median window capacity | best case top-2 (p 0.95) | top-4 (p 0.90) | top-10 (p 0.69) |
|---|---|---|---|---|---|---|---|
| code | 86.5 | 38.2 | 32.2 | 2 reads (25 % >= 3) | 76.7 (1.13x) | 74.5 (1.16x) | 74.3 (1.16x) |
| prose | 53.0 | 18.3 | 24.7 | 1 read (4 % >= 3) | 45.9 (1.16x) | 45.6 (1.16x) | 45.5 (1.16x) |
| edit | 161.1 | 83.5 | 46.5 | 4 reads (87 % >= 3) | 147.0 (1.10x) | 136.9 (1.18x) | 135.5 (1.19x) |
Best case = predicted reads for layer L+1 issued exactly at layer L's release, landing before L+1's routing (no contention,
zero issue/bookkeeping overhead), each useful read removing one demand read (0.165 ms) and a fully covered layer also its
0.191 ms latency. The idle time is real (25-47 ms/pass) but fragmented into 48 windows of 1-4 reads, so more candidates at
lower precision add nothing (capacity-bound). Breakthrough's hardware A/B of I8 (arm 1) measured 0-5 % against this 13-19 %
ideal: real overheads and waste eat most of it.
**Decision: do not build a new look-ahead prefetch as a main lever** (ideal +13-19 %, realistic <= ~5-10 %). At most a cheap
tuning of the existing JIT prefetch (top-2 per row, issued at the previous layer's release, sized to the window). The exact
line is at its floor-minus-overlap for now; the structural levers left are fewer bytes per read under the gate (none found:
2-bit tails fail) and fewer reads per pass (residency is weak).

## Non-exact deferral: quality passes, speed +22-28 % MODELLED BEFORE ENGINE OVERHEADS at 24 GB (07:15-07:25; one page for NewIdeas / Kamen)
**Quality** (NewIdeas' step scorer tf_defer125.py on fork125bt read-only, the Eval suite's tf reference, C=128 (24 GB),
exact 4-bit streaming loads; every 8 = 18 sequences / 3,216 positions; control every 16; data/stall/defer.txt):
| arm | gsm | code | ifeval | multi | pooled | late per token |
|---|---|---|---|---|---|---|
| dh_w4 (ranks 0-3 waited, cold 4-9 half a layer late) | 0.0015 (0.992) | 0.0006 (1.000) | 0.0087 (0.957) | 0.0041 (0.988) | 0.0026 | 59 of 82 reads |
| dh_w2 | 0.0021 (0.987) | 0.0008 (1.000) | 0.0098 (0.960) | 0.0040 (0.986) | 0.0031 | 72 |
| d1_w4 (one full layer late) | 0.0020 (0.991) | 0.0007 (1.000) | 0.0090 (0.962) | 0.0043 (0.986) | 0.0030 | 59 |
| exact streaming control (dh_w10) | 0.0009 (0.995) | 0.0006 (1.000) | 0.0089 (0.968) | 0.0019 (0.993) | 0.0016 | 0 |
| formal target 1.10 x ar4 (prefill path) | 0.0045 | 0.0025 | 0.0204 | 0.0206 | | |
Every arm is 2.3-5x under every per-part target; deferral costs +0.0006-0.0012 on gsm, +0.002 on multi, ~0 on code and
ifeval (ifeval sits at this path's own floor). Caveat: measured with exact 4-bit experts; on top of v3's 3-bit tiers the
costs add (code would be tight: v3 0.00172 + ~0.0006 vs 0.00253) -> needs one combined measurement.
**Speed** (stall/defer_sim.py: the traced v3 passes replayed with 72 % of each layer's reads late, one FIFO SSD at 13.1 GB/s
+ 0.191 ms per batch; late reads due before the next layer's MLP read; deltas applied to the measured verify time):
code 86.5 -> 69.7 ms (1.24x), prose 53.0 -> 43.5 (1.22x), edit 161.1 -> 125.9 (1.28x). One full layer late gives the same
(1.25 / 1.22 / 1.28x): with deferral the SSD becomes continuously busy, so the pass is then SSD-bytes-bound.
**Build cost**: phase A/B release exists single-row in Breakthrough's fork125bd ext; MTP verify passes need a multi-row phase B
(pairs index rank-major rows) - the one new piece, and it MUST BE CHEAP: Breakthrough's late-rank-1 attempt on v3 lost its whole
gain to two extra phase-B spins and an extra 3c call per layer. The +22-28 % is modelled before those engine overheads.
No build before Kamen decides (non-exact on the main track is his call). Pending: the combined v3 tiers + dh_w4 quality
measurement (code is the part that could fail).

## Gating number: v3 tiers + dh_w4 PASSES every part (Breakthrough, 2026-10-22 ~08:0x, the real v3f decoder)
tf_b12.py (the Eval suite's step_forward / var_head on load_b12(0.25): C_m 128, C_l 51, records3 plan 4433333333, HC bf16,
B12's own decode block); 18 sequences / 3,216 positions, the same for both arms; ~/Breakthrough/data/tfb12/:
| arm | gsm | code | ifeval | multi | top-1 (gsm/code/ifeval/multi) |
|---|---|---|---|---|---|
| v3 control (all top-10 waited) | 0.00298 | 0.00129 | 0.01763 | 0.00989 | 0.992 / 0.998 / 0.944 / 0.974 |
| v3 + dh_w4 (ranks 0-3 waited, cold 4-9 late) | 0.00334 | 0.00101 | 0.01807 | 0.01108 | 0.992 / 0.998 / 0.938 / 0.971 |
| target 1.10 x ar4 | 0.00449 | 0.00253 | 0.0204 | 0.0206 | |
Paired dh_w4 - control (bootstrap by position, optimistic): gsm +0.00036 [-0.00012, +0.00086] (74 % of target); code -0.00028
(40 %: code is SAFE, my caveat was wrong: v3's step-path code KLD is 0.0010-0.0013); ifeval +0.00045 [-0.00149, +0.00213]
(89 %; upper bound 0.02071 just crosses 0.0204: the TIGHT part); multi +0.00119 [+0.00035, +0.00213] (significant, 54 %).
My independent wrapper run (stall/tf_defer_v3.py) was cancelled as redundant.
**For Kamen:** v3 + half-layer deferral of cold ranks 4-9 keeps every part under the formal gate (IFEval tight), and is
modelled at +22-28 % per pass at 24 GB before engine overheads (the multi-row phase B must be cheap).
