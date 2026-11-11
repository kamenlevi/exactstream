# Breakthrough scoreboard: bytes per token, by goal

Model: Qwen3.8-Flash-Next 125B, the 4-bit MLX "bare" pack BigMoE runs (48 layers: 36 GDN + 12 QSA, hidden 2560 with 4
hyper-connection streams, 512 routed experts of width 640 per layer, top-10, one shared expert, vocab 248,320).
Numbers marked *sim* come from replaying the real 125B routing traces (MoEPrefetch `data/traces125/*_800.npz`: code, chat,
Bulgarian, math; 2,926 decode tokens; eager, one token per step): `scripts/ws_sim.py`, `ws_bounds.py`, `waitr_sim.py`,
outputs in `data/`. Everything else is measured by the sibling sessions (sources in brackets; MoonShots' handover,
`HANDOVER_FROM_MOONSHOTS.md`, has the details) or computed from the tensor shapes.

## 1. Machines (planning numbers)

| machine | RAM | memory bandwidth | SSD read | planning numbers used below |
|---|---|---|---|---|
| Kamen's MacBook Pro, M5 Pro | 64 GB | 280 GB/s achieved (307 rated) | 14.6 GB/s at queue depth 32, 6.95 at QD 1; 0.34 ms for one cold expert [MoonShots] | 250 GB/s, 10.2 GB/s (70 % of peak) |
| Cheapest new Mac mini (Oct 2026): **M6**, 16 GB / 256 GB, $899, on sale since Sep 22 | 16 GB | not found (M5 base: 153 GB/s rated) | **~3.2 GB/s** (256 GB; 512 GB+ models ~6.4) [ithinkdiff, Blackmagic] | 120 GB/s, 2.25 GB/s |
| (the older M4 mini, if it counts) | 16 GB | 120 GB/s rated | ~2.8 GB/s (256 GB) | about the same as the M6 |
| An 8 GB Mac (M1/M2/M3 base; Apple has sold no 8 GB Mac since late 2024) | 8 GB | 68-100 GB/s | ~1.5-2.9 GB/s (256 GB) | 75 GB/s, 1.4 GB/s |

> (a quote from the project owner, kept private)
chip, i.e. the M6 row above (memory bandwidth still to be looked up). Its 256 GB disk must also hold the model:
**~100 GB of model files at most.** Today's pack needs ~105 GB (trunk incl. experts 71.7 GB, n-gram table 32.0 GB, MTP
head 1.7 GB); with ~3.25-bit experts and the dense diet it is ~85 GB, and the 4-bit n-gram table is the next big item.

## 2. Where every byte goes today

| part | stored | read per eager token (1 row) | read per MTP depth-3 pass (4 rows, 2.8 tokens) | notes |
|---|---|---|---|---|
| routed experts | 24,576 x 2.76 MB = **68.0 GB** (SSD; 17-34 GB cached today) | 480 uses = **1.33 GB** | ~26 unique per layer = **3.45 GB** | the union of 4 rows is ~2.6x one row [MoonShots r10, r22] |
| GDN projections (4-bit) | 36 x 32.6 MB = 1.17 GB | 1.17 GB | 1.17 GB | |
| QSA projections + indexer | 12 x 29.8 MB = 0.36 GB | 0.36 GB | 0.36 GB | |
| hyper-connection mixers (**bf16**) | 1.27 GB | 1.27 GB | 1.27 GB | 37 % of the dense bytes; 8-bit = -0.63 GB, text coherent, GSM8K not run [MoonShots r28] |
| shared experts, routers, PLE projections | 0.13 + 0.07 + 0.07 GB | 0.27 GB | 0.27 GB | |
| lm_head (4-bit) | 0.36 GB | 0.36 GB | 0.36 GB | |
| **dense subtotal** | **3.4 GB** (3.2 by MoonShots' op count) | **3.4 GB** | **3.4 GB** | read in full every pass |
| MTP head | 1.67 GB | - | 3 drafts x (lm_head 0.36 + head ~0.13) = **1.46 GB** | the shared lm_head dominates the drafts |
| GDN state + QSA KV | 113 MB fp32 state; 24 KB per context token | ~0.28 GB | ~0.28 GB | QSA reads only ~2,048 selected tokens |
| n-gram table, embeddings | 32 GB + 0.36 GB (SSD-able) | ~16 rows, ~26 KB | same | depend only on token ids, so prefetchable |
| **total** | **~73 GB of weights** | **5.0 GB/token** | **8.6 GB/pass = 3.1 GB/token** | MoonShots' 2.3 GB/token leaves out the 3 draft lm_head calls |

- In RAM at 250 GB/s this caps one stream at ~81 tok/s with MTP (it doesn't fit in 64 GB at 4-bit anyway).
  MTPLX's own forge run of this pack, fully in RAM on unknown hardware: 47 tok/s plain, 76 with MTP depth 3.
- Today (BigMoE, exact): 43.5 tok/s at a 34 GB expert cache, 27.7 at 24 GB total. At 17 GB of experts half the wall
  time is the SSD loader at ~4.7 GB/s, a third of the SSD's rate [MoonShots r22b]; per-layer compute runs at ~45 % of
  the memory read ceiling [MoonShots r26b].
- **RAM floor with zero experts cached: dense 3.4 + MTP head 1.7 + runtime ~1.5 = ~6.6 GB.** The 6 GB goal is below
  today's floor before a single expert is cached; at 12 GB only ~5.4 GB (C = 40 of 512 per layer, 7.8 %) is left for experts.
- Projected today at 12 GB total (not measured): ~175 misses/token (*sim*) = 485 MB/token from the SSD, ~11-14 tok/s.

## 3. Corrections to the brief's physics framing

- "~4.5 GB per token shared over ~3.5 tokens": a verify pass reads ~8.6 GB (its 4 rows need ~26 experts per layer,
  plus 1.46 GB of draft lm_head reads) and commits ~2.8 tokens, so **3.1 GB per token** (2.3 without the drafts'
  lm_head). The single-stream ceiling in RAM is ~80-120 tok/s, not 150-250.
- "LRU at a 25 % cache hits ~73 %" is a Qwen3-30B number (FlashMoE). On this model's own traces LRU hits **~85 %** at
  25 % eager (*sim*: 72 misses/token; BigMoE measured 77.6) and ~80 % under MTP verify, because consecutive tokens share experts.
- Per-dependent-kernel latency (~20 us, MoonShots r31) is a second floor: today a layer costs ~1 ms per pass. Any goal
  above ~200 tok/s needs ~0.1 ms per layer-pass, i.e. a different kernel structure, not just fewer bytes.

## 4. What the routing traces say (*sim*, per token, 480 expert uses)

| expert cache per layer (C of 512) | 13 (2.5 %, 1.7 GB) | 26 (5 %, 3.5 GB) | 51 (10 %, 6.9 GB) | 128 (25 %, 17 GB) |
|---|---|---|---|---|
| LRU misses (= SSD loads) | 268 | 208 | 152 | 72 |
| Belady, the best any exact policy can do | 230 | 147 | 92 | 39 |
| exact, K tokens per pass: K = 16 / 64 / 256 | 169 / 100 / 48 | 149 / 92 / 46 | 120 / 79 / 42 | 64 / 50 / 33 |
| **wait for the top-2 ranks only, substitute or drop the rest** (MoEPrefetch's mode) | 40 | 30 | **20** | **10** |
| ...of which ranks substituted / dropped (quality risk) | 55 / 214 | 79 / 148 | 94 / 92 | 92 / 51 |

- Working set: within W tokens one layer touches ~10 (W = 1), 26 (4), 66 (16), 142 (64), 242 (256) and ~330 (800)
  distinct experts. A few hundred tokens touch half of the model, which is why caching alone saturates early.
- Exact levers are capped: a perfect cache policy gives 1.66x at 10 %; verifying K tokens per pass gives 1.06x at
  K = 4 (today) and needs K = 64 accepted tokens for 1.9x.
- Only changing the routing moves the SSD term by 5-10x: wait-top-2 at 10 % needs 7.5x fewer loads than LRU. Its quality
  is the whole question: ~190 of 480 uses per token are not the router's choice. Evidence so far: MoEPrefetch's 35B at
  19-25 % cache scored GSM8K-150 identical to the full model (ppl +3.7 to +6.2 %), pure substitution lost 14 points,
  and the 125B at 50 % scored 19/20 [MoEPrefetch FINDINGS]. (The sim substitutes only from the router's ranks 10-19,
  the ids the traces keep, so its drop counts are pessimistic.)

## 5. Budget per goal

RAM bytes/token = what memory bandwidth allows at the target speed; SSD bytes/token = what the SSD delivers at 70 % of
peak, assuming every read is hidden behind compute. "x" = how many times fewer bytes than today at that RAM size
(today: 3.1 GB/token from RAM; 485 MB/token from the SSD at 12 GB, 1.33 GB/token with no expert cache).

| goal | machine | tok/s | ms/token | RAM bytes/token | SSD bytes/token | RAM capacity for weights |
|---|---|---|---|---|---|---|
| small, lower bar | Kamen's Mac, 12 GB | 45-50 | 20-22 | 5.0-5.6 GB (ok today) | 204-227 MB (**2.1-2.4x**) | ~10.5 GB: fits today's dense + MTP head |
| small, lower bar | Kamen's Mac, 6 GB | 45-50 | 20-22 | ok | 204-227 MB (**5.9-6.5x**) | ~4.5 GB: dense + head must shrink ~1.2x, ~0 experts cached |
| small, target | M6 mini, 12 GB | 50-75 | 13-20 | 1.6-2.4 GB (**1.3-1.9x**) | 30-45 MB (**11-16x**) | ~10.5 GB |
| small, target | M6 mini, 6 GB | 50-75 | 13-20 | 1.6-2.4 GB (**1.3-1.9x**) | 30-45 MB (**30-44x**) | ~4.5 GB |
| same, on Kamen's Mac | Kamen's Mac, 6-12 GB | 100-150 | 6.7-10 | 1.7-2.5 GB (**1.2-1.8x**) | 68-102 MB (**4.7-7x** at 12 GB, 13-20x at 6 GB) | |
| small, ideal | Kamen's Mac, 6-12 GB | 400-600 | 1.7-2.5 | 0.42-0.62 GB (**5-7x**) | 17-26 MB (**19-29x** at 12 GB) | + ~0.1 ms per layer-pass |
| ultra-light | 8 GB Mac, 4 GB | 30-50 | 20-33 | 1.5-2.5 GB (**1.2-2.1x**) | 28-47 MB (**28-47x**) | ~3 GB: dense must be ~2x smaller, ~0 experts cached |
| **speed: the goal is one stream** (Kamen) | Kamen's Mac, in RAM | 1000 | 1.0 | 0.25 GB (**12x**) | - | whole model in ~45 GB = **~2.8 bits/weight** average |
| speed, many streams (Kamen's second target) | Kamen's Mac, in RAM | 1000 in total | - | ~45 GB read per step, B ~ 32-64 streams x 2.8 tokens | - | model ~30-45 GB at Q4 quality + 7-14 GB of per-stream state |

Notes:
- The many-streams line works on paper: one step reads (nearly) the whole model once, ~0.18 s at 250 GB/s, and carries
  B x 2.8 tokens; compute is 13 GFLOP/token, ~3.3 TFLOP per step at B = 64 (~0.13 s of the 25 TFLOPS). It needs the
  model at ~2-3 bits/weight with Q4 quality, and per-stream GDN state (113 MB fp32) plus KV.
- The one-stream line needs 12x fewer RAM bytes per token and ~3x fewer per-pass kernel steps. No known exact method gets there.

## 6. What each lever can give at most

| lever | best case on this model | evidence |
|---|---|---|
| exact cache policy | 1.66x fewer misses at 10 % (Belady), realistic 1.1-1.3x | *sim*; FlashMoE +7 % |
| exact amortization (MTP, drafts) | 1.06x at 2.8 tokens/pass; 1.9x needs 64 accepted tokens per pass | *sim* |
| fewer bits per expert, equal quality | Q4-equal at ~3.25 bits stored with a trellis code (1.4x bytes, 1.4x more cached), ~2.7 read with a rank-nested code (1.7x); no MoE result at <= 2.25 bits is Q4-equal | research/bits_limits.md; MoonShots 4433333333 GSM8K 427 vs 431 of 450 |
| routing change (substitute/drop low ranks) | up to 7.5x fewer SSD loads at 10 % (wait-top-2) | *sim*; quality: 35B GSM8K-150 identical at 19-25 % |
| dense part | HC mixers 8-bit -0.63 GB; dense at 3 bits untrained +12.5 % ppl (dead without training) | MoonShots r18, r28 |
| kernel structure | 1 ms -> ~0.1 ms per layer-pass needs a persistent/megakernel class design | MoonShots r6, r26-31 |

Reading: exact levers multiply to ~2-3x, enough for the lower bar at 12 GB on Kamen's Mac and nothing beyond it. Every
other goal needs a lossy lever that keeps quality. Routing change is the only one with a 5-10x lever on the SSD term;
fewer bits is the only one that shrinks RAM capacity and the RAM term together; the dense part (3.4 GB + 1.7 GB head)
must shrink for anything at or below 6 GB.

### How the levers combine (*sim*, `scripts/combo_sim.py`, `data/combo_sim.txt`)

SSD MB per token; r = how many top ranks are always exact (r = 10: exact routing); in brackets ranks substituted /
dropped per token (of 480). "Diet" = experts in a Q4-equal trellis code at ~3.25 bits/weight (research/bits_limits.md:
today's affine Q4 spends ~1 bit/weight more than its error needs), dense part cut to ~2.2 GB (8-bit HC mixers, 3.5-bit
projections), MTP head ~1.0 GB. Then 12 GB total leaves 7.3 GB of experts = 14.8 % resident; 6 GB leaves 1.3 GB = 2.5 %.
The 2.5-bit row is beyond what any published MoE result shows to be Q4-equal; it is kept as the optimistic bound.

| setup | exact (r = 10) | r = 3 | r = 2 | r = 1 |
|---|---|---|---|---|
| 12 GB total, today's format (7.8 % resident) | 480 MB | 106 [90/79] | 66 [90/111] | 31 [83/167] |
| **12 GB total + 3.25-bit diet (14.8 % resident)** | 234 MB | 47 [89/41] | **30 [95/68]** | 15 [92/130] |
| 12 GB total + 2.5-bit diet (20 % resident; not Q4-equal by published evidence) | 139 MB | 27 [84/31] | 18 [93/56] | 10 [93/123] |
| 6 GB total + 3.25-bit diet (2.5 % resident) | 541 MB | 126 [58/175] | 80 [55/214] | 38 [49/264] |
| today's 24 GB (25 % resident, 4.5 bits) | 199 MB | 41 [79/26] | 28 [92/51] | 18 [93/121] |

- **Fewer bits buy residency, and residency is what makes substitution safe.** At 12 GB, today's format leaves 7.8 %
  of experts resident; the 3.25-bit diet gives ~15 %, close to the 19-25 % where MoEPrefetch's 35B kept GSM8K-150
  identical with wait-top-2. There the SSD term drops to 30-47 MB/token (r = 2-3): ~50-75 tok/s on the M6 mini's SSD,
  ~220-340 on Kamen's, against 234 MB with exact routing.
- The RAM term is then the limit: per MTP pass dense ~2.2 GB + ~26 experts/layer x 2.0 MB (2.5 GB) + 3 drafts with a
  trimmed lm_head (~0.6 GB) + state (0.3 GB) = 5.6 GB for 2.8 tokens, ~2.0 GB/token: ~60 tok/s on a 120 GB/s mini,
  ~125 on Kamen's Mac. That is the small-target band on paper at 12 GB, if quality holds for both the trellis experts
  and the substitution. At 6 GB the residency (2.5 %) is below the ~4-token working set; nothing here reaches it yet.
- Multi-token passes add little once routing is rank-gated (K = 16 changes r = 2 by < 5 %): amortization only matters
  for exact routing, or for the 400-600 and 1000 tok/s goals where the RAM term dominates.

### Pass-level expert budget for long passes (*sim*, `scripts/pass_budget_sim.py`, `data/pass_budget_sim.txt`)

For the RAM-bound goals: a pass of K drafted tokens may use only U experts per layer (chosen by the rows' rank-weighted
votes). K = 16, U = 26: every row keeps its rank-0 expert, 97 % of its ranks 0-1 and 68 % of its top-10; with 2.5-bit
experts and a 2 GB dense part that is ~0.24 GB per token if all 16 tokens are kept, i.e. the bandwidth for ~1000
tok/s. K = 32, U = 40: rank-0 100 %, ranks 0-1 96 %, ~0.15 GB/token. So one-stream 1000 tok/s needs ~16 kept tokens
per pass AND budgeted routing AND ~2.5-3-bit weights AND per-pass latency under ~16 ms; research/sparse_consult.md
lists the other walls (the drafter must itself emit 1000 tok/s, i.e. a block or diffusion drafter).

## 7. Leaderboard (every idea is scored on these four numbers)

Quality follows the orchestrator's suite (`~/.local/share/pacer/briefs/quality_suite.md`): prefill-only KLD and
top-1 agreement on the reference's own outputs, letter-probability MMLU-Pro/GPQA, small paired decode sets, judged
against a same-weights floor and an accepted anchor (TOST +-2 points).

| idea | RAM bytes/token | SSD bytes/token | real RAM footprint | sustained tok/s (machine) | quality (suite; % answers equal to today's Q4) | status |
|---|---|---|---|---|---|---|
| today, BigMoE exact, 34 GB cache | 3.1 GB | ~0.1-0.2 GB | ~40-44 GB | 43.5 (M5 Pro) | reference | production |
| today, BigMoE exact, 24 GB total | 3.1 GB | ~0.22 GB | 24 GB | 27.7 (M5 Pro) | reference | production |
| today's engine at 12 GB total | 3.1 GB | ~0.49 GB (*sim*) | 12 GB | ~11-14 (projected) | reference | not measured |
| exact at 20 % residency (C = 102, MoEPrefetch fork, eager, MTP off) | 5.0 GB | not logged | 19.9 GB peak | 13.8 eager (M5 Pro) | reference | measured (expA) |
| D1 wait-top-2 at 20 % residency, background loads on (C = 102, eager, MTP off) | 5.0 GB | ~0.31 GB (14.4 waited + 97 background reads/token) | 19.9 GB peak | 21.5 eager (M5 Pro) | **GSM8K-150 145 vs 147; 144/150 answers identical (96 %; same-weights floor 147/150)** | measured (expA, 2026-10-19) |
| D1 wait-top-2 at 10 % residency, background loads on (C = 51) | 5.0 GB | ~0.45 GB (23.1 waited + 139 background) | 13.2 GB peak | 19.0 eager (M5 Pro) | **GSM8K-150 144 vs 147; 144/150 identical (96 %); exact-only right #12 (also a noise-arm flip), #139, #146** | measured (expA, 2026-10-19) |
| **D1 frugal wait-top-2 at 10 % residency** (no SSD reads for missing ranks >= 2: w2 path with BIGMOE_SPARE=0) | 5.0 GB | **78 MB** (28.1 waited reads/token; *sim* said 20) | **13.2 GB peak** | **22.0 eager** (M5 Pro; MTP off, fork's older base) | **GSM8K-150 146 vs 147; 145/150 answers identical (96.7 %)**; exact-only right #40, #58; new-only #87 | measured (2026-10-20) |
| D1 frugal at 5 % residency (C = 26) | 5.0 GB | 127 MB (45.7 waited reads/token) | **9.9 GB peak** | 18.8 eager | GSM8K-150 143 vs 147; 142/150 identical (94.7 %); exact-only right #12 (noise-type), #40, #59, #146, #147; new-only #87 | measured (2026-10-20) |
| **D1 frugal x MTP 3:3:1 at 10 %** (fork125bt, multi-row frugal verify, spec125; data/expS) | | | 14.3 GB (code 128 tok), 16.3 GB peak over GSM8K | **39.8 code (3:3:0) / 38.1 GSM8K / 30.3 prose** | **GSM8K-150 146 vs 147; 146/150 answers equal to exact eager**; Eval suite tfstep KLD (eager frugal 10 %) 0.057 vs ar4 0.0096 | measured 2026-10-20 |
| **D1 frugal x MTP 3:3:0 at 5 %** | | | 10.9 GB (code), 13.0 GB GSM8K peak | **34.4 code / 35.1 GSM8K** | GSM8K-150 144 vs 147; 144/150 equal | measured 2026-10-20 |
| D1 frugal at 20 % (C = 102) | 5.0 GB | 61 MB (21.9) | 19.9 GB | 23.3 eager | not run | speed only |
| D2 trellis (RHT + bitshift trellis, L = 12, Viterbi; data/d2/trellis_nmse.txt) | | | | | NMSE on real 35B experts: k=4 (4.05 b) 0.0052, k=3 (3.05 b) 0.0191, k=2 0.071; affine Q4 (4.5 b) 0.0085; Q4-equal NMSE at ~3.6 bits | measured |
| D2 trellis k=3, L=8, ALL routed experts of the 35B (fake-quant, data/noise35/experts_t3_L8.txt) | | | | | NMSE 0.0262, **KLD 0.0856** (top-1 90.1 %) vs the noise curve's ~0.087 at that NMSE: trellis error costs what white noise costs | measured 2026-10-20 |
| D2 trellis k=4, L=8, all routed experts of the 35B (data/noise35/experts_t4_L8.txt) | | | | | NMSE 0.0081, **KLD 0.0383** (top-1 93.7 %), below affine Q4 experts' 0.0408 (NMSE 0.0085) | measured 2026-10-20 |
| D2 matched white noise, all routed experts (data/noise35/experts_noise_matched.txt) | | | | | n0.026: KLD 0.0926 vs trellis k=3 0.0856; n0.0081: KLD 0.0397 vs trellis k=4 0.0383: **at equal NMSE the trellis costs 3-8 % less KLD than white noise, and Q4 the same as noise**: KLD is set by NMSE alone, so NMSE per bit ranks expert codes | measured 2026-10-20 17:14 |

## 8. Overnight measurements (2026-10-19/20)

### D2: noise-injection sweep on the 8-bit Qwen3.6-35B (`scripts/noise35.py`, `data/noise35/results.txt`)
KLD against the 8-bit model (near-bf16) over 3 x 1,365 teacher-forced tokens (prose, GSM8K solutions, code), one tensor
class changed at a time. "Q4" = that class requantized to affine 4-bit g64; "noise X" = white noise at NMSE X (an ideal
trellis code: 0.0086 ~3.5 bits, 0.012 ~3.25, 0.017 ~3.0, 0.024 ~2.75, 0.034 ~2.5).

| class | Q4 NMSE | Q4 KLD (top-1) | noise 0.0086 | noise 0.012 | noise 0.017 | noise 0.024 / 0.034 |
|---|---|---|---|---|---|---|
| routed experts gate/up | 0.0085 | 0.031 (94.6 %) | 0.032 | 0.039 | 0.044 | 0.058 / 0.074 |
| routed experts down | 0.0084 | 0.027 (95.3 %) | 0.022 | 0.028 | 0.040 | 0.042 / 0.047 |
| all routed experts | 0.0085 | 0.041 (94.1 %) | - | 0.052 | 0.065 | - |
| GDN projections | 0.0096 | **0.057** (91.0 %) | 0.049 | 0.070 | 0.087 | - |
| attention projections | 0.0094 | 0.035 (93.3 %) | 0.036 | 0.040 | 0.056 | - |
| shared expert | 0.0104 | 0.022 (94.7 %) | - | 0.025 | 0.028 | - |
| lm_head | 0.0090 | 0.009 (94.4 %) | - | 0.010 | 0.013 | - |
| **everything at Q4 (~ the 4-bit pack)** | 0.0086 | **0.112 (86.9 %)** | | | | |
| null: reload, nothing changed | | 0.000 (100 %) | | | | |
| null: experts gate/up re-quantized to 8-bit (NMSE 0.00001) | | **0.011 (96.9 %)** | | | | |

- **Floor: any change at all costs ~0.011 KLD / ~3 % top-1 on this MoE** (an NMSE of 1e-5 already does): routing
  near-ties flip and the flips propagate. Every comparison needs this floor; it is the KLD form of the same-weights
  GSM8K noise arm (147/150 identical).
- Q4's own error is NMSE ~0.0085-0.0105 in every class, as the bits brief predicted. At equal NMSE, white noise costs
  the same as Q4 on gate/up, attention and the shared expert, but **less on down_proj and GDN** (0.022 vs 0.027, 0.049
  vs 0.057): Q4's error there is structured, so a rotated (incoherent) code beats affine Q4 at equal bits.
- Trellis check (2026-10-20, all routed experts, L=8): k=3 NMSE 0.0262 KLD 0.0856 vs white noise at 0.0260 KLD 0.0926;
  k=4 NMSE 0.0081 KLD 0.0383 vs noise at 0.0081 KLD 0.0397 (single noise seeds). The trellis is no worse than white
  noise at equal NMSE (3-8 % better), so the noise curve prices any expert code from its NMSE alone: L=12 k=3 (NMSE
  0.0191, 3.05 bits) ~0.070; L=12 k=4 (0.0052, 4.05 bits) ~0.036 < Q4's 0.041 at 0.45 fewer bits; Q4-equal at ~3.6 bits.
- Experts at ~3.25 bits add +0.011 KLD over Q4 experts (+10 % of the whole pack's 0.112); at ~3.0 bits +0.024 (+21 %).
  GDN is the biggest single Q4 contributor (0.057), so a KLD budget is better spent there: experts at ~3.0-3.25 bits
  plus GDN better than Q4 (a rotated 4-5-bit code, +0.3-1 GB on the 125B) can stay at or under the pack's total KLD.

### D3: must-fix gap F, teacher-forced (`scripts/d3_mustfix.py`, `data/d3/analyze.txt`)
The 125B answered 40 prompts (20 GSM8K, 8 code, 8 knowledge, 4 Bulgarian; exact eager, C = 50 %; decode vs teacher-forced
top-1 agreement 99.2 %). Drafter tokens judged against the 125B's top-20, F = answer tokens per must-fix token:

| drafter | exact top-1 | in the 125B's top-3 | margin <= 3 nats | entropy AUROC for must-fix |
|---|---|---|---|---|
| Qwen3.5-2B | 5.6 | 16.0 | 12.0 | 0.89-0.94 |
| Qwen3.5-4B | 7.2 | 27.5 (GSM 71, code 40, knowledge 21, Bulgarian 11) | 22.2 | 0.90-0.95 |

**D3 is dead as an SSD lever** (it needs F >= 100-250; research/sparse_consult.md). For the one-stream speed goal it
needs ~16 kept tokens per pass from a drafter that costs far less than a 4B model per token, plus proof that lenient
acceptance keeps answers.

### Frugal's quality mechanism: the routing is flat (`scripts/gate_mass.py`, `data/gmass/gate_mass.txt`, 2026-10-20 22:05)
Router probabilities on 4 x 800 decode tokens x 48 layers (125B): of the renormalised top-10 mass, rank 0 holds 0.21,
ranks 0-1 0.36, ranks 0-3 0.58; 50 / 70 / 90 % of the mass needs 3.7 / 5.9 / 8.8 ranks. Frugal wait-top-2 at 10 %
(LRU) leaves **29 % of the top-10 gate mass uncomputed per token-layer (P99 66 %)**, at 20 % 23 %. Buying it back by
waiting is costly: wait-until-80 % mass = 98 loads/token (272 MB) at 10 % vs 20 (56 MB) for wait-top-2; 90 % = 352 MB.
Compare ACE (2609.05228): 20 % of slots skipped BY SCORE on the 35B is near free. The Eval suite's KLD 0.057 is
consistent with ~29 % of mass removed; the fixes must cover mass cheaply (cache policy, planes, cross-row sharing).
- Cache policy cannot recover it (`scripts/evict_sim.py`): missed mass at 10 % LRU 0.295, decayed-mass LFU 0.287,
  Belady oracle 0.266 (5 %: 0.371 / 0.355 / 0.333). The skipped experts are cold, not badly evicted.
- Per-expert gate threshold (`scripts/theta_sim.py`) is a cliff because top-10 shares bunch at 0.05-0.15: at 10 %,
  skip-below-0.10 leaves 0.211 skipped at 105 MB/token, 0.07 -> 0.077 at 266 MB, 0.05 -> 0.013 at 385 MB; reading only
  MSB-2 planes for ranks >= 2: 0.07 -> 0.099 skipped + 0.195 at 2 bits for 214 MB. 266 MB/token at 40 tok/s = 10.6 GB/s,
  near the SSD ceiling (frugal today uses ~2 GB/s, 15 %). The gate alone does not separate important experts; ACE adds
  an offline per-expert amplification score (gate x structural/router-direction amplification): next to test.

### Frugal KLD by mechanism (`scripts/tf_frugal_var.py`, `data/tfvar/results.txt`, 2026-10-20 22:06-22:37)
Eval suite's one-token-per-call teacher-forced scorer, its cached reference, 14 ifeval + multilingual sequences (every
4th), 3,308 positions, 125B at C=51 (10 %). Accepted ar4 (AutoRound) full-suite KLD 0.0096; Eval suite base 0.115/0.092.

| variant | KLD | ifeval / multi KLD | top-1 ifeval / multi | waited ranks |
|---|---|---|---|---|
| exact (all top-10 waited; harness floor) | 0.0043 | 0.0054 / 0.0032 | 0.978 / 0.982 | 10 |
| base frugal (wait 0-1, substitute 10-19, renorm) | 0.1066 | 0.127 / 0.086 | 0.862 / 0.903 | 2 |
| drop missing, renorm | 0.1408 | 0.175 / 0.106 | 0.838 / 0.887 | 2 |
| drop missing, no renorm | 0.1039 | 0.126 / 0.082 | 0.850 / 0.907 | 2 |
| substitute at own weight, no renorm | **0.0961** | 0.114 / 0.079 | 0.868 / 0.910 | 2 |
| wait ranks 0-3 | 0.0452 | 0.054 / 0.037 | 0.897 / 0.941 | 4 |
| wait until 70 % of top-10 mass | 0.0201 | 0.024 / 0.016 | 0.936 / 0.958 | 5.94 |
| wait until 85 % | **0.0095** (= ar4) | 0.012 / 0.007 | 0.960 / 0.969 | 7.95 |

Findings: (1) renormalising over the computed experts is harmful (drop+renorm 0.141 vs drop 0.104): missing outputs are
best estimated by ~0, not by scaling the rest; (2) substitution helps a little (0.104 -> 0.096 without renorm); (3) KLD
falls steeply with the covered gate mass (missed ~0.29 -> 0.10, ~0.2 -> 0.045, ~0.11 -> 0.020, ~0.05 -> 0.0095): the
quality bar needs ~85-90 % of the mass computed. At 10 % that is ~100 loads/token by waiting (theta_sim: 272 MB); the
open question is how much a zero-byte similar-expert substitute (SERE, queued) and plane reads (half bytes) buy.

### The 125B's routed experts are mutually orthogonal: no zero-byte substitute exists (`scripts/sim_table.py`, 2026-10-21 00:01)
All 512 experts of all 48 layers run on the same 512 calibration rows (MoE inputs captured from 24 held-out Eval-suite
sequences, exact prefill); per pair the output distance, per expert its norm and the best scaled common proxies.
- Best |cos| between an expert's outputs and ANY other expert's, over all layers: **median 0.024, P90 0.055**, max 0.40
  (layer 0 median 0.085); mean cosine over all pairs +0.002..+0.009. An optimally scaled best other expert leaves
  **100 % (median; 99.8 % at P10)** of the missing expert's energy; the nearest other expert at its own weight
  (distance / norm median 1.17-1.31) is WORSE than dropping it.
- Scaled shared expert, a resident mean-weights expert, and even the mean of all 512 outputs leave 99.5-100 % of a
  routed expert's energy (per-layer medians).
- Expert output norms vary little within a layer (CV 0.15, P90/P10 1.38; layer scale grows from ~1.0 at L8 to ~2.7 at
  L40), so a gate x norm wait rule would barely differ from waiting by gate mass.
**Consequences:** SERE-style substitution, SPICE's shared-expert base and mean-expert fallbacks cannot work on this
model; renormalisation hurts because it scales orthogonal outputs; a missing expert's contribution must be computed
(from SSD bytes, possibly at fewer bits) or accepted as an orthogonal error that scales with the missed gate mass.
The quality levers left: coverage per byte (MSB-3 / MSB-2 tail reads, records3), where the coverage goes (layer
sensitivity: band probes queued), and the training route (research/training_route.md).

### Confidence-gated coverage (BiLD-style fallback), first real runs (tf_frugal_var.py bild, CPU-synchronous residency)
Night subset = Eval suite ifeval + multi, every 8th sequence, 2,353 positions, C=51 (10 %).
| policy | KLD | ifeval / multi | top-1 ifeval / multi | re-runs | loads per step |
|---|---|---|---|---|---|
| bild tau 0.1 (frugal drop step, exact re-run if its entropy > 0.1) | **0.0133** | 0.0143 / 0.0120 | 0.956 / 0.963 | 67.1 % | 104.5 |
| m50_nr (wait by mass 50 %, no renorm) | 0.0432 | 0.0511 / 0.0338 | 0.909 / 0.938 | - | ~42 (replay) |
| b25 (substitutes, partial renorm 0.25) | 0.0912 | 0.1043 / 0.0755 | 0.868 / 0.908 | - | ~20 |
Per-position prediction for this subset (Eval suite data): 65 % re-runs (measured 67 %), mixture KLD 0.0042: the real
KLD is ~3x the mixture because tokens left frugal perturb the context of later (also re-run) tokens. On this hard subset
BiLD sits on the same KLD-per-load curve as wait-by-mass (wait-to-85 %: 0.0095 at ~111 loads); its case is content
adaptivity: predicted re-runs at tau 0.1 are 21.5 % on GSM and 11.4 % on code (vs 65 % here). GSM+code probes queued.
Update 03:38 (same harness; GSM+code = Eval suite tf_gsm + tf_code, every 8th sequence, 2,261 positions):
| policy | subset | KLD | per set | top-1 | re-runs | loads per step |
|---|---|---|---|---|---|---|
| noren (frugal drop, no renorm) | ifeval+multi | 0.1001 | 0.1175 / 0.0793 | 0.852 / 0.906 | - | ~20 |
| bild tau 0.5 | ifeval+multi | 0.0204 | 0.0232 / 0.0170 | 0.943 / 0.954 | 50.0 % | 85.1 |
| bild tau 0.1 | ifeval+multi | 0.0133 | 0.0143 / 0.0120 | 0.956 / 0.963 | 67.1 % | 104.5 |
| **bild tau 0.1** | **gsm+code** | **0.0073** | 0.0082 / 0.0048 | 0.986 / 0.996 | **22.5 %** | **64.6** |
Weighted by the Eval suite's position mix (gsm+code 60 %, ifeval+multi 40 %), bild tau 0.1 ~ KLD 0.0097 (= ar4's 0.0096)
at ~81 loads/token (~220 MB at 10 %); wait-to-85 % pays ~111 loads/token on every kind of text.
Update 04:30, easy text (gsm+code, every 8th): noren 0.0273 (gsm 0.0321 / code 0.0131, top-1 0.970 / 0.993); bild tau
0.1 0.0073 at 22.5 % re-runs / 64.6 reads; **m85_nr (wait to 85 % of the mass, no renorm) 0.0021** (gsm 0.0026 / code
0.0008, top-1 0.993 / 1.000; ~8.1 ranks waited, ~111 reads by replay). Paired on ifeval+multi vs noren (bootstrap 95 %):
bild_e10 -0.087 [-0.093, -0.081], bild_e50 -0.080, m50_nr -0.057, b25 -0.009 [-0.012, -0.006].
Reading so far: without renormalisation the mass rule is the stronger per-byte allocator on both kinds of text; BiLD's
all-or-nothing re-runs and the context effect cost it. Next probes: hybrids (mass-rule first pass + BiLD), adaptive theta
by the previous step's entropy (no re-runs), measured read counts for every variant (tf_frugal_var.py now counts them).

### Tail reads at fewer bits and the rate-distortion frontier (06:15; `rd_sim.py`, `data/gmass/rd_sim_c51*.txt`)
Measured on ifeval+multi (every 8th), 10 %: m85_nr 0.0094 at 106.6 measured reads (~295 MB/token); m85_nr_t2 (waited
tails read as MSB-2) **0.0151 at 19.8 full + 87.2 MSB-2 reads (~189 MB, -36 %)**; m100_nr_t1 (never drop, tails as MSB-1)
0.0461 at ~173 MB (MSB-1 truncation is no better than dropping). **Energy law:** KLD_excess ~ 0.0051 (E / 0.0032)^1.5
with E = missed gate energy + NMSE x energy read at low bits (MSB-2 NMSE ~0.10): predicts m85_nr_t2 at 0.0152 (measured
0.0151). Rate-distortion-optimal per-expert precision (drop / MSB-2 / MSB-3 / 4-bit, optionally trellis 2-bit / 1-bit
copies) on the 125B traces: at the hard subset's ar4-level KLD it needs ~280-310 MB/token without trellis copies and
~245 MB with them (-20 %): the flat, orthogonal tail (~130 missing experts per token at 10 %) must be covered at some
precision. On easy text the same policies sit 4-5x lower in KLD (m85_nr 0.0021 vs 0.0094), so the suite-average ar4
level is reached near wait-to-70 % (~200 MB/token at 4 bits; pending m70_nr on gsm+code).
**Inference-only ceiling:** Q4-like quality at 10 % residency (~14 GB) costs ~150-300 MB/token of SSD reads depending on
the text, i.e. ~50-100 tok/s on the M5 Pro's 14.6 GB/s with perfect load/compute overlap and ~10-20 tok/s on the base
Mac mini's 3.2 GB/s. Frugal's 56 MB/token (KLD 0.057 on the suite) is the only small-RAM regime with room for 100-150
tok/s; whether that KLD keeps "similar intelligence" is the Eval suite's decode verdict (held batch, resumes 08:45).
Update 07:47 (hybrids and adaptive theta, 10 %, every 8th; measured reads per step):
| policy | ifeval+multi KLD (reads) | gsm+code KLD (reads) |
|---|---|---|
| bild_m50_e10 (wait-by-mass 50 % first pass + exact re-run if entropy > 0.1) | 0.0090 (109; 65 % re-runs) | 0.0046 (88; 21 % re-runs) |
| ada_lo50_hi90_e10 (theta 0.5 after a confident step, 0.9 after an uncertain one; no re-runs) | 0.0120 (95.6) | 0.0041 (81.6) |
| m85_nr | 0.0094 (106.6) | 0.0021 (~107 replay) |
| bild_e10 | 0.0133 (104.5) | 0.0073 (64.6) |
All policies fall on roughly one KLD-vs-reads curve per kind of text; suite-mix ar4-level KLD at ~75-90 reads per token
(~200-240 MB at 10 %). Report sent to the orchestrator 07:50.
Routing per kind of text (125B traces, 800 decode tokens each, 10 % LRU): rank-0 mass 0.205-0.217, ranks 0-1 0.35-0.37,
ranks for 85 % 7.9-8.1 for code, chat, Bulgarian and math alike; wait-top-2 missed energy 0.021-0.028, hit rate 0.52
(math) to 0.61 (Bulgarian, the best). So ifeval/multilingual cost more KLD not because their routing is flatter or less
local, but because their next-token distributions are more sensitive (high entropy) to the same expert error: bytes per
unit of coverage are content-independent, only the damage differs.

### Frugal 10 % decode rows (Eval suite stage finished 10:31; PRELIMINARY, scored read-only with its own scorers)
| test | reference | same-weights floor | frugal 10 % (w2f_r10) | frugal vs ref |
|---|---|---|---|---|
| math (n=50) | 86.0 % | 88.0 % | **90.0 %** | lost 1, gained 3, McNemar p 0.62 |
| IFEval (n=100) | 90.0 % | 88.0 % | **88.0 %** | -2.0 [-6.7, +2.7] pt, lost 5 gained 3, p 0.73 (suite rule: > -5 pt) |
| answer agreement with ref | 100 % | 98 % (math) / 97 % (IFEval) | 92 % / 91 % | ar4's rows not on disk: the agreement rule is open |
HumanEval not scored here (it executes generated code: the Eval suite's harness does that). Despite the step KLD of 0.057,
no measurable task loss on math or IFEval: KLD flags drift, the decode tests are the verdict (research/skipping_principles.md).

### Speed of the quality policies, M5 Pro, 10 % residency, eager, cold SSD (`bt_bild.py`, data/day/sp_*, 11:21-11:27)
| policy | code tok/s | prose tok/s | SSD reads/token code / prose | peak |
|---|---|---|---|---|
| frugal (wait ranks 0-1, drop the rest, no renorm) | **25.3** | **26.6** | 29.5 / 25.4 (82 / 70 MB) | 13.2 / 12.1 GB |
| exact (all top-10 waited) | 12.9 | 12.4 | 152.7 / 165.9 (422 / 459 MB) | 13.2 / 12.1 GB |
| BiLD tau 0.5 (exact re-run if entropy > 0.5) | 18.2 | 11.3 | 59.7 / 121.6 (re-runs 25 % / 66 %) | |
| BiLD tau 0.1 | 15.4 | 9.5 | 77.1 / 142.6 (re-runs 39 % / 81 %) | |
Eager exact at 10 % residency is 12.9 tok/s on today's engine; BiLD costs compute on uncertain text (prose re-runs 66-81 %
of tokens: slower than exact). Wait-by-mass rows: the first run crashed in eager mode (BIGMOE_MTP_STREAM=1 without an MTP
head = phantom gate layer) and its MTP rows counted the prompt prefill (spec125 prefills inside): both fixed in
bt_mass.py, re-run queued (day_speed3.sh). 5 % with MTP and the per-pass cap (D1d) runs without aborts (10.7 GB peak).
Wait-by-mass on the fast path (fork125bf GPU wait flags, `bt_mass.py`, data/day/sp3_*, 13:51-13:56), 10 % unless noted:
| policy | code tok/s (SSD reads/token) | prose tok/s (reads/token) |
|---|---|---|
| m85 eager | 14.2 (112) | 13.7 (120) |
| m70 eager | 16.7 (78) | 16.0 (85) |
| ada 0.5/0.9/0.1 eager | 17.7 (72) | 14.0 (118) |
| m85 + MTP 3:3:0 | 19.6 (116; 3.20 tok/pass) | 13.3 (166; 2.17) |
| m70 + MTP | 24.4 (79; 3.37) | 12.3 (126; 2.06) |
| m50 + MTP | 29.6 (48; 3.28) | 18.8 (76; 2.13) |
| m85 + MTP at 5 % (C=26, per-pass cap D1d; 10.7 GB peak) | 21.3 (106; 3.46) | - |
| m70 + MTP at 5 % | 22.2 (98; 3.46) | - |
Per extra read on the critical path: (m85 70 ms - frugal 39.5 ms) / 82.5 reads = 0.37 ms; exact (77.5 - 39.5) / 123 = 0.31 ms:
latency-bound (2.77 MB at 14.6 GB/s = 0.19 ms). Q4-like quality at ~14 GB on today's engine = 14-20 tok/s (frugal 26-40,
main track exact at 24 GB ~29). High coverage + MTP loses on prose (acceptance ~2.1 tok/pass; rejected rows' reads wasted).

### OFFICIAL Eval suite verdicts on frugal (results/report_w2f_r10.txt, report_w2f_r05.txt, 13:25)
- frugal 10 %: **FAIL 5/8**: TF-step KLD 0.0570 = x5.97 ar4 [5.65, 6.31]; P99.9 x6.32; top-1 93.9 % (-3.50 pt vs ar4);
  decode IFEval 88.0 vs 90.0 (-2.0, PASS), HumanEval 90.0 vs 91.7 (-1.7, PASS), math 90 vs 86; agreement 92.4 % (floor
  97.1 %); length x0.968, hesitation, loops PASS.
- frugal 5 %: **FAIL 4/8**: KLD 0.0946 = x9.90; top-1 -5.24 pt; IFEval 83.0 vs 90.0 (-7.0 [-12.4, -1.6], FAIL);
  HumanEval +1.7 PASS; math 86 = ref.
- The fidelity rules are per corpus part (margins.json, pre-registered): 90 % CI upper bound of KLD / ar4's <= 1.10, same
  for P99.9; top-1 within -0.5 pt of ar4. ar4's TF KLD by part: code 0.0023, gsm 0.0041, ifeval 0.0186, multi 0.0187
  (plan4433333333: 0.0017 / 0.0028 / 0.0137 / 0.0110) -> targets code <= 0.0025, gsm <= 0.0045, ifeval <= 0.0204,
  multi <= 0.0206: **easy text binds**. On the every-8 subsets only m85_nr meets every part (0.0008 / 0.0026 / 0.0110 /
  0.0074); ada_lo50_hi90 misses gsm (0.0049); bild variants miss code and gsm.

### Frugal + look-ahead prefetch (day_speed4, data/day/sp4_*, 15:14-15:16, M5 Pro, quiet lock)
| run (10 % unless noted) | eager tok/s (waited misses/token) | + MTP 3:3:0 tok/s (tok/pass, waited/token) |
|---|---|---|
| frugal, code | 25.3 (29.5) [11:21] | **39.2** (3.20, 33.0) |
| frugal, prose | 26.6 (25.4) [11:21] | 28.6 (2.33, 42.7) |
| frugal + PF (12 of the 51 slots spare, PREFETCH 12, QD 4), code | 18.6 (27.0) | abort: "57 experts > C=52 at layer 48" |
| frugal + PF, prose | 11.8 (40.0) | abort (same) |
| frugal + PF at 5 % (C=26, 6 spare), code | 13.9 (40.6) | 23.1 (3.20, 53.9) |
Look-ahead prefetch as built is a net LOSS: the spare slots come out of C (C_main 51 -> 39), so prose waited misses rise
25.4 -> 40.0, and the predictor reads up to 12 predicted experts per layer while frugal consumes 2 (SSD contention with
the demand reads). With MTP the spare pool overflows on the MTP gate layer (multi-row: 57 ids > 52 slots). At equal RAM
the slots are worth more as cache. If revisited: post only the predicted ranks 0-1 (2-3 per layer), spare 3, cap the
MTP layer's row set. Not pursued: frugal failed the gate (below); the gate-passing policies need overlap, not prediction
(I1 deferral). The GSM8K-50 PF run aborted at load (same overflow), no accuracy row.

### PRE-REGISTERED (15:58, before the GPU probe): near-tie re-routing b25 by the gate-energy law (`reroute_energy.py`,
data/gmass/reroute_energy.txt; 125B traces, LRU replay; law fitted per kind of text on the measured mass-rule probes:
hard KLD = 0.0045 + 29.9 E^1.5 (exact/r2/m85, residuals <= 0.0004), easy = -0.0003 + 6.49 E^1.5 (r2/m50/m70/m85, <= 0.0014))
A swap's error = displaced w^2 + added w^2 (orthogonal experts, norms alike). Swapped mass at delta 0.25 is 5.3 % (hard) /
7.5 % (easy) of the top-10 mass per layer; NewIdeas' "0.7 %" is, I believe, the NET mass change (the added expert's
weight nearly replaces the displaced one's), but in output space the two errors add.
| C | delta | hard reads -> predicted KLD | easy reads -> predicted KLD |
|---|---|---|---|
| 51 | 0 | 131.3 -> 0.0045 | 171.6 -> 0.000 |
| 51 | **0.25** | **95.0 -> 0.019** | **118.7 -> 0.0055 (gsm ~0.0065 > 0.0045: FAIL predicted)** |
| 51 | 0.5 | 75.8 -> 0.036 | 95.6 -> 0.011 |
| 77 | 0.25 | 68.4 -> 0.017 | 83.1 -> 0.0051 |
| 97 | 0.25 | 53.3 -> 0.015 | 65.3 -> 0.0046 |
Per saved read b25 costs ~2.2x the gate energy of wait-by-mass (hard text at ~95 reads: m85 E 0.0028, measured KLD 0.0094;
b25 E 0.0061, predicted 0.019): the law says re-routing is DOMINATED by the mass rule. The d1_w10_b25 probe (first in
the queue) tests the law on swaps: near 0.019 / 0.0055 -> re-routing is dead under the gate; near m85's -> swaps cost
less than drops (norm kept) and the law needs a swap term.

### PRE-REGISTERED (16:05): deferral arms dh_wK by the same law (data/gmass/wk_energy.txt; C=51; drop = wait ranks < K,
use resident ranks >= K, drop the rest, no renorm; dh_wK ~ exact + f x drop excess, f = 0.19-0.23 on the 35B half-layer)
| K | waited reads hard / easy | drop KLD hard / easy | dh_wK at f 0.2: hard (ifeval ~1.17x) / easy (gsm ~1.2x, code ~0.45x) |
|---|---|---|---|
| 2 | 18.3 / 20.9 | 0.100 / 0.027 (measured noren 0.100 / 0.027) | 0.024 (ifeval ~0.028 FAIL) / 0.0051 (gsm ~0.006 FAIL) |
| 3 | 29.1 / 34.6 | 0.063 / 0.018 | 0.016 (ifeval ~0.019, marginal) / 0.0033 (gsm ~0.004, marginal) |
| **4** | **41.0 / 50.0** | 0.042 / 0.012 | **0.012 (ifeval ~0.014) / 0.0021 (gsm ~0.0025): PASS predicted** |
| 5 | 54.1 / 67.0 | 0.029 / 0.0078 | 0.0094 / 0.0013 |
| 6 | 67.8 / 85.4 | 0.020 / 0.0049 | 0.0076 / 0.0007 |
At equal waited reads no drop policy comes close (m50: ~47-59 reads, easy 0.0114 FAIL). Speed if the deferred reads hide:
dh_w4 eager ~28.4 + 0.375 x ~45 = ~45 ms (~22 tok/s at 10 %, vs m85's 14.2), SSD total = exact's 360-475 MB/token
(25-33 ms at 14.6 GB/s: near saturation, fits inside 45 ms). dh_w2 (queued) measures f on the 125B; dh_w4 follows in the
same turn unless b25 passes gsm+code (then b25's ifeval+multi runs first).

### PRE-REGISTERED (17:00): speed of the half-layer-deferral prototype (fork125bd + bt_defer.py, 10 %, eager, M5 Pro)
Model: compute 28.4 ms/token; exact streaming = 12.9 tok/s at ~150-170 reads/token (~0.33 ms per read on the critical
path). fork125 reads each expert as 9 parallel piece reads (no contiguous records: BigMoE replaced records/ by records3).
If late reads cost the same per read and overlap compute: max(28.4 + ~50 waited x 0.33, ~170 x 0.33) ~ 56 ms -> ~18 tok/s;
if each layer's late batch reads in parallel (0.34 ms + 0.19 ms per extra): ~47 ms SSD-bound -> ~21 tok/s.
Predictions: K10 (exact) ~12.9 | K4 same-layer ~15-17 | **K4 half-layer 17-22** | K2 half-layer 22-26 (code; prose similar).
Below ~17 for K4 dh = the loader serialises reads (then the next lever is one read per expert: records3 on BigMoE's tier
engine, IDEAS D1e). Equal-quality comparison: m85 eager 14.2 / 13.7 tok/s.

### 5 % with half-layer-late ranks (NewIdeas' probe, tf_defer125.py dh_w2_m50 at 0.05, ifeval+multi every 8th, 17:30)
KLD **0.1843** (ifeval 0.1820 top-1 0.829, multi 0.1870 top-1 0.842); per token 146.5 experts now, 33.4 late (gate mass
0.077 per layer), **300.1 dropped**, 62.4 reads; 284 s. tf_defer125's _m50 uses ONLY the ranks before 50 % of the top-10
mass, even resident ones (~6 of 10 ranks per layer dropped), unlike frugal (resident ranks 2-9 + substitutes 10-19), so
this measures the mass cut more than the deferral. The frugal-faithful version is tf_frugal_var.py `base_l2t4` at 0.05
(frugal + cold ranks 2-3 half a layer late), queued with its paired baseline `base` at 0.05 (diet turn 2).
Update 17:45 (NewIdeas' 35B frontier, valid rerun, ~/NewIdeas/IDEAS.md "Batch 4"): the swap prediction HOLDS on the 35B:
b25 exact streaming 92.7 reads at KLD 0.030 vs m85 97.9 at 0.028 (re-routing dominated by the mass rule; the 125B b25
probe was cancelled after Kamen's 17:10 bar change, so the 35B is the test). Also there: frugal with a cold rank 1 half a
layer late KLD 0.168 vs frugal 0.163 (+3 %) with waited reads 21.8 -> 9.5 per token (5 %: +0.005, 31.8 -> 14.2); dropping
rank 1 instead 0.290. 5 % margin candidates under frugal-10 %'s 0.163: w3 (51 reads, 0.125), cold-in-50 %-mass late (0.138).

### Frugal speed levers at 10 % (diet turn 1, 18:07-18:11, M5 Pro, quiet lock; data/day/pf*_*, df_*)
| run | eager tok/s (waited misses/token) | + MTP 3:3:0 tok/s (tok/pass, waited/token) |
|---|---|---|
| frugal, bt_frugal_spec (15:14 reference) | code 26.2 (25.5), prose 25.7 (26.9) | code 39.2 (3.20, 33.0), prose 28.6 (2.33, 42.7) |
| + precise prefetch, corrected d=1 predictor, 2/layer, 4 spare | code 22.7 (22.6), prose 21.7 (27.7) | code 37.6 (3.20, 31.5), prose 23.9 (2.17, 51.6) |
| + prefetch corrected, 4/layer, 6 spare | code 21.7 (22.8), prose 20.7 (29.3) | code 29.7 (3.20, 32.4), prose 21.0 (2.33, 51.1) |
| + prefetch uncorrected, 2/layer, 4 spare | code 21.8 (23.5) | code 28.1 (2.72, 42.5) |
| frugal on bt_defer (fork125bd flag mode, same semantics) | code 23.8 (27.1), prose 22.6 (31.2) | - |
| **frugal + cold rank 1 half a layer late** (waits for rank 0 only) | **code 25.4 (10.9 + 13.9 late), prose 24.6 (12.8 + 15.2 late)** | - |
Precise prefetch LOSES in every form at equal RAM (spare slots out of C; waited misses fall at most ~3/token: the d=1
predictor rarely names this layer's ranks 0-1 in time). Late rank 1: +6.7 % code / +8.8 % prose within the same harness,
waited misses -60 %; ~45-50 % of each moved read's latency hidden (2.6-3.5 ms saved for 16-18 moved reads). bt_defer's
harness is ~9 % slower than bt_frugal_spec's eager path, so the lever must move into the fast block (and the MTP verify
block: multi-row phase B) to pay. Checks: K=10 = exact streaming (139 reads/token, coherent); frugal +- late rank 1 coherent.
Peak 13.3 GB (code) / 12.2 GB (prose) eager at 10 %: the 12 GB fit needs the diet + MSB-3 pools (IDEAS D1g).

### Late rank 1 quality + RAM floors (diet turn 1, 18:11-18:27)
TF KLD at 10 %, every 8th, paired vs frugal (`base`, tf_frugal_var.py; `l1t2` = cold rank 1 half a layer late):
| subset | base | base_l1t2 | paired diff (95 % CI) | top-1 |
|---|---|---|---|---|
| ifeval+multi | 0.1009 (0.1155 / 0.0833) | 0.1065 (0.1248 / 0.0844) | **+0.0056 [+0.0018, +0.0093]** (+5.5 %) | 0.886 -> 0.878 |
| gsm+code | 0.0349 (0.0423 / 0.0131) | 0.0345 (0.0425 / 0.0108) | -0.0004 [-0.0038, +0.0031] (none) | 0.968 -> 0.971 |
Late rank 1 costs a small but significant +5.5 % on hard text (35B: +3 %) and nothing on easy text: below frugal's own
loss (0.10), but under the task-result bar it wants an IFEval decode row before it becomes the default.
RAM floor at C=16 (scripts/floor_diet.py; floor = MLX active minus expert pools; RSS misses Metal buffers):
| step | floor (MLX active) | vs base |
|---|---|---|
| base (n-gram hot cache 1024 MB) | 3.89 GB | - |
| hot cache 0 / 256 MB | 3.89 GB | 0 (file-backed pages, not in the floor) |
| HC mixers 8-bit (1337 -> 710 MB) | 3.27 GB | -0.62 |
| HC mixers 4-bit (1337 -> 376 MB) | 2.94 GB | -0.95 |
| hc4 + lm_head 3-bit + hot 0 | (4.13: artifact, the dequantised bf16 lm_head stays referenced) | expected ~-1.0 |
Budget at 10 % (peak = pools + floor + ~2.3 GB temporaries; measured 13.3 GB eager code): 4-bit pools 6.77 + 3.89 + 2.3
= 13.0; with HC 4-bit ~12.0; with v3's MSB-3 pools (5.26 + sidecar 0.74) + HC 4-bit ~11.3 GB. HC 4-bit KLD: turn 2.

### Pre-routed frugal (TrainLab's 35B lever) FAILS on the 125B (diet turn 2, 19:27-19:50; tf_frugal_var `pre2`)
Waited pair of layers 8-39 = layer L's router + corr125 on layer L-1's MoE input (top-2); rest = the true router's best
resident; TRUE weights renormalised; paired vs frugal `base` on the same positions (every 8th):
| residency | subset | base | base_pre2 | diff (95 % CI) | reads/step |
|---|---|---|---|---|---|
| 10 % | ifeval+multi | 0.1009 | 0.1273 | **+0.0264 [+0.0189, +0.0344] (+26 %)** | 20.9 -> 21.6 |
| 10 % | gsm+code | 0.0349 | 0.0389 | +0.0040 [-0.0010, +0.0087] | 28.1 -> 28.5 |
| 5 % | ifeval+multi | 0.1740 | 0.2126 | **+0.0387 [+0.0282, +0.0501] (+22 %)** | 31.8 -> 32.7 |
| 5 % | gsm+code | 0.0507 | 0.0597 | **+0.0090 [+0.0013, +0.0171] (+18 %)** | - |
Pre-pick precision (all four runs alike): true rank 0 among the two picks 69-79 % by 8-layer band, rank 1 46-52 %, top-2
overall 60-62 %, top-10 93-95 %: a quarter of the time the highest-weight expert is not waited for and, if cold, dropped.
No bytes saved either. The 35B's gain (0.141 vs 0.168) does not transfer; narrowing the band will not help (all bands
alike). GSM8K-50 with B12_PRE skipped by its guard. Also measured: frugal at 5 % on these subsets 0.1740 / 0.0507.
Plain B12 (frugal on v3, HC bf16, no late) checked fine: coherent, 6.00 GB pools (C_m 51, C_l 20), 9.90 GB active;
B12 with HC 4-bit failed to build Metal kernels (BigMoE's compiled HC read expects bf16 mixers): fixed, rerun in turn 3.

### B12 measured (diet turn 3, 21:54-22:01; frugal 10 % on the v3 copy, C_m 51 MSB-3 + C_l 20 sidecar, eager, M5 Pro)
| config | code tok/s (reads/token) | prose tok/s (reads/token) | peak GB code / prose |
|---|---|---|---|
| B12 full: HC 4-bit + late rank 1 | 25.3 (26.5 waited + 33.2 late) | 23.9 (32.4 + 37.4) | 10.4 / 10.2 |
| B12, HC 4-bit, no late | **26.0** (52.4) | 23.8 (64.8) | 10.4 / 10.2 |
| B12 plain: HC bf16 (compiled HC), no late | 23.8 (58.1) | 22.7 (66.3) | **11.3 / 10.6** |
| ref: frugal on fork125bf (4-bit pools) | 26.2 (25.5 reads, one per expert) | 25.7 | 13.3 / 12.2 |
GSM8K-20 (B12 full): 19/20, answers equal to exact eager 19/20, 25.0 tok/s. v3 reads count both record parts of a cold
4-bit expert (MSB + LSB). Late rank 1 gives NOTHING on v3 (two phase-B spins + a one-expert 3c call per layer eat the
overlap): drop it from B12. **Plain B12 already fits 12 GB (11.3 peak) without quantised mixers.**
Teacher-forced quality of the levers (every 8th, paired):
| change | hard (ifeval+multi) | easy (gsm+code) |
|---|---|---|
| HC mixers 4-bit, exact routing (exact_hc4 vs exact) | **+0.0243 [+0.0223, +0.0262]**, top-1 0.980 -> 0.939 | +0.0066 [+0.0056, +0.0076], 0.996 -> 0.987 |
| 5 %: frugal (base_r05) | 0.1740 (31.8 reads/step) | 0.0507 |
| 5 %: + cold ranks 2-3 half a layer late (base_l2t4_r05) | **0.0755: -0.0985 [-0.1081, -0.0893]** (67.4 reads, 37.3 late) | - |
| 5 %: wait for ranks 0-2 (w3_r05) | 0.1073: -0.0667 [-0.0758, -0.0581] (49.1 reads) | - |
HC 4-bit costs a quarter of frugal's own KLD on hard text: not free. 5 % + late ranks 2-3 beats frugal at 10 % (0.1009)
on hard text at half the cache: the 6-8 GB candidate, if its late reads hide.

### B8 and B12 (plain) measured (day_b8.sh, 23:14-23:28, M5 Pro, quiet lock; v3 copy, eager, 1-row decode)
| config | code tok/s (reads/token) | prose tok/s (reads/token) | peak GB code / prose | GSM8K-20 |
|---|---|---|---|---|
| **B8** = frugal 5 % (C_m 26, C_l 10; 3.05 GB pools) + cold MSB-3 ranks 2-3 half a layer late | 16.6 (88.2 waited + 44.0 late) | 16.1 (94.5 + 43.2) | **8.4 / 7.7** | **19/20**, answers = exact eager 19/20, 16.0 tok/s |
| B8 without late (frugal 5 % on v3) | 22.8 (70.0) | 20.7 (90.5) | 8.4 / 7.7 | - |
| **B12** plain = frugal 10 % (C_m 51, C_l 20; 6.00 GB) | 23.8 (58.1) | 22.7 (66.3) | 11.3 / 10.6 | **19/20**, = exact eager 19/20, 22.9 tok/s |
B8's late reads do NOT hide (60 ms vs 44 ms per token on code): 44 late reads/token at ~0.26 ms each, and the late loads
churn the 26-slot cache (waited misses 70 -> 88). B8 trades -22..27 % speed for hard-text KLD 0.0755 (frugal 5 % 0.1740,
frugal 10 % 0.1009) at 7.7-8.4 GB. Cheaper 5 % points to try next: late rank 2 only (l2t3), w3 (wait ranks 0-2; KLD 0.1073).

### B8-fast, B8's easy text, HC 8-bit (day_b8.sh, 23:53-00:13; batch paused 23:31-23:50 by the orchestrator, no timing row spans it)
| config (v3 copy, 5 %, eager) | code tok/s (reads/token) | prose tok/s | peak GB | GSM8K-20 | KLD hard / easy (every 8th) |
|---|---|---|---|---|---|
| B8 (cold ranks 2-3 half a layer late) | 16.6 (88.2 + 44.0 late) | 16.1 | 8.4 / 7.7 | 19/20 | **0.0755 / 0.0198** |
| **B8-fast** (B12_W=3: wait ranks 0-2) | **19.8** (93.9) | **18.3** (111.0) | 8.4 / 7.7 | **19/20** (= exact 19/20), 18.3 tok/s | 0.1073 / - |
| frugal 5 % (no late) | 22.8 (70.0) | 20.7 (90.5) | 8.4 / 7.7 | - | 0.1740 / 0.0507 |
| ref: frugal 10 % (fork125bt probe) | - | - | - | - | 0.1009 / 0.0349 |
B8 vs frugal 5 % on easy text: -0.0309 [-0.0379, -0.0243] (-61 %); B8 beats frugal 10 % on both kinds of text at 5 %.
HC mixers 8-bit (exact routing, paired): hard +0.0000 [-0.0003, +0.0003], easy +0.0001 [-0.0001, +0.0002], top-1
0.980 -> 0.979: FREE (-0.62 GB floor). HC 4-bit: +0.0243 / +0.0066. Speed of HC8 (eager HC path, no compiled HC) not
yet measured: B12 + HC8 and B8 + HC8 are the next timing rows.

### HC 8-bit speed + MTP on the v3 copy (day_hc8.sh, 00:55-00:58, M5 Pro, quiet lock; ext rebuilt: protect-what-fits)
| config (eager) | HC8 code / prose tok/s | HC bf16 (compiled HC) code / prose | peak GB HC8 | peak GB bf16 |
|---|---|---|---|---|
| B12 (10 %) | **24.2 / 22.2** | 23.8 / 22.7 | **10.7 / 10.4** | 11.3 / 10.6 |
| B8 (5 % + late ranks 2-3) | **18.9 / 16.0** | 16.6 / 16.1 | **7.7 / 7.5** | 8.4 / 7.7 |
| B8-fast (5 %, wait ranks 0-2) | **20.1 / 18.3** | 19.8 / 18.3 | **7.7 / 7.5** | 8.4 / 7.7 |
HC 8-bit: not slower, -0.6..0.7 GB, KLD free -> default B12_HC=8 from 01:00 (e0ab2d2); the Eval suite was told.
MTP 3:3:0 on the v3 copy (bf16 mixers; verify rows <= 4 on the multi-row frugal block):
| config | code tok/s (tok/pass, reads/token) | prose tok/s (tok/pass) | peak GB |
|---|---|---|---|
| B12 + MTP | **36.2** (3.37, 77.5) | **29.5** (2.67, 89.3) | 13.3 / 12.0 |
| B8-fast + MTP | 27.5 (2.91, 117.3) | - | 10.4 |
MTP: +52 % code / +30 % prose on B12, +39 % on B8-fast code. But MTP adds ~2 GB of peak (MTP head + its 64-slot expert pool
+ verify buffers): B12 + MTP is 13.3 GB on code; with HC8 ~12.7. To fit 12 GB: MTP_C 32, HC8, check FRSPEC's draft head.

### 12 GB trim of B12 + MTP (day_mtp12.sh, 01:13-01:15, HC8 default, MTP 3:3:0, quiet lock)
| config | code tok/s (tok/pass) | prose tok/s (tok/pass) | peak ACTIVE GB | max RSS GB |
|---|---|---|---|---|
| B12 + MTP, MTP_C 64 | **38.6** (3.12) | 26.4 (2.21) | 12.7 / 11.4 | 9.5 / 9.4 |
| B12 + MTP, MTP_C 32 | **38.6** (3.12) | 26.4 (2.21) | 12.6 / 11.3 | 9.4 / 9.3 |
| B12 + MTP_C 32, FRSPEC off | 37.2 (3.12) | - | 12.4 | 9.4 |
| B12 + MTP_C 32, MLX cache capped at 512 MB | 38.4 | - | 12.6 (cache 0.6) | 9.4 |
| B8-fast + MTP | **32.6** (3.37) | 18.1 (2.00) | 9.7 / 8.5 | 7.7 / 7.1 |
MTP_C 32 = 64 in speed (keep 32); FRSPEC +4 % (keep); the cache was only 0.3-0.6 GB. B12 + MTP = 38.6 tok/s on code (vs
the exact 24 GB engine's ~36). MEASUREMENT CAVEAT: MLX's peak ACTIVE (12.6) exceeds the process max RSS (9.4); RSS may
miss GPU-only allocations, peak active counts transient buffers. bt_b12 now also reports the lifetime-max physical
footprint (proc_pid_rusage v4, Activity Monitor's Memory); that number decides the 12 GB claim (next short run).

### TRUE peak footprint of the shipping configs (day_fp.sh, 01:24-01:27; proc_pid_rusage v4 lifetime-max phys footprint)
| config (HC8, code prompt incl. prefill) | MLX peak active | process max RSS | **peak physical footprint** | tok/s |
|---|---|---|---|---|
| B12 eager | 10.7 | 9.2 | **11.6 GB** | 24.2 |
| B12 + MTP 3:3:0 (MTP_C 32, FRSPEC) | 12.6 | 9.4 | **13.1 GB** | 38.6 |
| B8 eager | 7.7 | 7.4 | **8.7 GB** | 18.9 |
| B8-fast + MTP | 9.6 | 7.6 | **10.2 GB** | 32.6 |
Max RSS UNDERSTATES by 2.4-3.7 GB (Metal buffers); the footprint is peak active + ~0.6 GB cache + ~0.3 GB runtime.
Every "peak" in earlier sections is MLX peak active: read them as ~0.5-1.0 GB low. B12 eager fits 12 GB (0.4 margin);
B12 + MTP does not yet (13.1): next trim = MLX cache cap 64 MB, smaller prefill chunks, sidecar C_l 20 -> 12.

### PRE-REGISTERED by NewIdeas (02:10 Thu, model calibrated to the v3 MTP engine): prefetch A/B, B12 + MTP code
Baseline (model) 37.2 tok/s at 3 tok/pass (measured 38.6 at 3.12). Arm 1 (C=47 + 4 spares, corrected k=2, issued 'after'
the demand reads): **42.3 tok/s, +14 %**, waited reads per pass 61 -> 25, total reads 61 -> 107. Issued 'now' (PF_JIT):
38.0 (+2 %). Plain d=1 'after': 40.7 (+9 %). Global ring at C=51: 42.6 (+15 %). Below +5 % = the model is wrong somewhere;
the counters (waited per pass, prefetch issued / used / on time / late per pass, now printed by bt_b12's spec task) say what.
AMENDED 02:20 (before any A/B data; NewIdeas refitted the SSD model: floor 0.06 ms + min(8.1 GB/s per read, 14.0 shared),
fits QD1/2/4 = 6.95/12.7/14.0 GB/s): baseline 38.5 tok/s; arm 1 'after' 42.9 (+11 %); 'now' +2 %; plain d=1 'after' +8 %;
ring at C=51 +12 %; waited reads per pass 61 -> 25 under both models. Pre-registered range for arm 1: **+11..14 %**;
kill line < +5 %.

### Trim 2: B12 + MTP toward 12 GB of peak footprint (day_trim2.sh, 02:16-02:22; HC8, MTP 3:3:0, MTP_C 32)
| B12 config (cumulative) | code tok/s (tok/pass) | PEAK FOOTPRINT |
|---|---|---|
| + MTP, MLX cache cap 64 MB (set before load: ignored, cache stayed 0.6 GB; fixed to after load) | 36.0 (3.12) | 13.2 |
| + prefill chunk 256 | 39.4 (3.46) | 13.0 |
| + sidecar C_l 12 | 38.3 (3.46) | 12.7 |
| + base C_m 46 | 37.0 (3.37) | **12.2** |
| eager, cap + chunk 256 | 23.6 | 11.5 |
| + MTP, prose, cap + chunk 256 | 24.4 (2.21) | 11.9 |
| eager, ~2K-token prompt, chunk 2048 / 512 / 256 | 20.6 / 20.0 / 20.8 | **14.4 / 12.0 / 11.5** |
Long prompts: the exact-prefill scratch drives the peak (14.4 GB at chunk 2048); chunk 512 -> 12.0, 256 -> 11.5. MTP adds
~1.5 GB over eager at equal settings (not yet explained: MTP head, FR-Spec head, n-gram hot cache?). Prefill tok/s rows
were lost (the print sat in the MTP branch; fixed) and rerun in the A/B slot. Per-pass counters count waited misses per
record part (MSB and LSB separately): not the unit of NewIdeas' model.

### Prefetch A/B arm 1 + prefill speed + MTP RAM split + 32K context (day_pf.sh, 03:28-03:49, quiet lock, HC8, MTP_C 32)
Prefetch arm 1 (corrected k=2 per row, both record parts, issued after L's demand reads, 4 spares from C), B12 + MTP:
| run | tok/s | tok/pass | ms per pass | prefetch per pass: issued / used (on time, late) |
|---|---|---|---|---|
| code base (x2) | 37.2, 37.2 | 3.12 | 83.9 | - |
| code arm 1 (x2) | 33.0, 32.9 | 2.78 | **84.2 (+0 %)** | 210.5 / 125.3 (123.7, 1.7) |
| prose base | 25.3 | 2.21 | 87.4 | - |
| prose arm 1 | 25.7 | 2.13 | **82.9 (-5 %)** | 199.3 / 128.3 (126.5, 1.8) |
**Below the pre-registered +11..14 %** (per pass: +0 % code, +5 % prose). End-to-end code tok/s FELL 11 % because the
prefetches changed residency, hence the frugal outputs and the drafts' acceptance (3.12 -> 2.78). Prefetches land on time
(~99 %) and 60 % are used, yet per-pass time does not move: the model is wrong somewhere (candidates: in-flight prefetches
still contend with L+1's demand reads; the 4 spares cost hits; promotions are not free). The waited-miss counter counts
promoted prefetches as misses (not comparable across arms).
Prefill speed (B12 eager, 2617-token prompt; now cache-only chunks, emit_logits=False):
| chunk | prefill tok/s | PEAK FOOTPRINT |
|---|---|---|
| 2048 with logits (old code) | 173 | 13.9 |
| 2048 cache-only | 175 | **13.2** (logits were 0.7 GB) |
| 512 | 81 | 11.8 |
| 256 | 51 | 11.4 |
MTP RAM split (cache cap 64 MB now effective: cache 0.1 GB): B12 + MTP code 40.2 tok/s, 12.9 GB; n-gram hot cache 0: 12.9
(no effect); MTP head loaded, drafting off (spec0:0:0, verify width 1): 22.8 tok/s, **11.5 GB = eager**: the MTP head
costs nothing, the 4-row VERIFY costs ~1.4 GB.
Long context (37.3K-token prompt, chunk 2048, no cache cap): B12 eager decode 25.7 tok/s, prefill 189 tok/s, **PEAK
FOOTPRINT 18.7 GB** (peak active 17.0); B12 + MTP 25.5 tok/s (2.29 tok/pass), 19.1 GB. The prefill peak at long context
(attention over 37K keys per 2048-token chunk + the expert scratch + KV ~0.9 GB) far exceeds 12 GB.

### Where MTP's +1.4 GB lives (day_memck.sh, 04:02-04:03; B12 + MTP code, cache cap 64 MB, prefill chunk 256)
| verify width (spec) | tok/s (tok/pass) | loaded | first token: active / PREFILL PEAK | DECODE-phase peak active | lifetime footprint |
|---|---|---|---|---|---|
| 4 (3:3:0) | 39.0 (3.46) | 9.6 | 10.0 / **12.5** | **10.0** | 12.9 |
| 2 (1:1:0) | 29.5 (1.86) | 9.6 | 9.9 / 12.5 | 10.0 | 12.9 |
| 1 (0:0:0, head loaded, no drafts) | 23.3 (1.00) | 9.6 | 9.9 / 11.0 | 9.9 | 11.5 |
NOT the verify: decode-phase peak ~10.0 GB at every width (+0.1 GB at width 4; consistent with NewIdeas' GDN-snapshot
size but not the cause). The +1.5 GB is spec125's PREFILL with the MTP head on (with_head=True: per-chunk logits and the
captured hidden stream for the head, plus the MTP layer's own prefill). Decode itself sits at ~10.4 GB footprint. Fix =
a lean head prefill (cache-only chunks except the head's history), in my copy since spec125 is SpecDecode's.
TrainLab D1i interim (its copy, ~/TrainLab/d1i): exact grouped tier prefill (BIGMOE_TIER_PF_GROUP=64, read_rebuilt
double-buffered): same greedy sha, prefill 162/172 -> 204/206 tok/s at chunk 2048, footprint 13.2 -> 13.1 GB (the 2048-chunk
peak is attention-side: ~0.7 MB alive per chunk token at MoE entry); next: 512-token attention query chunks with the MoE
over 4096 tokens per layer.

### MTP's +1.4 GB fixed: grouped MTP-layer prefill (day_grp.sh, 04:17-04:20; B12 + MTP 3:3:0, MTP_C 32, cache cap 64 MB)
The extra prefill peak is the streamed MTP layer's plain `_sync_call` scratch (every expert a chunk uses). BigMoE's own
exact grouped path `BIGMOE_PF_GROUP=64` caps it:
| prompt, prefill chunk | PF_GROUP | tok/s (tok/pass) | prefill peak active | PEAK FOOTPRINT | greedy sha |
|---|---|---|---|---|---|
| code, 256 | off | 38.8 (3.46) | 12.5 | 12.9 | c5d2d4baef |
| code, 256 | 64 | 38.9 (3.46) | 11.0 | **11.5** | c5d2d4baef (same) |
| code, 2048 | 64 | 35.6 (3.12) | 11.2 | 11.7 | 262133df5a |
| long (2.6K), 512 | 64 | 23.8 (2.33) | 11.5 | 12.2 | 3ef8ca206f |
| prose, 256 | off | 23.8 (2.21) | 11.4 | 11.9 | a3c92bdf64 |
| prose, 256 | 64 | 24.2 (2.21) | 10.5 | **11.3** | a3c92bdf64 (same) |
Decode itself: ~10.0 GB active, ~10.4 GB footprint at every row. B12 + MTP is now within 12 GB at short context with
no speed or output change.

### D1i adopted from TrainLab (a45b99a; module API 4d42a59): sub-batched exact prefill on B12
TrainLab's measurements (~/TrainLab/d1i/PATCH.md; long prompt 2,617 tokens, cache cap 64 MB, cold / repeat):
| prefill | tok/s | PEAK FOOTPRINT |
|---|---|---|
| chunk 2048, cache-only chunks | 163 / 173 | 13.2 |
| + grouped tier prefill (BIGMOE_TIER_PF_GROUP=64; same sha) | 198-204 / 192-206 | 13.1 |
| chunk 256 (the verdict setting so far) | 50 / 50 | 11.5 |
| **decoupled: attention 512 / MoE once per 4096 tokens, groups 64** | **272 / 288** | **10.9** |
| 32K prompt (37,294 tokens), chunk 2048, groups 64 | 215 | 17.6 |
| **32K, decoupled 512 / 4096** | **270** | **12.5** |
| B12 + MTP, decoupled 256 / 4096 prefilled state + PF_GROUP 64 | 240 | 12.0 |
Not bitwise vs any chunked prefill (the MoE's gather kernel follows the rows per group); outputs fluent. Not valid with
B8's late writes (asserts). Rejected by TrainLab: GPU rebuild of records3 (slower), warming pools from the last chunk
(waited misses 0-16 rose 106 -> 116). API for harnesses: `b12_prefill(model, ids, cache, s_attn=512, s_moe=4096)` and
`b12_prefill_state(model, prompt_ids)` (spec125 `prefilled=`). Smoke test on my tree: day_smoke.sh (queued 04:25).

### Smoke test before the Eval suite's B12 verdict (day_smoke.sh, 04:34-, quiet lock; tree cc27412; cache cap 64 MB)
Settings: decoupled prefill D1I_ATTN=512 D1I_MOE=4096 D1I_SPLIT=1, grouped tier prefill BIGMOE_TIER_PF_GROUP=64; MTP rows
add BIGMOE_PF_GROUP=64, MTP_C 32, the d1i prefilled state (spec125 deep-copies it per call).
| row | result |
|---|---|
| B12 eager, long (2,617 tokens) | prefill 278 tok/s, decode 20.6 tok/s, **10.9 GB**, fluent (sha e0420271a0) |
| Eval suite path: MTPLX generate_ar, chunk 512, greedy (code / prose / IFEval-style) | 19.4 / 18.6 / 19.8 tok/s decode, **10.9 GB**, fluent; the IFEval-style answer keeps lowercase + `***` |
| same + the PF1 flags env.py forces (PF_PREFILL, PF_FULLSCR) | **identical shas** (ad39cb14d4 / d47091989a / 75c049fb42): harmless for v3f's tier layers |
| B12 + MTP 3:3:0, code | 35.3 tok/s, 3.12 tok/pass, **11.2 GB**; sha 262133df5a = the one-chunk prefill's (326 tokens: decoupled = one chunk) |
| GSM8K-20 through MTP 3:3:0 (spec125, d1i prefilled state) | **20/20**, all 20 answers equal to exact eager (eager B12 19/20), 34.7 tok/s |
| B12 + MTP 3:3:0, prose | 24.4 tok/s, 2.21 tok/pass, **11.2 GB** (sha a3c92bdf64, as at chunk 256: 53-token prompt) |
| 32K (37,123 tokens, data/prompts/ctx32k), eager | prefill 271 tok/s, decode 18.1 tok/s, **12.5 GB** (peak active 10.7): 0.5 GB over 12 |
Verdict path (Eval suite, told 04:45): MTPLX generate_ar at MTPLX_PREFILL_CHUNK_SIZE=512 + BIGMOE_TIER_PF_GROUP=64 (= the
decoupled prefill on prompts <= 512 tokens), AR only; MTP covered by the GSM8K-20 row. Chat1000 runs the paired long-prompt
task check (4-16K, decoupled vs chunk 512 vs chunk 2048).

### B8 / B8-fast with the decoupled prefill (TrainLab 04:50, its copy; assert relaxed in mine 04:55 to "no pending late writes")
On one-query-chunk prompts the decoupled prefill (512/4096) equals the plain prefill bit for bit with B8's layer_call
installed: B8 code 35f0847f33 / 6237ab3c44 (cold / repeat), prose b36b13759c / 986e52fde6, B12 control 13a7542443 /
fb7b3a2f33, the same on both paths. Late writes come only from decode rows (the frugal block takes S <= SMAX).
| config | prompt | prefill tok/s (cold / repeat) | PEAK FOOTPRINT | decode tok/s |
|---|---|---|---|---|
| B8-fast | long (2.6K) | 283 / 306 | 7.9 GB | 16.8 / 17.5 |
| B8 | long (2.6K) | 282 / 305 | 8.0 GB | 15.1 / 15.4 |
| B8-fast | 32K | 279 | **9.6 GB** | 16.6 |
| B8 | 32K | 280 | **9.6 GB** | 14.7 |
B8-fast at chunk 256: 58 tok/s prefill at the same 7.9 GB; B8-fast + TrainLab's streaming ring 4: 361 / 389 tok/s at 8.5 GB.
The 5 % line fits 12 GB up to 32K with room; B12 (10 %) is 12.5 GB at 32K.

### RAM per context + TrainLab's streaming on my tree (day_ctx.sh, 04:59-05:08, quiet lock; cache cap 64 MB, decoupled 512/4096)
| row | prefill | decode | PEAK FOOTPRINT | note |
|---|---|---|---|---|
| B12 + MTP, long 2.6K, prefill inside generate (D1I_FRESH, no state copy) | 258 tok/s | 24.6 tok/s (2.51/pass) | **11.3 GB** | the copied-state rows were ~0.7 GB higher; sha dc429bc75b |
| B12 + MTP, 32K (37,123 tokens), no copy | 252 tok/s | 24.5 tok/s (2.91/pass) | **13.6 GB** | [see 06:17 correction: MLX peak counters are reset per prefill MoE call]; decode phase 11.5 GB; sha 51c7a0cd52 |
| B12 eager, 32K, MoE chunk 2048 (vs 4096) | 200 tok/s (vs 271) | 18.1 | 12.5 GB (same) | NOT the MoE transient: peak active 10.7 both; footprint - active grows 1.0 -> 1.8 GB with context |
| B12 eager, long, streaming ring 6 + RB2 (ported) | **435 tok/s** (vs 278) | 20.9 | 11.8 GB | **sha e0420271a0 = non-streamed**: port exact |
| B8-fast eager, long, ring 4 + RB2 | 379 tok/s | 17.6 | 8.5 GB | |
| B12 + MTP, long, ring 4 + RB2, no copy | **341 tok/s** (vs 258) | 24.9 | **11.8 GB** | fits 12 GB: MTP can stream at ring 4 |
Crossover (eager B12, ring 6 vs the union read, same shas): 1,237 tokens 5.67 vs 7.01 s; 1,891 tokens 5.87 vs 8.11 s (2,617:
6.01 vs 9.43). The sweep costs ~5.6 s flat; the union read wins only below ~0.6K tokens (53 tokens 3.2 s, 326 ~5 s, cold).
D1I_STREAM_MIN default 1024 (was 2048 per NewIdeas' estimate).
Next (day_ctx2.sh, queued): MTP head fed per MoE chunk (D1I_HEAD_EACH=1, exact by construction: sha check) and MTPLX's n-gram
hot-row cache off (MTPLX_NGRAM_HOT_MB=0; up to 1 GB of anonymous rows that a long prompt fills) at 32K and 64K.

### Chat1000's paired long-context check of the decoupled prefill: PASSES (05:40; ~/Chat1000/REPORT.md section 8)
B12 = load_b12(0.10), my env (TIER_PF_GROUP 64, SPLIT 1, cache cap 64 MB, ASYNC_EVERY 1, streaming off), b12_generate, one
process per mode, same task order; 14 tasks, 114K prompt tokens (passkey 4K x4 / 8K x2, key-value ~10K x2, variable tracking
8K x2, 3-hop over a ~10K tool log x2, the Eval suite's RULER needle at 16K, one real ~9K agentic turn).
| mode | accuracy (13 scored) | identical to chunk 512 | prefill total | PEAK FOOTPRINT |
|---|---|---|---|---|
| chunk 512 (reference) | 11/13 | - | 1,036 s | 11.4 GB |
| **decoupled 512 / 4096** | **11/13** | **13/14** (one word choice at token 6) | **340 s (3.0x)** | 11.7 GB |
| chunk 2048 (floor pair) | 11/13 | 12/14 | 438 s | 14.1 GB |
Decoupled sits inside the variation between two plain chunkings. ADOPTED as the CLI's default prefill (1bb7fe9; D1I_ATTN=0 =
plain chunks). Not bitwise vs any plain chunking (no chunk size is canonical); the Eval suite's verdict path (MTPLX
generate_ar at chunk 512) is unaffected. Note: my HEAD moved during the run (opt-in code only): pin a copy next time.

### Small-call prefill breakdown + D1k look-ahead reads (day_ctx2.sh, 05:44-05:50; B12 eager, decoupled 512/4096, cache cap 64 MB)
Mechanism (la_advise_test.py, layer 30, 102 random experts): F_NOCACHE reads cold 23 ms (13.0 GB/s), the same set again
23 ms (not cached), after F_RDADVISE on a cached fd 5-9 ms (34-66 GB/s): F_NOCACHE reads DO use staged pages; an advisory
costs ~0.14 ms of its caller (now issued from a thread pool).
| prompt | prefill cold / repeat | t_sync (router waits) | t_moe (in prefill_call) | t_read (grouped reads) | with look-ahead (B12_PF_LA=1) |
|---|---|---|---|---|---|
| prose, 53 tokens | 2.94 / 2.95 s | 0.03 | 2.61 | 0.31 (11 grouped layers; 37 single, untimed) | **2.53 / 2.48 s (-14 %)**, 155 advised/layer |
| 'long' at 450 chars, 177 tokens | 3.83 / 3.68 s | 0.04 | 3.23 | 2.51 | 5.35 / 5.43 s (+40 %), 252 advised/layer |
| code, 326 tokens | 4.23 / 4.25 s | 0.05 | 3.45 | 2.77 | 6.26 / 6.25 s (+48 %), 297 advised/layer |
Shas identical with and without look-ahead in every pair (exact, as built). Cold = repeat: no first-call compile cost.
VERDICT: small-call prefill is SSD-BOUND on the union bytes (t_read ~2.5-2.8 s ~ 13 GB/s x the union; router waits are
negligible), so read-ahead has little to hide and its extra bytes (over-prediction, likely double reads of in-flight records)
cost directly: it helps only at ~50 tokens. D1k look-ahead: OFF (opt-in kept). The exact floor of a small call is its
union bytes / 13 GB/s (~2.5 s at 177 tokens); below that only fewer bytes help (skip both-part residents; non-exact:
frugal-routed prefill of short tool outputs, NewIdeas' B item).

### 32K fixes + GPU rebuild for small calls (day_ctx2.sh, 05:44-05:55; cache cap 64 MB, decoupled 512/4096)
| row | prefill | decode | PEAK FOOTPRINT | note |
|---|---|---|---|---|
| B12 + MTP, long, MTP head fed per MoE chunk (D1I_HEAD_EACH) | 259 tok/s | 24.6 | 11.4 GB | **sha dc429bc75b = without**: exact |
| B12 + MTP, 32K, head per chunk | 252 tok/s | 24.5 | **12.9 GB** (was 13.6) | **sha 51c7a0cd52 = without** (footprint: the honest number; the active peaks are not comparable, see 06:17) |
| B12 eager, 32K, n-gram hot cache off (MTPLX_NGRAM_HOT_MB=0) | 276 tok/s | 19.1 | 12.6 GB (12.5 on) | NOT the transient |
| B12 + MTP, 32K, head per chunk + hot cache off | 253 tok/s | 25.6 | 13.1 GB | no gain |
| prose 53 tokens, GPU rebuild path (old strided rebuild) | 2.94 s | | 11.0 GB | = CPU path (37 of 48 layers take the single path) |
| code 326 tokens, GPU rebuild path (old) | 5.37 s (CPU 4.23) | | 11.0 GB | same sha, slower |
HEAD_EACH is the default now (1642995). The 32K footprint exceeds MLX's peak active by ~1.8 GB only DURING prefill (after
decode both are 11.5 GB): a transient outside MLX's accounting, not the hot cache: footprint sampler queued (ctx3).

### CORRECTION 06:17 + small-call RB2 + where the 32K peak sits (day_ctx3.sh, 06:13-)
CORRECTION: TrainLab's prefill memory instrumentation (tier._mem_in / _mem_out, adopted 04:25) calls mx.reset_peak_memory()
at every prefill MoE call, so since then the "peak active" in my lines and the memck "prefill peak" cover only the stretch
after the last reset (in practice the decode phase / the last layer). The PEAK FOOTPRINT numbers (proc_pid_rusage lifetime
max) are unaffected and remain the RAM figures; prefill peaks are PF["peak_pre_gb"] (attention part) / ["peak_moe_gb"].
| row | prefill | PEAK FOOTPRINT | note |
|---|---|---|---|
| 177 tokens, raw preads + RB2 GPU rebuild (BIGMOE_TIER_PF_GPU=1 D1I_RB2=1) | 4.39 / 4.25 s (CPU path 3.83 / 3.68) | 11.0 | same shas; t_read 2.83 s > the C++ read_rebuilt's 2.51 (reads + rebuild) |
| 326 tokens, same | 5.02 / 4.95 s (CPU 4.23 / 4.25; old GPU 5.37) | 11.0 | same shas: the C++ path stays the fastest exact one |
| 32K eager, footprint sampler (50 ms) | 280 tok/s | 12.5 | at the footprint max (12.45) MLX active is 12.17, last MoE chunk, layer 43 / 47 attention (full-attention layers at max context) |
Per-phase prefill peaks at 32K (PF): attention 512 / MoE 4096: attention 12.23, MoE 11.89, active at MoE entry 10.9;
MoE 2048: 12.14 / 11.47 (10.79); TrainLab attention 256 / MoE 4096 (split 1): 11.58 / 11.88 -> the MoE binds, footprint 12.6.
So 256 / 2048 should lower both phases: row ctx32k_a256_m2048 appended.
| row | prefill | decode | PEAK FOOTPRINT | per-phase prefill peaks (active at MoE entry / attention / MoE) |
|---|---|---|---|---|
| B12 + MTP, 64K (65,730 tokens), no copy, head per chunk, hot cache off | 220 tok/s | 21.8 tok/s (2.78/pass) | **14.6 GB** | (spec task: not printed); decode phase 12.7 active |
| B12 eager, 64K, hot cache off | 240 tok/s | 17.3 | **14.1 GB** | 11.74 / **13.86** / 12.62 |
| B12 eager, 32K, attention 256 / MoE 2048 | 205 tok/s (512/4096: ~275) | 18.0 | **12.1 GB** (was 12.5) | 10.77 / 11.48 / 11.46 |
The attention transient grows with the keys (~1.3 GB at 37K, ~2.1 GB at 66K with 512-query chunks: about 512 x keys x 16
heads x 4 B of fp32 scores) and halves with the query chunk; the MoE transient follows the MoE chunk (~1.0 GB at 4096 rows,
~0.7 at 2048). RAM line now (peak physical footprint, cache cap 64 MB): eager 10.9 GB at 2.6K, 12.1 at 32K (long-context
chunks 256 / 2048), 14.1 at 64K; + MTP 11.3 / 12.9 / 14.6. At 64K the resident state alone (KV ~1.6 GB) is ~11.7 GB, so
64K under 12 GB needs a smaller KV (8-bit) or fewer pool slots at long context.

### B12 + MTP residency sweep (TrainLab 06:40, its copy; D1I_FRESH no-copy prefill, PF_GROUP 64, spec 3:3:0, 128 tokens)
| C_m (C_l = 0.4 x) | 'long' decode, PEAK FOOTPRINT | code decode, PEAK FOOTPRINT | hard-text KLD (tf_ifeval + tf_multi, every 8th) |
|---|---|---|---|
| 51 | 25.2 tok/s, 11.4 GB (mine: 24.6, 11.3) | 36.2 tok/s, 11.2 GB | 0.1030 |
| **56** | **26.9 tok/s, 12.0 GB** | **41.4 tok/s, 11.8 GB** | **0.0980** |
| 61 | 26.8 tok/s, 12.6 GB | 41.9 tok/s, 12.4 GB | |
| 66 | 26.5 tok/s, 13.2 GB | 39.7 tok/s, 13.0 GB | |
C_m 56: +7-14 % decode, -5 % KLD (reads per step 20.9 -> 19.4) for +0.6 GB: at the 12 GB line already at 2.6K with MTP.
My call: the verdict stays at C_m 51 (more residents only bring frugal closer to exact, so 51 is the conservative quality
case and keeps 12 GB to ~16-20K context with MTP); C_m 56 = the short-context speed setting (B12_CM=56).

### Skip-residents sizing + C_m 56 on my tree (day_res.sh, 06:43-06:47; decoupled prefill default, cache cap 64 MB)
Residency of a prefill's expert union (BIGMOE_TIER_PF_RESCOUNT=1): GSM8K-10 (10 prompts, each after the previous decode,
480 MoE calls, ~193 experts per layer per prompt): MSB-3 part resident 14.3 %, both parts 6.2 %; 177 tokens re-prefilled
after decoding it: 17.2 % / 6.9 % (cold: 0). Skip-residents would save <= ~12 % of a small call's bytes (6 % full skips +
8 % x 0.74 LSB-only reads), ~0.4 s of ~3.5 s: NOT built. GSM8K-10 at the decoupled default: 10/10, all = exact eager.
| B12 + MTP, no-copy server path | decode | tok/pass | passes/s | PEAK FOOTPRINT |
|---|---|---|---|---|
| C_m 51, long 2.6K (ctx_mtp_long) | 24.6 tok/s | 2.51 | 9.80 | 11.3 GB |
| **C_m 56, long 2.6K** | 25.7 tok/s | 2.56 | 10.04 (+2.4 %) | **11.9 GB** |
| C_m 51, code (smoke, copied state) | 35.3 tok/s | 3.12 | 11.31 | 11.2 GB |
| **C_m 56, code** | 34.2 tok/s | 2.98 | 11.48 (+1.5 %) | **11.8 GB** |
C_m 56 fits 12 GB at short context on my tree; per pass ~2 % faster here (TrainLab: +7-14 % tok/s incl. acceptance), KLD
-5 % (TrainLab). Default stays C_m 51 (context headroom); "B12-11" = load_b12(0.11) is the short-context option.

### B12_NORM gate + dh_w4 on v3 tiers for the MAIN TRACK (day_frpf.sh, 07:42-07:56; tf_b12.py = the Eval suite's scorer on v3f)
GATE: B12 with B12_NORM=mass (decode weights at the top-10 mass): GSM8K-20 19/20, 19 equal to exact eager (= renormalised
B12), 23.9 tok/s -> the Eval suite hook updated 07:47 (B12 / B8 / B8-fast with mass weights; they had not started).
Teacher-forced on the REAL v3f decoder (load_b12(0.25): C_m 128, C_l 51, plan 4433333333, HC bf16), 4 sets every 8th of
the combined list (18 seqs, 3,216 positions), same sequences both arms; targets 1.10 x ar4 (MoonShots):
| arm | gsm (0.00449) | code (0.00253) | ifeval (0.0204) | multi (0.0206) | all |
|---|---|---|---|---|---|
| v3 control (B12_W=10: all top-10 waited) | 0.00298 (top-1 0.992) | 0.00129 (0.998) | 0.01763 (0.944) | 0.00989 (0.974) | 0.0056, PASS x4 |
| **dh_w4** (B12_W=4 B12_LATE3=4,10) | 0.00334 (0.992) | 0.00101 (0.998) | 0.01807 (0.938) | 0.01108 (0.971) | **0.0060, PASS x4** |
Paired dh_w4 - control (bootstrap by position; optimistic): gsm +0.00036 [-0.00012, +0.00086]; code -0.00028 [-0.00063,
+0.00005]; ifeval +0.00045 [-0.00149, +0.00213] (dh_w4 at 89 % of target, 95 % upper 0.02071 just over 0.0204: the tight
part); multi +0.00119 [+0.00035, +0.00213] (54 % of target). Code is not the risk. BIGMOE_DEFER=6 works (6 late pairs).
Sent to MoonShots + NewIdeas 07:58. Per-position npz: data/tfb12/{v3ctl,dhw4}_c128_e8.npz.

### B12_NORM confirmation on the REAL v3f path (tf_b12.py, 10:28-10:32; C_m 51, C_l 20, HC8, plan 4433333333, ifeval + multi every 8)
| B12 decode weights | hard KLD | tf_ifeval | tf_multi |
|---|---|---|---|
| renormalised over the used (w2f) | 0.1230 | 0.1296 (top-1 0.861) | 0.1152 (top-1 0.890) |
| **top-10 mass (B12_NORM=mass, judged by the Eval suite)** | **0.1117 (-9 %)** | 0.1139 (0.861) | 0.1091 (0.900) |
Confirms the fork125bt probe (-10 %) on the real decoder. Note: on the tier path B12's hard KLD is ~0.02 above the
fork125bt (exact 4-bit) numbers (0.1009 there): the 3-bit tier loss, which my earlier KLD rows did not include.

### Repetition guard on the B12 path (orchestrator 09:04: B12 verdict interim failed the truncation rule 13 vs 11)
spec125 imports BigMoE's guard as `from bigmoe import repguard`; ~/BigMoE is not importable on the B12 path, so the guard
was silently OFF in the Eval suite's B12 runs and in mine. Fix 0418c6c: load_b12 registers v3f's byte-identical copy of
bigmoe/repguard.py (MTPLX LoopGuard; arms when a 12-gram recurs 4x in the last 2,048 tokens past 256) under that name.
Checks (10:32-10:36, the Eval suite's B12 + MTP setup with mass weights, one row per fresh process, guard off vs on):
| row | guard off | guard on |
|---|---|---|
| synthetic loop ("write this sentence 100 times") | loops to the 1,024 cap | **identical for 257 tokens, then armed -> EOS at 258** |
| ifeval/1248 (looped to the cap in the suite's run) cold | 470 tokens EOS | identical (never armed) |
| ifeval/2304 cold | 587 tokens EOS | identical (never armed) |
| math500 intermediate_algebra/1422 | (suite: 1,536 capped) | armed once, 61 steered steps, still 1,536 |
Frugal outputs depend on pool history: the suite's loops do not reproduce in cold replays even without the guard.
Note for the Eval suite: FOR_EVALSUITE.md (its session ended; orchestrator relays). Eager paths (CLI generate,
b12_generate) still have no guard: follow-up.

### EVAL SUITE VERDICT: B12 = FAIL, 3/5 rules (report_b12.txt 14:03; 210 core rows; mass weights, repetition guard, MTP)
| test | B12 | ref | floor | B12 vs ref [90 % CI] | rule |
|---|---|---|---|---|---|
| math (n=50) | 82.0 % | 86.0 % | 88.0 % | -4.0 [-10.6, +2.6], McNemar p 0.62 (3/1) | - |
| ifeval (n=100) | 85.0 % | 90.0 % | 88.0 % | -5.0 [-10.4, +0.4], p 0.23 (8/3) | **FAIL** |
| humaneval+ (n=60) | 90.0 % | 91.7 % | 95.0 % | -1.7 [-6.4, +3.1] | PASS |
| decode length ratio | x1.005 | | | | PASS |
| hesitation words / 1K | 0.44 | 0.27 | | | PASS |
| truncations / loops | 18 / 2 | 13 / 2 | | (loops: ifeval/1248, ifeval/374) | **FAIL** |
Answer agreement with ref 91.0 % (floor 97.1 %); token-identical 24/210 (floor 59/210). B8 (5 % + late ranks 2-3) has ~25 %
lower hard KLD than frugal 10 % on the same set (0.0755 vs 0.1009, fork125bt): its verdict is next (suggested to run its
core sets right after B12's extra sets). Quality lever left at 10 %: compute cold ranks late (B12_LATE3=2,4 or 2,10).

### The repetition guard corrupts long math (day_rg2.sh, 15:39-15:43; the Eval suite's 14:28 question)
math500/test/prealgebra/1640, the suite's exact B12 + MTP setup, 4 cold processes:
| run | tokens | finish | guard |
|---|---|---|---|
| guard off (a), (b) | 954, 954 (token-identical) | EOS | never armed |
| guard on (a), (b) | 1,536, 1,536 (token-identical, = the suite's saved guard row over all 1,536) | length cap | armed once, 65 penalised steps |
B12 IS reproducible cold to cold. The guard armed at token 735 on a normal formula line, the 12-gram '$$\text{Area} =
\frac{1}{' (at 627 / 649 / 688 / 723), and at token 740 penalised the repeated '32': guard off '\frac{1}{2} \times 32 \times
2 = 32', guard on '\frac{1}{2} \times 16 \times 2 = 16' -> a wrong number, the solution never finishes. The main track ships
the same guard (orchestrator told). B12's verdict rules (truncations; IFEval via the pool-history cascade) are confounded
by the guard: the reference was decoded without it; a like-for-like comparison decides (the Eval suite's choice).

### B12 re-scored without the guard's effect (NewIdeas 16:02, ~/NewIdeas/PLAN_B.md, scripts/b12_runnoise.py; the suite's checkers)
- Math: every math loss of the guard run (1 win / 3 losses vs ref) is in the 10 rows the guard could touch; on the other 40
  B12 passes / fails exactly the ref's rows. Without the guard: 88 % vs 86 %.
- IFEval: two B12 runs disagree with each other on 11 of 98 rows (5 / 6), as often as either disagrees with the ref; the
  deficit on the 96 rows the guard cannot touch is -2.1 / -3.1 pt (inside B12's run-to-run noise).
- Truncations without the guard: 14 vs 12, still one over the line (IFEval 8 vs 5).
=> B12 is BORDERLINE, not clearly failed. Method for the next B12 / B8 verdicts: two runs or a per-row pool reset; guard R1 or
none on both sides. 4-bit pools / late cold ranks (IDEAS D1n) = quality headroom, maybe not needed to pass.

### B8 + MTP crash fix verified (day_b8m.sh, 17:47; fix ad8779c: non-trunk layers run MTPLX's own call)
B8 = load_b12(0.05) + B12_LATE3=2,4 + B12_NORM=mass, MTP 3:3:0, code prompt, PF_GROUP 64, MTP_C 32: the MTP prefill with
late writes installed completes (d1i prefilled state 326 tokens in 5.6 s); decode 31.2 tok/s, 3.56 tok/pass, peak
footprint ~8.4 GB (decode phase 7.2 GB active). The crash the Eval suite hit (layer_call on the MTP layer) is gone.

### EVAL SUITE VERDICT: B8 = FAIL, 3/5 rules (report_b8.txt 19:42; guard OFF on both sides; eager generate_ar; 210 core rows)
| test | B8 | ref | floor | B8 vs ref [90 % CI] | rule |
|---|---|---|---|---|---|
| math (n=50) | 86.0 % | 86.0 % | 88.0 % | +0.0 [-4.7, +4.7] (1/1) | - |
| ifeval (n=100) | 84.0 % | 90.0 % | 88.0 % | -6.0 [-11.1, -0.9], McNemar p 0.11 (8/2) | **FAIL** |
| humaneval+ (n=60) | 88.3 % | 91.7 % | 95.0 % | -3.3 [-7.2, +0.5] | PASS |
| decode length ratio | x0.997 | | | | PASS |
| hesitation words / 1K | 0.29 | 0.27 | | | PASS |
| truncations / loops | 17 / 3 | 13 / 2 | | (loops: ifeval/1219, 1675, 3439) | **FAIL** |
Answer agreement with ref 93.3 % (floor 97.1 %); token-identical 31/210 (floor 59/210). Same shape as B12 (IFEval at the
bar, one run's noise ~2.4 pt per NewIdeas): a clean pass needs the quality step -> day_d1n.sh (4-bit pools at C_m 51 / 46,
B12 + late cold ranks, B8-mass) queued 20:10 (NewIdeas), candidate picked on KLD + MoonShots' phase-B speed numbers.

### Late writes under MTP: stale-entry bug (MoonShots 20:19) fixed 1a5eee1 (GPU check queued in day_d1n.sh)
After a partly accepted verify pass, MTPLX's commit_verified_window replays layer 1 alone on the kept rows; with keep == 1
B12's block took the 1-row late path (STASH[1] -> PENDH[2]); layer 2 is never replayed, so the NEXT verify pass applied
the stale entry at layer 2: reshape crash on a row mismatch (every 192-token B8 + MTP run in MoonShots' A/B), else a stale
late part on an unrelated token. Also the MTP head's layer (L 48) stashed a late part nobody wrote. Fix: clear STASH /
PENDH at layer 0 of each trunk forward; the MTP head's layer counts as last (waits for its cold ranks). Eager B8 unchanged.
B8's earlier 96-token MTP check (31.2 tok/s) likely never had a keep == 1 commit.

### Like-for-like verdicts (guard OFF on both sides; Eval suite 20:27; ~/EvalSuite/docs/RESULT_smallram_2026-10-22.md)
| variant | math | IFEval | HumanEval+ | truncations | rules |
|---|---|---|---|---|---|
| **B12** (C_m 51, mass, MTP, 210 rows) | 88.0 % (+2.0) | 86.0 % (-4.0, PASS) | 88.3 % (-3.3, PASS) | 16 vs 13 FAIL | **4/5** |
| B8 (C_m 26, mass, eager) | 86.0 % (0.0) | 84.0 % (-6.0 [-11.1, -0.9], FAIL) | 88.3 % (-3.3, PASS) | 17 vs 13 FAIL | 3/5 |
B12 (with guard) extra rows: MGSM 90.0 % vs ref 87.5 % (all languages within 1 problem); multilingual similarity to ref
0.42 vs floor 0.64 (-0.22 [-0.29, -0.15]) but 98 % in-script; agentic similarity 0.59 vs floor 0.58 (same). Second runs
b8.r2 / b12.r2 (pool-history noise) and B8-fast queued. B12 guard-off misses only the truncation rule: the quality step
(day_d1n.sh: 4-bit pools / late cold ranks) targets exactly that (more confident decoding, fewer rambling caps).

### QUALITY STEP screen on the real v3f path (day_d1n.sh, 21:44-; tf_b12 = the Eval suite scorer; mass weights; every 8)
Hard text (tf_ifeval + tf_multi, 7 seqs, 2,353 positions; strict per-part targets 0.0204 / 0.0206):
| candidate | RAM vs B12 | hard KLD | vs B12-mass | ifeval / multi |
|---|---|---|---|---|
| B12-mass (C_m 51, plan 4433333333) | - | 0.1117 | - | 0.1139 / 0.1091 |
| B12 + 4-bit pools C_m 51 (B12_PLAN4=1, C_l 51) | +1.1 GB | 0.0993 | -11 % | 0.1039 / 0.0938 |
| B12 + 4-bit pools C_m 46 (C_l 46) | +0.4 GB | 0.1082 | -3 % | 0.1126 / 0.1031 |
| **B12 + LATE3=2,4** (cold ranks 2-3 half a layer late) | same | **0.0508** | **-55 %** | 0.0561 / 0.0444 |
| **B12 + LATE3=2,10** (all cold ranks 2-9 late = dh at 10 %) | same | **0.0178** | **-84 %** | **0.0186 / 0.0170 (PASS strict)** |
The dominant quality cost of frugal is DROPPING cold ranks 2-9, not the 3-bit tiers (~11 % on the same runtime; my
cross-runtime +22 % estimate was overstated). Late computation at the same RAM recovers near-exact quality; the price is
the late reads (speed: day_spd.sh next; under MTP the verify rows wait for them until multi-row phase B lands).
B8 + MTP 192-token check of the late-write fix (1a5eee1): code 27.2 tok/s (3.25 / pass) 8.4 GB, prose 17.1 (2.21) 8.3 GB,
no crash.
Easy text (tf_gsm + tf_code, 11 seqs, 2,261 positions; strict targets gsm 0.00449, code 0.00253):
| candidate | easy KLD | vs B12-mass | gsm / code |
|---|---|---|---|
| B12-mass | 0.0244 | - | 0.0286 / 0.0118 |
| B12 + 4-bit pools C_m 51 | 0.0234 | -4 % | 0.0274 / 0.0116 |
| B12 + 4-bit pools C_m 46 | 0.0245 | 0 % | 0.0282 / 0.0137 |
| B12 + LATE3=2,4 | 0.0123 | -50 % | 0.0145 / 0.0057 |
| **B12 + LATE3=2,10** | **0.0042** | **-83 %** | 0.0051 (just over 0.0045) / **0.0013 (PASS)** |
B8-mass hard 0.0705 (ifeval 0.0809, multi 0.0579); its easy row was cut at the lock's end. Same pattern on both kinds of
text: late cold ranks are the quality lever; 4-bit pools barely move it.

### Speed + SSD of the quality candidates (day_spd.sh, 00:28-00:32 Fri; mass weights; raw lines, analysis at the next wake)
- spd_b12_code: 23.7 tok/s | PEAK FOOTPRINT 11.0 GB | SSD during decode: 1.8 GB/s, 76 MB/token, late reads 0.0/token
- spd_l24_code: 20.0 tok/s | PEAK FOOTPRINT 11.1 GB | SSD during decode: 2.6 GB/s, 130 MB/token, late reads 28.0/token
- spd_l210_code: 14.2 tok/s | PEAK FOOTPRINT 11.1 GB | SSD during decode: 5.1 GB/s, 356 MB/token, late reads 128.9/token
- spd_b12_mtp_code: 30.4 tok/s, 2.72 tok/pass | PEAK FOOTPRINT 11.3 GB | SSD during decode: 3.6 GB/s, 321 MB/pass, late reads 0.0/pass, waited 208.4/pass
- spd_l24_mtp_code: 33.7 tok/s, 3.46 tok/pass | PEAK FOOTPRINT 11.3 GB | SSD during decode: 4.7 GB/s, 482 MB/pass, late reads 0.0/pass, waited 270.6/pass
- spd_l210_mtp_code: 21.2 tok/s, 3.37 tok/pass | PEAK FOOTPRINT 11.3 GB | SSD during decode: 7.3 GB/s, 1144 MB/pass, late reads 0.0/pass, waited 574.8/pass
- spd_b12_prose: 23.0 tok/s | PEAK FOOTPRINT 10.9 GB | SSD during decode: 1.9 GB/s, 84 MB/token, late reads 0.0/token
- spd_l24_prose: 19.5 tok/s | PEAK FOOTPRINT 10.9 GB | SSD during decode: 2.7 GB/s, 136 MB/token, late reads 27.0/token
- spd_l210_prose: 13.9 tok/s | PEAK FOOTPRINT 10.9 GB | SSD during decode: 5.0 GB/s, 360 MB/token, late reads 124.9/token
- spd_b12_mtp_prose: 25.2 tok/s, 2.25 tok/pass | PEAK FOOTPRINT 11.2 GB | SSD during decode: 3.6 GB/s, 320 MB/pass, late reads 0.0/pass, waited 203.4/pass
- spd_l24_mtp_prose: 22.9 tok/s, 2.42 tok/pass | PEAK FOOTPRINT 11.2 GB | SSD during decode: 4.7 GB/s, 491 MB/pass, late reads 0.0/pass, waited 280.8/pass
- spd_l210_mtp_prose: 13.1 tok/s, 2.17 tok/pass | PEAK FOOTPRINT 11.2 GB | SSD during decode: 7.3 GB/s, 1202 MB/pass, late reads 2.0/pass, waited 605.7/pass
Per verify pass (MTP; tok/s / tok/pass, acceptance varies with the text): B12 89 ms code / 89 ms prose; + LATE3=2,4 103 / 106
ms (+15-19 %); + LATE3=2,10 159 / 166 ms (+79-86 %). Eager: B12 23.7 / 23.0 tok/s, + 2,4 20.0 / 19.5 (-16 %), + 2,10 14.2 /
13.9 (-40 %). SSD for 2,10 with MTP: 7.3 GB/s (1.1-1.2 GB / pass), well below the ~13 GB/s ceiling -> latency-bound, not
bandwidth-bound (NewIdeas' test). Footprints unchanged (10.9-11.3 GB). CANDIDATE TRADE: B12 + LATE3=2,4 = KLD -50-55 % at
-15-19 % speed (MTP 33.7 / 22.9 tok/s); B12 + LATE3=2,10 = near-exact (KLD -83-84 %) at ~-45 % speed (MTP 21.2 / 13.1).

### B12 second core run (report_b12.r2.txt, guard off both sides): FAIL 3/5
math 88.0 % (+2.0); IFEval 87.0 % (-3.0 [-7.9, +1.9], PASS); HumanEval+ 86.7 % (-5.0 [-11.1, +1.1], FAIL; run 1: 88.3 % PASS);
truncations 17 vs 13 FAIL (run 1: 16). Run-to-run noise flips HumanEval+; the truncation failure is consistent (16, 17 vs
13): B12L (late cold ranks 2-3, handed to the suite 02:4x) targets it.

### B12L (B12 + LATE3=2,4, mass, MTP, guard off) first core run: 4/5 (report_b12l.txt, Fri morning)
math 86.0 % = ref; IFEval 86.0 % (-4.0 [-8.0, -0.0], PASS); HumanEval+ 90.0 % (-1.7, PASS); truncations / loops 17 / 3 vs
13 / 2 FAIL. Halving the KLD did NOT move the truncation rule: 16-17 truncations for B12, B8 and B12L alike -> next wake:
row-level look at which rows hit the cap (legitimate long math vs loops; same rows as the ref?). Second run (b12l.r2) and
B12LL core follow in the suite's runner.
B12L both core runs (Eval suite 11:40, pooled vs ref): math 86.0 % (+0.0 [-4.7, +4.7]); IFEval -4.0 [-7.6, -0.4];
HumanEval+ -0.8 [-3.9, +2.3]; 4/5 in EACH run, only truncations / loops fail: 17 / 3 in both runs vs ref 13 / 2. vs B12
pooled: HumanEval+ better (-0.8 vs -4.2), IFEval same (-4.0 vs -3.5), math -2. Truncation excess systematic across
B12 / B12L / B8 (16-17 vs 13) -> row-level analysis of finish == length rows next (CPU only).
B12LL (B12 + LATE3=2,10, mass, MTP 3:3:0, guard off) core run 1 (Eval suite 17:28): 5/5 RULES PASS - the first frugal
config to pass the truncation rule. math 86.0 = ref; IFEval 87.0 vs 90.0 (-3.0 [-6.7, +0.7]); HumanEval+ 91.7 = ref
(100 % answer agreement); answer agreement 96.7 % vs floor 97.1 %; token-identical 50/210 (floor 59); truncations /
loops 14 / 1 vs 13 / 2. ~13 tok/s in the suite's harness (mine: MTP 21.2 / 13.1 code / prose, eager 14.2 / 13.9).
Second core run + MGSM / multilingual / agentic rows queued (runner 13, overnight).
B12LL second core run (Eval suite 22:43) = run 1 token for token (210/210): 5/5 TWICE (math 86.0, IFEval -3.0,
HumanEval+ = ref, truncations 14 vs 13). Run 2 used Fit12's shipping env (PF_GROUP 64 + ASYNC_EVERY) from chunk 2:
identical tokens, 11.28 GB. Fit12's B12LL + budget b16: IFEval -7.0 vs ref (FAIL); b20 being judged.
Fit12 memory (22:33): B12LL peaks 11.25-11.35 GB right after load (MLX peak ~10.8-10.9 vs 9.68 active), 10.4-10.7 GB at
short context; C_m 59 (+8 slots/layer) = exact, +6-7 % tok/s, but peaks 12.08 (load) / 12.12 (2.6K) / 12.75 GB (8K).

### Queue L2-07 (Sat 12:15, CPU): B12L's 17 truncations vs a decode-time run-on stop rule -> KILLED (2 fixable < 4)
Core rows (math 1536 / ifeval 1024 / humaneval 768 token caps), finish == length, from ~/EvalSuite/results/*/decode.jsonl:
ref 13. B12L 17 = 12 shared with ref + 5 extra: humaneval/147 (ref ended at 722 of 768) and math c&p/430 (ref 1497 of
1536) = near-cap length flips; humaneval/93 = rambling in code comments (ref ended at 340); ifeval/1219 ("A NIKE SNEAKER IS
FOR ...") and ifeval/2275 ("We have a start-up belief. We have a start-up trust. ...") = templated sentence loops with a
varying slot (2275 has NO repeated 12-gram, so the 12-gram repetition guard cannot see it). Only those 2 could be ended
correctly by a stop rule. Same picture in every variant: B12LL 14 = 12 shared + c&p/430 (extra in EVERY frugal variant)
+ ifeval/3126 (also truncated by 'floor' = same weights); B12 16-17, B8 13-17. The truncation rule sits at the noise of
2-3 borderline rows; B12L's excess over B12LL = 147 + 93 + 1219 + 2275.

### Queue L2-06 (Sun 04:20): 8-bit KV (b12_kv8, affine g64 on K and V of the QSA layers) vs bf16 KV, B12LL decode path
kv8_tf.py (ctx64k.txt; decoupled prefill of N tokens, then 256 teacher tokens in 4-row chunks; KL(bf16 || 8-bit)):
N=16384: KLD mean 0.0883 (p50 0.0035, p99 1.89, max 3.21), top-1 agree 0.969; QSA KV 0.403 -> 0.214 GB; ms/chunk 232 -> 233.
N=32768: KLD mean 0.2227 (p50 0.0054, p99 6.97, max 11.6), top-1 agree 0.941; QSA KV 0.805 -> 0.428 GB; ms/chunk 236 -> 245.
MUCH larger than 8-bit KV usually costs (~1e-3): suspect key-channel outliers under per-row groups (KIVI quantizes
keys per channel) or harness nondeterminism. Next: bf16-vs-bf16 control + K-only / V-only / g32 arms at 16K.
Sun 05:06, N=16384, same harness: bf16 vs bf16 control KLD 0.000000 / top-1 1.000 (deterministic). Every perturbation
gives the SAME size: kv:64 0.0883 (top-1 0.969), V only 0.0852 (0.945), K only 0.0945 (0.949), kv g32 0.0818 (0.961).
-> not quantization-limited: at 16K the B12LL path amplifies ANY tiny change (p50 ~0.003, but p99 1.5-2.2 from a few
flipped discrete choices: routing / QSA block selection). Floor check (benign prefill chunk 256 vs 512) running.
FLOOR (05:10): bf16 vs bf16 with the decoupled prefill's attention chunk 256 instead of 512 (benign runtime change):
KLD mean 0.1488 (p99 2.07, max 6.16), top-1 agree 0.891 - LARGER than 8-bit KV's 0.088 (and kv:64@256 vs bf16@512 0.080).
-> at 16K the teacher-forced KLD is far above the strict short-context targets for ANY runtime change; 8-bit KV sits
below that floor. The strict kill line cannot judge long context; task results can (Kamen's small-RAM bar).
OPEN: active memory after prefill is HIGHER with 8-bit KV (9.56 -> 9.74 GB) although QSA KV bytes halve (0.403 -> 0.214):
something keeps a dequantized copy alive. Next: one arm per process (footprint now / lifetime max) at 32K.

### GATE BUG (MoonShots 05:20, confirmed): torn routing-row reads under GPU load
The loader reads stale ids from the 2nd 64-byte sector of a routing-row line whose tag (word 0) is fresh (post_ids: ids,
thread_scope_device fence, tags; device scope does not order for the CPU). Under load 7 torn calls in 5 of 6 runs, quiet 0;
stale = the previous post's ids -> wrong experts ensured/released (explains B8 / B12L / B12LL load-only divergence).
Fix: per-word seq stamps (MoonShots builds v3h from v3g; I fold; also suggested for the release region). BigMoE told.
Per process (05:49, N=32768, no MLX cache cap): bf16 active 10.05 GB, footprint now / lifetime max 13.30 / 13.72 GB;
8-bit active 10.31 (+0.26), footprint 12.43 / 13.38 (-0.87 now, -0.34 lifetime); QSA KV 0.805 -> 0.428 GB. Next: 64K
(N=65000) with B12_CACHE_MB=64 as shipped, one arm per process, footprints after prefill and after 256 teacher tokens.
64K (07:01-07:07, N=65000, B12_CACHE_MB=64 as shipped, one arm per process, engine v3g both; data/kv8/mem64k_*.txt):
  QSA KV 1.598 -> 0.849 GB | after prefill: active 11.02 -> 11.80 (a live bf16 copy of the whole KV in the 8-bit arm, gone
  after the first steps), MLX peak 11.62 -> 12.40, footprint now 13.26 -> 12.51 | after 256 teacher tokens: active 11.01 ->
  10.27, footprint now 11.83 -> 11.29 | LIFETIME PEAK 13.96 -> 13.28 GB | ms per 4-row chunk 206 -> 224 (+9 %).
  Reading: steady decode at 64K fits < 12 GB even in bf16; the prefill peak is the 64K blocker. 8-bit saves 0.5-0.7 GB
  everywhere and costs ~9 % per pass at 64K (full dequantize per read) until a quantized attention kernel exists.
v3i (bd60b20, 07:02): default engine = v3g + MoonShots' gate seq stamps; smoke sha 259e67193e = v3g, 22.6 tok/s, 10.2 GB.

### Queue L2-06 long-context TASK check (Sun 10:33-10:48; scripts/kv8_long.py, data/kv8/long_*.json; B12LL, prefill 512/4096, cache 64 MB)
Real-code docs with planted facts, 3 questions as successive turns (mq 4 of 8 magic numbers, vt 5-step chain + 3-step
distractor chain, uuid 1 of 40). `kv8_long.py score` also counts distractors named.
32K (35,002-token first turn): bf16 mq 4/4, vt 5/5 (0 distractors), uuid 1/1, prefill 142 s | kv:64 mq 4/4, vt 5/5 PLUS all
  3 distractor-chain names (they hold a different value), uuid 1/1, prefill 140 s -> 8-bit KV made the one multi-hop error.
60K (63,022 tokens): bf16 mq 4/4, vt 5/5 (0 distractors), uuid 1/1, prefill 291 s | kv:64 LOST: the Mac slept 10:48-11:03
  mid-arm, my lock expired in the sleep, the runner stopped the job at its lock check (re-run with L2-10's follow-up).
Memory (20 Hz sampler: footprint, MLX active, PHASE): prefill footprint peak 32K bf16 12.61 GB (act 11.20) at (28672, 27,
  attn); 32K kv:64 12.30 (act 11.74) at (28672, 22, moe); 60K bf16 13.84 (act 12.79) at (57344, 39, attn) = a QSA layer
  in the last MoE chunk. Widest footprint - active 2.28 / 2.35 / 2.61 GB, all in attention phases. Q&A peaks 11.66 /
  11.26 / 12.54 GB. mx.get_peak_memory() read BELOW the sampled active in every arm (10.65 / 10.25 / 11.23): don't use it.
footprint -p every 3 s (data/kv8/fp_series_long.txt): the gap is 'Owned physical footprint (unmapped) (graphics)', i.e. GPU
  memory already freed but not yet reclaimed. It swings 0.07 -> 2.2 -> 0.07 GB within seconds, max 2.0-2.2 GB in every arm,
  1.0-2.1 GB of it inside the max-footprint samples.
FINDING (code + probe 10:47): b12_prefill never enters mtplx attention_phase("prefill"), which MTPLX's own generation does.
  So the sparse prefill lane never engages, though it is AUTO-supported and enabled on this M5 Pro (selector + NAX flash
  consumer for chunks with >= 32K history). Every 512-row chunk instead takes the stock dense path: a bool [S, T] mask
  and dense_band_sdpa, i.e. a [24, 512, T] score plane per QSA layer. That is the ~1.8 GB attention transient at 60K and
  likely the 10-22 MTP path's +2.1 GB.
  Second suspect: each s_attn chunk grows every QSA layer's stock KV by concatenation, freeing ~1.5 GB per chunk at 64K
  over 12 layers, which feeds the unmapped pool. mtplx qsa_rows_target(R) sizes all QSA buffers once (exact by design).
L2-10 batch: scripts/pk64.py + day_pk64.sh, one process per variant at N=65000: base / one-pass growth / lane / both /
  base + MTP / both + MTP. Exactness by token sha + first-step logits hash; footprint categories every 3 s; tier counters.
  Queued 11:06 (45 min).
L2-10 LINE (pre-registered Sun 13:10, with NewIdeas):
- PASS: the N=65000 prefill footprint peak on the plain B12LL path falls under 12.0 GB (base ~13.8-14.0). The MTP path is
  reported beside it.
- The exact change (one-pass growth) must keep the base's token sha and logits hash.
- The non-exact change (prefill lane) must leave the long-context tasks unchanged: 32K + 60K bf16 with the lane scoring
  mq 4/4, vt 5/5 with 0 distractors, uuid 1/1.
- KILL: under 1 GB of the base peak attributable to what we can change.

### Queue L2-10 batch 1 (Sun 13:11-13:39; scripts/pk64.py, data/pk64/results.txt + <tag>.json; N=65000, B12LL, cache 64 MB)
CAVEAT: run WITHOUT the shipped grouped tier prefill (BIGMOE_TIER_PF_GROUP=64, the Eval suite's SHIP_AR env; the library
path doesn't set it). So the MoE phase used _prefill_single's full-union scratch: ~512 experts x 2.77 MB = 1.4 GB per
layer call, freed and reallocated every call. The same holds for the L2-06 memory runs (kv8_tf, kv8_long).
| variant | prefill | PEAK footprint (MLX active) at | max active, attn / moe phase | after 8 decode fp/act | sha / logits |
|---|---|---|---|---|---|
| base (dense attention) | 306 s | 13.66 (13.26) at (61440, 35, attn) | 13.42 / 12.76 | 11.58 / 11.00 | cfa3402e3e / 4dcabb778f |
| one-pass growth | 302 s | 13.82 (12.66) at (61440, 16, moe) | 13.31 overall | 11.60 / 11.03 | = base (EXACT) |
| prefill lane | **235 s** | 13.69 (12.84) at (61440, 46, moe) | **11.43** / 12.84 | 11.59 / 11.00 | 56c4e038e8 / 1ccb5e8c59 |
| lane + one-pass | 233 s | 13.90 (12.53) at (61440, 46, moe) | 13.15 overall | 11.63 / 11.03 | = lane |
| base + MTP head | 297 s | 15.41 (14.24) at (61440, 16, moe) | 15.26 overall | 13.41 / 12.84 | = base |
| lane + one-pass + MTP | 232 s | 15.66 (14.64) at (20480, 22, moe) | 14.98 overall | 13.47 / 12.87 | = lane |
Readings:
- The lane works: 756 chunk x layer calls took metal_selector + flash_kernel. Attention-phase max MLX active fell 13.42 ->
  11.43 GB (-2.0 GB: the dense score plane), and prefill got 1.30x faster at 64K (306 -> 235 s; MTP path 297 -> 232 s).
  It is non-exact (sha differs); a task check is needed.
- The footprint peak did not move: the MoE phase (full-union scratch) peaks at the same ~13.7 GB.
- One-pass growth is exact but changes nothing measurable.
- The freed-not-reclaimed pool ('unmapped') maxes at 2.13-2.18 GB in every variant.
- The MTP head adds ~1.7 GB from load (before prefill 10.91 vs 9.24 GB) and ~1.8 GB in steady decode.
- Separate find: decode at 64K uses the dense [1, T] mask over the whole KV in all 12 QSA layers ('decode_dense_mask' 96
  per 8 steps). MTPLX's flash-skip decode lane exists but is opt-in (MTPLX_QSA_FLASH=1) and nothing of ours sets it.
Batch 2 (scripts/day_pk64b.sh) re-runs the variants in the shipped env (TIER_PF_GROUP 64, PF_GROUP 64, MTP_C 32,
ASYNC_EVERY 1), adds s_moe 2048 and a 1 GB MLX cache, and adds a paired decode A/B, flash-skip off/on, on the 64K cache.

### Queue L2-10 batch 2, SHIPPED ENV (Sun 14:03-14:31; TIER_PF_GROUP 64, PF_GROUP 64, MTP_C 32, ASYNC_EVERY 1; N=65000)
| variant | prefill | PEAK footprint (MLX active) at | max act | after prefill / 8 decode fp | unmapped max | sha |
|---|---|---|---|---|---|---|
| base (dense attention) | 312 s | 13.58 (13.38) at (61440, 47, attn) | 13.38 | 12.30 / 11.64 | 1.99 | cfa3402e3e (= batch 1 base: grouped = single here) |
| lane | **212 s** | **12.69** (11.60) at (61440, 43, post) | 12.14 | 12.49 / 11.56 | 1.36 | 56c4e038e8 |
| lane + s_moe 2048 | 298 s | **12.32** (11.15) at (61440, 46, post) | 11.72 | 12.11 / 11.56 | 1.04 | = lane (s_moe EXACT) |
| lane + MLX cache 1 GB | 208 s | 13.67 (12.12) at (57344, 34, moe) | 12.15 | 13.20 / 11.57 | 1.68 | = lane |
| base + MTP | 274 s | 15.25 (14.65) at (61440, 47, attn) | 15.16 | 13.55 / 13.36 | 1.85 | = base |
| lane + MTP | 211 s | 14.53 (13.98) at (57344, 44, moe) | 13.98 | 13.58 / 13.40 | 1.32 | = lane |
Readings:
- The lane in the shipped env: peak -0.89 GB (13.58 -> 12.69) and prefill 1.47x faster (312 -> 212 s).
- Adding s_moe 2048 brings the peak to 12.32 GB (-1.26 GB in all), but prefill is 40 % slower than the lane alone.
- A bigger MLX cache makes it worse.
- vs the line: PASS needs < 12.0 GB, best is 12.32, so NOT MET. Not KILLED either: 1.26 GB of the peak is attributable.
- Where the remaining peak sits: the post phase right after each MoE, with MLX active ~11.2-11.6 plus 1.0-1.4 GB of
  just-freed, not-yet-reclaimed GPU memory. Each grouped MoE layer call frees ~1.2 GB: 2 x 64-expert buffers (355 MB)
  and ~4 x 210 MB output copies.
- Next: persistent group buffers (BIGMOE_TIER_PF_PERSIST=1, v3i tier.py opt-in, default off, exact by construction).
Decode A/B, flash-skip off/on, paired on the 64K cache (B12LL, 3 interleaved rounds of 16 steps): dense 79.7 / 82.1 / 72.1
  ms/step, flash 74.3 / 65.2 / 81.2. Median ~-7 %, but round noise (+-8 ms; B12LL reads cold ranks from SSD) is larger:
  inconclusive. Expected saving ~5 ms per step (dense reads ~1.5 GB of KV per step at 64K).

### lane_ref (Sun 14:31-14:35; NewIdeas: is the lane the reference behaviour? 4.2K prompt, crossovers 2049, B12LL)
FINDING (code + the 14:04 crash): generate_ar chunks its prefill only with MTPLX_SUSTAINED_PREFILL=1, which MTPLX's profiles
  set and neither load_b12 nor the Eval suite does. So the suite's generate_ar rows prefill the whole prompt as ONE forward
  (A0: prefill_chunks 1). The lane never engages there, and the attention is a dense [24, T, T] plane: ~13 GB at RULER's
  16K, 203 GB at 65K (the crash).
First-token top-20 logprobs and 16 greedy tokens, per pair:
  A generate_ar sustained-512 (lane) vs B b12_prefill 512/512 + lane: 9/16 tokens, max |dlogprob| 0.56
  C b12 dense vs D generate_ar sustained, lane off: 11/16, 0.69
  A vs D (lane on/off inside generate_ar): 9/16, 0.56; B vs C (lane on/off in b12): 16/16 tokens, 1.00
  B vs E (flash-skip decode): 11/16, 0.63; A0 one-span vs C / D: 9/16, 1.48
-> No pair is bitwise. b12_prefill differs from generate_ar at the same chunking even with the lane off, and every
  runtime change moves the first token's logprobs by 0.5-1.5 at this length. This is the chaotic amplification L2-06
  measured. So the lane (and flash-skip decode) are runtime changes of the same size as the accepted b12_prefill vs
  generate_ar difference, and the 32K / 60K task check is the guard; adoption stays Kamen's call.
Persistent group buffers (14:36-14:39, BIGMOE_TIER_PF_PERSIST=1 on the lane, s_moe 4096): sha = lane (exact), prefill 207 s,
  PEAK 12.70 GB (lane 12.69). The unmapped pool max fell 1.36 -> 1.07 GB, but the always-allocated buffers add the same
  ~0.36 GB to MLX active; decode also kept them (11.37 vs 11.00 GB active: B12's decode path never calls tier's
  prefill_done). NO GAIN -> reverted (v3i tier.py = committed version again).
L2-10 STATUS (Sun 14:45): best plain-path 64K peak 12.32 GB (lane + s_moe 2048), 0.32 GB over the pre-registered 12.0 GB.
  1.26 GB is attributable to changes, so not killed. Untested levers left:
  - s_moe 1024 (exact, slower prefill)
  - s_attn 256 (non-exact)
  - fewer MoE output copies in _prefill_grouped (concat + take = 2 x 210 MB per layer; exact)
  - a smaller steady state (8-bit KV -0.75 GB at 64K, but it named distractors at 32K)
L2-10r (NewIdeas, Sun ~14:50; pre-registered):
- BANK the lane: 12.69 GB and 1.47x faster 64K prefill. The 32K / 60K task check (queued) is the guard; adoption is Kamen's.
- ONE more exact lever, kept small: drop 2 of the ~4 MoE output copies in tier._prefill_grouped (concat + take).
  PASS = >= 0.3 GB off the 64K plain-path lane peak (12.69 -> <= 12.39 GB) with sha identical (56c4e038e8 / 1ccb5e8c59).
  Otherwise stop L2-10.
- SKIPPED: s_moe 1024 (slower prefill) and s_attn 256 (non-exact).
- Projection if 8-bit KV passes its 60K + distractor check: lane + copy drop + 8-bit KV ~11.7 GB at 64K.

### Lane task check + the lost 60K 8-bit arm (Sun 14:54-15:13; kv8_long, B12LL; records long_<L>_g64[_lane].json, long_60000_kv64.json)
Shipped env (TIER_PF_GROUP 64, PF_GROUP 64, ASYNC_EVERY 1), bf16 KV, lane off vs on; `kv8_long.py score` counts distractors:
| run | 32K: mq / vt / uuid | 60K: mq / vt / uuid | prefill 32K / 60K | prefill peak 32K / 60K |
|---|---|---|---|---|
| lane off | 4/4, 5/5 + 0 distractors, 1/1 | 4/4, 5/5 + 0, 1/1 | 127 / 268 s | 11.84 / 13.40 GB |
| lane on | 4/4, 5/5 + 0, 1/1 (answers word for word as lane off) | same: identical answers | 121 / **210 s** | 11.73 / **12.70 GB** |
-> The lane is TASK-NEUTRAL on these items at both lengths (identical answers) and makes 60K prefill 1.28x faster.
   Adoption is Kamen's call.
8-bit KV, 60K (the first runs' env, no grouped prefill; compare long_60000.json bf16): mq 4/4, vt 5/5 + 0 distractors, uuid
  1/1, identical answers to bf16; prefill 288 s, peak 12.86 GB. So 8-bit KV named distractors at 32K (3 names) but not at
  60K. One item per length, so this stays a weak, mixed signal; NewIdeas wants more distractor items before any adoption.

### Queue L2-10r, the MoE output copy drop (Mon 08:05-08:14; pk64, N=65000, shipped env + lane; data/pk64/g_lane_sc.txt, g_lane_r2.txt, g64_lane_sc/_r2.json)
BIGMOE_TIER_PF_SCATTER=1: each group's MoE outputs scattered into one preallocated buffer instead of concat + take.
| run | scatter | peak footprint (active then) | MLX peak | peak by part attn / moe / post | prefill | sha / logits |
|---|---|---|---|---|---|---|
| g64_lane_sc | on (768 grouped calls) | **13.40** (11.89) GB | 11.62 | 12.72 / 13.40 / 13.07 | 302 s | 56c4e038e8 / 1ccb5e8c59 |
| g64_lane_r2, same-window control | off | 12.80 (12.14) GB | 11.62 | 12.33 / 12.80 / 12.79 | 241 s | same |
| g64_lane / g64_lane_ps (Sun) | off | 12.69 / 12.70 GB | 11.62 / 11.98 | | 210 / 207 s | same |
-> FAIL (line: <= 12.39 GB with sha identical). Exact, but the footprint peak went UP 0.60 GB against the same-window
   control (MLX's own peak unchanged), and prefill took 1.25x as long. Per the pre-registration, L2-10 STOPS; the opt-in
   is reverted (v3i tier.py back to its pre-5936bca text). Likely cause, untested: the scatter does not write in place
   (the async_eval'd graph still holds the old buffer), so each of the 64 groups copies the whole ~210 MB output buffer.
   Lane footprint spread over 3 runs: 12.69-12.80 GB.

### Queue L2-06t-dx, the 8-bit KV distractor check (Mon 09:44-10:01; kv8_long KV8L_SET=_dx, shipped env + lane; data/kv8/dx_*.txt, long_{32000,60000}_dx_g64_lane.json)
One document per length, 6 interleaved 4-step chains (distinct values), each chain asked after the document from a
restored cache (QSA trim + ArraysCache / STASH / PENDH restore + reset_generation_state), item 0 re-asked last.
| L | arm | gold names /24 | distractor names | items naming a (near-)whole other chain | fork check | decode tok/s |
|---|---|---|---|---|---|---|
| 32K | bf16 | 13 | 19 | 2 [0, 3] | DIFFERS | 8.62 |
| 32K | kv:64 | 14 | 16 | 2 [0, 3] | DIFFERS | 14.13 |
| 60K | bf16 | 5 | 1 | 0 | DIFFERS | 11.85 |
| 60K | kv:64 | 4 | 2 | 0 | same | 10.10 |
-> INVALID, no verdict on 8-bit KV:
   1. The fork is not exact: item 0 re-asked after the other 5 answers differently in 3 of 4 runs (32K: 3rd name on;
      60K bf16: same meaning, other words). So items 1-5 did not start from the post-document state (NewIdeas'
      condition). Suspects, untested: arrays the snapshot only references being mutated in place (MLX setitem
      rewrites the Python object), QSA state outside trim's reach, STASH/PENDH values shared by the shallow copy.
   2. The 48-token cap: every 60K answer (both arms) opens "Based on the text provided, here are the variables ..."
      and is cut after one name; at 32K 2 of 6 per arm are cut the same way. So 60K measures nothing.
   As numbers kv8 sits inside the line at both lengths ((a) 14 vs 13, 4 vs 5; (b) 16 vs 19, 2 vs 1; (c) the same
   items [0, 3] in both arms at 32K, none at 60K), and the only clean fork per run (item 0's first answer) names the
   same 11 names in both arms at 32K (4 gold + 7 distractors, two swapped in order). Not a pass: the procedure failed.
   bf16 itself is weak here (13/24 with 19 distractors at 32K). Decode tok/s: one short run per arm, noisy.
   Fix before a re-run: deep-copied snapshot + a small-context exactness self-test of snapshot/restore (prefill 2K,
   answer, restore, answer again: ids must match) and a longer cap (~160 tokens) or a "names only, one per line" ask.

### Queue L2-11, the PFR port A/B (Mon 13:46-13:55; scripts/day_pfr.sh + pfr_ab.py, data/pfr/; v3l, AR path, shipped env, no lane)
Fit12's PFR-a ported into bt_b12 (B12_PFR; engine v3l = v3i + Fit12's reset_layer / invalidate_tables). Fresh process per arm.
| arm | tokens | prefill | KV after prefill | first-step logits | sha32 | prefill-phase peak | lifetime peak |
|---|---|---|---|---|---|---|---|
| r8 (PFR off) | 8,474 | 27.5 s | ecd389371d | c57fe16fd1 | 142abd5d7e | 11.05 GB | 11.07 GB |
| p8 (PFR on, 1 chunk of 8,474) | 8,474 | **18.9 s** | = | = | = | **9.27 GB** | 9.92 GB |
| r64 (PFR off, s_moe 4096) | 65,000 | 265.1 s | 2227a92414 | 4dcabb778f | dc67a9270b | 13.54 GB | 13.72 GB |
| p64 (PFR on, 16384 x 3 + 15848) | 65,000 | **208.1 s** | = | = | = | **11.08 GB** | 12.10 GB* |
-> PASS: bit-exact at both lengths, and r8 / r64 equal Fit12's v3p references (so v3l = v3i = v3p on this path).
   Prefill -31 % / -21.5 %; released 6.0 GB in 0.03 s, re-created in 0.13-0.14 s; decode 12.46-12.73 tok/s in all arms.
   *with the KV hash's float32 copies; Fit12's clean a64m (no hash) peaks at 11.77 GB, set by the decode.
   DEFAULT since 445e301: B12_ENGINE=v3l, B12_PFR=1 on AR-path prefills > 4096 tokens (NewIdeas: bit-exact, so no Kamen
   decision). Skipped (noted once) where not A/B'd: MTP feeds (the shipped MTP path's per-chunk head feed), kept streams,
   the L2-10 lane, frugal prefill, the D1I ring. b12_prefill now asserts s_moe % s_attn == 0 (the 64K T2 break).

### Queue L2-06t-dx re-run, the 2K restore self-test (Mon 14:00-14:01; data/kv8/dx_run.txt): FAIL, stopped before the long runs
| arm | cache fingerprint after prefill / after each of 3 restores | item 0 first-step logits: first / again | item 0 ids |
|---|---|---|---|
| bf16 | c02d6276f1 / c02d6276f1 x 3 | 6a3b9ca775 / 7e1b585e89 | differ (63 vs 18 tokens) |
| kv:64 | 284da08a91 / 284da08a91 x 3 | 5f5ff2d58d / 06ce04bb87 | differ (61 vs 18 tokens) |
-> The new restore IS bit-exact (every buffer of every layer, fingerprinted), yet the same item answers differently after
   other items ran: the decode depends on state OUTSIDE the cache. Prime suspect (untested): the expert pools' residency.
   Under B12LL a cold MSB-3 rank 2-9 expert is added half a layer late while a resident one is added in place, so what
   earlier text left resident changes the numbers. Fresh-process runs reproduce (pk64 / PFR A/B shas) because every
   process starts with the same pools. Consequence: one-process comparisons of arms or items under B12LL (L2-06t's
   bf16 -> kv:64 in one process, the dx items) carry residency from what ran before them.
   Next (needs NewIdeas' OK): every item starts with empty pools (PFR's re-create + reset_layer before each item, the
   first included), self-test again; if it then passes, residency is confirmed as the cause.

### Queue L2-06t-dx run 3, pools emptied before every item (Tue 19:26-19:45; data/kv8/dx_run.txt, dx_score.txt, long_{32000,60000}_dx_g64_lane_r3.json)
Same documents, questions and lines as run 2; restore() now also empties the decode pools before every item, the first
included (bt_b12's PFR release + re-create, 6.0 GB in 0.03 + 0.15 s). 2K self-test: PASS in both arms (bf16 item 0
logits 6a3b9ca775 = 6a3b9ca775, 63 = 63 tokens; kv:64 1fdb44ffc7 = 1fdb44ffc7, 52 = 52; fingerprints equal after every
restore). So pool residency WAS the cause of run 2's failure. Fork check identical in all 4 runs: VALID.
| L | arm | gold names /24 | distractor names | items naming a (near-)whole other chain | per item hits / distractors | decode tok/s |
|---|---|---|---|---|---|---|
| 32K | bf16 | 14 | 23 | 2 [0, 3] | 4/7 3/0 1/1 2/15 2/0 2/0 | 13.63* |
| 32K | kv:64 | 15 | 5 | 0 | 1/1 3/0 1/2 4/0 4/2 2/0 | 11.93 |
| 60K | bf16 | 10 | 5 | 1 [1] | 1/1 0/3 4/0 3/1 1/0 1/0 | 12.29 |
| 60K | kv:64 | 15 | 8 | 2 [0, 1] | 1/3 0/3 4/0 3/1 4/0 3/1 | 9.31** |
| 12 items | bf16 | 24/48 | 28 | 3 | | |
| 12 items | kv:64 | **30/48** | **13** | 2 | | |
-> By the letter: (a) PASS (30 vs 24, kv8 +6), (b) PASS (13 <= 28 + 2), (c) FAIL by ONE item: 60K item 0, where kv8 names
   3 of the 83126 chain (RHTYL, MWWFH, FHXFZ) and bf16 names 1 (RHTYL).
   On inspection that item is not a wrong answer list: at 60K both arms answer in step-by-step traces cut at 160 tokens,
   and in item 0 both start from the 83126 chain (bf16 lists RHTYL as answer 1, then "Wait ... re-read"; kv8 traces
   "RHTYL is 83126. So MWWFH is 83126", ruling them out). The scorer counts names mentioned, so a longer trace counts
   more. Item 1 starts with the same 700 characters in both arms. At 32K the flagged items ARE answer lists: bf16 names
   11 names for item 0 and 10+ for item 3, while kv8 answers item 3 with exactly the 4 gold names.
   So run 1's alarm (kv8 named a whole wrong chain at 32K where bf16 didn't) does not reproduce once every item starts
   from the same state: at 32K it is the other way round, and whole-chain mentions occur in both arms (bf16 3 / 12 items,
   kv8 2 / 12). Verdict on the line: NewIdeas' call; adopting 8-bit KV stays Kamen's call.
   *32K bf16 overlapped the end of soft_end's resume ramp (until 19:32:38). **60K kv:64's prefill overlapped NewIdeas'
   one-core text search (19:41:20-19:43:26); its decode wrote 960 tokens vs bf16's 648. Timings are information only.
   Memory (one process, kv8 after bf16, lane on so no PFR, every item's pools re-created): prefill footprint peak 11.90 /
   12.75 GB at 32K and 13.59 / 14.34 GB at 60K (bf16 / kv:64); Q&A peak 11.77 / 11.18 and 14.20 / 13.93 GB.

### R-20 KV8M + R-19 W24-12 (NewIdeas' Tue 19:5x hand-outs, QUEUE 'RAPID SCREENS'; lines written Tue ~20:05, before any run)
NewIdeas' dx verdict (19:5x): not a PASS by the letter, not re-scored; recorded as "Monday's distractor alarm does not
reproduce; no blocker from dx". Adopting 8-bit KV stays Kamen's call; R-20 supplies its memory evidence first.
- R-20 KV8M: does 8-bit KV save memory at long context with PFR on? (dx ran lane-on without PFR; kv8's prefill peak was
  HIGHER there, 12.75 vs 11.90 GB at 32K.) One 64K prompt (65,000 tokens of ctx64k.txt), AR path, v3l, shipped env, PFR
  on, no KV hash, 32 decode tokens; bf16 vs kv:64 (b12_kv8, kv8_long's arm), a fresh process each (scripts/pfr_ab.py).
  PASS: kv8's lifetime peak >= 0.3 GB below bf16's (room for ~3 more C_m slots). KILL: < 0.1 GB. Phase peaks: information.
- R-19 W24-12: price W24-sp + W24-ec (MSB piece 2,211,840 -> ~1,900,544 B, -14.07 %; LSB 720,896 B unchanged) on bt_b12's
  AR decode at 2.6K and 8.5K (scripts/r19_price.py). Gain = SSD share x byte cut. SSD share = (T_cold - T_hot) / T_cold
  per decode token (steps 2-64): cold = shipped (C_m 51, PFR on, decode from empty pools); hot = C_m 160, PFR off, the
  same prompt's second decode (experts already resident). Byte cut = 14.07 % x the MSB share of the cold run's bytes (MSB
  / LSB reads solved from its bytes and read count). Two numbers: the UPPER bound (all SSD-attributable time scales with
  bytes) and a BANDWIDTH estimate (only the byte part of each read shrinks: 0.06 ms floor + bytes / 8.1 GB/s per read).
  PASS: bandwidth estimate >= +8 % at 8.5K. KILL: upper bound < +3 % at 8.5K. Between: report. The MTP path (verify rows
  wait for every read) is not priced here. rANS decode time on the GPU is not counted (MoonShots: 46 GB/s).
- AW-AR (NewIdeas GO ~20:1x, written before any run): Fit12's ALLWAIT (single-row trunk passes wait for every forced rank,
  like verify rows) on the AR path, where every decode step is a single-row pass, vs the shipped late rule. One process,
  v3l, shipped env, PFR on; at 2.6K and 8.5K two ABBA rounds (s a, a s), each run from empty pools, 256 greedy tokens;
  warm = tokens 33-256 (scripts/aw_ar.py; bt_b12 gets a run-time B12_ALLWAIT switch, default off, AW_N engagement count).
  PASS (fold ALLWAIT into AR too): ALLWAIT's mean warm decode >= shipped's - 2 % at BOTH lengths. KILL for AR: > 2 %
  slower at either (then ALLWAIT is folded on the MTP path only, where Fit12 measured it free). Token hashes both ways
  (ALLWAIT is expected to change AR bits; adopting it on AR is Kamen's call).

### Results: R-20, AW-AR, R-19 (Tue 20:01-20:33; scripts/runner_r1920.sh, one 10-min quiet lock per item; data/r1920/)
R-20 KV8M (64K = 65,000 tokens, AR path, PFR on: one release of 6.0 GB, 16384 x 3 + 15848; fresh process per arm):
| arm | prefill | prefill-phase peak | MLX peak | decode-phase peak | after decode | LIFETIME peak | decode (32) |
|---|---|---|---|---|---|---|---|
| bf16 | 205.2 s | 11.02 GB | 11.04 GB | 11.77 GB | 11.64 GB | **11.80 GB** | 12.57 tok/s |
| kv:64 | 206.9 s | 12.34 GB | 12.40 GB | 11.66 GB | 11.10 GB | **12.69 GB** | 9.26 tok/s |
-> KILL: 8-bit KV RAISES the 64K peak by 0.89 GB. The KV it stores is smaller (after decode -0.54 GB, decode-phase peak
   -0.11 GB), but its prefill transient is +1.3 GB (b12_kv8's attention appears to dequantize the history per chunk), and
   decode is -26 %. bf16's hashes = the L2-11 references (logits 4dcabb778f, sha32 dc67a9270b). Lead, not screened: a
   fused quantized attention (no dequantized history copy) would be the precondition for any 8-bit KV memory gain.
   (pfr_ab's header prints "B12_PFR 0" when the variable is unset; PFR ran, per the PFR log line.)
AW-AR (AR decode, 256 tokens from empty pools, warm = tokens 33-256, ABBA rounds in one process):
| length | shipped warm tok/s | ALLWAIT warm tok/s | ALLWAIT vs shipped | s1 = s2, a1 = a2 | tokens a = s | ALLWAIT layer calls s / a |
|---|---|---|---|---|---|---|
| 2.6K | 13.89 (13.82 / 13.97) | 14.42 (14.38 / 14.47) | **+3.8 %** | yes, yes | no (as expected) | 0 / 12,032 per run |
| 8.5K | 13.52 (13.48 / 13.57) | 13.04 (13.02 / 13.06) | **-3.6 %** | yes, yes | no | 0 / 12,032 per run |
-> KILL for the AR path (-3.6 % at 8.5K, line -2 %): ALLWAIT is folded on the MTP path only. The switch engaged only in
   the 'a' runs; each arm repeats bit for bit from the same empty pools; the shipped 8.5K run = the L2-11 r8 references
   (logits c57fe16fd1, sha32 142abd5d7e), so the switch (off) leaves the shipped path unchanged. Observation only: at
   2.6K ALLWAIT is faster (+3.8 %); a length-dependent rule would change AR bits (Kamen's call), not pursued.
R-19 W24-12 (AR decode, 64 tokens, steps 2-64; MB / reads per token from the process's disk bytes and ext misses):
| run | C_m | ms / token | MB / token | reads / token |
|---|---|---|---|---|
| cold 2.6K (shipped, empty pools) | 51 | 81.00 | 473.0 | 253.4 |
| cold 8.5K (shipped, empty pools) | 51 | 77.29 | 424.4 | 229.3 |
| hot 2.6K run 1 / run 2 | 160 | 66.40 / 52.83 | 288.4 / 158.1 | 159.9 / 90.6 |
| hot 8.5K run 1 / run 2 | 160 | 50.85 / 44.45 | 142.7 / 104.3 | 80.9 / 56.7 |
-> BETWEEN the lines (no PASS, no KILL). DEVIATION from the written method: the "hot" runs were not hot (104-158 MB / token:
   C_m 160 does not hold 64 tokens' experts, and the warm pools change the late rule's bits, so run 2's tokens differ
   from run 1's), so T_hot is not the no-SSD time. Used instead: step time against MB / token over all six runs, which is
   near-linear at ~0.10 ms per MB (8.5K: 0.103, intercept ~34 ms; 2.6K: 0.090, ~39 ms), i.e. reads cost about 10 GB/s
   on the critical path. SSD share of the shipped step = 56 % at 8.5K (43.5 of 77.3 ms), 52 % at 2.6K. Byte cut = -13.3 %
   per token (MoonShots' 12 GB replay, as NewIdeas directed; my own MSB / LSB solve gives MSB 90.6 % of bytes -> 12.7 %).
   UPPER bound (all of it scales with bytes): 8.5K -7.5 % time = **+8.1 %** tok/s; 2.6K -6.9 % = +7.5 %. BANDWIDTH estimate
   (the fixed ~0.06 ms per read stays: ~79 % of a 1.85 MB average read's time is bytes): 8.5K **+6.3 %**, 2.6K +5.8 %.
   The literal written method (cold vs the not-hot run 2) gives a lower share (42.5 %): +6.0 % / +4.7 %. The PASS line is
   on the bandwidth estimate (+6.3 %), so no PASS; only the 8.5K upper bound touches 8 %. Not counted, as written: the
   rANS decode, ~330 MB of codes per token at 46 GB/s = ~7.2 ms of GPU work,
   as large as the whole gain (~5.8 ms) unless it overlaps the SSD waits. Information: C_m 160 (pools 18.9 GB, process
   24.1 GB) runs 8.5K at 44-51 ms / token, 1.5-1.7x the 12 GB shipped decode.

### Fold prep: B12_FOLD (default off) + the v3m engine, fold-OFF regressions (Tue 20:54-20:59; data/fold/; runner_fold.sh)
NewIdeas 20:4x: fold Fit12's exact MTP-path stack into bt_b12 behind one flag, default off; run it only after Fit12's
R-01 confirmation (HELD at 20:45: a prompt prefilled a 2nd time in one process gives other final draft logits), then
an Eval-suite verdict. B12_FOLD=1 = PFR on every MTP prefill, C_m per prompt (fold_cm(n), B12_FOLD_CM points), the
QSA rows target n + max_new + 16 (R-02), MOV (spec125.restore_state pops a marked state), ALLWAIT inside
spec125.generate only (AR keeps the late rule: AW-AR). C_m per prompt needs v3m = v3l + reset_layer taking a new slot
count + drain() (waits until no async read is in flight; v3l's PFR release did not wait for prefetch / joint reads).
| check (fold OFF) | result |
|---|---|
| v3m build | clean (20:55) |
| AR 8,474 tokens, PFR on, v3m + new bt_b12 | KV ecd389371d, logits c57fe16fd1, sha32 142abd5d7e = L2-11's references |
| MTP p2k (2,617 tokens), 256 tokens: live bt_b12 on v3l vs new on v3m | both logits bd6097bdf7, sha32 82995cdb4e, sha256 a0774ec635, kv 37028a6545, head 8563204e11 |
| MTP p8k (8,474 tokens), same | both logits 7d045d7b7d, sha32 0b0d91cd43, sha256 ed67050746, kv 5a760388d6, head f59755cf27 |
-> PASS: with the fold off, the new bt_b12 (live since ae26248) and v3m change no bit on either path. In this prompt
   order (p2k, then p8k, fresh process) the stock MTP hashes equal Fit12's PFM / AW stack hashes, so a fold-ON run
   should reproduce them exactly (C_m 65 / 63 instead of 51 must not move a bit under ALLWAIT). Not run: waits for R-01.
- Fold-ON port check (NewIdeas GO 21:0x, before R-01; written before the run): MTP p2k then p8k, 256 tokens, v3m,
  B12_FOLD=1 (C_m 65 / 63 by the default points). PASS: runs clean, every lifetime peak <= 11.95 GB, and sha256 / kv /
  head (and p8k's logits) = Fit12's PFM / AW arms above. Any difference = the port: stop and diff before R-01.
- RESULT (Tue 21:09-21:10, data/fold/fold_on.txt): PASS. The port reproduces Fit12's arms bit for bit:
| prompt | C_m | logits | sha32 | sha256 | kv | head | lifetime peak | warm tok/s (stock) | prefill (stock) |
|---|---|---|---|---|---|---|---|---|---|
| p2k (2,617) | 65 | bd6097bdf7 | 82995cdb4e | a0774ec635 | 37028a6545 | 8563204e11 | 11.76 GB | 16.30 (15.06-15.17) | 10.6 s (10.0-10.3) |
| p8k (8,474) | 63 | 7d045d7b7d | 0b0d91cd43 | ed67050746 | 5a760388d6 | f59755cf27 | 11.79 GB | 18.29 (16.89-16.97) | 21.7 s (30.5-31.3) |
  Every hash = stock and = Fit12's PFM / AW. MOV took the state (keys left: h, logits); PFR re-created the pools at
  65 then 63 slots (7.45 / 7.24 GB, 0.18-0.19 s); ALLWAIT engaged in 18 / 16 single-row layer calls. Gains vs stock in
  this run: warm decode +7.5-8.3 %, 8.5K prefill -30 %, both peaks within 11.95 GB. The default flip waits for R-01
  (held: Fit12's pos-2 logits question) and then an Eval-suite verdict (NewIdeas).

### FR-Spec rebuild fix (NewIdeas GO ~21:4x; Tue 21:56-22:11; data/fold/frs_off.txt, frs_on.txt): PASS, live 07fec4a
Fit12's pos-2 finding (a prompt prefilled again in one process gave other draft logits) was the FR-Spec draft head:
spec125.generate rebuilds it with the prompt's ids only when it starts, after bt_b12's prefill had computed the first
draft with the previous request's subset. b12_prefill_state now rebuilds it first. v3m, MTP path, 256 tokens:
| run | prompts | p8k draft logits (every p8k) | p2k draft logits | sha256 / kv / head |
|---|---|---|---|---|
| fold off | p2k, p8k, p8k | 61b2520e58 x 2 | 98c6bc1c0d | = the regression's, every prompt |
| fold on | p8k (process-fresh), p2k, p8k | 61b2520e58 x 2 | 98c6bc1c0d | = the regression's, every prompt |
-> A repeated prompt now equals a fresh one on every hash. Tokens, KV and head are unchanged; only the draft-logits hash
   moved where a prompt has ids outside the base 64K set (draft-only: the trunk verifies every token). Information:
   the fold-off process's lifetime peak creeps 11.02 -> 11.84 -> 12.08 GB over three prompts; fold on stays 11.75-11.85.

### R-41: the write16 bisect (NewIdeas Wed; 15:47-16:04; data/w16/; scripts/w16_bisect.py, runner_w16.sh)
The Eval suite's fold verdict: 44 of the first 45 rows identical; agent/write16 (8,277-token prompt) splits at token 32
(on 321 tokens, off 466). Arms: one fresh process each, write16's own prompt_ids, decode._gen_mtp's call, 64 tokens.
| arm | pieces on | tokens | passes / accepted | C_m |
|---|---|---|---|---|
| off (live) | none | = suite OFF | 28 / 36 | 51 |
| on (live) | all | = suite ON | 27 / 37 | 60 |
| on, scratch defaults | all | = ON | 27 / 37 | 60 |
| fold-exact | MOV + PFR + rows target, C_m 51, no ALLWAIT | = ON | 27 / 37 | 51 |
| PFR only | PFR on the MTP prefill | = ON | 27 / 37 | 51 |
| PFR + C_m | PFR, C_m per prompt | = ON | 27 / 37 | 60 |
| rows target only | QSA rows target | = OFF | 28 / 36 | 51 |
| MOV only | MOV | = OFF | 28 / 36 | 51 |
| ALLWAIT only | fold off + B12_ALLWAIT=1 | = OFF | 28 / 36 | 51 |
-> PFR on the MTP path is the cause; ALLWAIT is not (EvalSuite's late counter: no cold rank 2-9 in any single-row pass,
   so ALLWAIT cannot change the MoE there). A fresh process reproduces both suite rows: no history effect.
   Hypothesis: the MoE chunk grid. Shipped s_moe 4096 splits 8,276 tokens 4096 + 4096 + 84; PFR's balanced grid is one
   chunk. A small tail chunk takes another gather kernel (stream.py: short calls are not segmented), so the last
   prompt rows' bits differ. L2-11's A/Bs had 282- and 3,560-row tails. Part 2 tests it: B12_PFR_GRID=keep (release the
   pools, keep the 4096 grid) on the MTP path and on the AR path (whose default is PFR on since 445e301).
R-41 part 2, the grid check (16:21-16:31; data/w16/w16_{pfr_keep,exact_keep,ar_*}.txt): CONFIRMED.
| arm (write16, 64 tokens) | MoE grid | tokens |
|---|---|---|
| MTP, PFR only, B12_PFR_GRID=keep | 4096 + 4096 + 84 (pools released) | = suite OFF (sha 5562a6070d) |
| MTP, fold-exact with keep (MOV + PFR-keep + rows target, C_m 51) | same | = suite OFF |
| AR, today's default (PFR balanced) | one 8,276-row chunk | sha 812892ac79: differs from PFR off at token 7 |
| AR, PFR off | 4096 + 4096 + 84 | sha c4bb59dcf7 |
| AR, PFR-keep | 4096 + 4096 + 84 (pools released) | sha c4bb59dcf7 = PFR off |
-> The balanced grid is the cause on both paths; keeping the 4096 grid while releasing the pools is exact. So the AR
   default since 445e301 (PFR balanced, flipped on 8,474 / 65,000 tokens, whose tails were 282 / 3,560 rows) is NOT
   exact on small-tail prompts. Prefill on write16 (AR): balanced 18.9 s, off 27.4 s, keep 28.3 s: keep gives up
   balanced's speed and keeps its pool release. NewIdeas' decision: AR default -> keep once verified (runner_fx);
   balanced becomes a non-exact option for Kamen with its speed numbers. (The first AR arms crashed at load: the harness
   had set the MTP env in AR mode; fixed and re-run.)
R-41 part 3, the rebuilt pieces (16:31-16:52; data/fx/; runner_fx.sh on the candidate: B12_PFR_GRID keep, MTP PFR only
above 4096 tokens or to resize): PASS.
AR, PFR-keep (pfr_ab, KV hash on):
| tokens | KV / logits / sha32 | = | prefill (keep / balanced / off) | prefill-phase peak (keep / balanced / off) | lifetime peak |
|---|---|---|---|---|---|
| 8,474 | ecd389371d / c57fe16fd1 / 142abd5d7e | r8 | 28.2 / 18.9 / 27.5 s | 9.46 / 9.27 / 11.05 GB | 9.92 GB |
| 65,000 | 2227a92414 / 4dcabb778f / dc67a9270b | r64 | 269.5 / 208.1 / 265.1 s | 11.25 / 11.08 / 13.54 GB | 12.00 GB (with the hash) |
MTP, shipped (fold off, v3l) vs fold-exact (MOV + PFR-keep + rows target, C_m 51, v3m), one process, 128 tokens:
| prompt | tokens | every hash (logits / sha32 / sha128 / kv / head) | lifetime peak shipped -> fold-exact | warm tok/s |
|---|---|---|---|---|
| p2k | 2,617 | equal | 10.93 -> 11.09 GB | 14.40 / 14.39 |
| p8k | 8,474 | equal | 11.73 -> 11.09 GB | 17.29 / 17.27 |
| p16k | 15,723 | equal | 12.07 -> 11.09 GB | 15.06 / 15.01 |
| p32k | 31,653 | equal | 13.03 -> 11.46 GB | 14.71 / 14.71 |
| write16 (8,276, 64 tokens) | | = suite OFF | | |
-> fold-exact PASSES R-41's line: bitwise on write16 and the four long prompts, every lifetime peak <= 11.95 GB (max
   11.46; shipped passes 11.95 from 16K and reaches 13.03 at 32K). No speed change (C_m 51, the 4096 grid). AR PFR-keep is
   exact at both old lengths with nearly balanced's prefill memory, at PFR-off's prefill time. Defaults to change
   (NewIdeas' decisions 1 and 3): B12_PFR_GRID keep, B12_FOLD on with its exact pieces only (ALLWAIT, C_m per prompt
   opt-in), engine v3m (bitwise = v3l; adds drain()). A no-switch check of the new defaults runs first (runner_def).

### Fold-fast pre-flight for the Eval suite's b12llff (Wed 20:3x; lines written before the run; runner_ff.sh, data/ff/)
The suite's confirmation (after R-40) runs three variants from my envs: old shipped, fold-exact (the defaults) and
fold-fast (B12_FOLD_AW=1 B12_FOLD_CM=1 B12_PFR_GRID=balanced). Fold-fast has not run on the rebuilt bt_b12 (part 3 checked
fold-exact only), so one short check on the LIVE code first:
- ff_mtp (p2k, p8k, p16k, p32k in one process, 256 tokens). PASS: rc 0; p2k / p8k every hash = the regression's
  (p2k 98c6bc1c0d / 82995cdb4e / a0774ec635 / 37028a6545 / 8563204e11; p8k 61b2520e58 / 0b0d91cd43 / ed67050746 /
  5a760388d6 / f59755cf27); C_m 64 / 60 / 56 / 47 (the fold points at 2,617 / 8,474 / 15,723 / 31,653 tokens); ALLWAIT
  calls > 0 on every prompt. Peaks recorded only (the 11.95 line is the suite's verdict).
- ff_w16 (write16, same env): = the suite's ON row. ff_w16k (grid keep, ALLWAIT + C_m): = the suite's OFF row.
- KILL (the suite holds b12llff until fixed): a crash, a p2k / p8k hash mismatch, C_m not resized, ALLWAIT 0.
Result (20:30-20:54, live bt_b12 56eadbc / 31b63af code): PASS on every line.
| prompt | tokens | C_m | ALLWAIT calls | hashes (256 tokens) | prefill fold-fast / fold-exact | lifetime peak |
|---|---|---|---|---|---|---|
| p2k | 2,617 | 64 | 18 | = the regression's (all five) | 10.5 / 10.1 s (resize PFR) | 11.65 GB (p2k decode) |
| p8k | 8,474 | 60 | 16 | = the regression's (all five) | 22.2 / ~30 s | 11.65 |
| p16k | 15,723 | 56 | 76 | sha32 a3b8a91daf, logits 1a402caa88 = shipped's; 256-token 4d1dea2be1 (no ref) | 38.7 / 54.4 s | 11.65 |
| p32k | 31,653 | 47 | 29 | sha32 149c35ee95, logits 2b3bba2c3f = shipped's; 256-token b2f310eae0 (no ref) | 87.3 / 120.8 s | 11.65 |
write16: balanced grid = the suite's ON row (sha 30cc46ce3d, prefill 21.8 s); grid keep = OFF (5562a6070d, 31.0 s).
-> b12llff is safe to run: on the rebuilt code it reproduces the old fold-on (ON tokens, Fit12's AW hashes), resizes C_m
   per prompt, engages ALLWAIT, and its lifetime peak over 2.6K -> 32K in one process is 11.65 GB (set by C_m 64 at
   2.6K). Its prefill is -28..-29 % at 16-32K. Told EvalSuite (peer say 20:5x).

### R-45 fix check (Thu 01:5x; lines written before the run; runner_gc.sh, data/gcfix/)
Fit12's R-45 (00:42, PASS (b)): the per-request creep of R-41C is cyclic garbage. mtplx's _full_vocab_head builds every
FR-Spec draft head in a local class that super()'s __class__ cell keeps in a cycle, and its closure holds the pruned head
(0.095 GB) until a gen-2 GC. Fix (exact, memory only): B12_FRS_NOCYCLE (default 1) wraps _full_vocab_head and empties the
closure's head / ids cells once built (only __init__ reads them; the methods use self.*). Candidate scripts/_bt_b12_gc.py =
live c3f957cba6 + the wrapper; the live file stays frozen for R-41C until part B ends (~12:00).
- gc_fx (Fit12's harness, scripts/creep_fix.py = fit_creep.py + BT_DIR; fold-exact; R-41C's first 6 math rows in one
  process). PASS: tokens = the suite's b12llfx rows on all 6; gc finds no mx garbage on rows 2-6 (<= 0.01 GB each;
  unfixed 0.095, Fit12 c1); MLX active at a row end <= 9.34 GB (unfixed 9.427); row maxima <= Fit12 c1's + 0.02
  (c1: 10.640 / 10.753 / 10.830 / 10.648 / 10.684 / 10.764, with gc after every row). KILL: a token differs, or >= 0.05 GB
  of mx garbage on any row 2-6.
- gc_mtp (fold_check p2k, p8k, 128 tokens): every hash = def_mtp's, the draft head's own output (logits) included.
Result (01:50-01:57, one quiet lock per item; data/gcfix/): PASS on every line. (gc_fx's first try stopped at the
harness's own sha check, 5 s, nothing measured; creep_fix.py now lets a BT_DIR candidate through and runner_gc2 re-ran it.)
| row (R-41C math, one process) | tokens = b12llfx | row max fixed / Fit12 c1 unfixed | MLX active at row end (unfixed) | gc: mx garbage (unfixed) |
|---|---|---|---|---|
| 1 prealgebra/1640 (904 tok) | yes | 10.640 / 10.640 | 9.332 (9.332) | 0.000 (0.000) |
| 2 intermediate_algebra/1422 (1,536) | yes | 10.659 / 10.753 | 9.332 (9.427) | 0.000 (0.095) |
| 3 algebra/694 (614) | yes | 10.732 / 10.830 | 9.332 (9.427) | 0.000 (0.095) |
| 4 intermediate_algebra/101 (1,536) | yes | 10.546 / 10.648 | 9.332 (9.427) | 0.000 (0.095) |
| 5 precalculus/323 (1,186) | yes | 10.590 / 10.684 | 9.332 (9.427) | 0.000 (0.095) |
| 6 number_theory/427 (856) | yes | 10.724 / 10.764 | 9.332 (9.427) | 0.000 (0.095) |
gc still finds the class's own 15 objects per row (cell 5, function 4, tuple 4, type 1; no arrays). Creep over rows 1-6:
0.107 GB (settled footprint 9.851 -> 9.958; hot cache +0.023) vs the suite's 0.631 over the same row starts.
gc_mtp: p2k 98c6bc1c0d / 82995cdb4e / 78a72d3ccb / fb2087cab5 / 76ecab8796 and p8k 61b2520e58 / 0b0d91cd43 / 0643da7828 /
f25e0ebd31 / b6d20a341e = def_mtp's (the draft head's output, logits, included); lifetime peak 11.01 GB (def_mtp 11.13).
-> The fix is exact and removes the per-request head garbage without any gc. It goes live only after R-41C part B ends
   (EvalSuite pins bt_b12 c3f957cba6 until then). Left: Fit12's R-45R residual (~18 MB/row of native mallocs; not the v3m
   engine by its source: per-layer maps are keyed by expert, queues drain per call, the trace vector grows only with
   BIGMOE_TRACE).

### FFX: fold-fast + the R-45 fix with C_m capped at 60 (Thu 15:5x; lines written before the run; runner_ffx.sh, data/ffx/)
For NewIdeas' R-41C part B call (B stopped at 45/239 rows at the full stop): keep the c3f957cba6 pin (B fails memory
whatever its scores: 13.20 GB), or move B to the fix with a C_m cap. Fit12 R-45F (gc between rows = the fix's effect):
fold-fast peaks 12.212 GB on 8 math rows at C_m 64, so C_m 60 (4 steps x 0.103 GB) should give ~11.80. One process, R-41C's
first 8 math rows of b12llff, Fit12's harness (creep_fix.py, F12_ARM=ff, F12_CM=2700:60,8600:60,16500:56,32800:47).
- PASS: tokens = the suite's b12llff rows on all 8 (C_m is bits-neutral under ALLWAIT; the fix is exact) and lifetime
  peak <= 11.85 GB (0.10 GB left for the suite's 16-row process). BETWEEN: 11.85-11.95 (cap 59 then).
- KILL: a token differs, or the lifetime peak > 11.95 GB.
Result (16:04-16:12, one quiet lock; data/ffx/): PASS. Tokens = the suite's b12llff rows on all 8 (C_m 60 vs the suite's
64: bits-neutral under ALLWAIT, as expected). Lifetime peak 11.680 GB (row maxima 11.049 / 11.631 / 11.668 / 11.466 /
11.520 / 11.647 / 11.612 / 11.628; MLX active 10.259 at every row end = 9.332 + 9 C_m steps). Creep over 8 rows 0.227 GB
(settled 10.778 -> 10.938, plus one 0.067 gc drop at row 5; the FR-Spec heads are gone) vs the suite's 0.821 unfixed.
Speed is NOT comparable: decode 482 s here vs 438 s for Fit12's R-45F (C_m 64, same harness, Wed 00:4x on an idle Mac);
this ran with Kamen's game on the GPU (Cellar live, WindowServer 66 %). R-41D's paired runs give the speed.
-> b12llff2 for R-41D = b12llff's env with B12_FOLD_CM=2700:60,8600:60,16500:56,32800:47 and B12_FRS_NOCYCLE=1.

### R-48 M01-12: BigMoE's M-01 records on the 12 GB line (Thu 17:3x; lines written before the run; runner_m01.sh, data/m01/)
NewIdeas' R-48 (Thu 17:2x): an exact port of BigMoE's R-44 (a) M-01 (fixed-width re-pack of the per-group scale / bias
fields; at 24 GB every sha equal, rusage bytes/tok -4.8 %, tok/s +2.0 %, ~/BigMoE/data/b44/ab.txt). The 12 GB line streams
the same records3 files (MSB part 2,162,688 B, LSB part 770,048 B), so BigMoE's records3m (2,048,000 / 720,896 B, 68 GB,
every expert verified by its converter) serves it as is: no second copy. Engine v3n = v3m + BigMoE's unpack (the m01_*
block verbatim; BIGMOE_M01=1 -> records3m; v3m stays pinned for R-41D). Expected per miss: MSB -5.3 %, LSB -6.4 %, full
record (prefill) -5.6 %; the streamed MTP layer (mtp.safetensors) and the embedding reads do not change, so decode
bytes/token ~ -5 %.
Pre-checks (CPU / SSD): (1) scripts/test_v3n_m01.py, L0 / L23 / L47 x 96 experts: read_rebuilt on records3m = the
safetensors 4-bit pieces and the MSB / LSB part pools = records3's part bytes (BIGMOE_M01=1), and the same on records3 with
M-01 off (--r3). (2) Page cache: no records3 / records3m layer file > 1 % resident (mincore) before the A/B (a cached arm
reads fewer SSD bytes; BigMoE's note 16:39).
Screen: one process per arm, fold-exact defaults (fold_check's env), in order p2k and p8k (fold_check's method:
b12_prefill_state + spec125.generate 3:3:0, 256 tokens) then write16 (w16_bisect's method, 256 tokens, stops at EOS). Arms
A = B12_ENGINE=v3m, S = B12_ENGINE=v3n BIGMOE_M01=1, equal pools (C_m 51 / C_l 20 at load, the fold's C_m per prompt);
order A1 S1 S2 A2 in one quiet lock. Per prompt: hashes (first-step logits from the state, sha of the tokens, decode-end
trunk kv and MTP head cache), rusage disk bytes read in prefill and in decode (ri_diskio_bytesread), decode tok/s (all
tokens and tokens 33+), lifetime peak.
- PASS: every hash equal in all 4 runs AND decode SSD bytes per token (rusage; the 3 prompts' decode bytes / decoded
  tokens, S1 + S2 vs A1 + A2) -2.5 % or better AND lifetime peak not higher (max S <= max A + 0.02 GB, the A1-A2 noise
  allowance) -> the 12 GB default on NewIdeas' word, after R-41D.
- KILL: any hash differs, or a decode bytes/token saving under 1.5 %.
- BETWEEN: report it; tok/s decides.
- INVALID (re-run): A1 vs A2 decode bytes/token differ by more than 1 %, or the arms' C_m / C_l differ.
Also reported: whole-process and prefill bytes/token (prefill expected -5.6 %), write16's tokens vs the suite's b12llfoff row.
Result (17:48-17:56, one quiet lock; runner_m01.sh, data/m01/): **PASS**. v3n's ext built (5 s); loader test PASS bitwise
on L0 / L23 / L47 x 96 experts with M-01 on (records3m: read_rebuilt = the safetensors 4-bit pieces, MSB / LSB part pools
= records3's part bytes) and off (--r3, the same on records3); page cache 0.00 % resident in both dirs.
| run | decode MB/tok | prefill KB / prompt tok | decode tok/s | lifetime peak GB |
|---|---|---|---|---|
| A1 (v3m, records3) | 457.16 | 22,494 | 16.06 | 10.992 |
| S1 (v3n + M-01, records3m) | 432.81 | 21,074 | 16.47 | 10.981 |
| S2 | 432.81 | 21,075 | 16.50 | 10.982 |
| A2 | 456.57 | 22,186 | 15.81 | 10.996 |
Every hash equal in all 4 runs: p2k 98c6bc1c0d / 82995cdb4e / a0774ec635 / 37028a6545 / 8563204e11 and p8k 61b2520e58 /
0b0d91cd43 / ed67050746 / 5a760388d6 / f59755cf27 (= the fold-exact regression's), write16 418633931b / a1789d46bf /
df432791d6 / 368d4f4e01 / 5c58dda19b (its 256 tokens = the suite's b12llfoff row); C_m 51 / C_l 20 in every run.
A1 vs A2 decode bytes/token noise 0.13 %. S vs A: decode bytes/token **-5.26 %** (per prompt p2k -5.29, p8k -5.26, write16
-5.24 %: S1 = S2 to the byte), prefill bytes/token -5.66 % (full record 2,768,896 / 2,932,736 = -5.6 %), decode tok/s
+3.45 % (warm p2k +2.7 %, write16 +2.4 %, p8k +5.8 % with A2's slow p8k, 15.57 warm), lifetime peak 10.982 vs 10.996 GB.
-> PASS: the 12 GB default on NewIdeas' word AFTER R-41D (v3m and bt_b12 stay pinned for it). Adoption = a bt_b12 default
   change (B12_ENGINE=v3n + BIGMOE_M01=1); records3 stays on disk until the 12 GB line and the suite stop reading it.

### Post-R-41D bt_b12 candidate check (NewIdeas Thu 18:3x item 1; lines written before the run; runner_cand.sh, data/cand/)
Candidate 23b42972ef (scripts/_bt_b12_m01.py; BT_DIR copy in the session scratchpad btm01/) = ce6d4c63d2 (live c3f957cba6
+ the R-45 fix, B12_FRS_NOCYCLE default 1) + the M-01 default: engine v3n, load_b12 sets BIGMOE_M01=1 unless B12_M01=0
(any other engine: unset), BIGMOE_M01 joins _EXT_KEYS. It goes live only after R-41D passes (NewIdeas' word). Run at its
own defaults (no B12_ENGINE / fold / M-01 keys in the env; scripts/cand_check.py):
- cand_hash: p2k, p8k, write16 (256 tokens) in one process: every hash (first-step logits, sha32, sha256, decode-end kv and
  MTP head) = R-48's (= the fold-exact regression's) and write16's tokens = the suite's b12llfx row.
- cand_rows: R-41C's first 16 math rows in ONE process, each as the suite's decode._gen_mtp runs it (no gc, no settle):
  tokens = b12llfx on all 16; lifetime peak <= 11.85 GB (the suite's unfixed b12llfx process: 11.511 GB after these rows).
  ~15 min of decode, so the process idles between two quiet locks (<= 15 min each) without GPU work.
- PASS: both. FAIL: any hash or token differs, or the lifetime peak > 11.85 GB.

### R-49 POM12: POM and POM + LP1 priced at 12 GB (NewIdeas Thu 18:3x; NON-exact; CPU price screen; lines written first)
MoonShots' M-03 POM (a miss used only at rank >= R2 reads 2 planes; a later rank < R2 use reads plane 3) and M-07 LP1 (a
rank-1-only LSB miss reads nothing) at the 12 GB pools (C_m 51 / C_l 20), with BigMoE's real layouts (903560e): full MSB
miss 2,048,000 B (records3m), 2-plane miss 1,425,408 (records3p seg1), plane-3 upgrade 704,512 (seg2), LSB 720,896
(records3m). scripts/pom12.py (MoonShots' pom24 / lpom24 replay rules) with the 12 GB engine's own rules: every row's
top-10 used and read, one call per layer per verify pass; LRU eviction (v3m / v3n) = the line's verdict; strict LFU
(BigMoE's engine) shown too. Variants: R2 6 / 5 / 4, each with and without LP1; LP1 alone. Traces: the 12 GB decode's own
verify passes (scripts/r49_capture.py, live bt_b12 at its defaults: p8k up to 512 tokens, agent bash17 / edit00 / text23
/ write16, math prealgebra/1640; first 10 passes of each warm the pools), captured in a quiet lock; a first read on the
expA traces (800 AR tokens, synthetic 4-row passes keeping 3) meanwhile. Quality by MoonShots' formula: predicted KLD =
v3 + f_e x gap(R2) + f_1 x (plan4333333333 - v3).
- PASS: some variant -15 % or better SSD bytes / token vs fold-exact + M-01 (R2 10, no LP1, same policy) with predicted
  KLD inside 1.10 x ar4 on all four parts -> MoonShots' harness confirms at that f (one lock), then an engine port behind
  a switch, then the strict test and Kamen.
- KILL: under -8 % for every variant inside the gate.
Also reported: the replay's baseline vs the measured decode rusage bytes / token (R-48's S arm 432.8 MB, which also holds
the MTP layer's and the embedding's reads), and LRU vs LFU at equal pools (an exact policy change, if it saves bytes).
R-49 AMENDMENT (NewIdeas 19:3x, before any 12 GB-trace result; the expA first read above came out 18:54): judge by SSD BUSY
TIME per token, not bytes (BigMoE's b45 at 24 GB: POM -19 % bytes but SSD busy -6.5 %, tok/s -1.9 %: the plane-3 upgrades
add reads, and 2-plane reads / upgrades pay a CPU rebuild). pom12.py models busy per (pass, layer) call, since the 12 GB
engine reads demand bursts with no prefetch: T0 0.2 ms + bytes / 14.0 GB/s + 4 us per read + (2-plane reads + upgrades) x
0.25 ms / 6 threads (MoonShots' ssd_qd; BigMoE's 0.084 ms rebuild per projection). Same thresholds, on modeled busy:
PASS -15 % or better inside the gate, KILL under -8 % for every variant inside the gate. The same model at 24 GB pools
(C_m 147 / C_l 46) is checked against b45's measured -6.5 %, as a calibration note.
R-49 RESULT (19:39 traces; replays 18:54-19:41; data/r49/: pom12_r49.out = the verdict run, pom12_expA*.out, the 6
captured traces): **KILL on modeled SSD busy time** (the amended line). Capture (one 10-min lock 19:33-19:38, live bt_b12 at
its defaults): p8k 512 tokens (173 passes, 2.96 kept / pass), agent bash17 / edit00 / text23 / write16 (114-128 tokens,
2.6-3.8 kept / pass), math prealgebra/1640 (904 tokens, 3.53 / pass); every pass 4 rows. Replay check: the LRU baseline
reads 432.8 MB / token on these traces (p8k 436) vs R-48's measured rusage 432.8 (p8k 450) on its own prompts: the replay
reproduces the 12 GB engine's bytes. LRU (the 12 GB engine), vs fold-exact + M-01:
| variant | bytes / token | busy, rebuild 0.25 ms | busy, 0.10 ms (calibrated) | busy, rebuild free | f_e / f_1 | predicted KLD vs gate |
|---|---|---|---|---|---|---|
| LP1 alone | -4.0 % | -3.8 % | -3.8 % | -3.8 % | - / 44 % | inside |
| R2 6 | -9.8 % | +4.6 % | -3.2 % | -8.4 % | 52 % / - | inside (code 87 % of the gate) |
| R2 5 | -12.4 % | +4.8 % | -4.5 % | -10.8 % | 54 % | OVER (code, ifeval) |
| R2 4 | -15.1 % | +4.6 % | -6.1 % | -13.2 % | 57 % | OVER (3 parts) |
| R2 6 + LP1 | -13.7 % | +0.8 % | -7.0 % | -12.2 % | 52 / 44 % | at the edge: code 0.00253 = gate |
| R2 5 + LP1 / R2 4 + LP1 | -16.4 / -19.1 % | +1.0 / +0.8 % | -8.4 / -9.9 % | -14.6 / -17.0 % | | OVER |
Calibration of the rebuild term: the same model at 24 GB pools (C_m 147 / C_l 46, expA, LFU, POM R2 4) gives busy -12.6 %
with a free rebuild and +3.2 % at 0.25 ms; BigMoE's b45 measured -6.5 % => ~0.10 ms per 2-plane read / upgrade (6 threads).
Why it fails at 12 GB: misses are ~2.5x the 24 GB rate, so half of the rank >= 6 gate energy sits at 2 planes (24 GB:
~20 %): the KLD budget is spent by R2 6, and each 2-plane read or upgrade costs a CPU rebuild that eats its byte saving.
Nothing inside the gate reaches -8 % busy at the calibrated cost (only a free rebuild puts R2 6 at -8.4 %). By bytes
alone it would have been BETWEEN (R2 6 -9.8 %). LFU (BigMoE's rule) shown in the file: same picture (best inside at the
calibrated cost: R2 6 + LP1 -8.7 %). The replay's LFU-vs-LRU gap is not a lever (Fit12 F1 engine A/B, 10-24: -2 %).

R-41D (EvalSuite c095ffe, 21:56) + LIVE SWAP (22:57:44, NewIdeas' word 22:08): b12llfx2 (fold-exact + R-45 fix) PASS: tokens
45/45 = b12llfx, lifetime peaks agent 11.52 / math 10.80 GB (line 11.95). b12llff2 (fold-fast + fix + C_m 60): tokens 45/45
but agent 12.47 GB (write14's +0.7 GB prompt-reading spike at n_prompt 1945 on top of creep) -> decision 5 stays open.
LIVE scripts/bt_b12.py = 23b42972ef (= ce6d4c63d2 + engine v3n + BIGMOE_M01=1, i.e. records3m; R-48 hashes + cand_hash
PASS). Old path: B12_ENGINE=v3m or B12_M01=0.

R-54 ECOSPEC12 LINES (NewIdeas Thu 20:47, QUEUE R-54; written before any result; EXACT by construction: only the number of
drafted rows per check changes, verification unchanged). In a 4-row MTP check every row's experts are read, but rows
after the first rejection are wasted reads (R-49 traces: 2.6-3.8 kept per check). Screen (CPU, scripts/eco12.py on R-49's
traces with pom12's LRU replay and busy model at the shipped 12 GB pools C_m 51 / C_l 20, v3n + M-01): per pass and row,
the marginal reads (not resident at the call's start, not already needed by an earlier row) and kept or not; (1) ORACLE:
each pass keeps only its first `kept` rows, replayed (so the skipped reads' cache effects are kept). KILL: oracle saves
< 6 % of modeled SSD busy ms per token. PASS: >= 12 % -> step 2 (one short capture of the MTP head's p1 per drafted token;
a policy dropping row j when p1(j) < tau and its marginal reads > m must save >= 6 % with tokens per check down by less
than the time saved -> an engine A/B). Between: report (BigMoE runs the same at 24 GB). Fixed 3- / 2-row horizons shown
as an approximation only.
R-54 RESULT (Fri 16:21, scripts/eco12.py, 4 s CPU; data/r49/eco12_51_20.txt, eco12_147_46.txt): **BETWEEN** (oracle -9.7 %
modeled SSD busy ms / token; KILL < 6, PASS >= 12). Replay baseline = 432.8 MB / token (= R-48's measured rusage).
| 12 GB pools (C_m 51 / C_l 20, LRU) | busy ms / token | MB / token | tokens |
|---|---|---|---|
| today (4 rows) | 34.79 | 432.8 | 1687 |
| ORACLE (cut after the last kept row) | 31.42 (-9.7 %) | 387.6 (-10.5 %) | 1687 |
| fixed 3 rows (approx.) | 36.16 (+3.9 %) | 441.7 (+2.1 %) | 1352 |
| fixed 2 rows (approx.) | 38.79 (+11.5 %) | 458.0 (+5.8 %) | 962 |
Per prompt (oracle): write16 -22.6 % (2.61 kept / pass), bash17 -13.5 %, p8k -12.3 %, math -7.4 % (3.53), edit00 -6.3 %,
text23 -2.8 % (3.80). Rows hardly share misses: every row adds ~3.4 MSB + 0.75 LSB new reads per layer call, kept or
rejected (rows 1-3 kept in 87 / 76 / 65 % of passes); rejected rows hold 16.1 % of today's bytes, but the replay saves only
10.5 %: about a third of the "wasted" reads are used by a later pass (an accidental prefetch). Static horizons lose (fewer
tokens per pass cost more than the reads they save). Same replay at 24 GB pools (C_m 147 / C_l 46, LRU stand-in, not
BigMoE's engine): oracle -9.1 % busy, -10.2 % bytes (16.72 -> 15.19 ms / token), also BETWEEN. Step 2 (p1 capture) not
opened (PASS only); NewIdeas' line: report, BigMoE runs it at 24 GB.

R-45 v3o LINES (NewIdeas Thu 19:52 GO, QUEUE R-45; written before the run): v3o = v3n + read_expert's bounce buffer in a
thread_local holder that frees it at thread exit (Fit12 R-45T: 87 % of the per-request growth = that bounce, leaked by each
GCD-retired worker, ~7 x 2.8 MiB per request). Candidate fd405861fa (scripts/_bt_b12_v3o.py = 23b42972ef with engine v3o,
M-01 on for v3n / v3o). One 15-min lock (scripts/runner_v3o.sh -> v3o_arms.sh): build v3o/ext/build; hashes by
cand_check (p2k / p8k / write16 vs R-48's = v3n's, write16 = b12llfx); creep by scripts/v3o_creep.py (Fit12's fit_creep.py
copied): 6 agent requests (write14, edit10, write13, twice; Fit12's ask: agent rows), a 10 s idle gap + gc after each,
arm v3o then arm v3n (live 23b42972ef) if >= 300 s remain. PASS: every hash = v3n AND after-gc growth G <= 3 MB per
request (v3o arm) -> v3o joins the 12 GB default on NewIdeas' word. KILL: G >= 15 MB. Also reported: vmmap MALLOC_LARGE per
request, and the v3n arm's G on the same rows (whether agent rows' creep is the same leak).
R-45 v3o RESULT (Fri 16:42-16:52, one quiet lock; data/v3o/: summary.txt, v3o_hash.txt, v3o_creep.txt + vmmap r1 / r6):
**hashes PASS, creep BETWEEN (G +3.6 MB per request; PASS <= 3, KILL >= 15)**. Build 6 s. cand_check on the v3o candidate
fd405861fa at its defaults (engine v3o, m01_on, records3m): p2k / p8k / write16 = R-48's hashes (= v3n's), write16 =
b12llfx's tokens; lifetime peak 11.061 GB. Creep, v3o arm, 6 agent requests (write14, edit10, write13, twice), 10 s idle
gap + gc after each: tokens = b12llfx on all 6; after-gc footprint 9.795 -> 9.813 GB (per row +3 +5 +3 +2 +5 MB);
vmmap MALLOC_LARGE 64.9 MiB / 7 regions -> 59.6 MiB / 5 regions (-1.1 MiB per request; v3m / v3n in R-45S / T: +7
regions = +19.6 MiB per request): the bounce leak is gone. The residue is not MALLOC_LARGE, not gc-tracked objects
(102,948 flat), not the n-gram hot cache (0.005 GB, flat from row 3; rows 4-6 repeat rows 1-3 and still add +2..+5 MB),
MLX active flat at 9.332 GB: likely small malloc zones (R-45T's 'rest': ~1.5 MiB / row of pymalloc arenas). Row 1's gc
still frees the load-time _FullVocabDraftHead (0.095 GB, Fit12's note). The v3n arm on the same rows was skipped (the
hash step took 7 min, not ~2: under 300 s left); its leak is known per request from R-45S / T.
LIVE SWAP (Fri 17:19:39, NewIdeas' word 16:54): scripts/bt_b12.py = fd405861fa (= 23b42972ef + engine v3o: the bounce-buffer
leak fix; M-01 on, records3m). Hash-exact (R-45 v3o, a763814). Older paths: B12_ENGINE=v3n (23b42972ef's engine) or v3m.

R-41E PFRMIN LINES (NewIdeas Fri 16:54 GO, after Breakthrough's 16:3x data: write14's +0.72 GB R-41D spike = the
short-prompt NO-PFR path; written before the run): candidate 9c3057ce62 (scripts/_bt_b12_pfrmin.py) = live fd405861fa +
B12_PFR_MIN, default 512: PFR (pools released while the prompt is read, re-created empty after) on prefills above 512
tokens instead of 4096, plus the fold's C_m-change rule as before. Exact by R-41's grid keep (below 4096 the MoE still
runs once over the prompt). One lock (scripts/runner_pfrmin.sh -> pfrmin_arms.sh): cand_check hashes at the candidate's
defaults, then agent/write14 + agent/edit10 in one process per arm, B12_PFR_MIN 4096 vs 512 (v3o_creep.py, 10 s gaps).
PASS: p2k / p8k / write16 = R-48's hashes with p2k taking PFR -> the fold-exact default on NewIdeas' word, then
b12llff3 (fold-fast + v3o + PFR_MIN 512 at today's C_m points) in the Eval suite's agent process for decision 5.
KILL: any hash differs. Reported, not gates: each row's rise (prefill max and row max minus row start; R-41D: +0.43
edit10 / +0.72 write14 without PFR, +0.06-0.25 with), lifetime peaks, and the prefill-time cost (the re-create ~0.2 s).
R-41E PFRMIN RESULT (Fri 17:36-17:40, one quiet lock; data/pfrmin/): **hashes PASS** (p2k / p8k / write16 = R-48's; p2k
now takes PFR: 2,618-token prefill 10.13 s; the check process's lifetime peak 10.502 GB vs 11.061 in the v3o check, whose
p2k ran without PFR). Spike A/B, one process per arm, write14 then edit10, 10 s gaps, C_m 51 (tokens = b12llfx in all 4):
| B12_PFR_MIN | row | prefill | PFR | prefill rise | row max rise | row time (prefill + decode) |
|---|---|---|---|---|---|---|
| 4096 (old) | write14 (1,945) | 9.72 s | no | +1.226 GB | +1.226 GB | 18.32 s |
| 4096 (old) | edit10 (1,565) | 9.23 s | no | +0.936 GB | +0.936 GB | 15.46 s |
| 512 | write14 | 10.40 s | yes | +0.028 GB | +0.527 GB | 20.06 s (+9.5 %) |
| 512 | edit10 | 12.86 s | yes | +0.149 GB | +0.410 GB | 20.65 s (+34 %) |
Process lifetime peak 10.914 -> 10.209 GB (-0.705). COST, larger than the ~0.2 s re-create in the code's comment: the
pools come back EMPTY after PFR, so each short row's decode starts cold (write14 decode 8.60 -> 9.66 s, edit10 6.23 ->
7.79 s, i.e. -11 % / -20 % tok/s) and edit10's prefill lost write14's resident experts (+3.6 s; one sample each). My 16:3x
"no visible time cost" (from R-41C/D rows whose C_m differed, 64 vs 60) was wrong; corrected to NewIdeas before any swap.
Not swapped live: the time cost goes back to NewIdeas (options: PFR_MIN 512 only in fold-fast's b12llff3, where the line
binds; or a budgeted rule: PFR on a short prompt only when footprint + its spike would cross the line).

R-41F REFILL LINES (NewIdeas Fri 17:5x GO for option (d); written before the run): candidate b9d5b491ad
(scripts/_bt_b12_refill.py) = 9c3057ce62 (PFR_MIN 512) + B12_PFR_REFILL, default 1: b12_prefill's moe_split keeps each
tier layer's routing of the prompt's last MoE chunk (a last chunk under 2,048 rows keeps the previous chunk's tail);
after PFR re-creates the pools, each layer's base pool takes its C_m most-used experts (pair counts, ties by id) and the
LSB sidecar its C_l most-used at ranks 0-1, by ext.ensure (parallel reads per layer, before the decode starts), most used
last (the LRU front). Exact by construction (cache contents only). One lock (scripts/runner_refill.sh ->
refill_arms.sh -> refill_sum.py): cand_check hashes at the candidate's defaults (p2k / p8k / write16, all three with PFR
+ refill); then arms in one process each on agent/write14, agent/edit10, agent/text23 (8,582 tokens), 10 s gaps:
A = today's live behaviour (B12_PFR_MIN 4096, no refill), B = the candidate, A2 = A again if >= 240 s are left.
PASS: every hash = R-48's AND write14 and edit10 in B within +3 % of A's row time (prefill + decode) AND text23's decode
tok/s in B >= +5 % over A's (today's PFR) -> then b12llff3 = fold-fast + v3o + PFR_MIN 512 + refill for decision 5.
KILL: neither time line holds. Between: one holds (report).
R-41F REFILL RESULT (Fri 18:36-18:44, one quiet lock; data/refill/): **KILL** (hashes PASS, neither time line). The refill
works as built: p2k / p8k / write16 = R-48's hashes with PFR + refill, 3,408 experts in 0.48-0.53 s per prefill; tokens =
b12llfx in all 9 rows. But B is slower than A everywhere (one process per arm; row = prefill + decode):
| row | A (PFR_MIN 4096) | B (512 + refill) | A2 (= A) |
|---|---|---|---|
| write14 row / decode | 18.21 s / 14.94 tok/s | 21.59 s / 12.62 | 21.84 s / 12.88 |
| edit10 row / decode | 15.11 s / 16.60 | 22.14 s / 13.07 | 15.96 s / 16.06 |
| text23 (8,582) prefill / decode | 33.33 s / 18.69 tok/s | 56.34 s / 16.29 | 39.61 s / 18.60 |
Lines: short rows +7.8 % (write14) / +42.5 % (edit10) vs A's mean (<= +3 %); text23 decode -12.6 % (>= +5 %). Noise is
large (A vs A2: write14 row +19.9 %, text23 row +16 %; the Mac sat at 13.7 GB swap / 8.5 GB compressor all along), but
text23's decode was steady in A / A2 (18.69 / 18.60) and B is below both. Why a fuller cache would decode slower is
unexplained (by LRU, refilled pools can only add hits): external load during B (all its prefills are slow too), or the
refill's 6.5 GB of buffered reads pushing hot file pages (the n-gram table, the streamed embedding rows) out of a page
cache squeezed by the Mac's memory pressure. Not chased (one screen per idea). Not swapped; PFR_MIN 512 alone stays the
R-41E result (exact, -0.7 GB on short rows, +9.5 / +34 % time on them).
LIVE SWAP (Fri 18:58:54, NewIdeas' decision-5 word 18:5x): scripts/bt_b12.py = 677c6558b6 = fd405861fa + B12_PFR_MIN read
from the environment, default 4096 (the same comparison as fd405861fa: default behaviour unchanged by construction; the
512 path is R-41E's hash-checked 9c3057ce62 rule). 512 is set only in b12llff3 (fold-fast + v3o + PFR_MIN 512, today's
C_m points, no refill) for the Eval suite's agent-process re-test (PASS: 29 / 29 = b12llff and lifetime <= 11.95 GB);
the short-row cost (+9.5 / +34 %, R-41E) goes into decision 5's description for Kamen.

> (a quote from the project owner, kept private)
LIVE SWAP 11:22:49: scripts/bt_b12.py = cf8bd7840a = 677c6558b6 + B12_MODE (default fast = b12llff3's fold-fast: FOLD_AW 1,
C_m per prompt at the FFX points 2700:60,8600:60,16500:56,32800:47, PFR_MIN 512, PFR_GRID balanced; exact = fold-exact,
the bit-exact mode: AW 0, CM 0, PFR_MIN 4096, grid keep; each variable still overrides) + engine v3p (= v3o + Fit12's R-53
prefill scratch reuse, exact, -6 % prefill; NewIdeas 00:3x) + MTPLX_NGRAM_HOT_MB default 16 (R-46, exact; NewIdeas 00:5x).
Check (shared GPU, at the candidate's defaults; data/ff/ffdef.txt): agent/edit10 and agent/bash17 tokens = b12llff3's rows
(103 / 114 tokens, both to EOS; engine v3p, mode fast, grid balanced, PFR_MIN 512); lifetime peak 11.28 GB.
EvalSuite's b12llff3 (c345bf1): 29/29 = b12llff, lifetime 11.51 GB, SSD 435.5 MB / token. Not bit-exact by design;
B12_MODE=exact for exact work.

R-60 MTPDROP LINES (NewIdeas Fri 23:1x; FAST INTEGRATION item; written before any run; exact by construction: drafts
only): MTP Head's trained MTP head D-22 b3 (~/TrainLab/ckpt/d22_b3.safetensors; its real runtime: code K15 7.78 -> 10.98,
gsm 6.07 -> 6.86) in bt_b12's MTP path (spec125 3:3:0) at the live defaults (fold-fast, cf8bd7840a). Candidate
68db9798c6 (scripts/_bt_b12_r60.py) = live + B12_MTP_HEAD=<ckpt> (opt-in): the MTP layer's dense parts (fc_hidden,
fc_embedding, q/k/v/o, router, shared expert + gate; 69.1M weights) as bf16 Linear modules (B12_MTP_HEAD_BITS=8: 8-bit
g64), mtp125.install without its training-only NativeMoE wrap (the streamed MoE block stays). Screen on a SHARED GPU
(scripts/r60_arms.sh -> r60_screen.py, one process per arm: stock, d22): p2k / p8k / write16 (256 tokens) + the Eval
suite's first 8 math rows. PASS: tokens identical to the stock arm on every prompt (math rows = b12llff2's) AND tokens
per verify pass +3 % or better -> a quiet confirmation -> default on NewIdeas' word. KILL: under +1 %. Also: decode
tok/s (rough), lifetime peak (<= 11.85 GB with the head's +~0.1 GB).
R-60 RESULT (Sat 19:57-20:21, Kamen's one-hour window, shared GPU, rough; data/r60/): **KILL** (tokens per pass +0.5 %,
line < +1 %) and the exactness premise fails in the default mode. Stock arm stopped by the swap guard after 9 of 11 jobs
(the Mac's swap at its limit); the d22 arm (b3, bf16) ran the same 9 (p2k / p8k / write16 + 6 math rows):
| prompt | stock tok / pass | d22 tok / pass | same tokens |
|---|---|---|---|
| p2k / p8k / write16 | 2.723 / 2.876 / 2.612 | 2.753 / 2.844 / 2.485 | no / no / no |
| math 1640 / 1422 / 694 | 3.531 / 3.657 / 3.529 | 3.587 / 3.666 / 3.529 | yes / no / yes |
| math 101 / 323 / 427 | 3.631 / 3.753 / 3.706 | 3.675 / 3.808 / 3.730 | no / no / no |
Pooled: 3.522 -> 3.541 (+0.5 %). In 3:3:0 (3 drafts) the head's K15 gains barely show. Tokens differ on 7 of 9 prompts:
the live default is fold-fast (decision 5), whose ALLWAIT makes the verify bits depend on the drafted rows, so in fast
mode a drafter change is NOT exact (stock's math rows = b12llff2's, 6 / 6). Drafter changes need B12_MODE=exact for a
bit check, or a same-scores test in fast mode. MEMORY FLAG: both arms reached a lifetime peak of 12.43 / 12.50 GB on
cf8bd7840a (p2k, p8k, write16, then long math decodes; it rose during the math rows: 11.38 -> 12.00 -> 12.31 -> 12.43),
above the 11.95 line, under heavy system swap; b12llff2's 16-row math process was 11.73 (R-41D). Needs a clean check
(e.g. the Eval suite's math process on the live file).

B-1 TWO-PASS VERIFY (new-bet candidate, FAST INTEGRATION rule 4; Sun 04:1x): prior art (2 web searches): no published match for
"a cheap pass of the target sets the exact verify length"; neighbours: AcceptMoE 2608.02989 (self-sized verifier expert
sets, NON-exact, 2.06x under offloading), EcoSpec 2607.12696 (expert-cost-aware selection), ASD 2608.03447 (approximate
verifier), Quasar 2603.01399 (quantized verifier), SPEQ / ML-SpecQD (quantized target as drafter), SpecDec++ (drafter-side
acceptance head). KILLED ON PAPER: the cheap pass still runs the dense part (attention / GDN, ~52 ms of a ~110 ms 45 GB
check, RANKED_45GB.md), so chat (2-3 kept) gains at most ~-8 % (168 -> ~154 ms) and code (~10 kept) loses (~199 ms),
while the drafter-confidence truncation (Leap S45c, published EVICT / EcoSpec) takes the chat rows' bytes without an
extra pass. Fit12's R-62P prices the oracle at 45 GB residency.
LOSSLESS RECORDS screen (Sun 04:2x, CPU 66 s, scripts/lossless_screen.py; lines PASS >= 10 % smaller, KILL < 5 %, sent to
NewIdeas before the run): **BETWEEN, ~8.5 % at the ideal bound -> not worth building now.** 40 experts (layers 0 / 12 /
24 / 36 / 47) of records3: the 3-bit MSB codes (196.6M values) have the histogram 0.029 / 0.047 / 0.114 / 0.202 / 0.246 /
0.199 / 0.109 / 0.054, order-0 entropy 2.716 of 3 bits (-9.5 % on the code planes); bf16 scales / biases 0.722 byte
entropy; real codecs on the raw parts: lzma-6 0.910 (MSB) / 0.916 (LSB), zlib-9 0.932 / 0.938. (The script's 28.5 % line is
WRONG: the index has no LSB piece offsets, so the 1-bit planes went uncounted; they are ~1 bit / bit.) records3m bound by
hand: code planes 1,843,200 B x 0.095 = -174.5 KB, bit planes ~0, the already re-packed scales / biases (311 KB) ~-20 %
-> ~-236 KB of 2,768,896 B = ~-8.5 % per record. Decoding ~7 GB/s of codes for the 12 GB decode would take ~4-7 CPU cores
(ANS / Huffman ~1-2 GB/s per core): too costly for <= 8.5 % fewer SSD bytes.

R-45V v3q LINES (Breakthrough Sun 04:5x, from Fit12's R-45U; NewIdeas told, its usage asleep; written before the run): engine
v3q = v3p + one shared bounce-buffer pool in read_expert (BigMoE 9f6c4f5's design, ext.bounce_stats(); exact: same
bytes read and copied). Candidate 9250688a7a (scripts/_bt_b12_v3q.py = live cf8bd7840a with engine v3q). One shared-lock
run (scripts/v3q_arms.sh): build; cand_check hashes with B12_MODE=exact (= R-48's); then fast mode (the live defaults),
one process per arm on agent write14 / edit10 / write13 x 2 with 10 s gaps (tokens vs b12llff3): arm v3p (live) vs v3q.
PASS: every hash = R-48's AND tokens = b12llff3 in both arms AND v3q's after-gc footprint after request 6 is >= 30 MiB
under v3p's AND v3q's growth G <= 1.0 MB / request -> live on NewIdeas' word. KILL: a hash / token differs, or v3q is
not lower.
R-45V RESULT (Sun 05:11-05:24, shared lock; data/v3q/; the Mac heavily loaded: swap 16 GB, p8k prefill 204 s): **BETWEEN, not
adopted.** Exact: p2k / p8k / write16 = R-48's hashes (B12_MODE=exact); fast-mode tokens = b12llff3 in all 12 requests.
After-gc footprint after request 6: v3q 10.727 vs v3p (live) 10.743 GB = -16 MB (line -30 MiB); growth G +5.4 vs +4.3
MB / request (noise under that load). Why so little: ext.bounce_stats() shows the pool at 26 buffers (66-72 MB) from
request 1: the engine runs ~26 expert reads at once, so a shared pool holds about as many buffers as there are live
workers; Fit12's estimate assumed a queue depth of 4-8 (11-21 MiB). A hard cap would serialize reads: not worth ~50 MB.
v3p's MALLOC_LARGE stayed flat (64.9 MiB / 7 regions) across the 6 requests: its residual +4.3 MB / request is not bounces.

TREES LINES (NewIdeas Sun 06:1x; RANKED #10 = Leap's option B; written before any run; CPU only): price exact draft trees on
DFlash2's own lattices before anyone builds them. Data: Leap's teacher-forced official loop on the 24 public HumanEval
samples (~/Leap/data/dfl2d_code) with the L1-20 drafter (l120_ft); scripts/tree_dump.py (a copy of Leap's cand21.py) dumps
per block, along the selector's own trajectory (its 9.54-tokens-per-pass run), the selector's top-16 candidates + softmax
per depth, its pick, the drafter's top-64 and the true tokens. scripts/tree_price.py: EAGLE-2-style trees (nodes = the
most probable paths by the product of per-depth selector probabilities: siblings only where it is unsure; Medusa-style,
since DFlash2 drafts every depth at once) vs today's chain at EQUAL verified rows; kept per check (accepted + 1) and GB per
kept token from BigMoE's rows -> GB curve (~/BigMoE/data/bytes45/bytes_study.txt, 4-bit at C 160: r 1 1.40, 2 2.29, 4 3.82,
8 6.29, 12 8.32, 16 10.03 GB; tree rows count like any rows). PASS: >= +10 % kept per check at equal bytes per kept token;
KILL: < +5 %. Not priced here: an exact tree check's own costs under REF2 (per-node GDN states, gathered attention: Leap's
bbaea5f notes).

TREES LINES ADDENDUM (Sun 08:3x, written before any result): (1) Chat priced too (NewIdeas 08:3x, after MTP Head's R-61N:
the 125B's #1 is in DFlash2's top-3 at 34 % of chat's first misses). Data: MTP Head's D-02a captures, same format, same
replay (tf_ifeval 30 + tf_multi 24 = chat; agent 12; mgsm 30; tf_gsm 60 = the math reference), with the same PASS / KILL
lines per domain. R-61N's "exact tokens per pass" per domain is the reproduction check for my chain (code: Leap's 9.54).
(2) The lattice now holds the selector's whole first-order table: P(candidate j at depth d | candidate i at depth d - 1);
candidate sets are path-free because DFlash2 drafts every depth at once. Checked on 35 blocks: its greedy path equals the
selector's pick 35/35, with identical probabilities. Trees are therefore ranked as the selector would grow them (siblings
only where it is unsure). The registered Medusa-style ranking (one probability row per depth, from the selector's own path)
is reported beside it, with a verdict for both. (3) Running it: the 4-bit lm_head is dequantized once to an f32 Linear,
with logits rounded to bf16 as on the GPU path. It runs on the CPU at nice 10 (<= 4 threads) under a CPU share lock.
Background QoS ran 1.3-115 s per block (~22 h for code alone; stopped after 2 of 24 samples); nice 10 runs ~0.16 s per block.
(4) Exact-capped check added: a depth where the answer's token is not the capture's own top-1 ends the block. That is 0.3 %
of code's generated positions, 2.5 % of ifeval's, 5.4 % of agent's.

TREES RESULT (Sun 11:38-11:46; registered price data/trees/price.txt, scripts/tree_price.py; side analysis
data/trees/frontier.txt, scripts/tree_frontier.py, NOT registered: written after the price). Lattices dumped 09:29-10:28
(CPU, share locks; data/trees/lattice_{code,chat,rest}.npz, 13,174 blocks from 180 answers).
Reproduction: my 16-row chain = the reference runs per domain: code 9.56 (Leap 9.54), chat 2.23 (R-61N 2.23), ifeval 2.45
(2.44), multi 2.05 (2.05), agent 3.00 (3.00), mgsm 3.52 (3.52), gsm 7.02 (6.98). Lattice greedy path = the pick 664 / 664.
Registered verdict, equal rows 16 (kept per check; first-order tree / pre-registered Medusa ranking; exact-capped tree gain;
tree shape = deepest node / branch points / off-chain nodes of 15):
  domain   chain 16  tree 16   tree         medusa          exact-capped  shape
  code       9.56    10.02   +4.8 % KILL   +2.8 % KILL      +4.4 %      11.8 / 1.2 / 3.4
  chat       2.23     2.72  +21.7 % PASS*  +18.4 % BETWEEN  +21.1 %      5.5 / 2.7 / 9.8
  ifeval     2.45     3.03  +23.5 % PASS*  +19.3 % BETWEEN  +22.5 %      5.9 / 2.7 / 9.5
  multi      2.05     2.46  +19.8 % PASS*  +17.4 % BETWEEN  +19.7 %      5.1 / 2.7 / 10.1
  agent      3.00     3.43  +14.5 % PASS*  +10.6 % BETWEEN  +13.6 %      6.9 / 2.4 / 8.4
  mgsm       3.52     3.99  +13.3 % PASS*  +10.8 % BETWEEN  +13.1 %      7.0 / 2.2 / 8.3
  gsm        7.02     7.56   +7.7 % BETWEEN +4.5 % KILL      +7.4 %     11.0 / 1.5 / 4.4
  * The PASS clause "best tree GB per kept <= best chain" holds only as a TIE: both bests are the same check (tree tau 0.5
    = chain t 0.5; no sibling clears tau 0.5). At equal bytes per kept token, the line's own words, trees give +0.0 %.
Equal cost per kept token (side analysis, four cost models: BigMoE's GB curve; '45GB' = the line through S45c's measured
checks, 65.9 + 14.19 x rows ms; Leap's L5 RAM and SSD models): the best tree is -10.1 % .. +0.2 % vs the best chain in every
domain under every model, also with off-chain tree rows at 0.75x (M-10 measured 0.84-0.91x for whole trees). The cheapest
check in every domain is a short confidence-truncated chain (chat ~2-2.5 rows, code ~9-10), where branching never pays;
trees win kept per check only at large rows, 2-3x off chat's cost optimum (chat: 16-row chain 4.39 GB per kept, t 0.8 1.22).
=> TREES = KILL (not worth building, for any domain or cost model).
[RETRACTED 11:5x, see S45D REPLAY below: on today's engine t 0.3 is at the optimum] Side lead (exact, not registered): the 45 GB default's truncation t 0.3 is below the optimum in every domain under every
model. Tokens per cost vs t 0.3, at t 0.5 / 0.6 / 0.7:
  45GB line: code +3.8 / +4.3 / +4.7 %, chat +3.1 / +2.3 / -0.1, agent +5.7 / +6.7 / +5.9, mgsm +3.6 / +3.5 / +1.4, gsm +5.8 / +7.1 / +7.0
  GB curve:  code +5.1 / +6.6 / +7.8 %, chat +14.3 / +18.9 / +21.6, agent +14.8 / +20.5 / +24.3, mgsm +11.4 / +15.1 / +17.0, gsm +8.6 / +12.0 / +14.1
The real gain depends on what a SHORT check costs on the 45 GB engine (the S45c line is two points at 9.8 and 16 rows, with a
66 ms intercept; the bytes curve is steeper at small rows). Nobody has run t > 0.3 there (S45c / S45c-2: 0.3 and 0.2).
Proposed S45d: t 0.3 / 0.5 / 0.7 on the 45 GB default, chat + agent + code prompts, exact by construction (tokens unchanged);
PASS >= +5 % tok/s on chat at the best t, KILL < +2 %. On chat, under the bytes model, t 0.3 is 7.3 % WORSE than no drafting
(1.40 GB per token plain vs 1.50 per kept): a second row costs 0.89 GB = 64 % of a token, while chat's first draft is kept 56 %.
The 12 GB line drafts with spec125's MTP chain, not DFlash2, so this lead is the 45 GB route's (MTP Head has the MTP-chain one).

CALSTOP LINES (Sun 11:5x; written before the run; CPU, the TREES lattices; published: SpecDec++ / AdaEDL / DISCO's
dynamic draft length): instead of a flat running-confidence threshold t, stop each check with a calibrated rule. q_d = P(the
chain is kept through depth d | the selector's running confidence bin, depth bin), fitted on half the answers. Per check,
verify n drafts maximizing (1 + sum_{j<n} q_j) - lambda x cost(n + 1); lambda = the rule's own tokens per cost (Dinkelbach,
iterated on the fitting half). Scored on the other half (2-fold over answers, both ways) against the best flat t chosen on
the fitting half, under the 45GB line and the GB curve, per domain. PASS: >= +3 % tokens per cost over that flat t on chat
AND code under both models. KILL: < +1 % on both. Exact by construction (it only picks how many drafts to verify).
CALSTOP RESULT (Sun 11:49; data/trees/calstop.txt, scripts/calstop.py): BETWEEN as registered (code +2.0 % on the GB curve
keeps it off KILL); in substance no gain. The calibrated rule vs a flat t fitted on the other half of the answers: 45GB line
-0.7 .. +0.1 % (every domain, per-domain and pooled), GB curve per domain code +2.0, chat +0.2, agent +0.6, mgsm +1.1, gsm
+1.3 %; pooled -0.8 .. +2.0 %. A flat running-confidence t carries the whole dynamic-length gain; no rule to build.
[The t part is RETRACTED 11:5x, see S45D REPLAY] It confirms the TREES side lead held out (t fitted on one half, scored on the other): one pooled t beats the live t 0.3 in
every domain. 45GB line: t 0.55, +4.0 code / +2.9 chat / +6.5 agent / +3.8 mgsm / +6.7 gsm %. GB curve: t 0.8, +8.1 / +22.7 /
+26.7 / +17.9 / +15.0 %. S45d's arms should be t 0.3 / 0.55 / 0.8 (the two models' pooled optima).

S45D REPLAY = the TREES side lead RETRACTED (Sun 11:5x; scripts/s45d_replay.py, data/trees/s45d_replay.txt; CPU, before any
GPU run). It replays MTP Head's D-23 log of the dfl arm (TODAY's 45 GB default: C 224, gate + REF2-T, t 0.3; 18 prompts, 1000
checks, read-only). Each check has its rows, kept, check / draft ms and all 15 step confidences, so a higher t only drops
rows. check(t) = logged ms - 7.82 ms x the rows dropped (D-23b's fixed-effects slope, reproduced). t 0.3 reproduces the
measured tok/s (median 1.000).
  whole set tok/s vs t 0.3: t 0.4 +0.3 %, 0.5 -1.1, 0.55 -1.7, 0.6 -2.5, 0.7 -4.6, 0.8 -7.7 (chat 0.5 -1.2 %; code at most +1.2 %)
The engine's measured cost per check is a large fixed part plus a small per-row cost. Fixed part = decode ms per check
- 7.82 x rows: code 110, math 80, chat 64, agent 179 ms. So a row costs 4-12 % of a check. The bytes curve and the S45c
line (pre-gate: 66 + 14.2 ms per row) overstated how much short checks save. Also withdrawn: "on chat, t 0.3 is worse than no
drafting" (plain would cost ~72 ms per token, ~14 tok/s, vs 22.9 measured).
With this measured model ('D23' in tree_frontier.py): best chain t 0.3-0.4 (<= +1.7 % vs 0.3); best tree vs best chain
+0.1 .. +2.1 % (agent +2.1; +2.6 with off-chain rows at 0.75x). TREES stays KILL under all five cost models. On the 45 GB
default the walls are the fixed part per check and the tokens kept per check, not rows.

PMIN12 LINES (Sun 11:5x; written before the run; CPU first): the 12 GB line runs spec125 3:3:0 with p_min 0 (always 3 MTP
drafts = 4 rows), and its rows are expensive. Warm AR step ~73 ms (AW-AR 13.7 tok/s); the 4-row pass ~173 ms (R-60
stock p2k: 16.07 s / 93 passes). So c(r) = 14.5 + 58.5 x r^0.7185 ms (Leap's SSD shape through those two points), ~33 ms per
added row. spec125 already has p_min (stop the chain where the head's running probability < p_min). Price p_min 0.1-0.6 on
the 3-draft chain by replaying MTP Head's D-23 mtp7 checks (head b3, 45 GB tokens; per check: the chain's per-step top-1 p
and kept): rows(p) = 1 + #of the first 3 steps with running p >= p_min, kept(p) = min(kept, 4, rows(p)); draft time
-2.2 ms per step not drafted (D-23b's early exit). Per type tok/s vs p_min 0. PASS (worth a real 12 GB A/B): >= +8 % on chat
or agent at one p_min, with code and math >= -2 %. KILL: < +3 % on every type. Exact by construction (drafts only).
PMIN12 RESULT, replay (Sun 11:57; scripts/pmin12.py, data/trees/pmin12.txt): PASS by the lines. 907 D-23 mtp7 checks replayed
as 3-draft chains at 12 GB costs, tok/s vs p_min 0: p_min 0.3: code +1.2, math +0.9, chat +24.5, agent +4.2, whole set +13.7 %;
p_min 0.5: +2.2 / +3.0 / +27.4 / +6.6 / +16.4 % (chat rows 4.0 -> 2.1, kept per pass 2.32 -> 1.91). Caveats: head b3 on
45 GB tokens (the 12 GB line drafts with the stock FR-Spec head, whose probabilities differ); the 12 GB cost curve is a
two-point fit. Never run at max_draft 3: SpecDecode's 10-18 'adaptive k<=5 (p_min 0.5)' arm also went to 5-6 rows, off the
sync-free gate (~2x verify), so it says nothing about the cut itself.
PMIN12 A/B LINES (Sun 11:58, before the run; scripts/pmin12_ab.py + pmin12_ab.sh; shared GPU, rough): one process on the
live default (bt_b12 31e4333196, fold-fast, stock head), spec125 3:3:0 as the suite runs it, p_min passed to
spec125.generate. Per prompt an untimed warm-up, then arms p_min 0 / 0.3 / 0.5 / 0 (baseline = the mean of the two 0
runs). Prompts: ifeval 3091 + 1051, multi zh15 + ja18 (chat), HumanEval 45 + 27 (code), MATH prealgebra 1973 + gsm8k 708
(math), agent edit10. 192 new tokens (agent 128), stop at EOS. PASS: the best p_min >= +8 % tok/s on chat, with code and
math >= -2 % (then a quiet confirmation and an exact-mode token check). KILL: < +3 % on chat. In fast mode a drafter
change can alter tokens (ALLWAIT, R-60), so token equality is reported, not required.
PMIN12 A/B RESULT (Sun 11:59-12:10, shared GPU, rough; data/pmin12/ab.txt + ab.jsonl): PASS at p_min 0.3. Live default
31e4333196 (fold-fast, v3o, stock head), spec125 3:3:0 + p_min. tok/s (sum tokens / sum decode s; p_min 0 = both runs):
  chat (ifeval 3091 + 1051, multi zh15 + ja18): 13.09 -> 0.3: 14.78 (+12.9 %), 0.5: 15.54 (+18.7 %); rows per pass 3.97 ->
    2.5-3.6 (0.3) / 2.1-3.1 (0.5); tokens per pass about the same (e.g. ja18 1.69 -> 1.68 / 1.60).
  code (HumanEval 45 + 27): 17.55 -> +0.6 / +0.5 %; math (prealgebra 1973 + gsm8k 708): 18.78 -> +3.1 % at 0.3; agent
    (edit10): +0.1 / -0.2 %. The head is confident there (rows stay ~4).
  all: 15.23 -> 16.46 (+8.1 %) at 0.3. At 0.5 math's gsm8k run STALLED: 171.6 s for the same 192 tokens and 53 passes
  (~3.2 s per pass vs 0.18). No sleep or pause in pmset; most likely another process's GPU load. Not caused by p_min (same
  tokens, same passes); it must be re-run.
  Tokens: both p_min 0 runs repeat each other bit for bit. In fast mode, chat's p_min runs diverge from p_min 0 at tokens
  16-160 (as R-60 found: in fast mode a drafter change alters tokens); code / math / agent runs are equal. Lifetime peak
  11.787 GB (< 11.95).
  The replay predicted chat +24.5 % at 0.3. The real gain is smaller: the stock FR-Spec head's probabilities differ, and
  spec125 drafts all 3 steps before cutting (no early exit).
Next: (1) exact-mode token check (B12_MODE=exact: p_min must leave tokens bit-equal; rows 1-4 in one band); (2) a cleaner
speed confirmation with more chat prompts, ABBA order, re-running gsm8k at 0.5; then propose B12 p_min to NewIdeas.

FIXED-PART RESULT (NewIdeas 12:1x; Sun 12:16; scripts/fixed_part.py, data/trees/fixed_part.txt; CPU, D-23's dfl logs:
18 prompts, 1000 checks, C 224, gate + REF2-T, t 0.3, no OOO). F = check + draft - 7.82 x rows per check.
  type   F (decode time / checks, as 11:5x) = steady F (median, checks 3..) + start-up excess + run rest
  code   128.7 = 105.7 + 23.1 + 0.1      math 82.6 = 69.1 + 9.4 + 0.1
  chat    65.9 =  57.9 +  5.6 + 0.1      agent 179.1 = 88.7 + 86.2 + 0.1
Steady split across the 18 prompts: F = 42.2 ms + 0.559 x demand SSD ms + 2.27 ms per K context (R^2 0.84; residuals
-18 .. +31 ms). SSD = 13.3 + 0.149 ms per miss. The constant includes the draft (11.3-13.7 ms short prompts, 13-17 agent),
so dense + gate + syncs ~30 ms.
  code   105.7 = constant 42.2 + SSD 65.3 + context 0.5 -> SSD 60 % (51 % of the 128.7 with start-up)
  math    69.1 = 42.2 + 31.0 + 0.5 -> constant 57 %
  chat    57.9 = 42.2 + 10.9 + 0.4 -> constant 79 % (of which draft ~12 ms = 21 %)
  agent   88.7 = 42.2 + 22.6 + 23.5 (10.4 K) -> constant 48 %; of agent's 179.1, the START-UP holds 48 %
Why agent is 179 vs chat 64. The first check after a 9-12K-token prefill (76-103 s) costs 1.6-3.4 s plus a ~190-220 ms
draft, against 82-191 ms steady. The second check is still ~290-380 ms. That is 2.1-3.8 s once per answer, spread over only
33-40 checks (bash18: 39 % of its decode time). The MTP arm shows the same 3.2-3.5 s first check, so it is the target's
first verify at long context, not the drafter. Candidates (not tested): the prefill's lazy tail evaluated inside check 1,
cold pools after the prefill, or one-time buffers / compiles at the new context. Then context (+23.5 ms at ~10K, attention
/ QSA) and SSD (+12 ms vs chat). Short prompts also pay a first-check excess (+150-350 ms), smaller.
C45 PREDICTIONS (pre-registered 12:16, before tonight's run; S45's six short code / math prompts, so the context and
long-prefill start-up parts are NOT tested). Per-run medians as c45_report prints them:
  P1 draft 11-14 ms per check on every prompt and arm.
  P2 demand SSD busy = 13 + 0.15 ms per miss (+-25 %) at every C.
  P3 no-OOO arm A at C 224: gate service on misses = 0.40-0.70 of demand SSD busy (my on-path share 0.56); lower on OOO arm B.
  P4 arm A: check + commit - gate service = ~30 ms + ~5 ms per row (+-25 %): the dense + syncs part is per-prompt constant.
  P5 C 224 -> 256 on the default: misses per check -20 .. -35 %, SSD busy down in proportion, cycle down by ~0.56 x the SSD
     drop (+-50 %): steady tok/s +5 .. +12 % on these code / math prompts.
  P6 reproduction (arm A, C 224) on the two prompts D-23 shares: prealgebra/1640 ~137 misses, SSD ~32 ms, cycle ~101 ms;
     algebra/694 ~232 misses, SSD ~58 ms, cycle ~153 ms (+-25 %).
  The split is WRONG if P3 lands outside 0.3-0.8 or P4's remainder grows with misses: then SSD is not ~half on the
  critical path, and the constant is not a constant.
  Leap 12:30 (R-80 / R-81, ~/Leap/data/r80_first_call.txt, r81_att_ctx.txt; shared GPU, rough):
  - Attention kernel builds are NOT the start-up. The first QSA call at a new length costs 1.4-1.7x a steady call
    (+1.4 .. +8.5 ms per layer; the process's very first call 6.9 vs 1.3 ms): tens of ms over 12 layers, not seconds.
    Left: the prefill's lazy tail, cold pools, or the GDN / MoE side (untested; a 16-row GDN-layer probe after a long
    prefill would close it).
  - Steady attention per check (REF2-T, 12 layers): 12.6 ms at 2K, 16.1 at 4K, 24.3 at 10K, 32.3 at 14K, 37.2 at 20K. That
    is ~1.2-1.4 ms per K, so ~55-60 % of my 2.27 ms per K context term is attention; the rest is elsewhere. The 16-row
    verify computes dense QK^T over every key under the selection mask. A block-sparse 16-row kernel could cap it near
    the 2K level: ~-12 ms per check at 10K (13 % of agent's steady 88.7 ms); not built.
PMIN12 EXACT-MODE CHECK (Sun 12:12-12:19, shared GPU; data/pmin12/ab_exact.*; B12_MODE=exact): p_min is NOT exact on the
12 GB engine. On all 4 chat prompts, p_min 0.3 and 0.5 diverge from p_min 0 (first difference at tokens 16-104); the two
p_min 0 runs repeat bit for bit. gsm8k (rows stay ~4) is equal. Cause: the engine's precision depends on its pools (cold
ranks read at MSB-3, LATE3), and which experts are cold depends on every verified row, so a shorter chain moves later
tokens. This is R-60's finding, and it holds in exact mode too. Speed in exact mode is consistent with fast mode: chat
per-prompt +17.9 % (0.3; noisy, -24 .. +70) / +34.2 % (0.5). One p_min 0 run stalled (62 s vs 19 s), so sums mislead;
per-prompt figures only.
=> Adoption is the non-exact path (NewIdeas 12:1x): an EvalSuite decode check on decision 5's core rows, then Kamen's call.
The exact mode keeps p_min 0. Knob ready in scratch: bt_b12 + opt-in B12_PMIN (default off; spec125.generate's p_min when
the caller passes none; unit-tested); candidate d4a320dfe4, not live.
PMIN12 CONFIRM LINES (Sun 12:19, before the run; shared GPU, fast mode = the default): 8 chat (ifeval 3091 / 1051 / 3223
/ 2467, multi zh15 / ja18 / ja20 / zh16) + 2 code (HumanEval 45, 27) + 2 math (prealgebra 1973, gsm8k 708) prompts, ABBA
p_min 0 / 0.3 / 0.5 / 0.5 / 0.3 / 0, 128 new tokens. Per prompt and arm, the faster of its two runs (a stall only slows);
per type, the geometric mean of the ratios to p_min 0. PASS (-> the EvalSuite decode check): chat >= +8 % at one p_min,
with code and math >= -2 %. KILL: chat < +3 %.
PMIN12 CONFIRM RESULT (Sun 12:20-12:34; data/pmin12/ab_conf.*, conf.txt; scripts/pmin12_conf.py; shared GPU, fast mode
= the default): PASS at p_min 0.3 and 0.5. Per prompt the faster of its two ABBA runs, then the geometric mean:
  chat (8): 0.3 +15.3 % (mean-of-2 +11.7, range +1 .. +39), 0.5 +21.3 % (+21.3, range +3 .. +50); rows per pass
    3.97 -> 3.06 / 2.62, tokens per pass 2.12 -> 2.07 / 1.99; tokens differ from p_min 0 (0.5: 0 of 16 equal).
  code (2): +0.5 / +1.1 %; math (2): +0.5 / -0.5 %. Tokens equal (rows stay ~4).
  All prompts by sums: +10.0 % (0.3) / +17.0 % (0.5). Lifetime peak 11.71 GB.
=> p_min 0.5 is the pick. Knob installed: live bt_b12 = e2f04e1d66 (31e4333196 + opt-in B12_PMIN, default off; unset =
the same behavior, spec125.generate untouched; unit-tested on and off). Next, NewIdeas' path: an EvalSuite decode check
of B12_PMIN=0.5 on decision 5's core rows, then SUMMARY_FOR_KAMEN (non-exact: Kamen's call).

Live bt_b12 = 8db71cc537 (Sun 12:40): default engine v3p with R-53's part a off (os.environ.setdefault("F12_R53A", "0")
before the engine import; ~/Breakthrough/v3p unchanged). Fit12 M-5b stage 3 (12:4x): part a alone caused v3p's short-prompt
peak. H (v3p + F12_R53A=0) 11.636 / 11.647 GB, lifetime 11.659, = v3o (11.641 / 11.612); part b off and part c off
change nothing (12.34 / 12.34 lifetime); tokens equal on all 9 rows. Exact (R-53 parts b / c as R-53C ran them). No speed
claim until a quiet prefill A/B (R-53D, Fit12, if NewIdeas wants it). Opt-in B12_PMIN kept. Fit12 runs M-5d (16 math rows,
lifetime line 11.95) on this sha.

RUNAHEAD-BUILD LINES (NewIdeas 12:4x; Sun 12:44, before the analysis; paper + code reading, CPU): what a live build of
MoonShots' M-16 needs. Pass k = [A: last pass's provisionally accepted rows, resident -> exact] + [B: the next block, misses
skipped]; pass k+1 re-checks B. From REF2's / BigMoE's / MTPLX's code: (1) can A + B share one call within REF2's 16-row
kernels, gate MAXK 160 and the key buckets, with A's rows bit-identical to a pass without B; (2) the cost of rolling back
B's GDN state and KV; (3) how a wrong provisional verdict is detected and what one bubble costs; (4) modelled tok/s with B
capped at 16 - |A| rows, simulated on D-23's 1000 logged dfl checks. Cost model: FIXED-PART's 42.2 ms + 2.27 ms per K
context + beta x (|A| + |B|), B's SSD off the critical path unless it exceeds the pass's compute. beta (pure compute per
row) is unknown until step 2 measures it, so I report the curve and a central estimate from the logs. Bubble rate
1 - U2 swept 1-5 %.
KILL: if A's bits cannot be kept (some REF2 path makes a row depend on later rows or on the call's row count within
16), or if the capped model gives < +15 % whole-set tok/s at the central beta. PASS (-> the build list for BigMoE / Leap,
pending step 2's beta and U2): >= +15 %.
RUNAHEAD-BUILD RESULT (Sun 12:49; scripts/runahead_model.py, data/trees/runahead_model.txt; code reading of Leap's ref2.py,
BigMoE's gated.py, MTPLX's commit_verified_window): KILL as registered: the capped model at the central beta is < +15 %.
It stays conditional on beta, which M-16 step 2 measures tonight.
(1) A's bits CAN be kept within 16 rows, with no kernel change. REF2 pads every <= 16-row call to a fixed 16-row form:
    - matmuls: qmm16c tiles, MLX's 16-row GEMM;
    - hyper-connection reads: padded;
    - QSA attention: padded with copies of the last row; key buckets set by the call's N = cache + 16 (the same with or
      without B); masked keys add exact zeros;
    - experts: each (row, rank) pair in its own threadgroups (3c / qmmg25);
    - GDN prework: bitwise for 1-16 rows.
    INV1 (all 16 rows = AR) already holds with arbitrary later rows, and REF2-T's precision is rank-based, so residency never
    moves A's bits (unlike the 12 GB engine, see PMIN12). Limits:
    (a) |A| + |B| <= 16: past 16 rows every REF2 path falls back to stock MLX. 32 rows would need 32-row tiles and attention
        classes, plus MAXK 320 > the routing row's 185-id ceiling.
    (b) The GATE is the real new piece. Each layer posts ONE routing row (<= 160 ids fits 16 rows x top-10), and the GPU
        spins until the loader released ALL of them, so B's misses would be waited on. B's ids need a no-wait class:
        prefetch-only, like the row's next-layer prediction lines, or a per-id no-spin flag; B's non-resident pairs become
        phase 2 = 0 in combine_phase. A C++ ext change in BigMoE.
    (c) The drafter's cache is repaired with A's exact taps once A is confirmed (U3).
(2) Rollback costs ~nothing extra per row. commit_verified_window(keep = A's exact accepted, verified = |A| + |B|) trims
    the QSA KV tail (offset only); each of the 36 GDN layers replays its recurrence over the KEPT rows from the snapshot; the
    PLE layer replays its forward over the kept rows. B rows cost only their capture. The real cost: a FULL accept skips the
    replay today (trim_n == 0); with B always discarded it always runs (~3-5 ms; modelled as 4 ms after a full accept).
(3) Bubble. Detected right after pass k's argmax: A's exact acceptance < its provisional length, or the exact bonus !=
    B[0]. Then commit A's exact rows, drop B, and re-draft from the exact bonus with the exact taps. B's queued reads (~150-375)
    still occupy the SSD for ~15-40 ms. The next pass has no verified A, so it re-checks only the bonus: ~one pass for 1 token.
    U2 0.95 vs 1.0 costs 2-4 points on the whole set.
(4) Model (D-23's 1000 checks in order; each prompt's own steady fixed part; SSD off the path only through beta; start-up kept
    on both sides; U2 0.98; whole set / code / chat):
    capped 16, conservative (all SSD per row): beta 2.5 +3.6 / -14.8 / +12.8 %; central beta 4.0 (= 7.82 - 0.149 x 25.3
      misses per row) -7.0 / -25.8 / +2.3 %; break-even beta 3.0, +15 % only below 1.2
    capped 16, optimistic (also 0.559 x io off the fixed part): beta 2.5 +24.6 %; 3.5 +14.5 %; central 4.0 +9.6 %
      (+15 % until beta 3.45)
    UNCAPPED 32 rows, optimistic: beta 2.5 +38.0 % (code +66.6, near MoonShots' +70-81 %); 4.0 +19.7 %
    The cap is the biggest single loss on code: tokens per pass 10.54 today -> 6.85 capped (A ~10.5 rows leaves ~5.5 for B),
    10.54 uncapped. Re-checking A nearly doubles the rows (code 14.8 per pass), so it only pays if beta is well below 7.82 / 2.
    RUNAHEAD does not touch the per-check fixed part (42-155 ms: dense padded to 16 rows, draft, syncs) or agent's start-up.
Lives if step 2 measures beta <= ~3.4 ms per row (capped, optimistic), or <= 2.5 with 32-row REF2 kernels + a wider gate
row (then +25 % capped / +38 % uncapped). MoonShots' ragged refinement (re-check A only from layer L1 whose misses the
runahead waited for) cuts A's cost by L1 / 48 and is the cheapest lever if beta lands at 3-4.
RUNAHEAD-BUILD addendum (Sun 12:5x): MoonShots' SHADOW CACHE (A exact on the committed cache, B on a second cache, as two
sequences of <= 16 rows each) does NOT avoid the kernel work. REF2's guards count rows as the product of the leading
dims, so batch 2 x 16 rows is a 32-row call and falls to stock MLX today, exactly like one contiguous 32-row sequence.
Lifting the cap needs the same pieces either way: dispatches whose rows run as independent 16-row tiles (qmm16c / GEMM /
3c pairs are already per-row within a fixed tile; REF2's R = 16 guards and MoonShots' GDN prework would extend to 32), and
the gate at 320 ids per layer (two routing rows, or a row past its 185-id ceiling). The shadow route adds a second cache
(KV + GDN + conv + PLE + indexer), a re-sync after every exact verdict, and drift: B runs on A's previous-cycle runahead
states, so U2 falls below tonight's number. The contiguous [A | B] sequence of up to 32 rows keeps B on A's exact states;
the per-row SDPA classes (sdpa_classes) already handle rows whose AR bucket differs. As two back-to-back passes (no batch-2
kernels) the shadow route pays the fixed part twice: whole set -11.5 / -16.8 / -19.4 % (optimistic, beta 2.5 / 3.5 /
4.0) and -29 .. -34 % (conservative), even with no drift cost. Review: if step 2's beta justifies lifting the cap
(uncapped optimistic +25.7 % at 3.5, +38.0 % at 2.5), build the contiguous 32-row route, not the shadow.

M-5d (Fit12 13:0x, Fit12's data; live 8db71cc537 pinned): PASS. Lifetime 11.774 GB over the 16 math rows (line 11.95),
tokens 16 / 16 equal; part a confirmed off (R-53 counters alloc 0); clean run (no other process over 8 GB, swap guard
quiet). Row maxima 11.05-11.76 GB (v3p with part a on: 11.05-12.34). The 12 GB memory line holds again on the live
default. The new binding peak: r10 at 11.757 GB, 0.11 s into its prefill (MLX active +0.88 GB over the row's start). The
same early transient set v3o's r13 max. 0.19 GB of room; that first-0.1 s allocation is the next lever if room is needed.

PMIN12 DECODE CHECK RESULT (EvalSuite runner 37, ended Sun 19:46; ~/EvalSuite/results/pmin12.txt, report_b12llff4p_vs_b12llff4.txt;
b12llff4p = live 8db71cc537 + B12_PMIN 0.5 vs b12llff4 = the control, 239 rows, shared GPU): **FAIL by the pre-registered
11.95 GB memory line only**. Decision 4 PASS: math 86.0 = 86.0 % (0/0 discordant), IFEval 87.0 = 87.0 % (3/3), HumanEval+
91.7 = 91.7 % (0/0); standard table 4/5 (truncations / loops 16 / 2 vs 14 / 1). Memory: one PMIN12 HumanEval+ process (26
rows) 11.99 GB; the CONTROL also reaches 11.95 GB (24-row HumanEval+ process), and the same rows sit at the top in both arms
(HumanEval/156, /45, /42; the control also /8). Process peaks: PMIN12 11.05-11.99, control 11.43-11.95. So the live default
has ~0 headroom on HumanEval+ processes (decision 5 measured 11.51 GB on the agent process only; M-5d 11.774 on math), and
PMIN12 adds at most ~0.04 GB there (maybe from the varying verify shapes, 2-4 rows). Tokens PMIN12 == control: math 21/50,
IFEval 22/100, HumanEval+ 53/60, agent 12/29; the control == decision 5's b12llff rows 239/239. SSD MB per token 359-385 vs
392-458; rough tok/s (shared, heat pauses; my ABBA is the speed number) math 13.2 vs 10.2, IFEval 10.0 vs 8.8, HumanEval+
12.1 vs 11.6, agent 18.0 vs 16.7. Agent creep: 29 rows in one process, lifetime 11.56 GB. Next: trim the default's HumanEval+
peak first (rows /156 /45 /42 /8), then a memory re-check on the 60 HumanEval+ rows per arm (~25 min each). Kamen's call.

DECISION MAP Q1 (Fable's mission 9 map, run by Breakthrough; NewIdeas 22:0x, orchestrator 01:0x; Mon 03:59-04:10, one
process, ~/FableGod/decision_map/out/: baseline / topk10 / topk2 .json, report.md, gate.txt, runner.log). Teacher-forced
prefill of 36 windows (12 HumanEval, 12 IFEval, 12 multilingual: the 125B's own greedy answers), D-02a's config (C 51,
4-bit, exact prefill); per answer position the argmax vs the baseline's, by the 125B's own class (sure p1 >= 0.99 / tie
p1 - p2 <= 0.1 / mid). Gate: baseline top-1 = D-02a's stored top-1 on 10316 / 10316 (100 %); topk10 through the score
wrapper bitwise = baseline. topk2 (ranks >= 2 weight 0, kept experts keep the model's weights; every position at top-2):
  code   sure 0.21 % (6 / 2922)  tie 62.0 % (31 / 50)    mid 25.6 % (101 / 394)    sure share 86.8 %
  IFEval sure 0.65 % (7 / 1076)  tie 64.8 % (188 / 290)  mid 34.5 % (700 / 2031)   sure share 31.7 %
  multi  sure 5.07 % (73 / 1439) tie 77.5 % (193 / 249)  mid 44.0 % (821 / 1865)   sure share 40.5 %
  mean margin change -2.34 nats. => Fable's outcome (a) on code (code-sure <= 1 %; his kill line > 5 %): Q1 opens on code,
  for SURE rows only; multilingual-sure sits at the 5 % line. NOT exact. Harsher than Q1 itself (the whole context ran at
  top-2); the classes are the 125B's, not the drafter's (Q1's emulation measures that). BIG sent 04:1x.
  topk2r (04:47; the kept top-2 RESCALED to the row's original routed weight) is far worse: code sure / tie / mid 2.33 %
  (68 / 2922) / 74.0 / 43.2 %, IFEval-sure 5.58 %, multi-sure 20.9 %. => Q1 must keep the top-2's own weights (no
  renorm), as the 12 GB line found (B12_NORM=mass).
DECISION MAP NIGHT 1 COMPLETE (Mon 04:42-05:37, attached run under a session-held quiet lock; 17 ablations; summary
~/FableGod/decision_map/out/NIGHT1_SUMMARY.md, table report.md). Flip % sure / tie / mid, code (IFEval-sure, multi-sure):
  topk5 0.00 / 40 / 8.6 (0.00, 0.07); shared_zero 0.68 / 66 / 30.5 (3.35, 7.44);
  MoE reuse of the previous position: g0 (layers 0-11) 22.7 / 88 / 67 (38.0, 42.5); g1 (12-23) 0.14 / 56 / 16.5 (0.09,
  0.00); g2 (24-35) 0.10 / 56 / 20.8 (0.00, 0.00); g3 (36-47) 18.9 / 82 / 64 (36.3, 69.6);
  att_reuse 29.0 / 54 / 29.2; gdn_reuse 88.5 / 100 / 96.7; hc_static (chunk-mean gates) 2.87 / 64 / 25.9; hc_mean_raw
  (raw stream mean) 100; 4-bit OUTPUTS: hc 0.00 / 44 / 9.1, gdn 0.00 / 40 / 3.8, att 0.00 / 44 / 2.3.
  => sure rows absorb nearly any single cut, ties flip 35-60 % under any cut, mid rows carry the damage. NEW LEAD: the
  middle layers' routed-expert outputs (12-35) reusable from the previous position on sure rows (0.10-0.14 % code-sure);
  untested together / stacked with top-2. Dead as single cuts: GDN / attention reuse, static HC, shared expert off on
  chat. BIG sent 05:4x. Night 2 (per-layer MoE reuse, 48 runs, ~3.5 h) wired, waits on NewIdeas / Fable.
DECISION MAP NIGHT 2, STAGE 1 (PLAN, written before the run; NewIdeas Mon 06:4x: night 2 in stages, speed answer first;
the orchestrator's lounge note: night 1's reuse shifted COMPUTED outputs, so every row's experts still ran, and an engine
that skips sure rows chains back to the last computed row). One quiet ticket (~25 min, queued 06:49), run_stage1.sh,
attached. New code (dm_common._reuse, CPU unit test test_reuse.py runs first): lag k with the carry across chunk calls
(night 1's lag 1 restarted at each call: ~1 row per chunk boundary differs), and CHAIN = rows whose BASELINE class is sure
take the last computed row's output (the 125B's own class as an oracle; prompt rows and non-sure rows compute).
  (1) moe_reuse_prev_L12_35: layers 12-35 at once, lag 1. Predict code-sure 0.3-0.6 % (g1 0.14 + g2 0.10, some
      compounding), code-mid ~30 %.
  (1b) moe_reuse_chain_L12_35: sure rows skip the middle routed experts, chained (code has long sure runs). Predict
      code-sure 0.5-2 % (staler reuse, but the computed rows keep the context near-exact), mid / tie below lag 1's.
  (2) moe_reuse_lag2_L12_35, moe_reuse_lag4_L12_35 (copy from t-2 / t-4, every row). Predict code-sure ~2x / ~3-4x lag 1.
  (3) edges: moe_reuse_prev_L08_11 and _L36_39 (lag 1). Predict 8-11 code-sure 0.5-3 %, 36-39 0.5-3 % (g0 / g3 were
      19-23 %, likely carried by the outer layers).
  Kill line as night 1: code-sure > 5 % = closed; <= 1 % = open. NOT exact by construction (a non-exact lead).
DECISION MAP NIGHT 2 STAGE 1 RESULT (Mon 06:53-07:18, one quiet ticket, attached; gate 10316/10316; unit test ALL OK; no
row-alignment warnings; swap flat; ~/FableGod/decision_map/out/NIGHT2_STAGE1.md, report.md, stage1_extra.txt). Flip %
sure / tie / mid (dmargin), code | IFEval | multi:
  12-35 at once, lag 1   1.03 / 66 / 33.0 | 1.77 / 67 / 39.8 | 0.69 / 71 / 29.7 (-1.59)   [predicted 0.3-0.6: WORSE]
  12-35 CHAINED          8.11 / 48 / 21.8 | 1.67 / 44 / 10.0 | 0.69 / 45 / 8.6  (-1.89)   [predicted 0.5-2: WORSE]
  12-35 lag 2            2.46 / 66 / 40.4 | 4.83 / 73 / 51.8 | 2.85 / 69 / 39.7 (-2.30)   [~2x lag 1: as predicted]
  12-35 lag 4            4.76 / 74 / 50.5 | 10.0 / 80 / 60.0 | 5.28 / 78 / 50.5 (-3.11)   [~4.6x: as predicted]
  edge 8-11, lag 1       0.00 / 46 / 4.3  | 0.00 / 46 / 5.8  | 0.00 / 43 / 4.2  (-0.04)   [predicted 0.5-3: BETTER]
  edge 36-39, lag 1      0.82 / 62 / 16.0 | 1.67 / 56 / 19.9 | 2.43 / 59 / 21.4 (-0.77)   [in the predicted range]
  Chained, skipped rows by staleness (code / IFEval / multi): 1 row 1.41 / 0.52 / 0.13 %; 2-4 rows 5.80 / 2.96 / 1.22 %;
  code 5-16 rows 10.3 %, 17-64 5.2 %, 65+ 10.5 %. Chain skip share: code 86.8 % (runs mean 13.7, max 398), IFEval 31.7 %,
  multi 40.5 % (runs ~2).
  => The halves compound ~4x; staleness grows the damage ~linearly; the engine pattern (chained skipping of sure rows) is
  KILLED on code (8.11 % > 5 % line), BETWEEN on chat. The band is ~8-35 (8-11 free, 36-39 costs). What survives: reuse
  at most 1 row old (every other sure row skips), not exact. Night 1's lead NARROWED. Stage 2a proposed to NewIdeas
  (bounded chain S=1 / S=2 on 8-35, + top-2, low edge 0-3 / 4-7; ~30 min); queued only after D-23d / runner 62 per NewIdeas.

## DAY9 PLAN (Mon 2026-11-02 10:4x; Kamen's day order via the orchestrator: conditions 1-3 of ROUTE_350)
- Decision map, D-02a configuration, 36 windows, share lock (flip rates). Realizable gate = DFlash2's cf for the draft
  each row verifies (MTP Head's HPD-0 walk on the same windows; checks identical 12/12 per domain); oracle column = the
  125B's own class. ~/FableGod/decision_map/run_day9.sh: capture (routing for MoonShots' bytes replay) ->
  Q1 gate_top2_cf099 / _cf0999 -> Q2 gate_bands (NEW-1) -> Q3 hc8w (HC mixers + PLE at 8 bits) / hc8w_mix.
- Lines: Q1 flips on drafter-sure code rows <= 1 % PASS, > 5 % KILL; Q2 KILL if the 0.9-0.99 band flips > 2x the sure
  band; Q3 map number goes to EvalSuite (its gate decides).
- Gate agreement (out/gate_dfl2_agreement.txt): code cf >= 0.99 = 65.7 % of rows (98.9 % of them oracle-sure);
  IFEval 20.1 % (87.7 %); multi 15.6 % (87.9 %).

## DAY9 #1 Q1 RESULT (Mon 2026-11-02 11:0x): PASS
- Top-2 at the model's own weights on DFlash2-sure rows, rest exact (decision map, D-02a 4-bit, 36 windows):
  cf >= 0.99 (65.7 % of code rows): gated code rows flip 0.09 % (2/2211, CI 0.02-0.33), oracle-sure gated 0/2186;
  cf >= 0.999 (47.7 %): 0.00 % (0/1607). IFEval / multi gated 0.59 / 1.26 % (0.99), 0 / 0 % (0.999).
- All code rows 1.34 / 1.25 % (ungated ties and mids through the context: 1.28 % of code rows at 0.99).
- MoonShots' bytes on the capture (C 256 + M-17): code SSD -40.3 % / -28.3 % -> implied 70.2-76.2 / 68.1-71.3 tok/s
  (63.6 today). Evidence ~/FableGod/decision_map/out/DAY9_q1.txt. #2 / #3 paused 11:01 for Leap's quiet window.

## DAY9 #2 Q2 + #3 Q3 RESULTS (Mon 2026-11-02 11:2x)
- Q2 bands (NEW-1 by DFlash2 cf): code >= 0.999 top-2 0.00 %, 0.99-0.999 top-4 0.17 %, 0.9-0.99 top-8 1.07 %, < 0.9
  exact 5.38 % (context), all 1.28 %. Line (mid > 2x sure) -> KILL by the letter; but the 0.9-0.99 band flips the same
  1.07 % when left exact (Q1 runs): own cut net 0 on code (+1 IFEval, +3 multi). out/DAY9_q2_attrib.txt.
- Q3 hc8w (HC mixers + PLE at 8-bit g64, 196 linears): sure 0.00 % everywhere, mids 0.5-0.7 %, all code 0.33 %.
  Map PASS; EvalSuite runner 39 GO (C 96) 11:20.
- Q3 closed 13:46: EvalSuite's same-scores gate PASS too (decision 4 vs REF2-T, 210 rows; all three sets p >= 0.5;
  a lean against it outside the line: the 5 changed answers are all losses, p 0.06). ~/EvalSuite/results/hc8_d4.txt.

## TWO-PASS CHECK on the map: PRE-REGISTRATION (Mon 2026-11-02 18:3x; the orchestrator 18:1x, Fable mission 20)
- Design (cloud Fable; ~/FableGod/CACHE.md section 3): pass 1 = top-k1 for every row; a row is certified when its own
  p1 under that cut is >= theta; pass 2 = ranks k1+1..10 for uncertified rows (exact). Certificates from the map's
  all-rows runs: top-2 at theta 0.99 (topk2: code 73.4 % of rows, IFEval 15.6, multi 10.8) and top-5 at theta 0.9
  (topk5: 89.8 / 43.0 / 51.7 %). Pass 1 alone flips 0 certified code rows at both (0/2472, 0/3024; multi top-2 4/385).
- Runs (one teacher-forced pass each, certified rows cut at the model's own weights, the rest exact, so the cut reaches
  later rows through the context as in DAY9's gated run): cert_top2_p099, cert_top5_p09. Read-out dm_cert_read.py ->
  ~/FableGod/decision_map/out/CERT_read.txt.
- Lines (DAY9 Q1's, on the certified code rows of the mixed run): flips <= 1 % PASS, > 5 % KILL, else BETWEEN.
  Reported beside them, no line: uncertified rows' flips through the context, all rows, windows with a flip, and
  certified rows whose p1 in the mix falls under theta; the comparison rows are Q1's gated run (all code rows 1.34 %),
  HC8 (0.33 %) and the t4 control (the summation-order floor, when run). The quality verdict stays EvalSuite's
  (decision 4); the map is mission 7's emulation.

## TWO-PASS THRESHOLD SWEEP: PRE-REGISTRATION (Mon 2026-11-02 19:3x; the orchestrator 19:2x, cloud Fable's 12 GB v2)
- Why: pass 2 re-reads the 3.44 GB of dense weights whatever its row count, so how often it runs is the lever (cloud
  Fable's estimate: pass 2 in 97 / 71 / 37 % of blocks at theta 0.99 / 0.9 / 0.7).
- Runs added to the two above, same harness and read-out (dm_cert_read.py), in this order: cert_top2_p099,
  cert_top2_p09, cert_top2_p07, cert_top2_p08, cert_top2_p095, cert_top5_p09.
- CPU columns, measured before the runs (topk2 + the exact walk's 391 code checks; "blocks needing pass 2" = a check
  with an uncertified row at or before its first exact miss, cache_replay.py's rule; a first check's prompt row is
  prefill's): code certified share / blocks needing pass 2 / pass-1-alone certified-row flips:
  0.99: 73.4 % / 77 % / 0 of 2472; 0.95: 82.3 / 63 / 0 of 2771; 0.9: 85.4 / 60 / 4 of 2873; 0.8: 88.4 / 55 / 10 of 2976;
  0.7: 91.6 / 47 / 23 of 3082. IFEval / multi blocks: 99 / 97 % at 0.99, 86 / 72 % at 0.7 (chat keeps 1.9-2.5 per check,
  and its first-miss row is mostly uncertified).
- Lines: each threshold takes the line above (certified code rows of the mixed run: <= 1 % PASS, > 5 % KILL, else
  BETWEEN). The sweep's pick = the lowest theta that PASSes, reported with its blocks-needing-pass-2 share; IFEval /
  multi reported without a line. The quality verdict stays EvalSuite's (decision 4).

## ADAPTIVE-THETA ARM: PRE-REGISTRATION (Mon 2026-11-02 20:2x; the orchestrator 20:2x, cloud Fable's 12 GB v3)
- Certificate (dm_acert.py; realizable: the drafter's cf and draft before the check, pass 1's argmax and p1 after it): a
  row is certified at p1 >= hi, or at p1 >= lo where the drafter is sure (cf >= s) and its draft equals pass 1's argmax.
  Runs (same harness and read-out as the sweep): acert_top2_p07_p09_cf099 (cloud's arm: lo 0.7, hi 0.9, s 0.99) and
  acert_top2_p05_p08_cf099 (the working point theta 0.8 plus a sure, agreeing drafter at p1 >= 0.5).
- CPU columns before the runs (code certified / blocks needing pass 2 / pass-1-alone certified flips): cloud's arm 86.5 %
  / 58 % / 4 (plain 0.9: 85.4 / 60 / 4); the 0.8 arm 89.1 % / 52 % / 10 (plain 0.8: 88.4 / 55 / 10). With s 0.9 instead
  of 0.99 flips rise (lo 0.5, hi 0.9: 12).
- Lines: the sweep's (certified code rows of the mixed run <= 1 % PASS, > 5 % KILL, else BETWEEN). An arm is worth taking
  over its plain theta when it PASSes and its all-code-row flips stay within the plain theta's 95 % CI; reported with its
  blocks-needing-pass-2 share. The quality verdict stays EvalSuite's.

## RANK-AWARE CACHE (CPU replay): PRE-REGISTRATION (Mon 2026-11-02 20:4x; the orchestrator 20:4x, cloud Fable's 12 GB v4)
- Policy: per-layer LRU at C slots that admits only experts used at ranks 0-1 (pass 1's reads; prefill admits its rows'
  ranks 0-1); ranks 2-9 (pass 2) hit when resident and are streamed otherwise, never admitted. Against today's policy:
  the same LRU admitting every read. Exact (bytes and timing only).
- Replay (dm_rankcache.py, CPU): DAY9's routing capture and the HPD-0 walks' checks (cache_replay.py's method, cache
  carried across a domain's windows, 2.2 MB per expert record); per check, per layer: pass 1 = the union of the block
  rows' top-2, then pass 2 = ranks 2-9 of the uncertified rows up to the first exact miss (top-2 certificate at theta
  0.8, the working point). Rows after the first miss carry a wrong-draft stand-in (row p + 97's routing), as the engine
  sees them; cache_replay.py's true-routing variant reported beside. Metrics: pass-1 hit rate, pass-1 / pass-2 / total
  bytes per check, pass-1 read time at 13 GB/s.
- Lines (the orchestrator's, on code at C 51, wrong-draft variant): PASS = pass-1 bytes <= 0.30 GB per check with total
  bytes not up > 10 % against today's policy; KILL = pass-1 bytes > 0.40 GB; else BETWEEN. C 160 and theta 0.99 reported
  beside, no line.

## RESULTS Mon 2026-11-02 20:5x: two-pass threshold sweep (4 of 6 runs) and the rank-aware cache
- Sweep (two-pass mixed runs on the map; certified code rows [95 % CI]; all code rows; IFEval / multi all rows in
  ~/FableGod/decision_map/out/CERT_read.txt): theta 0.99 0/2472 [0.00-0.16] PASS, all 1.16 %; 0.9 3/2873 = 0.10 % PASS,
  all 1.49 %; 0.8 8/2976 = 0.27 % [0.14-0.53] PASS, all 1.63 %, 9/12 code windows touched; 0.7 18/3082 = 0.58 % PASS,
  all 1.93 %. Certified rows flip at x0.75-0.80 of pass 1 alone (the mix's context is cleaner); the whole sequence is
  a theta-independent floor (1.16 % at 0.99 with no certified flip; Q1's gated run 1.34 %) plus 0.3-0.8 points. So the
  lowest theta that PASSes the line is 0.7 (pass 2 in 47 % of code blocks); the working point stays 0.8 (orchestrator,
  20:2x; pass 2 in 55 %). 0.95 and top-5 not yet run (a swap-guard stop at 20:43, requeued after the adaptive arms).
- Rank-aware cache: KILL by its line (code, C 51, theta 0.8, wrong-draft rows): pass-1 hit 64.1 -> 66.1 %, pass-1 bytes
  0.708 -> 0.669 GB per check, total 1.043 -> 1.060 (+1.6 %). Belady's MIN on pass 1's stream (the ceiling for any
  residency policy) reads 0.378 GB per check (80.8 % hit); the cloud's 0.26 / 80 % is that ceiling under cache_replay.py's
  optimistic routing (0.259, 86.6 %). C 160: pass 1 0.224 -> 0.154 but pass 2 0.127 -> 0.200, total flat; chat totals
  +8-55 %. Evidence out/RANKCACHE_read.txt.

## DOMAIN-ADAPTIVE CERTIFICATE ARM: PRE-REGISTRATION (Mon 2026-11-02 20:5x; Breakthrough's, from the sweep's chat numbers)
- Why: theta 0.8 holds on code but not on chat (certified rows flip IFEval 2.55 %, multi 4.63 %; all rows 8.51 / 7.04 %,
  vs 4.50 / 2.70 % at 0.99), and chat needs pass 2 in 79-91 % of blocks at 0.8 anyway. The drafter is sure mostly on
  code, so a sure-and-agreeing drafter can carry the low threshold where it pays.
- Run: acert_top2_p05_p095_cf099 (p1 >= 0.5 where cf >= 0.99 and the draft = pass 1's argmax, else p1 >= 0.95). CPU
  columns (certified / blocks needing pass 2 / pass-1 certified flips): code 85.1 % / 59 % / 0; IFEval 29.2 % / 97 % / 8
  (0.81 %); multi 28.2 % / 91 % / 13 (1.30 %); against theta 0.8: code 88.4 / 55 / 10, IFEval 40.4 / 91 / 38, multi
  40.8 / 79 / 66.
- Lines: the sweep's on certified code rows (<= 1 % PASS, > 5 % KILL); it beats theta 0.8 as the working point when it
  PASSes, its IFEval and multi all-row flips are below theta 0.8's (8.51 / 7.04 %) by more than their 95 % CIs, and code
  pass 2 stays <= 60 % of blocks. The quality verdict stays EvalSuite's.

## THE EXACT BYTE SET (CPU replay): PRE-REGISTRATION (Mon 2026-11-02 21:1x; the orchestrator 21:1x, cloud Fable's 12 GB v5)
- Set: (a) C 64 slots instead of 51 (the 4-bit drafter frees ~13 per layer); (b) block trimming: a check drafts up to
  and including its first draft with cf < 0.5, and pass 1 runs only those rows + 1 (the running-confidence cut at 0.3,
  the live engine's default, reported beside); (c) early-layer prefetch under pass 2: after a check that ran pass 2, the
  next check's pass-1 reads at layers 0-2 come off the critical path (an ORACLE bound: the next check's tokens are not
  known during pass 2).
- Replay (dm_byteset.py): dm_rankcache.py's LRU replay (today's policy), rows after the first miss holding wrong drafts
  (row p + 97), re-walked so trimmed checks end early (from an anchor inside an exact-walk check, the first miss stays
  the walk's; the drafts' cf are the walk's at the same target positions). Certificates: theta 0.8 and the
  domain-adaptive one (0.5 if cf >= 0.99 & agree, else 0.95). Each lever alone and the set; per check and per kept token.
- Line (the orchestrator's): PASS if the set gives pass-1 bytes <= 0.50 GB per check on code (raw replay; FLOOR's 1.3x
  calibrated value beside) at theta 0.8 and at the domain-adaptive certificate; KILL above 0.60; else BETWEEN. Guard: if
  trimming cuts kept per check by more than 5 %, the verdict uses pass-1 bytes per kept token x the untrimmed kept.
- RESULT 21:1x (exact byte set, dm_byteset.py; out/BYTESET_read.txt): BETWEEN by its line. Code, theta 0.8, raw:
  pass-1 0.708 -> 0.493 GB per check with the set, but trimming cuts kept 8.64 -> 8.12 (-6 %), so the judged value is
  0.524 (per kept token x the untrimmed kept); calibrated 0.403. The domain-adaptive certificate: the same (0.493 /
  0.524). Alone (pass-1 MB per kept token, today 81.9): C 64 69.8 (x0.85, not 1/C's x0.80), trim cf < 0.5 74.6, the live
  engine's own running cut 0.3 74.2 (already in today's engine), prefetch layers 0-2 (oracle) 77.4. On the engine's own
  trim the new levers (C 64 + prefetch) give 60.2 MB per kept token = 0.520 GB per check raw, 0.40 calibrated: inside
  120's <= 0.66, at 150's <= 0.45 only calibrated, so the 3-bit ranks 0-1 arm decides 150.

## SKETCH-ON-MISS (FAST, 45 / 24 GB lines): PRE-REGISTRATION (Mon 2026-11-02 22:2x; the orchestrator 22:2x, Fable A's
## NEWIDEAS_SCREEN.md section 2)
- Idea: a rank-32 RAM shadow of every expert serves a cache miss at once while the exact expert loads behind it.
- Shadows: rank-32 SVD of each routed expert's three matrices (gate / up 640 x 2560, down 2560 x 640) of the 4-bit
  model as served (~/BigMoE/models/bare, dequantized), stored int8 with per-row scales (W ~ A B, A = U_r S_r).
- Map arm (D-02a harness, 36 windows): the shadow set = exactly the experts an LRU at C 256 misses (DAY9's routing
  capture, the exact scheme's 16-row checks from the HPD-0 walks, wrong-draft rows after the first miss, cache across a
  domain's windows); for every row of a check that uses a missed expert at that layer, the expert's output is the
  shadow's (the router's weight kept). Control on the same set: the expert dropped (weight 0, no renorm).
- Lines (the orchestrator's, flips vs the baseline argmax over ALL rows, whole sequence): PASS if <= 0.1 % on code and
  <= 0.5 % on chat (IFEval and multi each); KILL if > 0.3 % on code; else BETWEEN. Also KILL (Fable's) if the shadow
  arm is no better than the drop control. Reported beside, no line: Fable's class lines (code-sure <= 0.1 %, code-mid
  <= 3 %, chat-sure <= 1 %, chat-mid <= 10 %), the flagged rows' own flips, the shadow's relative error per matrix.
  Caveat written now: the map's context floor for small all-row perturbations is 0.33 % of code rows (HC8), so the
  all-row line may sit near or below the floor; the drop control and the class split separate the two. EvalSuite's
  same-scores is the contract after a PASS.
- Measured before any map run (22:2x, dm_shadow_build.py --test and a spectrum check, CPU): the routed experts' spectrum
  is nearly flat. Energy kept at rank 16 / 32 / 64 / 128 / 256 = 6-10 / 12-16 / 22-26 / 38-42 / 63-66 % (layers 0 / 24 /
  47, gate and down, 6 experts each); the rank-32 shadow's relative Frobenius error is 0.92-0.94 (int8 adds nothing). Plain
  weight-SVD shadows are near a drop on isotropic inputs; the next step proposed is an activation-aware (whitened)
  shadow, judged first by the output error on captured MoE inputs (CPU), the map arm only if that clears.
- KILL, plain weight-SVD shadows (CPU basis; the orchestrator 22:3x): rank 32 keeps 12-16 % of each expert's energy
  (relative error 0.92-0.94), about a drop; the plain rank-32 map arm is skipped.

## SKETCH-ON-MISS, ACTIVATION-AWARE SHADOWS: CPU LINE PRE-REGISTRATION (Mon 2026-11-02 22:3x, before the capture)
- Capture (Tue, NewIdeas' slot, the map harness): every layer's MoE input x (the routed experts' input) on the answer
  rows of the 36 windows, fp16, with the routing capture's ids and router weights. Split: even-indexed windows of each
  domain fit, odd-indexed windows are held out.
- Shadows, rank 32 and 64, from the fit windows: gate / up whitened by the layer's input second moment
  (W C^1/2 -> SVD -> A B C^-1/2, small ridge), down whitened by the expert's own intermediate h = silu(W_g x) * (W_u x)
  over its routed fit rows (ridge-shrunk toward the layer mean); plain SVD beside.
- Metric, on held-out (row, expert) pairs as routed: the relative output error ||E(x) - S(x)|| / ||E(x)|| of the whole
  SwiGLU expert, its router-weighted mean over ranks 0-9 and over ranks 6-9 (the cold experts a miss mostly hits); a
  drop = 1.0.
- Line: GO to the map arm (shadow vs drop on the LRU C 256 miss set, the lines of ad1370c) if the whitened error is
  <= 0.5 at rank 32, or <= 0.5 at rank 64 (then the arm at rank 64 and the RAM doubled, ~5.6 GB, reported); STOP (KILL
  for whitened shadows too) if > 0.7 at rank 64; else BETWEEN, reported, the orchestrator decides.

## THE LAYER-WISE REUSE MAP (FAST; Fable A's ASSUMPTION_HUNT.md, the orchestrator 22:5x): LINES AND THE CPU BYTES
- Arm (built 22:5x, test_reuse ALL OK): pframe_map / pframe_map_q1 = the previous position's routed output reused in
  layers 8-11 on every answer row and in layers 12-35 on drafter-sure rows (cf >= 0.999), both bounded chains (S = 1:
  consecutive reusers alternate with a fresh row; the unbounded chain is KILLed already), _q1 stacks gate_top2_cf099.
- Lines (the orchestrator's, 22:5x, before any measurement): PASS = all-row code flips within +0.3 points of the gate's
  floor (Q1's gated run, 1.34 %) AND expert bytes <= x0.85; KILL = code flips > +1 point.
- CPU bytes, measured after the lines (dm_pframe_bytes.py; the FLOOR method: union of expert ids per layer over each
  16-row block, wrong-draft rows after the miss): code exact 7.89 GB per check, Q1 5.68, the map 7.21 (x0.91 of
  exact), the map + Q1 5.43 (x0.96 of Q1); IFEval / multi x0.97-0.98. Reused row-layers: 50 % in 8-11, 19 % in 12-35 on
  code (2 % on chat). The bytes condition (<= x0.85) fails against exact and against Q1, so the arm cannot PASS: reads
  are a union per block, and a skipped row saves only the experts no other row of the block uses. The pairs (compute)
  cut is ~14 % on code before Q1.
- KILL on bytes, CPU basis (the orchestrator 22:4x): the layer-wise reuse map's map run is skipped (no flips needed). The
  ~14 % pairs cut does not rescue it on the tensor kernel either (NAX6 pays per (expert, tier) tile, so only experts
  that leave the block's union save). Method rule from now on: price bytes and tiles on the union per 16-row block
  (FLOOR's method), never per row.

## Q1-ON-MISS (FAST; Fable C's C1-2, ~/FableGod/fablec/INVENT.md; the orchestrator 23:0x): PRE-REGISTRATION (Mon 23:0x)
- Idea: on drafter-sure rows (cf >= theta) keep ranks 0-1 AND every rank 2-9 expert that is resident; drop (weight 0, no
  renorm) only the rank 2-9 experts that would miss. Q1's SSD bytes by construction; sure rows keep most of their 10.
- Arm (Tue, the map harness): per-(row, layer, rank) keep masks from a CPU LRU replay at C 256 on DAY9's routing capture
  (the 45 GB line's slots; cache across a domain's windows, prefill top-10, the HPD-0 walks' 16-row checks, wrong-draft
  rows after the first miss), simulating the scheme itself: at each layer a check loads the union of what it needs
  (all ranks of unsure rows, ranks 0-1 of sure rows); a sure row's rank 2-9 expert is kept if resident after those
  loads. Runs: qom_cf099, qom_cf09, and Q1 at 0.9 (gate_top2_cf09) for the KILL comparison (Q1 at 0.99 exists).
- Lines (Fable C's): PASS = gated code flips <= 0.05 % at theta 0.99 and <= 0.5 % at 0.9; KILL = gated code flips >= Q1's
  at the same theta. Reported beside: all-row flips by class and domain, the kept share of rank 2-9 experts on sure rows.
- FLAG FOR KAMEN (with the result): the outputs depend on the cache state, so runs are not repeatable token for token.

## Q1-UNION (FAST; Fable C's AUDIT_350.md section 4; three sessions converged; the orchestrator 23:2x): PRE-REGISTRATION
## (Mon 23:2x; SUPERSEDES the Q1-on-miss arm above as the main arm, which becomes the cache-dependent variant)
- Arm: on drafter-sure rows (cf >= theta) keep top-2 plus every rank 2-9 expert already in the CHECK's union at that
  layer (another row of the block routes it: all ranks of the unsure rows, ranks 0-1 of the sure rows); drop only the
  experts that would be new to the check. Deterministic and token-repeatable; bytes and expert groups = Q1's by
  construction. Masks per (row, layer, rank) from DAY9's routing capture and the HPD-0 walks' 16-row blocks (the rows
  after a check's first miss enter the union with the wrong-draft stand-in, row p + 97); reported: the share of Q1's
  dropped pairs kept (Fable C's replay: 52 % at 0.99, 45 % at 0.9).
- Runs (Tue): q1u_cf099, q1u_cf09, and Q1 at 0.9 (gate_top2_cf09) for the comparison at 0.9. Variant if cheap: + the
  resident ones (Q1-on-miss, cache-dependent, not repeatable; flag for Kamen).
- Lines: PASS = gated code flips <= 0.03 % at theta 0.99 (Q1: 0.09 %) and <= 0.5 % at 0.9; KILL = gated code flips >=
  Q1's at the same theta. All-row flips by class and domain reported beside.

## t3_all (FAST, 45 GB's plan 3333333333 = the 1-bit sidecar dropped; the orchestrator 23:3x): PRE-REGISTRATION (Mon 23:4x)
- Arm: EVERY row (prompt and answer) takes all ten routed experts from records3 at 3 bits (MSB-3), the model's own
  weights; control t4_all = the same path at 4 bits (the bit-exact rebuild through dm_t3's summation order). Built:
  dm_t3.routed_topk (K = 10), gate ("t3all",), names t3_all / t4_all; test_t3.py ALL OK (K = 10 at bits 3 vs a
  dequantize reference: max rel err 1.6e-2, as top-2's 1.4e-2).
- Lines (the orchestrator's): PASS if all-row code flips stay within +0.3 points of the t4 control (t4_all) AND code-sure
  rows flip <= 1 %; KILL if code flips go > +1 point over t4_all; else BETWEEN. Reported: flips by the 125B's own class
  (sure / tie / mid) on code, IFEval and multi, t3_all vs t4_all, so it reads straight across to 45 GB. EvalSuite's
  same-scores follows a PASS.
- RESULT Tue 02:2x (CPU, dm_shadow_fit.py on the xcapture of 02:0x; out/SHADOW_read.txt): STOP, KILL for activation-aware
  shadows too. Router-weighted mean relative output error of the whole SwiGLU expert on held-out routed pairs (18 fit /
  18 held-out windows; layers 0 / 8 / 16 / 24 / 32 / 40 / 47, the 48 most-used experts each = 27-50 % of the held-out
  weight), ranks 0-9 / ranks 6-9, a drop = 1.0: plain r32 0.989 / 0.998, plain r64 0.967 / 0.982, whitened r32 0.920 /
  0.940, whitened r64 0.892 / 0.917 (line: GO <= 0.5, STOP > 0.7 at r64). Only the end layers compress a little (whitened
  r64: layer 0 0.80, layer 47 0.70); the middle layers stay at 0.92-0.97. The experts are near full rank in function
  space too, so no rank-32 / 64 shadow serves a miss better than a drop.
- RESULT t3_all Tue 02:2x (out/T3ALL_read.txt; t3_all 595 s, t4_all 684 s): BETWEEN on code. All-row code flips t3_all
  0.92 % (31/3366) vs the t4_all control 0.33 % (11/3366) = +0.59 points (line: PASS <= +0.3 with code-sure <= 1 %,
  KILL > +1); code-sure rows 0/2922 in both. By class, t3_all vs t4_all: code tie 38.0 vs 18.0 %, mid 3.05 vs 0.51 %;
  IFEval all rows 7.24 vs 2.15 % (+5.09; tie 47.9 vs 23.1 %, mid 5.27 vs 0.30 %); multi 5.83 vs 1.58 % (+4.25; tie 42.2
  vs 19.3 %, mid 5.47 vs 0.43 %); sure rows 0 flips everywhere (5,437 rows). The t4 path's own floor (bit-exact 4-bit
  weights, another summation order) is 0.33 / 2.15 / 1.58 %. Read across to 45 GB's plan 3333333333: sure rows hold,
  code moves a little, chat's mid rows move ~10x; EvalSuite's same-scores decides (chat is where it can fail).
- RESULTS Tue 03:1x (map, the night slot):
  - The 3-bit ranks 0-1 arm (M-18; DAY9_read.txt): PASS. Gated code rows flip 1/2211 at 3 bits = 1/2211 at the 4-bit
    control (gate_top2t4_cf099); cf >= 0.999 0/1607; all code rows 1.31 vs 1.19 % (+0.12), IFEval 5.50 vs 5.56, multi
    3.69 vs 3.63; gated multi 7 vs 5 of 554. The 3-bit records on sure rows add nothing beyond Q1's own deviation; the
    damage of t3_all lives on the uncertain rows. The t4 path vs the stock path: top-1 differs on 156 of 10316 rows.
  - The domain-adaptive arm (acert_top2_p05_p095_cf099; CERT_read.txt): PASS on its line (code certified 0/2863), pass 2 in
    59 % of code blocks, all code rows 1.57 % (theta 0.8: 1.63). Against theta 0.8 (b43930c's rule): multi all rows 5.32
    vs 7.04 % (CIs apart), IFEval 7.27 vs 8.51 % (CIs overlap) -> it does NOT formally beat theta 0.8 as the working point;
    chat certified-row flips 0.81 / 1.70 % vs 2.55 / 4.63 %. With the chat mode (one exact pass) chat's deviation is moot.
  - Q1 at 0.9 (gate_top2_cf09) ran; Q1-UNION's two runs failed on a 3-D score shape in the new mask path (fixed, tested,
    relaunched 03:1x).
- RESULT Q1-UNION Tue 03:1x (Q1U_read.txt; pre-registered a6dd04e): theta 0.9 PASS (gated code flips 7/2678 = 0.26 %
  [0.13-0.54] vs Q1 at 0.9 14/2678 = 0.52 %; line <= 0.5 %); theta 0.99 BETWEEN (1/2211 = 0.045 % vs the line 0.03 %
  and Q1's 2/2211 = 0.09 %; one flip). All rows, Q1-UNION vs Q1: theta 0.99 code 0.74 vs 1.34 %, IFEval 3.36 vs 5.42 %,
  multi 2.34 vs 3.69 %; theta 0.9 code 0.98 vs 1.78 %, IFEval 4.92 vs 8.86 %, multi 3.04 vs 7.60 %. Bytes and groups =
  Q1's at the same theta by construction (kept experts ~8.4 per row-layer). Q1-UNION at 0.9 has fewer flips than Q1 at
  0.99 on every measure while reading Q1-at-0.9's bytes: the lower theta Fable C predicted. Deterministic and
  token-repeatable (no cache state).
- PRIOR ART (NewIdeas Tue 07:0x): Q1-UNION's union step is published as OEA, Opportunistic Expert Activation (arXiv
  2511.02237, ICML 2026): every token keeps top-k0 and adds lower ranks only if the batch already loads them. New here:
  k0 set per row by the drafter's cf (the sure rows' k0 2).

## OEA AND THE UNION STACK (NewIdeas Tue 07:0x): PRE-REGISTRATION (Tue 07:1x, before the runs)
- Arms (masks ~/NewIdeas/data/dm_masks/<arm>, NewIdeas' generator; its q1u_0.9 equals my out/q1u_cf09 on 36 of 36
  windows; wired as q1_union(<abs dir>), mass, test_reuse ALL OK): oea_k8 / oea_k6 = every row keeps top-k0 + the
  rank > k0 experts already in its check's union; hyb_0.9_u8 / hyb_0.9_u6 = drafter-sure rows (cf >= 0.9) k0 2, the
  others k0 8 / 6. Union bytes per check vs exact (NewIdeas' CPU, same blocks; code / IFEval / multi): oea_k8 x0.828 /
  0.834 / 0.814, oea_k6 x0.647 / 0.661 / 0.631, hyb_0.9_u8 x0.527 / 0.756 / 0.762, hyb_0.9_u6 x0.435 / 0.603 / 0.594
  (Q1-UNION 0.9: x0.618 / 0.903 / 0.934).
- Lines (NewIdeas' proposal, kept): per domain, all-row flips <= Q1 at 0.99's level (code 1.34, IFEval 5.42, multi
  3.69 %) = PASS, up to 1.5x = BETWEEN, above = KILL; gated code flips (cf >= 0.99) reported beside. Read-out
  dm_oea_read.py -> out/OEA_read.txt. Slot: the orchestrator's (today's one MEM job, or Wed before 00:30).

## FLOOR NIGHT 1 (mission 13 narrowed; Fable's FLOOR_125B.md section 5; the orchestrator Tue 07:0x): PRE-REGISTRATION
## (Tue 07:2x, before the runs; Wed 00:30 -> deadline Wed 15:00, after BigMoE's measurement if that runs first)
- Windows: the 78 with HPD-0 cf walks: HumanEval 24, IFEval 30, multi 24 (measured rows 6,327 / 6,923 / 6,731 = 19,981,
  2x the 36-window map; 27,404 positions). GSM8K / agent / MBPP wait for MTP Head's walks (asked 07:2x) and a second night.
  Data: ~/Breakthrough/data/floor1 (78) and floor1_36 (the extras), DM_OUT; gate file built for 78 (dm_gate DM_MAX 30).
- Runs, in order: (1) capture = the baseline + routing (its flips vs itself 0), then the gate check (baseline vs D-02a's
  stored top-1 >= 99.9 %); (2) section 5's runs 1-4 on the 78: gate_bands, gate_top2_cf0999, gate_top5_cf099,
  gate_bands+hc8w; (3) extras 5-7 on the 36: gate_bands+moe_reuse_chain1_L08_35_dcf0999 (the bounded chain on cf >= 0.999
  rows stacked on the bands), gate_top0_cf0999_L08_35 (skip, k 0, on cf >= 0.999 rows in layers 8-35), gate_top4_cf099;
  (4) Q1-UNION 0.9 at scale (masks on CPU after the capture) and, if today's map passes an OEA / hybrid arm, the best one.
  Guards: the deadline, a swap guard (+2 GB stops the tree), others' quiet windows, PAUSE > 5 min, the lid.
- Lines (section 5's, as written; counts scale with the rows):
  - NEW WORLD-with-margin: band flip rates <= 0.02 % in the >= 0.999 band and <= 0.1 % in 0.99-0.999 on code, <= 0.3 %
    per band on IFEval / multi; the whole sequence at the twin's level: 125B-sure rows <= 0.1 %, non-tie rows <= 1 % on
    code and <= 5 % on chat; AND the C 256 replay on the night's routing <= 0.45 GB of expert reads per kept token at
    12.5 and <= 30 MB of misses per kept token.
  - EDGE: the >= 0.999 band flips > 0.1 %, or 125B-sure rows move > 0.3 % through the context, or bytes > 0.6 GB per
    kept token.
  - KILL of the mechanism: the 0.99-0.999 band at top-4 / top-5 > 1 %, or non-tie deviation > 2 % on code.
  - Side verdicts: run 5 adopts reuse only if < 1 % stacked on the gated rows with no extra sure-row movement; run 6
    decides skip; Q1-UNION / OEA at scale against their 36-window lines (a6dd04e, 78ccef8).
- Added to the same load (NewIdeas Tue 07:3x; optional; pre-registered before the runs): hyb_0.9_u8_t3 and q1u_cf09_t3 =
  the arm's mask with the sure rows' (cf >= 0.9) ranks 0-1 served from records3 at 3 bits through dm_t3 (gate ("t3",
  0.9)), their union-member ranks 2-9 at 4 bits; masks dm_q1u_masks.py <arm>:t3 (36/36: unsure rows = the arm's, sure
  rows' ranks 2-9 = the arm's, ranks 0-1 off). The 3-bit sure-rows arm was map-tested at cf 0.99 / 0.999 only. Line as
  the others: per domain all-row flips <= Q1 at 0.99's level = PASS, <= 1.5x BETWEEN, above KILL. My mask builder now
  reproduces NewIdeas' masks bit for bit (oea_k8 / k6, hyb_0.9_u8 / u6, q1u 0.9: 36/36 each) and builds them for the 78.
- RESULTS Tue 10:1x (one load, 09:31-10:18; OEA_read.txt / CERT_read.txt): per-domain all-row flips vs Q1 at 0.99's
  levels (code 1.34 / IFEval 5.42 / multi 3.69 %): oea_k8 0.62 / 3.83 / 3.07 % PASS / PASS / PASS (union bytes x0.83,
  gated code 0/2211); oea_k6 0.71 / 6.06 / 5.38 % PASS / BETWEEN / BETWEEN; hyb_0.9_u8 1.34 / 6.18 / 4.08 % PASS /
  BETWEEN / BETWEEN; hyb_0.9_u6 1.28 / 8.01 / 5.71 % PASS / BETWEEN / KILL; hyb_0.9_u8_t3 1.19 / 6.68 / 4.39 % PASS /
  BETWEEN / BETWEEN; q1u_cf09_t3 1.19 / 5.06 / 3.24 % PASS / PASS / PASS (3-bit sure rows on Q1-UNION 0.9 cost +0.2
  points on code, ~+0.1-0.2 on chat). The leftover two-pass arms all PASS their line: theta 0.95 0/2771 (all code 1.22 %),
  top-5 0.9 0/3024 (0.92 %), acert 0.7/0.9 3/2911 (1.49 % = plain 0.9's), acert 0.5/0.8 8/2999 (1.66 % vs plain 0.8's
  1.63 %, pass 2 in 52 vs 55 % of code blocks: worth taking over plain 0.8 by b43930c's rule). The night's step C takes
  q1u_cf09 + q1u_cf09_t3 + oea_k8 at scale (hyb_0.9_u8 is BETWEEN on chat, so not its +3-bit). P1-OEA's three JSONs are
  NewIdeas' to read.
- RESULTS FLOOR NIGHT 1 [K] (Wed 03:10-07:06; 78 windows = code 6,327 / IFEval 6,923 / multi 6,731 rows; data/floor1
  FLOOR_read.txt, DAY9_read.txt, Q1U_read.txt; extras on the 36 in data/floor1_36):
  - Bands (gate_bands, 78): >= 0.999 band 0.000 % (0/3163) / 0.000 % (0/710) / 0.202 % (1/494); 0.99-0.999 band 0.255 %
    (3/1176) / 0.842 % (6/713) / 0.330 % (2/606); 125B-sure rows 0.02 / 0.00 / 0.04 %; non-tie 0.67 / 1.76 / 1.43 %.
    NEW WORLD-with-margin NOT met (the 0.99-0.999 band is over 0.1 % on code and 0.3 % on chat). EDGE by one flip (multi's
    >= 0.999 band, 1/494). KILL not triggered (top-5 on the 0.99-0.999 band <= 0.56 %, code non-tie 0.67 % < 2 %).
    Verdict BETWEEN (dm_floor_read). The bytes part (C 256 replay) did not run: floor_rows.py needs the 36-window
    perturbation arms (topk2 ...), which the 78 never had; R's design gap, not a data fault.
  - Q1 at 0.999 (gate_top2_cf0999, 78): code drafter-sure rows 0.00 % [0.00-0.12] (0/3163) = PASS; IFEval 0/710, multi
    1/494. Q2 (DAY9's band structure): KILL (mid band 1.29 % vs >= 0.99 rows 0.07 % = 18.7x, kill > 2x).
  - Extras on the 36: chain reuse stacked on the bands (run 5) NOT adopted (code sure rows 0.17 % vs ~0.02 %, >= 0.999
    band 2/1607); skip k 0 on cf >= 0.999 in L8-35 (run 6) NOT adopted (>= 0.999 band 2/1607 = 0.124 % > the 0.1 % EDGE);
    top-4 on cf >= 0.99 (run 7): 0.99-0.999 band 0.17 / 0.54 / 0.36 %, under KILL, over the margin.
  - Step C at scale (78, all-row flips vs 1.34 / 5.42 / 3.69 %): q1u_cf09 (live) 1.04 / 6.37 / 4.12 % = PASS / BETWEEN /
    BETWEEN; q1u_cf09_t3 1.09 / 6.56 / 4.37 % = PASS / BETWEEN / BETWEEN; oea_k8 (16-row) 0.44 / 3.78 / 2.99 % = PASS x3
    (the 36's 0.62 / 3.83 / 3.07 % held); hyb_cold4_t3 1.14 / 11.18 / 10.73 %; u8+t3 live control 1.25 / 7.92 / 5.60 %;
    pmtc 0.98 / 8.44 / 6.21 %; len 1.04 / 8.44 / 6.21 %.
  - COLD on the 42 held-out windows (verdict lines 12b3d63; data/floor1/COLD_read_heldout.txt): hyb_cold4_t3 code PASS
    1.01 % (z -0.60 vs the control: no cost). PROMPT pmtc vs the control: code 0.81 vs 1.11 % (z -1.88), IFEval 8.45 vs
    8.08 % (z +0.92), multi 6.70 vs 5.66 % (z +2.78) -> CODE+, so be67ff0's GENERAL is WITHDRAWN (multi costs on
    held-out windows). len: the same chat rows (short prompts take pmtc); code 0.81 % (z -1.88) at cached bytes x0.196 vs
    the control's x0.272 (-28 %) and pmtc's x0.253 -> CODE+. So the prompt table is a code lever (len's form), not a
    general FAST step.

## 3-BIT PASS 1 (M-18 for the 12 GB two-pass; Fit12's item 4): PRE-REGISTRATION (Tue 10:4x, before the runs)
- Runs (36 windows, built and unit-tested; t3 path = test_t3 ALL OK): t3_top2_all = every row's top-2 from records3 at 3
  bits (the certificate source, as topk2 is for the 4-bit pass 1); t3cert_top2_p08 = rows whose own p1 under the 3-bit
  cut >= 0.8 take their top-2 at 3 bits, the rest exact. Read-out dm_cert_read.py (its row "t3cert_top2_p08").
- Lines: the sweep's (certified code rows of the mixed run <= 1 % PASS, > 5 % KILL); the 3-bit pass 1 is worth taking
  over the 4-bit one (cert_top2_p08: 8/2976 certified, all code rows 1.63 %, pass 2 in 55 % of code blocks) when it
  PASSes, its all-code-row flips stay within cert_top2_p08's 95 % CI, and its blocks needing pass 2 are <= +5 points.
  Bytes: pass-1 records at ~0.75 of 4-bit (MSB-3).
- RESULT Tue 11:00 (NewIdeas' read of the live-cut arms, ~/NewIdeas/data/LIVE_read.txt): q1u_0.9_live 1.25 / 6.45 / 3.77 %
  (code / IFEval / multi) = PASS / BETWEEN / BETWEEN, worse than the 16-row q1u_cf09 everywhere (paired z +2.2 / +4.3 /
  +2.6): the map's 16-row unions overstate Q1-UNION's chat quality; reach_0.9_0.5_live 1.28 / 6.80 / 4.31 % (a code-only
  add-on). P1 mixed runs: P1-K1 certified code flips 7/2997 = the control's 7/3019 (PASS), all code rows 1.52 vs 1.96 %.
  Consequence for FLOOR night 1: data/floor1/LIVE_CUT is set, so step B builds q1u_cf09 and q1u_cf09_t3 on live-cut
  checks (my builder: NewIdeas' live_segments copied, bit-identical 36/36); oea_k8 as is. The t3 arms now take their
  3-bit rows from the mask (gate ("t3mask",)), so a live cut's exact bonus row is never served twice.
- RESULT 3-bit pass 1 Tue 11:1x (t3_top2_all 339 s, t3cert_top2_p08 310 s; CERT_read.txt): PASS and worth taking over the
  4-bit pass 1 by its rule. Code: certified share 88.2 vs 88.4 %; certified-row flips 4/2970 = 0.13 % [0.05-0.35] vs
  8/2976 = 0.27 % (pass 1 alone 6 vs 10); all code rows 1.75 vs 1.63 % (inside cert_top2_p08's CI); pass 2 in 57 vs 55 %
  of code blocks (+2 <= +5). Chat all rows: IFEval 8.83 vs 8.51 %, multi 6.95 vs 7.04 %. With P1-K1 (pass 1 = top-1 +
  union, PASS, reads x0.465) the two quality-cleared pass-1 levers stack to ~0.71 x 0.465 x 0.75 = ~0.25 GB of pass-1
  reads per check at C 51 [P; the stack itself is not map-tested]; Fit12 times both in its 14:3x lock.
- FLOOR NIGHT 1 HEAD (S0; the orchestrator 14:5x, NewIdeas 14:54, pre-registered in its QUEUE): p1_oea_k1_ku4_t3 = the
  12 GB pass-1 stack, all rows: P1-K1 capped to each row's first 4 kept ranks (Fit12's KU 4; NewIdeas' masks oea_k1_ku4),
  every kept rank on the cut rows served from records3 at 3 bits (dm_t3 routed_topk K 10 with the rank mask; test_t3:
  1.9e-2 rel err vs a dequantize reference), uncut rows exact; then top_k8 (plain top-8, 4-bit). 36 windows. Lines
  (NewIdeas'): code, theta 0.8, vs p1_top_k2: certified flips <= 0.48 % and blocks needing pass 2 <= 63 % = PASS, > 0.96 %
  or > 73 % = KILL; top_k8 per domain vs Q1 at 0.99's levels. NewIdeas reads both (p1_cert_read.py, live_read.py).

## STAGED VERIFICATION (BREAKTHROUGH PUSH, the orchestrator Tue 17:0x; ~/FableGod/push/BREAKTHROUGH_PUSH.md): PRE-REGISTRATION (Tue 17:1x, before any replay)
- Question: does a cheap pass pick the right path out of the drafter's top-3 tree, so that ONE exact pass on that path
  keeps more per check? EXACT by construction (the exact pass keeps only tokens equal to its own argmax, + its bonus); the
  cheap pass only changes how much is kept. Prior art: trees (SpecInfer, EAGLE-2, Sequoia) and cascade / self-speculative
  verification are published; a tree pass at a cheap-row price (P1-K1 / 3-bit / resident-only) feeding one exact chain
  is adapted.
- Replay (CPU, decision_map/dm_staged.py): HPD-0's 16-row checks on the map windows (anchor A; draft j predicts ids[A+j+1]);
  truth = the window's tokens; drafter = DFlash2 L1-20's 16 candidates + teacher-forced selector softmax per depth (mission
  24's TL_WALK_TOP16 table: code = ~/TrainLab/data/d02b_walk_kw_code.npz now; IFEval / multi = MTP Head's aligned data when
  it lands); a node's children = its top-3 by that softmax.
- Trees: T-inf = the width-3 lattice to depth 15 (the ceiling); T_N = static best-first trees of N = 32 / 64 / 128 nodes
  (+ the anchor row) from the truth's per-depth rank frequencies, leave-one-window-out per domain (Sequoia / EAGLE-style);
  the chain 0^15 (self-check).
- Cheap pick: greedy from the anchor; at each node the child with the highest cheap logprob at that node's row (the arm's
  top-8 on the map; children outside it rank below; none inside = the drafter's first child, counted 'unresolved'). Exact
  pass: one pass on the chosen path; kept = its leading drafts equal to the truth + 1.
- Cheap arms = all-row map runs, teacher-forced through the whole window (so conservative vs a tree pass with exact KV
  before the anchor and a wider union): primary p1_oea_k1 (P1-K1 = top-1 + the check's union members); information
  t3_top2_all, topk2, t3_all, oea_k6, oea_k8, p1_top_k2, and S0's p1_oea_k1_ku4_t3 / top_k8 when they land. A resident-only
  C 256 cheap pass needs a GPU map arm (after FLOOR; pre-registered separately).
- Metrics per domain (code; chat = IFEval + multi pooled): kept per exact pass (chain / oracle / staged, per tree and arm);
  M2 = share of checks whose chosen path follows the truth as far as the tree holds it (staged depth = oracle depth);
  fidelity = (staged - chain) / (oracle - chain); unresolved picks; cheap and exact rows per check. Information: + a cheap
  tail (where the truth leaves the tree, the cheap pass's own argmax as one more draft); a cheap cut (the chosen path cut
  where the product of the cheap pick probabilities < 0.3) vs the chain with DFlash2's live cut (t 0.3), as kept per exact row.
- Excluded: each window's first check (its anchor row is not a measured map row).
- Self-checks: (S1) the chain tree gives the walk's k + 1 at every check for every arm; (S2) a perfect picker gives staged =
  oracle at every check and tree; (S3) per domain the drafter table's top-1 chain = the walk's k at every check (code
  391/391), else that domain is not read.
- Lines (primary arm p1_oea_k1, tree T_64): PASS (worth a GPU cost screen + Fable B's row price) = staged >= x1.25 the
  chain's kept per exact pass on chat AND fidelity >= 0.75 on chat, with code >= x1.05; KILL = the T_64 oracle itself
  < x1.15 on chat and < x1.05 on code (no top-3 headroom inside 64 rows), or fidelity < 0.5 on both domains (the cheap pass
  cannot pick); BETWEEN otherwise. tok/s is priced afterwards with Fable B's row costs (ROW_WALL.md) [P], gains halved.
- Prediction [G]: code chain ~11.4 per check, T_64 oracle +5-10 %, staged +4-8 %; chat chain ~2.8, T_64 oracle +30-45 %,
  staged +20-35 %, fidelity ~0.8; M2 ~0.85 code / ~0.9 chat: BETWEEN-to-PASS on chat.
- RESULT Tue 17:1x (decision_map/dm_staged.py, 20 s CPU; out/STAGED_read.txt; S1 / S2 / S3 ok; checks: code 379, IFEval
  1361, multi 1833 on the 36 windows; IFEval / multi tables = MTP Head's d02b_walk_kw_{ifeval,multi}.npz, 17:11): KILL by
  the line (the primary cheap pass cannot pick). p1_oea_k1 on T_64: code chain 8.43 -> oracle 9.17 (x1.087) -> staged
  8.78 (x1.041), fidelity 0.47, M2 0.873; chat 2.16 -> 2.70 (x1.247) -> 2.34 (x1.084), fidelity 0.34, M2 0.827. Lattice:
  code oracle x1.244, staged x1.135 (0.56); chat x1.309, x1.118 (0.38).
  Why (post-hoc breakdown, lattice true-path nodes): P1-K1 saves 82 / 72 / 74 % of the drafter's misses (code / IFEval /
  multi; truth = its #2 or #3) but breaks 1.2 / 6.3 / 7.0 % of its RIGHT picks (top-2-class arms the same): the tree's
  headroom sits exactly at the rows where cheap passes leave the exact model. Near-exact passes pick: oea_k8 / oea_k6 /
  t3_all save 93-98 % and break 0.1-1.5 %, fidelity 0.85-0.97 (T_64: code x1.08, chat x1.21-1.23), but they read
  x0.65-0.83 of exact's bytes per row, so a 65-row tree pass costs more than the 16-row exact pass it replaces.
  Headroom: even a perfect picker on T_64 gives chat x1.247 / code x1.087 (the line's x1.25 is out of reach for this
  tree); the lattice (x1.31 / x1.24) needs one cheap step per depth (MTP Head's 'beam').
  Information: the shipped chain with its live cut keeps code 8.23 at 10.1 exact rows (0.82 per row), chat 2.08 at 3.3
  (0.63); oea_k8 on T_64 with the cheap cut keeps code 8.98 at 9.1-10.0 rows (~0.94), chat 2.55 at 2.6-3.5 (~0.83); its
  tail token adds +0.8 / +1.0 per check. So a near-exact picker would make the exact pass nearly waste-free; it is a
  lever only if such a pass is nearly free (resident-only at C 256: quality unmeasured, a GPU map arm after FLOOR).

## RESIDENT-ONLY PICKER (the staged-verification lead; the orchestrator Tue 17:2x: "yes ... run it Wed after 15:00 with NewIdeas' QOM arms"): PRE-REGISTRATION (Tue 17:3x, before the masks' GPU run)
- Arm res_c256 (36 windows, decision_map/out): every realized row of each HPD-0 check (A .. first miss) keeps ONLY the
  experts resident in the base pool at its check's start, all ranks, 4-bit, the model's own weights (renorm off, as the
  union arms); prompt rows exact. The pool moves by the exact pass alone: each check then loads its union (all ranks of
  its 16 rows, the rows after the first miss by the wrong-draft stand-in p + 97). LRU replay = NewIdeas' (qom arms,
  ~/NewIdeas/scripts/dm_oea_masks.py) copied: per-layer base pool C_m of ~/MoonShots/data/laysz45_256.json (mean 295 of
  512), carried across a domain's windows in walk order, prompt rows warm it, plain LRU (no popularity prior: slightly
  pessimistic). Builder decision_map/dm_res_masks.py; self-check: its qom mode at theta 0 on live checks = NewIdeas'
  qom_0_live masks bit for bit.
- Read-out: dm_staged.py with res_c256 as the picker (the pre-registered replay, 54036cf), plus its all-row flips per
  domain vs the baseline argmax (dm_oea_read.py's method) and the kept share of realized row-layer experts.
- Lines (T_64, as the replay): PASS = fidelity >= 0.85 on code AND on chat (a resident-only tree pass picks like a
  near-exact one: then Fable B prices its compute for a ~65-row tree); KILL = fidelity < 0.6 on chat; BETWEEN otherwise.
  Information: staged/chain, the cheap cut (kept per exact row vs the shipped cut chain's 0.82 / 0.63), the tail token.
- Prediction [G]: kept share ~0.93-0.96 of row-layer experts; all-row flips between oea_k8's and topk5's; fidelity
  ~0.75 code / ~0.70 chat: BETWEEN. Flag: outputs depend on the cache state (not token-repeatable), FAST at best as a
  generator; as a picker the output stays EXACT (the exact pass decides every kept token).
- Added Tue 17:5x, before the run: res_c256 also reads as a FAST decode mode on its own (skip every miss, load it for
  later checks: no demand reads). Lines = dm_oea_read.py's (all-row flips per domain vs Q1 at 0.99's level: PASS <= 1x,
  BETWEEN <= 1.5x, KILL above). Prediction [G]: code BETWEEN (96.5 % kept, rank 0 98.8 %), chat PASS (99.1 %).
- WITHDRAWN Tue 17:4x, before any run (the orchestrator agreed): Fable B's ROW_WALL.md prices a resident-only row at
  2.64-3.0 ms today / ~1.0 at the floor engine (its resident expert pairs still run), so a ~65-row resident-only tree pass
  costs 65-195 ms per cycle against a 96.8 ms cycle, whatever its fidelity; and MoonShots' M-16 already measured the
  resident-only pass's agreement with exact (code 0.97-0.996, chat 0.85-0.985 of rows at C 224), which answers the FAST
  line. The masks stay in out/res_c256; Wed's day slot runs NewIdeas' three QOM arms only.

## COLD-CUT (FAST; Fable C's ~/FableGod/fablec/UNION_STACK.md section 5; the orchestrator Tue 18:5x: "run small breakthrough tests now"): PRE-REGISTRATION (Tue 19:1x, before the masks are built)
- Arm hyb_cold4_t3 (the 36 windows, decision_map/out, LIVE checks: live_segments, t 0.3). Sure rows (DFlash2 cf >= 0.9,
  the map's realizable gate) keep ranks 0-1 as base, served at 3 bits from records3 (dm_t3, the gate ("t3mask",), as
  hyb_0.9_u8_t3). Unsure rows keep ranks 0-3 plus every rank 4-9 whose expert is popular: share >= 0.05 in Fable C's
  ~/FableGod/fablec/masks/pop_cap45.npy (the share of cap45's 176 live checks whose exact layer-union holds it; six
  prompts, HumanEval 68 / 91 and four MATH-500, none of them a map window). Every realized row also keeps the experts
  already in its check's union (the rows' base sets; rows after the first miss by the stand-in p + 97). Prompt rows exact.
  Builder decision_map/dm_cold_masks.py; runner scripts/cold_slot.sh; read-out dm_cold_read.py -> out/COLD_read.txt.
- Checks before the GPU run: (1) the builder's u8_t3 rule equals dm_q1u_masks.py's hyb_0.9_u8:t3 at live lengths bit for
  bit on all 36 windows; (2) Fable C's mask check: union bytes on the map's code blocks (base experts x 2,162,688 B + the
  4-bit rows' ranks 0-1 x 770,048 B, no cache) <= x0.55 of EXACT's. If either fails, the arm does not run.
- Lines (Fable C's), all-row flips on code vs the baseline argmax: PASS <= Q1 at 0.99's 1.34 % (then it replaces u8 + t3
  in the FLOOR arms: -22 % bytes at C 320 in Fable C's replay); BETWEEN <= 2.0 % (then r 6 / p 0.05 next); KILL above
  (cold experts matter: drop the family, MISS-CUT too). Gated code flips (cf >= 0.99) expected unchanged (0-1 of 2,211).
  IFEval / multi columns are information (chat takes OEA).
- Added (mine): the CONTROL hyb_0.9_u8_live_t3 (hyb_0.9_u8_t3's rule at the same live checks, = the arm COLD-CUT would
  replace; no live u8 + t3 number exists) in the same process. COLD's own cost = McNemar only-COLD vs only-control flips
  on code: |z| < 2 = no measurable cost over u8 + t3 (then the bytes cut is free whatever the absolute line says).
- The chat form cold6_all (every row ranks 0-5 + popular 6-9 + members, no gate, no 3-bit rows) runs only if
  hyb_cold4_t3 PASSes, in a later slot; lines = dm_oea_read.py's on IFEval / multi (PASS <= Q1 at 0.99's 5.42 / 3.69 %,
  BETWEEN <= 1.5x, KILL above; oea_k8's 16-row 3.83 / 3.07 % quoted, no live oea_k8 exists).
- Prediction. Fable C [G]: 50 % PASS / 35 % BETWEEN / 15 % KILL. Mine [G]: BETWEEN, ~1.5-1.9 % on code (PASS 20 % /
  BETWEEN 65 % / KILL 15 %): the live checks already put Q1-UNION 0.9 at 1.25 % (42 flips vs the line's 45), and the
  3-bit sure rows (+0.2 pp at 16 rows) and the unsure-row cut come on top. COLD vs its control: |z| < 2. Union bytes on
  code ~x0.45-0.50 of EXACT's (no cache; Fable C's x0.35 at 45 GB includes the cache).
- Slot: the masks are CPU (1 core, ~2 min) after 19:40, once no quiet window is held or waited for. The flips need
  the 125B map run (~10 min, MEM 17 GB, GPU share 50), which waits for NO-SWAP >= 25 GB like the QOM arms (16 GB at 19:02
  with Kamen's apps open). It does not start after 23:45.
- Added Tue 19:4x, before the GPU run (CPU checks; decision_map/dm_cold_masks.py, dm_cold_lru.py -> out/COLD_lru.txt):
  self-check ALL EQUAL (36 windows); Fable C's mask check on code x0.471 <= x0.55: OK. No-cache union bytes vs EXACT
  (code / IFEval / multi): hyb_cold4_t3 x0.471 / 0.617 / 0.632; u8_t3 (the control's rule) x0.440 / 0.675 / 0.702.
  Unsure rows keep 9.29 / 7.36 / 7.23 of 10 experts under COLD vs 8.48 / 8.38 / 8.34 under u8: on code COLD drops FEWER
  pairs than u8. With the cache [K replay, plain LRU at laysz45_256's C_m, prompts warm it, carried across windows]:
  SSD bytes vs EXACT, code / IFEval / multi: hyb_cold4_t3 x0.254 / 0.498 / 0.532; u8_t3 x0.295 / 0.597 / 0.641
  (COLD -14 % / -17 % / -17 % vs u8 + t3; Fable C's cap45 replay -24 % on code). So cold = missed holds on the map's
  blocks: on code COLD keeps more experts AND reads fewer bytes. Prediction updated [G]: COLD vs the control on code
  |z| < 2 (it drops fewer pairs), the absolute line set by the control's level; chat, where the cross-domain prior cuts
  deeper than u8, is the risk for the chat form. Halved gain for decisions: ~-7 % bytes vs u8 + t3 at 45 GB.
  Tag: adapted (Fable C; SwapMoE / AdapMoE lineage). General: needs only a corpus popularity table, so any MoE model.
- RESULT Tue 19:55 (map run 19:45-19:55, ~10 min; out/COLD_read.txt): **hyb_cold4_t3 PASS on code by the pre-registered
  line: 1.25 % all-row flips (42 of 3,360; line 1.34 %); gated code flips 0 / 2,211.** The CONTROL hyb_0.9_u8_live_t3
  (u8 + t3 at the same live checks, first live number): 1.37 % (46). COLD's cost over it on code: only-COLD 11 vs
  only-control 15, McNemar z -0.78 = no measurable cost; with the 45 GB LRU it reads -14 % bytes (x0.254 vs x0.295 of
  EXACT's). So on code COLD-CUT dominates u8 + t3 (equal quality, fewer bytes) [K map + K replay].
  Chat (information; the domain switch gives chat OEA): COLD 10.60 % IFEval / 11.37 % multi vs the control's 7.74 / 5.54
  (z +6.1 / +11.7) and Q1 at 0.99's 5.42 / 3.69 = KILL level. Cause [G, to test]: Fable C's prior is code + math
  (cap45's six prompts), so on chat it marks needed experts cold: COLD's unsure chat rows keep 7.36 / 7.23 of 10 vs u8's
  8.38 / 8.34; the dropped "cold" experts are chat / multilingual specialists. Prediction scored: mine (BETWEEN, cost
  |z| < 2) half right (PASS; no cost); Fable C's 50 % PASS right on code.
- FOLLOW-UP PRE-REGISTRATION (Tue 19:5x, before any build or run): is the chat failure the PRIOR's domain or cold-dropping
  itself? Arm hyb_cold4_lowo_t3 = hyb_cold4_t3's rule with an IN-DOMAIN prior: per window, the share of the SAME domain's
  OTHER map windows' live checks whose exact layer-union holds the expert (leave-one-window-out on the map's own routing
  captures; pop_cap45's definition). Beside it, the pre-registered chat form cold6_all (cap45 prior). One process, ~9 min.
  Lines, chat (IFEval and multi), vs the control's 7.74 / 5.54 %: McNemar |z| < 2 on both = the prior's domain was the
  cause, COLD works with a domain-covering prior (GO: a corpus prior covering chat in the engine's table, next a mixed
  prior); z >= 2 on either = cold experts matter on chat even in-domain (COLD stays code-only). Code: information (an
  in-domain prior on code; PASS line 1.34 % as before). cold6_all: as pre-registered (IFEval / multi PASS <= 5.42 / 3.69 %,
  BETWEEN <= 1.5x, KILL above). Prediction [G]: lowo chat z < 2 on IFEval, >= 2 on multi (languages spread thin over 12
  windows); cold6_all BETWEEN on both. Bytes (45 GB LRU) printed beside for both arms.
- Added Tue 20:0x, before the follow-up's GPU run (CPU; dm_cold_masks.py rewritten with a per-window prior, its rebuilt
  hyb_cold4_t3 / cold6_all masks equal the ones the 19:45 run used bit for bit, 36 / 36 each): hyb_cold4_lowo_t3 bytes check
  on code x0.485 <= x0.55 OK. Unsure rows keep 9.54 / 8.54 / 8.36 of 10 (code / IFEval / multi; u8: 8.48 / 8.38 / 8.34;
  COLD-cap45: 9.29 / 7.36 / 7.23). Cached SSD bytes vs EXACT [K replay, 45 GB LRU]: lowo x0.267 / 0.363 / 0.372;
  COLD-cap45 x0.254 / 0.498 / 0.532; u8_t3 x0.295 / 0.597 / 0.641; OEA k8 (live) x0.708 / 0.726 / 0.729; cold6_all
  x0.625 / 0.703 / 0.703. So with an in-domain prior COLD keeps as many experts per unsure chat row as u8 yet reads
  -39 / -42 % chat bytes vs u8 + t3 and about half of OEA k8's: if its chat flips show no cost over the control, it is the
  chat byte lever too (halved: ~-20 %). Caveat: the prior comes from the same benchmark's other windows (other prompts,
  same style); a corpus prior is the engine's version.
- FOLLOW-UP RESULT Tue 20:12 (map run 20:02-20:12; out/COLD_read.txt): **by the pre-registered line, COST on multi ->
  COLD stays code-only.** hyb_cold4_lowo_t3 (in-domain leave-one-window-out prior) vs the control on chat: IFEval 7.68 %
  vs 7.74 % (only-lowo 96 / only-control 98, z -0.14: no cost, at x0.363 vs x0.597 of EXACT's cached bytes = -39 %);
  multi 6.67 % vs 5.54 % (108 / 68, z +3.02: cost). Code 1.37 % (= the control, z 0.00). Prediction (mine) right:
  IFEval no cost, multi cost. So cold-dropping is free where the table matches the text (code with cap45's code prompts,
  IFEval in-domain) and costs where it under-covers it (chat with a code + math table; multilingual even in-domain:
  4 languages x 3 windows, the rare experts carry the language).
  Chat form cold6_all (cap45 prior): IFEval 5.00 % PASS / multi 4.84 % BETWEEN (vs Q1 at 0.99: z -0.89 / +2.78), code
  0.65 %; cached chat bytes x0.703 vs OEA k8's x0.726-0.729 at live checks (-3 %): no byte lever on chat. Not pursued.
  Engine note: BigMoE's eviction prior (data/seed45/prior45.npz, scripts/seed45_prior.py) is the same cap45 code + math
  table, so a COLD rule on the engine's table repeats the chat failure.

## COLD-CUT WITH THE REQUEST'S OWN PROMPT AS THE TABLE (the orchestrator Tue 20:1x: "Yes to the last shot at 'general' ... Pre-register it first (lines on multi, IFEval and code vs the same controls)"): PRE-REGISTRATION (Tue 20:2x, before the masks)
- Tag: adapted. One search (arXiv 2607.00466 ELDR, IJCAI 2026 DoMoE, arXiv 2604.23150) found the prefill expert footprint
  used to place, cluster or prefetch, not to cut experts; nothing found doing the cut. Token-repeatable (the prompt's
  routing is fixed before decoding), needs no corpus: if it holds on every domain, a general method for any MoE model.
- Two arms, hyb_cold4_t3's rule (sure rows top-2 at 3 bits + members; unsure rows ranks 0-3 + popular ranks 4-9 +
  members; live checks), the table from the prefill: hyb_cold4_pmt_t3 = popular iff the request's own prompt rows routed
  to the expert at least once (any rank); hyb_cold4_pmtc_t3 = that OR Fable C's cap45 >= 0.05 (a corpus table plus the
  request's footprint). CPU profile before the run (kept experts and bytes only): unsure rows keep 9.07 / 7.86 / 7.53
  (pmt) and 9.73 / 8.91 / 8.89 (pmtc) of 10 vs u8's 8.48 / 8.38 / 8.34; cached bytes vs EXACT (45 GB LRU) x0.199 / 0.329 /
  0.343 (pmt) and x0.285 / 0.520 / 0.539 (pmtc) vs u8_t3's x0.295 / 0.597 / 0.641. Map prompts are short on chat
  (median 52 IFEval / 27 multi rows: ~145 / ~88 experts per layer seen) and longer on code (181 rows, ~249).
- Lines per arm, vs the same control hyb_0.9_u8_live_t3 (code 1.37 / IFEval 7.74 / multi 5.54 %), McNemar on each
  domain: GENERAL = z < 2 on all three (no measurable cost anywhere) at fewer cached bytes than the control on all three;
  CODE+ = no cost on code and IFEval only; else COLD stays code-only (cap45 table). If both arms are GENERAL, pmt (fewer
  bytes) is the method. Absolute lines beside (Q1 at 0.99: 1.34 / 5.42 / 3.69 %).
- Prediction [G]: pmtc GENERAL (it keeps more experts than u8 everywhere), at -3 / -13 / -16 % bytes vs u8 + t3 (halved
  ~-1 / -7 / -8 %); pmt costs on multi (the shortest prompts, 7.5 kept) and maybe IFEval, so CODE+ at best.
- Slot: both arms in one process (~11 min, MEM 17 GB) this evening if the gates clear; else in the 00:30 prefix (with QOM
  and the layout check that is ~25 min). Read-out lines in dm_cold_read.py (PROMPT lines).
- RESULT Tue 20:28 (map run 20:18-20:28; out/COLD_read.txt PROMPT lines): **hyb_cold4_pmtc_t3 = GENERAL by the line**
  (the request's prompt footprint OR the cap45 table): vs the control, code 1.13 % vs 1.37 % (z -1.79), IFEval 8.42 % vs
  7.74 % (z +1.63), multi 5.77 % vs 5.54 % (z +0.62): no measurable cost on any domain, at cached bytes x0.285 / 0.520 /
  0.539 vs the control's x0.295 / 0.597 / 0.641 (-3 / -13 / -16 %; halved ~-1.5 / -6.5 / -8 %) [K map + K replay].
  Honest edge: IFEval sits near the line (+0.7 pp, z 1.63); a scale check (78 windows) would firm it up.
  hyb_cold4_pmt_t3 (prompt footprint alone) = code-only: code 1.25 % (z -0.78) but IFEval 9.16 % (z +3.33) and multi
  6.92 % (z +3.56): short chat prompts (median 52 / 27 rows) don't reveal the experts the answer needs. Information: on
  CODE (prompts median 181 rows) it reads x0.199 of EXACT's cached bytes at no cost: -33 % vs u8 + t3, -22 % vs
  COLD-cap45 (x0.254) at the same 1.25 %. Predictions (mine) right on both arms.
  Mechanism: COLD-cap45 failed chat (z +6 / +12) because a code + math table calls chat specialists cold; adding the
  request's own prefill footprint restores them (the prompt is in the same language and domain as the answer) while the
  experts neither table knows stay dropped, and those are the ones that miss. So per request the table is: a generic
  corpus table + the request's prompt footprint, which needs no domain switch and no per-domain corpus. Tag: adapted
  (footprint known from ELDR / DoMoE-style work; used as a cut: not found in one search). General: any MoE with a router.
- Added Tue 22:3x (CPU, information; out/COLD_lru_c51.txt): the byte win depends on the RAM point. With 12 GB's cache
  (C 51 per layer, plain LRU) every COLD table reads MORE than u8 + t3: code / IFEval / multi pmtc x0.324 / 0.627 / 0.638,
  COLD-cap45 x0.294 / 0.543 / 0.574 vs u8_t3 x0.274 / 0.557 / 0.573 (pmtc +18 / +13 / +11 %). "Cold = missed" needs a
  cache that holds the warm set (45 GB's C_m ~295 per layer vs a warm set of ~250-300); at C 51 the warm experts miss too,
  and COLD keeps more of them than u8 does. So COLD-CUT (any table) is a 45 GB-class lever, not a 12 GB one.

## LENGTH-AWARE COLD TABLE + THE AT-SCALE CHECK: PRE-REGISTRATION (Tue 22:3x, before any mask or run at 78 windows)
- Arm hyb_cold4_len_t3: hyb_cold4_t3's rule with the table = the request's prompt footprint alone when its prompt has
  >= 128 rows (pmt), else footprint OR cap45 (pmtc). POST-HOC on the 36: every code prompt there has >= 136 rows and
  every chat prompt <= 120, so on the 36 it IS pmt on code + pmtc on chat (code no cost at -33 % bytes; chat GENERAL).
  It needs an out-of-sample test before any claim.
- Test: FLOOR night 1's step C (78 windows, tonight) runs hyb_cold4_t3, the control hyb_0.9_u8_live_t3, hyb_cold4_pmtc_t3
  and (appended now) hyb_cold4_len_t3. VERDICTS ON THE 42 HELD-OUT WINDOWS ONLY (the 78 minus the map's 36), each vs the
  control at the same windows, McNemar per domain: GENERAL = z < 2 on code, IFEval and multi AND fewer cached bytes than
  the control (45 GB LRU replay) on all three; else as before (CODE+ / code-only). pmtc's held-out verdict confirms or
  withdraws tonight's GENERAL (be67ff0); len's decides the length-aware table. Prediction [G]: both GENERAL; len at
  about -25-33 % bytes on code, as pmtc on chat.
- Added Tue 22:3x (CPU, information; out/COLD_lru.txt): U3 (Q1-UNION 0.9 + t3, unsure rows exact = the engine's FAST
  mode) reads x0.390 / 0.810 / 0.867 of EXACT's cached bytes at 45 GB, so the per-request table pmtc is -27 / -36 / -38 %
  vs U3 and len (code: the footprint alone) -49 % on code. On quality it matches u8 + t3, not U3 (U3 keeps all ten
  experts on unsure rows): an engine FAST step decided by task results.

## STALE-SLOT CHECK OF THE MAP'S C 51 PATH (the orchestrator 22:3x, forwarding Leap 50de28c: a CPU slot rewrite inside one command buffer is not seen by later dispatches): PRE-REGISTRATION (Tue 22:4x)
- At risk: the map runs BigMoE's streamed engine at C 51 with the spin gate (persistent per-layer slot pools that
  the ext fills in place; up to 160 ids per gate call vs 51 slots), so a layer's slot can be refilled between gate calls.
- Evidence already in hand [K]: three later runs equal the baseline bit for bit (tk and lp) on all 36 windows, 10,316
  rows. They are capture, topk10 (the identity arm) and xcapture (extra GPU traffic), run at different times and loads.
  So no timing-dependent staleness shows. A stale pattern that repeats identically per run is not excluded by this.
- Decisive check: the baseline with BIGMOE_GATE=event (its id copy pushes the command buffer past MLX's limit, so MLX
  commits a buffer per gate call: no slot is reread inside the buffer it was refilled in) on 9 windows (--max 3),
  DM_OUT=out_stale/event, against the stored out/baseline.json. PASS = tk and lp bitwise equal on every window: the
  spin path computed with the right weights, so past map verdicts stand. FAIL = any difference: the affected arms get
  re-run in event mode and their verdicts are flagged. Prediction [G]: PASS.
- Slot: after BigMoE's night ticket ends, before FLOOR's first step (scripts/stale_check.sh; ~3-8 min, MEM 17 GB).
- WITHDRAWN Tue 22:5x, before any run (code read, ~/BigMoE/bigmoe/stream.py + mtplx generation.py). The map's runs NEVER
  take the spin gate. Each window is one prefill chunk (MTPLX's auto chunk 2,048 rows > every window), so every MoE call
  carries hundreds to thousands of ids and takes stream._sync_call, the synchronous prefill path. The spin gate needs
  <= 160 ids AND (<= 92 or <= C), i.e. <= 9 rows. _sync_call has a command-buffer boundary between every slot rewrite
  and the next read of that slot, which is Leap's own listed fix:
  - np.array(indices) waits for the GPU before any load;
  - mx.eval runs between slot groups ("before the next group can evict these");
  - mx.eval(y) ends each layer.
  So the map's verdicts (all arms, incl. the certificates at C 51 and COLD-CUT) are structurally clear of the
  within-buffer staleness, and the three bitwise re-runs agree. The event-mode check would test the same path. It would
  also fail at import (event mode needs BIGMOE_MAXK <= 61; the map runs 160).
  The risk sits in DECODE engines (verify calls of <= 16 rows take the spin gate): BigMoE's 45 GB engine and the live
  12 GB bt_b12 (B12's 4-row calls), plus any two-pass that refills pools between passes without a sync. Their check is
  the same engine with and without a command-buffer boundary per pass or layer, bitwise (owners BigMoE / Fit12).
- Fit12 22:5x on its 12 GB paths: T3 never refills a pool pass 1 read inside one command buffer. A CPU sync
  (mx.eval + tolist) separates the passes, and every <= 4-row exact call ends with one. Inside a call there is one post
  per pool per layer, its ids fit the pool (<= 38 vs C_m 47, <= 32 vs C_l 32), and the loader evicts from the LRU tail
  (slots last read a call earlier). Evidence that discriminates [K, Fit12]: the MTP decode path (= the live 12 GB
  engine's) bitwise equal at C_m 51 / 59 / 65; G1 at C_m 47 / C_l 32 = the exact loop alone. Pool size changes which slots
  are rewritten when, so staleness would make outputs depend on C. The live spin path at C_m ~51 is therefore covered.
  Fit12's ~5-min sync-every-MoE-layer A/B is the confirmation if the orchestrator schedules it.
- K12 RESULT (Fit12, Wed 13:15) [K]: PASS 12/12. bt_b12 8db7 on v3p, C_m 47 / C_l 32, MAXK 40 gate, T3's calls, 4 HumanEval
  prompts x 3 arms in a fresh process. mx.eval at every MoE block's entry and exit (15,964 synced calls) = bitwise equal
  to T3R2's unsynced run (tokens, logits hash, per-check log).
- The modes K12 left open (MAXK 80 spec / MTP; Fit12: "yours to judge"), code read Wed 13:2x: CLEAR BY CONSTRUCTION, no run.
  - Every post_ids ends a command buffer. ext start() gives post_ids a 64 MB dummy input unless BIGMOE_NOPOSTCOMMIT is set
    (nothing sets it), so MLX commits right after each post (v3p ext/bigmoe_ext.cpp, g_post_bump). A call's spins and
    expert reads open a fresh buffer, and nothing earlier in that buffer read the pool.
  - Within a call the GPU reads only protected slots. The loader protects the waited and late ids, then resident ids in
    list order, and releases any id it evicts as -1 (flag-mode path). The late re-read after layer L+1's attention reads
    fresh victims outside that set.
  - MAXK 80 only widens the row (S <= 4 verify rows x 20 ids, one extra id line). The MTP head's chained drafts
    (spec125 draft_start: no CPU wait between steps) each post, so each commits. A CPU sync also ends every verify
    (spec125: mx.eval(g, hid); g.tolist()).
  - [K] backing: K12 on the same mechanism at MAXK 40; Fit12's cross-C bitwise runs on the MTP decode path (above).
  - Leap's hazard needs a slot read earlier in the SAME buffer: NOPOSTCOMMIT, or two reads of one pool between posts.
    Neither exists in the live bt_b12. Keep the post-commit on the live path. A gate that drops it (e.g. a zero-commit
    pickup) must read weights with coherent(system) loads (Leap B4: 0 stale at 1.09x plain bandwidth) or re-run K12.

## RECORDS3 LAYOUT CHECK (BigMoE's converted store + build_nat; prefix item, the orchestrator Tue): RESULT Wed 03:28 = PASS [K]
- ctrl = baseline + t3_top2_all on 3 windows with today's store and ext (out_layout/ctrl); test = the same with
  BIGMOE_EXT_BUILD=build_nat and DM_R3=r3_overlay (out_layout/test). Test vs ctrl and test vs the stored out/: all
  bitwise equal (decision_map/out_layout/LAYOUT_check.txt). BigMoE's layout patch (Wed 13:40) is clear on the map side.
- Also run this night, read by NewIdeas (no verdict here): qom_0_live, qom_0.5_live_t3, qom_0_live_t3, qom_0.5_live_c164
  (36 windows each, 03:22-03:58), and FLOOR night 1's S0 (p1_oea_k1_ku4_t3, top_k8; 03:10-03:21).

## WED NIGHT MAP SCREENS IN ONE FLOOR GAP (the orchestrator Wed 04:0x / 04:08): PRE-REGISTRATION (Wed 04:1x, before any run)
Runner scripts/screens_slot.sh (one map process, 36 windows, live checks, MEM 17 GB share lock, the next gap between FLOOR
night 1's steps); read-out decision_map/dm_screen_read.py -> out/SCREEN_read_<arm>.txt. All-row flips vs the baseline argmax
per domain; Q1 at 0.99's level (gate_top2_cf099) = 1.34 / 5.42 / 3.69 % (code / IFEval / multi; recomputed, equal).
- Screen 1(a), Fable C's BYTES10 (line written ~/FableGod/fablec/BYTES10.md:217-219, Wed 00:3x, before any run):
  qom_0_live_bonus = qom_0_live's masks with every drafted bonus row gated to ranks 0-1 + the block's union members (masks
  ~/FableGod/fablec/masks/qom_0_live_bonus, bytes10_bonus_masks.py, a subset of qom_0_live's keep, checked on one window).
  PASS = <= qom_0_live's own + 0.3 points AND <= the level; BETWEEN = otherwise <= 1.5x the level; KILL = above. Paired
  McNemar vs qom_0_live. qom_0_live's own flips [K, out/qom_0_live.json]: 0.42 / 2.53 / 1.77 %, so PASS needs <= 0.72 /
  2.83 / 2.07 %; BETWEEN up to 2.01 / 8.13 / 5.54 %. Prediction [G, Fable C]:
  PASS on code, BETWEEN on chat. Why it matters [P, NewIdeas' corrected ledger]: the 24 GB FAST start is 54.5 / 84.4 /
  81.7 MB per kept token with the bonus row exact vs 35.7 / 19.4 / 13.9 with it gated.
- Screen 2, NewIdeas (lines ~/NewIdeas/QUEUE.md:1181-1185, Wed 04:0x, before any run): hyb_0.9_k1_u10_live_t3 = sure rows
  (cf >= 0.9) keep rank 0 at 3 bits (dm_t3, t3k 1) + the block's union members at 4 bits; unsure rows and the bonus row
  exact. Twin q1u_0.9_live_t3 (the same with t3k 2). PASS = <= the level; BETWEEN = <= 1.5x; KILL = above; paired McNemar
  vs the twin. Prediction [G, Fable C]: BETWEEN on code (+0.3-0.6 points over top-2), KILL on chat. Joins the run only if
  the gap starts before 09:00 (FLOOR's D + B + C + R must still end by 13:00); otherwise it waits for 13:45.
- RESULTS [K] (Wed 04:37-04:55, one map process, 36 windows, 0 row-count warnings; out/SCREEN_read_<arm>.txt):
  - Screen 1(a) qom_0_live_bonus: 0.65 / 6.12 / 6.50 % = code PASS / IFEval BETWEEN / multi KILL (qom_0_live 0.42 / 2.53
    / 1.77 %). McNemar vs qom_0_live: code 14 / 6, z +1.79; IFEval 158 / 36, z +8.76; multi 193 / 25, z +11.38. Fable C's
    prediction held on code; chat was worse (multi KILL). Gating the bonus row to ranks 0-1 + union is free on code only.
  - Screen 2 hyb_0.9_k1_u10_live_t3: 1.60 / 8.36 / 4.84 % = code BETWEEN / IFEval KILL / multi BETWEEN; twin
    q1u_0.9_live_t3 1.34 / 6.62 / 4.33 %. McNemar vs the twin: code 21 / 12, z +1.57; IFEval 114 / 55, z +4.54; multi
    72 / 54, z +1.60. Fable C's prediction held (code BETWEEN at +0.26 points; chat KILL on IFEval).

## SCREEN 1 IN QOM'S FORM, qom_0_live_bq (NewIdeas; the orchestrator Wed 04:47): PRE-REGISTRATION (copied Wed 04:5x; NewIdeas' line
~/NewIdeas/QUEUE.md:1214-1229, Wed 04:4x, before any run)
- Arm: qom_0_live with each live segment's last row gated by the draft it verifies; the row keeps ranks 0-1 + the block's
  union members + the resident ranks 2-9 (QOM's own keep, 0 extra SSD reads). Masks ~/NewIdeas/data/dm_masks/qom_0_live_bq
  (q1_union, READY, 36 windows). Runner scripts/map_slot.sh, the gap after the screens run (or the next FLOOR gap).
- Lines (screen 1's): all-row flips <= qom_0_live + 0.3 points (0.72 / 2.83 / 2.07 %) AND <= 1.34 / 5.42 / 3.69 % = PASS;
  <= 1.5x the level (2.01 / 8.13 / 5.54 %) = BETWEEN; above = KILL. Paired McNemar vs qom_0_live and vs qom_0_live_bonus.
  Prediction [G, NewIdeas]: ~0.5 / 3.9 / 3.6 %: code PASS, chat BETWEEN.
- Screen 1(a) for reference [K, 04:4x]: qom_0_live_bonus 0.65 / 6.12 / 6.50 % = code PASS / IFEval BETWEEN / multi KILL.
- Conditional (pre-approved by the orchestrator if bq is not KILL on NewIdeas' read-out; NewIdeas says go):
  qom_0_live_bq_t3 = the same with every sure row's ranks 0-1 at 3 bits (hyb_t3, t3k 2), paired vs qom_0_live_t3.
- RESULT qom_0_live_bq [K] (Wed 05:02-05:06, 36 windows, 0 row-count warnings; out/SCREEN_read_qom_0_live_bq.txt):
  0.39 / 4.09 / 3.26 % = code PASS / IFEval BETWEEN / multi BETWEEN (NewIdeas' prediction ~0.5 / 3.9 / 3.6 % held). McNemar
  vs qom_0_live: code 6 / 7, z -0.28; IFEval 96 / 43, z +4.50; multi 77 / 24, z +5.27. Vs qom_0_live_bonus: code 5 / 14,
  z -2.06; IFEval 77 / 146, z -4.62; multi 55 / 170, z -7.67. QOM's resident keep on the gated row removes most of
  screen 1(a)'s chat cost (IFEval 6.12 -> 4.09 %, multi 6.50 -> 3.26 %), and code has no cost at all.
- qom_0_live_bq_t3 RUNS (Wed 05:0x, the same gap; not KILL on any domain = the orchestrator's pre-approval). Line, by
  analogy with screen 1 (NewIdeas' QUEUE gives the pairing only): <= qom_0_live_t3 + 0.3 points AND <= the level = PASS;
  <= 1.5x the level = BETWEEN; KILL above. qom_0_live_t3 [K]: 0.71 / 4.45 / 4.14 %, so PASS needs <= 1.01 / 4.75 / 3.69 %;
  BETWEEN up to 2.01 / 8.13 / 5.54 %. McNemar vs qom_0_live_t3 and vs qom_0_live_bq (written before the result).
- RESULT qom_0_live_bq_t3 [K] (Wed 05:07-05:12): 0.77 / 6.09 / 5.29 % = code PASS / IFEval BETWEEN / multi BETWEEN.
  McNemar vs qom_0_live_t3: code 9 / 7, z +0.50; IFEval 102 / 46, z +4.60; multi 92 / 51, z +3.43. Vs qom_0_live_bq (the
  3-bit cost): code 17 / 4, z +2.84; IFEval 130 / 62, z +4.91; multi 118 / 46, z +5.62. On code the gated row is free in
  both forms; 3 bits adds measurable flips, though code stays under the level.

## MORNING MAP BATCH AFTER FLOOR NIGHT 1 (the orchestrator Wed 06:2x): PRE-REGISTRATION (copied Wed 06:3x; NewIdeas' lines, before any run)
One process, right after FLOOR night 1's runner exits (scripts/map_slot.sh WAIT_PID): qom_0_k1_live_t3, then qom_0_live_bq_t2,
36 windows each (one process; screen 2b's verdict is read on code + multi only as pre-registered, its IFEval column is
information). Then "map batch done" to BigMoE; flips to NewIdeas. Read-out decision_map/dm_screen_read.py.
- Screen 2b qom_0_k1_live_t3 (~/NewIdeas/QUEUE.md:1189-1192 + 1250): QOM at cf 0 with sure rows at top-1 (rank 0 at 3 bits;
  union members and resident ranks 1-9 at 4 bits; t3k 1). Lines: all-row flips <= Q1 at 0.99's level (code 1.34 / multi
  3.69 %) = PASS; <= 1.5x = BETWEEN; above = KILL. Paired McNemar vs qom_0_live_t3 (0.71 / 4.45 / 4.14 %). Prediction
  [G, NewIdeas]: BETWEEN on code, KILL on chat.
- qom_0_live_bq_t2 (QUEUE.md:1305-1320, 05:5x): qom_0_live_bq_t3's masks; the gated rows' ranks 0-1 at 2 bits rebuilt from
  records3's 3-bit codes, w = s3 (2 (q3 >> 1) + 0.5) + b3 (MoonShots' to2bit.py c5af2d3, copied byte for byte into
  decision_map; dm_t3._proj bits == 2 via scripts/install_dm_t3_2bit.sh in the gap). Self-test [K, 06:3x, through dm_t3.read]
  OK on L12 e7 and L40 e300: 3-bit unpack exact, 2-bit triple within 0.26 % of the rule, gather_qmm within 0.7 % of dense,
  ~0.32 relative RMS from the 3-bit weights. Lines (bq_t3's): PASS = <= qom_0_live_bq_t3 + 0.3 points AND <= the level,
  i.e. <= 1.07 / 5.42 / 3.69 %; BETWEEN <= 2.01 / 8.13 / 5.54 %; KILL above. Paired McNemar vs qom_0_live_bq_t3 (the 2-bit
  cost) and vs qom_0_live_bq. Prediction [G, NewIdeas]: code BETWEEN (+0.5-1.0 points over 3 bits), chat KILL.
- RESULTS [K] (Wed 07:06-07:26, one process, 36 windows, 0 row-count warnings; out/SCREEN_read_<arm>.txt):
  - Screen 2b qom_0_k1_live_t3: 0.59 / 4.65 / 3.24 % = code PASS / multi PASS (IFEval 4.65 % under its level:
    information, outside the pre-registered domains). McNemar vs qom_0_live_t3: code 9 / 13, z -0.85; IFEval 56 / 49,
    z +0.68; multi 39 / 71, z -3.05. NewIdeas' prediction (BETWEEN code, KILL chat) missed. Fewer flips than top-2 at
    3 bits on multi, likely because rank 1 stays at 4 bits when resident. The 24 GB path's lever 2 holds in QOM's form.
  - qom_0_live_bq_t2: 1.49 / 10.13 / 9.77 % = code BETWEEN / IFEval KILL / multi KILL. McNemar vs qom_0_live_bq_t3: code
    34 / 10, z +3.62; IFEval 208 / 71, z +8.20; multi 228 / 69, z +9.23 (vs qom_0_live_bq z +5.40 / +11.23 / +12.77).
    NewIdeas' prediction held. 2 bits on the sure rows' top-2 is too coarse on chat.

## SCREEN 4: QOM AT THE 24 GB POOL (NewIdeas; the orchestrator Wed 07:5x): PRE-REGISTRATION (copied Wed 08:0x; NewIdeas' lines
~/NewIdeas/QUEUE.md:1343-1390, 07:4x + amendment 07:5x, before any run)
- Why: every QOM quality result so far kept the 45 GB pool's residents (laysz45_256, C_m mean 295), while the 24 GB prices
  replay 147 base slots per layer. Screen 4 re-tests at the 24 GB pool (masks suffix _c147, bit-checked, READY).
- Arms, one process after BigMoE's "morning block done" (scripts/map_slot.sh, MEM ~17 GB, 36 windows each):
  (C) qom_0_live_bq_c147_t3 = qom_0_live_bq_t3 at the 24 GB pool (t3mask, t3=3, t3k=2, its own masks);
  (B') qom_0_k1_live_bq_c147_t3keep = the cheap top-1 form: mask qom_0_k1_live_bq_c147 with the t3keep gate (p1_t3:
  t3=3, t3k=10), so rows the mask cuts take every kept rank from records3 at 3 bits (no LSB read on sure rows); sure rows
  with nothing dropped stay at 4 bits.
- Lines, per domain, all-row flips: PASS <= 1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x (2.01 / 8.13 / 5.54 %), KILL above
  (the 24 GB verdicts). Paired McNemar: C vs qom_0_live_bq_t3 (the pool's cost at top-2); B' vs C and vs qom_0_k1_live_t3.
- Predictions [G, NewIdeas]: C code PASS (~1.0 %), IFEval BETWEEN (~7 %), multi BETWEEN or KILL (~6 %); B' code BETWEEN
  (~1.5 %), IFEval KILL (~9 %), multi BETWEEN or KILL (~5.5-6.5 %).
- Prices [P, NewIdeas' ledger_top1.py, 24 GB, engine convention]: top-2 + 3-bit 35.7 / 19.4 / 13.9 MB per kept token;
  top-1 with rank 1 at 3 bits (B''s form) 24.0 / 11.5 / 7.7.
- RESULTS [K] (Wed 08:33-08:47, one process, 36 windows, 0 row-count warnings; out/SCREEN_read_<arm>.txt):
  - (C) qom_0_live_bq_c147_t3: 1.07 / 8.07 / 5.99 % = code PASS / IFEval BETWEEN / multi KILL (5.99 > 5.54). The pool's
    cost vs qom_0_live_bq_t3: code 20 / 10, z +1.83 (no measurable cost); IFEval z +4.93; multi z +2.24. The code FAST
    stack (top-2 at 3 bits + the gated last row) is map-PASS at the 24 GB pool, at 35.7 MB per kept token [P].
  - (B') qom_0_k1_live_bq_c147_t3keep: 1.69 / 11.25 / 8.61 % = code BETWEEN / IFEval KILL / multi KILL. Vs C: code 28 / 7,
    z +3.55; IFEval z +6.83; multi z +6.31. Vs qom_0_k1_live_t3: z +5.08 / +12.48 / +11.39. The cheap top-1 form (24.0 MB
    per kept token on code [P]) costs measurably on code and fails chat.
  - NewIdeas' predictions held for both arms.

## MAP SCREENS 6 + 7 (NewIdeas; the orchestrator 14:3x ROUTE: "build the per-layer weight-source hook ... Screens 6 + 7 run
tonight as one share-lock batch ... ~22:00"): HARNESS BUILT + BIT-CHECKED, PRE-REGISTRATION (copied Wed 14:5x; NewIdeas'
lines ~/NewIdeas/QUEUE.md:1709-1724 (screen 6) and 1742-1791 (screen 7, 14:1x + amendment 3de406e), before any run)
- Screen 6 = DRAFT17's quality at the 24 GB pool: qom_0_live_ball_c147_t3 = C (screen 4) with every last row gated, bonus
  rows included; tbits_mask(masks, 3, 2). Lines: PASS <= 1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x, KILL above (code's verdict
  is the one that matters). Prediction [G, NewIdeas]: code ~1.15-1.3 % PASS; IFEval ~8.1 %, multi ~6.0 % (C's).
- Screen 7 = the trellis FAST store's quality on layers S = {7, 18, 30, 41}. Masks qom_0_live_ball_c239_t3 (C + DRAFT17 at
  the mix's 239-slot pool; READY). Four arms, 36 windows each:
  - s7_ref (7-ref): tbits_mask(masks, 3, 2), C's form: the gated rows' ranks 0-1 from records3 MSB-3, the mask's kept
    ranks 2-9 at 4 bits on the stock path.
  - s7_trl (7-trl): s7_ref, except that at S the gated rows take EVERY kept expert (ranks 0-1 + the mask's 2-9) from Fit12's
    TRL view (the 2.40 b mix: gate / up k 2, down k 3; W = Q R^T decoded to an MLX 8-bit affine view, gather_qmm bits 8),
    with nothing on the stock path.
  - s7_cal (7-cal): the same rows and experts at S from records3 at MSB-2 (to2bit).
  - s7_cal_all (7-cal-all): s7_cal at all 48 layers.
- Read-out (NewIdeas' map_kl.py): est_KL = KL_ref + (KL_trl - KL_ref) x (KL_cal_all - KL_ref) / (KL_cal - KL_ref), then
  est_flips by the fit (NewIdeas data/ledger/map_kl_fit.txt). The lines are on est_flips: PASS <= 1.34 / 5.42 / 3.69 %,
  BETWEEN <= 1.5x, KILL above. A domain is "no estimate" if KL_cal - KL_ref < 2x its paired SE.
  Prediction [G, NewIdeas]: code ~1.0 % PASS, IFEval ~7.3 % BETWEEN, multi ~6.0 % KILL.
- Read as specified, flagged for NewIdeas: at S, 7-trl and 7-cal also move the kept ranks 2-9 (4 bits -> TRL / MSB-2;
  "every kept expert (any rank)"), while 7-ref keeps C's 4-bit ranks 2-9. So est_KL estimates "every kept expert of the
  gated rows TRL at all 48 layers" against C's form.
- HARNESS (Breakthrough, installed Wed 14:47 with no map process; scripts/dm_s7/: patch_s7.py, install_s7.sh, test_dm_s7.py,
  smoke_s7.py; pre-patch files in data/abl_scratch/s7/orig):
  - dm_t3.routed_kept(block, L, xg, src, keep): only the kept (row, rank) pairs, src = bits (records3 via read / _proj)
    or "trl:mix" (Fit12's trl_view.read / proj, read-only import). Reads go in groups of 128 experts (bounded memory); the
    combine is routed_topk's (K 10).
  - trl_gate refuses s7_trl unless all 512 experts of each S layer have whole files (safetensors size, shapes, and the
    metadata's layer / expert / proj / k). DM_TRL_FALLBACK=1 (smoke only) serves missing experts from MSB-3 and counts them.
  - dm_common: PERT['t3L'] = {layer: src}. At those layers the gated rows post nothing on the stock path (_row_rm) and
    _t3_add sends them to routed_kept, ranks < t3k forced in (the _t3 masks drop them). measure() stores per-window counts
    (res 'lsrc').
  - dm_ablations: qom_0_live_ball_c147_t3, s7_ref, s7_trl, s7_cal, s7_cal_all.
- BIT-CHECKS [K] (Wed 14:3x-14:49; data/dm_s7/):
  - CPU (36 checks):
    - at S the gated rows go to routed_kept, keep = ranks 0-1 + the mask's 2-9 (8.2-8.3 kept experts per row), and the
      stock path posts nothing for them;
    - other layers keep tbits_mask's form (routed_topk K 2; ranks 2-9 on the stock path);
    - the arms' PERT is as specified, and the TRL gate refuses today's partial store.
  - GPU unit (seconds):
    - routed_kept(src 3 / 2) == routed_topk(bits 3 / 2, K 10, keep), bit for bit;
    - the TRL path vs a dense fp32 reference (dequantize bits 8, at the kernels' bf16 rounding points): rms rel 4.3e-3.
      For scale: the live 3-bit path vs its own reference 1.8e-2; experts swapped (negative control) 1.36;
    - the MSB-3 fallback mix 9.9e-3; read-group size 3 vs 128 bit-equal.
  - Map smoke (one code window, HumanEval_135, 144 rows; 85 s, share lock, swap +0 MB):
    - C (qom_0_live_bq_c147_t3) re-run == the stored out/ JSON, bit for bit (tk and lp): existing arms are untouched;
    - s7_trl ran: 4 calls, 4,957 pairs, 970 expert reads (49 TRL + 921 counted fallbacks, since the store is partial by day);
    - s7_ref 9.2 s; s7_cal_all 55.4 s = 5.9x C (MSB-2's unpacking of every kept expert at 48 layers), MLX peak 15.4 GB vs
      12.5.
- COST, scaled from the smoke [K x 36]: s7_ref / s7_trl / s7_cal / screen 6 ~5-6 min each, s7_cal_all ~31 min; the batch
  ~55 min in one map process, MEM ~17 GB. (NewIdeas' ~28 min for screen 7 was low.) s7_trl waits for Fit12's full encode
  (ETA ~20:40).
- ADDENDUM Wed 15:5x, before any run (the orchestrator 15:3x / 15:0x; NewIdeas 14:5x):
  - s7_ref3 (NewIdeas' amendment 2): every kept expert of the gated rows at MSB-3 at all 48 layers. It uses lsrc_mask bits
    3 through routed_kept, the path that equals routed_topk bit for bit; same masks. The engine's FAST pool serves ranks
    2-9 at 3 bits, the map's C form at 4.
    Reading (NewIdeas, amended in QUEUE):
    - est_abs_TRL = KL_ref + (KL_trl - KL_ref) x (KL_cal_all - KL_ref) / (KL_cal - KL_ref);
    - in-frame = KL_ref + (est_abs_TRL - KL_ref3); the conservative est_abs_TRL is reported beside it;
    - without s7_ref3 the verdict uses the conservative one. s7_ref3 - s7_ref = the ranks 2-9 MSB-3 term at 239 slots.
  - Fable C's screens 1(b) / 1(c) (~/FableGod/fablec/BYTES10.md:255-260). The orchestrator 15:3x: the gate threshold is
    chat's last lever, per NewIdeas' sweep 385fa70.
    - Arms, q1_union form, the gate-file-cf rebuilds (never the 00:34 _bonus9 / _bonus5): q1u_0.9_live_bonus9g (7 bonus
      rows gated, their pairs x0.65) and qom_0.5_live_bonus5g (165 gated, x0.59).
    - Null checks. Per domain, all-row flips within +0.1 points of the ungated twin (q1u_0.9_live / qom_0.5_live) =
      PASS (expected), with paired McNemar. A larger cost = the gate file's cf and the walk's disagree. Read-out:
      dm_screen_read.py --null (new 15:4x: PASS / COST on the +0.1 line alone, the level printed for information).
    - Masks bit-checked 15:4x vs their twins: only the 7 / 165 bonus rows differ, by drops only; kept pairs x0.65 /
      x0.59, matching Fable C's build log.
  - Slot (scripts/dm_s7/s67_slot.sh, re-armed 15:5x for 20:00; the orchestrator 15:0x):
    - Order: 6, s7_ref, s7_ref3, s7_cal, s7_cal_all, 1(b), 1(c), then s7_trl once the TRL store is whole. One map process
      per arm.
    - An arm starts only before 23:10, and only if its estimate ends by 23:30 (the hard stop for RELAX's quiet arms).
      Leftovers go to Thu morning beside T0.
    - Estimates (min, with margin): 6 / 6 / 10 / 7 / 35 / 5 / 5 / 7. s7_ref3 ~8 from B''s 471 s on the same 3-bit
      all-ranks path (no MSB-2 unpacking), not ~31.
    - Expected end ~21:25 if the start is clear at 20:00 and the store is whole by ~21:15.
- ADDENDUM Wed 17:0x, before any run (NewIdeas 16:5x; the orchestrator's OK if it is in dm_ablations by ~19:30): DENSE-3 on
  the map, the batch's last optional arm.
  - Arm qom_0_live_ball_c147_t3+dense3 = screen 6's FAST code stack + Leap's DENSE-3 through ablation()'s "+" stack; twin =
    screen 6 (qom_0_live_ball_c147_t3), earlier in the same batch. DENSE-3 (~/Leap/scripts/dense3.py, 894b0ee, read-only,
    sha256 ec71be4478bc606f) sets the trunk's 4-bit g64 dense projections (GDN in / out, attention q / k / v / o; not the
    router, shared expert, indexer, experts, embed, head or MTP) to their MSB-3 values in the 4-bit format: codes & 0xEEEEEEEE,
    biases bf16(b + s / 2). Arm code = NewIdeas' data/arms/dense3_arm.py at 8fefaa3, pinned as scripts/dm_s7/dense3_arm.py,
    installed by scripts/dm_s7/install_dense3.sh (17:00). clear() restores the weights (TEARDOWN, as hc8w).
  - Reading (NewIdeas' QUEUE 16:5x, copied): map_kl.py as for the others. INVALID if its stored top-8 ids and log-probs equal
    screen 6's on every row (dense3_check.py). Verdict on code flips: PASS <= 1.34 %, BETWEEN <= 2.01 %, else FAIL. Priors:
    Leap FAIL (17-18 % relative rms per projection, 894b0ee), NewIdeas BETWEEN-KILL.
  - Bit-checks [K] (17:00):
    - CPU wiring test (test_dense3_arm.py, mock model, bf16 scales / biases): the stack's PERT == screen 6's; 120/120
      targets rewritten exactly; 51/51 decoys untouched (router, MTP, 8-bit, g32); clear() puts back the very same arrays
      (171/171); with BIGMOE_PKD1=1 or BIGMOE_SCOUT_BUNDLE=1 the setup raises before touching a weight. 9/9 ok.
    - GPU smoke (smoke_dense3.py, HumanEval_135, 144 rows, share lock MEM 16 GB, 38 s, swap +0): C re-run == stored JSON
      bit for bit; "[dm] dense3: 228 modules at MSB-3 (1509 MB of 4-bit weights rewritten)"; MLX peak 13.9 GB vs screen 6's
      12.6; 9.1 s per window (screen 6: 9.3); 0/144 rows identical to screen 6 (not INVALID); top-1 flips vs screen 6 0,
      mean |dlp1| 0.0069; screen 6 re-run after the DENSE-3 run == before, bit for bit (the restore works on the real model).
    - dense3_check.py: VALID on the smoke window; an arm vs itself = INVALID (negative control).
  - Slot (re-armed 17:02, same clock): the arm goes last in ARMS, est 10 min; it runs after 1(c) when s7_trl's store is not
    whole yet, else after s7_trl. Leap's dense3.py hash is logged at its go. Expected end ~21:35 with a clear start.
- ADDENDUM Wed 17:3x (the orchestrator 17:3x: "START THE MAP BATCH NOW instead of 20:00"; the GPU idle by day, BigMoE's
  lever 2 waits for apps closing, NO-SWAP 32 GB at 17:37, the only other jobs Fit12's 2 GB CPU encoder and a 0.5 GB Leap
  check): re-armed 17:38 with S67_START = the arming time; same arms, order, guards and clock (no new arm after 23:10,
  hard stop 23:30). s7_trl joins when Fit12's store is whole (~21:10). New: an interim read-out (the same read-outs as
  the end) the first time only s7_trl is left. Expected: the other eight arms end ~19:05 with clear checks; s7_trl ~21:20.
  Nothing in the pre-registration changes.
- ADDENDUM Wed 18:1x (by-day heat; nothing in the pre-registration changes): fanwatch set a heat pause every ~5 min today,
  3-6 min each (182-344 s), and froze every research job by working folder. The map runs ran from ~/FableGod/decision_map,
  outside the research folders, so they kept running under PAUSE (100 % CPU at 18:10) while the others were frozen, and
  the slot's "PAUSE over 5 min" guard stopped s7_ref at 18:05 after ~7 min of work. Screen 6 took 20.5 min (est 6), so its
  share lock lapsed ~4 min before the end. Fix (s67_slot.sh): each run starts from ~/Breakthrough (absolute paths only, checked),
  so fanwatch freezes and thaws it with the others; fanwatch's own PAUSE (PAUSE.fanwatch) no longer counts toward the 5-min
  stop; share locks last 3 x est + 10 min; stops of a frozen run use SIGKILL after 5 s (never SIGCONT). Swapped in by an
  atomic rename; the old slot is held after s7_ref by a placeholder process and then re-armed. Screen 6 (done, 36
  windows) stands: its run was not stopped, and heat pauses change no output.
- ADDENDUM Wed 18:4x (the orchestrator 18:4x, the order by the one-big-job rule; nothing in the pre-registration
  changes): finish s7_ref3 (done 18:43), s7_cal, s7_cal_all, 1(b), 1(c); then MTP Head's P-CAP (~32 GB, ~70 min) or
  s7_trl, whichever is ready first (Fit12's store ~21:10+, later with heat); DENSE-3 LAST, only once s7_trl is done and
  only if it can start before 23:10 (if the store is not whole tonight, DENSE-3 goes to Thu's leftovers with s7_trl).
  P-CAP's launcher waits for 3 clear minutes with no MEM >= 5 GB lock (MTP Head 18:4x), so it never starts between two
  of my arms. Slot: pick_next skips DENSE-3 while s7_trl is not done (dry test: six cases right); swapped in by an
  atomic rename and re-armed at the end of s7_cal (the old slot held by the placeholder, as at 18:21).
- RESULTS, interim (Wed 19:38, the slot's first wait for Fit12's TRL store) [K]. All arms 36 windows, 0 row-count
  warnings. Flips = all-row top-1 flips vs the baseline argmax, live checks; KL from map_kl.py. Data:
  data/dm_s7/results_wed/ (out JSONs gzipped, SCREEN_read files, the read-out, the slot's events).
  - Read-out table (code / IFEval / multi):

    | arm | code flips % / KL | IFEval | multi |
    |---|---|---|---|
    | C (qom_0_live_bq_c147_t3) | 1.07 / 0.0057 | 8.07 / 0.0386 | 5.99 / 0.0313 |
    | screen 6 (qom_0_live_ball_c147_t3) | 1.07 / 0.0061 | 7.83 / 0.0386 | 5.99 / 0.0313 |
    | s7_ref (screen 6's form, 239 slots) | 0.74 / 0.0040 | 6.48 / 0.0232 | 5.35 / 0.0236 |
    | s7_ref3 (every kept expert MSB-3, 48 layers) | 1.04 / 0.0051 | 7.62 / 0.0297 | 6.36 / 0.0324 |
    | s7_cal (MSB-2 at S) | 1.01 / 0.0052 | 7.12 / 0.0299 | 5.97 / 0.0308 |
    | s7_cal_all (MSB-2 at all 48) | 2.23 / 0.0180 | 13.92 / 0.1094 | 13.59 / 0.1465 |

  - SCREEN 6 (DRAFT17 at the 24 GB pool): PASS on code, the domain that matters.
    - code 1.07 % (36/3366) <= 1.34, the same rate as C (36/3366; McNemar 12 / 12, z 0).
    - IFEval 7.83 % BETWEEN (C 8.07; McNemar 13 / 21, z -1.37); multi 5.99 % KILL, identical to C (0 / 0: C had no
      ungated last rows in multi).
    - NewIdeas' prediction was code ~1.15-1.3 %, chat as C's. DRAFT17's gated last rows cost nothing measurable on code.
  - SCREEN 7: PENDING. s7_trl waits for Fit12's store; est_KL needs KL_trl.
    - Ready for it, the scale (KL_cal_all - KL_ref) / (KL_cal - KL_ref) = 11.1 / 12.8 / 17.1 (CORRECTED 19:5x from the unrounded KL, as NewIdeas: 11.7 / 12.9 came from the rounded table; 4 -> 48 layers; linear
      would be 12).
    - Information [K]: s7_ref3 - s7_ref = +0.30 / +1.14 / +1.01 points (KL +0.0011 / +0.0065 / +0.0088). That is the cost of
      serving the kept ranks 2-9 at MSB-3 instead of 4 bits at 239 slots (BigMoE's all-3-bit FAST pool question; code
      stays under 1.34 % at 239 slots).
      FRAME (NewIdeas 7fac438, Wed 20:0x): FAST certifies against REF2-T, which already serves ranks 2-9 at 3 bits, so this
      is REF2-T's own cost against the 4-bit model, not a FAST cost; tail_bits 3 is the like-for-like choice (NewIdeas'
      call 18:4x, d07577a).
  - 1(b) q1u_0.9_live_bonus9g, null check: PASS on all three domains.
    - code 1.25 % (= q1u_0.9_live; McNemar 0 / 0); IFEval 6.42 % (twin 6.45; 5 / 6); multi 3.86 % (twin 3.77, +0.09 <=
      +0.1; 6 / 3, z +1.00).
  - 1(c) qom_0.5_live_bonus5g, null check: COST on all three domains (each above the twin + 0.1 points).
    - code 0.48 % vs 0.36 % (+0.12; McNemar 5 / 1, z +1.63); IFEval 2.88 % vs 2.68 % (+0.20; 48 / 41, z +0.74); multi
      1.91 % vs 1.63 % (+0.28; 33 / 23, z +1.34).
    - Pooled 86 / 65, z +1.71 (two-sided p ~0.09): consistent in sign but not significant alone.
    - Per the pre-registration, a cost above +0.1 means the gate file's cf and the walk's disagree on the 165 bonus rows
      gated at 0.5 (BYTES10.md:255-260).
  - Left: s7_trl (when the store is whole; P-CAP may take the GPU first), then DENSE-3 (before 23:10 only).
- ADDENDUM Wed 19:5x (the orchestrator 19:5x; nothing in the pre-registration changes): MTP Head's P-OFF goes first
  (extraction ~20:40-21:15, training to ~22:15 if alive), then s7_trl once Fit12's store is whole (~22:15-22:30), all
  before BigMoE's 23:00 ticket. DENSE-3 is off tonight: a Thu leftover (BIGMOE_EXT_BUILD=build_nat after the conversion)
  or dropped if NewIdeas agrees (predicted FAIL). The slot was re-armed at 19:55 with ARMS ending at s7_trl, while it
  waited for the store with no arm running; the interim read-out was not repeated.
- ADDENDUM Wed 20:0x (NewIdeas): DENSE-3 is kept as a Thursday leftover (the ~13:10 spare after the records3 conversion,
  BIGMOE_EXT_BUILD=build_nat): "a measurement beats two predictions". The reading is unchanged (QUEUE 16:5x). NewIdeas runs
  s7_read.py (it writes into NewIdeas' repo) when s7_trl lands.
- ADDENDUM Wed 20:1x, before its run: NewIdeas' CHAT ROUTE arm (the orchestrator 20:1x; pre-registration copied from
  NewIdeas' QUEUE 20:1x, 4fb08ca; masks 4324e09).
  - Arm qom_0.5_live_c147_bonus5g = Fable C's 1(c) rule (QOM cf 0.5, the bonus row gated at 0.5, proxy g) on a 147-slot
    pool (the 24 GB split, SWEEP row 10a). Harness as 1(c): q1_union(masks dir), i.e. PERT {rmask, renorm False}; no t3,
    4-bit everywhere on the stock path; 36 windows.
  - Read: dm_screen_read.py on the lines, PASS <= 1.34 / 5.42 / 3.69 % on EVERY domain, BETWEEN <= 1.5x; paired McNemar
    vs qom_0.5_live_bonus5g (1(c)) and vs qom_0.5_live_c164 (information); map_kl.py beside. Prediction [G, NewIdeas]:
    ~0.7 / ~4.6 / ~3.0 %.
  - Install (20:13, install_chat.sh, test_chat_arm.py, 7/7 ok):
    - the PERT is exactly {rmask: NewIdeas' masks dir, renorm: False};
    - 36 windows, with names and shapes equal to 1(c)'s and to the routing's;
    - versus 1(c): drops only, ranks 0-1 identical, and 7,774 rows differ (NewIdeas' check: 7,774);
    - kept share 92.3 % over all rows, prompt rows included (NewIdeas' 82.5 % uses another denominator).
  - Slot: re-armed 20:13 with LASTGO 22:53 and DEADLINE 23:00 (BigMoE's ticket at 23:00, RELAX ~23:05).
    - The arm is listed in data/s67_extra.txt (est 5 min, start by 22:50, twins 1(c) then c164).
    - It runs right after s7_trl, only if it can start by 22:50; otherwise it is a Thursday leftover.
    - Dry test: ten cases right.
- ADDENDUM Wed 20:2x (NewIdeas: the chat arm needs no TRL store): from 22:40 the chat arm no longer waits for s7_trl, so it
  runs even if Fit12's store is late or not whole tonight; s7_trl still goes first while its store is whole, and after the
  chat arm if it still fits (LASTGO 22:53). Fit12 at 20:15: L18 + L30 whole, L7 388/512, L41 0/512; ETA 22:15-22:30, last
  chunk by 22:45. Re-armed 20:1x (dry test: five more cases right).
- ADDENDUM Wed 20:55 (the orchestrator 20:55): not now (Leap's quiet lock to ~21:01, then BigMoE's 44 GB early run). The
  chat arm's "alone" time moves from 22:40 to 22:15, after MTP Head's P-CAP resume (~21:50-22:15); the runner's yield checks
  keep it behind P-CAP if P-CAP runs late. s7_trl: Thursday is fine. The orchestrator places it at its 21:45 night-order
  pass: right after RELAX (~23:50) if Fit12's L41 is whole by then (before the conversion, no reader fix needed),
  otherwise after BigMoE's Thursday ticket. Tonight's runner still takes s7_trl if the store is whole by its last start.
  Re-armed 20:55 (S67_XALONE 22:15, LASTGO 22:53, DEADLINE 23:00; checked on the process).
- ADDENDUM Wed 21:18 (the orchestrator 21:18): BigMoE's early run moved to 23:00; MTP Head's P-CAP resume (MEM 36 GB) goes
  first; the chat arm starts as soon as P-CAP releases; Leap finishes Fit12's L41 with its fast encoder (bitwise), and
  s7_trl runs right after the chat arm (last start 22:53). Runner (re-armed 21:20, task bnfgvkno6):
  - data/s67_extra.txt holds "HOLD 0 22:15", so no arm starts before P-CAP releases. Without the HOLD, the runner would
    have started s7_trl as soon as the store was whole, taking the memory P-CAP waits for.
  - A watcher (task bzsrw1nzr) turns the HOLD into "ALONE 0 <time>" when P-CAP's MEM >= 30 GB lock disappears after being
    seen. From then on the chat arm goes first, then s7_trl. At 22:15 the HOLD expires by itself.
  - The old runner was parked by the placeholder during the swap.
  - Dry tests: chat first after ALONE, s7_trl after the chat arm, nothing during the HOLD (eleven cases right).
- ADDENDUM Wed 21:4x (the orchestrator 21:4x / 21:45): P-OFF's extraction was in the orchestrator's 20:0x order (ends
  ~22:10-22:20; its 22:33 lock end is a cap). The order holds: chat arm, s7_trl, then the Rule C passes if they can start
  by 22:50 (else Thursday after runner 42's start). If the chat arm has not started by 22:35, s7_trl goes first (a 22:35
  switch sets ALONE 23:59 in data/s67_extra.txt). Fit12's TRL store is whole (21:40; my trl_gate check 21:41 PASS, 6,144
  files; L41's e148-511 from Leap's Metal encoder, byte-checked by Fit12).
  - RULE C PASSES (data for MoonShots' CPU replay of Fable FULL's Round-2 rule C; ~/FableGod/full/FULL.md): the map
    harness's baseline configuration (as the map's out/baseline.json that M-39 replayed), teacher-forced, top-8 ids and
    log-probs per answer position.
    - Windows come from the HPD-0 walks, built as load_windows builds the map's: rulec_he12 = 12 windows of
      d02b_walk_hpd0_code.npz not in the map (5,118 ids, 2,961 answer positions; FULL.md counts 2,894 tested); rulec_mbpp
      = d02b_walk_hpd0_mbpp.npz's 100 windows (29,640 ids, 15,441 positions).
    - The walk's copies of the map's 12 code windows have the same ids as the map's set (12/12).
    - Outputs: data/rulec/rulec_he12.json, rulec_mbpp.json. Scripts: scripts/dm_s7/rulec_pass.py (check / run),
      rulec_slot.sh (after the s67 runner ends, same guards, last start 22:50, hard stop 23:00).
- ADDENDUM Wed 22:0x, before its run: NewIdeas' CF 0.3 CHAT ARM (NewIdeas 22:02, pre-registered QUEUE 22:0x 04ac744; the
  orchestrator 22:0x order after P-OFF: 1 chat arm (cf 0.5), 2 s7_trl, 3 the Rule C maps, 4 this arm only if it can start by
  22:50; if time runs short, s7_trl beats everything and the Rule C maps beat this arm; misses go to Thursday's first map slot).
  - Arm qom_0.3_live_c147_bonus3g: the chat route at cf 0.3 (the bonus row gated at 0.3), 147-slot pool, the chat arm's
    q1_union form (no t3, 4-bit stock path), 36 windows. Twins: the chat arm (primary), qom_0_live_bq_c147_t3.
  - Why (NewIdeas): a chat route at 49-52 MB per kept token against the cf 0.5 route's ~88 MB (x0.56-0.59).
  - Guess [G, NewIdeas]: IFEval 4.5-6.9 %, multi 2.9-4.8 %, against the lines 5.42 / 3.69 (a coin flip).
  - Install (22:04, install_cf03.sh + test_cf03_arm.py, 7/7 ok):
    - the PERT is exactly {rmask: NewIdeas' masks dir, renorm: False};
    - 36 windows, with names and shapes equal to the chat arm's and the routing's;
    - against the cf 0 base qom_0_live_c147 (= qom_0.3_live_c147, NewIdeas check A): drops only, ranks 0-1 identical,
      and exactly 528 rows differ (NewIdeas check B: the gated bonus rows).
    - My first test compared against the chat arm (cf 0.5) and failed by design: the pools evolve differently there, so
      it is not drops-only. Logged as install_cf03_2203.log; nothing was installed on that run.
  - Run: scripts/dm_s7/chain_cf03.sh waits for both runners to end, then (before 22:50) adds the arm to
    data/s67_extra.txt and runs a second s67 runner instance (same guards, hard stop 23:00).
    - A first inline chain was stopped at once: its own command line held the runner's name, so the pgrep gates of the
      Rule C runner and the dm_t3 install would have waited on it.
- RESULTS, final (Wed 22:21) [K]. All arms 36 windows, 0 row-count warnings, 0 failures; data in data/dm_s7/results_wed/
  (final_readout_2221.txt, out JSONs gzipped).
  - s7_trl (Fit12's 2.40 b trellis store at S = {7, 18, 30, 41}; every expert read from TRL, fallback 0):
    flips / KL 1.04 / 0.0046 | 6.89 / 0.0261 | 5.69 / 0.0278 (code | IFEval | multi).
  - SCREEN 7, my cross-check of amendment 2 (unrounded KL, the pre-registered fit; NewIdeas' s7_read.py, with its SE gate,
    is the verdict):

    | domain | scale | est_abs flips | in-frame flips | line | verdict |
    |---|---|---|---|---|---|
    | code | 11.12 | 1.54 % | 1.45 % | 1.34 | BETWEEN |
    | IFEval | 12.82 | 10.07 % | 9.54 % | 5.42 | KILL |
    | multi | 17.05 | 10.98 % | 10.44 % | 3.69 | KILL |

    NewIdeas' prediction was ~1.0 % PASS / ~7.3 % BETWEEN / ~6.0 % KILL; the TRL store extrapolated to 48 layers costs
    more.
  - CHAT ROUTE qom_0.5_live_c147_bonus5g (the 24 GB split, 147 slots): PASS on every domain.
    - Flips: code 0.71 % (24/3366), IFEval 4.30 % (146/3397), multi 2.73 % (97/3553), against the lines 1.34 / 5.42 /
      3.69. NewIdeas' prediction was ~0.7 / ~4.6 / ~3.0.
    - KL 0.0025 / 0.0115 / 0.0075.
    - Paired McNemar vs 1(c) (qom_0.5_live_bonus5g, the larger pool): 16 / 8, z +1.63; 79 / 31, z +4.58; 58 / 29,
      z +3.11. The smaller pool costs, but stays under every line.
    - Vs qom_0.5_live_c164 (information): 11 / 7, z +0.94; 73 / 45, z +2.58; 37 / 34, z +0.36.
  - Screen 6, 1(b) and 1(c): the same as the interim read-out (screen 6 PASS on code; 1(b) PASS; 1(c) COST).
  - After the batch: dm_t3 got the padded layout name (22:21:23) and r3_overlay was moved aside.
  - Then the Rule C passes started (22:21:30), and the cf 0.3 arm is chained after them (start by 22:50).
- RULE C PASSES done (Wed 22:21-22:38) [K]: data/rulec/rulec_he12.json (12 windows, 2,961 positions, 97 s) and
  rulec_mbpp.json (100 windows, 15,441 positions, 876 s); MLX peak 14.2 GB; every row 8 wide. Sent to MoonShots for the
  CPU replay of Fable FULL's rule C (FULL.md Round 2's pre-registered lines are theirs to read).
- RESULT, the cf 0.3 chat arm (Wed 22:38-22:43) [K]: qom_0.3_live_c147_bonus3g, 36 windows, 0 warnings: code PASS, IFEval
  and multi BETWEEN (just over the lines). Data: data/dm_s7/results_wed/cf03_readout_2243.txt.
  - Flips: code 0.74 % (25/3366), IFEval 5.56 % (189/3397), multi 3.86 % (137/3553), against the lines 1.34 / 5.42 / 3.69.
    NewIdeas' guess: IFEval 4.5-6.9 %, multi 2.9-4.8 %, "a coin flip".
  - KL 0.0038 / 0.0201 / 0.0126.
  - McNemar vs the chat arm (cf 0.5, primary): code 9 / 8, z +0.24; IFEval 100 / 57, z +3.43; multi 73 / 33, z +3.89. cf 0.3
    costs significantly on chat.
  - Vs C (information): 9 / 20, z -2.04; 78 / 163, z -5.48; 59 / 135, z -5.46.
  - Tonight's map work ended at 22:43 (hard stop 23:00 unused).
  - Thursday leftovers: DENSE-3 (after the records3 conversion, BIGMOE_EXT_BUILD=build_nat). The s7_trl_down build is on
    hold (NewIdeas 22:32: a 4-layer KL arm lacks the power; the orchestrator picks an alternative).
- ADDENDUM Wed 22:5x, before its run: NewIdeas' CF 0.4 CHAT ARM for Thursday (pre-registered QUEUE 22:4x, fb39925; the
  orchestrator: low priority, after DENSE-3, only if a share slot is free).
  - Arm qom_0.3_live_c147_bonus4g: the chat arms' q1_union form (no t3, 4-bit stock path), the bonus row gated at 0.4,
    147-slot pool, 36 windows, ~5 min. Twins: qom_0.3_live_c147_bonus3g (primary), qom_0.5_live_c147_bonus5g.
  - Install (22:50, install_cf04.sh + test_cf04_arm.py, 8/8 ok):
    - the PERT is exactly {rmask, renorm False}; 36 windows, shapes equal to the base's, cf 0.3's and the routing's;
    - against the cf 0 base qom_0_live_c147: drops only, ranks 0-1 identical, exactly 313 rows differ (NewIdeas: 313);
    - every changed row is also changed at cf 0.3 (313 within 528).
- NOTE Wed 22:5x [K], for NewIdeas' error-vs-KL question: s7_trl and s7_cal made exactly the same reads.
  - 144 calls and 368,983 (row, rank) pairs over the 36 windows, equal window by window (e.g. HumanEval_10: 4 calls,
    9,636 pairs in both).
  - By construction: the same masks; at S every kept expert of the gated rows (ranks 0-1 forced in + the mask's 2-9) from
    the source; prompt rows and ungated rows on the stock 4-bit path; other layers in s7_ref's form.
  - Only the unique experts at those slots drift slightly (1,165 vs 1,156 on HumanEval_10), because routing below layer 7
    sees different hidden states.
  - Under DRAFT17 (the ball masks), every realized check row, the last row included, is gated, so no unsure full-top-10
    rows take the source.
  - The 2.6-3.2x KL vs 1.1-1.8x error gap is not from different reads.

## CLOUD FABLE 3's MAP ARMS C3-R11..R17 (the orchestrator 06:5x; specs ~/FableGod/cloud3/live/cloud3/REQUESTS.md): MASK BIT-CHECKS
- Thu 06:5x [K] (scripts/dm_s7/check_c3_masks.py; data/c3/check_c3_*.log; CPU, read-only): an INDEPENDENT REBUILD of
  every band arm from its spec text plus the conventions in cloud3's builder docstrings.
  - Conventions: the MoonShots export's live-cut checks and dcf; stand-ins p + 97; only realized rows masked; the bonus row
    by its next draft; strict union members; reach = the product of the earlier drafts' cf.
  - c3_b2d, c3_b2c, c3_b2d_reach07, c3_b2d_reach07_u7, c3_b2b, c3_b2b_top1, c3_new1: bit-equal on 36/36 windows each (3,570
    checks, 14,093 rows).
  - c3_b2b_band5: bit-equal 36/36 once the layer 8-35 cap of 5 applies to DRAFTED rows with cf < 0.9 only.
    - With no-draft rows capped too, 14/36 windows differ: cloud3's build keeps a no-draft row exact. The R12 spec text is
      silent; flagged.
  - Format, every arm: keep [N, 48, 10] bool; N == the map's routing rows; the export's ids == the map's routing; rank 0
    kept on every row; rows outside every check all kept.
  - Nesting holds on 36/36: b2d within b2c, reach07 within b2d, reach07_u7 within reach07, b2b_top1 and b2b_band5 within
    b2b.
  - The walk's dcf equals the gate file's cf on all 8,587 realized drafted rows (max |diff| 5e-7), so the bands use the
    engine gate's confidence.
  - To redo on the final masks after Fable C's ~09:00 audit, then snapshot, install and pre-register.
- Thu 06:59: the scores-only capture installed for R6 (the orchestrator: option (i)), via install_scap.sh.
  - dm_common gets CAPR "s": each MoE layer's router top-10 scores for EVERY row, without the hidden states, saved to
    <DM_OUT>/scores/<window>.npz. run_map gets the name "scapture".
  - Diff: only the new lines; the default paths are unchanged. Import and clear() tested.
  - Runs with DM_OUT set to a scratch dir, so the map's out/routing stays untouched. A 1-window GPU smoke comes before the
    full pass.
- Thu 07:0x [K, partial: answer-row scores only, 88 % of checks] R6 massB_0.5's rule tested before the all-rows capture
  (scripts/dm_s7/build_massb.py --xcap --partial; the rule from REQUESTS C3-R6 + REPORT_P2 section 4).
  - Rule: per (check, layer), U = the check rows' distinct top-10 experts, B = ceil(0.5 |U|); keep the B experts with the
    largest summed renormalised weight; every realized row keeps its top-10 within that set.
  - Drop rates, on scored checks:

    | domain | rank 0 outside the budget | rank 1 outside | kept share |
    |---|---|---|---|
    | code | 1.05 % | 4.8 % | 0.73 |
    | IFEval | 0.43 % | 2.2 % | 0.65 |
    | multi | 0.40 % | 1.9 % | 0.64 |

  - The block counted from realized rows only (no stand-ins) gives 0.95 / 0.41 / 0.37 % and 4.3 / 2.0 / 1.6 %.
  - REPORT_P2 section 4 (one window per domain) said 0.000 for rank 0 and 0.000-0.002 for rank 1. On the 36 windows the
    half-union budget does drop rank 0 sometimes and rank 1 a few percent. Flagged for Fable C's audit; it is not a blocker,
    and the arm runs as defined.
- PRE-REGISTRATION Thu 07:1x (copied from NewIdeas' amended QUEUE Thu 07:0x / 07:1x, 5f97c1c; masks 3894480; the order is
  Fable C's audit: R15 first, then R15b), before any run.
  - C3-R15 = qom_0.5_u8_live_c147_bonus5g: the measured chat route with its unsure rows at top-8 + union members + resident.
  - C3-R15b = qom_0.5_u8_reach0.5_live_c147_bonus5g: R15 + REACH-CUT 0.5 (unsure rows with reach < 0.5 go to top-2 +
    members + resident).
  - Built with NewIdeas' dm_oea_masks.py (qom_<thr>_u<k>[_reach<tau>]) + Fable C's bonus gate at 0.5 (proxy g), exactly as
    the route, so each differs from it by its cut alone. This is not the _bq form Cloud3's request named.
  - NewIdeas' bit-checks (check_r15.log): u10 rebuilds the route's base 36/36; u8 drops only ranks 8-9; the reach cut keeps
    ranks 0-1; both gates drop only, on the route's 165 rows. The u8 base moves 7,724 sure rows (smaller unions, a lighter
    LRU pool), so neither arm is drops-only against the route.
  - Lines (Fable C's amendment):
    - PASS = under 1.34 / 5.42 / 3.69 on every domain AND within +0.5 points of the route's 0.71 / 4.30 / 2.73;
    - KILL = IFEval or multi over its line;
    - otherwise BETWEEN. Cloud3's 1.3x guard is printed for information.
  - Twins: the route qom_0.5_live_c147_bonus5g (primary); R15b also against R15. NewIdeas reads them with chat_arm_read.py,
    which writes into its repo, so NewIdeas runs it.
  - Install 07:14 (install_c3.sh with C3_SRC = NewIdeas' masks; snapshots in out/ni with sha256; test 8/8):
    - the PERT is exactly {rmask, renorm False};
    - 36 windows, byte-equal to the source, with routing shapes;
    - ranks 0-1 kept on every row.
- Thu 07:20, after Fable C's audit (AUDIT_1105.md 07:18; masks final, "as built"):
  - check_c3_masks.py re-run: the same 36/36 rebuilds and nesting.
  - install_c3.sh: c3_b2d, c3_b2c, c3_b2d_reach07, c3_b2b_band5, c3_b2b_top1 are in, from snapshots in out/c3 with sha256 (test:
    PERT, loader, 36 windows byte-equal, routing shapes, rank 0 kept). c3_b2d_reach07_u7 is not installed: the audit says no.
  - R6 builder per Fable C's rulings 2-3: ranks 0-1 always kept, then the budget (--as-written for the record form); a row
    without scores takes the rank profile (a fallback only).
  - Answer-row test, all checks: the budget alone leaves out rank 0 on 0.92 / 0.40 / 0.38 % of row-layers and rank 1 on
    4.4 / 2.1 / 1.8 %. The floor adds back 0.53 / 0.25 / 0.22 % of cells. Kept share 0.735 / 0.649 / 0.644 (Fable C's guess
    for code was 0.76-0.78).
- PRE-REGISTRATION Thu 07:2x: TEST 1 of Fable C's audit (AUDIT_1105 section 2; the orchestrator 07:2x; NewIdeas QUEUE Thu
  07:3x, masks 9b6451e), before any run.
  - split_0.9 = qom_0_split0.9_live_bq, split_0.99 = qom_0_split0.99_live_bq: the map's qom_0_live_bq form at the 45 GB pool
    (QOM cf 0 + the bonus row gated by its next draft). Drafted rows with cf >= 0.9 / 0.99 keep ranks 0-1 + union members
    only, with no resident keep; every other row is as QOM; loads unchanged. Harness q1_union(<masks>), no t3, 36 windows.
  - Twin: qom_0_live_bq (measured 0.39 / 4.09 / 3.26 %); each split also against the other.
  - Beside them on Friday: c3_b2d and c3_b2c (Cloud3's masks, as built).
  - Lines, code decisive, all-row flips: PASS <= 1.34 %; BETWEEN 1.34-2.0 %; KILL > 2.0 %. The lever dies if both splits are
    > 2.0 % on code (or if BigMoE's C3-R8 shows QOM's group count <= 0.6x exact's). Chat runs as the control.
  - Priors [G, Fable C]: split_0.9 1.2-1.5 %, split_0.99 0.9-1.2 %; c3_b2d 1.3-1.5, c3_b2c 1.2-1.4 on code; IFEval 6-8 %, multi
    4.5-6 % on chat. NewIdeas reads them with chat_arm_read.py, which writes into its repo, so NewIdeas runs it.
  - Then, per the audit: c3_b2d_reach07 on code only, and only if a code arm passes (PASS <= 1.34, KILL > 2.0); R15, then R15b
    only with >= 0.5 of margin on multi; band5 and top1 on code only; topk8; then the scores capture + massB last.
  - u7: not run.
  - Install 07:25 (install_c3.sh, snapshots in out/ni, 8/8: PERT, loader, 36 windows byte-equal, routing shapes, ranks 0-1
    kept on every row).
  - My count vs qom_0_live_bq: split_0.99 drops 910,285 experts on 5,691 rows (2,136 added by LRU drift); split_0.9 drops
    1,382,387 on 7,339 rows (2,239 added). Equal to NewIdeas' check_split.log.
  - Runner, Thu 07:2x: code-only arms ("<arm>+code", DM_SETS=code, 12 windows, read by code_read.py) and WHEN conditions
    (cond_eval.py: codepass / multile). Dry tests right.
    - A bash 3.2 parse error went into b6a2e29 (its commit ran after a failed syntax check) and was fixed in 6457a15, before
      any run.
- Thu 07:32, test 1's PRIMARY arms are the DRAFT17 _ball splits (the orchestrator 07:3x; NewIdeas QUEUE Thu 07:3x, masks
  c22da10; the lines unchanged), installed from snapshots (out/ni, 8/8 per arm):
  - qom_0_split0.9_live_ball and qom_0_split0.99_live_ball, and their twin qom_0_live_ball (QOM-KEEP + DRAFT17 at the 45 GB
    pool, never run on the map).
  - My count vs the twin: split_0.99 drops 953,764 experts on 5,849 rows (2,377 added by LRU drift); split_0.9 drops
    1,429,194 on 7,455 rows (2,376 added). The twin differs from qom_0_live_bq on 3,575 rows. All equal to NewIdeas'
    check_split_ball.log.
  - Code tiles per kept token at 45 GB (NewIdeas): _ball 291.4 -> 182.9 / 147.9 (x0.628 / 0.508); bq x0.654 / 0.537; the
    audit's replay x0.601 / 0.469.
  - Today's queue (data/thu_extra.txt), in order: split_0.9_ball+code, split_0.99_ball+code, qom_0_live_ball+code, DENSE-3,
    the bq pair +code, cf04.
- Thu 09:09, FRIDAY DRIVER ARMED (scripts/dm_s7/fri_c3.sh, data/fri_extra.txt; the orchestrator 07:2x / 09:0x).
  - It starts at Fri 00:30, behind the quiet queue (P-OFF, BigMoE), on a uniform converted records3, with build_nat.
  - Order: test 1 on 36 windows (the _ball splits, their twin, c3_b2d, c3_b2c); WHEN reach07+code (codepass over test 1);
    R15; WHEN R15b (multi <= 3.19); band5+code; top1+code; any Thursday leftovers; topk8; then the all-rows capture and
    massB_0.5 (built + installed in the driver).
  - Hard stop 10:30, the tail cut first by start-by times.
  - On every exit after 00:30 (the end, an abort, a stop) it stops its runs, unlocks, then touches
    ~/EvalSuite/results/q45/GO (EvalSuite runner 45). Before 00:30 a stop writes no GO.
  - Tested on scratch dirs: the stop path, the normal end, the pre-slot stop, and a missing GO dir (logged, never created).
  - Dry test of the pick order: right.
- Thu 09:2x, Thursday's run MOVED (the orchestrator 09:1x: EvalSuite's runners 43 + 44 fill ~10:30-15:50).
  - Driver scripts/dm_s7/thu_splits.sh, slot from 10:10. It waits for records3 to be CONVERTED (the in-place rewrite done; a
    row-major index is not enough) and no conversion process.
  - Then the capture smoke, then split_0.9_ball+code, split_0.99_ball+code and qom_0_live_ball+code (~20 min).
  - Last start 11:15, hard stop 11:45. On every exit from 10:10 it touches ~/EvalSuite/results/q44/GO (runner 44). Tested:
    no GO on a pre-slot stop; GO on a stop in the slot.
  - The bq pair, DENSE-3 and cf04 move to Friday's "Thursday leftovers". The 13:10 driver is stopped.
- RESULT, TEST 1 on the code windows (Thu 10:29-10:35) [K]: both DRAFT17 _ball splits PASS on code. 12 code windows, 3,366
  rows, all-row flips vs the baseline argmax; on the CONVERTED records3 with build_nat; data data/dm_s7/results_thu/.
  - The capture smoke first (the first map run on the converted store): routing ids == the map's stored routing on 100 % of
    (row, layer) cells, baseline top-1 == the map's baseline on 144/144 rows, scores |diff| 0.00 vs the 11-03 xcapture.
    PASS: the converted store reads bit-for-bit as before.

    | arm | code flips | vs the twin (only arm / only twin) | verdict |
    |---|---|---|---|
    | twin qom_0_live_ball (QOM-KEEP + DRAFT17, 45 GB) | 0.51 % (17/3366) | vs qom_0_live_bq (0.39 %): 11 / 7, z +0.94 | PASS |
    | split_0.99 (qom_0_split0.99_live_ball) | 0.92 % (31/3366) | 23 / 9, z +2.47 | PASS |
    | split_0.9 (qom_0_split0.9_live_ball) | 1.31 % (44/3366) | 37 / 10, z +3.94 | PASS, just under 1.34 |

  - split_0.9 vs split_0.99: 21 / 8, z +2.41.
  - Fable C's priors: split_0.9 1.2-1.5 %, split_0.99 0.9-1.2 %; both landed inside.
  - Each split costs significantly against QOM-KEEP (+0.41 / +0.80 points) and stays under the line. Tiles (NewIdeas):
    x0.628 / x0.508 of QOM's on code at 45 GB.
  - The audit's full PASS also needs BigMoE's C3-R8 (QOM's group count >= 0.85x exact's): the engine side.
  - The chat control (36 windows) runs Friday.
  - GO touched in ~/EvalSuite/results/q44 at 10:35:37; the lock was released first.
- Thu 10:3x (BigMoE's 10:28 lounge note after the conversion): the moved-aside r3_overlay.STALE-before-records3-conversion
  had its index.json renamed to index.json.STALE-row-major-entries-do-not-use, so a reader pointed at it fails loudly (no
  index) instead of decoding converted bytes as row-major. Nothing deleted.
- Thu 18:58 (the orchestrator 18:5x): PRE-REGISTRATION, Fable C's WALL arms for Friday (~/FableGod/fablec/build_1105,
  INSTALL.md sections 4-6). They run on the 12 code windows (3,366 rows, 3,316 non-tie), queued right after test 1
  (data/fri_extra.txt section 2, in place of reach07+code). All are on split 0.99 QOM-KEEP + DRAFT17 at 45 GB:

  | name | arm |
  |---|---|
  | K8 | `qom_0_split0.99_k8_live_ball` |
  | K6 | `qom_0_split0.99_k6_live_ball` |
  | A, split 0.99 + reach 0.7 | `qom_0_split0.99_r0.7_live_ball` |
  | A', middle-layer split 0.9 | `qom_0_split0.99_m0.9_live_ball` |
  | B, A + the z0.999 middle-layer skip | `qom_0_split0.99_r0.7_z0.999_live_ball` |
  | K4, a probe, last | `qom_0_split0.99_k4_live_ball` |

  - **Lines** (code_read.py VERDICT2): PASS = all-row flips <= 1.34 % AND non-tie flips <= 0.87 %; KILL over either.
    The code preset is the passing arm with the fewest tiles.
  - **Fable C's predictions** [G]: K8 / K6 non-tie ~0.7 (all-row ~1.15-1.2); A 0.6-0.9; A' 0.6-0.7; B = A + ~0.1; K4
    0.9-1.1.
  - **Anchors** (all-row / non-tie):

    | arm | flips | tiles |
    |---|---|---|
    | twin | 0.51 / 0.21 % | |
    | split 0.99 | 0.92 / 0.51 % | x0.628 |
    | split 0.9 | 1.31 / 0.78 % | x0.508 |

  - **Install** [K]: scripts/dm_s7/install_fc.sh + test_fc_arms.py, Fable C's copies of install_c3.sh / test_c3_arms.py.
    Reviewed: only the default dirs and B's skip check differ.
    - ALL OK for the six arms.
    - The out/fc snapshots match Fable C's manifest: 216/216 sha256.
    - B holds 2,215 skipped rows: nothing kept in all of layers 8-35, ranks 0-1 kept elsewhere.
  - **Read-out** [K]: the non-tie patches are applied to code_read.py and to the map's dm_screen_read.py. Each equals Fable
    C's patched copy byte for byte. On split 0.99, code_read reproduces Thursday's read:
    - all-row 0.92 % (31/3366), non-tie 0.51 % (17/3316);
    - McNemar 23 / 9, z +2.47 (non-tie 14 / 4, z +2.36);
    - VERDICT2 PASS.
  - **massB stand-ins**: build_massb.py's PROFILE is now the engine's mean renormalised weight by rank (0.227 ... 0.057). It
    was cloud3's (0.294 ... 0.010).
- Thu 18:5x: USES PREDICTION for BigMoE's Friday split A/B (keepset/uses_pred.py; CPU replay, code windows). Distinct
  experts computed per check, vs qall:

  | pool | s99 | s90 |
  |---|---|---|
  | 45 GB C_m | x0.628 | x0.508 |
  | C 160 | x0.666 | x0.558 |
  | C 160, full blocks | x0.724 | x0.630 |

  - Drafted code rows with cf >= 0.99: 0.827 of them in the first 32 generated tokens, 0.654 after.
  - BigMoE 19:2x: its pre-flight "uses" is need.size(), counted before the split's drop (~1.46x my count), so it cannot show
    the split. Friday's logs count groups computed after the drop and compare them with x0.72 / x0.63.
- EvalSuite T0 (17:52), on the 12 GB line: **KILL**.
  - IFEval 84.0 vs 90.0 (-6.0) and HumanEval+ 80.0 vs 91.7 (-11.7) fail.
  - Math 84 vs 86 PASS; lifetime peak <= 10.84 GB PASS.
  - The v3p freeze is lifted (T0_DONE).
- Thu 19:4x: PRE-REGISTRATION, Saturday night. The night starts Sat 7 Nov 00:30, after BigMoE's FAR-G1 and EvalSuite's
  runner 46 (the orchestrator 19:4x). Driver: scripts/dm_s7/sat_night.sh. Hard stop 09:45; no GO handoff.
  1. **Fable A's T1**: xs64 / xs48 / xs40 / xs32 on code, then xs48 on chat if code leaves room. It runs from
     sat_t1.sh, a copy of its run_t1.sh where only the times change. Its lines and decide.py are Fable A's
     (~/FableGod/lossy350a). Checked: the patch is env-gated (DM_XS=1), and it writes only under its run folder.
  2. **FAR's NEVER-WAIT tier 1 + 1b** on code, after its GPU self-check: c208 twin, nw0, nw1, nw1r1, nw1 at the published
     pool, split 0.99 + nw1, and the c266 pair. Lines are FAR's INSTALL.md section 4. Its harness patch is installed by
     hand on Friday after the Friday batch, after review. If it is not installed, or the self-check FAILs, these arms are
     dropped.
  3. **Step 0 of PLAN_12GB_90.md**: the 12 GB two-pass certificate on code.
     - On record [K], NewIdeas' Tue runs, read with VERDICT2:

       | arm | code flips | non-tie | verdict |
       |---|---|---|---|
       | `p1c_oea_k1_p08` (P1-K1 pass 1, θ 0.8, no KU cap, no cf gate) | 1.52 % | 1.00 % | BETWEEN, VERDICT2 FAIL |
       | `p1c_top_k2_p08` | 1.96 % | | |
       | `p1_oea_k1` (pass 1 on every row) | 4.04 % | | |

     - The masks come from scripts/dm_s7/build_p1c.py. It is NewIdeas' p1_mixed_masks rule plus T3R2's cf gate.
       It reproduces `p1c_oea_k1_p08` 36/36 windows bit-equal; code: 2,997 rows certified, 2.44 experts per row-layer.
     - The cf gate sends 332 code rows straight to exact.
     - Installed with install_c3.sh, ALL OK.

     | arm | what | code rows certified | prediction [G] |
     |---|---|---|---|
     | `p1c_oea_k1_ku4_p08_cf09` | T3R2's own form | 84.2 % | 1.4-1.6 % |
     | `p1c_oea_k1_ku4_p09_cf09` | | 81.9 % | 1.25-1.45 % |
     | `p1c_oea_k1_ku4_p095_cf09` | | 79.6 % | 1.1-1.3 % |
     | `p1c_oea_k2_p08_cf09` | 3.97 experts per row-layer | 85.6 % | 1.2-1.5 % |
     | `p1c_oea_k2_p09_cf09` | | 84.1 % | 1.0-1.3 % |

     - Line (code_read.py VERDICT2): PASS = all-row <= 1.34 % AND non-tie <= 0.87 %.
     - Decides: the fastest passing form is the 12 GB base for the plan's factors. If no P1 form passes, the base is a
       top-5 pass 1 (cert_top5_p09, 0.92 %), and Fit12 prices its pass-1 cost.
- Thu 19:5x: PRE-REGISTRATION, Saturday step 0b. Fit12's replay found the certificate token-exact, so the map's damage is
  pass 1's KV. Step 0b finds the cheapest pass-1 width that passes code.
  - Sources topk4 and topk3 run on all 36 windows (registered by install_topk34.sh, CPU test ALL OK).
  - Then cert_top4_p09, cert_top4_p08, cert_top5_p08 and cert_top3_p09 on code.
  - Line: VERDICT2 (all-row <= 1.34 % AND non-tie <= 0.87 %).
  - Predictions [G]: top-4 θ 0.9 0.95-1.2 %; top-4 θ 0.8 1.1-1.4 %; top-5 θ 0.8 1.0-1.25 %; top-3 θ 0.9 1.1-1.4 %.
  - On record: top-5 θ 0.9 0.92 %, top-2 θ 0.9 1.49 %.
  - Decides: which width Fit12 builds and prices. Speed for each width comes from Fit12's replay after one calibration
    run.
- Thu 20:0x: AMENDMENT before any run, Saturday step 0 / 0b PRUNED.
  - Fit12 (calibrated, full outputs, cf 0.9): top-3 is slower (34.5 / 33.1 tok/s at θ 0.9 / 0.95). Top-4 / top-5 cannot
    run at 16 rows: the union overflows the 47-slot pool and the engine aborts.
  - Every two-pass point except P1-K1 at θ 0.8 (45.8) is at or under the exact mode (40.8).
  - Only `p1c_oea_k1_ku4_p08_cf09+code` runs, with its line unchanged (VERDICT2). It decides whether the x1.12 FAST point
    is code-valid.
  - Dropped, not run: θ 0.9 / 0.95, both P1-K2 arms, topk3 / topk4 and the four width arms. The orchestrator asked for
    the same at 20:0x.
- RESULT, THE FRIDAY MAP BATCH (Thu 23:45 to Fri 03:54) [K]. It started 45 min early on Kamen's word (23:1x, via the
  orchestrator). It ran 14 runs on records3 with build_nat. Data: data/dm_s7/results_fri/ (the batch log without the
  per-window lines, the SCREEN_read files, fri_c3.log).
  - **Ran**:
    - test 1's three arms on all 36 windows, c3_b2d and c3_b2c;
    - Fable C's six code arms and R15;
    - then the all-rows capture (PASS) and massB_0.5.
  - **GO** was touched for EvalSuite's runner 45 at 03:54:25.
  - **Not run: 8 arms.** The runner stopped at its 14-run cap at 03:45. Two of the 14 runs were stopped by others' quiet
    windows and retried. The cap is now 40 (S67_MAXRUNS). The 8 arms move to Saturday after step 0 (sat_extra.txt section 3)
    with the same lines and twins.
  - **Full arms** (36 windows). Cells are all-row flips, with non-tie flips in brackets. Lines: code 1.34 [0.87], IFEval
    5.42 [2.22], multi 3.69 [1.63].

    | arm | code | IFEval | multi | VERDICT2 |
    |---|---|---|---|---|
    | twin `qom_0_live_ball` (QOM-KEEP + DRAFT17, 45 GB) | 0.51 [0.21] | 4.03 [1.00] | 3.26 [0.91] | PASS / PASS / PASS |
    | split 0.99 | 0.92 [0.51] | 6.01 [2.77] | 5.15 [2.27] | PASS / FAIL / FAIL |
    | split 0.9 | 1.31 [0.78] | 8.10 [4.47] | 6.73 [3.90] | PASS / FAIL / FAIL |
    | `c3_b2d` | 0.89 [0.54] | 6.59 [2.77] | 4.31 [1.91] | PASS / FAIL / FAIL |
    | `c3_b2c` | 0.95 [0.45] | 5.30 [1.87] | 3.74 [1.36] | PASS / PASS / FAIL (all-row 3.74 vs 3.69) |
    | R15 `qom_0.5_u8_live_c147_bonus5g` | 0.65 [0.33] | 4.39 [1.22] | 2.76 [0.61] | PASS / PASS / PASS |
    | `massB_0.5` | 1.31 [0.87] | 9.18 [5.28] | 8.22 [5.24] | FAIL / FAIL / FAIL |

  - **The split is code-only.** On chat each split costs significantly against the twin:
    - split 0.99: IFEval 110 / 43, z +5.42; multi 100 / 33, z +5.81;
    - split 0.9: z +9.30 / +8.43.
    - NewIdeas' read (cbd6bae) agrees.
  - **QOM-KEEP + DRAFT17 at the 45 GB pool** (the twin, in its first full run) passes all three domains.
  - **R15 matches the route it trims.** Its unsure rows run at top-8 + union + resident. Against the route
    (`qom_0.5_live_c147_bonus5g`) the McNemar is code 10 / 12, IFEval 52 / 49, multi 45 / 44 (all |z| < 0.5). R15b's
    condition holds (multi 2.76 <= 3.19), so R15b goes on Saturday's list.
  - **massB_0.5 (R6, Fable C's rulings) is KILLED.** Against `c3_b2d` it is IFEval 171 / 83 (z +5.52) and multi 210 / 71
    (z +8.29). Code's non-tie sits at the line (0.87 %).
  - **Fable C's six code arms** (12 code windows, 3,366 rows; all-row [non-tie] flips). The predictions are Fable C's, on
    non-tie flips [G].

    | arm | flips | prediction | VERDICT2 |
    |---|---|---|---|
    | K8 | 1.19 [0.69] | ~0.7 (all-row ~1.15-1.2) | **PASS** |
    | K6 | 1.49 [0.97] | ~0.7 | FAIL |
    | A, reach 0.7 | 1.40 [0.87] | 0.6-0.9 | FAIL (all-row) |
    | A', middle-layer split 0.9 | 1.40 [0.81] | 0.6-0.7 | FAIL (all-row) |
    | B, A + the z0.999 skip | 1.87 [1.21] | A + ~0.1 | FAIL |
    | K4, the probe | 2.02 [1.42] | 0.9-1.1 | KILL |

    - Against split 0.99, K8 is 17 / 8 (z +1.80; non-tie 10 / 4, z +1.60): not significantly worse.
    - K8 landed on its prediction. Every other arm came in 0.1-0.6 points over Fable C's prediction.
    - So K8 is the only code preset candidate from this set. It counts only if its tiles beat split 0.99's x0.628 (Fable C's
      count, engine side).
- Fri 04:01: FAR's NEVER-WAIT harness patch INSTALLED (install_nw.sh exit 0, after review; HANDOVER Fri 04:01). Saturday's
  tier 1 + 1b lines are unchanged; they run after the GPU self-check passes (PASS or REPORT).
- Fri 04:0x: PRE-REGISTRATION, LEVER 1 CPU REPLAY (PLAN_12GB_90 lever 1; the orchestrator Thu 20:0x: "kept per check").
  - **Data**: Fit12's calibration run c51 (`~/Fit12/data/scr/t3cal`, read-only), T3R2's 16 HumanEval prompts.
    - DFlash2's 15 drafts are logged per check. The outputs equal the exact tokens. Each output is capped at 256 tokens.
    - Prompt ids come from `~/EvalSuite/data/items/dec_humaneval.jsonl`.
    - Totals: 343 checks, 3,456 tokens; today's kept per check is 10.08.
  - **Method** (scripts/lever1_copy_replay.py, CPU, seconds):
    - Kept = 1 + the drafts' matched prefix. This must reproduce the logged kept on every check.
    - The copy draft is prompt lookup: the latest earlier occurrence of the longest suffix n-gram (n 8..1) in prompt +
      output so far, then the 15 tokens after it.
    - Off the logged starts, the drafter is taken to run to its next logged divergence.
    - The walks:
      - (A) today's: it must reproduce 343 checks;
      - (B) the tree: max(drafter, copy) per check, 31 rows, an upper bound;
      - (C) a realizable switch: copy when its n-gram match is >= N tokens (N 2 / 3 / 4 / 6), else the drafter;
      - (E) as B with copy checks 32 rows wide (31 copy drafts).
  - **Prediction** [G]: B +10-25 % kept per check over 10.08; the best C +3-10 %. The drafter already copies: the
    signature and docstring checks keep 16.
  - **Lines** (kept per check, 16 rows):
    - PASS = the best C >= +10 % (then a timing probe in Fit12's harness);
    - KILL = B < +10 % (copy adds nothing over DFlash2 on code at 12 GB).
- RESULT, LEVER 1 CPU REPLAY (Fri 04:0x) [P, CPU; data data/lever1/replay_g1k2.json + .log, scripts/lever1_copy_replay.py].
  - The replay reproduces all 343 logged checks exactly (0 mismatches). Today's kept per check is 10.08.

  | walk | checks | kept per check | vs today | rows per kept token |
  |---|---|---|---|---|
  | A, today (DFlash2, 16 rows) | 343 | 10.08 | x1.000 | 1.59 |
  | B, tree max(drafter, copy), 31 rows | 328 | 10.54 | x1.046 | 2.94 |
  | C3 / C4 / C6, copy if n-gram >= 3 / 4 / 6 (16 rows) | 359 / 350 / 343 | 9.63 / 9.87 / 10.08 | x0.955 / 0.980 / 1.000 | 1.66 / 1.62 / 1.59 |
  | E, tree with a 31-draft copy branch, 47 rows | 274 | 12.61 | x1.252 | 3.73 |
  | CW6 (post-hoc), a 32-row copy check if n-gram >= 6, else the drafter | 284 | 12.17 | x1.208 | 1.65 |
  | **CW8 (post-hoc)**, the same at n-gram >= 8 | **283** | **12.21** | **x1.212** | **1.64** |

  - **Copy at the check's own width (16 rows) is KILLED.** The oracle B is +4.6 %, under the +10 % line, and every switch
    is at or below today. DFlash2 already drafts the copied text: the copy runs that fit in 15 drafts, it gets right.
  - **The gain is in the width cap.** Long copy runs (re-emitted signatures and docstrings) go past the drafter's 15
    drafts. A 32-row copy check on long n-gram matches (CW8, ~25 % of checks) cuts checks by 17 % for +3 % rows per kept
    token.
    - This is a post-hoc LEAD, not a PASS: the pre-registered C line failed.
    - It counts only if a 32-row check costs well under 1.21x a 16-row one in the 12 GB exact mode. 1.1x would give ~x1.1
      [P, halved].
    - The next test is Fit12's: time 16- vs 32-row checks in t80ck2, on copy rows.
  - **Caveat**: outputs are capped at 256 tokens, which favours the copy-rich start of each answer. The drafter's run
    off the logged starts is approximated (to its next logged divergence).
- Fri 13:0x: PRE-REGISTRATION, NEVER-WAIT MOVED UP TO TODAY (NewIdeas' GPU ORDER item 6, b6e90b5, 12:5x: the expert-weight
  misses, -13.5..-15 ms per check, are the second-biggest gap to 300; a map verdict today lets BigMoE's engine A/B land on
  Saturday). Driver scripts/dm_s7/fri_nw.sh, list data/fri_nw_extra.txt. A share lock per run (MEM 18 GB), alone; hard stop
  14:10 (EvalSuite's long-code smoke follows).
  1. The GPU self-check first (FAR's INSTALL.md step 2), last start 13:30. PASS or REPORT runs the arms. Otherwise none run,
     and tonight's sat_night.sh keeps them as armed.
  2. Then Saturday's tier 1 + 1b code arms (Thu 19:4x item 2): the same 8 lines, twins and lines (FAR's INSTALL.md
     section 4). The decisive arm is nw1 at 208 slots against the c208 twin: PASS <= 1.34 % and D <= +0.40 points ("clean"
     D <= +0.15); BETWEEN (1.34, 2.0] % or D in (+0.40, +0.80]; KILL > 2.0 % or D > +0.80. FAR's predictions stand: twin at
     208 slots 0.6-0.9 %; nw1 D +0.1 to +0.4, PASS 60 %.
  3. Added at NewIdeas' request ("all three domains"; FAR's tier-2 chat form, 36 windows):
     - `qom_0_live_ball_nw1`, against its twin `qom_0_live_ball` on record (0.51 / 4.03 / 3.26);
     - then `qom_0_live_ball_c208` and `qom_0_live_ball_c208_nw1`.
     Lines: my VERDICT2 per domain (code 1.34 [0.87], IFEval 5.42 [2.22], multi 3.69 [1.63]). D against the own twin, with
     McNemar counts, is reported per domain. FAR registered no chat D line, so none is set now; a chat D over +0.40 points
     with |z| >= 2 is flagged.
  - Arms not done by 14:10 stay in tonight's list (the code ones). The 36-window arms are not added to tonight's list.
- Fri 13:2x: CORRECTION (Fit12, `~/Fit12/PLAN_12GB_90_fit12.md` "Lever 3 RESULT"): t80ck2 is FAST (top-2 pass 1, θ 0.8,
  cf 0.9; a map code KILL at 1.63 %), not the 12 GB exact mode. The exact loop runs 18.4-18.9 tok/s. PLAN_12GB_90 "Update
  Fri 13:2x" has the consequences; two land here.
  - PRE-REGISTRATION: two step 0 arms are back in Saturday's list (sat_extra.txt, after T3R2's own form, start by 09:10):
    P1-K1 KU 4 cf 0.9 at θ 0.9 and 0.95 (`p1c_oea_k1_ku4_p09_cf09+code`, `p1c_oea_k1_ku4_p095_cf09+code`; 41.0 / 37.9
    tok/s [P]). Thu 20:0x pruned them as "no faster than exact", which was the mislabel. Lines: VERDICT2 code 1.34
    [0.87]. Predictions [G]: θ 0.9 1.3-1.5 %, θ 0.95 1.1-1.3 % (top-2 ran 1.49 / 1.22 there; at θ 0.8, P1-K1 ran 1.52 against
    top-2's 1.63).
  - Lever 1's CW8 lead is KILLED at 12 GB [P, Fit12's arithmetic]. A 32-row pass 1 overflows the 47-slot pool, so it
    takes two 16-row pass-1 calls: ~35 ms per kept token against 25.9 (x0.74). Not timed; nothing routed.
- RESULT, NEVER-WAIT ON THE MAP (Fri 13:19-13:46; pre-registered Thu 19:4x item 2 and Fri 13:0x; fri_nw.sh, alone under
  share locks; data data/dm_s7/results_nw/: s67_nw_fri.log, fri_nw.log with nw_read's output, SCREEN_read_*.txt, the
  self-check's files).
  - **GPU self-check PASS** (2 code windows, 292 rows). (i) nwz equals the twin exactly. (ii) nw0: 0 flips, mean
    |d log p1| 0.0016. (iii) Rows and late flags are as expected. The twin is deterministic: its second run equals its
    first. On the 12 windows, nw_read's (ii) is REPORT (nw0 20 flips vs the twin, mean 0.0056).
  - **Code**, 12 windows, 3,366 rows. Flips are all-row [non-tie]. D = arm - twin in points; McNemar is only-arm /
    only-twin.

    | arm | flips | against its twin | verdict |
    |---|---|---|---|
    | `qom_0_live_ball_c208` (twin, 208 slots) | 0.74 [0.33] | vs the published pool's 0.51: +0.23, 14 / 6, z +1.79 | PASS (FAR: 0.6-0.9) |
    | `_c208_nw0` (control, lateness 0) | 0.62 [0.21] | -0.12, 8 / 12, z -0.89 | PASS |
    | **`_c208_nw1`** (every demand miss 1 layer late) | **0.68 [0.27]** | **-0.06**, 7 / 9, z -0.50 | **PASS, clean** |
    | `_c208_nw1r1` (rank 0 waits) | 0.74 [0.36] | 0.00, 10 / 10 | PASS (not needed: nw1 passes) |
    | `qom_0_live_ball_nw1` (published pool) | 0.59 [0.18] | +0.09, 9 / 6, z +0.77 | clean |
    | `qom_0_live_ball_c266` (twin, 266 slots) | 0.56 [0.12] | | PASS |
    | `_c266_nw1` | 0.65 [0.24] | +0.09, 8 / 5, z +0.83 | PASS, clean |
    | `qom_0_split0.99_live_ball_nw1` (the stack) | 1.07 [0.60] | +0.15 vs split 0.99 (12 / 7, z +1.15) | PASS |

    - Lateness itself (nw1 vs nw0) costs +0.06 points (8 / 6, z +0.53).
    - FAR's predictions: twin at 208 slots 0.6-0.9 (0.74, in); nw1 D +0.1 to +0.4 (-0.06, better); the published
      pool D +0.03 to +0.15 (+0.09, in); nw1r1 D +0.05 to +0.2 (0.00, better); the stack 0.95-1.15 (1.07, in).
    - The stack is +0.56 against `qom_0_live_ball` itself (27 / 8, z +3.21), mostly the split's own cost on code.
  - **Chat**, 36 windows (VERDICT2 lines: IFEval 5.42 [2.22], multi 3.69 [1.63]):

    | arm | code | IFEval | multi | VERDICT2 |
    |---|---|---|---|---|
    | `qom_0_live_ball` (twin, on record) | 0.51 [0.21] | 4.03 [1.00] | 3.26 [0.91] | PASS all |
    | `qom_0_live_ball_nw1` | 0.59 [0.18] | 3.97 [1.38] | 3.24 [0.85] | **PASS all** |
    | `qom_0_live_ball_c208` | 0.74 [0.33] | 5.27 [2.12] | 3.38 [1.30] | PASS all (IFEval at the line) |
    | `qom_0_live_ball_c208_nw1` | 0.68 [0.27] | 5.33 [2.35] | 3.72 [1.39] | code PASS / IFEval FAIL / multi FAIL |

    - At the published pool, nw1 equals its twin on all rows: IFEval 47 / 49 (z -0.20), multi 32 / 33 (z -0.12). On
      non-tie rows IFEval is 24 / 12 (z +2.00, +0.38 points): flagged, but under the 2.22 line.
    - At 208 slots the pool carries the chat cost. c208 alone vs the published pool: IFEval 86 / 44 (z +3.68), non-tie
      47 / 12 (z +4.56). nw1 on top adds nothing significant (IFEval 56 / 54, z +0.19; multi 50 / 38, z +1.28), but it
      tips IFEval non-tie (2.35) and multi all-row (3.72) over the lines.
  - **Reading**:
    - Never-wait passes code at every pool tested, clean.
    - On chat it passes at the published pool and fails at 208 slots, where the small pool itself sits at the lines.
    - So the engine A/B (BigMoE; Leap's engine side) is open for code. For chat, never-wait needs a pool at or above the
      published one, or a chat read at the engine's own pool.
    - Adoption still needs EvalSuite same-scores on code, and the speed needs the engine test (FAR's spec section 3).
- Fri 21:3x: PRE-REGISTRATION, NW3 = NEVER-WAIT v3 (b) at its worst case (NewIdeas QUEUE dc9aff8 S1 + 6fd0224; Leap's
  TABLE_350 "NEVER-WAIT v3 staged"). It runs in Saturday's map right after T1, before step 0.
  - **v3 in the engine.** A cold demand use goes late (nw1). (a) A cold QOM optional use is dropped (weight 0). (b) A
    cold LSB sidecar whose base is resident runs at 3 bits now (MSB only) instead of waiting.
  - **(a) changes nothing on the map.** The QOM masks already drop a cold optional use (keep 0, weight 0), and nw1's
    late flags lie within keep (FAR's check_masks.py).
  - **(b) is run at its worst case.** It cannot be modelled from residency: the map's pools hold whole experts, and the
    engine's sidecar pool is separate. So every unsure answer row (draft cf < 0.9, the engine's R57_U3_THR default, or no
    draft) is given a cold sidecar on ranks 0-1:
    - those ranks run at 3 bits on time (records3's MSB-3 record, the engine's 3-bit read, through dm_t3);
    - a late rank (a cold base) stays 4-bit and late, as in v3;
    - ranks 2-9 stay 4-bit, as in every map arm, so D against nw1 is (b) alone.
    If the worst case passes, (b) passes at any coldness.
  - **Built and installed** (21:35; dm_common.py 9e970279, dm_ablations.py 4d6bef35; old files *.pre_nw3): scripts/dm_s7/
    nw3_patch.py (anchored), test_nw3.py, install_nw3.sh. CPU tests ALL OK on the installed files (data/dm_s7/nw3_install/):
    - FAR's whole never-wait suite still passes (old arms bit-equal to the pre-never-wait harness);
    - test_nw3: old arms bit-equal to the live harness (baseline, twin, nw1, nw0, the t3 gate); nw3 through a 4-bit stand-in
      == nw1 (2.6e-16); nw3 at 3 bits == an independent reference (2.6e-16), with power (late ranks at 3 bits, the sure
      rows gated, no 3-bit term: 2e-3 to 6e-3 away); counts (3-bit terms == the masks', prompt rows never); guards;
      measure() end to end; no state leaks.
  - **Arms** (code = the 12 windows; twins = nw1 at the same pool, run Fri 13:xx):
    1. `qom_0_live_ball_c208_nw3c+code`: the control, the same rows through the same path at 4 bits. That path's summation
       order alone moves near-ties (on record, t4_all: 0.33 / 2.15 / 1.58 % with every rank on it).
    2. `qom_0_live_ball_c208_nw3+code` and 3. `qom_0_live_ball_nw3+code`: they decide.
    4. `qom_0_live_ball_nw3` and 5. `qom_0_live_ball_c208_nw3` (36 windows): chat on record.
  - **Lines** (NewIdeas: the usual lines vs REF2-T plus D vs nw1), per pool:
    - PASS: VERDICT2 code (all-row <= 1.34 % AND non-tie <= 0.87 %) AND D vs nw1 <= +0.40 points ("clean" <= +0.15);
    - BETWEEN: all-row in (1.34, 2.0] %, or D in (+0.40, +0.80];
    - KILL: all-row > 2.0 %, or D > +0.80.
    - nw3 PASS = PASS at both pools; PASS at the published pool only = pool-conditional, as in nw1's rule.
    - The control is reported beside it. If its own D against nw1 is over +0.20, nw3 is also read against the control,
      with a note on how much of nw3's D is the path rather than 3 bits.
    - A failed worst case is not a KILL of (b): the engine's count of 3-bit sidecars per check (NWCNT) says how much of the
      bound it really meets.
    - Chat (on record): VERDICT2 per domain (IFEval 5.42 [2.22], multi 3.69 [1.63]), and D vs nw1 with McNemar.
  - **Predictions** [G], from t3_all (+0.59 code / +5.09 IFEval / +4.25 multi against its control, all on uncertain rows)
    and M-18 (sure rows' ranks 0-1 at 3 bits: +0.12 code):
    - code: D +0.2 to +0.5 at either pool; PASS 55 %;
    - chat: IFEval / multi D +1 to +3 points; VERDICT2 FAIL likely at 208 slots, a coin toss at the published pool;
    - the control: D within +-0.15 on code.
  - Saturday's list: the 8 leftover 24 GB arms moved to data/sun_extra.txt (Sunday night; NewIdeas 6fd0224).

- Fri 23:4x: NW3 RE-TIMED (NewIdeas 7e74d6c: Kamen opened the window at 23:30). The lines and predictions above are unchanged.
  - New order: nw3's 3 code arms (the control, 208 slots, the published pool) run first in the map, right after
    EvalSuite's runner 49. They cannot run beside it: runner 49 declares MEM 46 GB, the map 18 GB, and that plus 8 GB is
    over 64 GB.
  - Then EvalSuite's CERT reference decodes, then the rest: nw3's 2 x 36-window arms (chat on record), step 0, the
    2-min NW self-check (record only; nw3's control arm checks the patched path on the GPU), then T1.
  - Driver: scripts/dm_s7/nw3_first.sh. The runner now keeps its share lock between arms (S67_KEEPLOCK=1), so EvalSuite's
    runner 53 cannot start in the gap between two code arms.

- Sat 01:4x: NW3 CODE AT 208 SLOTS: PASS [K] (D +0.18 vs nw1, just over the "clean" +0.15); the control is clean (D +0.06).
  The published pool's arm was still to run (below). Read: scripts/dm_s7/nw3_read.sh -> data/nw3_code_read.txt (code_read.py).

  | arm (12 code windows, 3366 rows) | all-row | non-tie | tie rows | vs nw1 at 208 slots (0.68 % [0.27]) |
  |---|---|---|---|---|
  | `qom_0_live_ball_c208_nw3c+code` (control) | 0.74 % (25) | 0.30 % | 30.0 % | D +0.06; only-arm / only-twin 7 / 5, z +0.58 |
  | `qom_0_live_ball_c208_nw3+code` | 0.86 % (29) | 0.45 % | 28.0 % | D +0.18; 13 / 7, z +1.34 (non-tie 8 / 2, z +1.90) |

  - nw3 against its control: 12 / 8, z +0.89 (non-tie 8 / 3, z +1.51). The control's own D is under +0.20, so the read
    against nw1 stands. VERDICT2 PASS on both arms. No flip difference is significant (|z| < 1.96).
  - Against the predictions [G]: code D +0.2 to +0.5 -> +0.18, at the low edge; the control within +-0.15 -> +0.06.
  - Timeline: nw3_first.sh saw the end of runner 49's first instance at 00:54, claimed at 00:56, and ran the control and the
    208-slot arm by 01:01 (121 s each). The published-pool arm was stopped twice by others' quiet windows (01:03, 01:30) and
    wrote nothing (a stopped arm writes nothing).
  - OVERLAP 01:29:09-01:30:31: that arm went in at the end of a quiet window, and runner 49's process 7 (C 224, MEM 46 GB)
    started 4 s later. Runner 49 passes its gate (others' MEM locks, then NO-SWAP memory) BEFORE taking its share lock, and
    retries a lock refused for a quiet window every 60 s with no new check. 82 s side by side; swap +0.37 GB (6.61 -> 6.97);
    runner 49's process 7 ended rc 0 (same-scores, no timing). My arm stopped itself for BigMoE's quiet ticket at 01:30:31;
    I stopped the driver at 01:31.
  - RE-KEYED (NewIdeas 01:3x: "take the NEXT runner-49 chunk boundary instead of runner 49's end"). The claim (a share lock,
    MEM 18 GB) goes in DURING a runner-49 process. The arm starts once runner 49's gate logs that it waits behind the claim,
    or once runner 49 already waited behind a MEM lock and its log has not moved for 75 s, or once runner 49 is over; always
    with NO-SWAP >= 25 GB. The s67 runner now ends after a stopped arm or a wait instead of retrying (S67_STOP_EXIT=1; unset
    = as before), and the driver claims again (at most 6 rounds). Re-armed 01:40:14 during runner 49's process 8; at 01:40:57
    runner 49 logged "stack fwd process 9: another session's MEM job holds a shared lock: waiting (day rule)" and the arm went.

- Sat 01:4x: NW3 CODE VERDICT: PASS AT BOTH POOLS [K] (nw3 PASS on code under the pre-registered rule; clean at the published
  pool). Sent to BigMoE and NewIdeas at once (S1's NW v3 engine A/B, final v1); BigMoE queued that A/B within minutes. Arm run
  01:41:21-01:43:39, alone (runner 49 waited behind the lock from 01:40:57; swap flat at 6.97 GB).

  | pool | arm | all-row | non-tie | D vs nw1 | McNemar vs nw1 (all-row; non-tie) | line |
  |---|---|---|---|---|---|---|
  | published | `qom_0_live_ball_nw3+code` | 0.71 % (24) | 0.30 % | +0.12 (nw1 0.59 % [0.18]) | 12 / 8, z +0.89; 7 / 3, z +1.26 | PASS, clean |
  | 208 slots | `qom_0_live_ball_c208_nw3+code` | 0.86 % (29) | 0.45 % | +0.18 (nw1 0.68 % [0.27]) | 13 / 7, z +1.34; 8 / 2, z +1.90 | PASS |
  | 208 slots | control `..._c208_nw3c+code` | 0.74 % (25) | 0.30 % | +0.06 | 7 / 5, z +0.58 | clean |

  - (b) ran at its worst case (every unsure answer row's ranks 0-1 at 3 bits on time), so on code it passes at any coldness.
  - Predictions [G]: code D +0.2 to +0.5 -> +0.12 / +0.18 (both under the range; PASS 55 % -> PASS); control +-0.15 -> +0.06.
  - Next on record: nw3's two 36-window arms (chat: IFEval / multi VERDICT2 and D vs nw1) in map_rest.sh, after the CERT
    reference decodes.

- Sat 04:2x: NW3 CHAT (on record) [K]: VERDICT2 FAIL on IFEval and multi at BOTH pools. D vs nw1 is +1.0 to +2.4 points,
  inside the predicted +1 to +3. The published pool's coin toss came down FAIL. Code stays PASS, as decided at 01:4x.
  Arms run 03:55:08-04:07:32 (36/36 windows each, swap flat at 6.9 GB). Read: data/map_rest_read.txt.

  | pool | arm | IFEval all-row [non-tie] | multi all-row [non-tie] | D vs nw1 (IFEval / multi) | McNemar vs nw1 (all-row; non-tie) | VERDICT2 |
  |---|---|---|---|---|---|---|
  | published | `qom_0_live_ball_nw1` (twin) | 3.97 % [1.38] | 3.24 % [0.85] | | | PASS / PASS |
  | published | `qom_0_live_ball_nw3` | 6.18 % [2.61] | 5.60 % [2.48] | +2.21 / +2.36 | IFEval 122 / 47, z +5.77; 52 / 14, z +4.68. multi 122 / 38, z +6.64; 67 / 13, z +6.04 | FAIL / FAIL |
  | 208 slots | `qom_0_live_ball_c208_nw1` (twin) | 5.33 % [2.35] | 3.72 % [1.39] | | | FAIL / FAIL (already) |
  | 208 slots | `qom_0_live_ball_c208_nw3` | 6.33 % [3.28] | 5.46 % [2.45] | +1.00 / +1.74 | IFEval 95 / 61, z +2.72; 54 / 25, z +3.26. multi 104 / 42, z +5.13; 55 / 20, z +4.04 | FAIL / FAIL |

  - Lines: IFEval 5.42 [2.22], multi 3.69 [1.63]. VERDICT (all-row): published IFEval BETWEEN, multi KILL (5.60 > x1.5
    5.54); 208 slots BETWEEN / BETWEEN.
  - **The worst-case rule sets the chat rate, not the pool.** nw3 lands at ~6.2-6.3 % IFEval and ~5.5-5.6 % multi at both
    pools; only nw1 differs by pool. That is why D is larger at the published pool.
  - **For v1:** (b) at its worst case costs chat, though not code. The engine's real share of it is NWCNT (3-bit
    sidecars per check) on chat prompts. So NW v3 is code-validated only. A chat claim needs one of: NWCNT on chat
    prompts, an engine chat run, or NW v3 off for chat.
  - Cross-check: t3_all (every rank at 3 bits on uncertain rows) cost +5.09 / +4.25 against its control; nw3 (ranks 0-1
    only) costs about half that.

- Sat 04:2x: STEP 0 RESULT (PLAN_12GB_90) [K]: ALL THREE P1-K1 KU 4 cf 0.9 forms PASS VERDICT2 on code. Per the
  pre-registered rule ("the fastest passing form is the 12 GB base"), the base is **T3R2's own form,
  `p1c_oea_k1_ku4_p08_cf09` (t80ck1u, θ 0.8): 45.8 tok/s [P, Fit12's replay, full outputs, cf 0.9]**. The fastest
  code-passing point before this was top-2 θ 0.95 at 35.9 [P] (+28 % [P]). Arms run 04:07:33-04:12:54 (12/12 windows each).

  | arm | θ | all-row | non-tie | tie | slack to the line (all-row / non-tie) | prediction [G] | tok/s [P] |
  |---|---|---|---|---|---|---|---|
  | `p1c_oea_k1_ku4_p08_cf09+code` (T3R2's form) | 0.8 | 1.31 % (44/3366) | 0.78 % (26/3316) | 36.0 % | 1 flip / 2 flips | 1.4-1.6 % | 45.8 |
  | `p1c_oea_k1_ku4_p09_cf09+code` | 0.9 | 1.16 % (39) | 0.66 % (22) | 34.0 % | 6 / 6 | 1.3-1.5 % | 41.0 |
  | `p1c_oea_k1_ku4_p095_cf09+code` | 0.95 | 1.10 % (37) | 0.60 % (20) | 34.0 % | 8 / 8 | 1.1-1.3 % | 37.9 |

  - Line: VERDICT2 code, all-row <= 1.34 % AND non-tie <= 0.87 %.
  - McNemar:
    - θ 0.8 vs cert_top5_p09 (0.92 %): 25 / 12, z +2.14 (non-tie 17 / 5, z +2.56). Top-5 pass 1 is still cleaner, but it
      overflows C = 47.
    - KU 4 + cf gate vs the plain P1-K1 θ 0.8 (`p1c_oea_k1_p08`, 1.52 %): 13 / 20, z -1.22 (non-tie 7 / 14, z -1.53). At
      θ 0.9: 15 / 27, z -1.85 (non-tie 8 / 19, z -2.12).
    - θ 0.9 vs θ 0.8: 7 / 12, z -1.15. θ 0.95 vs θ 0.8: 8 / 15, z -1.46.
  - Predictions [G]: all three came in at or under the bottom of their ranges.
  - **Caveats.** θ 0.8 passes by 1 all-row flip. A second sample or EvalSuite same-scores could tip it. θ 0.9 (41.0 [P],
    6 flips of slack) is the safe fallback. The speeds are Fit12's replay, calibrated within 2-4 % of T3R2's measured
    runs. This is FAST, not exact: adoption needs EvalSuite same-scores on code and Fit12's measured run on full outputs.
  - From 45.8, 90-130 tok/s needs x1.97-2.84. Sent to Fit12 (peer say) and to the orchestrator (next-caliber,
    12 GB line).

- Sat 04:2x: T1 RESULT (Fable A's LOSSY350_A T1) [K]. Input sparsity on the routed experts: every MoE block's switch reads
  x with all but the c largest-|x| of each 64-wide input group zeroed; the router and the shared expert keep the full x.
  Ran from sat_t1.sh, 04:13:18-04:21:59, swap flat at 6.9 GB. Lines and decide.py are Fable A's ("room" = non-tie flips
  left above today's preset).

  | arm | kept of 64 | code all-row | code non-tie | IFEval [non-tie] | multi [non-tie] | predicted non-tie [G, Fable A] | Fable A's line |
  |---|---|---|---|---|---|---|---|
  | xs64 | 64 (identity) | 0.00 % (0) | 0.00 % | | | 0 | OK: the wrapper is inert |
  | xs48 | 48 | 0.45 % (15) | 0.12 % | 3.74 % [0.87] | 2.67 % [0.67] | 0.05-0.07 code; 0.55-0.66 / 0.48-0.54 chat | FITS (room 0.66 / 1.03 / 0.72); lever alive (kill 0.21) |
  | xs40 | 40 | 0.95 % (32) | 0.33 % | | | 0.17-0.24 | FITS |
  | xs32 | 32 | 1.43 % (48) | 0.66 % | | | 0.42-0.60 | OVER; lever alive (kill 1.03) |

  - Every measured non-tie cost ran 1.2-2x over Fable A's predictions. xs48's multi room is thin (0.67 vs 0.72).
  - Stand-alone VERDICT2: xs48 passes all three domains; xs40 passes code; xs32 fails code all-row (1.43 > 1.34).
  - Quality only: the speed side (a kernel that skips the zeroed inputs) is Fable A's to price. No speed is claimed.

- Sun 00:1x: M1 (CHAT150 STRIKE, NewIdeas 7788571 / c16f510): THE 45 GB CHAT MISS LEVERS, RANKED. Bytes [P]: the
  BYTES10 replay (NewIdeas' bytes10_replay_fix.py core, unchanged, + 9 rules and pool splits: scripts/m1/m1_replay.py).
  Quality [K, map]: chat flips of the existing 45 GB-pool arms (scripts/m1/m1_quality.py). Data: data/m1/.
  - **Calibration.** The replay's EXACT chat row is within ~5 % of the engine's N45 exact run (BigMoE, nie1/nie2,
    676 checks each): 38.1 vs ~40 MB per kept token on IFEval.
    - The engine reads 101 MB and 89.8 misses per check (~1.13 MB per miss, so ~74 % of misses are 0.77 MB LSB sidecars).
      The replay reads 88.5 MB and 63 misses (55 % LSB).
    - So the engine is LSB-heavier than the replay, and the sidecar levers below should do at least as well there.
  - **Finding 1: QOM is not active on chat today.** In both the exact and the lossy (L) chat runs, the engine's
    qom_skip = qom_opt = 0 and NW is idle, and L's chat reads the same bytes as exact (99.8 vs 101.2 MB per check). L's
    chat gain is kernels, not misses.
  - **Finding 2: on chat, QOM costs nothing; gating the bonus row is what costs.**
    - QOM at cf 0 vs QOM at cf 0.5 (`qom_0_live` vs `qom_0.5_live`): z -0.54 / +0.70.
    - The bonus row gated at cf 0.5 (`qom_0.5_live_bonus5g`): z +0.74 / +1.34.
    - Every last row gated (`qom_0_live_ball`, DRAFT17's bound): z +4.10 / +5.69, but still inside the lines.
  - **Finding 3: a pool split pays only under QOM.** Moving S base slots per layer to 2.81 S sidecar slots (equal bytes):
    - EXACT: -1 % at S 16, +11 % at S 32 (no exact lever on its own);
    - QOM0 + ball: -41 % more at S 32-48.
    QOM frees the base pool (cold ranks 2-9 are no longer read), and the sidecars are where chat misses.

  | # | lever (45 GB pool, live checks, LRU) | MB per kept, IFEval / multi (vs EXACT) | misses per check, IFEval (MSB + LSB) | svc [P, 15 ms x MB ratio] | chat quality on the map, all-row [non-tie] IFEval / multi; McNemar vs `qom_0.5_live` | verdict |
  |---|---|---|---|---|---|---|
  | - | EXACT (today) | 38.1 / 33.2 | 63.3 (28.6 + 34.7) | 15 ms | (no flips by construction) | - |
  | 1 | QOM0: every drafted row skips its cold, non-union ranks 2-9; ranks 0-1 always; bonus row exact | 25.7 / 24.3 (-32 / -27 %) | 50.0 (15.3 + 34.7) | -4.8 | `qom_0_live` 2.53 [0.45] / 1.77 [0.33]; z -0.54 / +0.70 | free |
  | 1s | QOM0 + split S 16 | 22.5 / 22.6 (-41 / -32 %) | 35.3 (18.0 + 17.3) | -6.2 | + a ~5 % smaller base pool [P, small] | free [P] |
  | 2 | QOM0.5 + bonus row gated at the next draft's cf >= 0.5 (needs the MTP head's cf calibrated) | 22.8 / 22.0 (-40 / -34 %) | 46.9 | -6.0 | `qom_0.5_live_bonus5g` 2.88 [0.55] / 1.91 [0.39]; z +0.74 / +1.34 | ~free |
  | 2s | the same + split S 16-32 | 18.7 / 19.7 (-51 / -41 %) | 27.2-31.3 | -7.7 | as 2, plus the pool | ~free [P] |
  | 3 | QOM0 + every bonus row gated (the 45 GB FAST code rule, QOM-KEEP + DRAFT17, on chat) | 16.0 / 13.0 (-58 / -61 %) | 39.5 (4.8 + 34.7) | -8.7 | `qom_0_live_ball` 4.03 [1.00] / 3.26 [0.91] PASS; z +4.10 / +5.69 | costs; inside the lines (multi margin 0.43) |
  | 3s | the same + split S 32 | 9.4 / 8.4 (-75 / -75 %) | 17.2 (6.2 + 11.0) | -11.3 | as 3, plus the pool (~+0.3-0.5 points [G]; c208 cost +1.2 at -30 % slots) | the IFEval gate decides |
  | 4 | U3: a cold LSB sidecar is not read (ranks 0-1 at 3 bits when cold) = NW v3's (b) at its real rate | 26.6 / 24.7 (-30 / -26 %) | 28.6 (LSB 0: -55 % misses) | -4.5 (bytes) .. -8 (count) | worst case (`..._nw3`, every unsure row) FAIL: z +8.4 / +10.3. 3 bits on all sure rows (`qom_0.5_live_t3`) z +5.0 / +6.0. Real rate ~12 % of rank 0-1 LSB uses [P] | needs a fresh gate; after split 3s the LSB misses are already -68 % |
  | 5 | NW1 on chat (cold demand uses go late; no 3 bits) | bytes unchanged; hides waits | - | most of what is left [P, unpriced] | `qom_0_live_ball_nw1` vs its twin: z -0.20 / -0.12 (non-tie IFEval z +2.00, flagged) | ~free |
  | 6 | TOP8 on chat (CH5) | 30.2 / 25.9 (-21 / -22 %) | 54.8 | -3.2 | `top_k8` 5.24 [1.90] / 4.31 [1.54], multi FAIL; z +7.03 / +7.68 | DROP: dominated by QOM0 (fewer bytes saved, more damage) |
  | 7 | MSB-2 on ranks 2-9 (BC2, below) | after QOM, <= -4..-9 % of the miss bytes | - | <= -1 | s7 screens: +0.6 (4 layers) to +7.4 points (48 layers) IFEval | DROP for chat |
  | - | OPT (any residency policy's ceiling) | EXACT 19.6; QOM0+ball 9.0 | - | - | exact | PIN / RCP gained nothing (CACHE.md, BYTES10) |

  - **Handed to BigMoE** (engine A/Bs on 12 IFEval at C 256 after CERT; then the EvalSuite IFEval + multi gate vs
    NAX6-N, both orders):
    - **M1-A (~free on the map):** QOM on every drafted chat row (cf >= 0) + a 16-slot base-to-sidecar shift
      (C 256 -> 240, C_l +45). [P] -41 / -32 % MB per check, svc ~ -5..-6 ms.
    - **M1-B (max):** M1-A + the bonus row gated (DRAFT17 on chat) + a 32-slot shift. [P] -75 % MB per check, svc
      ~ -11 ms. It costs +1.4 / +1.6 points of flips vs M1-A on the map; the gate decides.
    - Then NW1 on chat, (b) off, on top of the winner.
    - Read per arm: svc, misses and MB per check, tok/s whole decode, and every token vs AR.
    - The best shift in the engine may differ: its pools are C 256 + VTIER 64 base, C_l 112 + 32 sidecar, against the
      replay's 295 / 89. Try S 16 and 32.
  - Caveats:
    - The map's cf is DFlash2's; chat runs the MTP head. QOM0 and the 'ball' bound need no cf; rule 2 does.
    - svc is priced proportional to MB per check. A count-bound SSD would favour the LSB levers (3s, 4).

- Sun 00:1x: BC2 ANSWER (NewIdeas 01:1x for Leap: a 2-bit / MSB-2 read path at 45 GB?) [records + map]:
  1. **Read path.** At 24 GB, yes: BigMoE's R-44b POM (LEDGER R-44b; HANDOVER Thu).
     - records3p seg1 (2 planes + fields) / seg2 (plane 3 + b3); the 2-plane slot is a 3-bit slot with plane 3 = 0 and
       bias2. It is bitwise EvalSuite's tier 2, with no kernel change.
     - b45 measured bytes -19 % but tok/s -1.9 %. Each 2-plane read or upgrade pays a ~0.1 ms CPU rebuild, so SSD busy
       fell only -6.5 %.
     - At 45 GB (r57 / ref2 tiers) I know of no port. It needs records3p for the converted store (disk), the POM pool
       state and upgrades in r57's tier pool, and, for any GPU-side byte cut, a true 2-bit slot and kernel. The 3-bit-slot
       trick saves SSD bytes only. BigMoE / Leap to confirm.
  2. **What is different now: QOM.** M1's QOM on chat already skips cold ranks 2-9 entirely, at ~zero map cost. After
     QOM, MSB misses are 30 % of chat bytes (QOM0) or 12 % (QOM0 + ball), so MSB-2 on misses has <= -4..-9 % left to
     save. MSB-2 on resident ranks saves no SSD bytes.
  3. **IFEval risk: high** [K, map, s7 screens at the 24 GB form, every kept expert of the gated rows]:
     - MSB-2 at 4 layers: +0.64 / +0.62 points (IFEval / multi) over s7_ref;
     - at all 48 layers: +7.4 / +8.2 points (13.92 / 13.59 %; KL x4.7 / x6.2);
     - MSB-3 on every kept expert (s7_ref3): +1.1 / +1.0.
     Ranks 2-9 only would be in between ([P, by weighted NMSE] ~+2-2.5 points IFEval): a likely FAIL on chat, for a
     small byte gain. Recommendation: drop BC2 for chat; QOM is the stronger form of the same cut.

- Sun 00:1x: 24 / 12 GB CHAT BYTES (NewIdeas 00:2x re-aim; parked until after Monday by NewIdeas 00:4x). On record from
  NewIdeas' own BYTES10 replay (~/NewIdeas/data/ledger/bytes10_replay_fix.out; IFEval / multi, live checks, LRU):
  - 24 GB (C_m 147 + C_l 46): EXACT 210 / 183 MB per kept token. QOM 0.5 (the 24 GB chat route, which PASSes chat at
    147 slots) 120 / 111. OPT 94 / 84. Synthetic kept per check: EXACT flat at ~127 / 112 when no row is wasted.
  - 12 GB (C_m 51 + C_l 32): EXACT 622 / 487. The best priced rule (COLD 0 t3) 194 / 156. Q1 0 t3 at 16 kept per check
    80 / 61 (no chat gate passes it).
  - <= 80 MB: at 24 GB, the passing route + fewer wasted rows (F2a) + the QOM pool split are the candidates (to price
    after Monday). At 12 GB no gated rule gets near it [P].

- Sun 04:2x: M1-B [K] AGAINST MY [P], AND WHAT THE ENGINE'S M1-B IS (BigMoE CHS2, data/chs/chs2_report.txt; NewIdeas
  e26eb0e).

  | arm (12 IFEval, C 256, whole decode) | MB per kept | misses per check | svc ms | tok/s | my [P] for MB per kept |
  |---|---|---|---|---|---|
  | x2c0 (L) | 41.8 | 97.8 | 15.8 | 44.8 | (replay EXACT 38.1) |
  | x2qm1 (QOM on drafted rows, bonus row exact) | 22.8 (-45 %) | 42.8 | 10.1 | 45.3 (+1.0 %) | QOM0 -32 % |
  | x2qm2 (+ the bonus row gated = M1-B) | 15.5 (-63 %) | 25.8 | 6.3 | 49.0 (+9.5 %) | QOM0+ball -58 % |
  | x2qm2s (+ BigMoE's pool split) | 15.1 (-64 %) | 28.2 | 7.9 | 52.0 (+16.1 %; kept 2.72 vs 2.41) | + split S 32: -75 % |

  - **The replay was low on the miss cut** (-38 % predicted vs -74 % measured for M1-B) **and high on the split** (-41 %
    more predicted vs ~0 measured).
  - The engine already holds more sidecar slots than the replay's 45 GB line: C_l 112 + VTIER 32 against 89. So the
    split's lever was mostly in place already.
  - x2qm2s's +16 % over x2qm2 rides on kept 2.72 vs 2.41 (lossy outputs differ); it is not a split effect.
  - **The engine's M1-B also includes U3.** r57_screen.py FASTARMS["QOM"] = _S(thr 0, q1u, t3=True, qom=True); "Tail
    3"; the screen asserts that QOM needs the t3 path.
    - Its t3 turns OFF the gated rows' sidecar READS. A pair whose LSB slot is resident still runs at 4 bits (LP1:
      "unchanged for slots >= 0"). So it is U3 at its real rate (cold sidecars only), not the map's t3, which puts every
      sure row's ranks 0-1 at 3 bits.
    - The map brackets it: `qom_0_live_ball` (no 3 bits) 4.03 [1.00] / 3.26 [0.91] PASS, and `qom_0_live_bq_t3` (every
      sure row at 3 bits) 6.09 [2.48] / 5.29 [2.36] FAIL. The real rate should sit near the first [P].
  - **If the IFEval gate fails M1-B,** the clean fallback (QOM without U3) needs a build: R57_QOM_TAIL=4 is "not built",
    and QOM asserts the t3 path.

- Sun 04:2x: 24 / 12 GB CHAT BYTES10, PRICED [P] (scripts/m1/m1_replay.py with M1_POOL=24 / 12; data/m1/m1_replay_24.out,
  _12.out; the ledger's pools: 24 GB = C_m 147 + C_l 46, 12 GB = 51 + 32; 12 IFEval + 12 multi windows, live checks,
  LRU). MB per kept token, IFEval / multi:

  | rule | 24 GB | 12 GB | chat gate on the map |
  |---|---|---|---|
  | EXACT | 210 / 183 (OPT 94 / 84) | 622 / 487 (OPT 387 / 304) | - |
  | QOM0 (drafted rows skip cold ranks 2-9) | 105 / 96 (-50 / -48 %) | 276 / 242 (-56 / -50 %) | free at 45 GB; untested at 147 / 51 slots |
  | QOM0.5 + bonus gated at the next draft's cf >= 0.5 (~ the 24 GB chat route) | 82 / 77 (-61 / -58 %) | 216 / 199 | `qom_0.5_live_c147_bonus5g` PASS (4.30 / 2.73 %) |
  | QOM0 + ball (= M1-B without U3) | 49 / 35 (-77 / -81 %) | 113 / 83 (-82 / -83 %) | c208 PASS thin (5.27 [2.12] / 3.38 [1.30]); 147 / 51 untested |
  | QOM0 + ball + split (best S) | 34 / 25 (S 32) | 111 / 82 (S 4; more slots hurt) | as above |
  | QOM0 + ball + U3 everywhere (`Q1 0 t3 +bonus all`) | 20 / 14 | 76 / 56 | the map's worst-case t3 FAILs (c147: 7.83 / 5.99) |
  | U3 alone | -14 / -11 % | -6 / -6 % | at 24 / 12 GB the base reads dominate, so U3 is small |

  - **The <= 80 MB line** (150 tok/s at ~12 GB/s):
    - 24 GB: reached by the map-PASSing chat route (~80) and by QOM0 + ball (49 / 35, gate open).
    - 12 GB: only QOM0 + ball + U3 reaches it (76 / 56), with no chat gate at 51 slots.
  - Absolute bytes differ by engine, so ratios are what to use. The 45 GB replay matched N45 within ~5 %. At 24 GB on
    code the replay reads 0.7x E24 (C 128, s10b); at 12 GB, 1.45x Fit12's loop.
  - **After the bytes, the GPU floor binds.** The same model's chat check has a fixed GPU cost of ~39-42 ms at 45 GB
    (BigMoE CHS2 FLOOR).
    - 24 GB with QOM0 + ball: ~113 MB per check, ~10-15 ms of SSD, so chat ~45-50 tok/s [P] (today ~20-30 [G]).
    - 12 GB: ~263 MB per check, so ~35-40 [P].
    - 150 at 24 / 12 GB needs the same B1 / G1 / G2 + K work as 45 GB, plus these byte cuts.
  - **Gates to pre-register before any run:** map arms `qom_0_live_ball_c147` and `qom_0_live_c147` (no 3 bits; masks
    by NewIdeas' dm_oea_masks rule at 147 slots), then the same at 51 slots. Each is ~5 min of GPU at 36 windows.
    Lines: VERDICT2 per domain; McNemar vs `qom_0.5_live_c147_bonus5g`. Waiting for NewIdeas to slot them (after
    Monday, or a gap).

- Sun 04:4x: QWEN 4 DAY-0 PRE-WORK, the decision map's capture with a model switch (NewIdeas' board, due Tue evening).
  INSTALLED.
  - Patched ~/FableGod/decision_map/dm_common.py (483b7247 -> 558671e0; the old file kept as dm_common.py.pre_q4) with
    the anchored patcher scripts/q4/q4_map_switch.py (10 anchors; a scratch copy tested first, then an atomic move, no map
    process running):
    - DM_MODEL picks the store (default today's ~/BigMoE/models/bare).
    - DM_ARCH picks the architecture module (default mtplx.models.qwen4_exp). A module without today's class names fails
      loudly at import.
    - The routing and x-capture ranks follow the MoE block's own top_k (CAPK, today 10).
    - The routing / scores / xcap write-outs stack over the discovered MoE layers (MOEL, today 0..47) instead of
      range(48).
  - CPU checks, all OK: the default env gives the same MD and module; DM_MODEL switches the store; a wrong DM_ARCH
    raises ModuleNotFoundError.
  - PRE-REGISTRATION, "today's capture reproduced" (GPU, ~20 min, share lock MEM 18 GB, when NewIdeas slots it; disk is
    at 55 GB free, so only small outputs):
    - Run: DM_OUT=<scratch>/q4repro DM_C=51 run_map.py capture (36 windows), then xcapture --max 1 (3 windows).
    - PASS = bit-identical tk / lp vs out/baseline.json, routing ids / tokens / gen vs out/routing, and x / w / ids vs
      out/xcap for those 3 windows. Anything else is a bug in the switch.
  - Day-0 hazards for the map (each fails loudly or needs a rebuild; none is silent):
    1. The windows are today's token ids (TrainLab's d02b walks, ids only). A new tokenizer needs windows rebuilt from
       the texts plus the new model's own AR answers (EvalSuite's AR generator with a model switch).
    2. cf-based arms need a drafter walk for the new model (gate_dfl2.json, floor_rows.json, the masks). QOM0 and 'ball'
       arms need no cf.
    3. The "gate" check compares against today's D-02a top-1. Skip it for a new model, or point it at the new
       baseline.
    4. The ablation paths (reuse, nw, t3, masks) still hold 48 / 10 literals. Sweep them before any arm runs on a new
       model; the capture itself is covered.

- Sun 11:3x: THE ORCHESTRATOR'S SLOT (GPU idle since 09:46; share lock, MEM 18 GB, done by 22:30): the Qwen 4 capture
  reproduction + the 24 / 12 GB chat map gates pre-registered at 04:2x / 04:4x. Runner scripts/q4/q4_repro_slot.sh
  (swap +2 GB guard, wall caps), check scripts/q4/q4_repro_check.py.
  - The 147-slot gates need no new masks: NewIdeas' qom_0_live_c147 and qom_0_live_ball_c147 (the 4-bit twins of the
    screen 4 / 6 t3 masks, built and checked Wed) were snapshotted into out/ni and registered with install_c3.sh (test
    ALL OK: PERT, loader path, 36/36 byte-equal, rank 0 kept on every row).
  - The 51-slot masks: built with my unchanged copy of NewIdeas' builder (scripts/m1/dm_oea_masks.py, sha256 a939c473,
    NewIdeas 9b6451e), 45 s on one core. Self-check: my copy rebuilds qom_0_live_c147 = NewIdeas' masks, 36/36 windows
    bit-equal. Experts kept per realized row-layer (code / IFEval / multi):
    | arm | code | IFEval | multi |
    |---|---|---|---|
    | qom_0_live_c147 | 8.30 | 9.09 | 9.24 |
    | qom_0_live_c51 | 6.35 | 7.22 | 7.85 |
    | qom_0_live_ball_c51 | 5.93 | 6.48 | 6.96 |
    At 51 slots the sure rows' rank 2-9 pairs kept fall to 56 / 62 % (IFEval / multi, ball) from 79 / 80 % at 147; load
    miss rate 27.5 / 20.7 %. Expect more flips than at 147.
  - RESULT, "today's capture reproduced" (Sun 11:35-11:40) [K]: PASS, bit-exact on every pre-registered comparison
    (data/q4/q4_repro_check.out). Board item QWEN 4 DAY-0 "Breakthrough" DONE.
    - capture, 36 windows (10,316 answer rows), C 51, 238 s, with the patched dm_common (log: "MoE layers 48 (0..47);
      capture ranks 10"). capture.json is byte-equal to out/baseline.json (11-02 04:03) and out/capture.json (11-02
      10:46). Routing ids / tokens / gen equal on 36/36 windows.
    - xcapture --max 1 (HumanEval_10, ifeval_1005, multi_ar21; 949 rows): x / w / ids / a equal to out/xcap (11-03),
      tk / lp equal to out/xcapture.json, its routing equal too.
    - Two harness notes. run_map's closing report() reads <OUT>/baseline.json, so with a scratch DM_OUT it exits 1
      after a complete capture; the runner now judges a step by its "[dm] <name> done in" lines. My check's first
      version doubled ".npz" and could pass on zero windows; both fixed before the verdict (0 windows now = FAIL).
    - So on day 0: DM_MODEL=<new store> DM_ARCH=<new module> run_map.py capture works as soon as the windows and the
      ablation literals (hazards 1-4 above) are dealt with.
  - RESULT, the 24 / 12 GB chat map gates (Sun 11:40-11:56, one process, 36 windows each, ~230 s each, 0 warnings)
    [K] (data/m1/chat_gates_24_12.out; under Kamen's 11-08 rule these flip lines are information that picks EvalSuite
    runs, not the gate). Lines: code 1.34 [0.87], IFEval 5.42 [2.22], multi 3.69 [1.63]; all-row % [non-tie].
    | arm | pool | code | IFEval | multi | map verdict | MB per kept [P] (IFEval / multi) |
    |---|---|---|---|---|---|---|
    | qom_0.5_live_c147_bonus5g (the 24 GB chat route, Wed) | 147 | 0.71 [0.33] | 4.30 [0.93] | 2.73 [0.85] | PASS | ~82 / 77 |
    | qom_0_live_c147 (QOM0, bonus row exact) | 147 | 0.74 [0.24] | 4.50 [1.26] | 3.26 [0.76] | PASS | 105 / 96 |
    | qom_0_live_ball_c147 (QOM0 + every last row) | 147 | 0.92 [0.54] | 6.36 [2.96] | 4.11 [1.85] | FAIL chat | 49 / 35 |
    | qom_0_live_c51 | 51 | 1.16 [0.54] | 8.80 [4.96] | 5.26 [2.66] | FAIL chat | 276 / 242 |
    | qom_0_live_ball_c51 | 51 | 1.40 [0.93] | 10.57 [6.53] | 7.99 [5.02] | FAIL all | 113 / 83 |
    - McNemar vs the route (all rows): qom_0_live_c147 IFEval z +0.62, multi z +2.06 (non-tie -0.60): the same class.
      qom_0_live_ball_c147 IFEval z +5.00, multi +3.91: clearly worse. At 45 GB the same ball rule passed (qom_0_live_ball
      4.03 / 3.26); with 147 resident slots instead of ~295, gating the last row drops more cold ranks 2-9.
    - 24 GB: no screened rule beats the route on bytes and stays clean. QOM0 with the bonus row exact is clean but costs
      more bytes than the route (105 vs 82). The cheap ball rule (49 / 35) fails both chat sets. So the 24 GB chat floor
      on clean rules stays ~80 MB per kept [P].
    - 12 GB: QOM at cf 0 fails chat even with the bonus row exact. A cf 0.5 rule (bytes 216 / 199 [P] with the bonus
      gated) is the next screen; the 80 MB line is not in reach at 12 GB with any clean rule found so far.
    - Next screens if NewIdeas wants them (~4 min GPU each, masks from my builder copy in seconds): qom_0_live_c147
      with the bonus row gated at 0.5 (between 105 and 50 MB [P]; both of its parents are clean), and qom_0.5_live_c51
      (+ bonus5g) for 12 GB.
- Sun 16:5x: NewIdeas' yes (11:59) to the two follow-up chat screens; READY, waiting for a GPU gap NewIdeas places
  (Strata's quiet ticket is first in line, so no share lock now). ~4 min GPU each, share lock MEM 18 GB, one command:
  Q4_ARMS="qom_0_live_c147_bonus5g qom_0.5_live_c51" scripts/q4/q4_repro_slot.sh <scratch> arms
  - qom_0_live_c147_bonus5g = qom_0_live_c147 (clean, 105 / 96 MB [P]) with the bonus row gated at 0.5 (proxy g) as in
    the route: built with my copy of Fable C's bytes10_bonus_masks.py (scripts/m1/bytes10_bonus_masks_bt.py, source
    sha256 c7f8970d; only the mask in / out dirs and an import path changed). Gated 165 of 1,368 bonus rows; only drops.
  - qom_0.5_live_c51 = cf 0.5 QOM at the 12 GB pool, bonus row exact (kept per realized row-layer 6.48 / 7.77 / 8.37).
  - Self-checks: my two copies rebuild NewIdeas' route masks qom_0.5_live_c147 and qom_0.5_live_c147_bonus5g, 36/36
    windows bit-equal each. Both arms installed with install_c3.sh (ALL OK; snapshots in out/bt).
