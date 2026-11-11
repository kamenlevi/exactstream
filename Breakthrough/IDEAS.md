# Breakthrough ideas (round 1, 2026-10-19)

Scored on the four numbers in SCOREBOARD.md: RAM bytes/token, SSD bytes/token, real RAM footprint, sustained tok/s,
plus quality (suite + % identical answers vs today's Q4). Budgets and the trace replays behind every *sim* number are in
SCOREBOARD.md sections 4-6. Research briefs (prior-art passes, 15 searches + 8 fetches each): `research/bits_limits.md`,
`research/routing_locality.md`, `research/sparse_consult.md`.

## The physics in one paragraph

At 6-12 GB the SSD term dominates: every token routes 480 expert uses, and routing is diverse enough that a few hundred
tokens touch half of the 68 GB of experts. No exact method moves that much: a perfect cache policy gives 1.66x at a 10 %
cache, and verifying K tokens per pass gives 1.06x at K = 4 and 1.9x only at K = 64. Three levers can: (1) change which
experts a token uses, so most uses hit RAM (up to 7.5x fewer SSD reads); (2) fewer bits per expert, which shrinks every
read and raises the share of experts that fit in RAM; (3) keep many tokens per big-model pass, which amortizes the
dense read and, with a per-pass expert budget, the expert reads. (1) and (2) multiply: at 12 GB a Q4-equal ~3.25-bit
trellis code plus a dense diet lifts residency from 7.8 % to ~15 %, and wait-top-2 there needs 30 MB/token from the SSD
(234 MB with exact routing). The RAM term then caps the target at ~60 tok/s on a 120 GB/s mini and ~125 on Kamen's Mac
(MTP, ~2.0 GB/token). The ideal (400-600) and one-stream 1000 tok/s need (3) and a different kernel structure on top.
6 GB and 4 GB are not reached by anything here yet: their residency (<= 2.5 %) is below even a 4-token working set.

## Status after night 1 (2026-10-20 01:30; details in SCOREBOARD.md section 8)
- **D1 alive.** The 125B with wait-top-2 kept GSM8K-150 at 145 (20 % residency) and 144 (10 %) vs 147 exact, with 144/150
  answers identical at both (same-weights floor 147/150). That mode still reads the missing low ranks in the
  background; the frugal mode, which saves the SSD bytes, hangs the fork's gate and is not measured yet. Next: a copy
  of fork125 with the frugal path fixed, and the gate-mass replay (`scripts/gate_mass.py`, traces with router probabilities).
- **D2 alive, reframed as a KLD budget.** Q4 costs 0.112 KLD (86.9 % top-1) against near-bf16 on the 35B; routed
  experts account for 0.041 of it. Experts at ~3.25 bits add +0.011, at ~3.0 bits +0.024. GDN is the most expensive Q4
  class (0.057) and, like down_proj, suffers more from affine Q4 than from white noise at equal error. So: experts at
  ~3-3.25 bits with a rotated code, GDN/down better than Q4, total KLD at or under the pack's.
- **D3 dead for the SSD.** F (tokens between must-fix tokens) is 7 exact / 28 lenient for Qwen3.5-4B; >= 100-250 is
  needed. Kept only for the one-stream speed goal, where the per-pass expert budget still applies.
- **Measurement fact:** on this MoE any perturbation, however small (NMSE 1e-5), costs ~0.011 KLD / ~3 % top-1 through
  routing flips. Judge every variant against that floor.

## The three directions

### D1. Budgeted routing: each token waits only for the experts that matter
- **Mechanism.** Per layer, a token waits for its top-r experts (r ~ 2, or a budget by gate mass x layer sensitivity);
  a missing lower rank is substituted by a resident expert the router also ranks (ranks 10-19), dropped and
  renormalized, or replaced by a cheap stand-in (the expert's mean output). Frugal mode reads nothing else from the
  SSD. Later: train the router with this miss policy in the loop.
- **Numbers.** SSD fetches/token at 10 % residency 152 -> 20 (7.5x); at 12 GB with D2's diet 234 -> 30 MB/token
  (*sim*). RAM footprint unchanged; tok/s then set by the RAM term (~60 mini / ~125 Kamen with D2).
- **Quality risk.** The whole question: ~160-190 of 480 rank-uses per token are not the router's choice. Evidence: the
  35B kept GSM8K-150 identical at 19-25 % residency (with background loads); substituting every rank lost 14 points
  while perplexity moved 4 %; flips come from top-2 substitutions in layers 10-39.
- **Prior art.** HOBBIT (2024, low-precision stand-ins for low-gate misses); "Which expert misses are worth waiting
  for?" (Oct 2026: fetch rank-1 or gate >= 0.15, PPL +0.35 % at ~25 % residency); cache-conditional experts (arXiv
  2412.00099, >50 % fewer misses, <= 64 experts); cache-aware joint router adaptation (2609.04895: -31 % misses at
  15.6 %, GSM8K equal, full-backbone tuning on task data); ReMoE (router-only, -4 to -7 % at 6-9 %); MoNE (mean-output
  stand-ins, static); MoEPrefetch's wait-top-2 (ours). Our 7.5x cut exceeds every published one. **Left for us:**
  reasoning-equal quality at <= 15 % residency on a 512-expert top-10 model, frugal loading, layer x gate-mass budgets,
  mean-output stand-ins for dynamic misses, a router trained under the miss policy. No paper shows reasoning-equal
  quality at <= 10 % on 256-512 experts.
- **First experiment (running tonight, `scripts/night_expA.sh` + `night_expA2.sh`, results in `data/expA/`).** The 125B,
  GSM8K-150 (MoEPrefetch's harness, copied read-only), wait-top-2 at 20 % and 10 % residency, (a) with background loads
  of the missing lower ranks (MoEPrefetch's mode) and (b) frugal (no reads for ranks >= 2), against BigMoE's exact eager
  reference from the same harness (147/150; the same-weights noise arm agrees on 147/150 answers). Pass: >= 145/150 and
  >= 94 % identical answers for frugal at 20 %; 10 % tells how far it goes. First probe: 20 % with background loads runs
  21.5 tok/s eager vs exact 13.8 (9.4 vs 37.1 stalled layers/token). ~2.5 h of GPU in the evening gap, before 02:30.
- **Next.** 35B variants (substitute / drop+renormalize / drop+mean / layer split) at 10 % and 5 %; re-trace the 125B
  with all 512 router scores to replay gate-mass budgets and near-tie reranking; teacher-forced KLD and top-1 agreement
  per the quality suite; router fine-tune under the policy only if a gap remains (63M params, ~3-5 h per 10M tokens on
  the Mac, or < 1 h layer-local).

### D2. Q4-equal experts at ~3.25 bits (trellis), nested by gate rank, and a dense diet
- **Mechanism.** Today's affine Q4 g64 spends ~1 bit/weight more than its error needs (NMSE ~0.008 is reachable at
  ~3.5 bits by the rate-distortion bound; a trellis code sits ~0.07 bit above it). Hadamard + LDLQ + a QTIP-style
  bitshift trellis with more bits for down_proj: ~3.25 bits stored; a nested code (2.5-bit base for every use, a 1-bit
  refinement read only for gate ranks 0-1) reads ~2.7. Dense: HC mixers bf16 -> 8-bit, projections and lm_head 3.5-bit
  trellis: 3.4 -> ~2.2 GB.
- **Numbers.** 1.4x fewer bytes per expert read (1.7x with nesting) and 1.4x more experts per GB (residency at 12 GB:
  7.8 % -> ~15 %); whole model ~49 GB (the in-RAM goal needs ~2.75 bits, beyond Q4-equal). Trellis decode costs <= 4
  ALU ops/weight: ~125 GB/s of compressed weights on an M2 Max (ALU-bound), >= 8x the SSD, so free for SSD-bound goals.
- **Quality risk.** Medium: no MoE result at <= 2.25 bits is Q4-equal, trained or not; 3-3.5 bits is where published
  trellis/VQ results sit within noise.
- **Prior art.** QTIP (3-bit Llama-2-70B +4.5 % ppl, zero-shot equal to FP16), EXL3, ik_llama.cpp IQ*_KT (trellis with
  Metal kernels), Q-Palette (fractional-bit trellis + allocation), HIGGS (linearity: equal weighted error = equal
  quality), Any-Precision LLM / MatQuant (nested codes), our MSB tiers (GSM8K 427 vs 431). **Left for us:** trellis vs
  Q4 on a 512-expert hybrid MoE (unpublished), a nested trellis read by gate rank, an MLX Metal kernel.
- **First experiment.** Noise-injection sweep on the 35B proxy (forward passes only, a few GPU hours, no quantizer
  needed): add white noise at the NMSE of ideal 3.5 / 3.25 / 3.0 / 2.75 / 2.5-bit codes to one tensor class at a time
  (expert gate/up, expert down, attention, GDN, HC mixers); measure KLD and top-1 agreement against the 8-bit 35B and
  against the 4-bit pack. Pass: experts at NMSE 0.012-0.017 (3.0-3.25 bits) add no more KLD than the 4-bit pack does.
- **Next.** Trellis fake-quant on the 35B (GSM8K-450, KLD, answer agreement), a Metal decode microbenchmark (pass:
  >= 60 GB/s compressed on the M5 Pro), then the 125B. Quantize from bf16, not from the 4-bit pack: the 125B's bf16
  experts are ~250 GB on HF, so shard by shard (~8 h at the home link's ~9 MB/s, within the 100 GB disk floor).

### D3. Long passes: keep many drafted tokens per big-model pass, with a per-pass expert budget
- **Mechanism.** A small same-tokenizer drafter (Qwen3.5-2B/4B are on disk; a block or diffusion drafter later)
  writes ahead. The 125B runs one prefill-like pass over a chunk of K drafted tokens: it catches up its state and judges
  every token (its rank, margin, a judge head). Everything up to the first must-fix token is kept, and the 125B's own
  token replaces that one. Inside the pass, the rows share a budget of U experts per layer. Rollback: truncate the 12
  attention layers' KV, restore the DeltaNet state from a checkpoint and replay its recurrence.
- **Numbers.** RAM: K = 16, U = 26 keeps every row's rank-0 expert and 97 % of ranks 0-1, at ~0.24 GB/token with
  2.5-3-bit weights, the bandwidth for ~1000 tok/s if all 16 are kept (*sim*). SSD: a chunk only beats eager decoding
  if kept/K > L(K)/152 (79 % at K = 16, 28 % at K = 256), so at 10 % residency the ceiling is 39-123 tok/s depending on
  F, the mean gap between must-fix tokens (research/sparse_consult.md).
- **Quality risk.** High: published judges keep 20-45 tokens per pass at about -1 GSM8K point; nobody measures whether
  final answers stay identical.
- **Prior art.** Judge Decoding (ICLR'25, 19.7 kept per pass 8B -> 405B), AutoJudge (20-45 per pass, GSM8K -1), R2R
  (NeurIPS'25, 12.9 % of tokens to the big model), SpecReason, DFlash (6.5 kept per pass, lossless), MoE-Spec / ECOSPEC
  (capped verify experts), Cerebras/B200 and diffusion LMs for the 1000 tok/s class. **Left for us:** chunks >= 16
  under SSD streaming, answer identity, the pass-level expert budget on a 512-expert top-10 hybrid, DeltaNet rollback
  by replay.
- **First experiment (offline, ~1 h of GPU).** The 125B writes ~50 reference answers (GSM8K, MATH, code, knowledge),
  logging its top-20 logits; teacher-force Qwen3.5-2B and 4B over them; for each drafter top-1 log the 125B's rank and
  margin and the drafter's entropy. Report F per acceptance rule (top-1; top-k + margin; judge head) and a stop
  predictor's AUROC. Pass: F >= 100 with AUROC >= 0.9, or F >= 250. Then the pass-level budget's quality: teacher-forced
  KLD/top-1 of K = 16 passes under U = 26.

## Other ideas from round 1

| idea | what it does | prior art | verdict |
|---|---|---|---|
| second-touch admission + a protected long-term set (CDN / codec long-term references) | exact: ranks 3-10 enter the cache only on a second touch | TinyLFU, S3-FIFO; none for experts | trace replay next (CPU, minutes); W-TinyLFU already lost to LRU here (MoonShots r17), so low odds |
| near-tie reranking | a resident expert replaces a missing one only within delta logits | cache-conditional experts | part of D1; needs all 512 router scores in the traces |
| prompt working set pinned + ranks 1-2 streamed | pin the prefill's hot experts | PreMoE, EASY-EP, SwapMoE | a static set loses to LRU on our traces (MoEPrefetch); maybe as a hybrid inside D1 |
| shared expert bases (MoBE) | -24-30 % expert params near-lossless on 235B-1T MoEs | MoBE (2508.05257) | later: never combined with quantization; a miss rebuilds from bases (compute) |
| expert pruning (REAP) | -25 % experts | REAP; REAP-288 on this model: HumanEval 93.9 -> 91.5 | beyond the quality bar |
| ASTC texture-decoded weights | free decode in texture hardware | Apple 2025 report | no better than 3-bit affine error; only if ALU-bound |
| catch up with resident experts only, full experts only at flagged tokens | the aggressive D3 variant (~152/F loads per token) | BiLD, R2R | the only candidate for 6 GB / 4 GB so far; quality risk high (SliceMoE reports a collapse) |

## D1b (2026-10-20 22:15): similarity substitution for frugal (SERE-style), next after the KLD variant probe
Frugal skips ~29 % of the top-10 gate mass at 10 % (flat routing; eviction and gate thresholds cannot recover it
cheaply: SCOREBOARD). SERE shows similar-expert re-routing turns a collapse (Qwen3-30B top-2 14.3) into 80.4 of 82.2.
Build output-similarity tables for the 125B; frugal substitutes each missing rank with its nearest resident expert at
the original gate weight; measure teacher-forced KLD (tf_frugal_var.py, new variant) on ifeval+multi vs base 0.115/0.092.
Cost at runtime: zero extra SSD bytes (one table lookup per miss).

## D1c (2026-10-21 01:25): confidence-gated coverage (BiLD-style fallback) -- the leading quality fix
Evidence: Eval suite tfstep data: frugal's own entropy tracks the reference's (r 0.956); re-running the most uncertain
25 % of tokens exactly would give KLD ~0.0098 (= ar4), 37 % -> 0.0029; confident tokens show 0 top-1 flips. Trigger:
entropy ~ 1 - p_top1 > margin. Real-run probe: tf_frugal_var.py bild_eNN (night chain 6).
Fast-path design (Wed, fork125bt):
- eager: frugal step -> GPU entropy (one scalar, synced with the argmax we already sync) -> if > tau: MTPLX
  snapshot_untrimmable_cache_lazy before the step, rollback_after_verify(verified_tokens=1), MODE exact (every top-10
  waited through the native gate, nw = 10), re-run the step. Est. 10 %: 37.7 ms + 0.25 x ~75 ms -> ~18 tok/s at ~ar4.
- MTP: the frugal verify pass decides acceptance as now; the first accepted row whose frugal entropy > tau ends the
  pass; rows from there are re-verified by one EXACT multi-row pass (the drafts after it ride along), so exact passes
  amortise like DraftExpert's verifier but run only on uncertain stretches. Commitment weighting (AcceptMoE) for the
  exact pass's later rows.
- Bytes: frugal ~20 loads/token + 0.25 x ~130 extra per re-run ~ 53 loads (~145 MB/token at 10 %), vs 111 for
  wait-to-85 % at the same KLD. Overlap of loads and compute (BigMoE/MoEPrefetch lane) applies to both passes.
- Layer-band probes may allow partial re-runs (only the sensitive layers exact).

## D1d (2026-10-21 10:15): multi-row waits at small residency need a per-pass cap (found by sp_m85spec05)
At 5 % (C=26) a 4-row MTP verify pass under wait-to-85 % can need more distinct waited experts than slots (replay: mean
12.1, P90 20, max > 26 per layer-pass) -> "bigmoe: 29 experts > C=26" abort. Fix on the GPU, no CPU sync: priority of
each (row, rank) = gate weight x commitment probability of the row (row 0 = 1, row j ~ a^j with the measured per-row
acceptance; AcceptMoE 2608.02989's principle), flag the top (C - 4) waited entries only (others fall back to resident
or dropped). Needed for the 16 GB-Mac configuration (~5 %); eager decoding at 5 % is fine as is (<= 10 waited).
Also measured this morning: frugal eager at 10 % 25.1 tok/s code (29.5 reads/token, 82 MB) / 26.8 prose (25.4, 70 MB).

## D1e (2026-10-21 11:30): port frugal onto BigMoE's 3-bit tier engine (after its validation settles; copy, never edit)
BigMoE HEAD (e48717e) decode on tier layers: block sorts the top-10 by score -> switch_mlp(x, inds) -> tier.tier_call:
route(L, ids) for the MSB-3 base pool (every rank), route(side0 + L, ids[ranks 0-1]) for the LSB-1 sidecar, then
MoonShots' fused 3/4-bit GLU (3c) with the gate-weighted combine inside (BIGMOE_TIER_COMBINE=1, scores passed in).
Frugal on it: (1) the ext's selective / flag waits with protect-what-fits (fork125bt/bf patches) on the BASE pool only;
the sidecar keeps waiting for ranks 0-1; (2) missing base slots -> a dummy resident slot + score 0 before the fused
combine (no renormalisation); (3) GPU wait flags from bt_mass.py (mass rule) or nw = 2 (frugal). Gains vs fork125bt:
MSB-3 residency (1.29x experts per GB), records3 single aligned reads, 3c (+21-23 % per BigMoE B5).
D1e refinement (15:05, after the official verdict): the gate is per part (KLD <= 1.10 x ar4) and the 3-bit tier already
spends much of the easy-text budget (plan4433333333 tf: code 0.0017 of 0.0025, gsm 0.0028 of 0.0045), so the
gate-passing coverage policy (m85, no renorm: code 0.0008, gsm 0.0026 at 4 bits) must run on BigMoE's tier engine with the
UNIFORM 4-bit plan (BIGMOE_TIER=4444444444: exact numerics through the MSB-3 + LSB-1 plane pools and 3c, BigMoE B6), not
on 3-bit tiers. What the port buys: OoO (resident experts computed during the loader spin; BigMoE af1af44, +3-8 % exact at
24 GB, more at 10 % where waits are 3x larger), 3c (+21-23 %), records3 single aligned reads. Needed in a copy of their
engine: flag waits + protect-what-fits in their ext (the base and LSB pools), missing ranks as score 0 in the fused
combine. Prefer after BigMoE's b9 (OoO verification) settles; tell BigMoE first.

### D1f (17:00 Wed): one read per expert under deferral (refines D1e)
fork125 reads every expert as 9 piece reads (no contiguous records; BigMoE deleted records/ for records3, 67 GB, and the
disk has 119 GB free, so rebuilding 68 GB of 4-bit records would cross the ~100 GB floor). Deferral turns stalls into SSD
throughput: dh_w4 keeps exact's ~150-170 reads/token, so its speed is set by how fast those reads stream (QD and request
size), not by latency. records3 gives one aligned read per expert (MSB-3 base + LSB sidecar) on BigMoE's tier engine,
which also has the 3c fused GLU (+21-23 %) and OoO. If the prototype lands below ~17 tok/s (reads serialised), port the
deferral (phase A / phase B release + late HC write) onto BigMoE's engine with the uniform 4-bit plan, after telling
BigMoE; if it lands near 20+, the port still adds the kernels and the record reads.

### D1g (17:45 Wed): frugal on BigMoE's v3 tier engine (the 12 GB fit; supersedes D1e's "uniform 4-bit" idea)
Survey (Explore agent, BigMoE HEAD 7b8c7b4 "V3 validated: GSM8K 450/472"): records3 = per expert an MSB-3 part (2.16 MB)
+ LSB-1 part (0.77 MB), 16 KB aligned, one preadv per part; base pool (MSB-3, ext layer L) + LSB sidecar (ext layer
side0+L) with their own LRUs; plan 4433333333 keeps every routed expert in the base pool and LSB only for ranks 0-1;
3c fused GLU (MoonShots tierglu.tier_glu) takes any slots + scores in rank order; OoO (tier.py _ooo_call) already does
phase 1 = resident pairs from the posted residency table, spin, phase 2 = the rest, combine_phase(scores). Missing:
selective waiting (release is all-or-nothing per layer). Port, in a copy (~/Breakthrough/v3f, after BigMoE/MoEPrefetch
answer): (1) ext: flag mode on the base and sidecar rows (bit 20 = load + wait, unflagged = protected if resident, never
read; optionally bit 21 = phase-B late); make_room already protects posted ids, so post the whole top-20. (2) block:
post top-20 to the base pool (ranks 0-1 flagged) + ranks 0-1 to the sidecar (flagged); frugal selection on the GPU (first
10 resident in rank order, renormalised); tier_glu / OoO phases with the selected slots, ranks 0-1 at 4 bits. RAM at
10 %: base 51 x 48 x 2.15 MB = 5.26 GB + sidecar ~20 x 48 x 0.77 MB = 0.74 GB (vs 6.77 GB of 4-bit pools) -> with a
diet floor ~5.5 GB, ~11.5 GB total. Quality: frugal (qualifies at 10 %) x MSB-3 on ranks 2-9 (v3's own GSM8K 450/472):
decode-check the combination (Eval suite) before any claim.

### D1h (01:45 Thu): prefetch done right (NewIdeas I8, ~/NewIdeas/scripts/stage_time.py), after the 12 GB trim
Why 18:10's prefetch lost: NVMe has no priority, so prefetches issued with a layer's demand reads share bandwidth and slow
the waited reads (model: -6 % on MTP passes at C=51), and spare slots came out of C. Fix: (1) DEMAND FIRST: issue layer
l+1's prefetches only after layer l's demand reads land; (2) a GLOBAL staging ring (8-16 slots, ~40 MB), promote-by-copy
into the LRU victim when a staged expert is routed at ranks 0-1, computed free at ranks 2-9; (3) corrected d=1 forecast
(corr125), k=2 per row. Model, fast engine, 4-row MTP: C=51 57.8 -> 63.6 (plain d=1) -> 67.8 (recall 0.69) tok/s; oracle
81.7. A/B on the v3 copy (B12 + MTP code, C=51): arm 1 = forecast posted on L's prediction lines, PF_JIT off (v3's non-JIT
loader prefetches L+1 after L's synchronous demand ensure + release), spares from C; arm 2 = global ring (new loader
feature). v3 caveat: a prefetch lands in the MSB base pool only; ranks 0-1 still need the LSB part (0.77 MB read) unless a
sidecar prediction line prefetches it too. Kill if 'after' + ring < +5 % on MTP code.

### D1i (02:05 Thu): sub-batched exact prefill (NewIdeas, ~/NewIdeas/scripts/prefill_union.py)
v3's prefill_call reads every expert a chunk uses from records3 into one scratch per layer per chunk (cache ignored):
distinct experts per layer per chunk 125 (64 tokens) / 221 (256) / 331 (1024) / 356 (1535) of 512, so the scratch peaks at
~1.05 GB per layer call at 2048 tokens and chunking smaller trades RAM for prefill speed (~585 -> ~127 tok/s est.). Fix
keeping big chunks: fill + run the scratch in expert sub-batches (64-128 experts, 0.2-0.35 GB), accumulate the gate-
weighted outputs (sum order changes: not bitwise vs the reference, fine for the task bar); skip experts already resident
(~13 % fewer reads at 10 %). After the B12 decode verdict.
D1h VERDICT (04:10 Thu): FALSIFIED for the eager v3 engine and SHELVED, arm 2 (global ring) included. Arm 1 measured +0 %
(code) / +5 % (prose) per pass vs the pre-registered +11..14 %, although 125 of 210 prefetches per pass were used, 99 % on
time. NewIdeas' model counted a waited read as pure SSD time (issue, bookkeeping, promotion free); the eager engine's time
is elsewhere (CPU-side per read / per promotion, residency side effects). Rule for MoonShots' new engine (M2): profile
per-layer wait vs compute before any prefetch work.
D1i SCOPE (NewIdeas -> TrainLab): sub-batched exact prefill + overlap reads / rebuild / compute + rebuild on the GPU (MoonShots'
3b kernel): prefill measured 175 tok/s at chunk 2048 vs a ~585 SSD floor; plus, for long context, decouple the attention
query chunk (e.g. 512) from the expert chunk (2048): 37K-token prompt peaked at 18.7 GB.
D1i RESULT (TrainLab 04:20, adopted 04:25 Thu, a45b99a / 4d42a59): grouped exact tier prefill (BIGMOE_TIER_PF_GROUP=64, same
sha, +25 %) + attention decoupled from the MoE chunk (512 / 4096): 2.6K prompt 10.9 GB at 278 tok/s (chunk 256: 11.5 GB
at 50); 32K 12.5 GB at 270 (chunk 2048: 17.6). Not bitwise vs chunked (routed-expert gather shapes). Rejected: GPU
records3 rebuild (slower), pool warming from the last chunk (waited misses 0-16 +10 %). Streaming (TrainLab 04:36,
D1I_STREAM=3 D1I_RING=6|8): 322-389 tok/s at 11.8-12.2 GB, exact; off at 32K (13.7 GB); NOT in my copy yet (eager-only
opt-in candidate: + ~0.9 GB would put B12 + MTP over 12 GB). The Eval suite's verdict path (MTPLX generate_ar) cannot use
the decoupled prefill: it runs MTPLX_PREFILL_CHUNK_SIZE=512 (= decoupled on prompts <= 512 tokens); Chat1000 runs the
paired long-prompt task check (4-16K, decoupled vs chunk 512 vs chunk 2048).

### D1j (04:50 Thu, proposal for NewIdeas): fewer stall bytes for B12 + MTP
MoonShots 04:xx: an all-hit 4-row pass costs ~38 ms; B12 + MTP passes cost ~85-90 ms, so ~50 ms/pass are stalls (~190-205
waited misses per pass, code / prose). Levers on bytes per waited miss (nothing built; quality first):
(a) waited ranks 0-1 at MSB-3 on a MISS only (a hit with a resident LSB part stays 4-bit): -26 % bytes per miss, and the
    LSB sidecar (C_l 20 = 0.74 GB) could shrink and pay ~7 more MSB-3 slots per layer (C_m 51 -> ~58: fewer misses).
    Prior art: HOBBIT (mixed-precision expert offloading, 2024) loads low-precision copies of less critical experts on a
    miss; this is the same move on the top ranks, with the 4-bit result available whenever the LSB part is resident.
(b) LSB sidecar only for rank 0 (rank 1 at MSB-3 always).
Quality probe: tf_frugal_var.py variants on ifeval+multi / gsm+code (hard / easy KLD vs frugal 10 %) before any trace sim.

### D1k (05:12 Thu): small-call prefill is the agentic bottleneck (NewIdeas 05:11)
Kamen's agentic calls bring a median 150 new tokens (p75 437); prefill at that size costs ~3-5 s cold (53 tokens 3.2 s,
326 ~5 s) vs ~4.4 s of decode per median call, i.e. ~40-50 % of a turn. Facts (code): v3's tier prefill reads every expert a
chunk uses with F_NOCACHE (no page cache) and never consults the decode pools (re-reads residents); per layer it syncs on the
router, reads + CPU-rebuilds (GCD-parallel), queues the MoE: 48 serial steps, no read/compute overlap below the streaming
threshold. Levers, exact first: (a) predicted-union look-ahead reads (layer L+1's router or a corrected predictor on layer
L's MoE input; read the predicted union while L computes, then only the stragglers) = pre-routing for READS only, outputs
exact; (b) skip both-parts-resident experts (<= C_l per layer; MSB-only residents still need the LSB part): ~15-25 % fewer
bytes (guess); (c) GPU rebuild for small calls (BIGMOE_TIER_PF_GPU=1: slower at 2.6K per TrainLab, unmeasured small);
then non-exact: (d) frugal-routed prefill for short tool outputs (NewIdeas; needs an agentic quality check).
Breakdown rows (cold / repeat, t_sync / t_moe, CPU vs GPU rebuild): day_ctx2.sh pfb_*.

### D1l (06:5x Thu): frugal PREFILL of short segments (NewIdeas B5c item 3) - prior art (search agent; citations NOT verified by me)
Built as an opt-in emulation (6ec0ba2, 5c4f224): B12_FRUGAL_PF=W, b12_generate(frugal=W, frugal_from=i), B12_FRUGAL_PF_NORM.
Reported prior art: Cache-Conditional Experts (Qualcomm, arXiv 2412.00099, TMLR 2025: boost cached experts' router scores,
keep top-J; misses -50 % for +0.1-3 % ppl); SwapMoE (2308.15030, ACL 2024: route only to RAM-resident "virtual experts");
EdgeXpert (2608.05303, MICRO 2026: prefill, unimportant tokens get fewer slots, top-1 kept, rest prefer experts the prompt
already used; prefill expert traffic -56 %, MC -1..-3 pt); OEA (2511.02237, ICML 2026: decode, top-k0 + already-active,
renormalised; Qwen3-30B k0=3 AIME 80.0 vs 80.4); ExFold (2608.24938: prefill top-4 of 8 costs 2.4-5.6 avg pts, folding
recovers); Wang (2610.22454: halving prefill experts costs 1.4-5.8 on LongBench, mostly answer FORM at answer-boundary /
instruction tokens; full routing on ~4 % of tokens recovers to within 1 F1); Chen & Yao (2609.04575, Qwen3.5-397B, 512 experts
k=10: RENORMALISING over 5 used costs MMLU 2.1 / GSM8K 4.2, normalising by the top-10 MASS 0.55 / 0.8 n.s.); Lynx
(2411.08982: prefill reassignment degrades more than decode, qualitative); HOBBIT (2411.01433). Verdict (agent): partly
covered; no paper combines a shared resident set + guaranteed top-2 + incremental tool-output prefill + SSD streaming.
Evaluation to copy (Wang): reduce prompt tokens only, decode identical, paired bootstrap CIs, score content vs format /
length separately, test by token position; add BFCL (tool calls) and a top-10-mass normalisation arm.
Also for DECODE (new lever, untested): B12's w2f renormalises over the 10 used (substitutes from ranks 10-19); a "mass"
arm (each used expert's exact probability / the true top-10 mass) is a cheap tf_frugal_var probe.

### D1m (13:04 Thu): multi-row phase B for B8 + MTP (NewIdeas offer; MoonShots ~/MoonShots/defer, bit-exact on BigMoE v3)
Today B12's block runs the late MSB-3 ranks (B8: B12_LATE3=2,4) only in 1-row decode; in MTP verify passes (S > 1)
FL_MR forces them to be WAITED, so B8 + MTP pays the cold reads in every pass. MoonShots' v3d makes phase B multi-row:
ext BIGMOE_DEFER_FULL=1 publishes every position's slot in phase B (a second slot vector; S x late ranks positions can
exceed the 46 pairs a 3-line region holds), and tier.py _dh_call / install_dh does phase-1 GLU (resident) -> spin A ->
phase-2 GLU (released) -> combine(which=0); after layer L+1's attention: spin B (full slots) -> phase-3 GLU over the late
pairs -> combine(which=1) -> HC write with layer L's inject. Port plan (next session; the v3f ext must NOT be rebuilt
while any process has it loaded, e.g. the Eval suite):
1. ext: copy MoonShots' DEFER_FULL change into v3f/ext (diff vs 7b8c7b4 in ~/MoonShots/defer/v3d), rebuild when free.
2. bt_b12 _install: for S > 1 with CFG["l3"], post late ranks with bit 21 (not FL_MR's forced wait), keep phase-A slots,
   and route the late pairs through a multi-row late3_fn (spin B full slots -> TG plan "3"*Mp over S rows, scores = the
   late ranks' weights) written by layer_call after L+1's attention (STASH / PENDH keyed by layer, S rows).
3. Check: bit-exact vs the 1-row path on S=1; B8 + MTP code / prose tok/s vs B8-fast + MTP (32.6 code); B8 KLD on
   tf_b12 (late ranks in verify rows change numerics only by timing, as in 1-row B8). Kill if B8 + MTP < +5 % per pass.
D1l VERDICT (13:05 Thu): KILLED (recommended to NewIdeas) by Chat1000's 07:42 evidence (REPORT.md section 10): the same routing
family (last tool result as 1-row B12 frugal steps, 9 real turns) gave 2/9 identical to exact vs the 1.0078 floor's 4/9,
with broken tool calls (an Edit old_string merged two lines; another bash command; a text answer turned into a Bash call).
Mechanism: agentic turns COPY exact strings out of tool results; frugal prefill perturbs those tokens' KV. Mass weights /
4-bit emulation do not address it; prior art's boundary-token fix targets answer form, not copying. Small-call prefill stays
exact (floor = union bytes at ~13 GB/s). Code kept opt-in (B12_FRUGAL_PF) for the record.

### D1n (14:08 Thu): buying IFEval back after B12's FAIL (next session; do NOT edit bt_b12's block during the suite's runs)
Diagnosis (HANDOVER STATE): v3's 3-bit lower ranks are the largest measured quality delta vs yesterday's passing-on-decode
w2f_r10 (+22 % hard KLD on the same positions). Candidates, all measurable with tf_b12 in ~2 min each before any verdict:
(a) 4-bit pools ("B12-4"): plan 4444444444. Needs B12's block to post the LSB sidecar for every USED rank (today: ranks
    0-1 only, sl [S, 2]) and a residency rule: a rank counts as resident only if BOTH parts are (else substitute), or
    HOBBIT-style compute it at 3 bits when only the MSB part is resident. RAM: C_m 51 all-4-bit ~ 7.2 GB pools (~12.4 GB
    with MTP, over 12); C_m ~46 ~ 11.8 GB. Cheapest first test: plan 4444444444 with C_l = C_m (sidecar = pool).
(b) Cold ranks computed late (B12_LATE3=2,4 / 2,10): exact routing for ranks 2..B-1 with a timing shift; KLD near the dh
    numbers; speed needs multi-row phase B (D1m) under MTP.
(c) B8-mass (5 % + late ranks 2-3): ~25 % below B12 already; its verdict decides whether (a)/(b) are needed at all.
Kill line: a candidate goes to a decode verdict only if its tf_b12 hard KLD is <= ~0.09 (w2f_r10's level adjusted for
mass weights) and it fits 12 GB at 2.6K with MTP.
D1m VERDICT (22:21 Thu, MoonShots ~/MoonShots/data/b8mr/ab.txt): DEAD. Deferring late ranks 2-3 inside MTP verify rows (multi-row
phase B) makes verify passes SLOWER: code +16.5 % per pass at 5 % (27.4 -> 22.8 tok/s), +16.4 % at 10 % (32.0 -> 26.8);
prose +1 % / +10 %. Waited reads fall 17-27 % but total reads rise 4-17 % and stalled spins ~91 -> ~130 per pass: at S=4 the
late reads are not hidden behind one attention block and churn a 26 / 51-slot cache that a 4-row pass (80 candidate ids)
already oversubscribes; S=4 KLD +9-35 % vs wait-all. KEEP FL_MR (wait-all) for S > 1. MoonShots' wait-all arm at 10 %
(B12 + late ranks 2-3 + MTP): code 32.0, prose 22.9 tok/s. My replay fix 1a5eee1 held (12 MTP runs, no crash, identical text).
Hazard noted (design only, unobserved): with C_m < 20 x S a phase-B read could evict a slot phase A still reads.
