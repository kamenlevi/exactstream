# RESULT, Breakthrough round 1 (2026-10-19, ~22:20)

Files (~/Breakthrough, commits 9f5db8c and 34a5350): `SCOREBOARD.md` (budgets), `IDEAS.md` (directions),
`research/{bits_limits,routing_locality,sparse_consult}.md` (prior-art briefs, 3 helpers), `scripts/` + `data/`
(trace replays, experiment A runners).

## Budget (SCOREBOARD.md)
- Today a token costs **3.1 GB of RAM traffic** with MTP (a 4-row verify pass reads 8.6 GB: 3.4 GB dense, ~26 experts
  per layer = 3.45 GB, 1.46 GB of draft lm_head reads; 2.8 tokens per pass). The in-RAM one-stream ceiling at 4-bit is
  ~80-120 tok/s, not 150-250, and the model (73 GB) doesn't fit in 64 GB.
- RAM floor with zero experts cached: dense 3.4 + MTP head 1.7 + runtime ~1.5 = **6.6 GB**. At 12 GB only ~5.4 GB
  (7.8 % of experts) is left for the cache; the SSD then supplies ~485 MB/token (*sim*), ~11-14 tok/s projected.
- Needed vs today: 12 GB lower bar 2.1-2.4x fewer SSD bytes; Kamen's Mac 100-150 tok/s 4.7-7x (12 GB) / 13-20x (6 GB);
  M6 mini 50-75 tok/s 11-16x SSD and 1.3-1.9x RAM bytes; ideal 400-600 19-29x SSD and 5-7x RAM; 4 GB ultra-light
  28-47x; one-stream 1000 tok/s 12x fewer RAM bytes plus the whole model in ~45 GB (~2.8 bits/weight).
- Corrections to the brief: LRU hits ~85 % at 25 % on this model (73 % is a Qwen3-30B number); verify passes don't
  share expert bytes (4 rows need ~2.6x one row's experts).

## What the routing traces say (125B, 2,926 tokens, *sim*)
- Exact levers are capped: Belady gives only 1.66x fewer misses than LRU at 10 % residency; K tokens per pass give
  1.06x at K = 4, 1.9x at K = 64. Routing is diverse: 256 tokens touch ~47 % of each layer's experts.
- Wait-top-2 (wait only for ranks 0-1, substitute/drop the rest) cuts SSD loads 152 -> 20 at 10 % residency (7.5x),
  more than any published routing change (routing brief). Quality there is unmeasured.
- Fewer bits buy residency: at 12 GB a Q4-equal ~3.25-bit trellis code plus a dense diet gives ~15 % residency, and
  wait-top-2 needs 30 MB/token (234 MB exact). Then the RAM term caps it at ~60 tok/s on a 120 GB/s mini and ~125 on
  Kamen's Mac: the small-target band on paper, if quality holds. 6 GB / 4 GB: residency <= 2.5 %, below a 4-token
  working set; nothing reaches them yet.
- Long passes with a per-pass expert budget (K = 16 rows, 26 experts/layer) keep every row's rank-0 expert and 97 % of
  ranks 0-1 at ~0.24 GB/token: the bandwidth for one-stream 1000 tok/s, if ~16 drafted tokens are kept per pass.

## Three directions and first experiments (IDEAS.md)
1. **D1 Budgeted routing**: each token waits only for its top experts (rank or gate-mass x layer budget), the rest
   substituted, dropped or replaced by a mean-output stand-in; router trained under the policy only if needed.
   Prior art: HOBBIT, "Which expert misses are worth waiting for?" (Oct 2026), cache-conditional experts, 2609.04895,
   ReMoE, MoNE. Open: reasoning-equal quality at <= 15 % residency on 512 experts.
   **First experiment running now** (GPU, evening gap, no job after 02:30, results in data/expA/summary.txt ~01:30):
   125B GSM8K-150 at 20 % and 10 % residency, wait-top-2 with and without background loads, vs BigMoE's exact eager
   reference (147/150; same-weights noise arm 147/150 identical). Pass: >= 145/150 and >= 94 % identical answers
   (frugal, 20 %). First probe at 20 %: 21.5 tok/s eager vs exact 13.8 (9.4 vs 37.1 stalled layers/token).
2. **D2 Q4-equal experts at ~3.25 bits (trellis), nested by gate rank (~2.7 read), dense 3.4 -> ~2.2 GB**. Today's
   affine Q4 spends ~1 bit/weight more than its error needs. Prior art: QTIP, EXL3, ik_llama.cpp *_KT (Metal),
   Q-Palette, HIGGS. Open: trellis vs Q4 on a 512-expert hybrid MoE; a nested trellis; an MLX kernel.
   **First experiment**: noise-injection sweep on the 35B proxy (forward only, a few GPU hours): white noise at the
   NMSE of ideal 3.5-2.5-bit codes per tensor class, KLD/top-1 vs the 8-bit 35B and vs the 4-bit pack. Pass: experts
   at 3.0-3.25-bit noise add no more KLD than the pack.
3. **D3 Long passes**: a same-tokenizer drafter (Qwen3.5-2B/4B on disk) writes ahead; one 125B pass catches up and
   judges a chunk; keep up to the first must-fix token; rows share a per-pass expert budget. Prior art: Judge
   Decoding, AutoJudge, R2R, SpecReason, DFlash, MoE-Spec. Open: answer identity, chunks under SSD streaming, the
   per-pass budget on a 512-expert hybrid. **First experiment** (offline, ~1 h GPU): the 125B writes ~50 reference
   answers logging top-20 logits; teacher-force the drafters; measure F (gap between must-fix tokens) per acceptance
   rule and a stop predictor's AUROC. Pass: F >= 100 with AUROC >= 0.9, or F >= 250.

## Utilization
- GPU: experiment A runs in the evening gap under queued quiet locks (50-problem chunks, keepalive, 85 % memory gate);
  the main-track runners start at 03:05 and are untouched. SSD in the wait-top-2 probe with background loads: ~111
  reads/token x 2.77 MB x 21.5 tok/s = ~6.6 GB/s (45 % of 14.6); frugal mode would read ~14/token (~0.8 GB/s, 6 %),
  so there compute and dispatch (~46 ms/token eager on the fork's older base) become the limit. Memory GB/s not
  measured in these runs. CPU: trace replays, one core at nice 10, minutes.
- Usage: the three research helpers cost ~0.69M subagent tokens (198k + 226k + 262k); pacer said SLEEP at 22:17 (10 % of the new window).

## Questions for Kamen (from the brief)
1. Does ~1000 tok/s mean one conversation, or the total across parallel streams? One stream needs ~16 kept tokens per
   pass, budgeted routing, ~3-bit weights and a block drafter (nothing published is near). Many streams need the model
   in <= ~45 GB at Q4 quality plus batched serving (plausible on paper).
2. Which Mac mini counts? The cheapest new one is the M6 (16 GB, 256 GB, $899, on sale since Sep 22, SSD ~3.2 GB/s).
   A 512 GB M6 reads ~6.4 GB/s, which halves every SSD factor; the older M4 ($599, ~2.8 GB/s) is about the same as the
   256 GB M6.
