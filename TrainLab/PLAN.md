# TrainLab plan (2026-10-21, started by the orchestrator for NewIdeas)

**Kamen's decisions 17:10 (via the orchestrator):** (1) the small-RAM line is judged by task results (decode tests within
a few points, similar answers), not the per-part KLD gate: frugal at 10 % qualifies, 5 % fails IFEval (-7 pt).
(2) No external GPUs, ever: every training step runs on this Mac. Consequences here: the first healing target is
frugal itself (drop_w2_C13 / C26 = pol35 ldrop): healing it could make 5 % (or less) pass; the 125B plan is a
Mac-local layer-streamed LoRA run (~100-180 tok/s est.: 1e7 tokens ~ 15-30 h of the machine; 1e9 is out of reach), so
a dataflow is a "go" only if the 35B heals within ~1e6-1e7 tokens.

Charter (TASKS.md, "Trainable dataflow"): the SSD-serial factor (GAPS B1, 2.5-3x in small RAM; A1 1.57x at 24 GB) cannot
be removed training-free. Find out whether cheap self-distillation (LoRA on attention / GDN / shared expert / router,
teacher = the unmodified model, KL on any text, routed experts frozen) makes the model tolerate dataflows that give every
SSD read slack with the true routing. First on the 35B proxy: a KLD-vs-training-tokens curve per dataflow with go/no-go;
for any "go", a 125B plan with its rented-GPU cost (Kamen's call; never an action here).

## Dataflows (train35.py spec)
| spec | what changes | slack it gives an SSD read | zero-shot (35B, defer35) |
|---|---|---|---|
| `late` | the routed MoE output of layer L is added at the end of layer L+1 (ScMoE / FarSkip-like) | a whole layer, every read; MoE off the dependent chain | 0.400 |
| `late_B20-40` | the same, last half of the layers only (FarSkip: far cheaper to heal) | half the reads | turn 1 |
| `half` | ranks >= 1 added after layer L+1's attention, before its router (rank 0 on time) | half a layer for 7 of 8 reads | turn 1 (C26/W2 with only misses late: 0.038) |
| `prew` / `pre` | layer L's experts chosen at layer L-1 by a trainable pre-router (Edge0-like); weights from L's router / the pre-router | a layer, every read, no misprediction | turn 1 |
| `k5`, `k4` (+`n8`) | static top-5 / top-4 of 8 (10 -> 6 on the 125B) | none: -37..-50 % reads | turn 1 |

## Method
- Model, held-out segments and reference: NewIdeas' defer35 harness (frozen prose / GSM8K / code segments, 1365 tokens,
  ref_1365.npz), so zero-shot numbers must reproduce defer35's (verified bitwise on a tiny random model: exact, late,
  late_w1, half, half_w2). Extra held-out: Bulgarian, French, German documents (Eval suite raw docs), a account-1 chat.
- Training text: data/train_tokens.npy, 4.7M tokens (Claude Code transcripts 2.0M, Python 1.5M, multilingual
  Wikipedia 0.6M in 10 languages, research notes 0.2M, Swift 0.2M, GSM8K items outside the held-out 200-399 0.2M).
- Loss: forward KL(teacher || student) on the teacher's top-64 + tail; teacher online (exact flow, LoRA off, Metal GDN).
- Differentiable chunked GDN (gdn_chunk.py; values and gradients equal mlx_lm's sequential reference to ~1e-7), per-layer
  checkpointing (gradients equal to the unchecked ones exactly), LoRA r16 (alpha 32) on GDN in/out projections,
  attention q/k/v/o, shared expert; Adam 1e-4 with 20-step warm-up; pre-routers 2e-5 + KL to the true router (aux 1.0).
- GPU: 45-min quiet-lock turns (run_turn.sh runs turn_job.sh at grant time; checkpoints resume across turns).

## Go / no-go (per dataflow)
- Floor: the 35B's bf16-perturbation floor is KLD 0.010 (any perturbation flips near-ties).
- Gate analogue: the 125B gate is per corpus part, KLD vs the reference <= 1.1 x the KLD of a different int4 build (ar4).
  The 35B analogue is NewIdeas' yardstick (LM Studio's separate 4-bit build vs our reference, per segment,
  ~/NewIdeas/data/defer35/yardstick.txt): **gate-go** = healed KLD <= 1.1 x yardstick on every segment.
- **Go** if the curve reaches <= 0.02 (2x floor) or closes >= 75 % of (zero-shot - floor) within ~10^7 tokens
  (extrapolated on log-tokens); **no-go** if < 50 % of the gap closes by ~3x10^6 tokens with a flattening slope.
- For a go: the 125B plan = tokens needed (scaled by the 35B curve and the literature), where (Mac-only run vs rented
  GPUs, research/training_cost_125b.md), and the speed it buys (GAPS B1/A1).

## 125B top-2 estimator (NewIdeas 19:40) - feasibility so far
- rerank125.py (CPU, 125B decode traces, layers 8-39, leave-one-prompt-out): training-free look-ahead top-2 precision
  0.549; a routing-history re-ranker over its top-32 candidates 0.582 (+0.033); the true top-2 is inside the 32
  candidates 95.5 % of the time (ranking fails, not the candidate set). Next: cap125_est.py capture (5 target layers,
  ~45K tokens) + fit125_est.py (tap / ridge / ridge+history / MLP+history). No-go below ~0.75 with the MLP.
- 21:45-22:30 capture (44.6K tokens, BigMoE streamed C25) + fit125_est.py, held out by document, target layers
  8/16/24/32/39: top-2 precision tap 0.527 | ridge 0.683 | ridge + history 0.689 | MLP + history 0.609 (overfit);
  ridge: rank-0 in top-2 0.646 -> 0.805, true top-2 in top-4 0.695 -> 0.853. Verdict: no-go for >= 0.85 (linear
  saturates ~0.7; history adds ~0.5-3 pt); a 5M-token MLP is the only lever left (prior <= 0.75-0.8).

## Results: the first healing curve (turn 4, 10-21 22:26-22:52)
frugal C13 (drop_w2_C13), LoRA r16 GDN/attn/shared + router, lr 1e-4, 2 x 1024 tokens/step: KLD 0.2017 (0) ->
0.1830 (131K) -> 0.1691 (246K) = 17 % of (zero-shot - floor); fit (KLD - 0.010) ~ tokens^-0.135 -> 0.142 @1M,
0.123 @3M (41 %), 0.107 @10M. The same LoRA: C26 0.168 -> 0.144; exact 0 -> 0.063 (a frugal-specific compensation).
Verdict (pre-registered): no-go at Mac budgets for this deletion-type dataflow, recipe check (lr 3e-5 x accum 4) open.
