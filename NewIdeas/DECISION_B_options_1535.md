# Decision for Kamen: how to reach 100-150 tok/s in 6-12 GB (NewIdeas, chief researcher; draft 2026-10-21 15:35)

## The physics, in one paragraph
Per token the 125B consults 480 expert slots (10 of 512 per layer x 48 layers), 2.77 MB each at 4 bits (2.15 at 3).
In 6-12 GB only ~10-19 % of experts fit, so a Q4-faithful token needs ~110 expert reads from the SSD (the per-part
fidelity gate passes only when ~85 % of each layer's router weight is computed). That is 245-305 MB per token; the SSD
gives 14.6 GB/s, so the ceiling is ~48-60 tok/s even if every read were hidden behind compute. Everything we measured
says the experts can't be made cheaper by structure (no low-rank, no neuron sparsity, no substitutes: their outputs
are mutually orthogonal), only by bits. The stalls themselves can be engineered away (our simulation on real 125B
traces: waited reads 153 -> 6 per token), and then bandwidth is the wall.

## Three options (all keep the main track exact; this is only about the small-RAM line)
| option | what it means | expected on the M5 Pro | Mac mini (3.2 GB/s) | cost / risk |
|---|---|---|---|---|
| 1. Keep the formal gate (KLD <= 1.1x the int4 build, per corpus part) | inference-only levers: 3-bit cache + dense diet (residency ~19 % at 12 GB), near-tie re-routing (-29 % reads, but probably too costly for the gate: probe pending), trellis-coded experts (Q4-equal error at ~3.25 bits: -28 % bytes; needs the bf16 weights, ~8 h download, and a Metal decoder; may pass the gate like the other int4 build), exact prefetch with trained estimators, ~2x engine | ~28-30 tok/s with MTP on today's engine (Breakthrough's estimate at C ~60-75 with b25); ~60-90 if the engine gets ~2x closer to its byte roofline (the near-tie re-routing lever is now doubtful: its swaps cost ~2.2x the gate energy per saved read of simply dropping the lowest-weight ranks, so the gate probably prefers the mass rule; probe pending); +35-70 % with one or two Thunderbolt-5 NVMe enclosures (measured 5-6.5 GB/s sustained reads each on M4 Pro Macs; Eclectic Light, TerraMaster), experts split across devices, once loading is stall-free (~+15 % for today's exact mode) | ~10-20 tok/s | no money (or ~EUR 250 for the SSD); weeks of engine work; each lever must pass the gate |
| 2. Judge by task results ("similar intelligence", your words) instead of per-token fidelity | frugal-class routing (wait for the top-2 experts, skip cold lower ranks) + prefetch + half-layer-late misses + kernels | ~100 tok/s at ~14 GB (Breakthrough's speed path) | ~40-55 tok/s with the dense diet | frugal today: math 90 vs 86, IFEval -2.0, HumanEval -1.7 pass, 91-92 % identical answers (same-weights floor 97-98 %); formal gate FAILS |
| 3. Train a small-RAM variant of the same model | distil the 125B to use ~6 experts instead of 10, prefer recently used experts, and choose the next layer's experts one layer early (published recipes: ZEDA, Edge0, cache-aware router training) | ~100-150 tok/s at ~12 GB (reads ~46/token, all hidden) | ~30 tok/s | rented GPUs (cost estimate below), published losses ~0.7-4 points average; a different model by the formal gate's definition |

Rented-GPU cost for option 3 (research/training_cost_125b.md, estimates from provider prices read 10-21; nothing run or
bought): the model is 125B with 6B active (dense 3.65B + routed 2.36B + head 0.64B). Stage 0, a dry run on the 35B
(~$30); stage 1 go/no-go, 3e8 tokens of self-distillation on one 8xH200/B200 node, ~20 node-hours: **$320-960** (go if
>= 75 % of the zero-shot gap closes and benchmarks stay within ~1 point; FarSkip's half-depth conversion lost 0.6 at this
budget); stage 2 to 3e9 tokens + on-policy: $1.6-4.8k; stage 3 (1e10): $3-11.5k. Whole program $5-7.5k on spot, $10-17k
on-demand. Tooling: ms-swift (+mcore-bridge) is the only stack with this model, expert parallelism, LoRA, forward/reverse
KL and on-policy rollouts. Risks: two-month-old software, 4-bit quantisation can erase the healing (keep trained dense
tensors at 8-bit or the LoRA unmerged), the MTP head needs a re-fit, and GSM8K-type collapses can hide behind good
perplexity.

## What I recommend
- Decide the bar for the small-RAM line first: it is worth more than any single mechanism (2-3x).
- Whatever the bar, build the stall-free loader (prefetch + near-tie + residency levers) and the kernel work: they help
  every option. Options 2 and 3 are compute-bound on today's engine (~30 ms of GPU per token eager, ~60-80 ms per 4-row
  MTP pass = ~40-50 tok/s): their 100-150 tok/s needs the engine ~2x closer to its byte floor (MoonShots' speed-of-light
  layer measures how much is possible).
- 1,000 tok/s for one chat is out of reach for this model on this Mac by bandwidth physics (even perfect copy-drafting
  of code edits caps ~640 tok/s; ~300-350 tok/s aggregate over many chats); only a trained parallel decoder changes that.
