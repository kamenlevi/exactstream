# A1 "greedy matching is the hidden cap": KILLED (Leap, Fri 2026-10-23 14:30, CPU only, no GPU)

**Verdict.** Lossless speculative sampling does not raise acceptance on this 125B. The prose / agentic drafter ceiling
comes from how closely the drafter matches the target's distribution, not from greedy matching. Every drafter from a
numeric twin to a 5 % near-copy gains at most +0.003 to +0.014 per token on IFEval at Qwen's official sampling settings.
The kill line was +0.02.

**Data.** The Eval suite's stored top-32 / top-256 distributions at the same teacher-forced positions: the 125B's own
greedy continuations, tf_gen (34,457 step-path positions; 296,597 prefill-path).
- Target p: the exact 125B (reference cache, or the reference step path).
- Drafter q, one of:
  - the numeric twin: the same model on the other kernel path;
  - AutoRound int4;
  - near-copies with 10 % / 5 % of experts resident (w2f_r10 / w2f_r05).
- Acceptance compared per position:
  - greedy: argmax q == argmax p;
  - lossless sampling: sum_x min(p'(x), q'(x)), where p' and q' get the temperature T, then top-k, then top-p.
- tau16 is simulated along each sequence (DP), the same unit as Chat1000's.
- Scripts: scripts/a1_ceiling.py, a1_lowT.py, a1_tolerance.py, a1_mtp_smoke.py. Outputs: data/a1_*.txt.

**IFEval** (the most fork-heavy class: 35 % of positions have p1 < 0.7), per token, then tau16:

| drafter | greedy | T1 k20 p0.95 (official thinking) | T0.7 k20 p0.8 (official instruct) | T1 full | best low T (0.05-0.5) |
|---|---|---|---|---|---|
| numeric twin | 0.974 / 12.8 | 0.978 / 13.1 | 0.976 / 12.9 | 0.968 / 12.2 | 0.976 / 13.0 |
| AutoRound int4 | 0.947 / 10.6 | 0.952 / 11.0 | 0.949 / 10.7 | 0.943 / 10.3 | 0.950 / 10.8 |
| near-copy 10 % | 0.872 / 6.64 | 0.881 / 6.98 | 0.874 / 6.73 | 0.871 / 6.61 | 0.875 / 6.77 |
| near-copy 5 % | 0.829 / 5.32 | 0.842 / 5.71 | 0.835 / 5.50 | 0.832 / 5.43 | 0.835 / 5.50 |

- Code and GSM: within +-0.001 of greedy for every drafter. Multilingual: +0.000 to +0.004.
- Why: sampling gains at forks (near-copy 10 %: 0.66 -> 0.75) but pays the drafter's whole-distribution mismatch at
  confident positions (0.987 -> 0.951). The two cancel.
- Sweeping T from 0.05 to 1.0 never moves acceptance more than about 0.01 from greedy agreement.
- Consistent with the literature: EAGLE-type drafters accept less at T=1 than at T=0. For example, EAGLE-3 on
  Llama-3.3-70B: MT-Bench 4.01 -> 3.76, HumanEval 5.18 -> 4.86 (LK Losses, 2602.23881). Leviathan et al. report gaps of
  0.1-0.2.

**Agentic.** No distributions are stored, so this is not measured directly. The near-copy's agentic misses are less
concentrated at forks (58 % vs 87 %), so the agentic gain should be at most the IFEval gain.

**MTP head.** Not run on IFEval / agentic: turn17 failed after 1 sequence. Its 105-position GSM smoke sample gives
greedy 0.943 vs sampling 0.937. This no longer decides anything for 350: even a twin-quality drafter is capped. When
turn17 reruns, a1_mtp_smoke.py reads its exact__tgt_top_* / exact__mtp_top_* arrays on CPU.

**The tie part of the greedy cap, measured.**
- The target's logits are bf16, so near |logit| 16-32 the step is 0.125 nats.
- Exact top-2 ties: 2.0 % of IFEval positions, 1.4 % multilingual, 1.2 % docs, 0.3-0.4 % code / GSM. At these, the
  argmax index rule decides the token, not the model.
- The twin's flips sit 1-3 bf16 steps apart (gaps 0.125-0.375 nats).
- Tolerances below 1 step add nothing. Tolerance 0.3 nats (about 2 steps) is Chat1000's floor-tolerance rule, which
  was already killed on answer changes.
- Small new EXACT lever (untested): a drafter-aware tie-break. Among exactly tied tokens, pick the one the MTP head
  (depth 1, from the true hidden state) ranks first, and do the same in plain decoding, so spec == plain still holds.
  - Worth at most about +0.7 % per token / +3.5 % tau16 on prose, ~+1 % on code.
  - It changes the reference shas at tie positions, so adopting it is Kamen's call.
  - Owner if wanted: SpecDecode / BigMoE.

**Implication for 350.** The per-token ceiling for ANY drafter whose numerics differ from the target is the twin:
IFEval 0.974-0.978 / tau16 ~13, agentic 0.959 (Chat1000) / tau16 ~11, whatever the acceptance rule. Agentic 350 still
needs A >= ~8 at 16-row passes <= ~31 ms. That means a drafter between the near-copy (6.8) and the twin (~11), which no
known mechanism produces.
