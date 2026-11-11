# TrainLab report: trainable dataflows on this Mac (10-22 01:00; MTP head section 03:30)

**Charter** (TASKS.md, NewIdeas 14:50): can cheap self-distillation (LoRA, KL to the unmodified model) make the model
tolerate dataflows that give every SSD read slack, with Kamen's rule that all training runs on this Mac? A dataflow is
a "go" only if the 35B proxy heals within ~1e6-1e7 tokens (a Mac-local 125B run costs ~2 h per 1e6 tokens).

## Verdict: no-go at Mac-scale token budgets
- Zero-shot damage (35B, KLD over prose / GSM8K / code; floor 0.010): frugal 10 % 0.168, frugal 5 % 0.202, half-late
  at C26 0.044, routed MoE one layer late 0.411 (last half of the layers 0.262), pre-routing layers 8-39 0.050,
  top-5 without renormalising 0.086, hard swap caps cap4 / cap2 / cap1 0.043 / 0.154 / 0.332.
- The one healing curve measured end to end (frugal 5 %, LoRA r16 on GDN / attention / shared expert + router,
  lr 1e-4, 2K tokens per step): 0.2017 -> 0.1830 at 131K tokens -> 0.1691 at 246K (17 % of the gap to the floor).
  Fit: (KLD - floor) ~ tokens^-0.135 -> ~0.142 at 1M, ~0.123 at 3M (41 %), ~0.107 at 10M.
- Recipe check (lr 3e-5, 4-window accumulation = 8K tokens per update): 0.1751 at 131K, 0.1651 at 262K, 0.1671 at
  393K, 0.1649 at 438K: a better early level, then a PLATEAU at ~0.165 (19 % of the gap). The slope does not improve
  with a better optimizer; the low-rank correction saturates. Same adapter at frugal 10 %: 0.141 (zero-shot 0.168);
  exact flow: 0.040 (less drift than the lr-1e-4 adapter's 0.063).

## What the measurements say
1. Deletion-type damage heals slowly: the healed frugal 5 % reaches the training-free x50 composition (0.145) only at
   ~1M tokens. This matches the literature's split: perturbations heal cheaply, deletions of computation do not.
2. The adapter transfers across residency: trained at 5 %, it lowers frugal 10 % from 0.168 to 0.144.
3. A healed model is a frugal-only variant: the same LoRA moves the exact flow by 0.063 KLD. It would ship as a
   separate adapter, never merged into the base (merging into int4 also erases LoRA deltas, Edge0).
4. Healing helps exactly the agentic fragility: the copy-heavy tool-log segment (floor 0.55 on its own) drops from
   1.46 to 0.76. Every MoE perturbation hurts copy-heavy text most.
5. 35B routing results do not transfer to the 125B: pre-routed frugal beats frugal on the 35B (0.141 vs 0.168) but
   loses on the 125B (+11-26 %, Breakthrough); cap4's locality gain shrinks from LRU 0.60 -> 0.72 to 0.676 -> 0.712
   (NewIdeas' trace sim). The 125B's routing is flatter; routing ideas must be checked on 125B traces.
6. The 125B's next-layer top-2 cannot be predicted well enough: on 44.6K captured tokens (held out by document), top-2
   precision is 0.527 for the training-free look-ahead, 0.683 with a full-rank ridge correction, 0.689 with routing
   history, 0.609 with a small MLP (overfit). History adds ~0.5-3 points. The routing-prediction line is closed.

## The 125B MTP head fine-tune (exact-safe, NewIdeas' top lever): killed (10-22 03:22)
Run mtp125n_d4: MTPLX's own head modules trained in place (fp32 projections, NativeMoE = MTPLX's plain MoE path with
stop_gradient on the routing indices; equal to MTPLX's output, max diff 0), DeepSeek-style depths 1-4, KL to the trunk's
top-64 per depth, lr 1e-5, 15 Claude Code transcripts from Kamen's sessions (131K positions, disjoint from the test turns).
- Agentic (Chat1000's 12 held-out turns, his agent125.py with the tuned head; greedy output token-identical to his
  stock run, so the comparison is exactly paired): tau16 4.04 -> 3.85 = -4.6 % [90 % -8.4, -0.6]; bash -0.2, edit
  -8.0 [-14.2, -0.7], text -4.7, write -6.2; tau9 -3.9 % [-7.2, -0.4] (agent_compare.py).
- General text (the 125B's own tf_gen continuations, 22 items round-robin over the 4 sets, sim125.py): -7.5 %
  [-9.9, -5.2]; GSM8K -15.3, multilingual -8.2, code -7.4, IFEval -1.7. The 8-bit stored head: -8.8 %.
- Why (step-wise acceptance, ALL agentic, stock -> tuned): depth 1 0.846 -> 0.858 (bash 0.781 -> 0.838; code tf_gen
  0.920 -> 0.969). Depth 1 is where training equals inference exactly, and there the adaptation works. Depths 2-4 went
  0.835 / 0.847 / 0.837 -> 0.813 / 0.827 / 0.827, although their training KL fell ~45 %: DeepSeek-style depth training
  gives depth j a fresh cache of depth-(j-1) outputs, while at inference the recursive head attends to the true-pair
  history plus its own chain. Depths 11-15 were never trained and got worse, which costs edit's long copy runs. Write
  got worse at every depth (Claude's file contents are not the 125B's own text).
- Storage (delta125.py): the trained changes sit far below half a 4-bit step (|d| > step/2 for <= 2 % of the weights),
  so re-storing in the stock 4-bit format would keep only 12-23 % of the change in those layers (33 % overall); 8-bit
  g64 keeps 99.9 % for ~+29 MB. Any future fine-tune of a 4-bit head must ship at 8 bits or in bf16, or train
  quantization-aware.
- Prior art (research/prior_art_mtp_finetune.md): FastMTP (2509.18362) got 1.21x -> 1.81x mainly by teaching recursion
  to a head trained for one step (vanilla acceptance per position 70 / 10 / 0 %), with ~10^8 self-distilled tokens and
  beta-0.6 weighted CE. The 125B head already chains at ~0.85 per step.
- What a working recipe would need (NewIdeas takes it to Kamen as a multi-night local proposal): the EAGLE-3 training-
  time test inside MTPLX's QSA attention (depth-j queries over true-pair keys plus their own chain keys) for all 15
  depths, and the 125B's own agentic continuations as data (~130 s to generate a 160-token turn here).

## Tools left behind (~/TrainLab)
- train35.py: LoRA self-distillation of the 35B under any dataflow (late / half / pre-routing / top-k / cache-dependent
  LRU flows with pol35 semantics / sticky caps / frugal), chunked differentiable GDN (gdn_chunk.py; the power-series
  triangular inverse NaNs on repetitive text, use recursive block inversion; MLX 0.32 gives wrong gradients for a strided
  slice of a concatenated array), per-layer checkpointing, gradient accumulation, teacher cache, NaN guard.
- decode35.py + score_decode.py: batched decode under any flow with the Eval suite's scorers.
- mtp35.py / mtp_onpolicy.py: the 35B MTP head's training-time test and Chat1000's tau protocol (stock tau16 4.88;
  a generic off-policy fine-tune stayed flat; the head is already strong on agentic text).
- mtp125.py (native training through MTPLX's head; --wdecay/--ck-loss/--mix-general/--init), agent125_tl.py,
  agent_compare.py (paired tuned vs stock agentic tau), sim125.py (general-text tau, one prefill for several heads,
  storage variants), delta125.py (what a run changed vs the stock storage).
- rerank125.py, cap125_est.py, fit125_est.py: the routing-prediction feasibility checks.

## D1i: B12 / B8 prefill in small RAM (NewIdeas 03:30, for Breakthrough; 10-22 05:00), ~/TrainLab/d1i/PATCH.md
The exact tier prefill read every expert a 2048-token chunk used into one scratch and computed full-vocab logits per
chunk position. Chunk 2048 was 13.9 GB at 162-172 tok/s on a 2.6K prompt; chunk 256 was 11.5 GB at 50 tok/s.
- Logit-free chunks (-0.7 GB, exact) and grouped expert sub-batches (v3's own _sync_grouped on records3; same sha;
  +25 %): adopted by Breakthrough.
- Decoupled prefill: attention in 256-query chunks, then the MoE once per layer over up to 4,096 tokens. Bit-identical
  to the plain prefill on one-query-chunk prompts (B8 and B12); fluent and GSM8K-20 20/20 through MTP on longer ones.
  2.6K: 10.9 GB at 280-308 tok/s. 32K: 12.5 GB at 270 tok/s (chunk 2048: 17.6-18.7 GB). Adopted by Breakthrough
  (b12_prefill).
- Fixed-order expert streaming (NewIdeas' design): every layer's 512 experts read from records3 through a ring of 6,
  in GIL-free threads, and rebuilt in place on the GPU by a strided rebuild4 variant. 2.6K: 11.8 GB at 426-474 tok/s,
  exact, decode unaffected. This passes the kill line (within 0.5 GB of 11.5 at >= 400). Ring 4: 11.5 GB at 388-423.
  Leave it off at 32K.
- The 6-8 GB end: B8 / B8-fast with the decoupled prefill peaks at 7.9-8.0 GB on 2.6K (283-306 tok/s; chunk 256 was
  58) and 9.6 GB at 32K (280 tok/s). With the MTP drafter: 8.2-8.5 GB, 23-28 tok/s decode.
- Dead: GPU rebuild without streaming (prefill is SSD-bound); warming the pools from the last chunk (+10 % waited
  misses: code prompt, prose answer); reading the model's safetensors directly (decode -17 % afterwards); the ext's
  read_rebuilt on a reader thread (it holds the GIL). The router readback is not the bottleneck (0.06 s per prompt).

## Main-track prefill (BigMoE's exact 24 GB engine, NewIdeas 05:45; 10-22 06:45), ~/TrainLab/main24min
D1i's bitwise parts, ported to a copy of BigMoE 86e548c without the decoupled attention split, which is not bitwise.
- The grouped exact tier prefill (BIGMOE_TIER_PF_GROUP=64) groups only where the reference call itself takes the
  sorted rhs gather (>= 4 pairs per expert), every group keeps that, and rows are padded to 64. BigMoE's old PF_GROUP
  lacked these conditions and failed on prose (LEDGER M24d/e). Logit-free greedy chunks (BIGMOE_PF_NOLOGITS=1).
- Bit-identical on all five prompts: prose / code / edit (BigMoE's v3b shas) and 2.9K / 5.7K. Time to first token
  (v3, C=128): prose 3.32 -> 2.95 s, code 6.23 -> 4.66 s, edit 6.42 -> 4.87 s, 2.9K 19.3 -> 12.7 s,
  5.7K 27.6 -> 21.3 s (-9 to -34 %). Footprint at 5.7K: 27.0 -> 25.5 GB.
- Handed to BigMoE (peer send, cc orchestrator). **ADOPTED into the v3 stack, 08:20 (BigMoE LEDGER PFG1)**: v3b shas
  at 96 and 256 tokens on code / edit / prose, spec == ar2, code prefill 5.95 -> 4.73 s in one window, GSM8K 0-39
  answers and token counts 40/40 identical to v3b.
- Not worth it here: the GPU raw-read rebuild (union reads are already at the SSD ceiling).
- Skipping pool-resident experts, the careful version v4 (~/TrainLab/main24min2, BIGMOE_TIER_PF_RESIDENT=1 on the
  adopted patch): ~30 % of a prompt's experts have their MSB-3 part in the base pool; for those only the LSB part is
  read (GIL-free pool), with a GPU rebuild4 applied as a lazy scatter, and cold rows go straight into their slots.
  Exact (same shas on all 5 prompts); vs the adopted patch, paired x2: code 4.74 -> 4.39 s (-0.35 s), 5.7K -1.3 s,
  but +1.2 GB footprint. PARKED: below NewIdeas' kill line (>= 0.5 s at 327 tokens). Earlier versions were slower
  (v1/v2: GPU assembly copies; v3: a fresh per-group buffer whose zeros + eval wait behind the GPU's previous
  group).
- Same session, B12 residency (frugal 10 %, + MTP): C_m 56 is the largest that stays <= 12 GB at 2.6K. Decode
  +7-14 % vs 51, hard-text KLD 0.1030 -> 0.0980.

## Trained sticky routing (NewIdeas 09:16): the 35B layer-local kill test -- KILLED 11:01; the lever exists zero-shot (per-pass union budget)
Feasibility (sent 09:16): no post-hoc method shows m=1 stickiness (path350's 1.7x) at kept quality; the realistic prize
is ~1.3-1.5x on pass bytes (m ~ 2-3, U(16) 32-40). Sticky routing changes the model's routing: a NON-EXACT route, so
adopting it is Kamen's call; gate = the Eval suite's per-part rule + Chat1000's agentic row.
- Zero-shot soft stickiness on the 125B traces (CPU; sticky/bonus125.py -> data/bonus125.txt, bonus125_vs_cap.txt).
  This is the training-free form of ReMoE's effect: +delta on the previous token's experts' logits before the top-10.
  Compared at equal U(16) with Chat1000's hard caps (same traces, energy and law), its error energy is 1.13-1.63x the
  hard cap's, on all 4 texts and under both trace bounds. The traces keep top-20 weights only, so the bounds are:
  "drop" (old experts below the top-20 can't stay) and "w20" (they get the 20th weight).
  The hard cap stays the zero-shot frontier; a uniform bias does not move it.
- Exact clairvoyant bound (CPU; sticky/oracle125.py -> data/oracle125.txt). This is the lowest error energy of ANY
  selection with union U per 16-token block, when left-out true experts cost w^2. It is 1.7-2.6x below the hard caps
  at equal U(16).
- The implementable per-pass union budget (CPU; sticky/budget125.py -> data/budget125.txt). Published as batch-aware
  routing: XShare (ICML'26) maximises the batch's total gating score; OEA (2511.02237) reports -39 % MoE decode latency
  on Qwen3-30B at batch 16. Method: in a 16-row pass every row's router output is known before the experts run. The
  pass set A = the U experts with the largest summed w^2; each row takes its top-10 within A.
  - On the 125B traces, at equal U(16), it has 1.4-1.9x LESS error energy than the hard caps (all 4 texts; fill-ins
    charged 0 or w20^2).
  - Chat1000's law then gives KLD 0.016-0.027 at U(16) = 32 on code / math and 0.043-0.083 on bulgarian / chat. The
    hard caps give 0.035-0.163 at the same union. Natural U(16) is 57-73, so this is ~1.8-2.2x fewer experts per
    16-row verify pass, inside frugal-10 %'s KLD band (0.10 hard text), with no training.
  - The 350 plan's verify passes are 16 rows. So the "union per pass" lever has a training-free published candidate
    that beats every sticky cap. Trained routers must be judged against it, not only against the caps.
  - Caveats: these are law estimates. It is non-exact (a row's routing depends on the other rows of its pass). Real KLD
    comes in the 35B GPU run (budget U = 16/24/32/40 added to sticky35).
- 35B kill test RESULT (sticky35.py; lock 10:36-11:01; data/sticky35_results.txt / .json, table data/sticky35_table.md).
  Setup: 1,212 steps x 1,024 tokens layer-local, 5 variants. The first start used betas 2 / 8 / 32: by step 200 their
  held-out U16 was 13-17, below cap 1's 19.9, so nothing could be compared at equal union. It was restarted at 10:40
  with betas 0.05 / 0.1 / 0.2 / 0.44 and the reuse-only control 0.2r.
  Converged held-out, locally (the original model's features):

  | beta | trust KL / layer | switch | U16 |
  |---|---|---|---|
  | original | 0 | 0.631 | 51.6 |
  | 0.05 | 0.0045 | 0.578 | 47.4 |
  | 0.1 | 0.015 | 0.534 | 43.8 |
  | 0.2 | 0.045 | 0.467 | 38.5 |
  | 0.44 | 0.144 | 0.359 | 29.9 |
  | 0.2r | 0.047 | 0.513 | 42.1 |

  The fp32 control (RouterFull(W0)) is bit-identical: KLD 0, per-layer err 0.
  - KILL RULE 1 FIRES. End-to-end, at equal U16, every one of the 15 trained points (5 variants x natural / cap3 /
    cap2) has MORE KLD than the zero-shot caps: 0.26-0.93x of their quality, where a >= 2x cut was required.
    - Natural top-k: beta 0.05 KLD 0.049 at U16 48.1 vs the caps' 0.021 (0.42x); beta 0.2 0.187 at 39.7 vs 0.058
      (0.31x); beta 0.44 (ReMoE's ratio) 0.404 at 30.8 vs 0.135 (0.33x).
    - With caps the gap narrows: beta 0.05 + cap2 0.93x, + cap3 0.91x.
    - The trained points are also worse than the untrained soft bonus.
  - KILL RULE 2 does NOT fire. The failure is not compounding:
    - Every trained point sits 1.01-1.35x above the zero-shot error law fitted on caps 1-6 (KLD = 1.06 err^0.85;
      err = each layer's MoE output error vs the original routing at the same input). The excess grows with beta.
    - Band additivity, all layers vs the sum of 4 bands of 10: 0.77x for beta 0.05 + cap2, 0.74x for zero-shot cap2.
  - Why it fails: the per-layer error itself. A small router-distribution KL (0.0045 nats) still moves every token's
    top-8 gate weights, i.e. the expert mixture. At equal union the trained routers' per-layer MoE error is 3x the
    caps' (beta 0.05: 0.021 at U16 48 vs cap6's 0.0067 at 49).
    - So the layer-local proxy (KL to the router + reuse) is the wrong objective.
    - An output-aware local objective (MoE-output reconstruction under the routing, NAEE-style) is untested. It
      would have to beat the zero-shot budget below, not just the caps.
- The union lever exists ZERO-SHOT: the per-pass union budget (sticky35's zero-shot rows, real KLD; batch-aware
  routing, published).
  - 16-row passes (natural U16 51.8). KLD cut vs the caps at equal U16:

    | budget | U16 | KLD (prose / code / gsm) | vs caps |
    |---|---|---|---|
    | 24 | 24.0 | 0.107 | 2.22x |
    | 32 | 31.8 (-39 %) | 0.069 (0.034 / 0.036 / 0.138) | 1.80x |
    | 40 | 38.9 (-25 %) | 0.038 | 1.67x |

  - 4-row passes, today's MTP verify width (natural U4 21.1):

    | budget | U4 | KLD (prose / code / gsm) | vs caps |
    |---|---|---|---|
    | 12 | 12.0 (-43 %) | 0.097 | 2.48x |
    | 16 | 15.8 (-25 %) | 0.046 (0.030 / 0.032 / 0.074) | 1.89x |
    | 20 | 18.7 (-11 %) | 0.026 | 1.18x |

  - OEA's published rule (union of the rows' top-k0) is 1.92-2.68x better than the caps at k0 3-5. The w^2 budget
    beats OEA by ~1.1x at equal union and by far more on multilingual text: German KLD 0.156 vs 0.439 at U16 ~23-24.
  - The soft +delta bonus is 0.84-1.21x of the caps: no real gain. The chat segment's KLD is high in every row
    (0.81-1.44, from layers 0-20) and needs its own look.
- DEAD entry (format of ~/NewIdeas/DEAD.md; sent to NewIdeas for that file):
  ## Trained sticky routing, layer-local (ReMoE's router terms without the LM loss) on the 35B (killed 11:01, TrainLab sticky35.py)
  - Setup: 40 routers trained on captured MoE inputs (1,212 x 1K tokens, frozen 35B, no cross-layer gradient). Loss:
    KL(P||P_ref) + beta (reuse + 0.25 smooth + 0.25 lag), ReMoE's weights. Betas 0.05 / 0.1 / 0.2 / 0.44 (ReMoE's
    ratio) + reuse-only 0.2r.
  - Locally sticky: held-out U16 51.6 -> 47.4 / 43.8 / 38.5 / 29.9 at router trust KL 0.0045-0.14 nats.
  - Killed end to end: all 15 points (natural top-k, cap3, cap2) have MORE KLD than the zero-shot caps at equal U16,
    at 0.26-0.93x of their quality, where a >= 2x cut was needed.
    - beta 0.05: 0.049 at U16 48 vs the caps' 0.021.
    - beta 0.44: 0.40 at U16 31 vs 0.135.
    - Best: beta 0.05 + cap2, 0.93x.
  - Not compounding: 1.01-1.35x over the caps' error law (KLD = 1.06 err^0.85); band additivity 0.77x vs 0.74x.
  - LESSON: KL-to-router is the wrong local proxy. A small router KL re-weights every token's top-8, so at equal union
    the per-layer MoE output error is 3x the caps' (0.021 vs 0.0067 at U16 ~48).
  - The union lever needs no training: the zero-shot per-pass union budget (batch-aware routing, XShare / OEA) cuts
    KLD 1.7-2.5x vs the caps at equal union (16-row U16 32: 0.069; 4-row U4 16: 0.046).
  - Revival path: an output-aware local objective (MoE-output reconstruction under the routing) that beats the
    budget, not just the caps.
- Reading for the 350 plan's U1 lever (union per 16-row pass) and for today's 4-row passes:
  - Sticky router training: KILLED.
  - A training-free per-pass expert budget gives the union cut at ~half the caps' KLD. This is non-exact (a row's
    routing depends on the other rows of its pass), so adopting it is Kamen's call.
  - Gate: the Eval suite's per-part KLD rule + task results + Chat1000's agentic row on the 125B.
  - Prepared, not run: sticky/budget125_tf.py measures real 125B KLD at R-row verify passes (Eval suite reference
    engine, gate wrapper, budget and OEA variants), ~30 min under a lock.

## 125B per-pass expert budget at 4-row verify passes (NewIdeas 11:12; lock 12:23-13:02) -- real KLD: CANDIDATE
sticky/budget125_tf.py on the Eval suite's exact reference engine (read-only import). The suite's four tf parts, every
4th sequence: 35 sequences, ~8K positions per arm. Each completion runs as 4-row decode_verify forwards (an MTP verify
pass whose drafts are all right). Arms: per pass, the U experts with the largest summed w^2 (bU), or OEA's union of the
rows' top-k0 (oK); each row takes its top-10 within the set. KLD is against the suite's cached reference.
data/budget125_tf.txt, budget125_tf_R4.json, table data/budget125_table.md.

| arm | distinct experts / pass / layer | code | gsm | ifeval | multi |
|---|---|---|---|---|---|
| exact 4-row floor | 27.1 (natural) | 0.0002 | 0.0007 | 0.0040 | 0.0023 |
| budget 18 | 17.9 (-34 %) | 0.0019 | 0.0050 | 0.0245 | 0.0126 |
| budget 16 | 16.0 (-41 %) | 0.0021 | 0.0070 | 0.0364 | 0.0177 |
| budget 14 | 14.0 (-48 %) | 0.0031 | 0.0096 | 0.0557 | 0.0268 |
| OEA k0 5 | 14.9 (-45 %) | 0.0036 | 0.0076 | 0.0539 | 0.0351 |
| OEA k0 4 | 14.4 (-47 %) | 0.0042 | 0.0126 | 0.0730 | 0.0374 |
| target (ar4 x 1.10) | | 0.0025 | 0.0045 | 0.0204 | 0.0205 |
| frugal-10 % (step) | | 0.0147 | 0.0307 | 0.1153 | 0.0924 |

- Against frugal-10 %'s level, every budget arm is below it on every part: budget 18 by 4.7-7.7x, 16 by 3.2-7x, 14 by
  2.1-4.7x.
- Against the strict per-part targets: budget 18 meets code and multi and misses gsm (0.0050 vs 0.0045) and ifeval
  (0.0245 vs 0.0204) narrowly. The 4-row floor alone is 0.0040 on ifeval. Budget 16 meets code and multi.
- These are point means on a subset. The suite's formal rule is the 90 % CI of the ratio vs ar4 (cluster bootstrap),
  so this is a screen, not a verdict.
- Top-1 agreement: budget 18 gives 0.994 / 0.986 / 0.943 / 0.969 against the floor's 0.999 / 0.995 / 0.976 / 0.985.
- The w^2 budget beats OEA at equal union: budget 14 vs OEA 4 is better on every part (ifeval 0.056 vs 0.073), and
  about equal to OEA 5 at a smaller union.
- Pass-time effect (for the 350 table / main track): distinct experts per 4-row pass drop 27.1 -> 17.9 / 16.0 / 14.0
  (0.66 / 0.59 / 0.52x).
  - In RAM, expert bytes per verify pass fall by the same factors. The routed experts are most of a 4-row pass, so
    budget 18 should cut ~20-25 % of the pass time.
  - At 24 GB, the misses per pass are bounded by the same cut.
  - Both need the engine hook (mask router logits before the top-k in the verify path; no kernel change) to measure.
- Caveats:
  - Non-exact: a row's routing depends on its pass-mates. In real decoding the pass-mates are the drafted tokens,
    wrong drafts included, so the target's outputs depend on the drafter. Kamen's call.
  - No agentic row here. On the 35B the copy-heavy chat segment loses ~0.8 KLD under ANY routing change, so
    Chat1000's agentic row must gate it.
  - Teacher-forced with all-right drafts only.
- Predicted misses at 24 GB (125B traces, LRU C=128, teacher-forced, no JIT; sticky/miss24.py):
  - 4-row passes: budget 18 = 0.44-0.56x, 16 = 0.36-0.46x of natural.
  - 16-row passes: budget 40 = 0.24-0.37x, 32 = 0.17-0.24x, 24 = 0.12-0.14x.
  - The dropped low-weight tail is what misses the pools, so misses fall far more than the union.
- Next (proposal):
  - The engine hook in a copy of BigMoE's verify block, then tok/s on the 24 GB main track (and B12) with budget
    18 / 16.
  - Then the Eval suite's decode sets + Chat1000's agentic turns.

## 24 GB main-track tok/s with the per-pass expert budget (turn14; lock 16:34-16:45) -- b16 ADOPT-CANDIDATE, b18 mixed
BigMoE v3 24 GB (C=128, tiers 4433333333, 17.07 GB pools), JIT prefetch on, MTP spec 3:3:0, 256 tokens. Router hook
sticky/budget_hook.py (TL_BUDGET); runner sticky/bench_budget24.py = SpecDecode's test_spec125.py unchanged + counters.
data/budget24.txt.

| prompt | arm | tok/s | vs natural | tok/pass | ms / pass | misses / pass | loader io ms / pass | JIT pf used / issued | output vs exact |
|---|---|---|---|---|---|---|---|---|---|
| code | natural | 34.38 / 34.37 | | 3.46 | 100.6 | 332 | 51.6 | 104 / 242 | identical |
| code | b18 | 42.45 | +23.5 % | 3.51 | 82.6 | 204 (0.61x) | 37.4 | 59 / 244 | differs at 182 |
| code | b16 | 45.35 | +31.9 % | 3.61 | 79.5 | 185 (0.56x) | 35.5 | 50 / 239 | differs at 24 |
| long (6K ctx) | natural | 24.98 / 24.99 | | 2.53 | 101.5 | 353 | 53.4 | 104 / 235 | identical |
| long | b18 | 21.80 | -12.7 % | 2.37 | 108.7 | 242 (0.69x) | 57.0 | 64 / 246 | differs at 5 |
| long | b16 | 30.58 | +22.4 % | 2.67 | 87.2 | 238 (0.68x) | 41.3 | 58 / 262 | differs at 5 |

- Natural runs reproduce to 0.04 %, and their outputs are identical to ar2: the exact path is untouched by the hook.
- Misses per verify pass fall to 0.56-0.69x, as the trace replay predicted (0.44-0.56x). Distinct experts per
  layer-pass: 27-28 -> 16-18.
- Budget 16 gains on both prompts: +31.9 % code, +22.4 % long.
- Budget 18 gains +23.5 % on code but loses 12.7 % on long. Its run wrote a different continuation (it diverged at
  token 5), with fewer tokens per pass (2.37 vs 2.53) and more loader time per pass (57 vs 53 ms) despite 31 % fewer
  misses. It is a single non-exact run, so the cause is open; per-pass time is the better metric across texts.
- The JIT prefetcher still issues ~240 reads per pass for natural routing; with the budget only 50-64 are used (104
  natural). A budget-aware prefetch (predict the pass set) is the next lever on top.
- Reading against the pre-set lines (>= +10 % on both prompts = adopt-candidate, < +5 % = kill):
  - b16 = ADOPT-CANDIDATE for Kamen's call. Its quality cost (125B KLD screen) is code 0.0021 / gsm 0.0070 /
    ifeval 0.0364 / multi 0.0177: 3-7x below frugal-10 %, over the strict targets on gsm and ifeval.
  - b18 = mixed (more prompts / runs needed).
- Non-exact: outputs depart from exact decoding within the first 5-182 tokens.
- Next: more prompts and 2-3 runs per arm, since text variance dominates non-exact tok/s; the budget-aware JIT;
  the Eval suite's decode sets + Chat1000's agentic row for b16 / b18; the R=16 KLD for the 350 table.
- Calibration (NewIdeas 16:55): the trace replay's miss ratios were optimistic by ~1.2-1.5x vs turn14.
  - b16 predicted 0.36-0.46x, measured 0.56-0.68x; b18 predicted 0.44-0.56x, measured 0.61-0.69x.
  - The replay is teacher-forced, with no wrong drafts in the pass and no JIT prefetch.
  - Its 16-row predictions are lower bounds until the replay includes drafted rows.

## Combined quality screen: the per-pass budget ON TOP of v3's tiers (turn15; lock 20:17-20:58) -- strict check FAILS (as pre-registered)
Eval suite variant plan4433333333: exact 4-bit prompt prefill, tiered 4-row verify passes. The pool-slot verify calls run
row by row through the suite's decode tier branch; the suite tiers multi-row calls only on its full-E prefill scratch,
and the 19:56 try asserted there. 33 of 35 sequences (time limit), ~7.5K positions per arm.
data/budget125_tf_R4_plan4433333333.json, per position _pos.npz; tables data/budget125_table_plan.md,
data/rule125_plan4433333333.txt.

| arm | distinct experts / pass / layer | code | gsm | ifeval | multi |
|---|---|---|---|---|---|
| v3 tiered floor | 27.0 | 0.0007 | 0.0024 | 0.0128 | 0.0113 |
| b22 | 21.4 (-21 %) | 0.0016 | 0.0045 | 0.0209 | 0.0152 |
| b20 | 19.7 (-27 %) | 0.0018 | 0.0053 | 0.0257 | 0.0162 |
| b18 | 17.9 (-34 %) | 0.0024 | 0.0065 | 0.0323 | 0.0200 |

The Eval suite's own rules (sticky/rule125.py) compare against ar4's KLD and top-1 at the SAME positions, with 90 %
cluster-bootstrap CIs.
- v3's tiers alone PASS every part: KLD x0.51-0.64, P99.9 x0.39-0.64.
- b22 FAILS. Mean KLD ratio: gsm x0.97 [..1.20], ifeval x1.04 [..1.18], code x1.10 [..1.52], multi x0.74. P99.9:
  code x1.80, ifeval x1.12 [..1.73]. Top-1 CI lower bounds: ifeval -1.9 pt, multi below -0.5.
- b20 fails at x1.16-1.28; b18 at x1.41-1.68.
- NewIdeas' pre-registered prediction holds (worst-part b22 1.21 / b20 1.48 / b18 1.87; its kill line, b20 or b22
  passing every part, is not met). The tails (P99.9) are the hardest rule: the budget creates rare large-KLD positions,
  most on code.
- The simple "ar4 whole-part mean x1.10" targets in report125 understate the strictness: on these subsets ar4's own
  KLD is lower than its part means.
- Reading: no budget with a useful cut passes Kamen's strict combined check. Whether an opt-in fast mode may be judged
  on task results instead is Kamen's call. turn16 measures b16 / b18 speed on six prompts for that question.
  NewIdeas' uncertainty-gated budget bet has its per-position data in _pos.npz.

## Six-prompt 24 GB tok/s: budget b16 / b18 and the budget-aware JIT (turn16; lock 22:18-22:45)
v3 24 GB, JIT on, MTP 3:3:0, 256 tokens, one run per arm and prompt (natural reproduces to 0.04 %). The agent prompt is a
real Claude Code turn (Chat1000 bash17, 8,969 tokens); math is GSM8K 0. data/budget24.txt.
| prompt | natural tok/s | b16 | b18 | b16+jit | ms/pass natural -> b16 / b18 / b16+jit |
|---|---|---|---|---|---|
| code | 32.4 | 39.4 (+22 %) | 40.1 (+24 %) | 45.4 (+40 %) | 107 -> 91 / 87 / 79 |
| edit | 32.5 | 41.5 (+28 %) | 37.6 (+16 %) | 49.3 (+52 %) | 121 -> 95 / 105 / 80 |
| long | 23.6 | 29.7 (+26 %) | 23.1 (-2 %) | 33.7 (+43 %) | 107 -> 90 / 103 / 79 |
| prose | 29.4 | 29.6 (+1 %) | 33.2 (+13 %) | 37.6 (+28 %) | 78 -> 81 / 75 / 64 |
| agent | 21.4 | 33.5 (+57 %) | 24.3 (+14 %) | 38.5 (+80 %) | 122 -> 105 / 101 / 91 |
| math | 28.9 | 37.0 (+28 %) | 33.2 (+15 %) | 42.3 (+46 %) | 95 -> 82 / 92 / 72 |
geomean tok/s gain: b16 +25.8 %, b18 +12.9 %, b16+jit +47.3 %
geomean per-pass speed-up: b16 x1.15, b18 x1.12, b16+jit x1.35
JIT reads per pass (issued / used), b16 -> b16+jit: code 233/50 -> 95/44, edit 287/45 -> 125/41, long 256/57 -> 101/52, prose 171/31 -> 61/27, agent 306/63 -> 130/59, math 234/50 -> 98/47

- The budget-aware JIT (TL_BUDGET_JIT=1: the look-ahead routers masked to the predicted pass set) is a large extra lever.
  It cuts prefetch reads per pass by ~60 % at nearly the same number used. b16 + JIT = +28..+80 % tok/s on every prompt
  (geomean +47 %, per-pass time x1.35), against b16 alone's +26 % (x1.15).
- tok/s mixes in acceptance changes, because the budget arms write different continuations (e.g. agent: 3.51 vs 2.61
  tok/pass). Per-pass time is the cleaner figure.
- Quality is unchanged by the JIT (it only changes what is prefetched, not the routing). b16 / b18 still FAIL Kamen's
  strict combined check (turn15), so this speed goes to Kamen as a task-result question, Kamen's call. The
  uncertainty-gated budget (turn17, NewIdeas) is the path to a strict pass.

## B2 screen: confidence-gated precision for cold reads (NewIdeas QUEUE B2; turn18, lock 04:23-04:59 Sun) -- data for the gate
Eval suite engine: exact 4-bit prompt prefill, 4-row verify passes on the tier path, 35 of 35 sequences (no drops; quiet-
window safe driver). Precision arms: every lower-rank expert, cold or resident, at the arm's bits, which is pessimistic vs
cold-only 2-bit. Per position KLD + the MTP drafter's depth-1 confidence:
~/TrainLab/data/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz; rules in data/rule125_plan4433333333_mtp_b2.txt.

| arm | gsm | ifeval | code | multi | suite rules vs ar4 (mean KLD ratio) |
|---|---|---|---|---|---|
| v3 4433333333 | 0.0023 | 0.0128 | 0.0007 | 0.0113 | PASS (x0.51-0.64) |
| ranks 4-9 MSB-2 (4433222222) | 0.0049 | 0.0301 | 0.0020 | 0.0294 | FAIL (x1.09-1.49) |
| ranks 2-9 MSB-2 (4422222222) | 0.0079 | 0.0484 | 0.0033 | 0.0526 | FAIL (x1.77-2.55) |

- The drafter's depth-1 top-1 = the next token 83-84 % in every arm (mean top-1 prob 0.86): 2-bit target hidden states
  do not degrade it.
- Ungated 2-bit fails, as expected. The pre-registered test is the gate: the 2-bit arm on passes the drafter is sure
  of, v3 otherwise. Kill: no gate passes the bar with >= 70 % of code passes on 2-bit. NewIdeas runs it on CPU
  (ugb_rule.py).

## B2 cold-only 2-bit at C=128 (turn18c128; 05:14-05:39 Sun)
An LRU replay of residency at 128 experts per layer, seeded by the prompt, then one update per 4-row verify pass. A cold
expert that every row of the pass uses at rank >= R is read and computed as MSB-2; everything else at v3's tiers.
35/35 sequences. data/budget125_tf_R4_plan4433333333_mtp_b2c128{.json,_pos.npz}, rules
data/rule125_plan4433333333_mtp_b2c128.txt.

| arm | gsm | ifeval | code | multi | SSD per token (v3 -> cold-only) | suite rules |
|---|---|---|---|---|---|---|
| v3 (turn18) | 0.0023 | 0.0128 | 0.0007 | 0.0113 | 168.7 MB | PASS |
| cold ranks 4-9 at 2-bit | 0.0025 | 0.0142 | 0.0009 | 0.0121 | 136.2 MB (-19.2 %) | FAIL: code P99.9 x1.39, ifeval top-1 CI -0.8 pt |
| cold ranks 2-9 at 2-bit | 0.0029 | 0.0155 | 0.0009 | 0.0129 | 128.7 MB (-23.7 %) | FAIL: tails (code, multi), top-1 CI (ifeval, multi) |

- Bytes corrected 06:19 Sun: a 2-bit read is records4's planar P2 part, 1,540,096 B (MoonShots), not 2/3 of the MSB-3
  part (1,441,792 B) as first counted (-22.3 % / -27.4 % then). The driver now counts Q2 bytes.
- Mean KLD stays within 5-15 % of v3; every mean-KLD rule passes (x0.52-0.77 of ar4).
- Ungated, the arms miss on the P99.9 tails and on top-1 lower CI bounds; the point estimates are positive and only
  6-8 sequences per part were run.
- 52-64 of 74 cold reads per token go 2-bit.
- NewIdeas runs the drafter gate on the per-position data. C=192 is next (turn18c192).
- NewIdeas' read-out (ugb_b2c128.txt): GATED B2 is KILLED by the pre-registered line (realizable drafter gate 51 % / 63 %
  of code passes vs 70 %; oracle 75-94 %). UNGATED cold-only r4 is the result: -22.3 % (-19.2 % after the 06:19 fix) SSD bytes at 24 GB, mean KLD near v3,
  fails only code P99.9 (1-2 positions) and an 8-seq top-1 CI -> a task-result candidate (Kamen's bar). It needs an engine
  that reads planar MSB-2 cold experts, then an Eval suite decode verdict.
- Implied speed (estimate, not measured; redone with the corrected -19.2 %): -19.2 % cold bytes x ~0.8 bytes-bound
  (Fit12's 12 GB breakdown) x 0.75-0.9 io on the critical path x 0.45-0.56 io share of a 24 GB pass (turn14 / turn16) =
  5-8 % shorter passes, ~+5-8 % tok/s on natural v3, ~+3-6 % on top of b16 + budget JIT. That assumes MSB-2 reads stream at the same GB/s and the dequant is free.
- Against v3 directly (sticky/vs_v3.py: same positions, 90 % cluster CI; data/vs_v3_plan4433333333_mtp_b2c128.txt):
  c128r4's mean KLD is x1.06 / 1.10 / 1.25 / 1.08 of v3's (gsm / ifeval / code / multi; code CI 0.79-1.54), P99.9 up to
  x2.17 (code); c128r2 x1.15-1.26. The ungated lower-rank MSB-2 arm (4433222222) is x2.10-2.69.

## MoonShots' 3B quality screen (NewIdeas Sun; turn19, queued) -- design
MoonShots' B9 replay: no LSB sidecar, every rank at MSB-3; today's 144.4 MB per token -> 87.0 MB (-39.8 %) at 24 GB.
- Arm p3333333333: every rank at MSB-3 on every verify pass (ranks 0-1 at 3 bits were never measured).
- Arm 3b140t32: 3B in full, per layer LRU replays seeded by the prompt's rows:
  - a base pool of 140 MSB-3 experts; misses at ranks 0-3 are read into it;
  - a cold expert at ranks 4-9 (best rank over the pass's rows) goes through a 2-bit cache of 32 (a Q2 read of
    1,540,096 B on a miss) and is computed at 2 bits; it never enters the base pool;
  - the bytes reference is v3's pools: base 147 + sidecar 46 (ranks 0-1).
- Arm c192r4: B2 cold-only at C = 192 (the 45 GB waypoint).
- The 2-bit compute is the suite's MSB-2 tier (plan digit 2). That is the same code 3B would read: MoonShots' Q2 is
  records4's planar MSB-2 part (P2, 1,540,096 B) computed by tierglu4's tier 2 (data/3b_prereq_prereg.txt). So the screen
  measures 3B's real numerics. (Corrected 06:16: an earlier line here called it a pessimistic proxy.) A re-fit 2-bit code
  (Fit12's B3 work) would be a separate gain.
- Kill (NewIdeas): 3B's mean KLD > 1.5x v3's on any part (vs_v3.py), or the strict rules failing by more than c128r4's
  margin (code P99.9 CI up to x1.62, ifeval top-1 CI down to -0.8 pt).
- Prior art (published): B2 cold-only and 3B's 2-bit cache are HOBBIT's idea (arXiv 2411.01433, mixed-precision expert
  offloading): on a cache miss, load a low-precision copy of a less important expert; its cache also keeps the low-precision
  copies. HOBBIT scores importance by gate weight against a threshold; our rule is the expert's best rank over the pass's
  rows (>= R).
- HOBBIT's weight rule, checked on CPU (sticky/wt_calib.py on Breakthrough's 125B traces, 4-token passes,
  data/wt_calib_C128.txt): at the same bytes saved as r4 (~19 %), its damage proxy (summed w^2 of the 2-bit uses per token)
  is only 3-4 % lower in every set. The 125B's routing is flat over ranks 4-9 (mean normalised top-10 weights 0.084 ->
  0.056), so the rank rule already almost is the weight rule. Not worth a GPU screen. Arm cC_wT stays in the driver.
- The same traces at C = 192: r4 saves ~20 % (19.9-20.5 %) of a smaller stream (code 100.4 -> 79.8 MB per token, vs 175.6 -> 141.1
  at C = 128); the damage proxy falls 0.289 -> 0.167 on code.

## UGB with the drafter gate at 4 rows, from existing data (CPU, Sun 06:22) -- KILLED; 8-16 rows predicted dead
- turn15's natural arm (v3 tiers, 4-row passes) is bit-identical to turn18's v3 arm at all 7,553 shared positions (KLD and
  top-1). So turn15's budget arms (b22 / b20 / b18) join turn18's drafter confidence (v3 arm's depth-1 top-1 log-prob)
  exactly. The turn17 rerun is not needed for 4 rows.
- NewIdeas' ugb_rule.py (read-only run; gate DRAFT_EX = the v3 arm's drafter; data/ugb_r4_draft_turn15x18.txt):
  - The largest code coverage at a full PASS is 65 % (b20 at thr -0.127); b22 and b18 pass only at 53 % (thr -0.030).
  - Pre-registered line 80 % (QUEUE L1-08 says 70 %): KILL either way.
  - What fails first as coverage rises: the code P99.9 tail, then the ifeval / multi top-1 CI.
  - The ORACLE gate (the arm's own min top-1, not realizable) survives at 85-90 %, so a better realizable gate would
    have room.
- L1-08 (8-16-row passes), predicted on CPU without a GPU run. At the thresholds that pass at 4 rows, the gate covers
  only 33-46 % of code passes at 8 rows and 16-25 % at 16 rows; even thr -0.348 gives 59 % / 38 %.
  - The matching budgets carry about the same damage per position (sticky/budget_rows.py on the traces: b24 at 8
    rows drops the w^2 share b16 drops at 4 rows, code 10.0 % vs 10.2 %; b32 ~ b20).
  - So the realizable drafter gate cannot reach 70 % of code passes at 8-16 rows. Caveat: the damage per position at 8
    rows is inferred from the energy proxy, not measured.
- Finer check (sticky/ugb_learned.py, data/ugb_learned_eps*.txt): with the true pass structure and a 5 % threshold grid,
  the drafter-min gate reaches 70 % of code passes at a full PASS for b22 and b20 (gsm 71 %, ifeval 12 %, multi 13 %);
  b18 reaches 53 %. ugb_rule's coarser grid gave 65 %. The pass structure is the prompt's last position alone (prefill),
  then chunks from p0 + 1; ugb_rule chunks from p0.
  - 4-row UGB is still KILLED by its pre-registered 80 % line (ugb_rule.py, turn17.sh).
  - L1-08's 70 % line is for 8-16 rows. There the gate covers 46-59 % of code passes at the thresholds that pass at
    4 rows, so it is still predicted dead.
- A learned gate does not help (CALM-style, 2207.07061; L2 logistic regression out of fold by sequence):
  - Features known before the pass: the drafter's lp1 at the pass's positions, and the previous pass's lp1 / margin /
    entropy.
  - It covers 19-32 % of code passes at a PASS, or none, for labels "excess KLD > 0.02 / 0.1 / 0.3". That is below the
    plain drafter minimum: the tail events that fail P99.9 are too rare to learn from 35 sequences.

## Byte replay cross-check vs MoonShots' B9 (CPU, Sun 06:28)
- sticky/bytes_xcheck.py runs the driver's own 3B replay (budget125_tf._observe_3b) on MoonShots' inputs: Breakthrough's
  traces, first half warm, second half measured, per token. data/bytes_xcheck_R1.txt.
- Results, mine vs MoonShots' budget24_q1540096.txt:

  | configuration | MB per token (mine) | MB per token (MoonShots) | saved (mine / MoonShots) |
  |---|---|---|---|
  | v3 147 + 46 | 143.2 | 144.4 | |
  | 3B 140 + 32 | 88.3 | 87.0 | -38.3 % / -39.8 % |
  | 3333333333 at 163 | 107.5 | 108.3 | -24.9 % / -25.0 % |

  With 4-token passes 3B saves -39.4 %. So B9's bytes are confirmed within ~1.5 %.
- The residue: my replay pins a pass's residents before inserting its misses, and does not warm the 2-bit cache during
  the warm-up.
- Trace lengths: chat holds 559 tokens and math 767, despite the "_800" file names. My first versions of
  bytes_xcheck / wt_calib / budget_rows assumed 800 and under-counted chat and math.
  - Fixed: real lengths, half / half.
  - The conclusions above are unchanged: HOBBIT's weight rule is -3-4 % damage proxy at matched bytes in every set; b24
    at 8 rows ~ b16 at 4 rows in every set.

## MoonShots' 3B quality screen (turn19; lock 06:13-06:52 Sun) -- 3B KILLED; and the cold-only replay was optimistic
35/35 sequences. data/budget125_tf_R4_plan4433333333_mtp_b3b{.json,_pos.npz}; data/vs_v3_plan4433333333_mtp_b3b.txt;
data/rule125_plan4433333333_mtp_b3b.txt.

| arm | gsm | ifeval | code | multi | mean KLD vs v3 (90 % CI upper) | suite rules vs ar4 |
|---|---|---|---|---|---|---|
| v3 (turn18) | 0.0023 | 0.0128 | 0.0007 | 0.0113 | x1 | PASS |
| p3333333333 (ranks 0-1 at 3 bits too) | 0.0050 | 0.0268 | 0.0027 | 0.0267 | x2.09-3.68 (..4.39) | FAIL everywhere |
| 3B (3b140t32) | 0.0056 | 0.0289 | 0.0029 | 0.0281 | x2.25-3.97 (..4.94) | FAIL everywhere |
| cold-only C=192 r4 (c192r4) | 0.0023 | 0.0137 | 0.0008 | 0.0119 | x0.98-1.11 (..1.37) | FAIL: ifeval P99.9 CI 1.15, multi top-1 CI -0.7 pt |

- 3B is KILLED by NewIdeas' rule: mean KLD x2.25-3.97 of v3 on every part (line 1.5), and every strict rule fails by
  far more than c128r4's margins.
- The damage is ranks 0-1 at 3 bits. p3333333333 alone is x2.09-3.68 of v3; the 2-bit cache adds 5-12 % on top. v3's
  LSB sidecar for the top-2 ranks is load-bearing.
- 3B's bytes on these sequences (prompt-seeded pools, 4-row passes): v3 153.3 -> 104.8 MB per token (-31.6 %);
  MoonShots' warm replay gives -39.8 %.
- c192r4 bytes as counted then: 98.1 -> 75.0 MB (old 2/3 size); 78.2 MB (-20.3 %) with the P2 size. But see the next
  point.
- CORRECTION, cold-only (turn18c128, turn19 c192r4): the replay was optimistic (sticky/cold_faithful.py on the traces;
  data/cold_faithful_C128_r4.txt, _C192_r4.txt).
  - In the replay, an expert read at 2 bits entered the LRU as if fully resident. Its later uses were free full-precision
    hits, in the bytes and in the computed quality alike.
  - A faithful single-pool engine keeps the 2-bit copy and must upgrade it when the expert is next used at ranks 0-3:

    | model (C = 128) | SSD bytes saved | 2-bit share of rank 4-9 uses |
    |---|---|---|
    | as screened (optimistic) | -18-20 % | 14-22 % |
    | faithful, upgrade = full MSB-3 re-read | -1.5-4 % | 23-38 % |
    | faithful, planar records4, upgrade = third plane (0.72 MB) | -11.7-12.9 % | 23-38 % |

    At C = 192, planar: -11.4-14.0 % with 11-24 % 2-bit uses (screened: 7-12 %).
  - MoonShots' B (budget24.py: a separate 2-bit cache, pools rebalanced) is faithful: -18.0 % at 24 GB (C_m 129, C_l 32,
    C_2 32) with 25 % of rank 4-9 uses at 2 bits.
  - So c128r4's x1.06-1.25 and c192r4's x0.98-1.11 of v3 understate a realizable design's damage; its 2-bit exposure is
    ~1.2-2x higher. Earlier lines in this file that call ungated cold-only r4 a -19-22 % candidate stand corrected.
  - A faithful screen of B (and of the planar single pool) is the next GPU step if this line continues.

## B10 v2 layer-sliced quality screen (NewIdeas Sun 07:00; turn20, ready 09:36) -- design
- Question: does trellis coding of the experts (Fit12's B10 v2: base at ~2.1 bits for ranks 2-9, base + residual at
  ~3.6-4.1 bits for ranks 0-1, RHT + BlockLDLQ, target W4) stay within x1.5 of v3's KLD per part with all 48 layers coded?
  The screen codes a few layers (2, 24, 46) and extrapolates.
- Why not "layer L trellis, the other layers v3, vs v3 at the same positions" (NewIdeas' first design): the kill line
  is a ~1 % KLD excess per layer (0.5 / 48). The arm's change at L mixes with v3's other 47 layers' errors and with the
  R-row floor (the all-4-bit R-row arm sits at 20-35 % of v3's KLD: gsm 0.00065 vs 0.00231, ifeval 0.00398 vs 0.01284,
  code 0.00024 vs 0.00073, multi 0.00233 vs 0.01128). That mixing term is zero on average but its per-position spread
  is far above 1 %, and turn18 has 6-15 sequences per part.
- Design: one layer ALONE, the other layers exact 4-bit (plan 4444444444, the same row-split path for every arm), scored
  as lkld = KLD(the all-4-bit arm || the arm) over the full vocabulary from both arms' logits in the same passes
  (budget125_tf TL_LOCAL=1). No mixing terms: lkld is layer L's own damage.
  - d<L>: v3's tiers at L (4-bit ranks 0-1, MSB-3 ranks 2-9) = layer L's share of v3's damage.
  - t<L>: the trellis at L. x<L> (only if the smoke asks): the dense path's own floor.
  - r_L = (t - x) / (d - x); all 48 layers ~ x((F + mean r x Q) / v3) at turn18's v3 positions, F = the floor arm's KLD
    vs the cached reference, Q = v3 - F. The 1.5 line is r ~1.6-1.7 per part. Also printed: 48 x mean (t - d) / v3,
    which assumes the sampled layers are average ones; the ratio form does not.
  - Additivity across layers is assumed (independent errors: their cross terms average out), as for v3 itself.
- Mechanics (sticky/dense_layers.py): in a chosen layer, on verify passes, the MoE block takes its 10 experts from dense
  float32 weights: V3Src = the engine's own 4-bit codes, MSB-3 via the suite's tier_weights (bit-identical to its fast
  decode path); TrSrc = Fit12's decode(L, e), held as int8 with a scale per 64 inputs (relative error ~3.5e-5, logged
  per run; the trellis's own error is estimated at ~4e-3 (full) to ~6e-2 (base), not measured). The streamed switch
  still runs on the same ids, its output unused.
  One process per layer: a trellis layer is ~5.4 GB in RAM next to the ~20 GB engine.
- Smoke (one multi sequence, the other layers v3): d<L1> must reproduce v3 (its lkld under 10 % of x<L1>'s, the KLD vs
  the reference within 2 %); 10-50 % adds the x<L> floor arms; above 50 % the run stops.
- Tests (CPU): tests/test_dense_layers.py (the dense path against the suite's gather_qmm per row and rank, at 4-bit and
  MSB-3; TrSrc against the int8 dequantization; the block against qwen4_exp's stock block) and tests/test_b10_extrap.py
  (synthetic data with known r and ratios; one-run-per-layer merge; a reference arm differing between runs stops it):
  ALL OK. The cold-only / 3B tests still pass.
- Prior (arithmetic, not a result): base at ~2.1 bits ~0.06 relative MSE vs W4 against MSB-3's ~0.026, so r ~2.4 if
  KLD tracked weight MSE. BlockLDLQ shapes the error toward what the layer's inputs see, so the screen decides.
- OUTCOME (Sun 10:31 / 17:12): the screen never ran. Fit12's own held-out checks killed the codes first (B10a: the
  2-bit base 2.49x MSB-3's error, line 1.5x; B10-r with a broad 32K-token calibration: 1.85x, unseen prose 2.29x), so
  NewIdeas closed the 2-bit route. turn20 is dropped; the reader (format A) stays for a possible TRL-3 bet.

## S5 (HC8) + S9 (top-8) strict quality screen (NewIdeas Sun 09:45, 350 STACK; turn21, ready Mon 03:09) -- design + pre-registration
- Question: are two cheap main-track pieces near-free in quality at v3's tiers (24 GB setting; tf has no repetition
  guard, so guard off)? HC8 = the trunk's hyper-connection mixers (input_mix_weight_down / up of the 97 GatedResiduals)
  and the PLE key / value projections in MLX affine 8-bit, group 64, instead of bf16 (~3 ms per check, -0.6 GB RAM).
  top-8 = each verify row keeps its top 8 of 10 experts, renormalized over the 8 (-16 % union; passed the task bar 5/5
  at 12 GB as b12llk8u2).
- Arms (turn18's positions: TL_MTP=1, every 4th sequence of gsm / ifeval / code / multi, 7,966 positions):
  p4433333333h (v3 + HC8), p44333333 (v3 + top-8), p44333333h (both). v3 = turn18's arm at the same positions when the
  smoke reproduces it bit for bit (else it reruns in the batch).
- Mechanics: HC8 is the pinned hc8.install (MoonShots 3055bf7 patches/hc8/hc8.py, the file Breakthrough pins; copied
  unchanged to sticky/hc8_3055bf7.py, md5 7993470...). sticky/hc8_switch.py runs it once and keeps the bf16 modules
  beside the 8-bit ones, so each arm puts its own set in place; the 8-bit mixers take MTPLX's eager HC read (its fused
  reads take bf16 only). HC8 holds in the prompt prefill too (stored weights). top-8 follows Fit12's TK1 hook: verify
  rows only, top_k 8 on the trunk's MoE blocks with an 8-digit plan (44333333). The prompt prefill and the MTP drafter
  keep top-10 at the plan padded to 10 (4433333333), so a top-8 arm's prompt state is v3's.
- Score (sticky/s5s9.py): (1) the Eval suite's strict per-part rules vs ar4 at the same positions, as rule125.py /
  ugb_rule.py: 90 % cluster-CI upper bounds of the mean KLD ratio and of the P99.9 ratio <= 1.10, the lower bound of the
  top-1 agreement change >= -0.5 points, in every part; (2) each arm's mean KLD as x of v3 at the same positions, and
  HC8's share twice (v3 + HC8 over v3; both over top-8).
- PRE-REGISTERED (NewIdeas' lines, written down here before the run):
  - S5: HC8 is NOT FREE if v3 + HC8 over v3, averaged over the four parts, is > 1.05, or any part's 90 % CI lower bound
    is > 1.05. Then it stays a 12 GB-only setting.
  - S9: top-8 FAILS if p44333333 fails the strict rules in any part. Then it can only be an opt-in judged by the task
    bar (Eval suite, after Thu).
  - Both pass: two near-free main-track pieces. Adopting either is Kamen's call.
- Smoke (one code sequence, 4 arms): v3 run after an HC8 arm must equal turn18's v3 bit for bit; HC8 and top-8 must
  differ from it; all KLDs finite; 196 modules swapped (MoonShots' hc8_smoke on this model).
- Tests (CPU): tests/test_hc8_switch.py (tiny real GatedResiduals + PLE-shaped projections + an mtp mixer: build =
  the pinned install's swaps, off = bit for bit the loaded tree, on = plain install bit for bit) and tests/test_s5s9.py
  (synthetic data: HC8 x1.02 FREE / x1.08 NOT FREE; top-8 x0.77 of ar4 PASS / x1.38 FAIL): ALL OK. s5s9.py on
  turn18's real v3 reproduces rule125's line exactly (gsm x0.52 [..0.59], P99.9 x0.38 [..0.49], ...).
- Prior (not a result): v3 sits at x0.51-0.64 of ar4 per part, so top-8 has ~1.7-2.1x of headroom on the mean rule; the
  tight one is P99.9 on code (v3's upper bound already 1.03 of the 1.10 line). Leap's L1-17: top-8 = 0.78x MSB-3's
  MoE-block error.
- turn21 started 08:23 Mon (lock to 09:12). Smoke (one code sequence, 274 positions): v3 after an HC8 arm == turn18's v3
  bit for bit (so v3 comes from turn18; 3 arms in the batch); HC8 x1.23 of v3, top-8 x3.05, both x2.54 on that one
  sequence; all finite. Full batch from 08:25 (35 sequences x 3 arms, ~38 min).
- RESULT (turn21, read-out data/s5s9_plan4433333333_mtp_s5s9.txt, 09:43 Mon): S5 HC8 NOT FREE; S9 top-8 FAILS.
  Coverage: 27 of 35 sequences, 6,094 positions per arm (gsm 7 of 15; ifeval 8, code 6, multi 6 = complete). gsm/620
  was dropped by the driver (2 spin gates timed out: the Mac slept 08:45:49-09:00:26, lid closed + 'Thermal Emergency
  Sleep'); the orchestrator's 09:07 lid hold froze the job in sequence 29; its time limit had passed, the 27 sequences
  were saved at 09:07:11 (checked readable, all finite), so I ended the frozen process at 09:43 and turn21.sh ran the
  read-out and released the lock. v3 = turn18's arm at the same (sequence, position) (smoke: bit for bit).
  - x of v3 (mean KLD, same positions, 90 % cluster CI):
      v3 + HC8          gsm x0.997 [0.914, 1.071] | ifeval x1.061 [1.000, 1.205] | code x1.244 [0.810, 1.616] |
                        multi x1.066 [1.031, 1.102] | mean of the parts x1.092
      v3 + top-8        gsm x1.49 | ifeval x2.39 | code x2.51 | multi x2.52 | mean x2.23
      v3 + HC8 + top-8  mean x2.29; HC8 on top of top-8 x1.033 (gsm x1.10, multi x1.06)
  - Strict rules vs ar4: v3 PASS (as turn18); v3 + HC8 FAIL (ifeval top-1 -0.2 [-1.4..] pt; code P99.9 x0.76
    [..2.91]); v3 + top-8 FAIL in ifeval (KLD x1.53 [..1.68], P99.9 [..2.03], top-1 -1.7 [-2.0..]), code (KLD [..1.71],
    P99.9 x2.82 [..3.38]) and multi (KLD x1.38 [..1.54], top-1 -1.1 [-2.4..]); gsm passes; both FAIL.
  - VERDICTS (pre-registered): S5 HC8 = x1.092 of v3 over the parts (> 1.05; highest part lower bound x1.031) -> NOT
    FREE: stays a 12 GB-only setting. Robust to the noisy code part: without it the mean is x1.041, but multi is above
    1 with its whole CI and HC8 alone breaks two strict rules. S9 top-8 FAILS the strict rules -> an opt-in only,
    judged by the task bar (Eval suite). Adopting anything stays Kamen's call.

## S10b long-lead expert predictor, offline fit (NewIdeas Sun 15:27; BigMoE b31 trace, Mon 08:22) -- design + pre-registration
- Question: can a learned per-layer predictor of layer T = L + 2's experts, from layer L's MoE input, recall >= 65 % of
  the misses (routed experts not resident at the 24 GB setting) in its top-16 on held-out prompts? Prefetch only, so
  decode stays exact. Published direction: ProMoE's stride predictors, Pre-gated MoE; inter-layer expert affinity.
- Data: BigMoE b31 (~/BigMoE/data/b31/s10b/, code 6809e3e): the shipped 24 GB v3 decode (JOINT, MTP, LFU 0.99, 25 %
  resident) over code / prose / edit / math x 5 prompts x 400 tokens; layer L's MoE input (bf16) at L = 2, 8, ..., 44,
  every layer's routed ids, miss flags per (pass, layer), routers of T = L+1..L+3. 3,838 verify passes, 7,017 rows per
  layer (BigMoE's check: 567 unequal-row passes skipped, no union mismatches, no missing records).
- Baseline (BigMoE's s10b_check, all prompts): the plain router of T on x_L, top-16 per row, recalls 57.6 / 49.1 / 46.4 %
  of the misses at lead 1 / 2 / 3 (code 60.4 / 53.0 / 49.9). Note: that is far above the 27 % lead-2 code recall b30
  measured for the engine's own forecaster lists, so the definitions differ (top-16 per row here); the kill line stays
  NewIdeas' absolute one.
- Models (sticky/s10b_fit.py; numpy on CPU, 4 BLAS threads): M0 plain router; M1 = a * router_T(x_L) + z_L dW + b
  (linear on top of the router, from zero); M1a = M1 + an affinity table over the 10 experts routed at L; M2 (small MLP,
  hidden 512) only if the best linear model is < 65 % on validation, and only picked if >= 1 point better there. Softmax
  cross-entropy on T's 10 routed ids; AdamW, batch 256; grid lr {3e-4, 1e-3, 3e-3} x wd {0.1, 10}, <= 10 epochs.
- Split by PROMPT: selection on lead 2, train prompts 0-2 of each domain, validate on prompt 3 (pooled over the 8
  layers). Judge (primary): refit on prompts 0-3, score prompt 4 of each domain (never used in selection). Secondary:
  leave-one-prompt-index-out (5 folds, pooled).
- Score: recall@16 of the misses, s10b_check's definition (top-16 over all 512 experts per row; misses = the row's routed
  ids at T flagged non-resident in that pass), summed over rows, layers, domains. Lead 2 decides; leads 1 / 3 beside.
- PRE-REGISTERED (NewIdeas' lines, written here before the run on the real trace): primary held-out lead-2 recall@16 of
  the misses < 55 % -> KILL; >= 65 % -> engine A/B (adopt at >= +5 % code tok/s, shas identical); in between: alive,
  NewIdeas decides. Code is reported beside (the domain that needs >= 2 layers of lead).
- Tests (CPU): tests/test_s10b_fit.py (a synthetic trace in b31's layout, with a planted linear drift between layers):
  loader counts right, MTP-only seqs and sidecar records skipped, M0 = an independent s10b_check-style count at leads
  1-3 exactly, the fit learns the drift (held-out lead 2: 88.4 % vs M0 47.5 %): ALL OK.
- RESULT (Mon 09:01; data/s10b_fit.txt/.json, log logs/s10b_fit.out, 1498 s CPU): ALIVE, BELOW THE A/B LINE.
  Join: BigMoE's rule (the FIRST call per (pass, layer); MTPLX's commit re-runs layer 1 over the kept rows of a partly
  accepted verify under the same seq): 4,405 passes, 8,834 rows per layer, 567 re-run calls skipped, nothing else lost.
  (A first pass that kept the LAST call lost those 567 passes as unequal-row passes, exactly as BigMoE's first checker
  did: 3,838 passes; there M2 held-out lead 2 = 52.7 % vs M0 48.1 % (KILL side), leave-one-out 54.7 % vs 49.1 %;
  kept as data/s10b_fit_first3838.*. The pre-registered verdict is the all-pass run below.)
  M0 on all prompts: 58.5 / 50.2 / 47.7 % at lead 1 / 2 / 3 (3,838-pass run: 57.6 / 49.1 / 46.4 = BigMoE's checker).
  Selection (validation = prompt 3, lead 2, M0 50.9 %): linear models top out at 53.2 % (M1 lr 1e-3 / 3e-4, wd 10);
  M1a (+ affinity table) the same 53.1; M2 (MLP 512) lr 3e-3 wd 10 reaches 55.6 % at epoch 10 (still rising slowly:
  the corner of the grid) -> picked M2 (>= 1 point over the best linear).
  JUDGE (held-out prompt 4 of each domain), recall@16 of the misses, M2 vs M0:
    lead 2: 55.6 % vs 49.6 % (20,323 misses) | code 61.1 (55.4) | prose 52.3 (51.1) | edit 40.4 (41.6) | math 62.1 (47.7)
    lead 1: 52.0 % vs 58.9 % (WORSE) ; lead 3: 51.8 % vs 47.2 %
    lead 2 by layer: L2 55.4 (50.5) | L8 66.0 (60.8) | L14 39.3 (50.5) | L20 61.7 (46.5) | L26 64.4 (64.9) | L32 56.3
    (36.9) | L38 30.8 (33.3) | L44 43.6 (42.4)
  Leave-one-prompt-index-out (5 folds pooled): lead 2 57.2 % vs 50.2 % (code 58.7 vs 53.8, math 63.6 vs 49.5).
  -> 55.6 % sits between the kill line (55) and the A/B line (65): pre-registered outcome 'alive, NewIdeas decides'.
- Reading: the predictor learns the ROUTED set much better (lead-2 recall@16 of all routed ids 76.0 % vs 58.8 %), but
  its 16 slots go to popular experts, most of them resident, so the misses gain only ~6 points (and lose ~7 at lead 1,
  where the plain router is already good). The objective (all routed ids) is not the score (misses). Gains are uneven
  by layer (L20 +15, L32 +19; L14 -11) and by domain (math +14, code +6, edit -1).
- Not tested here (would change the score, not the predictor): the engine knows what is resident, so at prefetch time it
  can skip resident experts; the deployable score is recall of the misses by the top-16 NON-resident predictions. The
  trace has residency only for served experts, so that needs a resident bitmap per (pass, layer) (~14 MB); asked BigMoE.
- EXPLORATORY next (not pre-registered, labelled as such): the same models / grid / split / judge trained on each row's
  MISSED ids only (S10B_TARGET=miss), i.e. the objective the score counts; cache-dependent (24 GB, LFU 0.99).

## S10b on b32: the DEPLOYABLE score (NewIdeas 10-26 09:46 GO) -- pre-registration (written before the run, 13:50)
- Data: BigMoE b32 (~/BigMoE/data/b32/s10b/, code 3eb8ada): b31's capture (same 20 prompts and engine settings) plus
  res.bin, pool T's residency (T = L+1..L+3) at each base layer L's prefetch decision. 4,363 passes, 8,861 rows per layer,
  605 layer-1 re-runs skipped; BigMoE's checker clean (no unequal rows, no union mismatches, every scored (pass, L -> T)
  has a residency record).
- PRIMARY score (NewIdeas): DEPLOYABLE recall@16 = per row, the top-16 of the model's scores among the experts NOT
  resident in the (pass, L -> T) snapshot (the engine skips resident experts); recall of the row's misses at T
  (s10b_check's definition; misses resident at the decision but evicted before T stay in the denominator and cannot be
  caught). M0 (plain router) on the same basis must reproduce BigMoE's checker: held-out 75.3 / 63.9 / 63.1 % at
  lead 1 / 2 / 3, all prompts 74.9 / 64.1 / 63.5 %.
- Beside it (BigMoE's ask, matched budgets): per row at k = 2 / 4 / 8 / 12; per pass and target layer (the union over the
  pass's rows = what the SSD reads): recall of the pass's misses, unique reads, reads never routed at T in the pass,
  precision, at k = 2 / 4 / 8 / 12 / 16 and all12 (top-12 over all 512, resident skipped; for M0 at lead 1 = the shipped
  engine's policy, BigMoE's engine12: 51.2 % with 4.8 reads per pass-layer). M0 must reproduce check2.txt's lines.
- Models, grid, split, judge, leave-one-out: as the b31 run above, with two changes made before seeing any b32 fit:
  selection (validation = prompt 3, lead 2) uses the deployable recall@16, the decision metric (b31's top-16 over all
  512 printed beside it); M2 is always tried (the plain router alone sits at ~64 % here, so the 65 % gate would stop
  the search at the line it is judged on), and is picked only if >= 1 point better than the best linear model.
- PRE-REGISTERED (NewIdeas' lines): held-out lead-2 deployable recall@16 >= 65 % -> engine A/B (BigMoE); 55-65 % ->
  NewIdeas' call; < 55 % -> stop S10b. Leave-one-out, per domain, leads 1 / 3 and the budget lines beside it. Since M0
  alone is at 63.9 %, the A/B line needs only a small gain; whether a pass would pay on an SSD that already runs ~93 %
  busy is read from the budget lines (recall at matched reads, precision at k <= 4; BigMoE 13:16): NewIdeas' call.
- Label caveat (BigMoE 13:16, data/b32/budget_note.txt): a 'miss' = not in the pool table when T is served; a
  prefetch that lands on time joins the table only at T's service, so on-time prefetches are labelled misses too
  (~1/4 in the Oct-19 engine runs; b32 did not save the split). M0 and the fit are scored on the same labels; a
  late + demand-only re-score needs BigMoE's b33 (NewIdeas' call).
- EXPLORATORY (b), not pre-registered, labelled as such: the same pipeline with S10B_TARGET=miss (trained on the row's
  missed ids only). It cannot change the verdict alone; if it beats (a), a pre-registered re-run on a fresh capture.
- Tests (CPU): tests/test_s10b_fit.py now builds b32's layout (res.bin with the MTP pool, nT = 0 at L = NL, passes
  without records, a later all-resident record after each re-run pass): M0's deployable recall at k = 2..16 (all prompts
  and held-out) and the read budget (recall, reads, unused, misses, precision; k2..k16 and all12, leads 1-3) equal an
  independent s10b_check-style count; the fit learns the planted drift (held-out lead 2 deployable 93.4 % vs M0 58.2 %);
  without res.bin the script falls back to b31's score. ALL OK for both targets (routed, miss).
- Run: data/s10b_fit_b32.txt/.json and data/s10b_fit_b32_miss.txt/.json; logs logs/s10b_fit_b32*.out.
- AMENDMENT 13:50, still before any b32 fit (NewIdeas 13:46, QUEUE row S10b: "the bar re-anchored to reads"): the
  DECISION metric is now the per-pass catch of the misses at the shipped budget, ~4.8 unique reads per pass-layer, lead
  2, held-out prompt 4. Computed from the top-k-among-non-resident sweep (k = 1, 2, 3, 4, 5, 6, 8, 12, 16; union over
  the pass's rows), linear in reads between the two budgets around 4.8 (NewIdeas' "router alone ~34 %" is this
  interpolation of k2 / k4). Lines: >= 50.4 % (the shipped lead-1 policy's held-out catch at that budget, so the MLP
  could REPLACE the lead-1 reads, a layer earlier) -> pass: b33 labels (late + demand), then an engine A/B design;
  < 45 % -> park S10b; between -> b33 labels decide. Selection (validation prompt 3, lead 2) uses this catch. Beside it:
  the shipped rule's family (top-K over all 512, resident skipped, K = 4..64) at 4.8 reads, leave-one-out, per domain,
  leads 1 / 3, the k2 / k4 / k8 lines (recall, reads, precision; BigMoE's ask), and the per-row deployable recall@16
  (the 09:46 lines 65 / 55 %, now context only). Test extended: M0's catch at 4.8 reads in both families (all prompts
  and held-out) = an independent interpolation of an independent sweep; ALL OK for both targets.
- RESULT (Mon 13:55-14:00, 5 min CPU; data/s10b_fit_b32.txt/.json, logs/s10b_fit_b32.out): PRE-REGISTERED VERDICT
  PARK (missed 45 % by 0.6 point), BUT THE ENGINE'S OWN PICK RULE CLEARS THE 50.4 % BAR: NewIdeas' call.
  M0 reproduces BigMoE's checker exactly (deployable held-out 75.3 / 63.9 / 63.1 %; every check2 budget line, incl.
  engine12 = all12 at lead 1: 51.2 | 50.4 % at 4.8 reads, precision 22.6 %).
  Selection (validation prompt 3, lead 2, catch at 4.8 reads, M0 37.0 %): M1a lr 3e-4 wd 10, 5 epochs 46.7 % (M1 46.6;
  M2 46.7 at lr 3e-3 wd 10 ep 8, not >= 1 point better) -> M1a.
  JUDGE (held-out prompt 4), lead 2, per pass at 4.8 reads per pass-layer:
    top-k among non-resident (PRIMARY): M1a 44.4 % vs M0 32.9 % -> < 45 % -> park S10b; leave-one-out 46.2 vs 35.2.
    the shipped rule (top-K over all 512, resident skipped; secondary): M1a 53.3 % vs M0 36.5 %; leave-one-out 55.6 vs
    39.0. Without interpolation: M1a's top-16 over all 512 catches 51.9 % with 4.31 reads per pass-layer, where the
    shipped lead-1 policy catches 50.4 % of the lead-1 misses with 5.14 reads on the same held-out passes (48.4 % at
    4.8 in its own family: the 50.4 % bar is its catch at 5.14 held-out reads).
    BigMoE's lines, lead 2 held-out, M1a vs M0: k2 38.4 vs 28.4 % (3.7 reads, precision 28.5 vs 21.0 %); k4 54.3 vs
    41.4 % (7.1, 21.1 vs 16.0 %); k8 68.0 vs 54.3 % (13.5, 13.8 vs 11.0 %); all12 44.6 % with 2.8 reads, precision
    43.4 % (M0 43.2 % with 6.6 reads, 18.1 %).
    Lead 1 at 4.8 reads: 44.5 vs 42.1 % (top-k), 50.2 vs 48.4 % (shipped rule); lead 3: 42.9 vs 31.5, 51.0 vs 34.9.
    Lead 2 by domain (top-k, 4.8 reads): code 39.3 (31.3), prose 41.0 (33.5), edit 40.6 (28.4), math 52.8 (35.7).
    Per-row deployable recall@16 (the superseded 09:46 score): 75.9 vs 63.9 % (code 79.4, prose 70.5, edit 68.8, math 80.5).
  EXPLORATORY miss-only targets (data/s10b_fit_b32_miss.*): worse everywhere (lead 2 held-out at 4.8 reads 38.7 %
  top-k, 43.4 % shipped rule; picked M1a at 1 epoch): training on the misses alone does not help; no re-run.
- Reading: the linear correction (+ affinity table) makes the top of the list precise (lead 2, shipped rule at 2.8
  reads: 43 % precision vs the router's 18 %). 'Top-k among non-resident' spends k reads per row even when a row's best
  candidates are resident; the shipped rule reads only experts that rank high overall, so at a fixed budget it catches
  more (53.3 vs 44.4 %). Caveats: the shipped-rule family was pre-registered as secondary; ~1/4 of the 'misses' may be
  on-time prefetches (label caveat); b33 would relabel the same passes, not new prompts. Next is NewIdeas' call: park
  per the primary line, or b33 (late + demand labels) + an engine A/B design on the shipped rule, model frozen.

## S10c-1: frozen M1a on b33's late + demand misses (NewIdeas 10-26 14:08, QUEUE R-08 / S10b row) -- pre-registration (written before the run, Tue 19:25)
- Question: does the S10b predictor's edge survive when only the misses a long lead can help count? b32's 'miss'
  included prefetches that landed on time. BigMoE's b33 (10-26 16:16-16:24, ~/BigMoE/data/b33/s10b, b33_note.txt)
  re-ran the same 20 prompts (exact: output shas = b32) with labels per miss: kind 0 demand, 1 on time, 2 late,
  3 queued (miss.bin entries e | miss << 16 | kind << 17). Late + demand = miss and kind != 1 (queued counts with
  demand), as BigMoE's `S10B_LABEL=ld s10b_check.py`. b33 is scored on its own x, residency and labels (no join to b32:
  pass boundaries depend on timing).
- Model FROZEN, no refit (NewIdeas 14:08): b32's pick, M1a (linear on router_T + affinity table), lr 3e-4, wd 10,
  5 epochs, trained on b32 with S10b's splits and seeds (judge: b32 prompts 0-3 of each domain; leave-one-out: the
  other four prompt indices), i.e. S10b's own models; standardization from b32's training rows. Each model is
  scored on b33's rows of the prompts it never saw (held-out prompt 4; leave-one-out folds pooled = all 20 prompts).
  No selection and no other recipe on b33.
- Score (PRIMARY): lead 2, per pass and target layer, the catch of the pass's late + demand misses at 4.8 unique reads
  per pass-layer under the SHIPPED RULE's family (top-K over all 512 experts, resident skipped; K = 4..64), linear in
  reads between the two budgets around 4.8 (S10b's definition); M1a minus the plain router (same family, same
  interpolation, same passes).
- Lines (NewIdeas 14:08): PASS: M1a >= router + 10 points -> (2) engine A/B (BigMoE designs and pre-registers it).
  KILL: < +5 points (its edge was on-time prefetches; S10b / S10c closed). Between: (2) only if BigMoE prices the
  integration as small (NewIdeas' call). Held-out prompt 4 decides; leave-one-out must agree: PASS needs both >= +10,
  KILL needs both < +5, anything else is 'between'.
- Beside it (no verdict): the top-k-among-non-resident family at 4.8 reads; leads 1 / 3; BigMoE's k2 / k4 / k8 lines
  (recall, reads, precision); per-row deployable recall; domains; the kind split.
- Checks before reading the verdict: (a) the plain router on b33 must reproduce BigMoE's check_ld.txt: deployable
  lead 2 60.8 | 60.7 % (all | held-out); budget lead 2 k2 25.7 | 25.2 % at 3.6 reads, k4 38.5 | 38.1 % at 6.9,
  k8 51.7 | 51.3 % at 13.2, k16 64.4 | 64.4 % at 24.9; lead 1 engine12 (all12) 43.6 | 43.4 % at 4.9; all-512 recall@16
  of misses lead 2 46.8 %. (b) The frozen models on b32's own held-out passes must reproduce S10b (lead 2 at 4.8
  reads: shipped rule 53.3 % vs 36.5, top-k 44.4 vs 32.9; leave-one-out 55.6 vs 39.0, 46.2 vs 35.2), proving they are
  S10b's models. If (a) or (b) fails, no verdict until the cause is found.
- Cost: one CPU screen <= 10 min including both loads (4 BLAS threads, nice 10), outputs only (~KB).
  sticky/s10b_fit.py gains S10B_LABEL=ld, S10B_FROZEN=M1a,3e-4,10,5 and S10B_TRAIN=DIR (train there, score TRACE_DIR);
  the test grows a b33-like copy of its synthetic trace (labels with kinds, passes in another order) and checks the
  label filter against an independent count and the cross-trace scoring against the same-trace frozen run.
- RESULT (Tue 19:45:54-19:47:59 under my quiet lock, 125 s CPU; data/s10b_fit_b33ld.txt/.json, logs/s10b_fit_b33ld.out):
  PASS, +18.8 points held-out and +18.9 leave-one-out (line +10).
  Checks: (a) the router on b33 reproduces BigMoE's check_ld.txt on every line (deployable all | held-out
  70.2 | 70.6 / 60.8 | 60.7 / 60.4 | 60.3 at leads 1 / 2 / 3, domains identical; every budget line incl. lead-1 all12
  43.6 | 43.4 % at 4.9 reads; all-512 recall@16 51.5 / 46.8 / 44.9, held-out 52.2 / 46.2 / 44.6). (b) The frozen models
  on b32's own held-out passes give S10b's numbers exactly (shipped rule 53.3 vs 36.5, top-k 44.4 vs 32.9; leave-one-out
  55.6 vs 39.0, 46.2 vs 35.2): they are S10b's models. b33 kinds over the trunk records: demand 298,424, on time
  75,064 (13.6 %), late 115,780, queued 61,277 (= BigMoE's pf_qdemand exactly). 4228 passes, 8802 rows (as BigMoE).
  PRIMARY (lead 2, late + demand misses, shipped rule at 4.8 reads per pass-layer): held-out M1a 52.5 % vs router
  33.7 % (+18.8); leave-one-out 54.3 vs 35.4 (+18.9). On all misses (b32) the gap was +16.8: the edge did not come
  from on-time prefetches (the relabel cost the router 2.8 points, M1a 0.8).
  Without interpolation (held-out): M1a's top-16 over all 512 (resident skipped) catches 51.2 % with 4.34 reads per
  pass-layer (precision 29 %), its top-12 43.8 % with 2.85 (37 %); the router's top-12 40.6 % with 6.61 (15 %). The
  shipped lead-1 policy on the same held-out passes: 43.4 % of the lead-1 late + demand misses with 5.23 reads.
  Beside it (held-out): top-k among non-resident at 4.8 reads 42.9 vs 29.6 % (LOO 43.9 vs 30.9); shipped rule at
  4.8 reads lead 1 47.3 vs 41.0, lead 3 49.5 vs 32.5; per-row deployable recall@16 lead 2 74.5 vs 60.7 % (code 77.6,
  prose 69.1, edit 67.4, math 79.6); per pass k2 / k4 / k8: 36.6 / 53.0 / 66.9 % vs 25.2 / 38.1 / 51.3 % at ~3.7 /
  7.1 / 13.5 reads (precision 24 / 18 / 12 vs 17 / 13 / 9 %); deployable lead 2 by layer: M1a >= the router at all
  8 predictor layers held-out (L14 level in leave-one-out, 64.2 vs 64.4).
- Reading: S10c-1 passes its pre-registered line by a wide margin, so the next step is (2), BigMoE's engine A/B
  (M1a's lead-2 picks replace the lead-1 router picks at the same rule and budget; exact; NewIdeas' line +5 % tok/s
  on code / prose, KILL <= +2 %). Engine cost: M1a folds into one linear map per (L, T):
  s = x (a W_T + diag(1/sd) dW) + (b - (mu/sd) dW) + sum of A[e] over the 10 experts routed at L, i.e. the same
  [2560 x 512] matvec as today's router forecast plus a 10-row gather-sum. Caveats: an offline proxy (catch of
  labelled misses; reads = unique non-resident experts per pass-layer; no timing, no SSD queueing); b33 runs the same
  20 prompts as b32 (other passes, same tokens); models exist only for the traces' 8 predictor layers (x at L = 2, 8,
  ..., 44), so an A/B over all layers needs x at every layer (a new capture) or runs on these 8 (L -> L+2) pairs.

## R-22 M1aB: M1a's read-budget curve on b33's late + demand misses (NewIdeas Tue ~19:58) -- a read-out, not a new run (20:02)
- Line (NewIdeas, QUEUE R-22): PASS if M1a (lead 2, the shipped rule: top-K over all 512, resident skipped) matches
  the shipped lead-1 policy's 43.4 % catch at <= 3.0 unique reads per pass-layer (then BigMoE's A/B also runs at that
  budget); KILL if it needs >= 4.5. The curve already existed: the S10c-1 run (19:47) computed it as pre-registered
  'beside' lines (data/s10b_fit_b33ld.json, budget all4..all64), and its K = 12 point was in my 19:50 note to BigMoE
  and REPORT 'S10c-1'. No new data exists to test it afresh (b33 is the only late + demand trace; b32 trained the models).
- Held-out prompt 4, per pass-layer: catch | reads | wasted (never routed at T in the pass) | precision:
  M1a lead 2: K=8 32.1 % | 1.62 | 0.70 | 48.3 %; K=12 43.8 % | 2.85 | 1.60 | 37.5 %; K=16 51.2 % | 4.34 | 2.88 |
  28.8 %; K=24 61.3 % | 7.91 | 6.16 | 18.9 %.
  The shipped lead-1 policy (router, K=12, lead 1): 43.4 % | 5.23 | 4.06 | 16.3 %.
  The router at lead 2 (M1a's targets): K=12 40.6 % | 6.61 | 5.38 | 15.0 %; K=16 47.8 % | 9.65 | 8.21 | 12.1 %.
- -> PASS: M1a reaches 43.4 % at 2.81 reads per pass-layer (linear between K=8 and K=12; the real K=12 point catches
  43.8 % with 2.85), against 5.23 for the shipped policy (-46 % reads) and ~7.8 for the router at lead 2. Wasted reads
  1.6 vs 4.1 per pass-layer (-61 %). Leave-one-out: the shipped policy's 43.6 % (4.88 reads, 3.77 wasted) is matched
  at 2.66 reads; K=12 catches 45.4 % with 2.85 (1.59 wasted).
- Caveats: a read-out of numbers that existed before the line (see above); the shipped policy's targets are the
  lead-1 layers (T = L+1) and M1a's the lead-2 layers (T = L+2), same passes and label rule; reads = unique
  non-resident experts in the pass's picks, ignoring overlap with the lead-1 prefetch, timing and queueing. For
  BigMoE's A/B: K = 12 is this budget (~2.85 reads per pass-layer); K = 16 (~4.34 reads, 51.2 %) is about the shipped one.

## R-26 DYNK-Q: strict / KLD screen of dynamic top-k at tau 0.06 (NewIdeas Tue ~20:15; MoonShots' R-25) -- pre-registration (written before the run, 20:19)
- Rule (MoonShots' R-25, NON-exact): each verify row drops its rank-8 / rank-9 expert (top-10, rank order) only when
  that expert's top-10-renormalized gate weight is < tau; the kept experts are renormalized. tau 0.06 at the exact
  24 GB stack: -13.2 % SSD bytes (82.0 MB / token) for 4.7 % of the gate mass (fixed top-8: -25.7 %, 12.1 %).
- Harness = S9's (turn21): sticky/budget125_tf.py, TL_MTP=1, v3's tiers (plan4433333333), R = 4 verify rows,
  teacher-forced on the suite's tf sets, scored against ar4; the prompt prefill and the MTP drafter keep top-10.
  Arm q060 = v3 + DYNK 0.06 through the router logits (sticky/dynk.py): a row that drops gets -inf outside its top-10
  and (row max - 80) at the dropped ranks, so the suite's block keeps its own softmax / top-10 / renormalisation / rank
  tiers and gives the dropped experts ~1e-35; a row that keeps all 10 passes its logits through bit for bit.
  tests/test_dynk.py (CPU: against the suite's block routing and an independent float64 rule) must be ALL OK first.
- Size (one 10-min screen, turns/turn22.sh under run_locked.sh 10): S9's sequences (every 4th) of the three parts
  where fixed top-8 FAILED: ifeval (8), code (6), multi (6) = 20 sequences, parts interleaved; gsm (top-8 passed it,
  x1.49 of v3) is left out to fit; the driver's time limit may cut the last ifeval sequences.
- Gate = S9's: the Eval suite's strict per-part rules vs ar4 (90 % cluster-CI upper bounds of the mean KLD ratio and
  of the P99.9 ratio <= 1.10; lower bound of the top-1 agreement change >= -0.5 points), read-out sticky/s5s9.py.
- Lines (NewIdeas, QUEUE R-26): PASS: q060 passes the strict rules in ifeval, code and multi, all three of the parts
  where fixed top-8 failed (then gsm and a longer confirmation; only a PASS earns one). KILL: it fails in any of them
  (then only a task-test bar could carry it, like top-8 at 12 GB: Kamen's decision 2). taus 0.05 / 0.07 do not fit
  the 10 minutes: not run.
- Beside it (no verdict): mean KLD as x of v3 per part (v3 = turn18's arm at the same positions; since turn21's smoke
  the engine's repos changed only BigMoE's prefetch path, stream.py + m1a.py behind BIGMOE_M1A, not re-verified bit for
  bit here); the measured drop rates and dropped gate mass per verify row-layer (MoonShots: 4.7 % of the mass).
- AMENDMENT 20:31 (before any R-26 run): the first hook (router logits) failed tests/test_dynk.py: bf16 score ties
  re-ordered the kept experts vs v3 in ~13 % of the dropping rows (tiers could move); the lock was released unused.
  The drop now runs inside the suite's block on v3's own sorted scores (dynk.block_dynk = the suite's sorted block line
  for line + dynk_scores; trunk layers only), and the run adds a smoke on its first sequence: q000 (the DYNK block with
  nothing dropped) must equal turn18's v3 bit for bit. Lines, harness and size unchanged.
- RESULT (turn22 under my lock 20:33:02-20:40:28, all 20 sequences, 443 s; read-out 20:44 at the start of the next
  lock, back-to-back quiet windows kept it out before; data/s5s9_plan4433333333_mtp_r26.txt,
  budget125_tf_R4_plan4433333333_mtp_r26*.json / _pos.npz): KILL: tau 0.06 FAILS the strict gate in all three parts.
  Smoke PASS: q000 at 505 positions = turn18's v3 bit for bit (the block copy and the engine are v3's); q060 differs
  there by mean |dKLD| 6.2e-3. DYNK measured over 226,368 verify row-layers: rank 9 dropped in 47.4 %, ranks 8 + 9 in
  36.1 %, 0.84 experts and 4.21 % of the top-10 gate mass per row-layer (MoonShots' replay: 4.7 %).
  Strict rules vs ar4 (90 % cluster CI): v3 PASS in all three (turn18's arm, as S9). v3 + DYNK 0.06 FAIL:
    ifeval KLD x0.91 [..0.99] ok, P99.9 x0.87 [..1.22] FAIL, top-1 -0.3 [-0.9..] pt FAIL;
    code   KLD x1.01 [..1.22] FAIL, P99.9 x2.31 [..2.38] FAIL, top-1 +0.1 ok;
    multi  KLD x0.92 [..1.04] ok, P99.9 x0.90 [..2.05] FAIL, top-1 -0.2 [-1.3..] pt FAIL.
  Mean KLD as x of v3: ifeval x1.43 [1.34, 1.57], code x1.98 [1.45, 2.33], multi x1.68 [1.53, 1.79], mean x1.70
  (fixed top-8 on the same three parts, S9: x2.39 / x2.51 / x2.52, mean x2.47).
- Reading: the threshold drop keeps 35 % of top-8's dropped mass (4.2 vs 12.1 %) but costs ~half of its extra KLD
  (x1.70 vs x2.47), and it breaks the same tail rule on code (P99.9 x2.31): low-weight rank-8 / 9 experts still carry
  the tail. Per the pre-registered line, only a task-test bar could carry it (Kamen's decision 2, as top-8 at 12 GB).

## R-29 LAYSZ-ON: online per-layer pool sizes at 24 GB (NewIdeas Tue ~20:30; MoonShots' R-27) -- pre-registration (written before the run, 20:40)
- MoonShots' strict bound (20:3x, data/laysz_159_44_lo15_r16a10_lfus.txt / _r4a3_): static held-out -10.3 % (16/10)
  / -8.9 % (4/3); self-calibrated on the scored set (in-sample ceiling of a static allocation) -14.7 % / -13.6 %, i.e.
  4.4 / 4.7 points beyond static on the mean (code / math only 2.2-3.5): not below NewIdeas' 4-point bar, so it runs.
- Online rule (sticky/laysz_online.py): pools start uniform 159 / 44; at token 400 and every 200 tokens after (400,
  600), each layer's (C_m, C_l) is re-fit by R-27's Lagrangian (same RAM, same grid C_m 9-339 / C_l 8-104) on the
  trace's OWN miss curves seen so far (fixed-size strict-LFU replays at every grid size over the checks fully inside
  [200, t)); the pools resize at check boundaries (a shrink drops the lowest LFU scores, the strict rule's own trim).
  Scored like R-27: SSD bytes per token over checks starting in [400, 800), 24 GB strict LFU, both shapes (16/10, 4/3).
  Replay = MoonShots' (read-only import of wall24.py; laysz24.py's run_strict, constants and Lagrangian copied, since
  it writes at import). Check first: my static held-out and self-calibrated per-set numbers must equal MoonShots'.
- Lines (NewIdeas, QUEUE R-29): PASS: online >= 4 points beyond static held-out (mean of the 4 sets) in both shapes;
  KILL: < 2 points in both; else between (NewIdeas' call). CPU, one core (taskpolicy -b), outside others' quiet locks.
- RESULT (20:35:53-20:38:26, 152 s, one core; data/laysz_online.txt, logs/laysz_online.out): KILL. The online re-fit
  is WORSE than static: -4.6 points (16/10) and -2.1 points (4/3) beyond static held-out.
  Check: static held-out and self-calibrated reproduce MoonShots' strict numbers on every set in both shapes (16/10
  static -15.8 / -16.8 / -4.1 / -4.7, mean -10.3 %; self -22.3 / -22.1 / -7.6 / -6.9, mean -14.7 %; 4/3 static -13.6 /
  -13.6 / -4.0 / -4.3, mean -8.9 %; self -20.8 / -18.9 / -7.4 / -7.3, mean -13.6 %).
  Online, bulgarian / chat / code / math, mean: 16/10 -13.0 / -6.1 / -2.4 / -1.6, mean -5.8 %; 4/3 -13.6 / -5.6 / -4.4 /
  -3.6, mean -6.8 % (vs uniform 159 / 44). Re-fits at tokens 400 (19 checks at 16/10, 66 at 4/3) and 600; RAM within
  18.13 GB every time.
- Reading: a set's own first 200 tokens of curves (19-66 checks) predict its second half worse than three other full
  traces do; chat loses most (-10.7 / -8.0 points vs static; its trace is the shortest, 55 checks at 16/10). The
  self-calibrated gap is in-sample on the scored tokens, so it does not transfer to a causal re-fit at this trace length.
  Not tried (would be a new screen): blending the static prior with the online curves, or re-fitting only layers whose
  curves move. R-27's static allocation stays the lever.

## R-31 UNIQ-Q: strict / KLD screen of MoonShots' check-level UNIQ rule (NewIdeas Tue ~20:45; MoonShots' R-30) -- pre-registration (written before the run, 20:43)
- Rule (R-30 check-level, uniq24.py UNIQ_MODE=mass, NON-exact): in each verify check, an expert that the check's rows
  use only at ranks >= 4 is dropped from every row when its summed top-10-renormalized weight over those rows is
  < theta; the kept scores are renormalized per row. Bytes (MoonShots' 16/10 replay): theta 0.06 -19 % at 2.4 % gate
  mass, 0.08 -41 % at 7.6 %, 0.12 -53.5 % at 12.0 % (top-8's mass; with R-27 ~39.5 MB / token, under 350's 40.6).
- Harness: budget125_tf.py with 16-row verify passes of 16 consecutive TRUE tokens (teacher-forced: the proxy for real
  drafted 16-row checks, which this harness does not have), v3's tiers, KLD per position against the suite's cached
  reference and ar4; the prompt prefill and the MTP drafter keep top-10; TL_MTP off (no drafter outputs needed). Arms:
  v3 at 16 rows (p4433333333, the 16-row control), u080 and u120 (renormalized); smoke on the first sequence: u000
  (theta 0) must equal v3 at 16 rows bit for bit. The rule runs inside the suite's block (dynk.uniq_scores;
  tests/test_dynk.py: = a numpy copy of uniq24.py's rule, and theta 0 = the suite's block bit for bit).
- Size (one 10-min screen; the lock also carries R-26's read-out, ~1 min of CPU that back-to-back quiet windows kept
  out): every 8th sequence of ifeval / code / multi (10 sequences, the parts fixed top-8 failed); the time limit may cut
  the last ones. Not in this screen: theta 0.06, the not-renormalized arms, gsm.
- Gate: S9's strict rules vs ar4 per part (s5s9.py) for each arm; v3 at 16 rows must pass them itself, else no verdict
  (then x of v3 only).
- Lines (NewIdeas, QUEUE R-31): PASS: theta 0.08 passes the strict gate (then 0.12, gsm, a longer confirmation and the
  Eval suite's task tests: Kamen's bar). KILL: even theta 0.06 fails; since 0.06 is not run here, a 0.08 FAIL reads
  '0.08 FAIL, 0.06 untested' and the KILL needs the 0.06 arm (the next screen). theta 0.12 (the 350 stack's point)
  is reported beside it, with x of v3 at 16 rows. Kin, beside: the per-pass union budget at 4 rows (b14-b18, OEA k0 4-5;
  REPORT '125B per-pass expert budget') missed the strict ifeval target.
- RESULT (turn23 under my lock 20:44:35-20:54:26, 20:55): NO DATA, the screen did not fit. The first sequence
  (tf_ifeval/1005, 505 positions x 4 arms at 16 rows) had not finished when run_locked's 570 s ran out (rc 143,
  nothing saved): at 16 rows this harness costs >= 0.27 s per position and arm, >= 3.5x its 4-row cost (0.075 s in
  R-26). I sized the screen on the 4-row timing without a 16-row timing check first; the slot was spent for nothing.
  The 16-row path is the suite's reference engine with v3's tiers computed row by row ('plan variant' split), not the
  production engine, so this says nothing about 16-row speed in the engine itself.
- Options (NewIdeas' call): (a) a confirmation-size run, ~35-45 min, 16 rows, v3 + u080 + u120 on ~6 sequences;
  (b) a <= 10-min profile of the 16-row path first (where the 3.5x goes; a cheap fix would make (a) a screen again);
  (c) wait for real drafted 16-row checks (Leap's real-draft dump, MoonShots' R-30 redo).
- NewIdeas 21:0x: (b), with memory too (cap positions, stop at swap + 1 GB); a cheap fix -> theta 0.08 first, else (c).
- PROFILE (turns/turn24.sh, 20:59-21:00, ~1 min GPU, 64 positions of the first code sequence per run; swap flat at
  13.7 GB): v3 at 4 rows 0.07 s / position (192 single-row tier calls per pass), RSS 20.2 GB, MLX peak 23.8 GB; v3 at
  16 rows 0.30 s / position: every pass's 16-row expert calls take the engine's full-E scratch (48 full re-tiers of all
  512 experts per pass, no decode tier call), RSS 23.3 GB, MLX peak 25.3 GB; exact (no tiers) at 16 rows 0.26 s /
  position, RSS 23.9 GB. So the time is the engine's > 8-row path (the suite's re-tiering adds ~15 %), and memory is
  not the issue (20-25 GB in every run; the 20:54 swap came from two 20 GB processes overlapping).
- AMENDMENT (written before the screen, 21:05): fix = TL_CHUNK=8: every arm's verify passes run the expert computation
  in 8-row calls inside the block copy (dynk.block_dynk chunk=8; routing and UNIQ's check-level rule still see all 16
  rows; rows are independent in the expert sum; test: chunked = unchunked bit for bit). Gate in the same lock: 64
  positions at 16 rows chunked <= 0.105 s / position (1.5x the 4-row cost) and RSS < 30 GB, else no screen (-> (c)).
  Screen: v3 at 16 rows chunked (the control) + u080 only (NewIdeas: theta 0.08 first), every 8th sequence of ifeval /
  code / multi, turns/turn25.sh under run_locked 11. Lines unchanged (PASS: u080 passes the strict gate; the control
  must pass itself); u120 and theta 0.06 are later screens.
- GATE (turn25 under my lock 21:10:25-21:11:15; logs/r31_gate.txt): FAIL, so no screen ran. Chunked 16-row passes
  cost 0.62 s / position, 2x the unchunked 0.30. The engine picks its full-E path for the whole 16-row forward, not
  per expert call: both 8-row halves still went through the full-E scratch (96 multi-row tier calls per pass, the
  re-tier itself memoized once per layer, 48), no decode tier call. RSS 23.3 GB, swap flat.
- So, per NewIdeas' rule, (c): R-31 waits for a real-drafted 16-row quality harness; no 40-min lock on the true-token
  proxy. Not cheap, left for whoever builds that harness: forcing the engine's decode (pool) path for 16-row verify
  forwards (its > 8-row switch), or splitting a check into two 8-row forwards (not equivalent: UNIQ needs all 16
  rows' routing per layer, and rows 8-15 at layer L need rows 0-7's outputs).

## D-03 NGRAM: does the 125B's own n-gram (PLE) table carry next-token information? (NewIdeas Wed ~15:40, drafter track D98) -- pre-registration (written before any probe is fitted, Wed 15:48)
- Question: the 125B's 51B hashed n-gram table (layer 1's PLE: 320,001,536 rows x 160, 4-bit g32) gives every position
  16 rows (8 hash heads each for the 2-gram and the 3-gram ending at that token) = one 2,560-d vector. If a linear map of
  it predicts the next token clearly better than the token's own embedding does, a drafter could read the same rows (the
  verify fetches them anyway) for information a small drafter does not have.
- Data: L1-18 code (Leap): 24 HumanEval samples, ids = prompt + the 125B's greedy code (data/dfl2d_code), labels =
  data/l118_tgt (row i = the 125B's top-32 ids / log-probs for token i + 1; asst marks completion tokens).
  gsm (information): data/l120/gsm_*.npz (29 samples, same label format).
- Rows: mtplx's exact host arithmetic (_ngram_rows_np, its source exec'd from BigMoE's .venv324, not imported) with the
  checkpoint's layer_multipliers / ngram_heads_vocab_sizes / ngram_heads_offsets (layers.1.ple.ple_embedding, also
  checked against the derived values), history starting [eos, eos] (eos 248044, as at a fresh sequence); gathered from
  models/bare/ngram-table.safetensors (memmap, read-only), dequantized w * s + b per 32-group, concatenated (16 x 160).
- Arms: N = the n-gram vector of the prefix ending at token i. C (control) = token i's input embedding (embed_tokens,
  4-bit g64, dequantized). N+C (information) = both, 5,120-d.
- Probe: ridge from x (centred) to the LM-head row (lm_head, 4-bit g64, dequantized f32) of the 125B's top-1 at row i;
  lambda per arm and fold = alpha * tr(X'X) / d, alpha in {1e-3, 1e-2, 0.1, 1, 10} by 3-fold CV MSE over the training
  half's samples. Score: nearest LM-head row (Euclidean) over all 248,320 tokens; top-1 and top-4 vs the 125B's top-1.
- Halves: whole samples (sorted by name; even -> A, odd -> B), two folds (fit A score B, fit B score A), results pooled.
  Positions in one sample share n-grams, so a random-position split could memorize a document's own repeats. Fit on all
  rows of the training half (prompt + completion); score the test half's completion rows (asst of token i + 1).
- Reported: top-1 / top-4 overall and on rows where the 125B's top-1 p >= 0.9; DFlash2's first misses are not scored
  (headroom18 keeps no per-position list; that needs a GPU replay).
- Lines (NewIdeas' row, code): PASS = N top-1 >= 30 % on held-out p >= 0.9 rows AND >= C + 10 points. KILL = N top-1
  < 10 % there, or <= C + 3 points. Else BETWEEN. gsm is information only.
- Limits: CPU only (taskpolicy -b, nice 10, 4 BLAS threads), < 45 GB; one screen (<= 10 min incl. load); no lock (the
  table read is ~0.2 M random rows from the SSD, done outside others' quiet windows).
- RESULT (Wed 16:03-16:21, sticky/d03_ngram.py, logs/d03_ngram.out, data/d03_ngram.txt / .json): **KILL**.
  - Checks: the checkpoint's hash constants = the derived ones (multipliers, 16 head sizes 20000003..20000171,
    offsets); numpy dequantize = MLX's bit for bit on lm_head and table rows; the 125B's top-1 = the next token on
    99.67 % of code completion rows; code used 55,173 distinct table rows of 177,520 lookups (gathered in 1.9 s).
  - code, held-out completion rows (6,375; 5,925 with p >= 0.9), top-1 / top-4 by nearest LM-head row:
    | arm | all rows | p >= 0.9 | p < 0.9 top-1 |
    |---|---|---|---|
    | N (n-gram rows) | 5.3 / 8.4 % | **5.6** / 8.9 % | 1.3 % |
    | C (token embedding, control) | 8.7 / 11.7 % | **9.3** / 12.4 % | 1.1 % |
    | N+C (information) | 9.2 / 13.3 % | 9.8 / 14.2 % | 1.3 % |
  - Verdict: N 5.6 % vs C 9.3 % on p >= 0.9 rows (-3.7 points): KILL on both lines (< 10 %, <= C + 3). Adding the
    rows to the embedding buys +0.5 top-1 / +1.8 top-4 points.
  - gsm (information; 7,061 completion rows, 6,106 with p >= 0.9): N 14.3 % vs C 19.2 %; N+C 15.4 %.
  - Reading: the table's 2,560-d vector carries less linearly decodable next-token identity than the token's own
    embedding. In the model it enters through layer 1's gated key / value projections as knowledge for later layers,
    not as a next-token code. Caveats that don't change the verdict: the CV picked the top of the alpha grid (10) in
    most fits, so every arm might gain a little from more ridge; a non-linear readout was not tried (the 30 % line is
    5x away). DFlash2's first misses were not scored (no per-position list).
  - Cost: the code verdict took 9.0 min (inside the 10-min line); the gsm information arm another 9.2 min (the
    5,120-d N+C ridge CV ~5 min per set). Next time CV only the primary arms.

## D-05 MTPHYB + D-08 CALC walk (MTP Head; QUEUE D98 rows D-05 (moved from BigMoE) and NewIdeas' D-08 ask) -- pre-registration (written before the run, Wed 16:56)
- D-05 question: does the 125B's native MTP head beat DFlash2 per token at shallow depths (a hybrid: MTP for depths
  1-2, DFlash2 after)? Measure: the native chain (k = 3) in the shipped 24 GB v3 engine's greedy decode, every pass
  verifying exactly its 3 drafts (BigMoE's scripts/mtp_depth.py a1af5e4, copied as sticky/mtp_depth_ids.py so a prompt
  can be EvalSuite's tf prompt_ids as stored; env = mtp_depth.sh's lines). N = 300 tokens per prompt; 8 prompts per
  domain from EvalSuite tf_prompts (sorted by key): tf_code [::3], tf_gsm [::8], tf_ifeval [::4], tf_multi [::3]
  (data/d05_prompts.jsonl). Agent: Chat1000's 12 agent turns (data/agent125/*.npz: out + the stored 15-deep MTP chain
  per position), on CPU, alignment fixed by the higher depth-1 agreement of mtp[i, 0] vs out[i + 1] / out[i + 2].
- Per-depth rate j = P(draft j right | drafts 1..j-1 right). DFlash2 reference (D-02b, Leap): code 0.941 / 0.922 at
  depths 1 / 2, math 0.908 / 0.887 (L1-20, the exact 125B's full completions, teacher-forced). IFEval / multi / agent
  have no DFlash2 numbers yet (D-02a): their MTP rates are reported now and judged when D-02a lands.
- Lines (the row): PASS = MTP >= DFlash2 + 3 points per token at depths 1 AND 2 on >= 2 domains. KILL = MTP <=
  DFlash2 + 1 point at depths 1-2 on every domain with a DFlash2 number. Else BETWEEN. Read today on code + math.
- Caveats written now: MTP runs against v3's greedy tokens over the first 300 tokens (live decode), DFlash2 against the
  exact 125B's full completions; the positions differ, the alignment is the same (depth 1 = the token after the
  verify's bonus token, given target states up to the row before it).
- D-08 CALC (NewIdeas' ask; information for its row): Leap's D-02b walk copied (sticky/d02b_walk.py), saving every
  check's anchor per sample (data/d02b_walk_{code,math}.npz); the walk must reproduce D-02b (kept 9.57 code / 7.00
  math). Then on CPU, NewIdeas' detector (~/NewIdeas/scripts/d08_calc.py) marks calculator-determined completion tokens
  of the 60 GSM8K answers; reported: the share of them that are DFlash2 first misses, and the share of all first
  misses they make up. NewIdeas' follow-up line: PASS if >= 40 % of calculator tokens are missed by DFlash2.
- Run: turns/turn26.sh under one quiet lock (run_locked.sh 15): the walk (~1 min) then the MTP decode (~8-9 min).
- RAN under my lock 17:04:24-17:14:06 (logs/turn_job.out), rc 0.
- Walk: reproduces D-02b exactly (code 666 checks, kept 9.57, per token 0.9239; math 2,046 checks, kept 7.00, 0.8707);
  every check's anchor saved (data/d02b_walk_{code,math}.npz).
- D-08 CALC RESULT (sticky/d08_calc_walk.py, data/d08_calc_walk.txt): FAIL on NewIdeas' follow-up line.
  - Detector region: NewIdeas' detector as written scans the LAST gen tokens (start = n - gen), but dfl2d's gen is the
    absolute index where generation starts (= the tf prompt length: 98 / 139 / 171 for gsm 600 / 601 / 640). So its
    3.11 % (179 / 5,764, reproduced here) counts a slice of the completions over the sum of PROMPT lengths.
  - Fixed to the completion (ids[gen - off:]): 326 calculator tokens of 14,320 = 2.28 %.
  - DFlash2 already drafts 266 of them right (81.6 %); 54 are its first misses (16.6 %); 6 are the given first token.
  - They make 3.0 % of DFlash2's 1,822 first misses, at median depth 10. Line (>= 40 % missed): FAIL. Fixing all 54
    would extend ~2.6 % of checks by the few tokens after a depth-10 miss: < ~1 % kept.
- D-05 RAW (data/d05_mtp_depth.json; 32 prompts x 300 tokens, 15-38 tok/s): code 0.943 / 0.963 / 0.907, gsm 0.543 /
  0.874 / 0.790, ifeval 0.605 / 0.634 / 0.680, multi 0.469 / 0.472 / 0.558 (depths 1 / 2 / 3).
  - NOT the read-out: mtp_depth.py decodes 300 tokens without stopping at the end of the turn. Short answers continue
    into post-turn text: ifeval/3669 opens new 'assistant' / 'user' turns repeating its answer; gsm/600 emits runs of
    newlines after 'Answer: 10'. There the MTP head collapses (per-prompt depth 1 down to 0.17).
  - Read-out = the same passes cut at the first committed end token (sticky/d05_truncate.py; written 17:2x, run when
    EvalSuite's quiet window ends).
- D-05 agent (CPU; data/d05_agent_mtp.txt): Chat1000's stored 15-deep chains line up as mtp[i, j] -> out[i + j]
  (depth-1 agreement 0.845 there vs <= 0.022 at every other offset). The pre-registered rule named only out[i + 1] /
  out[i + 2], so the alignment was taken from the data; Chat1000's own P(first 1 right) = 0.85 agrees. Depth 1 / 2 / 3
  = 0.846 / 0.841 / 0.853 (1,920 positions), P(first 3 right) 0.607. No DFlash2 agent number yet (D-02a).
- READ-OUT 19:25 (with NewIdeas' 18:49 asks: same prompts and cut on both sides, counts, drafting cost). Two findings
  first: (1) turn26's MTP run used BigMoE's v3 PLAN mode (BIGMOE_TIER=4433333333: gate ranks 3-10 run 3-bit MSB
  tiers, not bitwise); (2) mtp_depth.py decodes 300 tokens past <|im_end|>. So turn27 (19:15-19:23, quiet lock, swap
  flat) re-ran the chain in BigMoE's EXACT 24 GB mode (plane pools 4444444444, bitwise; MTP streamed) on the 8 code + 8
  gsm prompts, saving the decode and spec125's timers. It also timed k = 3 / k = 1 with the MTP layer resident (as at
  45 GB) on 4 code prompts. Scoring: sticky/d05_compare.py -> data/d05_compare_exact.txt (v3 run:
  data/d05_compare.txt). Completion positions 1..L, L = the answer's first end token (capped at 300). Position 0 is the
  prefill's on DFlash2's side, so the MTP's position-0 trial is dropped. DFlash2 = the D-02b walk's checks on the same
  16 prompts, cut at L. Conditional per-depth acceptance (hits / trials):

  | domain | MTP chain k = 3, exact mode: d1 / d2 / d3 | DFlash2 L1-20: d1 / d2 / d3 | MTP - DFlash2 (points) | tokens per check at width 3 |
  |---|---|---|---|---|
  | code (same text *) | 0.972 / 0.981 / 0.947 (515/530, 511/521, 482/509) | 0.969 / 0.941 / 0.943 (188/194, 175/186, 164/174) | +0.3 / +4.0 / +0.4 | 3.83 vs 3.74 |
  | gsm (same questions **) | 0.918 / 0.898 / 0.874 (416/453, 377/420, 327/374) | 0.904 / 0.885 / 0.858 (208/230, 184/208, 157/183) | +1.4 / +1.3 / +1.6 | 3.46 vs 3.39 |

  * 2,047 positions compared, 83 differ, all in HumanEval/118 after token 206. ** Even BigMoE's bitwise exact mode
  differs from dfl2d's gsm answers from token 0 / 6 / 16 (gsm 600 / 608 / 632). dfl2d's texts come from another exact
  engine (Leap's harness; near-ties resolve differently), so gsm compares two greedy answer sets to the same 8
  questions, not paired positions. v3 plan mode gives the same picture: code 0.972 / 0.977 / 0.955, gsm 0.924 / 0.875 /
  0.893. On these 16 prompts DFlash2 keeps 10.55 (code) / 6.87 (gsm) per block-16 check (D-02b's 24 / 60: 9.57 /
  7.00); its depths 4-15 run 0.98-0.86 on code and 0.91-0.68 on gsm. IFEval / multi (MTP only, v3, cut): 0.637 / 0.550
  / 0.540 and 0.464 / 0.464 / 0.504.
- Drafting cost (spec125 timers, decode only, t_draft + t_head per pass): the MTP chain at k = 3 with the MTP layer
  resident costs 4.48 ms per check (k = 1: 1.95, so ~1.3 ms per extra step); streamed (24 GB exact mode) 8.26 ms.
  DFlash2's one parallel step: 12.5 ms per block-16 check (Leap DR: forward 7.5, top-16 head 4.5, selector 0.8; 8.5 to
  11.2 in the R-06 / R-10 / R-15 variants).
- Hybrid arithmetic at 45 GB (MTP for depths 1-3, DFlash2 for 4-15): if DFlash2's depth 4-15 rates held, the MTP
  prefix (0.903 vs 0.860 on code, 0.721 vs 0.687 on gsm) would lift tokens per check from 10.55 to 10.98 (+4.1 %) on
  code and from 6.87 to 7.11 (+3.5 %) on gsm. That is an UPPER bound: DFlash2 drafts all depths in parallel, so its
  depth-4+ guesses follow its own depth 1-3 guesses. Cost: +4.5 ms on a ~155 ms check (143-150 verify + 12.5 drafting,
  Leap CA) = +2.9 %, so net <= +1 % on code and <= +0.6 % on gsm (MTP for depths 1-2 only: +3.8 % / +2.4 % for +3.2 ms
  = +2.1 %). At 24 GB (MTP streamed, and R-43: 16-row checks need ~12+ kept) the hybrid gains nothing.
- RESULT D-05 = BETWEEN (19:25). Like-with-like: depth 1 +0.3 / +1.4 points and depth 2 +4.0 / +1.3 (code / gsm). PASS
  needs +3 at depths 1 AND 2 on 2 domains: code fails depth 1, gsm fails both. KILL needs <= +1 at depths 1-2 on
  every domain: code depth 2 is +4.0, gsm +1.4 / +1.3. Against the pre-registered D-02b figures (0.941 / 0.922,
  0.908 / 0.887) it is also BETWEEN (code +3.1 / +5.9, gsm +1.0 / +1.1). Noise: DFlash2 has ~190-230 trials per depth,
  SE ~1.6-2.2 points; only code depth 2 (+4.0, ~2.2 SE) stands out. As a speed lever the hybrid is not worth a build
  (net <= +1 %). For D98: the 1-layer native head, chained on its own tokens, matches or beats the 5-layer parallel
  block drafter at depths 1-3, so the near depths are not the gap (code ~0.97 for both). The gap to ~0.98 is at depths
  4-15 (code 0.86-0.98, gsm 0.68-0.91).

## D-09 DSCALE: is DFlash2's far-depth gap a DATA problem? (NewIdeas GO Wed 19:31; QUEUE D98 row D-09) -- pre-registration (written before any training, Wed 22:10)
- Why: D-05 put both drafters at ~0.97 for depths 1-3 on code. The D98 gap is depths 4-15 (DFlash2 L1-20 on code 0.98
  -> 0.86, on gsm 0.91 -> 0.68). L1-20 learned from only 22.5K anchors of the 125B's outputs. Data scaling of drafters is
  published (EAGLE-3's scaling result) but untested here.
- Method: Leap's L1-20 recipe unchanged. sticky/d09_train.py is a verbatim copy of ~/Leap/scripts/dfl2c_train.py, run
  with --bs 32 (block) --micro 1 --epochs 2 --minutes 50 --pmask 0.5: incoai init, lr 2e-5, S 4, warm-up min(20,
  steps/10 + 1), cosine to 10 %, the script's seeds. It trains on nested subsets of L1-20's 99 training windows,
  stratified by domain (sticky/d09_subsets.py, seed 9, symlinks to Leap's files):
  - 1/4 = 25 windows (18 he + 7 gsm), 5,774 anchors (0.257 of 22,504);
  - 1/2 = 50 windows (36 + 14), 11,392 anchors (0.506);
  - the same 10 val windows for both.
  At 2 epochs the step count scales with the data (~1/4 and ~1/2 of L1-20's 201).
- Score: sticky/d02b_walk.py (my walk, which reproduces D-02b's L1-20 9.57 / 7.00 exactly), with TL_DRAFT = the new
  weights. Block 16, official 1-step loop, kept per check (accepted + 1), on the 24 held-out HumanEval samples
  (dfl2d_code) and the 60 GSM8K answers (dfl2d_math). The full-data point is L1-20 itself (9.57 code / 7.00 math).
- Fit: per domain, the least-squares slope of kept per check against log2(anchor share) over the three points (0.257,
  0.506, 1) = tokens per doubling of data.
- Lines (NewIdeas'): PASS >= +0.4 per doubling on code or gsm. KILL <= +0.15 per doubling on both. Else BETWEEN.
- Caveat: one seed per point, so the slope's own noise is not measured (a retrained full-data point would measure it;
  it is not part of this screen).
- Prediction: incoai (no data) gives 8.88 on code and L1-20 9.54-9.57, so I expect a flattening curve: 1/4 ~9.2, 1/2
  ~9.4, a slope of ~+0.2 per doubling on code and less on gsm (only 27 gsm windows). 25 % PASS, 35 % KILL.
- Guards: start only with >= 40 GB free (22:08: ~43.7 GB free + inactive + purgeable); stop if swap grows > 2 GB; one
  quiet lock (~35 min).

### D-09 RESULT: BETWEEN (read-out Thu 02:15; run Wed 22:11-22:34, turns/turn28.sh)
- Ran as pre-registered: both trainings "done" (1/4: 51 steps, 1/2: 101; L1-20: 201), peak 29 GB, swap flat (11.93 ->
  11.92 GB). Walks: data/d02b_walk_d09q.txt, data/d02b_walk_d09h.txt; the full point is data/d02b_walk.txt (L1-20).

| data (anchor share) | code kept / check | math kept / check | code P(first 15 right) | math P(first 15 right) |
|---|---|---|---|---|
| 1/4 (0.257) | 9.37 | 6.65 | 0.203 | 0.056 |
| 1/2 (0.506) | 9.46 | 6.82 | 0.231 | 0.075 |
| all (1.0, L1-20) | 9.57 | 7.00 | 0.276 | 0.086 |

- Fit (least squares against log2 of the anchor share, as pre-registered): code +0.10, math +0.18 tokens per doubling.
  PASS needs >= +0.4 on either: no. KILL needs <= +0.15 on both: no (math +0.18). RESULT D-09 = BETWEEN. Rounding of
  the printed 2-decimal values moves a slope by < 0.01, so the verdict does not depend on it.
- Where the gain lands (conditional per-depth rates, mean of depths 1-5 / mean of depths 11-15): code 0.934 / 0.844
  (1/4), 0.932 / 0.856 (1/2), 0.934 / 0.889 (all); math 0.878 / 0.771, 0.883 / 0.808, 0.884 / 0.813. The near depths
  are flat; the far depths gain ~+0.02 per token per doubling. So the far-depth gap does respond to data, but slowly.
- For D98: on the log-linear line, 16x the data (4 doublings, ~360K anchors) gives code ~10.0 and math ~7.7 at block
  16; even 1000x would give ~10.6 / ~8.8 if the line held. D98 needs ~14. More data under this recipe cannot close the
  gap; the lever is the drafter's architecture and capacity (the MTP-head design question), which is KILL's reading.
  Math's +0.18 sits within the unmeasured retrain noise of the KILL line.
- Prediction check: I expected code 1/4 ~9.2, 1/2 ~9.4, a slope of ~+0.2 and less on gsm. Code came out flatter
  (+0.10) and math steeper (+0.18). BETWEEN was my 40 % case.
- Caveats: one seed per point (retrain noise unmeasured; the whole code effect is 0.20 tokens). Steps scale with the
  data at 2 epochs (the pre-registered 'same steps per token'), so the 1/4 run had 51 optimizer steps; a fixed-step
  design would answer a different question.

## D-02a: DFlash2 L1-20 on IFEval / multi / agent, and how sure the 125B is at DFlash2's first misses (NewIdeas GO Thu 15:5x; QUEUE D98 row D-02) -- pre-registration (written before the capture, Thu 15:55)
- Why: the design step needs to know what kind of first misses DFlash2 makes. Where the 125B is >= 0.99 sure, the miss
  is a drafter knowledge problem (capacity, inputs or the training signal can fix it). Where it is a near-tie (p1 - p2
  <= 0.1), the miss is the exact ceiling, decided by rounding; only near-tie acceptance helps there, and Kamen hasn't
  wanted that. D-02's three missing domains come with it.
- Data: code's top-32 is already stored in ~/Leap/data/l118_tgt (ids equal dfl2d_code's, 24 / 24). One capture,
  sticky/d02a_extract.py (Leap's dfl2_extract.py plus dfl2c_extract.py's top-32 hook, same runtime), covers the 60 GSM8K
  texts (p only; their features stay dfl2d_math's), IFEval (30), multi (24) and the 12 agent turns (features + p) into
  data/d02a/. ~146K prefill tokens, ~7 min at Leap's ~370 tok/s.
- Walk: sticky/d02b_walk.py (L1-20, block 16, official 1-step loop; TL_WALK_SETS) on IFEval / multi / agent: kept per
  check, per-token rate, rate per depth.
- Table: per domain and depth band (1-3, 4-7, 8-11, 12-15), each first miss (a check with k < m: the token at a + k + 1)
  is split by the 125B's p1 / p2 on that token's row: sure (p1 >= 0.99), near-tie (p1 - p2 <= 0.1) or between. The same
  split over all scored positions gives the base rate. Positions where the 125B's prefill top-1 is not the text token
  are counted apart.
- Route (NewIdeas'): sure >= 50 % of the first misses at depths >= 4 on code or math -> a knowledge problem (the design
  study, plus one TVD / reverse-KL screen); near-tie >= 50 % -> the remaining gap there is the exact ceiling; else mixed.
- Prediction (depths >= 4): code sure ~45 %, near-tie ~15 %; math sure ~35 %, near-tie ~25 %. Kept per check: IFEval
  ~5, multi ~4.5, agent ~5 (near-copy 6.6 / 8.0 / 6.8).
- Guards: start only with >= 24 GB free (Leap's captures peaked at 22.4-22.5 GB on <= 42K-token sets, 40.9 GB on its
  268K-token set); stop if swap grows > 2 GB; one quiet lock of 15 min; PAUSE-aware (pausewait).

### D-02a RESULT (Thu 19:13; data/d02b_walk_d02a.txt, data/d02b_walk_d02a_orig.txt, data/d02a_miss.txt)
- Capture: 126 keys in data/d02a (two quiet locks, 16:17-16:31 and 16:55-17:00; peak 20.5 GB). On 12 math files its
  features and top-1 equal Leap's dfl2d_math bit for bit, so the pipeline is the one D-02b used.
- Walk at block 16, 1 step (kept per check / per-token rate):

| set | L1-20 | incoai's original |
|---|---|---|
| IFEval (30) | 2.45 / 0.594 | 2.52 / 0.604 |
| multi, 8 languages (24) | 2.06 / 0.515 | 2.11 / 0.526 |
| agent turns (12) | 3.01 / 0.672 | 3.02 / 0.672 |
| code (24) | 9.57 / 0.924 | 8.91 / 0.895 |
| math (60) | 7.00 / 0.871 | 6.58 / 0.852 |

  The near-copy keeps 6.6 / 8.0 / 6.8 on instructions / chat / agent. So the weak spot is DFlash2's own: L1-20's tune
  added +0.66 on code and +0.42 on math and cost <= 0.07 elsewhere. Leap's agent-text tunes (L3 / L4, up to 268K
  tokens; ~/Leap/data/dfl2_result_l3*/l4*_agent.txt) moved agent by only +0.05 to +0.11, and NewIdeas killed D-11 on that.
- First misses by the 125B's confidence on the missed token. Sure = p1 >= 0.99, tie = p1 - p2 <= 0.1, mid = the rest;
  shares of on-path first misses. The base column is all drafted completion positions.

| set | base sure / tie | depths 1-3 sure / tie / mid | 4-7 sure | 8-11 sure | 12-15 sure | depths >= 4 sure / tie |
|---|---|---|---|---|---|---|
| code | 0.89 / 0.01 | 0.10 / 0.13 / 0.78 | 0.27 | 0.72 | 0.85 | 0.61 / 0.05 |
| math | 0.80 / 0.01 | 0.14 / 0.08 / 0.78 | 0.33 | 0.53 | 0.71 | 0.46 / 0.04 |
| IFEval | 0.34 / 0.08 | 0.08 / 0.15 / 0.76 | 0.11 | 0.45 | 0.92 (12) | 0.17 / 0.11 |
| multi | 0.39 / 0.05 | 0.26 / 0.09 / 0.66 | 0.16 | 0.23 | - | 0.17 / 0.10 |
| agent | 0.62 / 0.04 | 0.33 / 0.08 / 0.59 | 0.44 | 0.70 | 0.85 | 0.53 / 0.07 |

- Reading:
  - Near-ties are only 5-15 % of the first misses everywhere, so the exact ceiling is not what stops DFlash2.
  - The shallow misses (depths 1-3, which set the per-token rate) fall on mid tokens: the 125B clearly prefers one
    token but is < 0.99 sure (code 0.78, math 0.78, IFEval 0.76, multi 0.66, agent 0.59). A drafter must match the
    125B's moderate preferences there.
  - The far misses fall on tokens the 125B finds obvious given the prefix (code 0.72-0.85 sure at depths 8-15, math
    0.53-0.71): lookahead failures.
  - On IFEval / multi the 125B itself is >= 0.99 sure on only 0.34 / 0.39 of the tokens (code 0.89, math 0.80). These
    texts carry real choice, which is why a 0.59B drafter falls so far below the near-copy there.
- Route (pre-registered): code, depths >= 4 sure 0.61 -> knowledge; math 0.46 -> mixed. By NewIdeas' rule that allows
  one divergence-form screen: D-13 below.
- Prediction check: I expected near-ties at ~15 / ~25 % of the deep misses on code / math; they are 5 / 4 %. The sure
  shares (0.61 / 0.46) are above my ~45 / ~35 %. Kept per check was far below my IFEval ~5 / multi ~4.5 / agent ~5.

## D-13 DIVERGE: mode-seeking targets for greedy acceptance (MTP Head; from D-02a, NewIdeas' one-screen rule) -- pre-registration (written before the run, Thu 19:13)
- Why: D-02a puts the shallow first misses on mid tokens. Greedy verification accepts only the 125B's argmax, but
  L1-20 trains forward KL to the whole top-32. Targets weighted toward the mode may move the drafter's argmax onto the
  125B's at those tokens. Published: DistillSpec (the divergence choice depends on the decoding mode); untested here.
- Arms (Leap's R-05 rapid protocol: 21 steps from L1-20, warm-up 3, cosine, L1-20's recipe otherwise;
  sticky/d13_train.py = r05_train.py + --div):
  - U = forward KL (the control);
  - S = forward KL to softmax(lp / 0.5) (sharpened);
  - V = total variation on the top-32 plus one bucket for the rest of the vocabulary.
- Score: the D-02b walk (block 16, 1 step) on code (24) + math (60). Ratios S / U and V / U of kept per check, with a
  90 % bootstrap CI over samples (sticky/d13_ratio.py).
- Lines: PASS: S / U or V / U >= 1.03 on code or math, with the CI's low end > 1.00 -> the full 2-epoch run of the winner
  (~28 min). KILL: both <= 1.01 on both domains. Else BETWEEN.
- Prediction: S / U 1.00-1.02 (KILL 45 %, BETWEEN 40 %, PASS 15 %). V / U <= 1.00, because TVD's bounded gradient
  moves little in 21 steps. R-05 found the protocol's U itself ~2 % below L1-20 (21 more steps of a plateaued recipe).
- Guards: >= 34 GB free to start (L1-20 peaked 33.7 GB); stop if swap grows > 2 GB; one quiet lock (15 min),
  resumable per arm.

### D-13 RESULT: KILL (Thu 19:33; run 19:22-19:33 under one quiet lock, turns/turn31.sh; data/d13_ratio.txt, data/d02b_walk_d13{U,S,V}.txt)

| arm | code kept / check | math kept / check | vs U, code (90 % CI) | vs U, math (90 % CI) |
|---|---|---|---|---|
| U forward KL (control) | 9.40 | 6.71 | - | - |
| S sharpened targets (T 0.5) | 9.39 | 6.67 | 0.999 (0.992-1.005) | 0.994 (0.988-0.999) |
| V total variation | 9.32 | 6.68 | 0.991 (0.975-1.007) | 0.996 (0.984-1.008) |

- Every ratio is <= 1.01 on both domains: KILL, as I predicted at 45 % for S (V <= 1.00 predicted). U sits ~2 % under
  L1-20 (9.57 / 7.00), the same drift R-05 found for 21 more steps of a plateaued recipe. Swap flat (10.8 -> 10.3 GB).
- Reading: how the target distribution is weighted doesn't move the drafter's argmax onto the 125B's at the mid tokens
  where its shallow misses sit. With D-09 (more data: +0.10 / +0.18 per doubling) and Leap's SD (capacity by depth,
  AR / semi-AR, selectors: <= +3-6 %), the remaining levers are the drafter's INPUTS and what it can compute per
  position at the same kernel count (width), which is the design study's question.

## D-14 probes (design study, CPU, stored data; Thu 20:47; sticky/d14_probe.py -> data/d14_probe.txt)
- A. Depth-1 hit rate of DFlash2 L1-20 on tokens the 125B is >= 0.99 sure of: code 0.996, math 0.993, IFEval 0.899,
  agent 0.795, multi 0.622. On mid tokens: 0.791 / 0.744 / 0.565 / 0.529 / 0.452. So outside code and math the drafter
  misses even tokens the 125B finds obvious: a knowledge / coverage gap, not only real choice.
- B. Multi by language, depth-1 hits on sure tokens: es 0.936, fr 0.824, de 0.734, ru 0.697, ja 0.683, zh 0.672,
  ar 0.454, bg 0.365. Kept per check: es 2.85, fr 2.74, de 2.39, ru 2.11, ja 2.08, zh 2.00, bg 1.60, ar 1.59. That fits
  training-data coverage of incoai's drafter (the 125B itself is sure on 29-47 % of tokens in every language).
- C. Agent: the true 4-gram at a first miss occurs inside the drafter's 2048-token window 14 %, only beyond it 13 %,
  nowhere earlier 73 %. A longer window could reach at most ~13 % of the agent misses, so the window is not the main
  gap. Leap's agent-text tunes (+0.05-0.11) already say coverage there is hard to buy with data.

## D-15a LANGPROBE: can a little of the 125B's own text in a language teach the drafter that language? (MTP Head, from D-14 B) -- pre-registration (written before the run, Thu 20:47)
- Why: D-14 B shows DFlash2 missing 55-64 % of the tokens the 125B is sure of in Bulgarian and Arabic, but only 6 % in
  Spanish. If that is coverage, a tune on the 125B's own answers in those languages should move it fast. Then the
  chat wall (2.06 kept per check; the near-copy keeps 8.0) has a cheap path: the 125B's answers to new multilingual
  prompts plus a proper tune (D-15).
- Screen: D-02a's 24 multilingual answers, 3 per language. Pack two per language (16) as training windows
  (sticky/d15_pack.py; h = incoai's frozen fc + hidden_norm as in dfl2c_extract.py) and hold out the third (bg2 de5 fr8
  es11 ru14 zh17 ja20 ar23). A rapid tune from L1-20 on the 16 uses L1-20's recipe (forward KL to the top-32, block
  32, pmask 0.5, lr 2e-5), 2 epochs (~38 steps; sticky/d13_train.py --div fkl).
- Score: the D-02b walk (block 16, 1 step) on the 8 held-out answers against L1-20's walk of the same 8: kept per
  check, and the depth-1 hit rate on the 125B's >= 0.99-sure tokens, per language (sticky/d15_read.py). Code + math
  are walked for forgetting, against D-13's U (9.40 / 6.71, the same drift for a few more steps).
- Lines: PASS = held-out kept per check +0.3 or more over L1-20 AND the sure-token depth-1 hit rate +0.10 or more, with
  code / math within -3 % of U -> D-15 (new multilingual prompts, the 125B's answers, a proper tune; and the same test
  on IFEval). KILL = held-out kept +0.1 or less. Else BETWEEN.
- Prediction: +0.3-0.6 kept on the held-out 8, most in bg / ar (PASS 45 %, BETWEEN 35 %, KILL 20 %). Code / math -1
  to -3 %.
- Guards: >= 34 GB free to start, stop if swap grows > 2 GB; one quiet lock (15 min). Disk: 1.1 GB of weights.

### D-15a RESULT: KILL (Thu 21:00; run 20:54-21:0x, one quiet lock, turns/turn32.sh; data/d15_read.txt)
- The 8 held-out multilingual answers kept 2.07 per check with L1-20 and 2.06 after the tune: -0.01 (line +0.1). The
  depth-1 hit rate on the 125B's sure tokens went 0.655 -> 0.643. By language (kept, L1-20 -> tuned): ar 1.70 -> 1.72,
  bg 1.63 -> 1.63, de 2.50 -> 2.31, es 3.10 -> 3.25, fr 2.65 -> 2.94, ja 2.00 -> 1.97, ru 2.24 -> 2.13, zh 1.91 ->
  1.90. Bulgarian and Arabic did not move; the other moves look like noise.
- Forgetting: code 9.28 vs D-13's U 9.40 (-1.3 %), math 6.73 vs 6.71. Small.
- vs prediction: I gave PASS 45 %; it is a clean KILL. Two answers per language (~4.8K anchors, ~38 steps) teach the
  drafter nothing about the language. In D-09, ~5.8K anchors of code / math from incoai gave +0.49 on code.
- Reading: the multilingual gap is not cheap coverage. The native MTP head is just as weak there (D-05: depth 1 0.464 on
  8 of these answers). Two small heads with very different training both fail on tokens the 125B is sure of, which
  points at what a small head can compute per token in these languages (sub-word spelling across many scripts), not at
  missing data. That moves the chat / instruction / agent walls toward target-class compute: the near-copy keeps
  6.6-8.0 there.

## D98 DESIGN STUDY 1 (MTP Head, Thu 21:09): what keeps a drafter target-class, and the next screens
Extends Leap's SD study (TABLE_350 'SD', Mon) with today's measurements and a literature sweep of June-October 2026
(subagent, web reading only; arXiv ids as reported, numbers from the papers, not re-checked here).
- **Our evidence today.**
  - The per-token rate at shallow depths is the limit (SD). On code / math DFlash2 gets ~99.5 % of the tokens the 125B
    is sure of; its shallow misses are mid tokens, and its deep misses are sure tokens it cannot foresee (D-02a).
  - Outside code / math it misses tokens the 125B is sure of: IFEval 10 %, agent 20 %, multilingual 38 % (bg 64 %),
    the D-14 probes. The native MTP head is just as weak on multilingual (D-05).
  - Dead here: more data (D-09, slow), loss shape (D-13, R-05), a little in-language data (D-15a), agent-text tunes
    (Leap L3 / L4), copy alone, recycled verify rows (M-06), a longer window (D-14 C).
  - Measured positives: the 16-row tree +9 % (L1-22). The native MTP chain keeps 4.00 per pass on agent turns vs
    DFlash2's 3.01 (Chat1000 suffix_agentic, Leap L2b: 'Agentic = MTP alone').
- **Literature since SD** (the closest to us first):
  - Native MTP heads can stay accurate when chained, if trained unrolled on their own outputs.
    - AngelSpec (2607.25852; Tencent, Hy3 MoE): a shared-weight MTP unrolled D steps on its own argmax, LK then TV loss.
      Cumulative depth-3 acceptance 0.266 -> 0.524; MTP tau MT-Bench 2.16 -> 2.45, GSM8K 2.70 -> 3.42.
    - Red Hat's FastMTP note on Qwen3-Next-80B: ~8K GSM8K samples, per-position acceptance 0.897 / 0.719 / 0.476 ->
      0.912 / 0.776 / 0.616.
    - Microsoft (2610.00888): chained one-layer heads on the target's own answers, depth-3 +12-15 points on chat.
  - Routing per round between an AR / MTP drafter and a block drafter (WhiFlash, 2606.07710, Qwen3-8B; tau EAGLE-3 /
    DFlash / router): chat 5.24 / 3.40 / 5.47, agentic 4.42 / 4.12 / 4.90, math 5.58 / 7.38 / 7.72. The router is the
    target's entropy or a tiny MLP, < 1 % of drafting time.
  - Exact trees on GDN hybrids are solved, with a fixed kernel count per layer.
    - Bole (2608.01651): Qwen3.5-122B-A10B, the same family as ours, with native-MTP trees. MBPP tau 6.40; tree
      verification 3.4-7.7x faster than the baselines.
    - TreeWY (2608.20961): one triangular solve per node. Trees pay only with deep drafters; DFlash2's block 16 is one.
  - Richer target inputs for agent text. DLite (2610.23787) is a history-free drafter reading 12-14 tapped layers;
    agentic tau 2.5-3.2 vs DFlash's 1.1-1.8 (Qwen3-4B / 8B, K = 7).
  - Matched-data study (2610.24993): with data matched, block-drafter design buys ~0.2 tokens on chat. The chat gap is
    drafter class / capacity.
  - Multilingual needs a pretrained drafter body. Osprey (2609.09338): a drafter body pretrained on ~500B tokens of web
    text gives G-MMLU (42 languages) 2.63 -> 3.26 on a 229B target. 'No Refund for the Token Tax': Qwen3.5-9B's native
    drafter 3.16 in English vs 1.89-2.81 in 20 other languages, the same weakness as ours.
  - Online adaptation. Test-Time Speculation (2605.09329): one gradient step per round on the verified tokens, DFlash
    +41 % on average on reasoning sets (Qwen3.5-122B-A10B AIME25 2.9 -> 4.9). Its gains come from long generations
    where DFlash decays; ours are short.
  - Best reported chat numbers are ~5.5 per 16-token block (~0.82 per token), and nothing reaches 0.96-0.98 outside
    code / math. Our near-copy's 6.6-8.0 is already above every chat / agent number in the sweep.
- **What keeps a drafter target-class?** The 125B's own MTP head reads the full PRE-MIXER WIDENED stream (4 x 2560 =
  10,240 per position; qwen4_exp.py: "this family has exactly one draft input: the pre-mixer WIDENED stream"), and the
  125B's readout is a learned mix of the streams. DFlash2 gets only the plain mean of the 4 streams at 5 taps. The
  native head is target-class at depths 1-2 on code and beats DFlash2 by +1.0 per pass on agent turns. Its decay past
  depth 3 is what the unrolled training fixes in the papers above.
- **Candidates, ranked, each with a short kill screen:**
  1. **D-16 HCFULL (new, general for hyper-connection / mHC targets): give DFlash2 the target's final widened stream
     (the MTP head's own input) as an extra context input.** A zero-initialised 10,240 -> 2,560 projection added to h
     keeps L1-20's function at the start. Screen: capture the widened stream for L1-20's 109 windows + the code / math
     walk sets (~1.5 GB), Leap's rapid protocol (U vs H from L1-20, the new projection at a higher learning rate), walk
     code / math (+ agent). Two quiet locks. PASS >= +3 % on code or math, or >= +0.3 kept on agent.
  2. **D-17 MTPDEEP: how fast does our native MTP chain decay past depth 3, on the same five sets?** k = 7 in exact
     mode (mtp_depth_ids.py) on D-02a's prompts, plus agent125's stored chain. One lock. If depths 4-7 fall well below
     DFlash2's rates, the unrolled MTP training (AngelSpec / FastMTP) has room, and the MTP chain + router line
     (WhiFlash) becomes the head architecture for agent / chat. The training itself is a bigger build: LoRA on the
     MoE MTP layer in MLX, inputs = the same widened stream as D-16.
  3. **Trees on GDN (+9 % measured at 16 rows, L1-22): Leap's kernel line.** Bole / TreeWY give the recipe for this
     family. Handed to Leap with the references.
  4. Width at the same kernel count (2x FFN at 4-bit, Net2Wider from L1-20, ~1 h local training): DFlash Table 6 / Scylla
     say ~+6 % per 1.6x capacity, and the matched-data study caps architecture tweaks at ~0.2 tokens on chat. Parked
     behind 1-2.
  5. Multilingual: a pretrained drafter body (Osprey) needs a small pretrained LM, a download and Kamen's call. Parked;
     priced only if NewIdeas wants it.
- **Stack estimate (not measured)** on code: trees +9 % and HCFULL / width +3-6 % each would give ~10.8-11.5, against
  the wall at 10.57 (+1 kept). On agent: route to the MTP chain (4.00, +1.0 today) and train it unrolled (the papers'
  depth-3 gains) toward ~5.

## D-17 MTPDEEP: the native MTP chain past depth 3, and the per-round router's oracle (NewIdeas' row D-17; from design study 1) -- pre-registration (written before the run, Thu 21:13)
- Why: the native MTP head reads the 125B's full widened stream and beats DFlash2 by +1.0 per pass on agent turns.
  Papers since Monday make chained MTP heads accurate when trained unrolled (AngelSpec, FastMTP), and route per round
  between an MTP / AR drafter and a block drafter (WhiFlash). Two questions: how fast does our chain decay past depth
  3, and what would the best per-round choice between it and DFlash2 keep?
- Part 1 (one lock, turns/turn33.sh): plain greedy decode of D-05's 32 prompts (8 code / gsm / ifeval / multi, 300
  tokens) with the native MTP head drafting a 15-token chain at every position (sticky/d17_mtpchain.py = Chat1000's
  agent125.py without the route hook). Agent: Chat1000's stored 15-token chains on the 12 turns.
- Part 2 (one lock, turns/turn34.sh): DFlash2's features on the same texts (d02a_extract.py, D02A_SRC=d17), then DFlash2
  L1-20 drafted at every position (d02b_walk.py, TL_WALK_EVERY); agent from D-02a's features.
- Read-out (sticky/d17_read.py, sticky/d17_route.py): the MTP chain's conditional rate at depths 1-15. Tokens per pass
  for DFlash2 alone, MTP alone (K 7 / 15), the per-round oracle (the better of the two at each pass), and a confidence
  router (MTP when the 125B's p on the bonus token is below a threshold; in-sample, so optimistic).
- Lines (NewIdeas'): PASS = oracle >= +0.5 kept over DFlash2 on chat (multi) or agent. KILL = < +0.2 on all three of
  ifeval / multi / agent.
- Prediction: agent oracle ~+1.2 over DFlash2 (MTP alone already +1.0): PASS on agent. ifeval / multi oracle +0.2-0.4
  (D-05 had MTP ~ DFlash2 at depths 1-3 there). Code / math: DFlash2 wins most passes, oracle +0.3-0.6 at K 15. The
  chain decays to ~0.7-0.8 by depth 7 on code / math. 60 % PASS (on agent), 10 % KILL.
- Guards: >= 30 GB free to start, stop if swap grows > 2 GB; two quiet locks of <= 15 min; resumable.

## D-16 HCFULL: give DFlash2 the input the native MTP head reads (MTP Head; NewIdeas' row D-16; from design study 1) -- pre-registration (written before any capture, Thu 21:17)
- Why: the 125B carries 4 hyper-connection streams per position (4 x 2560 = 10,240). Its own MTP head drafts from the
  full PRE-MIXER WIDENED stream, and its readout is a learned mix of the streams. DFlash2 sees only the plain mean of the
  streams at 5 taps. The native head beats DFlash2 by +1.0 per pass on agent turns and is target-class at depths 1-2
  on code. Richer target inputs drive the agent gains in DLite (2610.23787). Untested here, and general for any
  hyper-connection / mHC target.
- Design: each DFlash2 context row becomes h + proj(rmsnorm(wd)), where wd is the final widened stream at that
  position and proj is a new 10,240 -> 2,560 linear map, zero-initialised, so the start is L1-20 exactly
  (sticky/d16_train.py, sticky/d02b_walk.py TL_HC). Drafting cost: one more 10,240 x 2,560 matmul over the new context
  rows only (~26M weights): a few kernels and ~0.05 GB per check at 4-bit.
- Part 1 (turns/turn35.sh, one lock): capture wd (Qwen4ExpTextModel._last_widened, fp16; sticky/d16_extract.py) for
  L1-20's 109 training windows (42,246 rows), the code / gsm walk sets and the 12 agent turns, ~2 GB.
- Part 2 (turns/turn36.sh, one lock): arm H = Leap's rapid protocol from L1-20 (21 steps, L1-20's recipe) plus wd, with
  proj at lr 1e-3 (the rest at L1-20's 2e-5 schedule). Control = D-13's U (the same 21 steps without wd; its code / math
  walks exist, agent is walked here). Walks at block 16 on code (24), math (60) and agent (12); H / U with 90 % bootstrap
  CIs (sticky/d16_read.py).
- Lines (NewIdeas'): PASS = H / U >= 1.03 on code or math (CI low > 1.00), or agent kept +0.3 -> the full 2-epoch run,
  then the 45 GB price with Leap. KILL = H / U <= 1.01 on all three. Else BETWEEN.
- Prediction: code / math +1-3 % (the mean already carries most of what the target's own readout uses); agent +0.2-0.5,
  where the native head's lead is. 21 steps is little time for a new 26M-weight projection, so this can only
  under-read: PASS 30 %, BETWEEN 35 %, KILL 35 %.
- Guards: >= 30 GB free for the capture (C = 128 resident experts if >= 42 GB free), >= 34 GB for training; stop if swap
  grows > 2 GB; two quiet locks of <= 15 min; resumable.

### D-17 part 1 read-out (Thu 21:36; data/d17_read.txt; 24 of 32 prompts, multi pending; run 21:21-21:35, one lock)
- The native MTP chain from the true prefix vs DFlash2 L1-20 at block 16 on the same answers (tokens per pass):

| set | MTP K3 | MTP K5 | MTP K7 | MTP K15 | DFlash2 | MTP rate at depth 1 / 7 / 15 |
|---|---|---|---|---|---|---|
| code (8) | 3.81 | 5.32 | 6.21 | 7.78 | 10.62 | 0.977 / 0.869 / 0.697 |
| gsm (8) | 3.47 | 4.45 | 5.27 | 6.07 | 6.73 | 0.937 / 0.835 / 0.685 |
| ifeval (8) | 2.24 | 2.39 | 2.42 | 2.43 | 2.29 | 0.690 / 0.500 / - |
| agent (12) | 3.01 | 3.60 | 3.91 | 4.24 | 3.01 | 0.870 / 0.799 / 0.735 |

- The chain decays slowly on code / math (~0.97 -> ~0.87 by depth 7, ~0.70 at 15) but stays behind DFlash2's block 16
  there. On agent it keeps +1.23 per pass over DFlash2 (Chat1000's 4.00 re-measured); on IFEval +0.14.
- For 45 GB tok/s (an estimate, not measured: ~1.5 ms per MTP step, DFlash2 ~8 ms per block, a ~40 ms check):
  - On agent text an MTP chain at K 5-7 gives ~3.6-3.9 per pass for ~8-11 ms of drafting, ~+20 % over DFlash2.
  - On code DFlash2 stays far ahead (10.62 vs 6.21 at K 7).
  - So the router matters for agent text, and DFlash2's own quality (D-16) for code, the 45 GB headline.
- Part 2 (the per-round oracle on the same texts) follows after D-16.

### D-17 RESULT: PASS on NewIdeas' line (Fri 02:33; part 2 run 02:07-02:29, two quiet locks, turns/turn39.sh; data/d17_route.txt, data/d17_ci.txt)
- Both drafters at every position of the same answers (each cut at its first end token), walked per pass; tokens per
  pass = accepted + 1; 90 % bootstrap CIs over answers (sticky/d17_ci.py):

| set | DFlash2 L1-20 | MTP K15 | MTP K15 / DFlash2 | oracle K15 (better of the two per pass) | oracle / DFlash2 | confidence router (in-sample) |
|---|---|---|---|---|---|---|
| code (8) | 10.44 | 7.78 | 0.746 (0.703-0.791) | 11.62 (+1.18; MTP on 15 % of passes) | 1.113 (1.052-1.166) | 10.49 (+0.05) |
| gsm (8) | 6.60 | 6.07 | 0.919 (0.886-0.947) | 7.45 (+0.84; 21 %) | 1.128 (1.089-1.162) | 6.60 (+0.00) |
| ifeval (8) | 2.24 | 2.43 | 1.084 (1.032-1.176) | 2.66 (+0.42; 24 %) | 1.186 (1.146-1.261) | 2.40 (+0.16) |
| multi (8) | 2.22 | 2.92 | 1.315 (1.184-1.464) | 3.18 (+0.96; 41 %) | 1.434 (1.325-1.556) | 2.84 (+0.63) |
| agent (12) | 3.28 | 4.24 | 1.295 (1.208-1.372) | 4.69 (+1.41; 35 %) | 1.431 (1.340-1.535) | 4.12 (+0.84) |

- PASS: oracle +0.96 on chat (multi) and +1.41 on agent, over the +0.5 line. Prediction (agent ~+1.2, ifeval / multi
  +0.2-0.4, code / math +0.3-0.6) was low everywhere: the two drafters are more complementary than expected, on code too.
- The cheap router (MTP when the 125B's p on the bonus token is low) gets none of the code / math oracle; on multi /
  agent its best threshold just picks MTP on 72-88 % of passes, i.e. it approximates "MTP alone".
- On chat and agent text the native MTP chain ALONE beats DFlash2 by 29-32 % per pass (CIs above 1.18) at the same
  16-row check. For 45 GB tok/s (estimate, not measured; ~150 ms per pass today, ~1.5 ms per MTP step): a text-type
  switch (MTP K7 on chat / agent, DFlash2 on code / math) is ~+12-22 % on chat / agent text and 0 on code.
- Code, the 45 GB headline: the oracle is +11 % (CI 1.05-1.17) but needs a selector that sees which chain will be
  right, or a 2-branch tree (both chains in one check; the extra rows cost bytes at 45 GB). Not capturable with what
  we have; a learned selector needs both drafters' per-depth confidences (not stored).
- Next: the switch is an engine decision (BigMoE / Leap own the 45 GB loop); the 2-branch tree is Leap's tree line.

### D-18 MTP1: the MTP head's first guess revealed to DFlash2 -- desk KILL (Fri 02:33; sticky/d18_hybrid.py -> data/d18_hybrid.txt)
- Idea: L1-20 is trained to use revealed block tokens (pmask 0.5), so the native head could draft position 1 and
  DFlash2 positions 2-15 in one check. Bounds from D-17's every-position data (LOW: only the MTP token where DFlash2
  missed depth 1; HIGH: DFlash2 anchored one token later, optimistic):
  code 10.44 -> 10.18 / 10.18 (-2.5 %), gsm 6.60 -> 6.44 / 6.71, ifeval 2.24 -> 2.26 / 2.39, multi 2.22 -> 2.39 / 2.64,
  agent 3.28 -> 3.45 / 3.68.
- Why it fails: once answers are cut at their end token, DFlash2's depth 1 on code is 0.986 vs the MTP head's 0.977
  (the 0.89 in the raw every-position walk came from text past the end token). On chat / agent the hybrid stays below
  MTP alone. KILL at the desk, no GPU run.

### D-16b: the H arm DIVERGED -- a setup error, not a result (Fri 01:57; turns/turn36.sh; data/d16_loss.txt)
- H read 1.27 / 1.18 / 1.10 per check on code / math / agent (U 9.40 / 6.71 / 2.94). Per-step training loss on the
  same batches as U: equal at step 1 (3.326), +0.11 at step 2, +5.8 from step 3 on, when the projection's lr reached
  its 1e-3 peak. One Adam step moves every output of a zero-init 10,240-fan-in projection by up to lr x 10,240 (x the
  normed input), so 1e-3 swamps the context rows; even the 1e-5 first step cost +0.11.
- Retry (turns/turn38.sh, queued): fan-in-matched rates, Hc 5e-6 (main layers: fan-in 2,560 at 2e-5) and Hb 1.5e-6,
  training only, screened by the paired per-step loss vs U (sticky/d16_loss.py: each step's loss is taken before its
  update, an online held-out loss on the same batch for both arms). An arm below U with the CI under 0 gets the walks;
  else D-16 KILL at the rapid protocol (21 steps may be too few for a new 26M-weight projection: an under-read).

## D-19 SWITCH: past-only switch rules between the native MTP chain and DFlash2 (NewIdeas' GO, Fri 02:4x; CPU replay on D-17's data) -- pre-registration (written before the run)
- Why: D-17's oracle is +0.96 / +1.41 on chat / agent, and MTP alone already +0.70 / +0.96 there, but an engine can
  only use the past. Which rules keep most of it without hurting code / math?
- Rules (sticky/d19_switch.py; MTP K15 vs DFlash2 L1-20 block 16; same walk as D-17): type (MTP on ifeval / multi /
  agent, DFlash2 on code / gsm, the text type known from the prompt); bandit over the last W = 2 / 4 / 8 passes with
  partial feedback (only the drafter that ran is scored) and with full feedback (DFlash2 drafted every pass, scored
  on the verified tokens: capped at the chosen arm's accepted + 1; MTP scored only when it ran); both (type start +
  full-feedback bandit).
- Lines (NewIdeas'): PASS = >= +0.4 kept on chat (multi) or agent with code / math >= -0.1.
- Prediction: type passes by construction (+0.70 multi, +0.96 agent, 0 on code / math). The bandits pass on agent
  (+0.5-0.8) but explore MTP on code, where an MTP pass costs ~2.7 tokens: code -0.05 to -0.3, so W 8 / full feedback
  may pass and W 2 partial may not. 75 % PASS (type alone suffices), 50 % that a bandit without the type passes too.
- Cost is NOT in the line: MTP K15 is 15 sequential MTP-layer steps per pass (estimate ~1.5 ms each at 45 GB).

### D-19 RESULT: PASS by the text-type rule only (Fri 02:48; CPU, sticky/d19_switch.py -> data/d19_switch.txt)
- Tokens per pass vs DFlash2 L1-20 (MTP K15 = the native chain; D-17's every-position data, same walk):

| rule | code | gsm | ifeval | multi | agent |
|---|---|---|---|---|---|
| type (MTP on ifeval / multi / agent) | +0.00 | +0.00 | +0.19 | +0.70 | +0.97 -> PASS |
| bandit, partial feedback, W 2 / 4 / 8 | -0.46 / -0.10 / -0.16 | -0.29 / -0.37 / -0.56 | +0.10 / +0.13 / +0.18 | +0.52 / +0.54 / +0.50 | +0.67 / +0.68 / +0.63 |
| bandit, full feedback, W 2 / 4 / 8 | -0.36 / -0.26 / -0.60 | -0.34 / +0.06 / -0.27 | +0.16 / +0.16 / +0.15 | +0.51 / +0.48 / +0.57 | +0.67 / +0.71 / +0.61 |
| both (type start + full bandit), W 2 / 4 / 8 | -0.36 / -0.26 / -0.60 | -0.34 / +0.06 / -0.27 | +0.16 / +0.15 / +0.13 | +0.52 / +0.51 / +0.64 | +0.68 / +0.85 / +0.81 |

- PASS (>= +0.4 on chat or agent with code / math >= -0.1) only for the type rule. Every bandit costs code / math
  0.1-0.6 per pass: it keeps trying MTP there, where an MTP pass costs ~2.7 tokens on code, and switches back too late.
  On chat / agent the bandits keep 70-90 % of the type rule's gain.
- Prediction: type passes (yes); bandits +0.5-0.8 on agent (yes, +0.61-0.85); code -0.05 to -0.3 (worse: down to
  -0.60); "a bandit without the type passes too" 50 % (no).
- So the engine needs the text type up front (an agent / tool turn, or the prompt's language), not a learned switch.
  At 45 GB (estimate, not measured): MTP K15 adds 15 sequential MTP-layer steps per pass; MTP K7 halves that and keeps
  94 % of the multi gain (2.88 vs 2.92) but 66 % of the agent gain (3.91 vs 4.24, DFlash2 3.28).

### D-16 RESULT: KILL at the rapid protocol (Fri 05:33; arms 01:44-03:34, turns/turn36.sh + turn38.sh; data/d16_loss.txt)
- Paired per-step training loss vs D-13's U on the same batches (each step's loss taken before its update; 90 %
  bootstrap CI over steps 11-21):
  - H, hc-lr 1e-3: diverged, +4.72 (CI +4.44 to +5.01); walks 1.27 / 1.18 / 1.10 per check (U 9.40 / 6.71 / 2.94).
  - Hc, hc-lr 5e-6 (fan-in-matched): +0.144 (CI +0.111 to +0.175); worse than U on every step from step 4.
  - Hb, hc-lr 1.5e-6: not finished (the Mac slept mid-run; killed before its first checkpoint). Not re-run: the delta
    shrinks with the rate but never turns negative, so a lower rate can at best approach U.
- No arm beats U, so no walks: KILL by the pre-registered screen. The new 26M-weight projection adds per-batch noise
  to every context row faster than 21 steps can turn it into signal. Prediction was PASS 30 / BETWEEN 35 / KILL 35.
- Caveat (pre-registered): 21 steps can only under-read a new input. A full 2-epoch run (~1-2 h of GPU) is the only
  remaining test of the input itself; not proposed now, because D-17 / D-19 put the native head's lead on chat /
  agent down to its chain (a switch captures it), not to an input DFlash2 lacks, and code (the 45 GB headline) has
  no MTP lead to transfer (DFlash2 0.986 vs MTP 0.977 at depth 1).

## D-22 MTPTTT: train the native MTP head the way it chains (MTP Head, Fri 09:53) -- pre-registration (written before any training)
- Why: Kamen's goal is the best MTP head. The native head is target-class at depth 1 (code 0.977, DFlash2 0.986) but
  its per-depth rate falls to 0.869 by depth 7 and 0.697 by depth 15 (D-17), so its chain keeps 7.78 per pass on code
  vs DFlash2's 10.44. SD (Leap) says only per-token quality near 0.96-0.98 reaches 12-14 at block 16; a target-class
  head that held ~0.97 per depth would keep ~12.9 on code. The 10-22 fine-tune (mtp125n_d4) killed depths 2-4 because
  it trained depth j over a fresh cache of depth-(j-1) outputs, while the chain at inference attends to the TRUE-PAIR
  history plus its own chain entries (REPORT 'The 125B MTP head fine-tune'). Papers that train the chain the way it
  runs (AngelSpec, FastMTP on Qwen3-Next-80B, Microsoft 2610.00888) gain at depths 2-3.
- Design (sticky/d22_ttt.py; mtp125.py's plain-MLX head forward, fp32 copies of fc_hidden, fc_embedding, attention
  q/k/v/o, router, shared expert; frozen 4-bit experts, HC mixers, norms, lm_head): the EAGLE-3 training-time test
  inside the head's attention. Depth 1 = causal over true pairs (trunk widened stream at u + token u+1). Depth j for
  anchor s: input = the head's own depth-(j-1) output + token s+j (teacher-forced: on the accepted prefix the chain's
  own tokens ARE the true ones), query at position s+j-1, keys = true pairs u <= s plus this anchor's own chain entries
  at s+1..s+j-1 (the MTP cache at inference). KL to the 125B's top-32 at row s+j, depths 1-8, assistant targets only.
- Data: L1-20's 99 training windows (80 code, 29 math overall; the 125B's own outputs) with D-16a's widened stream;
  no new 125B generation. Held out: the 24 HumanEval walk samples (L1-20 held them out) and the 60 GSM8K ones.
- Validation gate (before any training claim): the untrained head under my forward reproduces MTPLX's real chain
  (D-17's stored 15-token chains on the 8 code + 8 gsm prompts, same positions): argmax agreement >= 0.98 at depth 1
  and >= 0.95 over depths 1-7. Else the forward is fixed first.
- Read-out: the chain from the true prefix at every position of the held-out sets, walked (tokens per pass, MTP K7 /
  K15) as in D-17; trained vs untrained head, 90 % bootstrap CI over samples.
- Lines: PASS = code K15 >= 1.15x the untrained head with gsm >= 1.00x (then the agent data question and a 45 GB
  pricing with the D-19 switch); BIG if code K15 reaches DFlash2's level (>= 10.4). KILL = < 1.05x on code and gsm.
- Prediction: depths 2-7 gain (the papers' size: +5-15 points at depth 3), code K15 7.8 -> 8.5-9.5 (1.10-1.20x),
  gsm 1.05-1.15x; DFlash2's level is unlikely in one short run. PASS 40 %, KILL 30 %.
- Guards: training in <= 15-min quiet-lock chunks, checkpointed and resumable; process memory <= 36 GB; stop if swap
  grows > 2 GB; any shipped head stays bf16 / 8-bit (REPORT 10-22: 4-bit re-storage erases most of a fine-tune).

### D-22 interim (Fri 19:39): the training-time-test MTP head PASSES on code by my forward; real-runtime confirmation running
- Gate (data/d22_gate.txt): my forward with the untrained head reproduces MTPLX's real chain (D-17's stored chains, 16
  prompts): argmax agreement depth 1 0.993, depths 1-7 0.974 (lines 0.98 / 0.95); it drifts to 0.86-0.93 at depths 9-15
  (fp32 vs bf16 compounding), so the K15 claim also gets a real-runtime read-out (sticky/d22_real.py).
- Training run 'base' (sticky/d22_ttt.py: depths 1-8, lr 1e-5, wdecay 0.8, 99 windows, 69.1M trainable, ~3 s per step,
  peak 13 GB). Paired screen at step 76 (0.77 epoch; data/d22_loss_*.npz): online held-out loss below the untrained
  head's on the same unseen windows in every 20-step block (-0.10 to -0.17, CIs < 0); validation windows' KL per depth
  0.059 / 0.181 / 0.221 / 0.415 / 0.563 / 0.791 / 1.043 / 1.301 -> 0.036 / 0.100 / 0.140 / 0.215 / 0.262 / 0.301 /
  0.330 / 0.383.
- Read-out with my forward (held-out answers; no answer shares a 32-gram with any training window), paired vs untrained:

| head | code (24) K7 | code K15 | gsm (60) K7 | gsm K15 |
|---|---|---|---|---|
| untrained | 6.17 | 7.54 | 5.48 | 6.54 |
| step 76 | 6.54 = 1.059x (1.033-1.084) | 9.21 = 1.221x (1.160-1.290) | 5.50 = 1.004x | 6.58 = 1.006x (0.988-1.025) |
| step 198 (2 epochs) | 6.24 = 1.010x | 8.71 = 1.154x (1.094-1.225) | 5.28 = 0.964x | 6.33 = 0.968x (0.949-0.988) |

- So step 76 meets the PASS line (code K15 >= 1.15x with gsm >= 1.00x); longer training (2 epochs, two optimizer
  restarts at resume) over-fits the 99 windows and costs gsm. Next: a run with a decaying lr stopped near 1 epoch, and
  chat / agent data (the switch uses the MTP chain there). Not yet DFlash2's level on code (10.4): no BIG.
- Real runtime (data/d22r_base76, MTPLX with the step-76 head installed, the same greedy decode as D-17): 8 code + 1 gsm
  prompts decoded so far (per-prompt depth-1 hit 0.936-0.997); the paired read-out follows when the gsm prompts finish.

### D-22 REAL-RUNTIME RESULT (Fri 20:54; data/d22r_base76_vs_stock.txt; quiet lock 19:02-19:16, turns/turn46.sh): the trained head holds ~0.94 per depth
- MTPLX's own runtime (spec125, k_mtp 15, the same greedy decode as D-17) with the step-76 head installed in bf16, on D-05's
  held-out prompts; every answer token-identical to the stock run (drafts only: exact).
- code (8 answers): tokens per pass K3 3.81 -> 3.86 (1.015x), K7 6.21 -> 6.87 (1.105x, CI 1.070-1.142), K15 7.78 -> 9.83
  (1.263x, CI 1.185-1.366). Conditional rate per depth, stock -> trained:
  0.98>0.99 0.97>0.98 0.95>0.96 0.93>0.96 0.93>0.95 0.90>0.95 0.87>0.94 0.84>0.94 0.83>0.94 0.79>0.94 0.75>0.94 0.75>0.94
  0.72>0.94 0.75>0.94 0.70>0.93: the stock chain decays to 0.70 by depth 15; the trained one stays at ~0.94 through it.
- gsm (8 answers, identical; decoded 21:02-21:10 on a shared GPU at C 51): K3 3.47 -> 3.49 (1.004x), K7 5.27 -> 5.41
  (1.027x), K15 6.07 -> 6.36 (1.048x, CI 1.001-1.109); per-depth rates about unchanged (0.94 / 0.91 / 0.90 at depths
  1-3, 0.68 -> 0.72 at 15).
- VERDICT: PASS on the pre-registered line in MTPLX's own runtime (code K15 >= 1.15x: 1.263x; gsm >= 1.00x: 1.048x).
- Against DFlash2 L1-20 on the same 8 code answers (D-17's every-position walk): 10.44 per pass. The trained MTP chain is at
  94 % of DFlash2 on code after 12 minutes of training on ~25K tokens; on chat / agent the stock chain already beats
  DFlash2 by ~30 % (D-17 / D-19) and the switch (D-20) runs it there.

### D-22 run b3 (Fri 22:38; shared GPU training 21:42-22:38): depths 1-15 + chat windows + one cosine epoch
- sticky/d22_ttt.py train --run b3 --data l120,d22c --depth 15 --wdecay 0.9 --epochs 1 --sched cos: L1-20's 99 code / math
  windows + 73 IFEval windows (the exact ref's answers; D-20's prompts held out), 172 steps, peak 25 GB.
- Read-out with my forward (held-out; paired vs the untrained head, 90 % CI):
  code (24) K7 6.70 = 1.085x (1.051-1.116), K15 10.12 = 1.341x (1.275-1.416); gsm (60) K7 5.85 = 1.067x (1.053-1.081),
  K15 7.57 = 1.157x (1.137-1.178). Step 76 of 'base' had code K15 1.221x, gsm 1.006x: the deep-depth training, the
  decaying lr and / or the extra windows lift both (not separated). Real-runtime read-out (32 prompts) next.

### D-22 b3 REAL-RUNTIME RESULT (Fri 23:39; quiet locks 23:15-23:38, turns/turn54.sh; data/d22r_b3_vs_stock.txt): the MTP head beats DFlash2 on code
- MTPLX's runtime with b3 installed, D-05's 32 held-out prompts, every answer token-identical to the stock run (exact):

| set (8 answers) | K7 stock -> b3 | K15 stock -> b3 | K15 ratio (90 % CI) | DFlash2 L1-20 on the same answers |
|---|---|---|---|---|
| code | 6.21 -> 6.98 (1.124x) | 7.78 -> **10.98** | 1.411x (1.287-1.565) | 10.44 |
| gsm | 5.27 -> 5.53 (1.049x) | 6.07 -> 6.86 | 1.131x (1.093-1.180) | 6.60 |
| ifeval | 2.42 -> 2.48 | 2.43 -> 2.49 | 1.024x (0.976-1.072) | 2.24 |
| multi | 2.88 -> 2.94 | 2.92 -> 3.00 | 1.028x (0.994-1.062) | 2.22 |

- Code: the per-depth rate is ~0.97 at EVERY depth 1-15 (stock 0.98 -> 0.70): a target-class chain. K15 10.98 is above
  DFlash2 L1-20's 10.44 on the same answers (block 16 = the same 16-row check): the pre-registered BIG line (>= 10.4) is
  met. gsm 6.86 is above DFlash2's 6.60 too. Chat (ifeval / multi) barely moves: 73 IFEval windows are not enough.
- For 45 GB tok/s (estimate): on the exact 16-row route the MTP chain's 15 sequential steps cost ~20 ms vs DFlash2's
  12.5 ms (D-05), so code is about even there at today's ~150 ms checks; on the shipped MTP line (H50, adaptive width
  <= 7 drafts) the K7 gain (+12 % code, +5 % gsm) applies directly: that A/B (turns/turn53.sh) waits for >= 50 GB
  available.

### D-22 on the shipped 45 GB line, first A/B (Sat 00:26; quiet lock 00:22-00:26, turns/turn53.sh; data/d22_ship_b3.txt): ORDER-CONFOUNDED, redo queued
- H50 stack (BigMoE 21:56) at C 160 (NO-SWAP short of C 256's need), adaptive width, 9 prompts x 256 tokens, the stock head
  installed vs b3 installed (same code path); outputs identical across arms on every prompt (exact).
- tok/s stock -> b3: code (4) 42.9 -> 47.1 (+9.7 %), gsm (2) 28.4 -> 37.9 (+33 %), ifeval (2) 38.3 -> 42.7 (+12 %),
  multi (2) 23.7 -> 25.6 (+8 %). NOT a result: b3 always ran second on a prompt the stock arm had just warmed (expert
  cache), and code / ifeval got faster while keeping FEWER tokens per pass (4.98 -> 4.31, 3.35 -> 3.07).
- The fewer tokens per pass are a finding: the shipped width rule (AdaptiveWidth: keep step j while the head's product of
  probabilities ^ 2 pays for its row) stops b3's chains earlier. b3 is trained to the 125B's own (softer) distribution, so
  its p is lower even where its argmax is right: its confidence is calibrated differently from the stock head's. The
  shipped line would need the width rule re-fitted (alpha / V0) for b3, or a fixed chain.
- Redo (turns/turn55q.sh, after D-20): a discarded warm-up then A-B-A per prompt, drafted / accepted per pass logged, in
  the shipped adapt mode and with a fixed 7-draft chain (k7).

### D-20 SWITCH45 RESULT: PASS (Sat 02:02; quiet tickets 00:51-01:5x, turns/turn47.sh; data/d20_runs.jsonl)
- Leap's REF2b exact live loop (copy: sticky/d20_switch45.py, Leap reviewed), C 160 (the NO-SWAP fallback; C 224 did not
  fit), 256 tokens, the trained MTP head b3; per prompt arms interleaved: dfl (DFlash2 L1-20, 16-row checks) / mtp7 (the
  MTP chain, 8-row checks) / mtp15 (16 rows). Every arm's tokens identical on all 11 prompts; == REF2 rows where they exist.
- Mean wall-clock tok/s by type: chat (5: ifeval x3, multi x2) dfl 10.91 / mtp7 14.86 (+36.2 %) / mtp15 11.76 (+7.8 %);
  agent (2) 10.51 / 15.87 (+51.0 %) / 16.08 (+52.9 %); code (2) 25.12 / 20.40 (-18.8 %); math (2) 16.78 / 18.07 (+7.7 %).
  Two mtp15 runs (HumanEval/91, math500/1282: 1.2 tok/s) were frozen by a pause and are excluded from any comparison.
- Why mtp7 wins on chat / agent: it keeps about as much per check as the 16-row arms (chat 2.4-6.8 vs 2.1-9.0) at an
  8-row check that is ~15-25 % cheaper (e.g. multi/bg0 check 185 vs 220 ms), for ~5-8 ms more drafting. On code
  DFlash2 keeps 15.4 / 16 on HumanEval/68 and its 16-row check pays.
- Verdict on NewIdeas' line: PASS (chat / agent mean +36 % / +51 % with drafting cost included; code / math stay on
  DFlash2 under the switch, so unchanged). NewIdeas' arms: (c) the switch (DFlash2 on code, MTP K7 on chat / agent) beats
  (b) MTP everywhere (code -19 %). Caveats: C 160, 2-5 prompts per type, one run per arm, night (quiet locks).

### D-22 b4 REAL-RUNTIME RESULT (Sat 10-31 04:45, read Sun 01:02; turns/turn56.sh + turn57.sh; data/d22r_b4_vs_stock.txt)
- b4 = b3's recipe with ALL 147 chat windows (IFEval 97 + MGSM 50) + L1-20's 99 (246 steps). MTPLX runtime, D-05's 32
  prompts, answers identical. K15 stock -> b4: code 7.78 -> 10.65 (1.368x, CI 1.284-1.469; b3 10.98), gsm 6.07 -> 7.04
  (1.161x, CI 1.102-1.236; b3 6.86), ifeval 2.43 -> 2.51 (1.031x, n.s.; b3 2.49), multi 2.92 -> 3.03 (1.038x, CI
  1.013-1.060; b3 3.00). K7: code 1.118x, gsm 1.060x, ifeval 1.033x, multi 1.039x (CI 1.006-1.069).
- Twice the chat windows move chat by ~1 % more: chat's wall is the per-token accuracy at depth 1 (ifeval 0.72, multi
  0.80 vs code 0.98), not the chain decay that the training-time test fixed. b3 stays the code head; b4 is the better
  all-round head (gsm / multi up, code -3 %). The chat follow-up fits NewIdeas' CHAT-NT1 (accept-set training).
- The shipped-line A-B-A redo did not run on 10-31 (seven grants 02:01-02:04, NO-SWAP short each time); re-queued
  Sun 01:0x (turns/night2_q.sh, 15-min waits after a short grant).

### D-22 shipped-line A-B-A redo (Sun 11-01 02:25; quiet tickets 01:45-02:24, turns/night2_q.sh -> turn53.sh; data/d22_ship_b3_{adapt,k7}_aba.json): b3 pays in a fixed chain only; the shipped width rule hides it
- H50 stack at C 160 (the NO-SWAP fallback), 10 prompts x 256 tokens; per prompt one discarded warm-up decode, then A-B-A
  (A = the stock head installed through the same code path, B = b3). Outputs identical across all arms on every prompt
  (exact). Tokens per pass are deterministic (A1 == A2 on every prompt), so they are the clean read; tok/s = B over the
  mean of A1 and A2 (one stalled A1, k7 multi/bg0 at 1.9 tok/s, replaced by its A2).
- k7 (a fixed 7-draft MTP chain, no lookup), tokens per pass stock -> b3: code (4) 5.59 -> 6.28 (+12.2 %), gsm (2)
  4.17 -> 4.50 (+7.8 %), ifeval (2) 4.27 -> 4.50 (+5.5 %), multi (2) 1.77 -> 1.77 (0). tok/s geo-mean ratio: code 1.125
  (4 of 4 faster), gsm 1.080, ifeval 0.982, multi 0.982.
- adapt (the shipped rule: k_mtp 1 + n-gram lookup + AdaptiveWidth): code 4.75 -> 4.33 (-8.9 %: prompt_code 3.16 -> 1.58,
  tok/s 0.72x; the other three 1.00-1.07x), gsm 1.58 -> 2.28 (+44 %: gsm/600 1.88 -> 3.28, 1.24x), ifeval -2.2 %, multi
  0. tok/s geo-mean ratio 0.944 / 1.119 / 0.974 / 1.007.
- Read: the shipped rule drafts ONE MTP token per pass and fills the rest from the n-gram lookup, so b3's chain (depths
  2-15, the thing the training-time test fixed) is never used there. What is left is the width rule's reaction to b3's
  softer depth-1 probability, which swings single prompts both ways. adapt also beats k7 on multi (k7 drafts 7 and keeps
  0.5-1.0 of them: multi/ar21 12.3-12.8 tok/s in k7 vs 22.7-23.2 in adapt).
- Verdict: b3 does not go into the H50 adapt line as it is. It pays in a fixed chain: here code +12.5 % tok/s, and on the
  45 GB REF2b loop the switch (D-20, D-20b below). For the shipped line it would need a width rule fitted to b3's chain
  probabilities (k_mtp up to 7, alpha / V0 re-fitted) or a per-type k (k7 on code / gsm, adapt on multi). BigMoE's 24 GB
  screens (Sun 05:20-05:43) agree: b3 keeps +5-8 % per pass there, tok/s flat (24 GB decode is SSD-bound per token).

### D-20b SWITCH45 confirmation: PASS at C 224 (Sun 11-01 03:22; quiet tickets 02:24-03:22, turns/night3_q.sh -> turn58.sh; data/d20_runs.jsonl, data/d20b_summary.txt)
- 18 new prompts (4 HumanEval, 4 MATH-500, 3 IFEval, 3 multi-turn, 4 agent), the REF2b exact loop, head b3, 256-token
  cap, arms dfl (DFlash2 L1-20) / mtp7. 17 prompts have both arms at both C 160 and C 224 (agent/edit02 has dfl only: its
  57 s prefill ran the last ticket out). The night queue's counter ignored C while the harness's resume key has it, so the
  set ran at C 160 (the first grant's NO-SWAP tier) and again at C 224: two tiers instead of one.
- Outputs identical across arms at both tiers; == the REF2 rows where they exist. NewIdeas' text rule routed all 17 to
  their dataset's type.
- C 224 (the 45 GB tier), mean tok/s dfl -> mtp7: code (4) 31.6 -> 27.7 (-12.2 %, mtp7 faster on 0 of 4), math (4) 23.1 ->
  27.3 (+18.3 %, 4 of 4), chat (6) 12.9 -> 16.2 (+26.0 %, 6 of 6), agent (3) 14.0 -> 17.8 (+27.0 %, 3 of 3). Whole set
  (3602 tokens): always DFlash2 16.0 tok/s, always MTP 19.8, the switch 20.3 = +27.3 %.
- C 160: code -3.2 % (1 of 4), math +22.2 % (4 of 4), chat +33.5 % (6 of 6), agent +2.4 % (agent/bash18's mtp7 run stalled
  at 2.3 tok/s, 14.3 at C 224; the other two +17 / +45 %); the switch +10.1 % over the set.
- Verdict: PASS. The switch holds on new prompts, and math is now clearly an MTP type (D-20: +8 % on 2 prompts; here
  +18-22 % on 4 of 4). Rule for the 45 GB route: DFlash2 on code, the MTP chain K7 with b3 on math / chat / agent.
  Caveats: one run per arm, night quiet locks, 256-token cap.

### RANKED_45GB #20 BIGDRAFT: a DFlash2-style drafter 4-8x larger (MoonShots' bet) -- desk judgment (Sun 11-01 08:2x; NewIdeas' ask)
- Size: DFlash2 = 5 layers x (hidden 2560, FFN 7680, 32 / 8 heads) ~ 0.43B + fc ~ 0.03B = ~0.47B (embedding / head shared
  with the 125B). 4-8x = 1.9-3.8B: over the local-training line (~1B), so the bet as stated is a paid run, not local.
- Data: a drafter trained from scratch at that size needs on the order of 100M-1B tokens of the 125B's own answers (DFlash /
  EAGLE-3 recipes train on ~0.5-1M regenerated samples). Here the 125B writes ~20-50 tok/s, so ~1-2.5M tokens per day of an
  otherwise idle Mac: 100M tokens = months. The answers on disk today (l120 99 + d22c 147 + d16w / d02a windows) are ~0.3M.
  Storage is fine only after the frozen fc (5 KB per token: 1M tokens = 5 GB; raw taps are 26 KB per token).
- Expected gain, from prior art and our own data (not from the twin ceilings):
  - DFlash's own capacity table (Leap TABLE_350 row (iv)): 3 / 5 / 8 layers = Math500 5.64 / 5.99 / 6.33 tokens, MT-Bench
    3.37 -> 3.50 (5 -> 8 layers); Scylla (2505.07858): log-linear in drafter capacity up to ~10 layers. That is ~+6 % kept
    per 1.6x on math and ~+4 % on chat, so 4x ~ +12-18 %, 8x ~ +17-27 %, not 2x. Leap marked depth scaling dead (net ~0)
    because drafting cost grows with layers; on our SSD-bound cycle a WIDER 4-bit drafter costs only ~+5-10 ms per pass
    (2-4 % of a 150-300 ms cycle), so net +8-20 % is the plausible range.
  - Why capacity will not reach the twin ceilings on chat: in CHAT-NT1 (R-61N, below) 69 % of chat's first misses fall where
    the 125B itself is < 0.9 sure, i.e. its own near-ties, which only the 125B's own arithmetic decides. The MTP head already
    has far more capacity and the trunk's final hidden state, and it reaches chat depth-1 0.72-0.80 (DFlash2 0.50-0.63), yet
    keeps 3.08 per 8-row check on chat vs DFlash2's 2.91 per 16 (D-20b): more depth-1 skill moved chat little.
- Verdict: not local (size + data); as a paid run it is a proposal for Kamen with an expected +10-25 % kept on code / math and
  +10-17 % on chat. Cost sketch (rented 8x H100-class node, ~$2.5-3 per GPU-h): 300-500M tokens of the 125B's answers ~15-30 h
  ($400-800, if the 125B runs in vLLM / SGLang there), drafter training 2-4B with the target's taps ~10 h (~$250): ~$650-1,050
  plus pipeline work; downloads and rentals are Kamen's call. It ranks below b5 (AUF), the exact top-3 / tree lever (34 % of
  chat's first misses have the 125B's #1 among DFlash2's top-3, CHAT-NT1) and the switch on the default (D-23).

## R-61N CHAT-NT1: near-tie (NON-exact) acceptance on chat (MTP Head, Sun 11-01; NewIdeas' QUEUE row from the orchestrator's night brief) -- CPU/GPU replay on stored captures
- Pre-registered in the QUEUE row (written before the run): PASS = chat tokens per pass +30 % or better at rho 0.25; KILL =
  under +10 % at rho 0.1. Method = a copy of NewIdeas' nt_accept.py + nt_analyze.py (sticky/r61n_chatnt.py): L1-20 with
  cand21's selector at t 0.0, teacher-forced on D-02a's captures (the 125B's own top-32 per position), forgive a first miss
  when the drafter's token has p_d >= rho x p_top1; LB = +1 token per forgiven block, MOD = the run continues with the exact
  replay's survival (an estimate). Added: the drafter's top-3 at each first miss. Shared GPU, 08:13-08:24.
- Harness check: exact tokens per pass ifeval 2.44 / multi 2.05 / agent 3.00 at block 16 = D-02a's walk (2.45 / 2.06 / 3.01).
### R-61N RESULT: BETWEEN (Sun 11-01 08:24; data/r61n/chatnt_b16.txt, chatnt_b32.txt; block 32 = block 16 within 1 %)
- Chat (ifeval + multi, 6,145 blocks from 54 answers), block 16: exact 2.23 tokens per pass. First misses by the 125B's
  top-1 p: >= 0.99 17 %, 0.9-0.99 15 %, 0.5-0.9 40 %, < 0.5 29 % (code's NT-1: 56 % at >= 0.99). So chat misses are mostly
  where the 125B is unsure, but the drafter's token is usually far from its choice: median p_d 0.000-0.021.
- Forgiven share / tokens per pass (LB .. MOD): rho 0.5 9 % (+4 .. +10 %); rho 0.25 18 % (+8 .. +23 %); rho 0.1 27 % (+12 ..
  +39 %). By set: ifeval rho 0.25 +10 .. +32 %, multi +6 .. +15 %, agent +4 .. +16 %, mgsm +3 .. +14 %, gsm +3 .. +18 %.
- Verdict on the pre-registered line: BETWEEN (chat rho 0.25 MOD +23 % < +30 %; rho 0.1 MOD +39 % > +10 %). No GPU decode +
  quality test is proposed from this alone.
- The drafter's top-3 oracle (any of its three best inside the accept set): chat rho 0.25 forgives 49 % (+22 .. +102 %), rho
  0.1 57 % (+25 .. +136 %). The accept-set follow-up (a drafter trained so its #1 lands in the set) has that much headroom,
  non-exact, Kamen's call.
- EXACT lead from the same data: the 125B's own #1 is among DFlash2's top-3 at 34 % of chat's first misses (ifeval 40 %,
  multi 28 %, agent 37 %, mgsm 37 %, gsm 65 %): +15 % (LB) .. +54 % (MOD) tokens per pass on chat if a 2-3 way branch at
  uncertain positions could be verified (draft trees, RANKED #10; cost = extra rows per check), or if a reranker fixed the
  order. That lever keeps every token, so it ranks ahead of near-tie acceptance.

### D-23 SWITCHDEF RESULT: KILL (Sun 11-01 09:11; quiet tickets 08:24-09:11, turns/after_d23.sh -> turn59.sh; data/d23_switchdef.txt, d23_runs.jsonl)
- The 45 GB default as it ships (Leap's k69_live R57_PRESET=45: REF2-T 4433333333 / 3c, C_l 112, gate MAXK 160, build_next,
  L1-14 truncation t 0.3) at C 224 with head b3 resident; dfl (the default) vs mtp7 (8-row checks), D-20b's 18 prompts, order
  alternating, one discarded warm-up per arm. 18 / 18 clean; every token equal across arms; HumanEval/16 and /19 == REF2-T AR.
- dfl -> mtp7 by type (mean tok/s): code 50.2 -> 47.0 (-6.5 %), math 43.7 -> 40.7 (-6.7 %), chat 26.6 -> 23.5 (-11.8 %),
  agent 20.2 -> 21.3 (+5.6 %). Whole set: dfl 29.4 tok/s, mtp7 everywhere 27.8, the switch (dfl on code only) 28.6 = -2.7 %
  -> KILL (line +4 %). Peak 42.4 GB.
- Why D-20b's +27 % (REF2b, no cut) does not carry over: REF2-T makes dfl 1.84x faster (16.0 -> 29.4) but mtp7 only 1.37x
  (20.3 -> 27.8). The default's DFlash2 verifies only rows whose running confidence stays >= 0.3 (chat 3.9 rows; check
  76-85 ms), while mtp7 verified a fixed 8 (check 79-102 ms) and drafts in 17.5 ms vs 11.5. mtp7 still keeps more on chat
  (3.07 vs 2.71 per check): the loss is the verify width, not the head.

### D-23b MTP TRUNC: CPU replay of D-23's checks with the default's cut on the MTP chain (Sun 11-01 11:4x; sticky/d23_trunc.py, data/d23_trunc.txt)
- Per logged check (rows, kept, check ms, draft ms, the chain's per-step top-1 p): rows(t) = 1 + steps whose running product
  of top-1 p is >= t, kept(t) = min(kept, rows(t)) (greedy verify: exact by construction). Check cost 7.82 ms per verified row
  (fixed-effects fit on the dfl arm's checks; it predicts the mtp7 arm's mean 8-row check at a median ratio 0.999, per prompt
  0.39-1.49). Overhead per check kept as measured. The estimator reproduces D-23's whole set (dfl 29.5 vs 29.4 measured, mtp7
  28.0 vs 27.8). Draft (A) all 7 steps as now; (B) early exit, 2.2 ms saved per step not drafted (D-20: K7 17-21 ms, K15
  37-43 ms) after 0.3 ms per extra sync.
- Type sums (tokens / time; the dfl row): code 51.1, math 43.9, chat 22.9, agent 20.7. MTP with the cut at t 0.3: (A) code
  -18.7 %, math -5.5 %, chat +27.8 %, agent +6.0 %; (B) code -18.2 %, math -3.8 %, chat +50.5 %, agent +7.2 %. t 0.1-0.4 is
  flat (chat +25-30 % A, +43-50 % B); t 0 = D-23 (chat -4 %).
- The switch (dfl on code + math, MTP + cut on chat + agent): (A) +12.2 %, (B) +19.4 % whole set at t 0.3; leave-one-prompt-
  out t per type +13.3 % (A) / +19.4 % (B). Replay estimate, in-sample prompts: D-23c measures it (turn62, quiet, tonight).
- D-23c = d23_switchdef.py + arm mtp7e (each step evaluated, drafting stops at the first step whose running product < 0.3,
  the leading drafts at >= 0.3 verified = the dfl arm's rule; mock-tested against draft + cut: 300 / 300 equal, cache trimmed
  back). D-23's 6 chat + 4 agent prompts, dfl vs mtp7e; code / math dfl times from D-23. PASS (pre-registered): the switch >=
  +10 % whole set with every token equal; KILL < +4 %.

### D-22 b5 (Spec-AUF) trained; read-out pending (Sun 11-01 09:28-10:14, shared, turns/turn60.sh)
- b5 = b4's recipe (L1-20's 99 + 147 chat windows, depths 1-15, wdecay 0.9, lr 1e-5 cosine, 1 epoch) + --auf: each anchor's
  loss stops after its chain's first argmax failure (the failing depth kept), depths weighted by their alive share. 246 steps
  (two shared slots, resumed at 156), peak 24.9 GB.
- Read-out so far (teacher-forced, walked as D-17): code K7 6.72 / K15 10.12 (b3 6.70 / 10.12), gsm K7 5.88 / K15 7.66 (b3
  5.85 / 7.57). The chat sets crashed the eval (an answer of 5 tokens has a 2-depth chain; the depth-3 hit read indexed past
  it): fixed (nan for missing depths). b4 has no teacher-forced read-out yet (only MTPLX's real runtime), so turn61 runs b4 and
  b5 on all five sets (shared, when NO-SWAP >= 21 GB; day4_chain.sh).

### D-22 b5 (Spec-AUF) RESULT: small, math-side (Sun 11-01 13:23; shared 13:11-13:23, turns/turn61.sh; data/d22_eval_b5.txt, d22_eval_b4.txt, d22_b5_vs_b4.txt)
- b4 vs b5 differ only in the loss support (AUF: each anchor's CE stops after its chain's first argmax failure). Teacher-forced
  chain on the five held-out sets, paired per answer, 90 % CI: code K7 1.001x (0.984-1.019), K15 1.007x; gsm K7 1.014x
  (1.001-1.027), K15 1.019x (0.998-1.041); ifeval K7 1.003x, K15 1.003x; multi K7 1.007x, K15 1.015x (0.999-1.032); mgsm K7
  1.023x (1.012-1.033), K15 1.030x (1.018-1.042). Depth-1 hit unchanged (ifeval 0.726, multi 0.78, code 0.981).
- Verdict: AUF at greedy T 0 = +0-3 %, significant on math only (the paper's +9 % tau is for DFlash at T 1). b5 is the
  all-round head (>= b4 on every set, = b3 on code: K7 6.72 vs 6.70, K15 10.12 vs 10.12; gsm K15 7.66 vs b3 7.57). It does
  not move chat (the depth-1 wall). No real-runtime run for +1-3 % alone; b5 replaces b3 in later MTP arms after D-23c
  (which stays on b3, the head of D-23's logged checks). Eval peak footprint 6.8-10.8 GB (gate 20 GB was right).

### D-24 MTPTOP3: the MTP head's top-3 at chain misses + cut / tree policies (Sun 11-01 13:44-13:51 shared; sticky/d24_mtptop3.py, d24_read.py; data/d24/, data/d24_read.txt)
- Head b5, teacher-forced chain on the held-out answers (exact up to the first miss, and along a rescued branch past it).
  Depth 1, the true token at rank 0 / <=1 / <=2: ifeval 0.756 / 0.867 / 0.908, multi 0.784 / 0.879 / 0.913, mgsm 0.924 /
  0.965 / 0.978, gsm 0.961 / 0.989 / 0.996, code 0.978 / 0.992 / 0.997. At first misses (any depth) the true token is the
  head's #2 or #3: chat 0.44-0.48 (DFlash2 0.34, R-61N), math 0.60-0.76, code 0.76.
- Policies scored with D-23's cost model (check F + 7.82 ms x rows, 2.2 ms per drafted step, F fitted per type), tok/s vs the
  cut chain K7 (t 0.3): chain K7 without the cut ifeval -28 % / multi -27 % (the D-23 loss); K15 + cut code +14.5 %, gsm
  +4.2 %, mgsm +2.3 %, chat +0.1-0.6 %; leaf branches (rank-1 / rank-2 candidates at kept depths with p < 0.5 or 0.8, no
  extra draft steps, a tree-aware verify assumed at chain-row cost) -7 % .. +0.9 % on every set: a rescued token is worth
  ~34 ms on chat and a leaf row costs 7.8 ms, so a leaf needs > ~23 % rescue odds, which spread leaves do not reach.
  Oracle width 2 / 3 at every depth (no row budget): chat 3.6 / 4.05 tokens per pass vs 2.3-2.5 cut (the headroom a cheaper
  verify row would open).
- Consequence for tonight: D-23c becomes v2 (pre-registered in sticky/d23c_mtptrunc.py before any run): dfl vs mtp15e (K15 +
  the cut + early exit), head b5, all 18 of D-23's prompts, one 30-min quiet ticket (turns/turn62.sh, night5_chain.sh);
  PASS = S1 (dfl on code + math, mtp15e on chat + agent) >= +10 % whole set, every token equal; KILL < +4 %; S2 (dfl on code
  only) and mtp15e everywhere reported too. No tree arm.
- D-24 addendum (14:1x, CPU): the cut threshold for K15 + cut, est. tok/s by t 0.1-0.7 per set: the best t is 0.3 on ifeval /
  multi / mgsm, 0.4 on gsm (+1.1 %), 0.2 on code (+0.3 %); the curve is flat around 0.3, so D-23c keeps the default's t 0.3.

### FR-Spec-style reduced draft vocab for the MTP head: dropped at the desk (Sun 11-01, CPU; written up 16:1x for ~/TRIED)
- Idea: the MTP head drafts over the top-K most frequent tokens only (FR-Spec), the 125B still verifies on the full vocab, so
  outputs stay exact. Desk read (CPU coverage check, recorded here from the session's notes): a smaller lm_head saves only
  ~1-3 % of a check at 45 GB, because the verify dominates the cycle. A 32K vocab covers only 0.53 of the multilingual
  answers' tokens, so chat would lose drafts. KILL at the desk; no GPU run.

> (a quote from the project owner, kept private)
- Build (sticky/e5x.py, e5_train.py): L1-20's body frozen; after drafter layer i the block's hidden state is normed and projected
  (in_i, identity init) into the 125B's MoE input at tap layer T_i (5/15/25/35/45), routed by the 125B's 8-bit router over the
  224 experts most used there in Breakthrough's expA traces (a stand-in for the C 224 pool), top-1, and the frozen 4-bit expert's
  SwiGLU output comes back through out_i (zero init, so step 0 = L1-20). Trained: norm_i / in_i / out_i (65.5M).
  Checks: gather_qmm's input gradient == a dequantized reference (max diff 9e-7); at zero init the trainer gives d13_train's
  numbers exactly (--check max logit diff 8.013e-02, argmax agreement 0.9738 for both; that check's 0.99 line is not met by
  d13_train itself at block 32 on multilingual text). Memory (--time): S 32 / A 8: E5 peak 11.0 GB, 8.5-10 s per 256 anchors;
  the L1-20 recipe (E1) 14.4 GB, 10.3-11.3 s; S 4 / A 64: 29.3 / 33.4 GB.
- E5a quick screen (21:16-21:37, shared lock; turns/e5a.sh): branch-only training on D-15a's split (16 train / 8 held-out
  multilingual answers), 2 epochs = 35 steps, lr 2e-5 and 2e-4; walk of the 8 held-out (block 16, TL_E5):

| arm | kept per check (L1-20 2.07) | sure-token depth-1 hit (L1-20 0.655) |
|---|---|---|
| E5a lr 2e-5 | 2.08 | 0.654 |
| E5a lr 2e-4 | 2.08 | 0.646 |

  Null at both rates, like D-15a's full-body tune on the same 16 answers (2.07 -> 2.06). Read with the D-16 caveat: 35 steps
  is little for a fresh 65M branch (training loss ema 7.185 -> 7.12 / 7.15), so this is not E5's verdict. E5 proper = the same
  branch at lr 1e-4 on tonight's DLOG set (20K answer tokens per language, ~14x the anchors), read on the full held-out D-02a
  multi (24) + IFEval (30) with Fable's lines (sticky/e15_read.py), after E1 on the same set (turns/night8_train.sh).

### D-23c v2b at C 224: stopped by the swap guard after 3 of 18 prompts (Mon 11-02 01:35-01:38; turn69 -> turn68 -> turn62)
- Gate passed at exactly 44 GB NO-SWAP; the v2b preset engaged (OOO calls after the warm-ups 1213); then swap grew +2.4 GB
  (8.8 -> 11.2 GB) and the guard stopped the run. 6 arm runs are saved (data/d23c_runs.jsonl, resumable): HumanEval/16 dfl
  47.0 / mtp15e 71.9 tok/s, prealgebra/1640 mtp15e 42.9 / dfl 49.1, ifeval/2853 dfl 24.1 / mtp15e 29.4. NOT a result: 3
  prompts, and each prompt's first arm ran on a cold expert cache (code dfl 1325 misses vs mtp15e 172). No verdict; C 224 with
  the MTP head needs more headroom than tonight's idle ~44-45 GB (or a re-pre-registered lower C).
- DLOG (Sun 23:29-00:42 at C 160, frozen 00:42-01:02 by Kamen's quiet window, ended by its ticket limit 01:04): 174 prompts,
  de 17,966 / ru 17,480 / ja 17,370 answer tokens, 1.85 GB, logged top-1 == next token on every answer row of every prompt.

### D-23c' at C 192: pre-registration (written before the run, Mon 11-02 01:5x; orchestrator 01:5x suggestion, my call)
- Why: tonight's idle memory (~44-45 GB NO-SWAP) does not hold D-23c at C 224 with the MTP head resident (the swap guard
  stopped it at +2.4 GB after 3 prompts). C 192 is ~4.5 GB lighter (peak ~38 GB; gate 42 GB = peak + 4).
- Same as D-23c v2 in everything else: sticky/d23c_mtptrunc.py v2b (the current default's preset 45+ooo, OOO + TIER_COMBINE
  via BigMoE's scores hook; NEWDEF=0 fallback in the same ticket on a crash), arms dfl vs mtp15e (K15 + the 0.3 cut + early
  exit), head b5, D-23's 18 prompts, interleaved arms, one discarded warm-up per arm. Both arms at the same C, so the paired
  comparison and the lines stay as registered: PASS = S1 (dfl on code + math, mtp15e on chat + agent) >= +10 % whole-set
  tok/s with every token equal; KILL < +4 %; BETWEEN otherwise. S2 and mtp15e-everywhere reported beside it.
- Only the absolute tok/s differ from the default (C 224). D-20b's switch gained more at higher C (+10 % at C 160, +27 % at
  C 224), so a C 192 verdict is a lower-cache reading of the same comparison; the C 224 confirmation stays open for a night
  with more free memory. BigMoE's M-17 per-layer pool sizes (offered) are NOT used: the preset stays the shipped default.
- Ticket: one 45-min quiet ticket after the E1 / E5 ticket's release (turns/after69.sh -> turns/turn70.sh; budget 2300 s /
  hard 2400 s of running time, gate retried up to 6 times). Resumable rows are keyed by C, so C 224's 6 runs stay separate.

### E1 COVERAGE RESULT: KILL (Mon 11-02 02:53; train 01:38-02:50 in turns/turn69's ticket; logs/night8_e1train.out, data/e15_read.txt)
- Fable's E1 (MISSION 4 sec 4): DFlash2 L1-20 tuned on the 125B's own de / ru / ja answers (tonight's DLOG: 174 MGSM-seeded
  prompts, 17,966 / 17,480 / 17,370 answer tokens = 88 % of the 20K spec, 52,642 anchors, ~11x D-15a), L1-20's recipe
  (forward KL to the top-32, block 32, pmask 0.5, lr 2e-5 cosine), 2 epochs = 416 steps (S 32 / A 8 micro-batches); held-out
  D-02a answers (multi 24 + IFEval 30; none in the training set), block 16 walk (sticky/e15_read.py, D-14's definitions).

| set | L1-20 | E1 |
|---|---|---|
| de / ru / ja sure-token depth-1 hit (pooled, n 362) | 0.707 | 0.729 (+0.022) |
| multi kept per check | 2.06 | 2.05 |
| multi depth-1 hit sure / mid / tie | 0.622 / 0.452 / 0.224 | 0.643 / 0.450 / 0.237 |
| IFEval kept per check | 2.45 | 2.41 |
| IFEval depth-1 hit sure / mid / tie | 0.899 / 0.565 / 0.292 | 0.907 / 0.553 / 0.278 |
| code / math kept (forgetting) | 9.57 / 7.00 | 9.40 / 6.74 (= D-13's usual drift) |

- Per language sure-hit / kept, L1-20 -> E1: ar 0.454 -> 0.488 / 1.59 -> 1.60, bg 0.365 -> 0.393 / 1.60 -> 1.59, de 0.734 ->
  0.771 / 2.39 -> 2.31, es 0.936 -> 0.928 / 2.85 -> 2.75, fr 0.824 -> 0.841 / 2.74 -> 2.82, ja 0.683 -> 0.689 / 2.08 -> 2.19,
  ru 0.697 -> 0.713 / 2.11 -> 2.10, zh 0.672 -> 0.686 / 2.00 -> 1.93.
- Verdict on Fable's line (PASS: de/ru/ja sure-hit >= 0.85 and multi kept +0.5; KILL: gain < +0.05): KILL (+0.022, kept
  -0.01). The training loss fell (ema 6.61 -> ~4.9), so the drafter fit the new text, but held out it moved the sure tokens
  by 2 points and the kept tokens not at all: at ~10x D-15a's data (D-15a: -0.01 kept) the coverage gap still is not buyable
  with the 125B's own answers in the language, in line with D-09's +0.10-0.18 kept per data doubling. Caveats: MGSM-seeded
  (math word problems in four styles) rather than open chat; 88 % of the 20K spec.

### E5 DFlash-X RESULT: KILL (Mon 11-02 04:00; train 02:52-03:56 in turns/turn69's ticket, walk to 03:59; logs/night8_e5train.out, data/e15_read.txt)
- E5 proper: the branch above (L1-20's body frozen in bf16, one borrowed 4-bit expert of the 125B per drafter layer through
  the 125B's own router at T_i over the 224-expert subset, top-1; norm_i / in_i / out_i trained, 65.5M), lr 1e-4 cosine, on
  E1's set (52,642 anchors of the 125B's own de / ru / ja answers), 2 epochs = 416 steps (S 32 / A 8), 62.2 min, peak 11.0 GB.
  Training loss ema 6.61 -> 5.80 (E1, full body, same set: -> 4.81). Same held-out walks and definitions as E1.

| set | L1-20 | E1 (full body) | E5 (borrowed experts) |
|---|---|---|---|
| multi kept per check | 2.06 | 2.05 | 2.06 |
| multi depth-1 hit sure / mid / tie | 0.622 / 0.452 / 0.224 | 0.643 / 0.450 / 0.237 | 0.631 / 0.467 / 0.259 |
| de / ru / ja sure-token hit (pooled, n 362) | 0.707 | 0.729 | 0.713 |
| IFEval kept per check | 2.45 | 2.41 | 2.44 |
| IFEval depth-1 hit sure / mid / tie | 0.899 / 0.565 / 0.292 | 0.907 / 0.553 / 0.278 | 0.902 / 0.555 / 0.282 |
| code / math kept (forgetting) | 9.57 / 7.00 | 9.40 / 6.74 | 9.53 / 6.89 |

- Per language sure-hit / kept, L1-20 -> E5: ar 0.454 -> 0.459 / 1.59 -> 1.60, bg 0.365 -> 0.402 / 1.60 -> 1.59, de 0.734 ->
  0.766 / 2.39 -> 2.30, es 0.936 -> 0.912 / 2.85 -> 2.87, fr 0.824 -> 0.843 / 2.74 -> 2.78, ja 0.683 -> 0.654 / 2.08 -> 2.09,
  ru 0.697 -> 0.702 / 2.11 -> 2.12, zh 0.672 -> 0.677 / 2.00 -> 1.99.
- Verdict on Fable's lines (PASS: multi sure-hit >= 0.75 OR IFEval mid-hit >= 0.62; KILL: < +0.1 on both the multi sure-hit
  and IFEval kept): KILL (multi sure-hit +0.009, IFEval mid-hit -0.010, IFEval kept -0.01). Against E1 (same data, no
  experts), the experts' own share: multi sure-hit -0.012, IFEval mid-hit +0.002, multi kept +0.01, i.e. nothing.
- Reading: the branch trained (loss -0.8) but transferred nothing held out; the frozen body kept code / math nearly intact
  (9.53 / 6.89), which is the only difference from E1. Routing a drafter's own hidden state through the 125B's router into
  frozen experts did not hand the drafter the 125B's knowledge at this size and data; with E1 also KILL, neither more of the
  125B's own answers nor its experts move the drafter's held-out multilingual / instruction acceptance. Caveats: a 224-expert
  subset picked from expA traces, top-1 only, MGSM-seeded data, one learning rate (E5a's 2e-5 / 2e-4 were null too).

## HPD-1: hard-position distillation on L1-20's own data -- pre-registration (written before any run, Mon 11-02 04:13; Fable ~/FableGod/FINDINGS.md MISSION 8 sec 3, the orchestrator's Monday lead, NewIdeas' Monday list)
- Why: Fable's best new drafter direction. The drafter's misses sit on positions it already flags (cf < 0.999: half of code,
  92 % of the breaks; D-23's live logs). TRIED holds no training set selected by the drafter's own confidence: D-09 more
  uniform data (+0.10 code kept per doubling), SD1 / R-05 depth weights (0.97x), D-13 loss forms (0.99x), L1-21 picker
  (+1.2 %), D-15a / E1 / E5 language tunes (null).
- Confidence = the live engine's cf (the selector's softmax over its 16 candidates at its pick; d23_switchdef.py's chain), of
  L1-20, with the true predecessor, at every on-path position of L1-20's own training data (~/Leap/data/l120: 99 train
  windows of 80 HumanEval+ / 29 GSM8K answers of the 125B, ~22.6K answer tokens), training block 32
  (sticky/hpd_train.py --conf-out).
- Arms, both from L1-20 with its recipe (forward KL to the top-32, block 32, pmask 0.5, lr 2e-5 cosine, S 32 / A 8 = 256
  anchors per step), 2 epochs (~178 steps): H = loss only where L1-20's cf < 0.999 (sticky/hpd_train.py --hpd-conf);
  U = every position (sticky/d13_train.py), i.e. 2 more epochs of the same recipe: the control for the extra training.
- Not Fable's full form (one doubling of FRESH data selected the same way): tonight's DLOG data is de / ru / ja word
  problems, and every HumanEval problem is already used (80 train / 24 held-out / 60 live benchmark prompts, disjoint).
  HPD-1 is the re-weighting of the existing set; a fresh-data round (HPD-2) needs new code prompts.
- Score (sticky/hpd_read.py, CPU): held-out code (24 HumanEval+), math (60 GSM8K), multi (24), IFEval (30), block 16,
  the official 1-step loop. L1-20's own walk (TL_WALK_CF=1) defines the positions: the depths it reached at each check
  (1 .. accepted + 1); uncertain = its cf < 0.999 (0.99 beside it). Each arm is walked at L1-20's anchors and scored with
  teacher-forced picks at exactly those positions (paired); 95 % intervals by resampling samples. Beside it: accuracy at
  the sure positions (must hold), kept per check at L1-20's anchors and on the arm's own walk.
- Line (Fable's, the orchestrator's): KILL if H's code accuracy on L1-20's uncertain positions (cf < 0.999) moves < +0.01
  against L1-20; else ALIVE. The selection's own share = H - U on the same positions; if H - U < +0.005, any gain belongs
  to the extra epochs, not to the selection.
- Prediction: H ~ U ~ L1-20 + 0.00-0.01 on code (L1-20 has fit this set; the uncertain positions left in it are the
  hardest; D-09 and E1 say held-out positions move slowly with data). 60 % KILL.
- Self-checks before any arm: the CF walk reproduces L1-20's kept per check (code 9.57, math 7.00, multi 2.06, IFEval 2.45),
  and L1-20 walked at its own anchors gives identical checks and picks (turns/hpd1.sh walk0).
- Resources: ~75 min, peak ~14.4 GB (training), walks ~5 GB, no timing. Slot: last in the night queue (in place of my DLOG
  top-ups: with E1 KILL, more de / ru / ja data has little use now) or under a shared lock by day (NO-SWAP >= peak + 8).

### D-23c' at C 192 RESULT: BETWEEN on 17 of 18 prompts (Mon 11-02 04:23-04:42, turns/after69 -> turn70; data/d23c_mtptrunc_c192_17p.txt, d23c_runs.jsonl)
- As pre-registered (v2b on the current default: OOO engaged, 92,472 OOO calls; C 192, head b5, cut t 0.3), 1142 s, peak
  38.2 GB, swap flat. 17 prompts clean, every token equal across arms; agent/edit01's dfl run was paused (a SIGSTOP freeze)
  and is excluded by the script's rule (the record is re-run on the next invocation).

| type (prompts) | dfl tok/s | mtp15e tok/s | change | kept per check dfl -> mtp15e | draft / check ms dfl -> mtp15e |
|---|---|---|---|---|---|
| code (4) | 53.3 | 53.0 | -0.7 % | 11.20 -> 11.92 | 11.9 / 189.0 -> 37.3 / 172.7 |
| math (4) | 47.5 | 43.4 | -8.6 % | 6.76 -> 6.64 | 11.6 / 118.8 -> 22.8 / 115.5 |
| chat (6) | 27.4 | 31.2 | +14.0 % (geo-mean 1.186, faster on 4) | 2.71 -> 3.13 | 11.5 / 72.4 -> 9.4 / 75.0 |
| agent (3) | 22.5 | 21.1 | -6.2 % | 4.17 -> 6.63 | 13.3 / 118.7 -> 23.9 / 148.7 |

- Whole set (17 prompts, 3,542 tokens): dfl 32.5 tok/s; S1 (dfl on code + math, mtp15e on chat + agent) 35.5 = +9.4 %;
  S2 (dfl on code only) +6.9 %; mtp15e everywhere +6.1 %. Verdict on S1 (PASS >= +10 %, KILL < +4 %): BETWEEN.
- Reading: the MTP chain with the default's cut is a real exact gain on chat (+14 %, kept 2.71 -> 3.13 at fewer rows); on
  agent it keeps far more (4.2 -> 6.6) but its longer chains cost more drafting and checking than they save here (D-23b's
  replay had agent +6-7 %); math stays a DFlash2 domain. The pre-registered set is completed by one 15-min ticket that
  re-runs the paused record (queued 04:4x); the final read-out on 18 prompts follows here.

## D-23d CHATSWITCH: MTP on chat (and agent) on fresh prompts -- pre-registration (written before any run, Mon 11-02 06:0x)
- Why: D-23c' (C 192, 17 clean prompts) put the MTP chain (K15, head b5, the default's cut t 0.3) at chat +14.0 % (mean
  tok/s; geo-mean 1.186, faster on 4 of 6) and +25.8 % time-weighted (chat dfl 53.1 s -> 42.2 s), but on only 6 chat prompts
  with a wide spread (ifeval/374 -12 %, multi/bg1 +64 %), and a chat-only switch (MTP on chat, DFlash2 elsewhere) is only a
  post-hoc +11.0 % whole set. Agent was -6.2 % on 3 prompts (the 4th, edit01, is being re-run). Head b5 was trained on L1-20's
  code + math windows only, so no chat or agent prompt is in its training; DFlash2 L1-20 was trained on code + math too.
- Prompts (fixed by rule before any run; none in D-23 / D-23c'): dec_multi, the first unused key per language in file order:
  bg0 de3 fr7 es10 ru12 zh15 ja18 ar21 (8); dec_ifeval, the first 8 unused keys in file order: 1220 3280 1561 2239 2590 2870
  2439 1219; dec_agent, the first unused key per kind: bash17 edit00 text23 write11. Interleaved by type, 20 prompts.
- Run: the D-23c' harness unchanged (sticky/d23c_mtptrunc.py, D23_NEWDEF=1 = the current 45 GB default with OOO, C 192, CAP
  256, arms dfl and mtp15e alternating first, one warm-up per arm), one quiet ticket (~25 min; budget 2000 s, resumable),
  gate NO-SWAP >= 42 GB. turns/turn72.sh; output logs/d23d.out; read-out sticky/d23d_read.py -> data/d23d_read.txt.
- PRIMARY (chat-only switch), on the 16 fresh chat prompts (set ifeval / multi), every token equal across arms:
  time-weighted chat gain G = sum(time dfl) / sum(time mtp15e) - 1. PASS: G >= +10 % and per-prompt geo-mean ratio >= 1.03;
  KILL: G < +3 %; BETWEEN otherwise. (D-23c' had G +25.8 %, geo 1.186: PASS leaves room for regression to the mean.)
- SECONDARY (agent joins the switch?): pooled agent G on the 4 fresh + D-23c''s 4 (after the edit01 re-run): agent goes on
  MTP only if pooled G >= +8 % and fresh G >= 0; else agent stays on DFlash2. Information: NewIdeas' type rule's label for
  each fresh prompt (the switch routes by the rule: a chat prompt the rule calls code / math stays on dfl, scored that way
  in a second line), and the projected whole-set gain of the chosen switch on D-23c''s mix with chat pooled (22 prompts).
- Predictions: chat PASS 55 %, BETWEEN 30 %, KILL 15 %; agent pooled G >= +8 % 35 %.
- On PASS: the chat switch (type rule -> MTP K15 + cut on chat) goes to the integration owners (Leap / BigMoE) as an exact
  +G on chat, with the whole-set projection; BIG to the orchestrator only if the projection is >= +10 % whole set.

### D-23c' at C 192 FINAL: PASS on all 18 prompts, by a thin margin (Mon 11-02 06:39-06:41 re-run of the paused record, turn70 in a 15-min ticket; data/d23c_mtptrunc.txt C 192 block, d23c_runs.jsonl; written 06:42)
- The paused record re-run clean: agent/edit01 dfl 31.8 tok/s (steady 39.1; prefill 89.8 s, the paused one had 203.8 s) vs
  mtp15e 41.8 (+31 %). 18 prompts clean, every token equal across arms (3,793 tokens).
- Whole set: dfl (the default) 32.5 tok/s; **S1 (dfl on code + math, MTP K15 + cut on chat + agent) 35.9 = +10.6 % -> PASS**
  (pre-registered PASS >= +10 %, KILL < +4 %); S2 (dfl on code only) +8.3 %; mtp15e everywhere +7.5 %. A chat-only switch
  (post-hoc) +10.2 %.
- Per type: code -0.7 %, math -8.6 %, chat +14.0 % mean (geo 1.186, faster on 4 of 6; time-weighted +25.6 %), agent +5.9 %
  mean (geo 0.997, faster on 2 of 4: edit01 +31 %, edit02 +9 %, bash18 -18 %, bash19 -16 %).
- How fragile: the margin is 0.6 points. Leave-one-out: without multi/bg1 (dfl 16.6 -> MTP 27.3 tok/s; DFlash2 keeps 1.26
  tokens per check on Bulgarian) S1 is +5.3 %; without agent/edit01 +9.4 %. The chat gain is broad (4 of 6, geo 1.186); the
  agent gain is two long edit prompts against two bash prompts. So: a PASS by the letter, exact, on the current default, and
  the right next step is D-23d (pre-registered above, before this read-out: 20 fresh prompts, 16 chat + 4 agent) before
  anything ships. Integration needs no new parts: NewIdeas' type rule (in the harness) routes; the MTP arm is the native MTP
  chain with head ~/TrainLab/ckpt/d22_b5.safetensors, K15, the default's running-confidence cut 0.3 with early exit.

### D-23c' CORRECTION: the PASS does not stand: an order effect and in-sample IFEval (Mon 11-02 06:52, before any D-23d run)
- Order effect: the harness runs the two arms back to back on each prompt (the first arm alternating by prompt index).
  Both arms decode the same tokens, so the second arm verifies on the experts the first just loaded: it had fewer misses
  on 17 of 18 prompts (HumanEval/16 dfl first 1,482 vs MTP 517; ifeval/374 MTP first 548 vs dfl 236). From the balanced
  types, ln ratio = ln R +- ln w: the second arm is faster by w = 1.19 (code, 2 + 2), 1.12 (math, 2 + 2); the same model
  gives 1.21 for chat and 1.20 for agent. The order was not balanced where S1 uses MTP: chat had dfl first on 4 of 6
  (MTP warm), agent on 1 of 4.
- Order-corrected per-type ratios (geo, MTP / dfl): code 0.998, math 0.909, chat 1.115 (reported 1.186), agent 1.093
  (reported 0.997). Both arms put at their first-run (cold) times with each type's w: chat time-weighted +14.0 % (reported
  +25.6 %), agent +11.3 %, S1 about +8.7 % whole set (reported +10.6 %) -> BETWEEN, not PASS; chat-only switch about +5.7 %.
  These are model-based estimates (a constant w per type, fitted on the same 18 prompts), not measurements.
- In-sample: head b5's training (D-22 b4 / b5 recipe) includes 97 IFEval windows of the 125B's exact answers, among them
  all three IFEval prompts here (2853, 374, 3407). After the order correction their joint gain is small (dfl 19.4 s -> MTP
  18.8 s); the chat gain is in the multilingual prompts (bg1, es9, fr6; multi is in no training set). My D-23d text above
  ("b5 was trained on code + math only") was wrong: 5 of D-23d's 8 IFEval keys were in b5's training too.
- Verdict: D-23c' = BETWEEN (order-corrected); the BIG of 06:42 is corrected (sent 06:52). D-23d decides, with the design
  fixed below. Measurement lesson for every A/B at this memory (SSD-bound): arms back to back on the same prompt favour the
  later arm by ~12-20 %; separate them or balance the order and correct.

### D-23d amendment 1 (Mon 11-02 06:52, before any D-23d run): order and prompts
- Order: D23_ORDER=sep (new in sticky/d23c_mtptrunc.py; the default order unchanged): each arm gets its own pass over all
  prompts, so a prompt's two runs are 22 prompts apart (4 of them agent prompts with ~9K-token contexts); the arm of the
  first pass alternates within each set (multi 8 / 8, one of each per language; IFEval 2 / 1; agent 2 / 2). Read-out
  check: w = sqrt(geo ratio where dfl ran first / where MTP ran first) should be near 1.
- Prompts: only prompts outside every training set in play (L1-20: HumanEval / GSM8K; b5: those + d22c's 97 IFEval and 50
  MGSM windows). dec_ifeval has 3 such prompts (1220, 3280, 1561); dec_multi gives 16 (two per language: bg0 bg2, de3 de4,
  fr7 fr8, es10 es11, ru12 ru13, zh15 zh16, ja18 ja19, ar21 ar22); agent bash17 edit00 text23 write11 as before. 23 prompts.
- Lines (same thresholds): PRIMARY chat = the 19 chat prompts (16 multi + 3 IFEval), time-weighted G: PASS >= +10 % and
  geo >= 1.03; KILL < +3 %; else BETWEEN. multi and IFEval reported apart (IFEval n = 3: information only; English chat
  is not measured well by this set). SECONDARY agent: on MTP only if its G >= +8 % and geo >= 1.00 (fresh 4; no pooling
  with D-23c', which carries the order effect). Projection on D-23c''s mix as information.
- Predictions (revised for the correction): chat PASS 40 %, BETWEEN 35 %, KILL 25 %; agent joins 30 %.

### D-23d CHATSWITCH RESULT: PASS -- chat goes on the MTP head; agent stays on DFlash2 (Mon 11-02 08:15-08:32, after_d23c2 -> turn72 in a 40-min quiet ticket; logs/d23d.out, data/d23d_read.txt, d23c_runs.jsonl; written 08:35)
- Run: 23 of 23 prompts clean in both arms, every token equal across arms (exact); D23_ORDER=sep (each arm its own pass,
  the first arm alternating within each set); start 08:15:34 at 45 GB NO-SWAP, done 08:32:31 (swap 10.8 -> 11.4 GB).
- PRIMARY chat (19 prompts): time-weighted G +24.5 %, geo-mean ratio 1.223, MTP faster on 19 of 19 -> PASS (line: G >= +10 %
  and geo >= 1.03). Chat tok/s (tokens / seconds) 23.1 -> 28.8. The type rule labels all 19 chat (routed G +24.5 %).
  - multi (16; bg de fr es ru zh ja ar, two each): G +27.4 %, geo 1.265, 16 of 16 faster; kept per check 2.09 -> 2.65;
    tok/s 22.9 -> 29.1. Per prompt: bg +49 / +50, ar +40 / +37, zh +25 / +46, ja +37 / +14, ru +31 / +26, de +15 / +24,
    es +16 / +12, fr +5 / +10 %: the further the language is from English, the more DFlash2 (code + math training) loses.
  - IFEval (3, English; information only, n = 3): G +1.4 %, geo 1.022 (+0.8, +1.4, +4.5 %); kept 4.55 -> 5.36.
- Order check: chat w 0.992 (geo 1.213 where DFlash2 ran first, 1.234 where MTP did), all prompts 0.976 (agent 0.894 on
  2 + 2). The separate passes removed D-23c''s back-to-back bias (w 1.12-1.20).
- SECONDARY agent (4): G -0.5 %, geo 0.981 -> agent stays on DFlash2 (line G >= +8 % and geo >= 1.00). MTP keeps far more per
  check on agent (4.80 -> 8.20, +71 %), but its longer chains verify more rows, which touch more cold experts (misses
  edit00 986 -> 1404, text23 662 -> 1165, bash17 299 -> 372), so tok/s is flat (21.5 -> 21.4) at C 192.
- Whole-set projection on D-23c''s mix (chat = 53.1 of 116.9 DFlash2 seconds, 45 %; agent, code, math on DFlash2): chat
  pooled x1.245 -> +9.8 % (the pre-registered information line; under the +10 % BIG line); multi and IFEval at their own
  rates -> +7.0 %; all chat English -> +0.6 %. The whole-set gain is the non-English share of the chat traffic.
- Predictions: chat PASS 40 % (came true); agent joins 30 % (did not).
- Verdict: D-23d = PASS. The switch to ship: the type rule -> chat on the MTP head (b5, K15, cut t 0.3, early exit);
  code, math, agent on DFlash2 L1-20 as today. Exact. No BIG (projection +9.8 %, my line +10 %). Handoff:
  INTEGRATION_CHATSWITCH.md to the owners of the live default (Leap / BigMoE). Open: C 224 (the default's C; D-20b's switch
  gained more at higher C) on a night with ~46 GB free; English chat on more than 3 prompts; agent at C 224, where MTP's
  extra kept tokens may pay for its extra rows.

### D-23e: the chat switch at C 224, the default's C -- pre-registration (written before any run, Mon 11-02 08:42)
- Why: D-23d passed at C 192 (chat +24.5 %). The default ships at C 224, where D-20b's switch gained more, and where MTP's
  extra agent rows (4.80 -> 8.20 kept per check at C 192, but +24-76 % misses) may pay. One confirmation at the shipping C
  before the owners wire it, with the agent line read again.
- Run: D-23d unchanged except C 224: the same 23 prompts in the same order, D23_ORDER=sep, head b5, cut t 0.3, CAP 256, one
  warm-up per arm; gate NO-SWAP >= 48 GB (v2b at C 224 swapped at 44 GB), swap guard +2 GB; one quiet 40-min ticket (budget
  1900 / 2000 s). turns/turn74.sh -> turn74_d23e.sh; logs/d23e.out; read-out sticky/d23d_read.py with D23_C=224 (now
  also prints the projection with multi and IFEval at their own rates) -> data/d23d_read.txt.
- Lines (D-23d's): chat (19) time-weighted G: CONFIRMED if G >= +10 % and geo >= 1.03; the switch FAILS at the default's C
  if G < +3 %; BETWEEN otherwise. Agent joins the switch only if its G >= +8 % and geo >= 1.00. Any token differing across
  arms = FAIL. The prompts are D-23d's: not new to me, but still outside every training set (nothing trained since).
- Predictions: chat CONFIRMED 80 %, BETWEEN 15 %, FAILS 5 %; agent joins 35 %.
- When: after HPD-1's shared lock ends (turns/after_hpd.sh queues the quiet ticket; by day only if NO-SWAP reaches 48 GB,
  else tonight).

### D-23d note: agent's MTP loss sits in the first check after the long prefill (CPU, D-23d's per-check records; written 08:48)
- Agent prompts have ~9K-token contexts (prefill 65-78 s). Over the 4 agent prompts, each prompt's first check cost
  DFlash2 4.7 s in all and MTP 9.9 s; agent decode in all: dfl 26.5 s, MTP 26.7 s. Without each prompt's first check,
  agent is 26.1 -> 34.0 tok/s (+30 %) for MTP; after the first check, the two arms' checks are similar (2nd / 3rd checks
  105-440 ms in both).
- MTP's first check after a cold prefill: 1.9-2.8 s in 7 of 8 runs (D-23c' + D-23d; 685 ms only for edit01, which ran
  second on warm experts), whatever its rows (write11: 3 rows, 2.78 s). DFlash2's first check: 0.3-2.5 s (cheap in 4 of 9;
  its first draft takes 160-190 ms against MTP's ~40 ms). In D-23d's second pass, every agent first check was slow
  (1.5-2.8 s, both arms).
- Not explained yet: a cold decode working set after the prefill (the prefill path's staging is dropped), memory pressure
  after a 9K prefill (swap rose 0.6 GB during D-23d), or something MTP-specific. sticky/d23c_mtptrunc.py now saves each
  check's misses and io ms in "pc" (logging only), so D-23e shows whether the first check is a cold-expert load.
- For the live engine this stall is part of an agent turn's latency after prefill (~1.5-2.8 s), whichever drafter: a lead
  for BigMoE / Leap once D-23e's per-check misses say what it is. The agent verdict (stays on DFlash2) stands as measured.

### HPD-1 RESULT: KILL (Mon 11-02 09:24; H trained 08:47-09:19 under a shared lock, walkH + readH 09:19-09:23; logs/hpd1_readH.txt)
- Self-checks passed (walk0: L1-20's CF walk 9.57 / 7.00 / 2.06 / 2.45; fixed-anchor walk == own walk on every set).
- Code, L1-20's uncertain positions (cf < 0.999, 48.4 % of reached positions, n 2990): L1-20 0.8428 -> H 0.8351
  (-0.0077, 95 % -0.0204 .. +0.0042); at cf < 0.99: -0.0077. Sure positions hold (1.0000). Line: KILL if < +0.01 -> KILL.
- H is slightly worse at the uncertain positions everywhere: math -0.0064 (95 % -0.0116 .. -0.0007), IFEval -0.0090
  (-0.0156 .. -0.0028), multi -0.0058 (n.s.). Kept per check at L1-20's anchors: code 9.52 (-0.05), math 6.92 (-0.07),
  multi 2.03, IFEval 2.41; own walk code 9.66 (+0.09), math 6.94.
- Prediction was 60 % KILL. Two more epochs on L1-20's own 22.6K tokens with the loss only on its uncertain positions
  do not move them: L1-20 has fit this set, and its hard positions are not learnable from it.
- U (the same 2 epochs on every position: the control) is training (09:23, shared); its read gives H - U, i.e. whether
  the small loss is the selection's or the extra epochs'. That decides HPD-2's form (fresh code data selected by cf): if U
  loses as much as H, the selection is not what hurts and HPD-2 stays possible on fresh data; if U holds, masking itself
  hurts.

### D-23e amendment 1 (Mon 11-02 10:01, before any D-23e run): gate 47 GB
- By day NO-SWAP sits at 47 GB (the Claude app and sessions hold ~21 GB). C 224's peak is ~42.4 GB (D-23c' peaked at
  38.2 GB at C 192; the pools grow 25.3 -> 29.5 GB at C 224), so a 47 GB gate keeps the quiet rule's +4 GB; the +2 GB swap
  guard stays. Lines, prompts and order unchanged. turns/after_hpd2.sh (replaces after_hpd.sh) queues D-23e at 47 GB.

### HPD-1 control U (09:23-09:57, shared; logs/hpd1_read.txt): the selection's own share H - U = +0.0023
- U (the same 2 epochs on every position) lost about as much as H at L1-20's uncertain code positions, so the small loss
  is the extra epochs on a set L1-20 has already fit, not the masking. HPD-1 stays KILL; HPD-2 (fresh code data selected
  by cf) is not ruled out by it, but it needs new code prompts (MBPP: Kamen's download OK).

### D-23x: the first-check stall, traced (NewIdeas 09:5x; a diagnostic, not a timing result; turns/turn75.sh after D-23e)
- The D-23d harness at C 192 with BIGMOE_TRACE=1 + D23_TRACE=1 (new: sticky/d23c_mtptrunc.py records each prompt's prefill
  and first 6 checks' expert reads, by pool: base MSB-3 vs LSB-1 sidecar, by tag: demand / prefetch / joint, and by layer;
  own files data/d23x_runs.jsonl, d23x_mtptrunc.txt). 4 agent prompts + 2 short chat prompts (bg0, ifeval/1220), both
  arms, separate passes. Read-out sticky/d23x_read.py -> data/d23x_read.txt.
- Question: is the base (3-bit) pool cold after the 4-bit prefill, so the first check reads it from the SSD? If so, a pool
  warm-up at the end of the prefill (TRIED: PARKED) is an exact candidate fix for agent turns in the live engine.

## E2 / D2 DATASLOPE-MBPP: the data slope's next points on fresh code (ROUTE_350 condition 8, kept >= 13 on code) -- pre-registration (written before any capture, Mon 11-02 10:35; Kamen's day order via the orchestrator ~10:3x; MBPP download approved, ~/TrainLab/data/mbpp, SHA256 checked)
- Why: D-09 measured +0.10 (code) / +0.18 (math) kept per check per doubling over 1/4 -> 1/2 -> 1 of L1-20's data. Whether
  the slope holds past L1-20's data decides if more data (the harvest, Fable's E6) is a road to code >= 13. HumanEval is used
  up; MBPP (974 problems, none in any set we score) gives fresh code prompts.
- Data: the 125B's own answers to MBPP prompts (task text + its 3 tests; sticky/dlog_capture.py DLOG_SRC=mbpp, a fixed
  random order, answers capped at 512 tokens, thinking off), captured with the default's harness (D-23's preset, exact), rendered
  as L1-20's data, until 52K code answer tokens. L1-20's train data: code 16,171 + math 6,432 answer tokens (72 + 27 windows;
  the same 10 val windows everywhere). Arms: x2 = L1-20's windows + the first MBPP answers adding >= 16,171 code tokens;
  x4 = + >= 48,513 (two doublings of the code part; math unchanged).
- Training: D-09's protocol exactly (sticky/d09_train.py = Leap's dfl2c_train.py: incoai init, lr 2e-5, S 4 / A 64, block 32
  in training, pmask 0.5, 2 epochs, so the steps scale with the data; --cache-gb 0.5, --minutes 100 / 180 as caps only).
- Score: sticky/d02b_walk.py (reproduces L1-20's 9.57 / 7.00), block 16, the official 1-step loop, kept per check on the 24
  held-out HumanEval+ samples (code) and 60 GSM8K (math), as D-09.
- Line (the orchestrator's / Fable's): code slope per doubling of code data, fitted on L1-20 (9.57), x2, x4 (log2 of the code
  share 0, 1, 2): KILL < +0.05; PASS >= +0.10; BETWEEN otherwise. If only x2 finishes today, the one-doubling step decides
  with the same lines (noisier: one seed per point). Beside it: the 5-point fit with D-09's 1/4 and 1/2 points; math kept
  (its data is unchanged); what code >= 13 would need at the measured slope.
- Prediction: the slope flattens past L1-20's data and MBPP is easier than HumanEval: x2 ~9.62, x4 ~9.66, slope ~+0.05:
  KILL 45 %, BETWEEN 40 %, PASS 15 %.
- Resources: all under shared locks (no timing). Capture C by NO-SWAP (C 64 >= 29 GB, 96 >= 33, 128 >= 37, 160 >= 41;
  the shared rule's +8 GB; by day NO-SWAP ~29-30 GB with Kamen's Chrome), ~1.8 GB of data; training peak ~15 GB; disk:
  hpdH / hpdU / e1_dlog checkpoints (KILLed today, results in REPORT) deleted for room (floor 114 GB).
- E2 amendment 1 (10:38, before any capture): by day NO-SWAP is 28 GB, so the capture also runs at C 48 when NO-SWAP >= 27
  (peak ~19 GB + the shared rule's 8); it stops at 17K code tokens until x2 is walked, then goes on to 52K (x2 trains
  sooner; the data order and the arms are unchanged). C only changes speed, not the 125B's tokens (exact).
- E2 amendment 2 (10:41, before any capture): the engine aborts when a layer call needs more experts than its base pool
  (ext/bigmoe_ext.cpp:582), and a 16-row check routes up to 160 (top-10 each). So the small-C capture caps the rows per
  check (DLOG_ROWS: C 48 4, 64 6, 96 9, 128 13, 160 16; BIGMOE_MAXK = 10 x rows), scales the LSB sidecar pool with C
  (BIGMOE_TIER_CL = 112 C / 192), and lowers the rows further if the loaded base pool is smaller. Rows and pool sizes change
  only speed: the 125B's tokens and logged rows are the same (exact verify).

> (a quote from the project owner, kept private)
- Why: the realizable gate (top-2 on rows where DFlash2's cf >= 0.999) covers 48.9 % of code rows at 0 top-2 flips (Fable:
  47.7 % on all rows); the cheap pass's own certificate (p1 under top-2 >= 0.99) covers 73.1 % at 0 flips but is known only
  after the pass. A small head on what the drafter knows before the check, trained to predict the certificate (not the
  drafter's correctness), could recover part of the gap: Fable's prize up to 2.2x on expert bytes at 45 GB. TRIED has no
  learned gate (UGB used the raw cf; CALSTOP was a stop rule).
- Rows: the decision map's 10,316 (~/FableGod/floor_rows.json; 36 windows, 12 each code / IFEval / multi), those with a
  drafter cf (code 3,283). Label: cert2 = exp(top-1 lp under topk2) >= 0.99 (decision_map/out/topk2.json). Flips: floor_rows'
  flip['topk2'] (every row perturbed, so conservative; code 138 = 4.2 %).
- Features (realizable before the check), per row = draft j + 1 of the HPD-0 walk's check at anchor A (the map's gate
  source, dm_gate.py): DFlash2's hidden at draft position j (2560), the pick's selector p1 / p2 / entropy, the unary top-16
  p1 / p2 / entropy, cf, log(1 - cf), the running cf product, depth. New TL_WALK_KH=1 in sticky/d02b_walk.py, walked at the
  HPD-0 anchors (data/d02b_walk_kh_<dom>.npz; its cf must equal HPD-0's within 1e-5 and floor_rows' cf).
- Heads (sticky/kh_train.py, MLX on the CPU): LOGIT-S (logistic on the 10 scalars: a re-calibrated cf), LOGIT-H (logistic
  on hidden + scalars), MLP-H (PRIMARY: 2 layers, 256 hidden, ReLU, Adam 1e-3, 30 epochs, wd 1e-4, standardized inputs).
- CV: 6 folds of 2 code windows; each fold trains on every other row of all three domains and scores its 2 held-out
  code windows; pooled out-of-fold scores.
- LINE (MLP-H, code): out-of-fold coverage at <= 0.1 % top-2 flips among the gated rows: KILL < 60 %; PASS >= 60 % and the
  deployment check holds (each fold's threshold = the lowest score with 0 flips on its training code rows, applied to its
  held-out rows: pooled flips <= 0.1 % of the gated rows); BETWEEN otherwise. Beside it: coverage at 0 flips, LOGIT-S /
  LOGIT-H, and the baselines computed the same way (cf alone ~48.9 %; the certificate itself, the ceiling, 73.1 %).
- Prediction: the drafter's hidden carries part of the 125B's margin: MLP-H ~55-62 %. KILL 50 %, BETWEEN 15 %, PASS 35 %.
- Resources: the walk is DFlash2 on stored taps (~5 GB, a few minutes, shared lock); the heads are CPU seconds.

### K-HEAD RESULT: KILL -- a learned head on the drafter's features does not beat the drafter's own cf (Mon 11-02 12:07; walk 12:01-12:04 shared, heads on the CPU; data/kh_read.txt, d02b_walk_kh_<set>.npz)
- The feature walk reproduced the map's gate: 10,232 rows found (0 missing), every cf equal to HPD-0's and floor_rows' (1e-5).
- Code, out-of-fold (6 folds of 2 windows), coverage at <= 0.1 % top-2 flips / at 0 flips; deployment = each fold's 0-flip
  training threshold on its held-out rows:
  | head | <= 0.1 % flips | 0 flips | deployment: covered, flips |
  |---|---|---|---|
  | LOGIT-S (10 scalars) | 53.1 % | 44.7 % | 47.5 %, 1 / 1,561 (0.06 %) |
  | LOGIT-H (hidden + scalars) | 51.9 % | 50.8 % | 75.3 %, 14 / 2,472 (0.57 %) |
  | MLP-H (primary) | 56.0 % | 53.9 % | 74.4 %, 17 / 2,441 (0.70 %) |
  | cf alone (same sweep) | 63.5 % | 56.7 % | - |
  | the certificate itself (ceiling) | 73.1 % | 73.1 % | - |
- LINE (MLP-H, <= 0.1 % flips, KILL < 60 %): 56.0 % -> KILL. The heads also over-certify: thresholds at 0 training flips
  give 0.57-0.70 % flips on held-out windows.
- Post-hoc (not the verdict; KH_CODEONLY=1, trained on code rows only): MLP-H 53.8 %, LOGIT-H 37.7 % (2,570 inputs on
  ~2.7K rows overfit), LOGIT-S 68.2 % at <= 0.1 % but 46.7 % at 0 flips and 50.1 % covered with 0.12 % flips in deployment:
  above cf only at the sweep's top, where 12 windows cannot separate them.
- Reading: the drafter's hidden state does not carry the 125B's top-2 margin better than its own cf does; Fable's 3.5 (1)
  "re-calibrated drafter confidence" does not open the 2.2x gap on this data. Also: the cf gate's own zero-flip point on
  these rows is lower than 0.999 (56.7 % covered at 0 flips; 63.5 % at <= 0.1 %, in line with Breakthrough's gated 0.99:
  2 / 2,211), so the realizable gate is ~57-64 %, not 47.7 %. Left: the router's top-2 mass per layer (logged Tuesday
  night) and the ~55K-row certificate log for a v2 (a scalar re-calibration is the one variant worth a retest there).
- Prediction was KILL 50 %.

### D-23e DROPPED (Mon 11-02 13:11, before any run; the orchestrator's call)
- BigMoE's night ticket (~/BigMoE/scripts/night_day9.sh group 2) measures the wired chat switch at C 256, the default's
  coming C: chA_256 (DFlash2 everywhere) vs chB_256 (R57_CHATSW=1: d22 b5, K15, cut 0.3 with early exit; checked in
  r57_screen.py) on D-23d's 19 chat prompts, ABBA order, with a reference-token check. That supersedes D-23e's C 224
  confirmation (data/D23E_DROP). D-23x (NewIdeas' first-check trace) stays, after BigMoE's ticket, done by 07:00.

## DUALDRAFT: two block drafters, the higher-cf pick per depth (Fable DRAFTER_98.md 3.2) -- pre-registration (written before any run, Mon 11-02 13:44; the orchestrator's queue for Tue by day, after E2's verdict)
- Why: two drafters are wrong at different places (Fable 1.5: either-right 94.0 % vs 90.9 % alone for near-twins; the
  cf pick keeps 92.7 %). Proposing, at each depth, the draft of whichever drafter is more confident could lift kept per
  check at +1-2 ms per block (one batched block pass) and nothing on the check. Exact (the check decides). TRIED: not there.
- Drafters: L1-20 and E2's x2 checkpoint (data/e2_x2.safetensors: incoai init, L1-20's data + one doubling of fresh MBPP
  code; a different model, Fable's "diverse" case). HPD-1's H / U (Fable's near-twin example) were deleted for disk this
  morning; if x2 is not trained by Tuesday, a seed-2 retrain of L1-20's recipe (~35 min shared) is the second drafter.
- Mechanism in the walk (new TL_WALK_ENS=<second drafter> in sticky/d02b_walk.py): per check both drafters run their block
  pass on the same context (each its own cache); a JOINT chain picks per depth: each drafter's selector scores its 16
  candidates given the chosen predecessor, the candidate with the higher cf (selector softmax at its pick) is proposed and
  becomes the predecessor for both at the next depth (implementable as is: no true tokens used).
- Score: D-02b's official 1-step loop, block 16, kept per check on the 24 held-out HumanEval+ code and 60 GSM8K samples,
  with L1-20 alone and x2 alone walked by the same script as references. Diagnostics: the share of depths taken from each
  drafter; teacher-forced either-right and cf-pick per-position accuracy.
- LINE (Fable's "kill if kept(16) < 11.3, i.e. < +8 %", restated for this walk, where L1-20 keeps 9.57): code kept per check
  of the ensemble >= 1.08 x L1-20's in the same walk (~10.34) -> PASS; < +4 % -> KILL; BETWEEN otherwise. Math reported
  beside it.
- Prediction: x2 is a different model but trained on overlapping data, so its errors overlap L1-20's more than random:
  ensemble +3-6 % on code. KILL 40 %, BETWEEN 40 %, PASS 20 %.
- DUALDRAFT tooling (13:45): sticky/d02b_walk_ens.py (a copy of d02b_walk.py, which today's E2 walks use unchanged) with
  TL_WALK_ENS. Self-check before the run: TL_WALK_ENS = L1-20 itself must give L1-20's 9.57 / 7.00 exactly and 0 % depths
  from the second drafter (equal cf: the first drafter's pick, i.e. the official chain).

### E2 / D2 x2 RESULT: PASS on the one-doubling step (condition 8's first verdict; Mon 11-02 15:18; capture 13:49-14:11 C 96 / 9 rows, train 14:11-15:09, walk 15:13; data/e2_read.txt, d02b_walk_e2x2.txt)
- Data: L1-20's 109 windows + the first 127 MBPP answers of the 125B = +16,212 code answer tokens (x2 of L1-20's 16,171);
  every captured row on-path (logged top-1 == the next token), the 125B's own tokens. Training: D-09's protocol, 359 steps
  in 55.6 min (2 epochs), from incoai.
- Walk (block 16, official loop, held-out): code 9.57 -> 9.79 kept per check (+0.22; per-token rate 0.9239 -> 0.9271;
  P(first 15 right) 0.276 -> 0.285); math 7.00 -> 6.99 (its data unchanged).
- LINE (one step L1-20 -> x2, KILL < +0.05, PASS >= +0.10): +0.220 per doubling -> PASS. All 4 points (D-09's 1/4, 1/2,
  L1-20, x2): code +0.139 per doubling. The slope does not flatten past L1-20's data on fresh on-policy code (D-09 was +0.10).
- What it means for condition 8 (kept >= 13): at +0.22 per doubling, 13 needs ~15 more doublings (x2.5e4 the data):
  data alone is a real but slow lever for a drafter of this size; it supports the harvest as one input, not the route to 13.
  One seed per point: x4 (two doublings) checks the slope. Prediction was KILL 45 % / BETWEEN 40 % / PASS 15 %.
- x4: 18,331 of the 48,513 code tokens are captured; the capture slot that started at 15:13 was stopped at 15:17 to give
  the GPU to BigMoE's A6 (the orchestrator's order); x4 resumes after A6 (chain v6).

## MATCH: the session match override (Fable DRAFTER_WALL.md 2.1) + the match-driven block length (2.5) -- pre-registration (written before any run, Mon 11-02 16:46; the orchestrator's queue for tonight after x4's walk; shared lock)
- Why: on code the 125B restates its own docstrings and signatures; Fable's probe on my HPD-0 walks: 58.2 % of HumanEval+
  positions sit inside a >= 8-token match of the window, and the match's next token is right 99.2 % there (the drafter
  96.1 %). Exact and training-free: the check decides. TRIED: prompt lookup / SUFFIX / LK1 / M-05 were standalone drafters
  (few, imprecise matches on agent / math); an override on top of the block drafter, gated by match length, is not there.
- Mechanism (Fable's, on the CPU after the drafter's block): a table of every k-gram (k 8-16) of the committed window
  (prompt + accepted tokens) -> the token after its most recent occurrence; walking the drafted block, at each position the
  longest k >= 8 suffix (the committed window + this block's tokens so far, overrides included) found in the table replaces
  the draft with that next token; else the drafter's token. 2.5: at 32 rows, the check takes 32 rows when >= 24 of the 32
  drafted positions carry such a match, else the first 16 rows.
- Walk: sticky/d02b_walk_match.py (a copy of d02b_walk.py; TL_WALK_MATCH, TL_WALK_BS, TL_WALK_LEN25), DFlash2 L1-20, the
  official 1-step loop with anchors moving with the overridden drafts. Sets: the 24 HumanEval+ windows (dfl2d_code), 100
  of E2's MBPP answers (data/dlog/d1102mbpp, capture order; held-out for L1-20), the 12 agent windows (data/d02a/agent_*).
  Arms per set: base16, match16, base32, match32, len25. Math is off (the type rule) and not walked.
- LINES (Fable's): HumanEval+ kept(16) match/base >= 1.06 AND kept(32) match/base >= 1.25 -> PASS; MBPP and agent kept(16)
  >= 1.03 each (secondary); KILL if HumanEval+ kept(16) < 1.03. 2.5: kept per check-ms with ms(rows) = 88 + 7.82 x rows
  (D-23c''s 45 GB code check, 189 ms at 12.9 rows, and D-24's 7.82 ms per row): len25 >= 1.10 x match16 -> PASS, < 1.04
  -> KILL. Beside it: the share of positions overridden, the override's precision, kept by depth.
- Prediction: the probe's x1.071 / x1.375 come from a teacher-forced profile; the real walk keeps most of it on code:
  match16 +5-7 %, match32 +20-30 %. PASS 45 %, BETWEEN 35 %, KILL 20 %. 2.5: BETWEEN.

### MATCH RESULT: BETWEEN on code (Mon 11-02 17:08; run 17:01-17:08 shared, x4 paused at a checkpoint for it on Kamen's priority call; data/match_read.txt, d02b_walk_m_<arm>.txt)
- First attempt 16:59 crashed on the MBPP set: the capture's h is hidden_norm(fc(taps)) as bf16 bits (L1-20's training
  format), so the walk now views the bits exactly and runs that set with fc / hidden_norm as identity, as the trainer does.
- Kept per check (DFlash2 L1-20, official loop, anchors moving with the override):
  | set | base16 | match16 | base32 | match32 | len25 (rows / check) |
  |---|---|---|---|---|---|
  | HumanEval+ (24) | 9.57 | 9.93 (x1.038) | 10.26 | 12.35 (x1.204) | 12.03 (19.8) |
  | MBPP (100, held-out for L1-20) | 5.94 | 5.82 (x0.980) | 6.00 | 5.91 (x0.985) | 5.85 (16.7) |
  | agent (12) | 3.01 | 3.15 (x1.047) | 3.02 | 3.19 (x1.056) | 3.18 (17.4) |
- The override in the real walk: code 16 rows: a match at 39.7 % of reached positions, right 93.6 % (the probe's
  teacher-forced 99.2 %: here the suffix carries the block's own drafts); it changed 618 drafts, 381 to right where the
  drafter had 145. 32 rows: 28.2 % / 86.2 %, 2,236 changed (1,664 vs 131). MBPP: 5.3 % / 32.1 % (the 125B does not
  restate MBPP's prompt; its short answers copy little); agent: 10.5 % / 51.3 %.
- LINE (HumanEval+ kept(16) >= x1.06 and kept(32) >= x1.25 PASS, < x1.03 KILL): x1.038 / x1.204 -> BETWEEN. Secondary
  (>= x1.03): agent x1.047 yes, MBPP x0.980 no (the override costs there).
- 2.5 (kept per check-ms, ms = 88 + 7.82 x rows): code len25 x1.063 over match16 -> BETWEEN (PASS 1.10, KILL 1.04);
  MBPP x0.981, agent x0.962. Report only (the orchestrator), C0's SSD-hidden ms = 36.1 + 2.98 x rows: code x1.067.
  Always-32 loses on cost (code 36.5 vs 46.6 kept per s at the pre-registered model).
- Reading: on code the override is real (match32 keeps 12.35 = x1.29 over today's 16-row 9.57) but below Fable's profile
  (x1.071 / x1.375), because in-block drafts feed the suffix. It needs a gate where matches are imprecise (MBPP): a
  per-session precision tracker or k >= 12 is the obvious next step (post-hoc, not tested). Prediction was PASS 45 %.

## EARLYTAP: can DFlash2 draft from early taps only? (Fable ANE_DRAFTER.md, mission 18) -- pre-registration (written before any run, Mon 11-02 17:38; the orchestrator's queue, before DIVERSE2; tonight after x4, before BigMoE's ticket)
- Why: DFlash2 reads the check's own taps at layers 5 / 15 / 25 / 35 / 45 of 48, so the next block's draft can start only
  after ~94 % of the check (a ~12 ms window today). A drafter reading only taps <= 35 or <= 25 opens 43-75 ms today
  (14-26 ms at the floor), enough for a Neural Engine drafter (Leap's ANE-1: a block in 23.8-25.6 ms).
- Data: L1-20's own windows (~/Leap/data/l120_windows.jsonl, 109: 99 train + 10 val), re-extracted with the 125B
  (sticky/taps_extract.py = Leap's dfl2c_extract.py + tap variants): per position the 5 taps (mean of the 4 HC streams),
  then h = hidden_norm(fc(taps)) with the base drafter's frozen fc / hidden_norm, as L1-20's recipe, for three tap sets:
  h5 (all: the control; must reproduce ~/Leap/data/l120's h), h4 (tap 45 zeroed: taps <= 35), h3 (taps 35, 45 zeroed:
  taps <= 25). Zeroing a tap before the frozen fc = removing its columns. A trained fc might do better: this is the
  conservative, recipe-exact version.
- Training: D-09's protocol (sticky/d09_train.py, incoai init, 2 epochs, pmask 0.5, ~201 steps, ~31 min each) on each set.
- Walk: block 16, official loop (sticky/d02b_walk_match.py with TL_WALK_TAPMASK: the same taps zeroed in the walk sets'
  raw taps): the 24 HumanEval+ windows, the same 100 MBPP answers as MATCH (raw taps re-extracted, held-out for these
  drafters), the 12 agent windows.
- LINES (Fable's, on HumanEval+ kept per check vs the retrained control): h4 >= 0.95 x -> the <= 35 pipeline is alive;
  h3 >= 0.90 x -> the <= 25 pipeline is alive; KILL if h4 < 0.93 x. MBPP / agent reported beside. Also: the control
  against L1-20 itself (9.57) = this recipe's seed noise.
- Prediction (Fable's): h4 loses 3-6 %, h3 8-15 %. Mine: h4 -4 %, h3 -10 %: h4 alive 55 %, h3 alive 40 %, KILL 25 %.
- EARLYTAP notes (18:30): (1) the orchestrator: frozen fc with zeroed taps is a lower bound; if h4 lands BETWEEN
  (0.93-0.95 x the control), a trained-fc arm is the follow-up before calling the <= 35 pipeline dead. (2) The extractor
  also keeps the raw per-layer taps (5 / 15 / 25 / 35 / 45) of L1-20's 109 windows with the 125B's top-32
  (data/taps_raw/*.npz: f [n, 5, 2560] f16, tk, lp, ids, asst, split) for Fable's CPU 'sureness probe' at layers 25 / 35.
- COPYGATE (Fable's copygate session's pre-registered gate for MATCH's override; its lines in ~/FableGod/copygate/GATE.md
  section 5): run by chain v9 after x4, before EARLYTAP, in a shared slot (turns/copygate_job.sh -> its gate_walk.sh;
  reviewed: walks only, the gate's session duel learns from positions <= k, i.e. what a check reveals).

### E2 / D2 RESULT (x2 + x4): PASS by the line, a flattening slope (condition 8; Mon 11-02 19:13; x4 = +48,640 MBPP code tokens, 669 steps over three slots 16:14-19:08 (paused twice for MATCH, one swap-guard stop; resumed each time from its checkpoint); data/e2_read.txt)
- Code kept per check (block 16, 24 held-out HumanEval+): L1-20 9.57 -> x2 9.79 (+0.22) -> x4 9.85 (+0.06); math 7.00 /
  6.99 / 7.06 (its data unchanged). LINE (slope on L1-20, x2, x4; KILL < +0.05, PASS >= +0.10): +0.140 -> PASS.
  All 5 points (with D-09's 1/4, 1/2): code +0.130, math +0.100 per doubling.
- Reading: more of the 125B's own code still helps past L1-20's data, but the second doubling gave a third of the first
  (one seed per point, so the step-by-step slope is noisy: x2's +0.22 may be partly seed). At ~+0.14 per doubling code >= 13
  needs ~23 more doublings: for condition 8 data volume is a real, slow input (Fable's E6 harvest), not the route.
- Prediction was PASS 15 % / BETWEEN 40 % / KILL 45 %.

### COPYGATE RESULT: BETWEEN (Fable's copygate session's gate for MATCH's override; run by MTP Head 19:14-19:21 shared; data/gate_read.txt, d02b_walk_g_<arm>.*; its pre-registration ~/FableGod/copygate/GATE.md section 5; written 19:22)
- Sanity: the gate's 'all' mode equals MATCH's override bit for bit on 3 of 3 sets.
- Kept per check vs the same-rows base walk (m_base16 / m_base32):
  | arm | HumanEval+ | MBPP | agent |
  |---|---|---|---|
  | G2 copygate, 16 rows | x1.041 | x1.000 | x1.032 |
  | G2 copygate, 32 rows | x1.211 | x1.001 | x1.036 |
  | G1 pduel, 16 / 32 | x1.041 / x1.211 | x0.995 / x0.997 | x1.044 / x1.048 |
  | G0 prompt-only, 16 | x1.041 | x0.994 | x1.041 |
- LINE (G2: HumanEval+ >= x1.035 and x1.18 at 32, MBPP >= x1.00, agent >= x1.04 PASS; HumanEval+ < x1.03 or < x1.15 at 32 or
  MBPP < x0.99 KILL): x1.041 / x1.211 / x1.000 / x1.032 -> BETWEEN (agent short). The pre-registered fallback (G2 misses
  only agent -> G1 decides): G1 misses MBPP (x0.995 < x1.00), not a KILL -> BETWEEN.
- Reading: the gate removes MBPP's loss (x0.980 -> x1.000) and keeps code's gain (x1.038 -> x1.041; x1.204 -> x1.211): the
  override becomes safe to wire everywhere; its remaining gain is code's, at 32 rows x1.21 (12.42 kept per check).

## DIVERSE2: a second drafter trained to be wrong elsewhere (cloud Fable's committee answer via the orchestrator) -- pre-registration (written before any run, Mon 11-02 20:42; Kamen's Ultra week: queue item 1)
- Why: two DFlash2 drafters miss together: P(member 2 wrong | L1-20 wrong) = 0.61 (hpdU) / 0.66 (hpdH) on drafter_ensemble.out;
  with independent errors the pair would be right 97.7 % per position instead of 94.0-94.4 %. So member 2 should be
  trained to be wrong elsewhere, not to be better.
- Data: L1-20's training data (~/Leap/data/l120). L1-20's teacher-forced right / wrong per (anchor, depth) on it from the
  conf-out pass (sticky/hpd_train.py --conf-out, new: data/hpd/conf_l120_right); L1-20's live block starts (block 16, the
  official loop) simulated on the CPU from that (up to the first miss teacher-forced and own picks agree) -> data/hpd/starts_l120.
- Arms, all from incoai's base (no --init), L1-20's recipe in sticky/hpd_train.py (S 32 / A 8, fkl, pmask 0.5, lr 2e-5, 2
  epochs, the same seeds and order): M2-0 (no weights: the control), M2-W (x4 loss where L1-20 was wrong), M2-WS (x4 where
  wrong, and x4 for blocks anchored at L1-20's live block starts; D-09's trainer anchors at every position).
- Measurement: each member at HPD-0's anchors (TL_WALK_CF, teacher-forced picks) on the 24 held-out HumanEval+ and 60
  GSM8K: own per-position accuracy, q = P(member wrong | L1-20 wrong), either-right, the higher-cf pick; then the joint-chain
  ensemble L1-20 + member (sticky/d02b_walk_ens.py) kept per check.
- LINES (code, the orchestrator's suggestion): PASS if M2-W or M2-WS has q <= 0.50 with its own accuracy >= 0.90; KILL if
  both weighted arms have q >= 0.60 (no better than hpdU); BETWEEN otherwise. Beside it: M2-0's q (the recipe's own
  overlap), E2-x2's q (DUALDRAFT tonight), and the ensembles' kept vs L1-20 (DUALDRAFT's +8 % line).
- Prediction: the x4 weight lowers q modestly (~0.61 -> ~0.55) at ~1 point of own accuracy: BETWEEN 55 %, PASS 20 %, KILL 25 %.
- Schedule (Kamen's laptop rule): conf-out now (small, ~6 min shared); the three trainings (~30 min each) in the big window
  (Tue after the night chain, before 15:00, or Wed from 00:30); walks after.

## KWALL: the knowledge wall at the 125B's mid rows (Fable KNOWLEDGE_WALL.md 0.5, mission 24 + cloud Fable's deciding experiment) -- pre-registration (written before any run, Mon 11-02 20:56; Ultra week queue items 2-3)
- Rows: the 12 code map windows, the 125B's class per row (~/FableGod/floor_rows.json: sure / mid / tie; 393 mid rows in
  Fable's count), at the positions the official loop reached at HPD-0's anchors (the map's gate source).
- (a) Mission 24 (a small GPU walk; by day only with a MEM share lock): DFlash2 L1-20's 16 candidates and its selector's
  teacher-forced softmax at every tested position (new TL_WALK_TOP16 in sticky/d02b_walk_match.py, HPD-0's anchors, the 24
  HumanEval+ windows; data/d02b_walk_kw_code.npz); the native MTP chain's draft from the same anchor and depth = D-24's
  teacher-forced chain, head b5 (data/d24/code_b5.npz, rank 0). Read-out sticky/kw_read.py.
  LINE (mid rows): the 125B's token in DFlash2's top-3 >= 0.85 OR the union (DFlash2 right or MTP right) >= 0.80 -> PASS
  (a selector problem worth one learned picker); top-3 < 0.75 AND union < 0.72 -> KILL (the mid rows are the model's);
  BETWEEN otherwise. Prediction: top-3 ~0.85 (Fable's ten misses: 7 of 10 in the 125B's top-3), union ~0.75: BETWEEN 50 %.
- (b) Cloud's deciding experiment, MTP half (CPU, now): the MTP head's (b5) depth-1 accuracy per class at the 3,283 code map
  rows, from D-24's stored teacher-forced ranks (row i predicts token i + 1 = D-24's anchor i, depth 1). LINE (cloud's): MTP
  mid >= 0.90 -> a cheap reader of the trunk state exists (a 2-4-layer distilled head is the next design); MTP mid <= 0.85
  -> the wall is real IF the near-copy's mid is >= 0.90 (that half needs the near-copy's per-row predictions, not in my
  data); between -> BETWEEN. Prediction: MTP mid ~0.75-0.80 (D-24: depth 1 on code 0.978 overall; mids are the hard 12 %).

### KWALL (b) RESULT, the MTP half: BETWEEN [numbers corrected Tue 11-03 15:35: 'ALIGNMENT FIX'] (Mon 11-02 20:56; CPU on D-24's stored ranks; sticky/kwall_mtp.py, data/kwall_mtp.txt)
- The native MTP head b5 at depth 1 (teacher-forced: the trunk's state at the previous row + the true token), at the 12 code
  map windows' rows by the 125B's class: sure 2,922 rows 0.985 (top-3 0.999); mid 394 rows 0.860 (top-3 0.982); tie 50 rows
  0.800 (top-3 0.920); all 3,366 0.968. DFlash2 at the same rows (its draft at the row's depth in the official loop, so
  deeper on average, not a like-for-like column): sure 0.955, mid 0.659, tie 0.300.
- LINE (cloud's): MTP mid 0.860 -> BETWEEN (>= 0.90 would be a cheap reader; <= 0.85 the wall if the near-copy >= 0.90).
  The near-copy half needs its per-row predictions (not in my data): open.
- Reading: one MoE layer on the trunk's state gets 86 % of the 125B's wording choices right at the very next position and
  has the 125B's token in its top-3 at 98 % of them. So at depth 1 the information is cheaply readable; the drafters' mid-row
  losses (DFlash2 0.66) come from depth (no trunk state past the anchor) as much as from the choices themselves. Lead (not
  tested): a depth-1 top-3 MTP proposal or a distilled 2-4-layer head at depth 1 (the cloud's next design if mid >= 0.90).
- DIVERSE2 note (21:00): the conf-out pass run by day at 20:43 took a 51 GB footprint (it built 32 groups x 8 anchors per
  call: full-vocab f32 logits for 256 blocks x 31 depths ~8 GB per call, kept by MLX's cache), collided with Breakthrough's
  map run (swap 17 GB, memory pressure on Kamen's laptop) and was stopped by the orchestrator at 20:57 (33 of 109 windows).
  Fixed: micro-batched (--micro groups per call), mx.clear_cache per window, the MLX peak printed. It re-runs inside
  turns/dv_job.sh in Tuesday's big window (gate 23 GB until its peak is measured). HPD-1's conf phase (08:37) used the same
  code with 53 GB free, so its "peak ~12" estimate was never true.

## DEPTH32: the AR-flat deep half (Fable DEPTH_WALL.md section 4, mission 23) -- pre-registration (written before any run, Mon 11-02 21:02; the orchestrator's queue, after EARLYTAP / DUALDRAFT; big window)
- Why: DFlash2's per-depth rate decays (0.95 shallow -> 0.86 deep on code); the D-22 MTP chain is the one measured object
  without decay (~0.94 flat). A 32-row check with DFlash2 drafting depths 1-8 and the trained MTP chain 9-32 could keep
  14+ per check on code.
- Measurement (exact up to the first miss, as D-22's evaluation): (1) the MTP head b5's teacher-forced hits and argmax at
  every anchor, depths 1-31 (sticky/d23h_mtp31.py = D-24's script at D = 31; the trunk's widened stream: data/d16w for
  HumanEval+, the capture's wd for MBPP; ~7-15 GB, MLX peak printed); (2) DFlash2 L1-20's own picks at every position
  (d02b_walk.py, TL_WALK_EVERY + TL_WALK_CF, block 16; HumanEval+ raw taps, MBPP: data/taps_mbpp_walk); (3) CPU: the
  official loop at 32 rows on those tables (sticky/d23h_sim.py): the draft = DFlash2's own picks at depths 1-8 + the MTP
  argmax at 9-31; arms: DFlash2-32 (reference: the real walk m_base32, 10.26 / 6.00), MTP-32 (the MTP chain 1-31), HYB
  (1-8 DFlash2 + 9-31 MTP), HYB+G (COPYGATE's gate, ~/FableGod/copygate/copygate.py, on the whole 32-row draft).
- Sets: the 24 HumanEval+ windows, the same 100 MBPP answers as MATCH / EARLYTAP.
- LINE (Fable's, on the best realizable arm, HYB or HYB+G): kept(32) >= 14 on HumanEval+ AND >= 7.5 on MBPP -> PASS;
  < 13 on HumanEval+ or < 6.5 on MBPP -> KILL; BETWEEN otherwise. The step cost (24 MTP steps per block) is Leap's half.
- Prediction: the MTP chain is flat near 0.94, so from depth 9 the hybrid adds ~3 kept over DFlash2-32 on code: HYB
  ~13-13.5, +G ~14 on HumanEval+; MBPP ~6.5-7: BETWEEN 50 %, PASS 25 %, KILL 25 %.

## ANCHORCOV: anchor coverage for the ANE's three pre-drafted next blocks (cloud task 15, ~/FableGod/CLOUD_ANSWER_ROAD350.md; the orchestrator 21:4x) -- pre-registration (written before the replay, Mon 11-02 21:38; CPU)
- Design: while the check finishes its last layers, the Neural Engine drafts the next block from 3 anchors: (a) all drafts
  accepted, (b), (c) right after the drafter's two lowest-confidence rows; the one matching the real accepted length goes in.
- Replay: HPD-0's walks of DFlash2 L1-20 (code, math, multi, IFEval; per check accepted k of m checkable drafts and the own
  picks' cf), block 16, the walk's loop (no 0.3 cut). A check is covered when its outcome (all m accepted, or the first
  miss at row k) is among the anchors: "all" + the N - 1 lowest-cf rows; N = 2, 3, 4. Beside it (information): the rows
  ranked by the drafter's own first-miss probability (prod_{i<j} cf_i x (1 - cf_j)) instead of the raw lowest cf.
- LINE (cloud's): coverage >= 50 % at N = 3 on code and chat (multi + IFEval) -> PASS; else KILL for the 3-anchor design
  (report N = 2 / 4 beside it). L1-20's cf stands in for the early-tap drafter's (EARLYTAP's walks have no cf).
- Prediction: code ~55-65 % (many checks accept all 15 or miss at the one unsure row), chat ~40-50 %.

### ANCHORCOV RESULT: KILL by the stated rule; alive with first-miss-probability anchors (Mon 11-02 21:38; CPU, sticky/anchorcov.py, data/anchorcov.txt)
- Coverage of the check's real outcome by N anchors ("all accepted" + the N - 1 lowest-cf rows) vs (information) the rows
  ranked by the drafter's own first-miss probability prod_{i<j} cf_i x (1 - cf_j):
  | set (checks) | all accepted | lowest cf N=2 / 3 / 4 | first-miss prob. N=2 / 3 / 4 |
  |---|---|---|---|
  | code (666) | 0.294 | 0.494 / 0.595 / 0.677 | 0.641 / 0.782 / 0.857 |
  | math (2,046) | 0.109 | 0.275 / 0.416 / 0.532 | 0.481 / 0.672 / 0.776 |
  | multi (3,294) | 0.005 | 0.105 / 0.182 / 0.255 | 0.614 / 0.804 / 0.893 |
  | IFEval (2,850) | 0.010 | 0.142 / 0.241 / 0.320 | 0.577 / 0.799 / 0.886 |
- LINE (cloud's, the stated rule: >= 0.50 at N = 3 on code and chat): code 0.595, chat 0.210 -> KILL.
- Reading: on chat the first miss is almost always at depth 1-2 while the lowest-cf rows sit deeper, so raw lowest cf picks
  the wrong rows; the drafter's own first-miss probability picks them: N = 3 covers 0.78 of code checks and 0.80 of chat.
  The 3-anchor ANE design is alive with those anchors (pre-registered as information; a design change, to be pre-registered
  as such by its owner). L1-20's cf stands in for the early-tap drafter's.
- EARLYTAP addendum (the orchestrator 21:4x, cloud's extra lines, report only): kept >= 0.97 x the control with early taps
  (the h3 / h4 rows of et_read), the ANE's 12.4 ms under a busy GPU, a sync-free hand-off (Leap's halves).

### D-23x RESULT: the first-check stall is a cold pool after the prefill, plus slow reads (Tue 11-03 06:56; run 06:37-06:53 quiet, C 192, read trace; data/d23x_read.txt, d23x_runs.jsonl)
- 11 traced runs (4 agent + 2 chat prompts x 2 arms; one short). The prefill's reads go through the staging path and are
  not in the pool trace (0 reads); no prefetch or joint reads in this preset (all demand).
- Mean per check, first check vs checks 2-6:
  | set, arm | base MSB-3 reads | LSB-1 sidecar reads | read s (summed) | check ms | rows |
  |---|---|---|---|---|---|
  | agent, DFlash2 | 1,074 vs 489 | 325 vs 171 | 5.76 vs 1.13 | 2,526 vs 205 | 7.5 vs 6.2 |
  | agent, MTP | 1,528 vs 717 | 534 vs 245 | 8.97 vs 1.76 | 3,078 vs 255 | 15.0 vs 11.0 |
  | multi bg0, DFlash2 / MTP | 865 / 995 vs ~220 | 266 / 333 vs ~78 | 2.3-2.8 vs 0.33 | 297 / 342 vs ~110 | 6-8 vs 4.4 |
  | IFEval 1220 | 271-478 vs ~258 | 82-151 vs ~85 | 0.4-1.0 vs 0.4 | 351-355 vs ~121 | 1-2 vs 2-3 |
- Per layer (agent): the first check's base reads spread over all 48 layers (by quarter 360 / 334 / 324 / 250 vs 181 / 151 /
  129 / 125 later): a pool cold everywhere, not a few layers. Both tiers are cold (NewIdeas' question: base and sidecar alike).
- Reads are also slower, not only more: agent text23's first check (9 rows) read 776 records over a 3.2 s span (later
  checks ~550 in 0.3 s); edit00 2,428 records in 2.3 s. So ~2-3x the reads at ~4-10x the wall time per read: a cold pool
  AND a slow I/O path right after the long prefill (page cache state or the prefill's I/O tail; not measured here).
- Lead for BigMoE / Leap (exact): warm the decode pools at the end of the prefill (read the base + sidecar records of the
  experts the last prefill rows routed to, while the prompt's last chunk finishes), which removes most of the extra reads;
  the latency half needs a timed look at the I/O queue after prefill. TRIED's PARKED pool warm-up is this; the trace says it
  pays most on long prompts (agent: ~2.3 s per turn).

### EARLYTAP RESULT: KILL (Tue 11-03 09:30; extraction Mon 19:22-19:52, trainings Tue 06:53-09:23 shared, walks 09:23-09:26; data/et_read.txt, d02b_walk_et_<v>.txt)
- Self-check: the 5-tap re-extraction equals L1-20's stored h bit for bit on 109 / 109 windows. The retrained control
  (h5, D-09's recipe) keeps 9.66 on HumanEval+ vs L1-20's 9.57 (seed noise +0.09).
- Kept per check (block 16): | set | h5 control | h4 (taps <= 35) | h3 (taps <= 25) |
  code 9.66 | 7.04 (x0.729) | 5.43 (x0.562); MBPP 5.88 | 3.92 (x0.667) | 3.00 (x0.510); agent 3.00 | 1.91 (x0.637) | 1.68 (x0.560).
- LINE (HumanEval+ vs the control: h4 >= 0.95 alive, < 0.93 KILL; h3 >= 0.90 alive): h4 x0.729 -> KILL; h3 x0.562 not alive.
  Cloud's report-only line (>= 0.97 x): no for both.
- Reading: the last tap (layer 45 of 48) carries a large share of what DFlash2 drafts from: without it a quarter of code's
  kept is gone. This is the frozen-fc version (dropped taps zeroed before the base drafter's fc): a lower bound; a trained fc
  (and wider layers on the early taps) could recover part, but at x0.73 the ANE pipeline needs that as its own test.
  Prediction was h4 -4 % / h3 -10 % (KILL 25 %).

### DUALDRAFT RESULT: KILL (Tue 11-03 09:30; run 09:26-09:29 shared; data/dd_read.txt, d02b_walk_dd_*.txt)
- Self-check (L1-20 twice): 9.57 / 7.00 and 0 % second-drafter depths: OK.
- Joint-chain ensemble L1-20 + E2's x2: code 9.82 (x1.026 vs L1-20, x1.003 vs x2 alone 9.79), math 7.12 (x1.017); x2
  proposed 60 % / 51 % of the reached depths. LINE (>= x1.08 PASS, < x1.04 KILL): x1.026 -> KILL.
- Why (teacher-forced at HPD-0's anchors, all 15 depths): L1-20 right 0.752, x2 right 0.757 per position on code, and
  q = P(x2 wrong | L1-20 wrong) = 0.866 (math 0.892): two drafters trained on overlapping data with the same recipe are
  wrong at the same places (the orchestrator's hpdU / hpdH figure was 0.61 / 0.66 on another position set). This is
  DIVERSE2's baseline: its weighted members have to bring q from ~0.87 toward 0.50.
  Prediction was KILL 40 %.
- D-23x note (09:30): BigMoE already has the fix built: mission 20, R57_A6=warm (PREFILL_WARM: the prompt's routed experts
  into the base + sidecar pools by frequency at prefill end; C 160 / 256, 3 code + 3 agent prompts); last night's 45 GB
  runs were memory-blocked, so its measurement is pending. D-23x is the trace that says where the time goes.

### KWALL (a) RESULT: PASS -- the mid rows are a selector problem [union corrected 15:35: 'ALIGNMENT FIX'] (Tue 11-03 11:46; sticky/kw_read.py, data/kw_read.txt; written 15:12)
- At the 125B's mid rows of the 12 code map windows (positions the official loop reached at HPD-0's anchors): the 125B's
  token is inside DFlash2 L1-20's top-3 (its teacher-forced selector softmax over the 16 candidates) at 0.888; the union
  (DFlash2's pick right or the MTP chain's, b5 teacher-forced) 0.903. LINE (top-3 >= 0.85 or union >= 0.80 PASS): PASS.
- With KWALL (b) (the MTP head's depth-1 top-3 holds it at 0.982 of mid rows) the knowledge wall is not in the candidates:
  the right token is almost always proposed, and the pick is what fails. Fable's reading: one learned picker is worth a
  test despite L1-21 / SD-sel's +1.2 % (its lattice was top-64 with a 12M picker).

### DIVERSE2 RESULT: KILL -- re-weighting does not make a second drafter wrong elsewhere (Tue 11-03 15:12; conf-out 11:46-11:51 (micro-batched, 281 s), trainings 11:51-13:20, walks to 13:24, all shared; data/dv_read.txt)
- Members from incoai, L1-20's recipe (179 steps each): teacher-forced at HPD-0's anchors on code (9,750 positions) / math:
  | member | own (L1-20 0.752) | q = P(wrong given L1-20 wrong) | either right | ensemble kept (code) |
  |---|---|---|---|---|
  | M2-0 control | 0.750 | 0.901 | 0.776 | 9.67 (x1.010) |
  | M2-W (x4 on L1-20's misses) | 0.734 | 0.890 | 0.779 | 9.64 (x1.007) |
  | M2-WS (+ x4 at live block starts) | 0.735 | 0.893 | 0.778 | 9.60 (x1.003) |
  Math: q 0.889 / 0.885 / 0.883; ensembles x1.010 / x0.994 / x0.991.
- LINE (q <= 0.50 with own >= 0.90 PASS; both weighted arms q >= 0.60 KILL): q 0.890 / 0.893 -> KILL. (The own >= 0.90 bar was
  out of reach on this measure, all 15 depths teacher-forced, where L1-20 itself is 0.752; q decides.)
- Reading: even an independently trained twin (M2-0) is wrong at 90 % of L1-20's misses, so the misses belong to the data
  (the 125B's wording choices), not to the training run; x4 weights move q by one point and cost 1.7 points of own accuracy.
  With KWALL (a): the drafters' candidates already hold the right token at mid rows; a second drafter adds the same
  candidates. The remaining lever is the pick (a learned re-ranker over the top-3 / top-16 at mid rows), not a second member.
  Prediction was BETWEEN 55 % / KILL 25 %.

### WITHDRAWN 15:35 (off-by-one, see 'ALIGNMENT FIX' and 'MTPPICK aligned'): MTPPICK (lead, CPU): DFlash2 and the native MTP head are wrong at different places; a per-depth confidence pick keeps +23 % (code) / +46 % (math) per check (Tue 11-03 15:15; sticky/mtp_overlap.py + the pick simulation, data/mtp_overlap.txt)
- Both teacher-forced at HPD-0's anchors, depths 1-15 (exact up to the first miss): DFlash2 L1-20's pick (HPD-0 walks) and the
  MTP head b5's draft (D-24). Per position: code DFlash2 0.751 / MTP 0.861, q = P(MTP wrong | DFlash2 wrong) 0.301 (DFlash2
  twins 0.87-0.90), either 0.925, the higher-confidence pick (DFlash2's tcf vs MTP's top-1 p) 0.888; math 0.682 / 0.838,
  q 0.240, either 0.924, pick 0.862; multi 0.213 / 0.438, q 0.606; IFEval 0.281 / 0.450, q 0.585.
- Kept per 16-row check at the same anchors (the chain takes at each depth the more confident drafter's token and stops at its
  first wrong pick): code DFlash2 9.54 (its real walk 9.57) / MTP 10.13 / pick 11.70 / oracle 12.73; math 6.97 / 8.01 /
  10.17 / 12.01; multi 2.05 / 3.03 / 3.38 / 4.15; IFEval 2.44 / 2.93 / 3.63 / 4.48.
- The pick needs no extra rows (one chain is verified); its cost is running both drafters: DFlash2's block pass (~12 ms) and
  the MTP chain's 15 sequential steps fed the picked tokens (~2.2 ms each today, 33 ms; <= 1 ms each in Leap's fused /
  ANE target). At C 192's measured checks: code 9.54 per 202 ms -> 11.70 per ~235 ms (+5 %), math 6.97 / 130 ms -> 10.17 /
  ~163 ms (+16 %); at a 1 ms step the gain is most of the kept gain.
- TRIED: not there. D-05 MTPHYB split depths (near MTP, far DFlash2: <= +1 %); D-18 MTP at positions 1-2 (KILL); L1-06 verified
  both chains as rows (KILL on rows); D-19 / TYPERULE switch per prompt; DUALDRAFT / DIVERSE2 paired DFlash2 twins (q ~0.9,
  KILL). New: a per-position confidence pick between two architecturally different drafters (q 0.24-0.30), no extra rows.
- Next: a live, exact A/B in the D-23 harness (an arm drafting with both and picking per depth), timed, at C 192 / 224
  (a 45 GB-route quiet ticket); pre-registration next.

### ALIGNMENT FIX: four cross-drafter read-outs read D-24's MTP rows one position late (found Tue 11-03 after the 15:15 MTPPICK commit; CPU re-runs 15:35)
- The bug: D-24's tables (data/d24/<set>_b5.npz) index the MTP anchor s = a0 + r, whose depth d predicts ids[s + d + 1]
  (d22_ttt.chain; D-24's own walk used anchor gen + p - 2, which is right). The scripts that put the MTP chain beside a DFlash2
  check at anchor A (drafts predict ids[A + 1 ..]) or a map row t (predicts t + 1) used r = A - a0, i.e. s = A; the live chain
  there is s = A - 1. The wrong rows predicted one token later, from the trunk's hidden at A, which the loop does not have yet.
- Test (CPU, HPD-0's checks, depth 1): P(MTP wrong | DFlash2 wrong) = 0.237 / 0.112 (code / math) with s = A and 0.615 /
  0.543 with s = A - 1: two drafters share hard tokens, so the right alignment shows the overlap.
- Affected, re-run aligned: sticky/mtp_overlap.py (the MTPPICK lead: withdrawn, next section); sticky/kwall_mtp.py (KWALL (b):
  sure 0.985 -> 0.998, mid 0.860 -> 0.855 (top-3 0.982 -> 0.975), tie 0.800 -> 0.340; still BETWEEN); sticky/kw_read.py (KWALL
  (a): MTP chain right at mid rows 0.786 -> 0.707, union 0.903 -> 0.796; the PASS stands on DFlash2's top-3 0.888); MoonShots'
  export data/mtp_map_code (v2 = data/mtp_map_code_v2 by sticky/mtp_map_export.py; v1 marked WRONG_ALIGNMENT.txt; the
  per-row depth-1 flags agree on 0.84-1.00 of rows, so per-row decisions built on v1 moved by one row). Fixed before any run:
  sticky/d23h_sim.py (DEPTH32's HYB arm). Not affected: D-24's and D-23's MTP-only numbers, everything DFlash2-only.

### MTPPICK aligned: KILL on paper -- the pick keeps +7 % (code) / +11 % (math) per check, the MTP chain's 33 ms per cycle costs more (Tue 11-03 15:35; sticky/mtp_overlap.py fixed, sticky/mtppick_sim.py; data/mtp_overlap.txt, data/mtppick_sim.txt)
- Aligned overlap at HPD-0's anchors, depths 1-15: code DFlash2 0.752 / MTP 0.859, q 0.436 (the bug said 0.301), either 0.892,
  pick 0.852; math 0.682 / 0.835, q 0.400, either 0.873, pick 0.808; multi q 0.676; IFEval q 0.702 (DFlash2 twins 0.87-0.90).
- Exact official loop (new tables: DFlash2 at every position, turns/every_walk.sh 15:23-15:33 shared, MLX ~2 GB; each arm runs
  its own loop), 16-row checks, kept DFlash2 / MTP / pick / oracle: code 9.58 / 10.20 / 10.28 (x1.073) / 11.74; math 7.01 /
  7.73 / 7.78 (x1.110) / 9.34; multi 2.06 / 2.92 / 2.72; IFEval 2.45 / 2.74 / 2.70.
- Est. tok/s (D-24's model: F + 7.82 ms x rows, + 12 ms DFlash2 block pass, + 2.2 ms per MTP step): code 43.3 / 42.1 / 40.4
  (pick x0.933), math 34.5 / 34.5 / 32.9 (x0.955); with the shipped cut (t 0.3): code 51.0 / 48.7 / 46.8 (x0.916 vs dfl), math
  44.3 / 44.7 / 39.3 (x0.887); on chat the pick loses to the MTP chain it already runs (multi x0.804).
- At 1 ms per MTP step the code pick is flat (x1.005 at 16 rows): it gains only if the second drafter is nearly free.
  Verdict: KILL (on paper). The 15:15 'lead' (+23 % / +46 %) was the alignment bug; nothing was sent on it.

### TOP3-USE: tree-of-top-3 vs a learned picker vs MTPPICK, prior art + paper price on the 16-row checks (the orchestrator's item 2, Tue 11-03 15:35; desk + the CPU tables above) -- none pre-registered
- Question: KWALL (a) puts the 125B's token in DFlash2's top-3 at 0.888 of mid rows (top-1 0.659). The published ways to use
  it: verify several candidates per position (a tree), or pick a better single candidate (a learned picker); the third is a
  second drafter's opinion (MTPPICK). Priced on the 45 GB route: check = F(type) + 7.82 ms x rows (F code 84.1 ms), DFlash2
  block pass 12 ms, MTP step 2.2 ms; base = the shipped DFlash2 chain (cut t 0.3): code 9.24 kept at 10.9 rows, ~51 tok/s
  est.; at a fixed 16 rows 9.58 kept, ~43 tok/s.
- Prior art, trees (tag published): Medusa (arXiv 2401.10774: per-offset heads, a calibrated sparse Cartesian tree, tree
  attention), SpecInfer (ASPLOS 2024: trees merged from several small drafters, topology-aware mask), EAGLE-2 (arXiv
  2406.16858: dynamic trees, expand by path confidence, rerank, keep the top m), Sequoia (arXiv 2402.12374: DP-optimal tree
  per budget, hardware-aware size), SpecExec (arXiv 2406.02532: offloaded dense models, thousands of nodes because the verify
  cost is flat when weight streaming dominates). That premise fails for a routed MoE: every row brings new experts
  (DraftExpert arXiv 2607.24434: expert-set expansion; ours 7.82 ms per row), and OEA sharing is small: a sibling row adds
  2.35-2.55 new experts per layer vs 2.79 for a late chain row (TRIED M-10, 0.84-0.91x).
- Prior art, pickers (tag published; L1-21 adapted): DFlash2's selector chain over its top-16 is already a learned
  first-order picker; higher-order selectors (SD-sel, LiLiCorr-style) publish about level with it; L1-21's 12 M-param
  lattice picker: +1.2 % kept (9.54 -> 9.65) against an oracle pick of 14.11. KWALL (a)'s top-3 headroom is that oracle gap.
- Prior art, two drafters (MTPPICK tag adapted): TAPS (arXiv 2603.27027, ICML 2026) routes each request to one of two
  drafters by mean tree-node confidence; its best arm verifies both drafts as one merged tree (rows); HedgeSpec (arXiv
  2510.20064) picks a drafter online; multi-draft speculative sampling (arXiv 2410.18234) selects among verified drafts.
- Price, code, vs the shipped chain at equal cost:
  - tree of top-3 over DFlash2's lattice (EAGLE-2 / Sequoia-style best-first; TRIED L1-22, TREES): 10.43 kept at 16 rows
    (+9 % vs the 16-row chain) but -10.1 .. +2.1 % vs the best chain at equal cost under 5 cost models incl. D-23's measured;
    L1-22 294 vs 311 tok/s vs the cut chain. Needs a tree check on the Gated DeltaNet layers: per-node recurrent states
    (~50 MB per layer) + per-row key gathers (TRIED 'Exact tree checks'; Bole / TreeWY reorder the recurrence, not bitwise,
    FAST at best). Not built.
  - learned picker over DFlash2's top-3 / top-16 (L1-21): +1.2 % kept, no rows; ~+0.3 % tok/s after its own cost.
  - MTPPICK (aligned, above): +7 % kept, x0.93 (16 rows) / x0.92 (cut).
- Pick: none. The tree is the only one that uses the top-3 headroom, and it does so by verifying more rows, which is
  where the 45 GB route pays (7.82 ms per row, siblings 0.84-0.91x); the picker is ~0 (+0.3 %), and a live A/B resolves
  ~+-3 % at best (order w 0.97-0.99), so a pre-registered run of either would be inside its noise. The top-3 lever is the
  row cost, not the pick: revisit the tree when a verify row gets much cheaper (warm pools, sidecar drop, the FAST row-cost
  work), where D-24's oracle width 2-3 headroom (chat 3.6-4.05 vs 2.3-2.5 tokens per pass) opens.

### DEPTH32 amendment 1 (Tue 11-03 15:35, before any run of steps 1 and 3)
- Step 2 (DFlash2 at every position, code + MBPP) ran today 15:23-15:33 as a small shared job (turns/every_walk.sh, also math /
  multi / IFEval for MTPPICK); turns/d23h_job.sh skips it. sticky/d23h_mtp31.py also writes the top-1 p (p1_<i>, D-24's
  definition). sticky/d23h_sim.py: the MTP row is r = A - 1 - a0 ('ALIGNMENT FIX'; the first version read one position late).
  Left: step 1 (the MTP head to depth 31 on 24 HumanEval+ and 100 MBPP answers, ~5-8 min, MLX ~7-11 GB as D-24) + step 3 (CPU).

### CUTSWEEP (information, CPU): the shipped DFlash2 cut t 0.3 is within 0-2 % of the per-type best -- no knob to turn (Tue 11-03 15:38; sticky/cut_sweep.py, data/cut_sweep.txt)
- DFlash2's own running-confidence cut swept t 0-0.5 in the exact official loop (every-position tables: own picks and own cf,
  so the rows are exact), D-24's cost model. Best t vs t 0.3, est. tok/s: code 0.40 x1.003, math 0.40 x1.018, MBPP 0.35
  x1.000, multi 0.40 x1.012, IFEval 0.35 x1.009; the curve is flat over 0.3-0.4 (code 51.0 / 51.1 / 51.2). Agrees with S45d
  (t 0.5 -1.1 % on D-23's measured checks). A per-type t would buy <= +1.8 % (math), inside a live A/B's noise: not pursued.

### DEPTH32 RESULT: KILL -- the best realizable 32-row arm keeps 12.60 (HumanEval+) / 6.10 (MBPP) per check, line 14 / 7.5 (Tue 11-03 15:40-15:42 shared, MEM 12 GB; turns/d23h_job.sh; data/d23h/, data/d23h_read.txt; written 15:42)
- Step 1 (the MTP head b5, teacher-forced to depth 31): depth-1 right code 0.977, MBPP 0.935; 29 s + 107 s; MLX peak 9.7 /
  10.1 GB (the pre-registration said ~7-15). Step 2 = turns/every_walk.sh (15:23-15:33). Step 3 (CPU, the official loop at
  32 rows on the aligned tables, amendment 1).
- Kept per 32-row check: code DFlash2-32 10.26 (the real walk), MTP-32 12.55 (+G 12.85), HYB (DFlash2 1-8 + MTP 9-31) 12.19
  (+G 12.60); MBPP 6.00 / 6.23 (+G 6.21) / 6.11 (+G 6.10). LINE (hyb+g): 12.60 / 6.10 -> KILL (< 13 or < 6.5).
- The MTP chain alone beats the hybrid at 32 rows: DFlash2's shallow depths add nothing once the deep half is the MTP head.
  On MBPP no arm moves much (6.0-6.2): the deep half is not where MBPP's misses are.
- Priced (D-24's model): even the PASS line would run ~34 tok/s vs the shipped chain's ~51 on code; rows 17-32 keep ~0.15-0.2
  tokens each here (code (12.6 - 9.6) / 16), against ~0.40 needed at 7.82 ms per row. Prediction: BETWEEN 50 %, PASS 25 %,
  KILL 25 % (came KILL).

### HEADBENCH (information, < 1 GB GPU, shared): the MTP draft step's lm_head costs 1.47 ms of ~2.2 ms, at 242 GB/s = 85-94 % of the M5 Pro's streaming bandwidth (Tue 11-03 15:45; sticky/head_bench.py, data/head_bench.txt)
- The live step (MTPHead.draft_cut): embed, the MTP layer, the HC mixer, the full 248,320 x 2560 lm_head (4-bit g64, 358 MB),
  argmax + f32 softmax max, one mx.eval sync. The head alone with the same post-ops and sync: median 1.475 ms (p1 by
  logsumexp 1.473, argmax only 1.509; an empty op + sync 0.146 ms). So ~0.6 ms is the rest of the step (layer, HC, embed).
- 358 MB / 1.475 ms = 242 GB/s, against 258-287 GB/s measured streaming on this M5 Pro (Leap / FableGod probes): a faster
  matvec kernel has <= ~10-15 % to give. Only fewer head bytes help (draft-side only, so outputs stay exact): a 2-bit draft
  copy (~0.75 ms), a vocab subset (FR-Spec: dropped at the desk 11-01, 32K covers only 0.53 of multilingual answer tokens),
  or a trained low-rank head. Priced on D-23d's chat checks (draft 8.9 of 86 ms, ~3.5 steps): saving 0.7-1.1 ms per step =
  chat +3-4.5 %, agent on MTP ~+3 % over DFlash2. Under a live A/B's noise; not pursued unless bundled.

### ROUTEPRED: a learned expert-route predictor for prefetch (the orchestrator's item, Tue 11-03 15:57; steps 1-2 on the CPU; step 3 not pre-registered) -- KILL on paper for the next-check predictor
- Step 1, prior art. Layer-ahead gate predictors (published): Pre-gated MoE (a pre-gate picks the next layer's experts),
  HOBBIT (arXiv 2411.01433: next-layer prediction from gating inputs; mixed precision makes wrong guesses cheap), AdapMoE
  (current gating inputs through later gates), ProMoE (a learned classifier, sliding window, on the CPU), MoE-Infinity
  (request-level activation tracing). Draft-driven verifier-expert prediction (published): SP-MoE (arXiv 2510.10302: draft
  attention outputs through the target's gates, a cutoff-layer policy, async batched prefetch), MoE-SpeQ (a quantized draft
  model's multi-step expert look-ahead), DraftExpert (arXiv 2607.24434: router-agreement distillation predicts the
  verifier's expert sets; a secondary summary reports 86-88 % prefetch hits). Ours would be adapted. TRIED: V1 look-ahead
  ADOPTED (64 GB era), CORR PASS +2.9 % (24 GB), S10b / S10c M1a (mine) KILL at 24 GB (+0.3-0.9 %), R-57 / R-57b KILL at
  45 GB (wrong guesses evict, late reads), R-70D draft-window prefetch PARKED, P2-PREFETCH (12 GB two-pass) PASS on its CPU
  model (Fit12's T3R-PF tonight).
- Step 2, headroom (BigMoE's ~/BigMoE/data/c50/c45l_256.txt, median check at C 256): cycle 96.8 ms = draft 12.0 (SSD idle)
  + check 84.2 (demand SSD busy 42.0, gate spin on misses 42.6), 171 misses. Full-hiding ceiling (every read early, the
  check -> max(GPU ~42, SSD 42)): ~x1.5-1.8. Draft-window ceiling (a perfect next-check predictor: 12 ms x 11.7 GB/s = 88
  reads of ~1.6 MB): 51 % of 171 misses -> ~x1.28 on short checks; on HumanEval/68's 1.45 GB checks ~x1.07.
- Step 2, predictability (sticky/nextcheck_pred.py on BigMoE's 45 GB-route capture ~/BigMoE/data/cap45: 199 checks of the
  same 6 prompts, every row's top-10 and MoE input; router self-check 345 / 400 rows reproduce the captured top-10 set).
  Check p + 1's misses from what the draft window knows (check p), budget B reads per check, candidates not served in
  checks p and p - 1: the router on check p's last row, precision 0.136 / 0.084 / 0.066 at C 160 (measured flags) / 224 /
  256 (LRU replay) at B 88 = recall 1.6-1.7 % of the misses; the rows' mean the same; popularity 0.00-0.02. The perfect
  predictor at B 88: recall 0.13 / 0.21 / 0.25. Realized at C 256: ~6 of ~356 misses per check -> ~+0.5 % cycle. A learned
  model would need ~7x the precision for +10 %; S10b's learned correction gave x1.35 at lead 2.
- The within-check look-ahead on the same capture (sticky/lead1_recall.py): L+1's router on L's MoE input, per-row recall@10
  0.611 / @16 0.738; the rows' union of top-10 names 0.505 of the check's misses at L+1 at 3.49 reads per miss (top-16: 0.670
  at 6.83). The predictive information is one layer ahead, inside the check, where the SSD is busy (R-57's failure: the
  wrong reads compete and evict). That is BigMoE's lane tonight (pre-route on the partial sum, la45 look-ahead +
  keepalive at C 224); a learned lead-1 correction adds little there (S10b: lead 1 +2.4 points).
- Verdict: the next-check (draft-window) predictor = KILL on paper (the window is SSD-idle but the information is not
  there). Not pre-registered. Follow-up only if tonight's pre-route misses its recall line: a learned correction of the L+1
  forecast on cap45 (x, ids on disk).

### TREE3 + STAGED (the Breakthrough Push, Tue 11-03 17:03-17:12; the orchestrator's ask: aligned top-3 tree data for Breakthrough's staged verification, then pricing)
- Data (turns/tree3_job.sh 17:03-17:07, one GPU job: DFlash2 walk MLX 2.03 GB, MTP head 12.7 GB; turns/kw2_job.sh 17:10-17:11,
  1.99 GB): data/tree3_map/<code|ifeval|multi>/<window>.npz + README (the 36 decision-map windows; every anchor A, depths
  0-14 = map rows A + j): DFlash2's top-3 + selector probabilities (teacher-forced), its own chain, the MTP head's top-3 +
  probabilities (aligned: its chain at anchor A starts from the trunk's hidden at A - 1), the 125B's top-3 (the map's
  baseline.json). Breakthrough's own format at HPD-0's anchors: data/d02b_walk_kw_ifeval.npz / _multi.npz (checks
  identical to HPD-0's on 2850 / 2850 and 3294 / 3294; chain k from the tsm top-1 = HPD-0's k on every check).
- Caveat: the every-position walk chunks the drafter's context differently from a live loop: at HPD-0's anchors 1.3 /
  2.2 / 2.5 % of its teacher-forced picks differ from the per-check walk (code / IFEval / multi; k differs on ~1 % of
  checks): near-ties. For an exact replay of a given loop use a walk at that loop's anchors (TL_WALK_ANCHORS).
- Coverage, all depths, the 125B's token in the top-3: DFlash2 code 0.885 / IFEval 0.430 / multi 0.301; MTP head 0.965 /
  0.646 / 0.580; either 0.974 / 0.683 / 0.609 (depth 0: DFlash2 0.995 / 0.875 / 0.699, MTP 0.996 / 0.917 / 0.903). The map's
  baseline top-1 equals the window token on 0.996 / 0.977 / 0.984 of rows.
- STAGED (sticky/staged_sim.py; a map run as the cheap pass, teacher-forced = exact along the true path; each arm its own
  loop; kept per check / exact rows): code shipped cut 8.39 / 10.1 (MTP chain cut 8.54 / 9.9); oracle trees DFlash2 11.41,
  MTP 12.55, union 13.61. Cheap = topk2: beam (a pick at every depth) DFlash2 9.47 / 10.1, MTP 10.06 / 10.5, union 10.33 /
  10.8; cat (one tree pass, the path ends at the first sibling) DFlash2 8.30 / 9.2, MTP 8.73 / 9.7. Cheap = topk5: beam
  union 12.01 / 12.4, MTP 11.18 / 11.6; cat MTP 9.57 / 10.3. Cheap = p1_top_k2: beam union 10.90 / 11.3. Multi: shipped
  mcut 2.50 / 3.2 -> MTP beam topk5 3.52 / 4.5; IFEval mcut 2.37 / 3.1 -> 3.73 / 4.7 (data/staged_sim.txt).
- Reading [G]: the kept gains (+23-43 % on code) need one cheap step per depth (10-15 sequential cheap forwards); one
  cheap tree pass (cat) gets +2-14 %; a static EAGLE-2-style tree sits between. Whether it pays is the cheap row's cost
  (Fable B's ROW_WALL) plus the cheap pass's fixed cost; the exact pass's rows stay at ~10-12. Sent to Breakthrough and
  the orchestrator; the static-tree arm next if Breakthrough wants it.

### LEAD1CORR (prep for WALL1's 'learned L+1 predictor', exploratory, CPU; Tue 11-03 18:4x-18:58; sticky/lead1_corr.py, data/lead1_corr.txt)
- BigMoE's 45 GB capture cap45 (6 prompts, every row's MoE input at every layer + its true top-10), leave-one-prompt-out.
  Plain = layer L+1's router on layer L's MoE input (R-57's forecast). Corrected = a per-layer full-rank ridge map x_L ->
  x_{L+1} (residual form; lambda from an inner prompt), then L+1's router on the forecast (SlotStream / CORR style).
- Per-row recall of L+1's true top-10, layers L+1 = 2..47, pooled over held-out prompts: @10 plain 0.612 -> corrected 0.729
  (+11.7 points; every held-out prompt +11-12); @16 0.739 -> 0.851. The router here is routers.npz (fp16; it reproduces the
  captured top-10 sets on 0.86 of rows), so the gain is relative.
- Context [K]: Fit12's corrected forecast in R-57b-R's replay had recall 0.693 (line 0.75); tonight's pre-route on L's partial
  MoE output has its own line p3 >= 0.85 / p6 >= 0.90 @10. Cost of the map: 46 x 2560^2 (0.6 GB bf16; rank-128 would be ~60 MB,
  untested), ~13 MFLOP per row and layer.
- Use: if the pre-route misses its line, the same ridge correction on the pre-route's input (L's partial output) is the
  follow-up; it needs those vectors captured (cap45 has only x_L). No verdict here (no line).

### S2 (Fable B's pre-registered screen, ROW_WALL.md sec 5; run by MTP Head on the orchestrator's routing, Tue 11-03 19:4x-19:42; sticky/s2_route.py, data/s2_route.txt): PASS by the letter, carried by D2's kept +15 % assumption
- Replay: BigMoE's cap45 through the adopted C 256 per-layer pools exactly as Fable B's row_split.py (MoonShots' laysz45
  loader, real 8-bit router: self-check 0.982; LRU + 30 x popularity; prompts 2-6), rows capped at the live chain's 10.
  The cheap chain over rows 1-9 one cycle early takes the exact routes (no drift: cap45 has no cheap-pass states).
- Misses per check (rows 0-9, MSB + LSB): code 292 (25 checks), math 204 (146); 7-8 % only on row 0 (the base union, which
  the cheap chain never sees). Recall of the exact pass's misses: TOP-2 chain code 0.517 / math 0.457; ALL-RESIDENT 0.932 /
  0.917. Requests per hit 1.00 by construction.
- Model (ROW_WALL sec 3, code rows 9.84, kept 10.54, the replay's misses; base 126 ms -> 83.4 tok/s): TOP-2 (16 rows x 0.6
  ms + dense extra 9.8), no drift: route-only +3.5 %, with D2's kept +15 % +19.0 %; drift 0.1: +1.5 / +16.7 %; drift 0.2:
  -0.4 / +14.6 %. ALL-RESIDENT (2.64 ms rows): route-only -6.9 %, with kept +15 % +7.1 %.
- LINE (Fable B: top-2 recall >= 0.45 with <= 1.3 requests per hit and the model >= +10 % on code PASS; recall < 0.30 or the
  model < +4 % KILL): recall 0.517, model +19.0 % -> PASS by the letter. Reading [G]: the route prediction alone is +3.5 % (KILL
  side by itself) and goes to ~0 at 20 % route drift; the PASS is D2's kept +15 % (the cheap chain pre-verifying the next
  check's start), which this screen does not test, and ROW_WALL's own section 2 puts a cheap verdict's excess at 0.04
  tokens per row on code (16 rows ~ +6 % kept, not +15). Next would be measuring that kept gain and the route drift (a
  top-2 pass's real routes), not the recall.

### M-23b FIRSTMISS-MTP RESULT: KILL (code -1.1 %, chat -0.9 % vs t 0.3; MoonShots' pre-registered row in ~/NewIdeas/QUEUE.md; run by MTP Head Tue 11-03 20:26-20:31, CPU; sticky/m23b.py, data/m23b.txt; written 20:32)
- Harness: MoonShots' stop45.py coupled walks on Breakthrough's 36 capture windows (read-only imports; walk / evaluate with a
  pluggable feature function), SSD-aware tok/s at C 256 + M-17, EXACT gate, 2 folds by window, mu tuned per head on the
  train fold, 2 seeds. Heads A (stop45's), B (+ Fable B's content signals for the drafted token, context through the anchor
  token), C_d (+ the MTP head's flag / top-1 p / p of the draft at depths < d, from tree3_map; d chosen on the train fold =
  1 in both folds of every domain), charged d x 2.2 ms per check.
- Test folds, vs t 0.3 (kept / rows per check; tok/s): code t 0.3 8.00 / 10.62, 86.2; A -1.5 %; B +0.6 %; C1 -1.1 % (MTP cost
  hidden +1.3 %), C2 -4.1 % (+0.5), C3 -5.8 % (+1.1); oracle +12.3 %. IFEval: A +2.3, B +2.2, C1 -1.7 (hidden +3.4), C3 -8.7
  (hidden +6.1); oracle +21.7. Multi: A +2.4, B +2.4, C1 -0.1 (hidden +5.9), C3 -7.9 (hidden +8.6); oracle +17.1.
- LINE (code C_d vs t 0.3: >= +3 % PASS, < +1 % KILL): -1.12 % -> KILL; C_d vs B -1.74 %. Chat (IFEval + multi pooled, reported
  apart): -0.89 % -> KILL; C_d vs B -3.11 %.
- Diagnostic (cf >= 0.99 rows, the MTP flag at depth 0: marks >= 0.30 of first misses, flags <= 0.05 of rows): code 262 rows,
  flags 0.004, 0 first misses (undefined); IFEval flags 0.003, marks 0 / 4; multi flags 0.054, marks 9 / 10. Too few cf >= 0.99
  first misses on the capture walks to judge (any depth: code 4 / 4 marked, IFEval 2 / 9, multi 11 / 14).
- Reading [G]: the MTP head is a real second opinion (cost hidden: chat +3.4-8.6 %), but each MTP step on the critical path
  (2.2 ms) costs more than the rows it saves. On the shipped chat path the MTP chain already runs (its features are free there),
  but its own running-confidence cut already uses them; this harness replays DFlash2's walks, so that case is not measured.
  Prediction (MoonShots): code BETWEEN (+1-3 %), chat BETWEEN-PASS; came KILL / KILL.

### CPUMTP (the orchestrator 20:4x: run the MTP head's draft steps on the CPU during the GPU's gate spins) -- prior art + pre-registration (written before any run, Tue 11-03 20:34)
- Idea: the GPU spins ~42 ms per check on SSD gates (c45l_256), the CPU idles; an MTP step on the CPU would hide its 2.2 ms,
  making M-23b's cost-hidden gain real, or letting the MTP chain draft ahead while the GPU verifies (PEARL-style post-verify).
- Prior art (adapted): Ghidorah (arXiv 2505.23219: hetero-core model parallelism, CPU + GPU on unified memory, Medusa),
  DuoDecoding (arXiv 2503.00784: the draft on the CPU, the target on the GPU), PEARL (parallel draft / verify). TRIED: A7
  (MoonShots 10-23: the MTP drafter on CPU / ANE in parallel, KILL: drafter share 8-11 % of a pass = the ceiling), M-19
  CPUDRAFT (MoonShots 11-02: DFlash2 on the CPU, KILL: MLX's CPU 4-bit matmul is a reference loop, 1.4 s per linear; CPU
  aggregate 35-57 GB/s at 4-8 threads), Leap's WHOLE-CHIP-2 (ANE handoffs 78-147 us per layer, KILL).
- Desk bound first (M-23b, data/m23b.txt): with the MTP cost fully hidden, C_d on code is +0.5-1.3 % vs t 0.3 (below M-23b's
  +3 % line even when free); on chat M-23b's +3.4-8.6 % is against DFlash2-t 0.3, while chat ships on the MTP chain (D-23d
  +24.5 % over DFlash2). So hiding the cost cannot make M-23b pass; a CPU MTP step matters only for drafting ahead.
- Step 1 (now, < 4 GB, ~2 min, sticky/cpu_head_bench.py): the lm_head part of an MTP step (248,320 x 2560; 358 MB at 4-bit,
  1.47 ms on the GPU, HEADBENCH) on the CPU: MLX CPU stream 4-bit / bf16 / fp32, numpy (Accelerate) fp32, and a 32K-row
  subset (a reduced draft vocabulary), alone and beside a GPU load (the GPU lm_head in a loop in another process).
  The head is a floor for the whole step (the MTP layer adds its own weights).
  LINE: PASS (worth Step 2) if the best full-vocab head beside the GPU load is <= 3.0 ms (a whole step can then fit the
  ~5.6 ms per draft a code check leaves: 15 drafts in ~84 ms); KILL if it is > 5.6 ms AND the 32K-subset head is > 3.0 ms;
  BETWEEN otherwise (only a reduced-vocab CPU head could keep pace). Prediction: KILL 80 % (M-19's 35-57 GB/s).
- CPUMTP STEP 1 RESULT: KILL (Tue 11-03 20:35; sticky/cpu_head_bench.py, data/cpu_head_bench.txt; ~4.2 GB, under a share lock).
  The lm_head on the CPU, median per step (+ argmax), alone / beside a GPU process running the 4-bit head in a loop: MLX CPU
  4-bit ~2.6 s (a reference loop, scaled from 8K rows); MLX bf16 55 / 72 ms; fp32 (2.54 GB) MLX 19.9 / 39.2 ms, numpy
  (Accelerate) 20.0 / 38.7 ms; the 32K-row subset fp32 2.7 / 5.1 ms. GPU 4-bit: 1.47 ms (HEADBENCH).
  LINE: best full-vocab head beside the GPU load 38.7 ms (> 5.6) and the subset 5.1 ms (> 3.0) -> KILL (prediction KILL 80 %).
  [K] side number: the CPU alone streams fp32 at ~125-128 GB/s on this M5 Pro (2.54 GB in 20 ms), well above M-19's 35-57
  GB/s through MLX's 4-thread path; beside a bandwidth-bound GPU kernel it falls to ~65 GB/s. A hand-written 4-bit CPU head
  at that rate would still need ~2.9 ms alone / ~5.5 ms beside the GPU for the 358 MB, above the GPU's 1.47 ms: the CPU can
  hide an MTP step only where the GPU idles (gate spins), and with M-23b's cost-hidden ceiling of +0.5-1.3 % on code there
  is nothing to win on the shipped paths. Not pursued; no Step 2.

### MoonShots' WALL 2 sweep (~/MoonShots/research/sweep_1103/WALL2_KEPT.md, Tue 11-03 22:5x): three drafter items taken -- pre-registrations (written before any run, Tue 11-03 23:2x-00:0x)
- Context: the night chain (BigMoE's quiet lock to 04:28, then FLOOR to 13:00) blocks every program, so these are scripted and
  pre-registered tonight and run as small tests from Wed 13:45 (share lock, MEM <= 6 GB). Prior-art pass = MoonShots' sweep
  (45 searches, 25 pages, grep evidence in WALL2_KEPT.md); the TTS paper re-read here (arXiv HTML): Adam, K = 1 step per round
  best, S = 1 best acceptance / S = 5 best throughput, forward KL to the target over all canvas rows with decreasing weights;
  lr, lambda and the parameter set are NOT stated; gains +15-17 % over the first 10K tokens, +183-245 % at 20-30K; no short-
  output numbers.

#### SELGAP: DFlash2's selector on vs off (MoonShots' 10-minute check on DCore, arXiv 2610.24993)
- Question: DCore puts a trained causal tail (DSpark / Domino) at ~+19 % kept over plain DFlash (Qwen3-8B, T 0, B 16). TRIED
  SD-ARtail is KILL on paper because DFlash2 already has a first-order selector. How much does that selector add over plain
  DFlash (each position's own argmax) here?
- Measurement (sticky/selector_gap.py, ~1 min GPU): the HPD-0 k-head walks (data/d02b_walk_kh_{code,ifeval,multi}.npz: the
  drafter's normed hidden at every draft position, the teacher-forced selector picks); unary argmax = the 4-bit lm_head on the
  stored hidden. Both teacher-forced at HPD-0's anchors (exact up to the first miss), kept = accepted + 1 per 16-row check, no
  cut. Validation: the selector picks must sit in the recomputed exact top-16; the own chain's k must equal the walk's.
- LINE (MoonShots): selector gap < +10 % -> a trained tail deserves a screen (proposal: a DSpark-style tail on L1-20);
  >= +10 % -> the selector already holds most of a tail's gain, SD-ARtail's KILL stands. Prediction: code +8-14 %, chat
  +4-9 %, i.e. split (code KILL-side, chat screen-side), 50 / 50 overall.
- Prior art read after writing the line (DCore HTML, Table 1, Qwen3-8B, B 16, greedy, low / high entropy): DFlash 6.78 / 4.03,
  DFlare 7.06, D-PACE 7.46, DCore 7.59 / 4.39 (no tail: target-layer mixing + position-weighted CE, drafting 1.79 ms as
  DFlash), Domino 8.10 / 4.83, DSpark 8.04 / 4.79 (a parallel 5-layer backbone + a token-dependent correction path; DSpark's
  is a Markov head, ~78 M params at 8B; drafting 2.64-2.71 ms). Greedy throughput: DCore best (772.6 vs 755.8 / 752.7 TPS);
  the tails pay only under sampled drafting. Our selector is DSpark's family (first order over the top-16, 0.30 ms after
  R-77 CHAIN), so a NEW tail must beat it, and Domino ~ DSpark says a richer tail adds ~1 % over a first-order one. The
  line stands; a sub-10 % gap would point at our selector's capacity (rank-limited bilinear) more than at a missing tail.
- RESULT (Wed 11-04 05:42-05:43, beside FLOOR's CPU map under a share lock; data/selector_gap.txt): selector gap code +6.7 %
  (kept 9.744 vs 9.131 unary argmax, 634 checks), IFEval +3.0 % (2.447 / 2.376, 2690), multi +2.8 % (2.065 / 2.008, 3114).
  P(depth >= 15) on code 0.276 vs 0.142 (the selector doubles the long runs); at depth 1 ON = OFF (0.940 / 0.938, 0.629 /
  0.639, 0.505 / 0.505: no predecessor drafted yet). Validation: own-chain k equals the walk's on every check; picks outside
  the recomputed top-16 0.13 / 0.26 / 0.18 % (near-ties of the f16-stored hidden). Prediction (code +8-14, chat +4-9) missed
  low on both.
  - VERDICT by the letter: gap < +10 % everywhere -> a trained tail deserves a screen.
  - Reading against DCore's Table 1: on code our first-order selector already holds a DSpark-class gain (+6.7 % vs
    DSpark-over-DCore +5.9 % at low entropy), and Domino adds +0.7 % over DSpark -> no tail screen on code. On chat ours is
    +3 % vs DSpark's +9.1 % at high entropy: the one open item is a stronger selector for chat. Cap by the depth arithmetic:
    a tail acts from depth 2, and chat checks end at depth 0-1 in 36-50 % of cases; doubling the selector's measured effect
    would add +3 % on chat. Not queued: small against the D98 bar; offered to the orchestrator.

#### TOOLSPEC (a): schema tokens among DFlash2's agent first misses (ToolSpec, arXiv 2604.13519)
- Measurement (sticky/toolspec_screen.py, CPU seconds): the 12 held-out agentic turns (data/d02b_walk_d02a_agent.npz, 637
  official-loop checks, kept 4.8 per check); the 125B writes tool calls as XML (<tool_call>, <function=N>, <parameter=P>,
  </parameter>, </function>, </tool_call>). For each first miss: on a tag token at all (upper bound), and FSM-fillable
  (excluding the value-ending token, the next-parameter-or-end choice, call / stop, and NAME tokens).
- Caveat recorded first: each capture starts inside a tool call (gen after '<parameter=...>'), 160 tokens, mostly values.
- LINE (MoonShots): PASS >= 10 % of agent first misses FSM-fillable AND kept bound >= x1.05; KILL tag share < 4 % OR kept
  bound < x1.02; BETWEEN otherwise. Prediction: KILL 75 % (tags are the drafter's easiest tokens; misses sit in values).
- RESULT (Wed 11-04 05:42, CPU seconds; data/toolspec_screen.txt): KILL. 637 checks, kept per check 3.01 on this set (k + 1,
  official loop; the 4.8 above was MoonShots' agent base, not these 12 turns). Tag tokens 12.7 % of the answer tokens.
  First misses 625: on a tag token 45 (7.2 %), FSM-fillable 14 (2.2 %). Kept bound x1.007 (FSM); x1.023 even if every
  tag first miss were filled. KILL by the OR (kept < x1.02); the PASS side (x1.05) is out of reach even at the upper bound.
  Caveat: the '="' and '">' pieces of parameter tags count as NAME (decision) chars here; reclassifying them moves at most
  a few of the 45 and stays under x1.023. Prediction (KILL 75 %) held.

#### TTS1: test-time LoRA on DFlash2, teacher-forced replay (TTS step 1 as MoonShots specified, adapted to our captures)
- Replay (sticky/tts_replay.py, GPU ~20 min, ~4 GB): the live official loop with the t 0.3 cut (kept = min(k, L) + 1) on code
  (24 HumanEval+ windows, Leap's captures, hard CE), IFEval (30), multi (24), agent (12) (D-02a captures: soft CE to the
  125B's top-32). After every check one Adam step on LoRA r 16 / alpha 32 on every Linear of the 5 layers' attention and MLP,
  with L1-20's training loss (soft CE + the selector's CE with the true predecessor) on the VERIFIED rows only (accepted
  drafts + the bonus row: leak-free; TTS also trains on the rows after the first rejection, which a teacher-forced replay does
  not have). Context K / V as the drafting pass cached them (no recompute: live semantics). Reset per window. Arms: lr 0
  (frozen, same code path), 1e-4, 1e-3. Validation: the update's forward reproduces the drafting pass (max |d hid| ~ 0).
- Caveat recorded first: our answers stop at ~840 tokens (agent 160); TTS's published gains are averaged over 10K+ tokens
  and grow with length, so this tests the early regime only. MoonShots' step 0 on Breakthrough's 36 windows: chat kept per
  check flat or rising with position (2.4 -> 2.7, 1.9 -> 2.1), code falling (12.7 -> 5.7: HumanEval's echoed signature and
  docstring early). A KILL here kills TTS for today's suites, not for long agent sessions (those need long 125B captures).
- LINE: PASS if at one lr chat (IFEval or multi) or agent kept per check (cut) >= +6 % vs frozen AND code >= -0.5 % at that
  lr (then: price the update (ms per step x S) and an S = 5 arm, plus a per-session agent arm); KILL if < +2 % on every set
  at both lrs; BETWEEN otherwise. Prediction: KILL 70 %, BETWEEN 25 %, PASS 5 %.
- TTS1 RESULT: KILL (Wed 11-04 09:29-09:48 shared, pulled forward by the orchestrator; data/tts_replay.txt + _<set>.npz;
  written 09:5x). Kept per check (cut), frozen -> lr 1e-4 / lr 1e-3: code 9.248 -> 9.238 (-0.1 %) / 1.752 (-81.1 %);
  IFEval 2.342 -> 2.297 (-1.9 %) / 1.153 (-50.8 %); multi 1.988 -> 1.954 (-1.7 %) / 1.059 (-46.7 %); agent 2.876 -> 2.790
  (-3.0 %) / 1.173 (-59.2 %). Every set falls at both lrs. Validation max |d hid| 0 in every arm. Update 25-34 ms median
  per step. Prediction KILL 70 % -> held.
  - Smoke first (09:01-09:03, 2 multi windows) found a loader bug: d02a's tf_* and agent windows hold one top-32 row per
    position from off, and the loader read them as rows from gen (misaligned, the target's top-1 matched the answer
    0.00-0.09). Fixed (94dd7eb, 3627f9d): aligned, it matches 0.936-0.994 on IFEval / multi and 0.899-1.0 on agent (the
    check's threshold is now 0.80).
  - Reading: at lr 1e-3 every set collapses within a window (the drafter forgets faster than it adapts). At 1e-4 the small
    losses are largest where a window has the most updates (agent -3.0 %). No early-regime gain; the caveat above stands
    (answers <= ~840 tokens, agent 160), so this kills TTS for today's suites, not for long agent sessions.
  - Memory: peak footprint 17.5 GB (max RSS 4.0 GB) against "MEM 5 GB" in the lock, sized from the smoke's 4.4 GB. Likely
    MLX's buffer cache over the long agent windows and the 1e-3 arms (up to 6.4K checks); not checked. Swap stayed flat
    (11.4 -> 10.9 GB). A re-run caps the cache (mx.set_cache_limit) or declares ~18 GB.

### PR160 read check + SAMELAYER-T (the 05:37 wake's pre-agreed follow-up to tonight's pre-route; written before any run, Wed 11-04 05:5x)
- pr_160 (BigMoE's night, C 160, ~/BigMoE/data/r57/pr_160_preroute.txt): per-row recall@10, all domains: r57 0.634, full
  0.565, p3 0.565, p6 0.566 -> KILL by its line (PASS p3 >= 0.85 AND p6 >= 0.90; KILL p3 < 0.75). Code 0.553 / 0.475,
  chat 0.667 / 0.602 (r57 / full).
- READ CHECK (code reading, no run): 'full', 'p3' and 'p6' score L+1's router on `nxt.attn_hyper_connection(h)[0]`
  (~/BigMoE/data/r57/snap/r57_screen_s160.py:521 and :526; scripts/r57_screen.py the same) = L+1's ATTENTION-side
  hyper-connection read. In qwen4_exp every block's read has its own hc_norm weight and its own learned read mix
  (ref/qwen4_exp.py:1243-1290; there are no input / post-attention layernorms), and the router sees the MLP-side read
  `mlp_hyper_connection(hyper, pending=(attn_out, inject))`. The pre-attention forecast of L+1's router input is the
  MLP-side read with no attention write: `nxt.mlp_hyper_connection(h)[0]` (pending=None).
  - Two signs the measured read is not the router's: 'full' holds strictly more information than r57 (L's MoE output added)
    yet scores 7 points lower, and NewIdeas' L2-08 measured the MLP-side read of the pre-attention streams (QUEUE L2-08,
    ~/NewIdeas/scripts/tf_samelayer125.py:62, data/samelayer/results.txt) at per-row recall@10 0.735 code / 0.820
    IFEval + multi, against pr_160's 'full' 0.475 / 0.602.
  - Prediction for a re-run with the MLP-side read (BigMoE's code and verdict, not mine): full code 0.70-0.76, chat
    0.78-0.84; p3 within 0.01 of full -> 'all' ~0.76-0.81 = BETWEEN by the letter (code alone may stay < 0.75).
- SAMELAYER-T (MoonShots' WALL1_SSD item 3 = WALL2_KEPT item 4, arXiv 2511.10676 "Pre-Attention Expert Prediction":
  linear maps on the pre-attention activations, 93-97.6 % on 15-30B MoEs): per layer L, a trained forecast router
  W_L = R_L + dW_L (2560 -> 512) on L's MLP-side read of the pre-attention streams (the corrected 'full' input), ridge on
  L's true router logits, residual form (lambda from an inner prompt; dW = 0 is the untrained forecast); also scored on the
  p3 input (L-1's partial output written), its live timing. Leave-one-prompt-out, per-row recall@10 of L's true top-10.
  - Run cost: none over the untrained pre-route (W_L replaces the router weight in the same GEMV; 48 x 2.6 MB bf16 =
    126 MB resident).
  - Data: a new teacher-forced capture (the 125B exact streaming through NewIdeas' tf_samelayer125 path, C 51, ~13 GB,
    ~15-20 min): per decode row and layer the MLP-side pre-attention read (2560) and L's true MoE input (2560) + top-10;
    tf_code + tf_ifeval + tf_multi, every 8 (~5K rows). Or BigMoE's corrected pre-route re-run with those vectors saved.
  - LINE (MoonShots' item-3 screen, unchanged): PASS recall@10 >= 0.85 code AND >= 0.90 chat; KILL < 0.78 code (less
    than +0.05 over the untrained read); BETWEEN otherwise. Then: exposure_replay exposed3 with the forecast window
    0.45 ms (MoonShots' [G]: D2's exposure ~12.4 -> ~6-9 ms per check).
  - Prediction (LEAD1CORR's ridge gave +11.7 points at lead 1; MoEPrefetch's rank-128 correction +12 on the D1 tap):
    code 0.80-0.86, chat 0.87-0.92 -> PASS 35 %, BETWEEN 45 %, KILL 20 %.
  - When: not tonight (main at 39 % against the orchestrator's <= 45 % at 13:45; night = scheduled jobs only). The capture
    is one > 5 GB GPU job: Wed day in a free slot, else Wed night, routed by the orchestrator.
  - UPDATE 05:5x: BigMoE confirmed the read bug, withdrew the KILL (~/BigMoE 958ea4b: full = L+1's MLP-side read at its start,
    PLE included; p3 / p6 on the MLP side; fullA = the old read as a control) and dumps the data in its re-run (WED_TESTS 3b,
    R57_PR_DUMP=1: data/r57/pr_dump/pr2_160_NN.npz, 18 prompts, ~124K row-layers, decode verify rows incl. rejected drafts;
    X = full's input, Y = the true MoE input, f32). No capture of mine needed.
  - Script (sticky/samelayer_t.py, CPU): ridge on the logit residual (Y - X) R_L^T with R_L the real 8-bit router, folded
    into the router at run time; leave-one-prompt-out, lambda from two inner prompts. Check before the data (cap45 as a fake
    dump, X = x_{L-1}, Y = x_L = LEAD1CORR's lead-1 setup, 16 layers; sticky/samelayer_t_fake.py): untrained 0.622 ->
    trained 0.742 (+12.0 points) vs LEAD1CORR's 0.612 -> 0.729 (+11.7) = the pipeline reproduces. Lambda grid widened to
    0.1-10 x mean eigenvalue after the check (its picks sat at 1.0).
  - UPDATE 07:3x: the data landed (BigMoE's 3b done 07:31: 18 files, 2.4 GB; 12 HumanEval = code, 3 IFEval + 3 multi =
    chat; 123,845 row-layers, layers 1..47). BigMoE's corrected pre-route (~/BigMoE/data/r57/pr2_160_preroute.txt): full
    0.712 code / 0.827 chat / 0.794 all, p3 0.789 all -> BETWEEN. The re-run prediction above (full code 0.70-0.76, chat
    0.78-0.84, all ~0.76-0.81 BETWEEN) held on all three. For SAMELAYER-T this puts the untrained baseline at 0.712 code
    (the low end of my guess): PASS needs +13.8 points on code (and +7.3 on chat); staying out of KILL needs +6.8 on code.
    The prediction and the LINE stay as registered. Before the run, the script's memory was cut: the per-prompt Grams are
    rebuilt on the fly instead of stored (peak ~3 GB, under the 5 GB day line), with an independent check of the
    leave-one-out totals at layer 1. It runs at the 13:47 wake under a CPU share lock, since BigMoE's morning block holds
    quiet to ~12:50 and Fit12 has 13:05-13:40.
  - RESULT: BETWEEN (Wed 11-04 09:18-09:29 shared, CPU 8 threads, 648 s, max RSS 3.0 GB; data/samelayer_t.txt; written
    09:5x). Per-row recall@10, leave-one-prompt-out, untrained -> trained: code (12 HumanEval) 0.706 -> 0.823 (+11.7
    points), chat 0.821 -> 0.851 (+3.1; IFEval 0.809 -> 0.843, multi 0.831 -> 0.859), all 0.748 -> 0.833 (+8.5). LINE
    (PASS >= 0.85 code AND >= 0.90 chat; KILL < 0.78 code) -> BETWEEN. Prediction: code 0.80-0.86 held, chat 0.87-0.92
    missed low; the call (BETWEEN 45 %) held.
    - Checks: untrained vs BigMoE's corrected 'full' (0.712 code / 0.827 chat): -0.006 / -0.006 (the same rows, pooled
      per row here). Leave-one-out totals vs a direct computation at layer 1: max rel err 1.5e-15 / 5.4e-15.
    - Bands (layers 1-7 / 8-39 / 40-47, trained): code 0.777 / 0.825 / 0.854, chat 0.779 / 0.857 / 0.892. The late layers
      reach the code line; the first 7 stay lowest. Lambda picks 1 x524, 3 x239 (grid edges 0.1 x2, 10 x14).
    - Reading: the learned map removes ~40 % of code's untrained misses (0.294 -> 0.177) but ~17 % of chat's (0.179 ->
      0.149). Chat is 6 of 18 prompts (37 % of rows), so the fold fits are code-heavy; a chat-only fit would have 5
      training prompts per fold. Not scored: the p3 input (the dump holds full's input only; BigMoE's table puts p3 within
      0.005 of full) and the union per 16-row block (the prefetch's real unit).
    - Next (not queued: BETWEEN has no pre-registered follow-up): pricing is cheap, since the trained router is the same
      GEMV (47 x 2.6 MB bf16). Replay the union-per-block misses with the trained forecast in exposure_replay (MoonShots'
      exposed3, 0.45 ms window). The orchestrator's and MoonShots' call.
    - UPDATE 10:0x, MoonShots priced it (exploratory, ~/MoonShots/scripts/price_samelayer.py: untrained -> trained recall per
      layer on M-25b's held-out demand at C 256): code, the demand the forecast fails to name 13.7 -> 8.1 MB per kept
      token (-40 %), ~-2.4 ms exposed per check as a bound; chat -18 to -19 %, ~-0.1 ms per check. These are my per-row
      miss ratios (0.294 -> 0.177, 0.179 -> 0.149) carried to the bytes, so they are upper bounds: the prefetch's unit is the
      union per check (per-row estimates have overstated before, e.g. the reuse map x0.96 vs -40 %). The next cheap step,
      if the orchestrator wants it: union-per-check recall of the trained forecast from the dump's chk field (CPU, ~11 min).

## DRAFT17: a draft for the bonus row, the block pass at 17 positions (NewIdeas 13:44 for build #6's gated last row; the orchestrator routed it 13:5x) -- pre-registration (written before any run, Wed 11-04 13:58)
- Question: what does one more draft position per check cost in ms in the engine's drafter pass? (block 17 = the anchor +
  16 masks, so the check's last row, the bonus row, has a draft to gate; on code 32 % of segments lack one today.)
- What the code does today: BigMoE's 45 GB default (r57_screen.py, R57_D45 + R57_COMBINE -> K83_EXACT) runs propose() =
  hidden_states(block, ctx, cache, 1) -> head2s (the 4-bit LM head, two-stage top-16) -> hidden_projection -> Leap's
  chain_fused, with Leap's draft_fused installed (fused conv, memoized mask, and R-78's 16-row "forms": the MLP's gate / up /
  down and q_proj keep W transposed; at any other row count they take x @ contiguous(Wt.T).T, a copy of each weight per call).
- Harness: sticky/draft17_time.py copies that path verbatim. DFlash2 L1-20 (bf16, loaded over incoai's base as the
  engine does), the target's 4-bit embed / LM head (shard 16). Per arm its own drafter cache, prefilled with CTX rows
  (600 and 1,900), then +10 new context rows per call (a check's accepted tokens); the mask memo is cleared before each call
  (the engine builds it once per check); arms interleaved, order reversed every other rep; 5 warm + 40 timed calls; median.
- Arms: R16 (today); R17 and R32 as the code stands (forms fall back); R17g and R32g with the forms opened to that row count
  (draft_fused._rows16 patched). Masking the extra position from rows 0-15 (so drafts 1-15 stay as today) changes only the
  memoized mask: same cost, not timed. The extra draft's accuracy is not measured here.
- Read (a cost answer, no PASS / KILL): R17 and R17g minus R16, in ms. Expected: R17g +0.1-0.5 ms; R17 as is +3-6 ms (the
  per-call weight copies, ~0.7 GB written per pass). Sanity: R17 vs R17g give the same drafts on most calls.
- Lock: quiet, MEM 3 GB, the routed slot after BigMoE's agent leave-one-out (start by 14:30, ~3 min); footprint by
  /usr/bin/time -l.

### DRAFT17 RESULT: the bonus row's draft costs ~0 ms once the 16-row forms accept 17 rows; +8.6 ms as the code stands (Wed 11-04 14:03; quiet lock, 9 s run; data/draft17_time.txt, .log)
- The engine's K83 drafter path, L1-20 bf16, median of 40 timed calls per arm, context 600 / 1,900 rows:
  | arm | positions | ms | vs R16 |
  |---|---|---|---|
  | R16 (today) | 16 | 11.11 / 12.23 | - |
  | R17 as the code stands | 17 | 19.64 / 20.88 | +8.53 / +8.64 |
  | R17g (forms open to 17) | 17 | 10.94 / 12.37 | -0.16 / +0.14 |
  | R32 as the code stands | 32 | 20.22 / 21.24 | +9.11 / +9.01 |
  | R32g (forms open to 32) | 32 | 11.07 / 12.30 | -0.04 / +0.07 |
  p10-p90 spans are ~1 ms. R17 / R17g and R32 / R32g drafted the same tokens on 100 % of calls. R16 matches the engine's
  ~10-12 ms pass (K-83: 10.2 ms at C 160).
- Answer for #6: a 17th position costs nothing measurable (within +-0.2 ms) once draft_fused's R-78 forms accept the
  block's row count (its _rows16 test). As the code stands it costs +8.6 ms per check: every row count other than 16 makes
  a fresh copy of the MLP and q_proj weights on each call. Prediction was R17g +0.1-0.5 / R17 +3-6 ms; the fallback is worse
  than predicted.
- Also found: 32-row blocks pay the same +9 ms today under K83, which has been in the 45 GB default since 0365ada.
  r57_screen.py uses R57_ROWS=32 where the copy runs (MATCH 2.5; combo_tue's cmb32). The same one-line change in
  ~/Leap/scripts/draft_fused.py (Leap's file) removes it.
- Masking the extra position from rows 0-15 changes only the memoized mask (same cost). The extra draft's accuracy is not
  measured.
- Memory: peak footprint 3.38 GB (/usr/bin/time -l), mlx peak 2.48 GB, max RSS 2.46 GB. That is 0.4 GB over the declared
  MEM 3 GB (the cache limit was 256 MB).
- UPDATE 14:1x (Leap's lounge notes): Leap's 05fafe7 makes the forms' default 16..64 rows (R-78b: both forms are bitwise at
  every row count >= 16 and differ at 2-15), so the +8.6 / +9 ms fallback is gone from the code and 17 / 32-row drafts stay
  bitwise. NewIdeas (14:08): with the bonus row gated, code's bytes go x0.665-0.706 at 24 GB (35.7 -> 25.2 MB per kept
  token today) and x0.742 at 12 GB.

## SAMELAYER-T FIT: the final router weights for lever 2 (BigMoE's ask 14:5x, build #2) -- written at launch (14:57), before results
- Not a test: the deliverable. Ridge on all 18 pr2_160 prompts (no hold-out), lambda per layer from the 18-fold LOPO grid
  (each of 0.1 / 0.3 / 1 / 3 / 10 x G's mean eigenvalue scored on every held-out prompt, pooled per-row recall@10; the best is
  refit on all prompts). sticky/samelayer_t_fit.py -> data/samelayer_t_W.npz (bf16 bits, 47 x 512 x 2560, + the bias)
  and data/samelayer_t_fit.txt.
- Folded form, the router's own GEMV: logits_L = X W_L^T + b_L, W_L = R_L + dW_L^T, b_L = e0 - mu dW_L; top-10. The bias
  is part of the forecast (a per-expert offset moves the top-10).
- Input: X = layer L's MLP-side read of its pre-attention stream, r57_screen.py:582 (pre = self.mlp_hyper_connection(hidden)[0]),
  after the PLE add at :578 ('full'). The dump has no p3 input, so p3 recall can't be scored from it.
- Reported: LOPO recall at the chosen lambdas (slightly optimistic vs SAMELAYER-T's nested 0.823 / 0.851, since the pick
  sees the outer folds); in-sample recall with the fp64 W, the saved bf16 W, and R exact + dW in bf16 (the rounding check).
### SAMELAYER-T FIT RESULT (Wed 11-04 14:57-15:07, CPU, share lock; data/samelayer_t_fit.txt, .log; weights data/samelayer_t_W.npz, 118 MB, not in git)
- LOPO recall at the chosen lambdas: code 0.706 -> 0.824, chat 0.821 -> 0.851 (the nested pick gave 0.823 / 0.851: the
  outer-fold pick adds nothing). Lambda 1 x mean eigenvalue on 41 of 47 layers (0.3 x3, 3 x3).
- In-sample (all 18 prompts, optimistic): code 0.880, chat 0.926. The saved bf16 W gives the same to 4 decimals (0.8802 /
  0.9258), and so does R exact + dW in bf16: rounding costs nothing.
- Footprint 3.57 GB (MEM 4 GB declared), 593 s. Sent to BigMoE with the input (r57_screen.py:582 after :578) and the bias.

> (a quote from the project owner, kept private)
- Data: the TOP16 walks (data/d02b_walk_kw_{code,ifeval,multi}.npz: L1-20 at HPD-0's anchors, block 16; per check the 16
  candidates and the teacher-forced selector softmax at each of the 15 depths; the 125B's tokens).
- Simulated at the same anchors (no anchor shift): kept per check = 1 + the leading run of right picks, for: the official pick
  (top-1 by the selector; self-check against the walk's own kept) and oracle picks that are right whenever the 125B's token is
  among the top-2 / top-3 / top-16 candidates. Also: per depth, the share of first misses where the right token was rank 2 / 3.
- Read: the ceiling for any re-ranker over DFlash2's candidates. Prior on the same anchors: E1 COVERAGE's DFlash2 + MTP-chain
  confidence pick, code 9.54 -> 11.70 (oracle 12.73), REPORT :2452.
- LINE (does a live re-ranker deserve a build?): top-3 oracle >= +20 % kept on code -> yes (the MTP-assisted pick first);
  < +10 % -> the pick is not the lever; between: only with a cheap re-ranker.
- Prediction: top-3 oracle code +25-35 % (KWALL: 0.888 top-3 at mid rows vs ~0.75 top-1).
### RERANK-0 RESULT: PASS by its line -- a perfect top-3 pick is worth x1.21 on code, the whole candidate set x1.29 (Wed 11-04 15:10; sticky/rerank0.py, data/rerank0.txt; CPU, seconds)
- Code (24 windows, 666 checks): the official pick re-simulated 9.57 = the walk's 9.57 on every check (self-check OK).
  Oracle kept per check: top-2 11.00 (x1.149), top-3 11.57 (x1.209), top-16 12.37 (x1.292). At the official first miss the
  125B's token is rank 2 in 54.7 %, rank 3 in 14.9 %, 4-16 in 19.6 %, not a candidate in 10.9 %.
- Chat: ifeval 2.45 -> top-3 3.31 (x1.351) / top-16 4.19; multi 2.06 -> 2.62 (x1.273) / 3.28. Chat's misses are often not
  candidates at all (30 / 44 %): chat's wall is the candidates, code's is the pick.
- LINE (top-3 >= +20 % code -> a realistic re-ranker deserves a test): x1.209 -> PASS, at the edge. Prediction was +25-35 %.
- What it means for 300: even a perfect pick over all 16 candidates keeps 12.4 per 16-row check; 14 needs a better drafter
  or longer checks. [CORRECTED 15:3x: the next sentences quoted the WITHDRAWN off-by-one MTPPICK numbers (11-03 ALIGNMENT
  FIX), not E1 COVERAGE; aligned, the MTP-assisted pick keeps +7 % (MTPPICK aligned) / +9.8 % (MTPSCREEN, full head 10.51).]
  The only realistic re-ranker measured so far is the target's MTP chain: E1 COVERAGE's DFlash2 + MTP
  confidence pick, 9.54 -> 11.70 (x1.226, above the top-3 oracle because the chain proposes its own tokens). It costs 15
  sequential MTP steps (2.2 ms each today, ~33 ms). At a 90 ms cycle it pays only below ~1.4 ms per step; at a 50 ms cycle
  only below ~0.75 ms, or when the chain runs beside the block pass instead of after it. A learned picker over DFlash2's own
  state gave +1.2 % before (L1-21 / SD-sel).

## DA-0: the DRAFT-AHEAD replay -- how often a block pre-drafted during the check is the one needed (the orchestrator's item 2, BREAK_THE_WALL 4.2) -- pre-registration (written before any run, Wed 11-04 15:2x)
- On the engine's cut chain (t 0.3) over the TOP16 walks (L1-20, HPD-0's anchors): the check ends at the segment end (all
  checked drafts right) or at its first miss. Pre-drafted prefixes: the segment end + the two lowest-cf depths in the
  segment. The next block's anchor is unknown before the check ends, so the pre-draft guesses it: the own draft past the cut
  (segment end; none exists at a full block today) or the selector's 2nd-ranked candidate (at a miss). Hit = the check ends
  at a pre-drafted prefix AND the guess is right; then the pre-drafted block is exactly the serial one.
- Read: the hit rate on code. With D7b's 6.3 ms drafter, a hit saves 6.3 ms of serial draft; the 3 pre-drafts cost GPU time
  inside the check, which pays only where the GPU waits (after layer 45, the last tap, or on the SSD).
- LINE: code hit >= 50 % -> build the scheduler patch; < 30 % -> DRAFT-AHEAD can't pay at its design's 3 prefixes.
- Prediction: ends at a prefix ~70 % (the design's [K]), anchor guesses ~75 % / ~55 % -> hit ~45-50 % on code.
### DA-0 RESULT: BETWEEN -- the pre-drafted block is usable on 34 % of code checks (43 % with DRAFT17's extra draft) (Wed 11-04 15:14; sticky/da_replay.py, data/da_replay.txt; CPU)
- Code (666 checks, cut chain kept 9.35): the check ends at a pre-drafted prefix on 89.8 % (segment end 49.1 %, a full
  block 24.3 % of them; one of the two lowest-cf depths 40.7 %), but the anchor guess is often wrong: the own draft past the
  cut is right 34.5 % (the cut sits where confidence drops), the 2nd-ranked candidate at a lowest-cf miss 63.1 %.
  HIT 34.2 %; 42.6 % if DRAFT17's extra draft fixes full blocks at the same rate (likely higher there: unmeasured).
- Chat: ifeval 25.6 %, multi 17.8 %.
- LINE (>= 50 % build, < 30 % can't pay): BETWEEN. Prediction was ~45-50 %: the segment-end guess is the miss.
- Worth: with D7b's 6.3 ms drafter, a 34-43 % hit saves 2.1-2.7 ms per check, and only if the pre-drafts run in GPU-idle
  time. At 45 GB FAST, where the GPU is the bottleneck, they don't. Pre-drafting 2 anchors per prefix raises the hit rate
  but costs more drafts. The orchestrator asked for the patch anyway (by day, offline same-drafts check).

## DA-1: DRAFT-AHEAD's offline same-drafts check (the orchestrator's "minimal test", by day) -- pre-registration (written before any run, Wed 11-04 15:20)
- sticky/draft_ahead.py (plan / launch on cloned drafter caches / resolve; PATCH notes for r57_screen.py) and sticky/da_check.py:
  the engine's K83 drafter path (L1-20 bf16, forms + chain_fused, head2s K 16) replayed on the 24 code windows of
  ~/Leap/data/dfl2d_code in the engine's order (propose -> cut t 0.3 -> check against the 125B's tokens -> commit -> next
  propose on the taps of rows 0..k). Prefixes: end / low1 / low2 (DA-0's three), + end2 (the 2nd-ranked guess past the cut).
- LINE: identical drafts on 100 % of hits (path, cf and the drafter cache, bit for bit) and the serial cache untouched on 100 %
  of checks -> the patch is exact and goes to BigMoE; any difference -> fix before handing over. Hit rate reported beside
  (DA-0's replay 34.2 %), with the serial propose's ms.
### DA-1 RESULT: PASS -- the patch is exact; worth it only where the check leaves ~20-33 ms of GPU idle (Wed 11-04 15:20; sticky/draft_ahead.py, sticky/da_check.py, data/da_check.txt; drafter only, 38 s)
- 689 checks over 24 code windows (cut-chain kept 9.25), 3.25 pre-drafts per check. HIT 35.4 % with the 3 prefixes (DA-0's
  replay 34.2 %), 40.9 % adding the 2nd-ranked guess past the cut.
- Same drafts: 281 of 281 hits identical (path, cf and the drafter cache, bit for bit); the serial cache untouched by the
  pre-drafts on 689 of 689 checks. LINE (100 % / 100 %): PASS.
- Time: serial propose 11.7 ms, each pre-draft 10.0 ms. If the pre-drafts cost nothing, the drafter's critical path falls
  11.7 -> 7.6 ms (3 prefixes) / 6.9 (4). But they are ~33 ms of GPU per check (3.25 x 10; ~20 with D7b), so they pay only
  where the check leaves that much GPU idle: the SSD-bound 24 / 12 GB lines, not 45 GB FAST, where the GPU is the bottleneck.
  Batching the prefixes into one pass would cut the cost, but it changes the GEMM shapes: the drafts would no longer be
  bitwise the serial ones.
- Footprint 2.85 GB. PATCH notes for r57_screen.py are in draft_ahead.py (R57_DA=1; safe form after the eval, overlap form
  on a second stream from layer 45's taps).

## MTPSCREEN: does the MTP-assisted pick survive a screened MTP head? (Leap's question, 15:3x: the 4-bit lm_head is ~1.7 of the ~2.2 ms MTP step) -- pre-registration (written before any run, Wed 11-04 15:26)
- Why: RERANK-0 / TOP3-USE: the MTP-assisted pick keeps 11.70 vs 9.54 on code (x1.23, E1 COVERAGE) but its 15 sequential
  steps (~2.2 ms each) make it x0.92 net. Leap measured a screened head at 0.24 ms (32K rows) / 0.02 ms (~2K rows): a step of
  ~0.75 / ~0.5 ms. Then the pick costs ~7.5-11 ms per check (15 steps; fewer with the cut) and could net ~+10-13 % at a 90 ms
  cycle, if the screen keeps its quality.
- sticky/mtpscreen.py: d24_mtptop3's teacher-forced b5 chain on the 24 held-out HumanEval+ windows, at the TOP16 walk's HPD-0
  anchors (MTP anchor s = A - 1); per depth the MTP's top-1 and p under heads full / a (FR-Spec top-32K, ~/Leap/data/
  frspec_rank.npy, + session tokens) / b (DFlash2's 16 candidates + session) / b0 (the 16 alone); E1's pick (the more confident
  drafter per depth), plus a threshold sweep (optimistic) and the union.
- Self-checks: DFlash2 alone ~9.54-9.57, MTP alone (full) ~10.13, pick (full) ~11.70, union ~12.73 (E1's numbers).
- LINE: head b keeps >= 95 % of full's pick gain (kept_b - 9.54 >= 0.95 x (kept_full - 9.54)) -> PASS: the pick at ~0.5 ms per step
  is worth building; head a >= 95 % -> PASS at ~0.75 ms; both < 80 % -> KILL the screened pick.
- Prediction: b ~85-95 % (the MTP's wins at a DFlash2 miss are mostly the 125B's token, which is among the 16 candidates
  89 % of the time at first misses, RERANK-0); a ~99 %.
- Needs ~7-11 GB of GPU (the D-24 harness), ~15-20 min: by day only in the one > 5 GB slot, else tonight.

### D17-1 (information for BigMoE's R57_D17 + QOM, Wed 11-04 15:30; sticky/d17_check.py, data/d17_check.txt; drafter only, 14 s, 2.6 GB)
- The 17-position drafter pass vs today's 16 on the K83 path (forms 16..64, Leap 05fafe7), 689 code checks in the engine
  order: drafts 1-15 identical tokens on 83.3 % of checks (96.48 % per draft), cf never bit-equal (the extra mask position
  enters the bidirectional block attention); the t 0.3 cut lands at the same depth on 89.3 %. 9.67 vs 9.76 ms: no extra cost.
- Full blocks (15 checked drafts, all right): 23.7 % of checks; the 16th draft is the 125B's token on 79.8 % of them (the
  quality of QOM's last-row gate there, and DA-1's full-block anchor guess: DA's hit would rise ~34 -> ~43-44 %).
- Patch steps for r57_screen.py: sticky/draft17.py (PATCH17, split17, last_row_draft). Not measured: kept under the
  17-position drafts; the masked variant (drafts 1-15 as today).
### MTPSCREEN RESULT: PASS by its line (a screened head keeps the pick: b 10.53, a 10.51 = full 10.51), but the pick is x1.10, not x1.23, and the step's layer alone is 1.28 ms (Wed 11-04 15:30-15:31; sticky/mtpscreen.py, data/mtpscreen.txt, mtpscreen_code.npz; 28 s, footprint 10.1 GB)
- 666 code checks at HPD-0's anchors (MTP anchor s = A - 1, the 11-03 ALIGNMENT FIX's): DFlash2 alone 9.57 (= the walk);
  MTP b5 alone 9.97; union oracle 11.35. The pick (the more confident drafter per depth): full head 10.51 (x1.098); head a
  (FR-Spec 32K + session) 10.51; head b (DFlash2's 16 candidates + session) 10.53 (x1.101); b0 (the 16 alone) 10.40 (x1.087).
  The best threshold shift is 0 for every head: the subset's inflated p does not hurt the pick here.
- Self-check vs the pre-registration: it expected E1 COVERAGE's 11.70 / 10.13 / 12.73. Those were the WITHDRAWN off-by-one
  MTPPICK numbers (11-03), misattributed to E1 in RERANK-0's text today (corrected above). The aligned MTPPICK said +7 %;
  this run says +9.8 %.
- LINE (b keeps >= 95 % of full's gain -> PASS): (10.53 - 9.57) / (10.51 - 9.57) = 102 % -> PASS; a 100 %.
- The 1-row step (mtp_forward + argmax / softmax max + one sync, 400-pair cache, shared GPU): full head 2.571 ms, Leap's
  screened 32K head 1.353 ms, a 16-column stand-in (the layer + mixer alone) 1.280 ms. The head is 1.29 ms of it (Leap: 1.257).
- Worth: the pick adds +0.94 kept per check (x1.10) for 15 sequential steps. At 1.35 ms per screened step that is ~20 ms, so
  x1.10 x 90 / 110 = x0.90 at a 90 ms cycle: a loss. It breaks even near 0.6 ms per step (the layer ~2x faster than today),
  and pays only below that. The screened head does halve every MTP step: that helps the chat switch's MTP chain (R57_CHATSW,
  2.2-2.6 ms per step today); the drafts change only where the full top-1 is outside the set (a: 0.3 % of depths).

### D7B-1 (the orchestrator's D7b discrepancy, Wed 11-04 15:36-15:41; sticky/d7b_check.py, data/d7b_check.txt, d7b_check_g32.log; drafter only, 3.3 GB)
- Question: BigMoE's live all-in arms (C 128, 12 HumanEval prompts) saw R57_D7B=code cut code kept 10.35 -> 9.91 (-4.3 %);
  Leap's R-85b replay said +0.00 %. Which form does the engine apply, and does anything else differ?
- The engine applies arm E (d7b_live.apply: the 5 layers' Linears 4-bit RTN g64; fc, selector and head bf16). The bf16
  drafter gets R-78's forms, the 4-bit one doesn't (forms need nn.Linear); both get draft_fused's conv kernel, mask memo and
  chain_fused. This replay wires it the same way.
- Engine-order walk with the t 0.3 cut, paired per window: code (24 held-out HumanEval+) 9.25 -> 9.13 (x0.987, 90 % CI over
  windows x0.965-x1.007); math (60 GSM8K) 6.67 -> 6.52 (x0.978, x0.967-x0.988). g32 instead of g64: code x0.986
  (x0.974-x0.999), math x0.989 (halves math's loss, not code's).
- Reading: the code cost is ~-1.3 % here (CI to -3.5 %); R-85b's +0.00 % was optimistic, and BigMoE's -4.3 % on 12 prompts
  is likely the same effect plus sampling. Net at a 90 ms cycle: -4.1 ms (~4.5 %) vs -1.3 .. -4.3 % kept, i.e. ~+0 .. +3 %.
  The trade improves as the cycle shrinks (at 50 ms, -4.1 ms is 8 %). A 4-bit-aware fine-tune (~1 h GPU, ~11 GB) is
  what would remove the kept loss.

> (a quote from the project owner, kept private)
- Why: the plan's probes decide whether a cloud drafter build can pass ~12 kept per 16-row check (Kamen's 95-96 % per
  token). Data at DFlash2's size adds +0.13 kept per doubling (D-09 + E2/D2). Capacity is the untested input.
- Method (published: block expansion, LLaMA Pro, Wu et al. 2024): incoai's DFlash2 grows from 5 to 10 layers. After each
  layer comes a copy whose o_proj and mlp.down_proj are zero, so the start computes exactly incoai's function.
  - sticky/pcap_expand.py self-test: max |logit diff| 0.000, PASS.
  - sticky/pcap_train.py = sticky/d09_train.py (= Leap's dfl2c_train.py, the L1-20 recipe) + the expansion + a second
    AdamW for the new layers (MultiOptimizer).
  - --check: the training forward matches the official 10-layer path (argmax agreement 0.996, PASS, L1-20's own check).
- Run: L1-20's args (data ~/Leap/data/l120, 2 epochs, lr 2e-5, pmask 0.5, block 32, warm-up + cosine to 10 %, the
  script's seeds), with three changes:
  1. The new layers' peak lr is 2e-4 (10x), since their zeroed outputs must grow within ~200 steps.
  2. Groups of 32 anchors, 8 per step (L1-20: 64 x 4). Same ~256 anchors per step, same per-anchor loss, ~201 steps;
     the activation memory halves.
  3. --minutes 75: 10 layers take ~2x per step, and L1-20's 50-minute budget would cut the 201 steps.
- Score: the D-02b walk (sticky/d02b_walk.py, TL_EXPAND=1, TL_DRAFT = the new weights), block 16, the official 1-step
  loop. Kept per check on the 24 held-out HumanEval+ samples (code) and the 60 GSM8K answers (math). Control = L1-20
  (9.57 / 7.00): the same data and recipe at 5 layers.
- Lines:
  - PASS = code kept >= +0.30 over L1-20 (>= 9.87). Capacity is a lever at today's data; the cloud build is worth it.
  - KILL = code < +0.10 (< 9.67) AND math < +0.10 (< 7.10). Extra layers don't pay at this data; the cloud build's
    case then rests on P-OFF (more data) alone.
  - Else BETWEEN.
- Caveats:
  - One seed: retrain noise is unmeasured (D-09's whole data effect was 0.20).
  - The 32 x 8 regrouping has no separate control.
  - The new layers learn from ~22.5K anchors only. A KILL doesn't rule out capacity WITH much more data; that is the
    joint question for P-OFF.
- Also measured: the 10-layer drafter's propose time per 16-row check vs L1-20's, to price the extra layers.
- Prediction: code +0.2 (BETWEEN); 30 % PASS, 30 % KILL.
- Guards: start after Breakthrough's map batch (the orchestrator's 19:05 slot) with >= 36 GB free (free + inactive +
  purgeable). Share lock MEM 32 GB until the first step's memory line says otherwise. Swap guard: stop at +2 GB (the
  trainer checkpoints; --resume continues). The walk runs under the same lock.

## MTP-BLOCK: the native MTP layer as a one-pass block drafter (probe 3 of notes/DRAFTER_PLAN_1104.md) -- KILL by analysis, no run (Wed 11-04 18:04)
- The plan's idea: draft all 15 depths of a check in ONE pass of the 125B's MTP layer (DFlash-style mask rows), instead
  of the 15 sequential chain steps (1.35 ms each, MTPSCREEN), keeping the target-class weights.
- Why it can't work as designed: in a one-pass block, rows 2..15 have no known token and no chain state. Their inputs
  are a learned mask token, plus at most the anchor's widened stream. These inputs are identical within a block except
  for the RoPE position. With ONE layer, the mask rows can attend only to the context (the true pairs u <= s) and to each
  other's inputs. Those inputs carry no anchor-specific information beyond what each row already has, so the rows can't
  coordinate on the intermediate tokens. Each row becomes an independent per-offset predictor from the context.
  - That is Medusa's design (arXiv 2401.10774: per-offset heads from the last hidden state; their top-1 accuracy falls
    fast with the offset).
  - DFlash gets its far depths from several layers of mask rows attending to each other's updated states (DFlash2: 5
    layers, 0.86-0.98 per depth on code). The MTP chain gets them from its own drafted tokens.
- Expected kept at 16 rows: well below DFlash2's 9.57, since depths >= 3 would fall to per-offset-head levels. It can't
  reach Kamen's 95-96 % per token. Its only possible win would be a cheaper drafter at lower kept: at a 50 ms cycle,
  ~8.5 kept at ~3 ms vs 9.57 at ~10 ms is about break-even.
- Untested variant, kept as an idea (TRIED): MTP-LOOP. Apply the MTP layer 2-3 times over the block with tied weights,
  so the mask rows can exchange information (DFlash with tied layers). Cost ~2-3 MTP-layer passes at 16 rows per check;
  it needs the D-22 trainer extended to a block loss (~1 h build).
- The GPU slot goes to P-OFF (below), which tests the data lever with the corpus Kamen approved at 17:5x.

> (a quote from the project owner, kept private)
- Why: data at DFlash2's size adds +0.10-0.13 kept per doubling ON-policy (D-09 + E2/D2). On-policy tokens cost the
  125B's decode (~50-110 tok/s). Off-policy prefill costs ~300-690 tok/s, so it is the only cheap route to 20-100M
  tokens for a cloud build.
- Data:
  - Source: Magicoder-OSS-Instruct-75K (ise-uiuc, MIT), data-oss_instruct-decontaminated.jsonl, 203,201,169 bytes,
    sha256 f988f14d3516a9f9...; downloaded 17:55-17:56 into ~/TrainLab/ext/magicoder_oss75k (untrusted data; parsed
    as JSON only, python -I).
  - sticky/poff_prep.py: Python items only (38,284), shuffled with seed 0, in the 125B's chat template exactly as L1-20's
    windows (enable_thinking False; asst = the solution + <|im_end|>; gen = its first token). Result: 1,698 windows,
    892,137 tokens to prefill, 400,124 assistant tokens (17.8x L1-20's 22,504 anchors). ext/poff/poff_windows.jsonl.
- Extraction (GPU, ~40 min at ~370 tok/s): Leap's dfl2c_extract.py (DFL2C_IN / DFL2C_OUT, C 51) gives h, the top-32
  labels and the on-path rate.
  - Diagnostic (written now): L1-20's recipe trains depth j of an anchor only while the TEXT equals the 125B's greedy
    path for j-1 tokens.
  - On off-policy text, far-depth signal therefore scales like (on-path rate)^(j-1). At a rate of 0.7, depth 8 keeps
    ~8 %; at 0.9, ~48 %.
  - A low on-path rate predicts that P-OFF can't move the far depths, which is where the D98 gap is (D-09).
- Training (sticky/poff_train.py = sticky/pcap_train.py without --expand, plus --init):
  - Start from L1-20, continuing L1-20's recipe (lr 2e-5 warm-up + cosine to 10 %, pmask 0.5, block 32, A 64 x S 4,
    micro 1) for 1 epoch over L1-20's 99 train windows plus the first 160K assistant tokens of the off-policy windows
    (~180K anchors, ~700 steps, ~1.7 h). Val = L1-20's 10 val windows.
  - Compared at the on-policy slope, 22.5K -> ~183K anchors is ~3.0 doublings, which predicts +0.3-0.4 code if
    off-policy tokens were as good as on-policy ones.
- Score: the D-02b walk (code 24 HumanEval+ / math 60 GSM8K) vs L1-20 (9.57 / 7.00).
- Lines:
  - PASS = code >= +0.30 (off-policy prefill works at about the on-policy slope; the cloud build's data plan is
    prefill-first).
  - KILL = code < +0.10 (the cheap route doesn't move kept; data must be on-policy decode, ~5x slower to collect).
  - Else BETWEEN. Math is reported but not judged: the new data is code only.
- Prediction: on-path rate ~0.75 on assistant tokens; code +0.10 (BETWEEN/KILL edge); 20 % PASS, 45 % KILL.
- Guards: the extraction and the training each run in the orchestrator's GPU slot (share lock with MEM from a 3-window
  smoke, swap guard +2 GB, stop under 110 GB of free disk = the extractor's own guard).

### P-OFF amendment (Wed 11-04 18:1x, before any extraction or training): sized to the orchestrator's slots
- Slots: extraction ~21:00-21:40 (~16 GB). Training only if the on-path rate looks alive, cut to <= ~65 min including
  the walk, in 21:50-22:55. Otherwise a Thursday slot for the full 160K version.
- "Alive" line, written now: on-path rate >= 0.75 on assistant tokens. Below it, far depths get < 9 % of the anchors at
  depth 8, so this recipe can't move the far depths. Training is then skipped, and P-OFF = KILL for prefill-only data
  under L1-20's recipe. The remaining route is on-policy: decode the 125B's own answers to these prompts (~5x slower).
- The 65-minute version:
  - Data: the first 60K off-policy assistant tokens (+ L1-20's 99 train windows; ~355 steps, ~50 min at L1-20's 8.3 s
    per step; --minutes 55 as a hard cap).
  - Walk: code only (TL_WALK_SETS code = the same 24 dfl2d_code files, identical computation), ~4 min.
  - 22.5K -> ~82.5K anchors is ~1.9 doublings, so the on-policy slope predicts +0.19-0.24.
- Lines for this version, replacing the 160K lines above:
  - PASS >= +0.18 code: about the on-policy expectation, so off-policy prefill works.
  - KILL <= +0.05.
  - Else BETWEEN.
  - Retrain noise is unmeasured; D-09's whole 2-doubling effect was +0.20.
- The full 160K version keeps the original lines if it runs on Thursday.

### P-OFF amendment 2 (Wed 11-04 19:4x, before any extraction or training): tonight right after P-CAP; extraction sized from Leap's measured speed
- The orchestrator 19:4x: P-OFF is back on tonight, right after P-CAP. The extraction gives up if it can't start by 21:40
  (then Thursday). Training runs only if the rate is alive and it starts by 21:55, done by ~22:55. Breakthrough's s7_trl
  runs after it.
- **Correction to my 18:1x estimate.** I assumed ~370 tok/s for the extraction (~40 min for all 892K tokens). Leap's own
  log (~/Leap/logs/job.out, the L1-20 extraction) measured 109 windows / 42,355 tokens in 763 s: 56 tok/s, ~7 s per
  short window. The cost is SSD-bound: each window streams nearly every non-resident expert (C 51). That run's MLX peak
  was 22.4 GB on windows of <= 609 tokens, not ~16 GB. All 1,698 windows would take ~3.3 h.
- **Tonight's extraction is therefore the training prefix only:** the first 260 windows of poff_windows.jsonl, which
  hold 60,162 assistant tokens and 136,845 tokens to prefill. The longest is 1,271 tokens. Estimate ~32 min.
  - Smoke: the 3 longest windows of that prefix (ext/poff/smoke3.jsonl), so the smoke footprint bounds the run's. It
    stops above 27 GB. The share lock declares MEM 28 GB.
  - The rest (windows 261-1,698) waits for Thursday's 160K version. The extraction is resumable; ~680 windows ≈ 80 min.
- **The decision rate** is the extractor's own "on-path rate (assistant)" line over windows 1-260. It excludes the 3 smoke
  windows, which were extracted first. The alive line stays >= 0.75.
  - Also reported: the same count over all 260 windows (sticky/poff_data.py). It reproduces the extractor's L1-20 number
    (0.9978 vs 0.998 printed).
- Training, walk and lines are unchanged from amendment 1 (60K version: PASS >= +0.18 code, KILL <= +0.05). The training
  lock is 90 min, not 70, for heat pauses; the cap is still --minutes 55 of trained time.

### P-OFF amendment 3 (Wed 11-04 20:0x, before any extraction or training): after BigMoE's 45 GB run; training gate 22:15 with >= 36 GB
- The orchestrator 20:0x: Kamen is closing Chrome so BigMoE's 45 GB quiet run (WAITMEM 44) goes right after P-CAP,
  ~20:40-21:30. P-OFF's extraction must not start before it.
- The wrapper (turns/poff_night.sh) now waits for:
  - P-CAP's runner to end;
  - then BigMoE's lock to come and go (or 21:00 if it hasn't started);
  - then 3 clear minutes with >= 30 GB available.
  The extraction still gives up at 21:40.
- Training may now start up to 22:15 (was 21:55), so it still ends before ~23:10. It needs 3 clear minutes with >= 36 GB
  available (free + inactive + speculative + purgeable), since Chrome may be reopened after ~21:30. Otherwise Thursday.
  Breakthrough's s7_trl may run first if its store is whole first.
- Data, rate line (>= 0.75), recipe and PASS / KILL lines are unchanged.

### P-OFF amendment 4 (Wed 11-04 20:1x, before any extraction): extraction only tonight; training on Thursday
- P-CAP is heat-stretched to ~20:57: its steps take 20.3 s on the wall clock vs 13.7 s trained. BigMoE's early 45 GB run
  then takes ~21:00-21:50, so the 21:40 extraction give-up would fire.
- The orchestrator 20:0x said yes to my option: the extraction alone may start as late as 22:00 (~35 min, MEM 28 GB,
  done by ~22:35). It gives tonight's on-path rate:
  - >= 0.75: alive. The 60K training runs on Thursday, with amendment 1's lines.
  - < 0.75: KILL by the pre-registered diagnostic. No training.
- Breakthrough's s7_trl goes after the extraction, or before it if its store is whole first.

### P-CAP: run STOPPED at step 148/188 by its swap guard (Wed 11-04 20:37); my CPU port tests caused it. No walk yet.
- Cause: from 20:19, my PORT tests ran on the CPU beside it. An MLX CPU reference process (~4-5 GB) and a 5.6 GB PyTorch
  step-1 check overlapped from 20:37. Swap went 8378 -> 10433 MB (+2 GB) and the runner's guard sent SIGTERM.
  - The trainer finished the step and saved its checkpoint: ext/pcap/pcap10.safetensors (bf16) and .state.json, step
    148, data order kept.
  - "CPU only" was not "memory free". Rule from now on: port tests total <= 4 GB, one process at a time, footprint
    measured first, none during a quiet run.
- Memory: MLX peak 35.4 GB, so the lock was re-declared MEM 36 at step 20 (declared 32). The start guard was 34 GB
  available (pre-registered: 36 GB free).
- Early read, NOT the pre-registered measure (that is the walk). Validation at step 100, where both runs have seen 25.6K
  anchors:

  | | argmax, mean gain d1-15 | selector, mean gain d1-15 | depth 10 argmax | depth 15 argmax |
  |---|---|---|---|---|
  | L1-20 (5 layers, A 64 S 4) | +0.095 | +0.092 | 0.472 -> 0.619 | 0.087 -> 0.303 |
  | P-CAP (10 layers, A 32 S 8) | +0.134 | +0.126 | 0.528 -> 0.712 | 0.101 -> 0.378 |

  - P-CAP learns ~40 % more per depth at equal data. Caveats: the 24 validation groups differ (A 32 vs A 64 grouping),
    the batch shapes differ, and the new layers run at 10x the learning rate.
  - It says nothing yet about kept per check.
- To finish: resume the last 40 steps (the MLX trainer restarts AdamW with a 10-step ramp; a deviation to record),
  then the walk. ~25 min at MEM 36 GB, in a slot from the orchestrator (asked 20:4x).
- 20:4x, the orchestrator: the resume goes FIRST after BigMoE's early 45 GB run releases its lock (~21:35-21:50). If that
  run hasn't started by 21:15, it moves to 23:00, and the resume takes the GPU as soon as it is free. turns/pcap_resume.sh:
  - start rule: >= 36 GB available (the pre-registered guard), 3 clear minutes; gives up at 22:25;
  - share lock MEM 36 GB; swap guard +2 GB;
  - the same pass / kill lines.
  **Deviation:** the MLX trainer's resume restarts AdamW's moments, with a 10-step learning-rate ramp, for steps 149-188.
  P-OFF's extraction moves to Thursday. It runs tonight only if BigMoE's run moved to 23:00 and the extraction can start
  by 21:45.

### PORT step-1 parity: which embedding rounding the GPU used (pre-registered 20:5x, before the A/B)
- MLX 0.32.2's CPU dequantize rounds twice: bf16(bf16(s * q) + b), bitwise vs MLX-CPU. Its GPU kernel very likely rounds
  once (an fma in float). L1-20 trained on the GPU.
- Two full-size step-1 runs so far:
  - 5.3434 with one rounding (attn chunk 16, rows 256, 6 threads, fp32 storage);
  - 5.3474 with two (chunk 8, rows 128, 4 threads, bf16 storage; a 6.8 GB footprint, over my own 4 GB rule; the
    bf16-storage build kept a second full copy, since fixed).
- A/B rule, written now: rerun both roundings with identical settings. The rounding whose loss is closer to L1-20's
  logged step-1 loss is taken as the GPU's and becomes the port's default. The other stays an option ("twice" =
  MLX-CPU tests).
- **RESULT 20:56** (both runs: --bf16-store, chunk 8, rows 128, 4 threads; only the embedding rounding differs):
  - once: 5.343469, -0.000507 vs L1-20's logged 5.343977 (-9.5e-5 relative);
  - twice: 5.347406, +0.003429 (+6.4e-4).
  By the rule above, **one rounding is the GPU's arithmetic** and stays the default. The port reproduces L1-20's GPU
  step-1 loss to 1e-4 relative. MLX-CPU's two-rounding dequantize differs from its GPU kernel.
- Footprint 4.87 GB per run, again over my own 4 GB rule (by 0.9 GB, ~17 s each; only Fit12's 2 GB run beside it).
  The full-size step-1 check runs again only when nothing else runs, or after cutting it below 4 GB.

### PORT walk validated (Wed 11-04 21:0x): port/walk_port.py vs the MLX walk, L1-20 weights
| | checks | kept per check | per-token rate | P(first 15) |
|---|---|---|---|---|
| code, MLX walk (bf16 drafter) | 666 | 9.57 | 0.9239 | 0.276 |
| code, PORT walk (fp32 drafter) | 660 | 9.66 | 0.9258 | 0.288 |
| math, MLX walk | 2046 | 7.00 | 0.8707 | 0.086 |
| math, PORT walk | 2043 | 7.01 | 0.8710 | 0.087 |

- The port runs the drafter in fp32, the trainer's regime; MLX's walk runs it in bf16. A few near-tie picks differ: +0.09
  kept on code, +0.01 on math. Per-depth rates agree within ~0.01.
- Good enough for Kaggle to score checkpoints against each other. The Mac's bf16 walk stays the reference for absolute
  numbers.
- Footprint: the math run peaked at 5.97 GB, over my own 4 GB rule (nothing big beside it; Fit12 2 GB). Cause: the
  head's 512-row transients (~1.8 GB). Fix in the code (one bf16 output buffer, 256 rows; head parity still PASS,
  2.55 GB). The walk's footprint is NOT re-measured yet: rerun the math walk (batch 30) when the Mac is quiet.

### P-CAP RESULT (Wed 11-04 21:33): BETWEEN. Code 9.57 -> 9.75 kept (+0.18), math 7.00 -> 6.92 (-0.08). Capacity alone, at today's data, is a small lever.
| | checks | kept per check | per-token rate | P(first 15) |
|---|---|---|---|---|
| code, L1-20 (5 layers, 1.21 GB) | 666 | 9.57 | 0.9239 | 0.276 |
| code, P-CAP (10 layers, 2.09 GB) | 654 | **9.75** | 0.9281 | 0.301 |
| math, L1-20 | 2046 | 7.00 | 0.8707 | 0.086 |
| math, P-CAP | 2069 | **6.92** | 0.8703 | 0.093 |

- **Pre-registered lines:** PASS = code >= 9.87, which it misses. KILL = code < 9.67 AND math < 7.10, which it isn't,
  since code is 9.75. The verdict is **BETWEEN**. The prediction was +0.2 on code (30 % PASS / 30 % KILL); the result
  is +0.18.
- **Code, per depth 1-15:** P-CAP 0.943 0.920 0.920 0.931 0.936 0.946 0.953 0.927 0.943 0.939 0.913 0.925 0.911 0.874
  0.883 vs L1-20 0.941 0.922 0.906 0.946 0.955 0.931 0.926 0.939 0.943 0.916 0.908 0.906 0.888 0.887 0.858. Most of the
  gain is at depths 6-15. Math is flat (rate 0.8703 vs 0.8707).
- **For the drafter plan's decision** ("+0.3 = capacity is the lever, < +0.1 = the route is closed"): +0.18 is in
  between, worth about 1.5 data doublings at the on-policy slope.
  - The cloud build's case now rests on capacity x data compounding, which no probe tests yet.
  - P-OFF (more data, off-policy) is the next input. Its extraction runs now.
- **As an engine lever on its own: no.** 2x the layers is roughly 2x the drafter's ms per check, against +1.9 % kept on
  code (-1.1 % on math). The pre-registered "also measured" propose time is not built yet; the sign is clear without it.
- **Deviations:**
  1. My CPU port tests tripped the swap guard at step 148. The resume restarted AdamW's moments with the trainer's
     10-step ramp for steps 149-188, at lr 4.4e-7 -> 2e-6, the cosine tail.
  2. The lock was declared MEM 32 and re-declared 36. The resumed segment's footprint was 37.4 GB.
  3. The first segment started on >= 34 GB available, against the pre-registered 36 GB free. The resume started on 36.
  4. Single runs: retrain noise is unmeasured, and D-09's whole 2-doubling effect was +0.20. So +0.18 / -0.08 is within
     what noise could move.
- Files: ext/pcap/pcap10.safetensors (gitignored); data/d02b_walk_pcap.txt and data/d02b_walk_pcap_{code,math}.npz;
  logs data/pcap_train.log, data/pcap_train2.log, data/pcap_walk.log.

### PORT steps 1-5 vs L1-20's log: pre-registration (Wed 11-04 22:0x, before any run)
- Thursday on the orchestrator's 21:5x rule: a share lock MEM 12 GB once runner 42 runs, only if NO-SWAP stays >= 20 GB
  beside it. Script: turns/port_steps_thu.sh.
- Run: train_port.py with L1-20's exact recipe (fp32 weights + MLX AdamW), 5 steps, on the CPU.
- **PASS = every step's loss within 2e-3 relative** of L1-20's logged loss (5.343977, 4.695799, 4.796175, 4.645471,
  3.723465). Step 1 already agrees to 1e-4; later steps add GPU-vs-CPU accumulation drift through Adam. FAIL otherwise.
  Tool: port/compare_l120.py.
- Slot amendment (Thu 09:13; PASS line unchanged): runner 42 ran as a shared run (MEM 36 GB), but only 15 GB was
  available beside it (swap 10.7 GB), so the NO-SWAP >= 20 GB rule failed and the check did not start beside it. Fallback
  (the orchestrator's conversion window): turns/steps_after_conv.sh waits for BigMoE's records3 conversion (a quiet lock
  right after 42's verdict, ~10:15-10:25) to end, then runs port_steps_thu.sh with its rules unchanged (no quiet lock or
  ticket, >= 32 GB available for 3 clear minutes); it gives up at 11:30. The walk re-measure follows under the same lock.
- Cap (Thu 09:2x, the orchestrator): EvalSuite holds runner 43 until this check ends, cap ~10:50. So the steps start
  only by 10:38, anything still running at 10:50 is stopped (every step's loss is in the log, so a stop leaves a partial
  read-out over the steps done), and the walk starts only by 10:44. If 43 can't be held: the 43-44 gap (~14:10).
- Gate, replacing the cap (Thu 09:2x, the orchestrator's 2nd note): runner 43 now waits for ~/EvalSuite/results/q43/GO
  (this script touches it on every exit path) or 10:50, and while any share lock shows MEM > 5 GB (mine says MEM 12
  GB). By day this check and Breakthrough's split code arms (~20 min, also right after the conversion) go one at a
  time, in either order: the gate also waits while another session's lock shows MEM > 5 GB, and backs off if two locks
  land together. Start by 11:30 (the give-up stays); the walk only by 11:40; 12:00 = runaway stop.

#### RESULT (Thu 11-05 10:38-10:43): **FAIL**. Step 1 matches L1-20 to 1.1e-4; steps 2-5 are 1.5-3.8 % ABOVE its log.
| step | port loss | L1-20 logged | rel | port grad-norm | lr (both: the same schedule) |
|---|---|---|---|---|---|
| 1 | 5.343394 | 5.343977 | 1.09e-4 | 20.61 | 2.00e-7 |
| 2 | 4.851293 | 4.695799 | **3.31e-2** | 27.06 | 1.19e-6 |
| 3 | 4.978863 | 4.796175 | **3.81e-2** | 24.19 | 2.18e-6 |
| 4 | 4.796092 | 4.645471 | **3.24e-2** | 21.14 | 3.17e-6 |
| 5 | 3.777776 | 3.723465 | **1.46e-2** | 29.24 | 4.16e-6 |
- Reading: the forward is right (step 1, and the step-to-step shape follows L1-20's batches, so data order and masks
  match), but after the first update the port's model is worse on every next batch. Something in the update differs:
  the gradient (e.g. a stop-gradient placed differently, so the loss value matches but its gradient doesn't), the
  trainable set, clip, or AdamW. L1-20's trainer is sticky/d09_train.py (identical to Leap's scripts/dfl2c_train.py,
  last changed 10-24 16:42, before L1-20's run): trains layers + norm + candidate_selector, AdamW(wd 0), clip 1.0, warm-up
  1 % -> 100 % over 20 steps (the port's logged lr matches it). Diagnosis next (code first, then a small differential).
- The port is NOT yet validated past step 1: Kaggle runs wait for this fix.
- Run: 258 s wall (41-59 s per step at 6 threads, CPU). Footprint **14.4 GB** vs MEM 12 GB declared (fp32 weights +
  AdamW moments + grads). Swap flat (10,629 -> 10,621 MB); 47 GB available after it. Runner 43's GO touched 10:44:52.

### P-OFF diagnostic RESULT (Wed 11-04 22:10): ALIVE. The 125B's greedy token equals the corpus answer on 0.86 of answer tokens
- On-path rate (assistant tokens): **0.860** by the extractor's line over windows 4-260 (the decision line), **0.8622**
  over all 260 (sticky/poff_data.py). The pre-registered alive line was >= 0.75, so it passes.
- Per window it ran ~0.84-0.96; the 3 smoke windows (the longest) gave 0.960.
- What it means for training: the loss weights depth j only while the text stays on the 125B's path. At 0.86, depth 8
  keeps ~0.86^7 = 35 % of its anchors and depth 15 ~12 %. Prefill-only code data trains the near depths fully and the far
  depths partly. It is worth the 60K training run (amendment 1's lines: PASS >= +0.18 code, KILL <= +0.05).
- Extraction, measured: 260 windows / 136,845 tokens in 36.7 min (21:33:44-22:10:25). That is 62 tok/s, ~8.5 s per
  window (Leap's log: 56 tok/s). Footprint 17.7 GB (smoke 16.8, MEM 28 declared). Swap flat.
- The training folder is built: ext/poff/train60 = L1-20's 109 windows + the 260 off-policy windows (60,162 assistant
  tokens). The training (turns/poff_t.sh, ~55 min + walk, MEM 34 GB) needs a Thursday slot from the orchestrator.
- Slot (Thu 11-05 23:2x; lines and steps unchanged): turns/poff_fri.sh queues its quiet ticket at Thu 23:45, not Fri
> (a quote from the project owner, kept private)
  comparable with L1-20.

### P-OFF 60K RESULT (Fri 11-06 00:52): BETWEEN. Code walk kept per check 9.68 vs L1-20's 9.57 (+0.11; PASS >= +0.18, KILL <= +0.05). After SEED (Fri 02:57): NO EFFECT, since one reseed alone moves code by 0.16
- Run: turns/poff_fri.sh, quiet lock granted Thu 23:45:06. sticky/poff_train.py from L1-20 (l120_ft), 1 epoch of
  ext/poff/train60 (369 windows: L1-20's + the 260 off-policy ones; 1,453 train groups, 82,406 anchors), L1-20's recipe
  (lr 2e-5, S 4, A 64, micro 1, bs 32, pmask 0.5), MLX's default TF32 like L1-20.
- The 55-min cap bound: 341 of 364 steps. Steps took 9.7 s, not the 8.3 s assumed. The 23 missing steps would have run
  at lr <= 2.2e-6, about 11 % of the peak.
- Memory: MLX peak 34.4 GB active + 2.2 GB cache; footprint 36.8 GB (time -l). Swap flat (8,224 -> 8,208 MB).
- Code walk (sticky/d02b_walk.py, the same 24 dfl2d_code files):

  | | checks | kept per check | per-token rate | P(first 15 right) |
  |---|---|---|---|---|
  | P-OFF | 658 | 9.68 | 0.9270 | 0.300 |
  | L1-20 | 666 | 9.57 | 0.9239 | |

  The walk's header still reads "L1-20" (fixed text). The numbers differ from L1-20's exactly reproduced 666 / 9.57,
  so the P-OFF weights were the ones walked.
- VAL (L1-20's 24 val groups, on-path top-1): depths 1-2 up 0.001-0.003. Depths 3-16 down 0.003-0.021 (depth 7
  0.807 -> 0.793, depth 14 0.446 -> 0.425).
- Reading: +0.11 is about half the on-policy slope's prediction (+0.19-0.24). Retrain noise is unmeasured, so it can't
  be told apart from noise. L1-20 stays the drafter. Settling it needs one of: a noise probe (L1-20's recipe rerun) or
  the full 160K version. Both need night slots; data: data/poff_*.log, data/d02b_walk_poff*.

### SEED: L1-20 retrain noise (pre-registration, Fri 11-06 01:4x, before the run; the orchestrator 01:2x)
- Why: P-OFF 60K came out +0.11 code kept per check over L1-20 (BETWEEN), and retrain noise was never measured. The
  TF32-off retrain probe's "no effect" line (|delta| < 0.10) needs it too.
- Run: sticky/seed_train.py = sticky/d09_train.py (L1-20's trainer) plus --seed. --seed 1 reseeds the training data order
  and the pmask draws; the 24 VAL groups stay L1-20's. Everything else is L1-20's recorded args (l120_ft.state.json):
  --epochs 2 --minutes 50 --lr 2e-5 --S 4 --A 64 --val-every 100 --micro 1 --cache-gb 2 --bs 32 --pmask 0.5, data
  ~/Leap/data/l120, MLX's default TF32 like L1-20. Then the D-02b walk, code + math (the default sets, which reproduce
  L1-20's 9.57 / 7.00 exactly).
- Slot: turns/seed_l120.sh, not before 02:10. It queues a quiet ticket after 3 clear minutes with no other quiet lock
  or ticket, so it runs after Leap's NATIVE-FRI and FLAT-2.
  - MEM 37 GB (P-OFF's footprint was 36.8). Swap guard +2 GB.
  - The chunk must end by 10:15: no start after 09:25, and a stop at 10:12.
- Measure: |dcode| = |kept(seed 1) - 9.57| and |dmath| = |kept(seed 1) - 7.00|. That is one sample of the seed-to-seed
  difference, so it gives a rough scale only.
- Reading for P-OFF's +0.11:
  - |dcode| >= 0.11: P-OFF is within one seed difference. Read it as no effect; the 160K version isn't worth a slot on
    this evidence.
  - |dcode| <= 0.04: +0.11 is ~3x one seed difference. Likely a small real gain (still below PASS); the 160K version is
    worth a slot.
  - Between: still unreadable; a second seed would be needed.
- For the TF32-off probe: if |dcode| or |dmath| >= 0.10, one run per arm can't separate TF32 from noise. It is
  re-priced with seeds before it runs.
- Void if the run doesn't finish all 201 steps (cap, guard or the 10:12 stop).

#### RESULT (Fri 11-06 02:13-02:57): |dcode| 0.16, |dmath| 0.01. P-OFF's +0.11 is within one seed difference: NO EFFECT
- Valid: all 201 steps ("DONE: saved ... 32.5 min of training, 201 steps"). The quiet lock was granted at 02:13:45
  (#1 in the queue, 15 s wait) and released at 02:57:28. Footprint 35.4 GB (MLX peak 33.7 active + 2.2 cache); walk
  5.8 GB. Swap flat (11,004 -> 10,988 MB).
- Walk (sticky/d02b_walk.py, the default code + math sets, ext/seed/l120_s1.safetensors):

  | | code checks | code kept per check | code per-token rate | math checks | math kept per check | math per-token rate |
  |---|---|---|---|---|---|---|
  | L1-20 (seed 0) | 666 | 9.57 | 0.9239 | 2,046 | 7.00 | 0.8707 |
  | seed 1 | 655 | 9.73 | 0.9260 | 2,047 | 6.99 | 0.8701 |
  | P-OFF 60K | 658 | 9.68 | 0.9270 | | | |

  dcode = +0.16, dmath = -0.01. The walk header's "L1-20" is fixed text; the numbers differ from L1-20's, so the
  seed-1 weights were the ones walked.
- VAL (the same 24 groups, on-path top-1, depths 1-16): seed 1 vs L1-20 ranges -0.027 to +0.005 (mean -0.0055).
  P-OFF's ranged -0.021 to +0.003 (mean -0.0083). P-OFF's VAL drops are the size of a reseed's.
- Reading (as pre-registered): |dcode| 0.16 >= 0.11, so P-OFF's +0.11 reads as no effect. The 160K version isn't worth
  a slot on this evidence: parked.
- Scale (one sample only): code (24 samples, ~660 checks) moves ~0.16 per retrain; math (60 samples, ~2,050 checks)
  moves 0.01. So a single-run code delta below ~0.3 can't be told from retrain noise; that includes P-CAP's +0.18. Math
  is the steadier read; P-CAP's math -0.08 is 8x this sample, but it is still one sample.
- TF32-off probe, re-priced because |dcode| >= 0.10: one run per arm can't be read on code. It would need >= 2 seeds
  per arm (~4 x 35 min GPU + walks), with math as the primary measure. TF32's rounding is noise, not a bias, so the
  expected effect on kept rate is small. The retrain is parked. TF32 off stays the rule where port parity matters; the
  full-scale confirmation (5 steps) runs as planned.
- Data: data/seed_l120.log, seed_lock.log, seed_train.log, seed_walk.log, d02b_walk_seed1.txt. The weights
  (ext/seed/l120_s1.safetensors) are not in git.

### PORT walk footprint re-measure: pre-registration (Wed 11-04 23:4x, before the run)
- Why: the 21:0x math walk peaked at 5.97 GB (section 'PORT walk validated'), over my 4 GB rule for port tests. The fix
  (one bf16 head output buffer, 256-row chunks) passed head parity at 2.55 GB, but the walk itself was not re-run.
- Run: turns/walk_foot.sh. The same command as the 21:0x validation: math, batch 30, --bf16-store, 4 threads, CPU only
  (154 s then). Share lock MEM 7 GB. It starts only in a real gap: no other session's quiet lock or queued ticket, no
  other MEM >= 5 GB lock, >= 24 GB available, lid open, 3 clear minutes. It gives up at Thu 06:45 (BigMoE's 07:05).
  Tonight EvalSuite's RELAX holds quiet locks in hourly chunks, so it may give up; then it waits for a later gap.
- Lines: footprint (/usr/bin/time -l peak memory footprint) <= 4.0 GB = FIXED; above = still over (walks beside other
  jobs keep --batch <= 15). Kept per check must stay 7.01 +- 0.02 (2043 checks); outside that, the head change broke
  the walk.
- Thu 06:45: NOT RUN. turns/walk_foot.sh found no gap overnight: EvalSuite's RELAX held quiet locks, then a MEM 30 GB
  shared replay (06:45: big=1, 14 GB available). New slot (amendment, lines unchanged): inside the steps check's share
  lock (turns/port_steps_thu.sh, MEM 12 GB), after the steps check exits (one port process at a time), only if the
  steps check wasn't stopped, no other session holds or queues a quiet lock, and >= 26 GB is available; it stops if
  available < 20 GB or swap +2 GB.

#### RESULT (Thu 11-05 10:42:52-10:44:52): NOT FIXED. Footprint **4.74 GB** (was 5.97; line <= 4.0). The walk itself is fine.
- math, batch 30, --bf16-store, 4 threads: 60 samples, 2043 checks, kept per check **7.01**, per-token rate 0.8710,
  P(first 15 right) 0.087. Identical to the 21:0x run, so the head-buffer change didn't break the walk. 112 s (was 154).
- The head fix saved 1.2 GB; the remaining ~0.7 GB over the line is elsewhere (likely the 30 samples' context features
  and the batch's fp32 context tensor). Rule stays: walks beside other jobs use --batch <= 15.

### PORT steps FAIL, diagnosis 1 (Thu 11-05 10:59): one batch's gradient, MLX (GPU, the trainer's own code) vs port, first layer only: MATCH
- Setup: port/ref_mlx.py grad --gpu vs port/test_parity.py grad; first HF layer, A 8, 1 train group, pmask 0.5 (the
  trainer's PM_RNG); per-parameter norms + 8 hashed +-1 projections (port/grad_stats.py, identical in both venvs).
  Data: data/port/gradpar/ (mlx.json, port.json, compare.txt).
- Loss 12.749070 vs 12.748799 (rel 2.1e-5). Total gradient norm 16.159 vs 16.170 (ratio 1.0007).
- Every layer-0 parameter and the final norm: relative difference 3e-4 to 3e-3, the bf16 roundings' level. Exact on the
  small ones (q_norm / k_norm 9e-4, base kernels 1.4-1.7e-3, final norm 3.3e-4).
- NOT covered: the candidate selector's gradient is ZERO on both sides (a 1-layer drafter never ranks the true token in
  its top 16, so the selector loss is inactive). Also not covered: the full run's --ckpt-layers, --attn-chunk 8 and
  --rows 128 (this test used no recompute and no chunking), 5 layers, and the optimizer over steps.
- Footprints: MLX side 5.47 GB (MLX peak 3.85), port 4.10 GB; the MLX side is over my 4 GB port-test rule (short,
  share lock MEM 4 GB, 25 GB available, swap flat).

### PORT steps FAIL, diagnoses 2-3 (Thu 11-05 11:02-11:05): recompute exact, selector gradients match, data identical over 5 steps
- 2a (port only, 1 layer, A 8): --ckpt-layers + --attn-chunk 4 + --rows 128 vs none: loss identical, gradients
  identical to <= 7e-8. The full run's recompute and chunking are not the cause.
- 2b (MLX GPU vs port, all 5 layers fp32 and frozen, so the final norm and the candidate selector get gradients and the
  selector loss is active): loss 2.022318 vs 2.025148 (1.4e-3); gradient norms within 0.1-0.4 %; estimated relative
  differences ~1e-2 (successor codebook 1.1e-2, hidden projection 7.7e-3, predecessor codebook 7.4e-3, norm 6.1e-3
  exact). Larger than layer 0's 1e-3 but no systematic error; consistent with near-tie flips at the top-16 boundary of
  the bf16-rounded logits.
- 3 (no model; port/test_data_steps.py): the MLX trainer's own load_windows / groups_of / make_batch (from its source)
  plus its pre-loop RNG use and item selection vs the port's, over L1-20's first 5 steps: the same groups (step 1:
  gsm_21, gsm_20, he_160, gsm_7) and BITWISE IDENTICAL batch arrays (context, positions, tokens, top-32 targets, masks,
  weights). 0.7 GB.
- So far: data, forward, one-batch gradients (layers, norm, selector) and recompute all match. Untested: the update
  applied over steps (diagnosis 4, running) and gradients at the full scale.
- Footprints: 3.87-4.91 GB (port), 5.21 GB (MLX side); share lock MEM 5 GB; swap flat. Data: data/port/gradpar2/,
  data/port/data_steps.txt.

### PORT steps FAIL: ROOT CAUSE FOUND (Thu 11-05 11:05-11:20). MLX runs fp32 matmuls as TF32 by default; the port is right.
- MLX 0.32.2: mlx/include/mlx/utils.h `enable_tf32() { get_var("MLX_ENABLE_TF32", 1) }`, i.e. ON unless
  MLX_ENABLE_TF32=0 (libmlx also carries M5 "nax" GEMM kernels). Nothing here sets it, so L1-20 (and every MLX run
  with "fp32 master weights": P-CAP, P-OFF, D-09 ...) trained with TF32 matmuls.
- Direct test (port/test_mlx_tf32.py, 256 x 2560 @ 2560 x 640, GPU): fp32 matmul error vs float64 4.15e-4 by default,
  6.5e-7 with MLX_ENABLE_TF32=0. A +-1e-5 relative weight change moves the output ~51x more than the true change by
  default (both signs), 1.00x with TF32 off.
- The chain that found it (1 layer, A 8, gsm_0; data/port/{trainpar,dppar,lr0par,swappar,bisect,tf32off}/):
  - diagnosis 4: 5 optimizer steps, MLX vs port: step 1 equal, step 2 MLX 12.085 vs port 12.332 (the full run's sign).
  - 5: the step-1 parameter change is the same on both sides (norm 0.004999 each).
  - 6: with lr 0, step 2 is identical (12.343787 / 12.343735): same batch, same forward.
  - 7 (swap): each side's forward gives its own number with EITHER side's step-1 weights (MLX 12.0849 / 12.0848, port
    12.3318 / 12.3318): the forwards differ once weights leave their bf16-exact start.
  - 8: the gap is in every matmul group (MLX 15-100x the port's per-group effect); the port's change equals the
    first-order prediction grad . dW (-0.0120 vs -0.0117); MLX's is 22x it. MLX's loss jumps already at t = 0.01 of the
    step (12.149) and saturates (t = 0.1 -> 3: 12.086 -> 12.082): a discontinuity, all in the soft CE (selector 0).
  - stage dump: identical weights, MLX vs port, the first linear layer (conv kernel projection) differs 4e-4 at the
    start and its output moves 20x the port's after the update. That is TF32's error, reshuffled by tiny weight changes.
- Confirmation with MLX_ENABLE_TF32=0: the jump is gone (t = 0.01: 12.343785; t = 1: 12.332231 vs port 12.331776),
  and 5 optimizer steps of MLX vs port agree to <= 1.2e-4 (12.748735 / 12.748799, 12.332228 / 12.331774, 11.793780 /
  11.793807, 11.211055 / 11.211951, 8.971849 / 8.972967), well inside the 2e-3 line.
- So: the port reproduces the MLX trainer's arithmetic in true fp32. The pre-registered line (L1-20's LOGGED losses)
  cannot pass without emulating TF32's rounding, and that rounding is noise: in the warm-up (lr 2e-7 .. 4e-6) Adam's
  steps (~6e-7 .. 1e-5) are below TF32's resolution for typical weights, so MLX's early steps move the model by
  re-rounding, not by the gradient (here 22x the gradient's effect, in a direction the gradient doesn't predict).
- The steps check stays FAIL as pre-registered (its reference was TF32). Re-framed validation, PASS at small scale:
  the port = MLX with TF32 off over 5 steps. Full-scale confirmation (optional): MLX's own trainer for 5 steps of
  L1-20's recipe with MLX_ENABLE_TF32=0 vs data/port/steps_thu.json (GPU ~30 GB, ~3 min; needs a slot).
- For MLX training here: MLX_ENABLE_TF32=0 gives true fp32 (speed cost unmeasured). Whether TF32 hurt or helped L1-20's
  drafter is untested (a TF32-off retrain, ~30 min GPU, is a probe). The Friday P-OFF run keeps the default, so it
  stays comparable with L1-20 and P-CAP (both TF32).
- Footprints of all diagnosis runs: 3.4-5.7 GB, one process at a time, share locks MEM 4-5 GB, swap flat.

### PORT full-scale confirmation: pre-registration (Thu 11-05 11:3x, before any run; the orchestrator 11:2x)
- When: Friday by day right after EvalSuite's runner 45 ends, if it can start by 14:50 (>= 10 min before 15:00);
  otherwise first on the Fri -> Sat night after FAR-G1 (re-armed then). Runner turns/mlx_fp32_fri.sh (share lock MEM 34
  GB; gate: no other quiet lock / ticket, no other MEM > 5 GB lock, >= 40 GB available; swap guard +2 GB).
- What: MLX's own trainer code (port/ref_mlx_steps.py executes sticky/d09_train.py up to its loop: its data, rng, step(),
  clip, AdamW, schedule), 5 steps of L1-20's recipe, no val (it draws no random numbers).
- Run 1, MLX_ENABLE_TF32=0, vs the port's steps check (data/port/steps_thu.json): **PASS = every step within 2e-3
  relative** (the small-scale test gave <= 1.2e-4); FAIL otherwise (then the root cause is incomplete).
- Run 2 (if >= 5 min remain before 15:00), TF32 on (MLX's default), vs L1-20's log: reproduced = within 2e-3 (MLX is
  deterministic enough that the log is today's TF32 trajectory); else the log isn't reproducible either (record it).
- Amendment Fri 11-06 07:3x (the orchestrator 07:2x; Kamen made the 45 GB dense-kernel rebuild today's top priority):
  start only after Leap's part B quiet lock is released (or from 10:00 if it hasn't come and gone by then). BigMoE's
  share pre-flights may run beside it. Lines unchanged. The runner is turns/mlx_fp32_fri2.sh (a copy; the first runner
  was stopped in its wait), log data/port/fp32_fri2.log.
- Amendment 2, Fri 11-06 08:0x (the orchestrator 08:0x): BigMoE's FAST-NAX engine A/B (quiet, ~30-40 min) goes first.
  Start after BigMoE's quiet lock is released, or at 09:00 if BigMoE hasn't queued by then. The gate wants 3 clear
  minutes. Lines unchanged. fri2 was stopped at 08:01, just after runner 45 ended (08:00:13) and before its gate, with
  no lock taken. The runner is turns/mlx_fp32_fri3.sh, log data/port/fp32_fri3.log. It saw BigMoE's ticket (#1 in
  line) at 08:02:00.
- Amendment 3, Fri 11-06 09:4x (the orchestrator 09:4x): EvalSuite's agent check (runner 48, shared MEM 39 GB, 09:40 to
  ~11:10-12:10) took the GPU, and swap hit 13.65 GB. Start only after runner 48 ends and EvalSuite holds no lock, with
  no other MEM > 5 GB lock (no exceptions now), and NO-SWAP >= 40 GB (PORT's ~37 GB peak + 3 GB margin; Leap's noswap10
  definition). Start by 14:50, else Saturday by day. Lines unchanged.
  - fri3 never saw runner 45's end: EvalSuite deleted q45.pid when runner 45 ended (08:00:13). It was stopped at
    09:44 in its step 1 (no lock).
  - turns/mlx_fp32_fri4.sh takes runner 48's pid when armed (09:46:01). Log data/port/fp32_fri4.log.

#### RESULT (Fri 11-06 11:28:37-11:30:27): PASS on both runs. The root cause holds at full scale.
- Slot: runner 48 ended at 11:24:03. The gate cleared at 11:28:36 (NO-SWAP 47 GB). Shared lock MEM 37 GB, ~2 min.
  Footprints 32.4 / 32.5 GB. Swap flat (14,504 -> 14,480 MB).
- Run 1, MLX_ENABLE_TF32=0, MLX's own trainer code vs the port's steps check (data/port/steps_thu.json):

  | step | port | MLX, TF32 off | rel |
  |---|---|---|---|
  | 1 | 5.343394 | 5.342842 | 1.03e-4 |
  | 2 | 4.851293 | 4.858737 | 1.53e-3 |
  | 3 | 4.978863 | 4.971444 | 1.49e-3 |
  | 4 | 4.796092 | 4.796075 | 3.6e-6 |
  | 5 | 3.777776 | 3.773581 | 1.11e-3 |

  Max 1.53e-3 is inside the 2e-3 line, so PASS. It is ~10x the 1-layer test's 1.2e-4: at 5 layers, A 64 and an active
  selector, bf16 roundings and top-16 near-ties add up (diagnosis 2b's ~1e-2 per-gradient level). No systematic sign.
- Run 2, MLX's default TF32, vs L1-20's logged losses: 5.343977 / 4.695799 / 4.796175 / 4.645471 / 3.723465, equal to
  the log at every step (rel 0). MLX's TF32 trajectory is deterministic, and the log reproduces.
- TF32 on vs off, same code and data: steps 2-5 differ by -3.4 / -3.5 / -3.1 / -1.3 %. That is the original 'PORT
  steps FAIL' gap, now shown at full scale on MLX alone.
- So: the PyTorch port equals MLX's trainer in true fp32 at full scale. MLX's default TF32 explains the whole FAIL.
  L1-20 (and every MLX run here) trained with TF32, and that is reproducible. The TF32-off retrain stays parked (SEED).
- Data: data/port/fp32_fri4.log, steps_mlx_{fp32,tf32}.{json,log}.

### MTP3BIT (the orchestrator 19:4x; CPU from stored logs, data/mtp_3bit_vs_4bit.txt): does the 3-bit trunk hurt the MTP head's drafting? No, not measurably
- Public rig for comparison (github.com/crgregersen/m10dis, docs/RESULTS.md, as quoted by the orchestrator): this model's
  own MTP head on the FP8 model, per-position acceptance at drafts 1-3: natural text 70 / 46 / 26 %, one HumanEval
  prompt 97 / 96 / 92 % (read as cumulative, P(drafts 1..k all right); conditional below too).
- Stock head, our 4-bit trunk (D-17 / D-22, D-05's prompts, MTPLX runtime): code conditional 0.98 / 0.97 / 0.95 =
  cumulative 98 / 95 / 90 %; IFEval 0.69 / 0.60 / 0.59 = 69 / 41 / 24 %; multi 0.77 / 0.70 / 0.65 = 77 / 54 / 35 %. Same as
  the FP8 rig on code and on natural text.
- Same head (b3), same 17 prompts, 4-bit REF2b (D-20b) -> 3-bit REF2-T (D-23, the 45 GB default), mtp7 (8-row checks, no
  cut), C 224, mean kept per check: code 7.24 -> 7.06, math 5.81 -> 5.78, chat 2.59 -> 2.60, agent 3.98 -> 5.19 (3 prompts),
  all 4.02 -> 4.09. Each trunk drafts against its own greedy text, so the continuations differ; no loss beyond noise.
- b3 on the 3-bit trunk per position (907 checks): code conditional 0.964 / 0.953 / 0.971 = cumulative 96 / 92 / 89 %;
  math 93 / 87 / 76 %; chat 69 / 40 / 23 %; agent 87 / 77 / 70 %.

### DRAFTOFF estimate (the orchestrator 19:4x, desk, no build): what it takes to get the drafter off the GPU's critical path
- On the path now: the drafter's block pass after each verify, ~3.5-4.5 ms of a ~34 ms 45 GB check (ZERO / Fable B);
  BigMoE's D8-NAX 10.4 -> 5.8 ms per code check, R57_DSUB ~-1 ms more.
- The only exact overlap is DRAFT-AHEAD (draft check n+1's block during check n's verify, assuming every draft is kept):
  the patch is exact (DA-1, sticky/draft_ahead.py, 281 / 281 identical); usable on 34 % of code checks, 43 % with the
  17th draft (DA-0). Ceiling 0.34-0.43 x ~4 ms = ~1.4-1.7 ms per code check (+4-5 % tok/s), less on chat (full blocks rare).
- It pays only if the speculative pass runs in GPU idle time (SSD waits): it needs >= ~4-5 ms of idle per check now that
  the drafter is ~4-6 ms (DA-1's "20-33 ms" was at the 10 ms drafter). Without idle both passes are bandwidth-bound and it
  nets ~0. More idle at 12 GB, so likelier there.
- Not viable: the drafter on the CPU / ANE in parallel (~30-60 ms per block pass, and it still needs the verify's result).
- Unmeasured lever: draft ahead at the predicted cut (the engine's confidence cut) instead of at a full block; DA-0's CPU
  replay can price its hit rate in ~15 min.
- Build cost: BigMoE's R57_DA wiring (a second command queue so the draft overlaps the verify's waits; discard on a miss)
  + one timeline measurement of GPU idle per 45 GB check. The surer lever stays a cheaper drafter (D8-NAX, DSUB, 4-bit
  weights + a short 4-bit-aware fine-tune: D7B-1), which stacks with draft-ahead.
- BigMoE's number (Thu 23:1x, quoted as given; C 208 FAST, medians): all-in svc 19.3 ms per check (p90 38.6; an upper
  bound on GPU idle, since svc includes the spin on loads), U3C 33.4; the drafter takes 10.0 ms; checks run 86-96 ms. So
  the 10 ms draft fits in the idle only if most of svc is real idle: plausible, not shown. If it fits, draft-ahead saves
  ~0.34-0.43 x 10 ms = 3.4-4.3 ms per check (~+4-5 %, the same share as estimated). A timeline of real GPU idle decides.

### DA-1 re-priced from Leap's step 0 (Fri 11-06 07:3x; the orchestrator's note; desk, no run): 0 on FLAT, was -1.4..-1.7
- Input: Leap's TABLE_350 'STEP 0, PART A', from FLAT-2's synced records. The warm 2-row check is GPU-bound: 47.8 ms of
  GPU work (plus head and embed) in a 50.3 ms unsynced check. So the GPU is idle <= ~2.5 ms per check, and FLAT-2's
  draft is 9.1 ms, serial.
- DRAFTOFF's condition (GPU idle >= the draft's time) fails: ~2.5 vs 9.1 ms.
- Fitting the draft into idle gaps only: drafting ahead on every check delays the check by >= 9.1 - 2.5 = 6.6 ms. It
  saves 9.1 ms only on hits (h = 0.34-0.43 of code checks), so the net is +2.7..+3.5 ms per check, i.e. slower. It breaks
  even only at h >= 0.73, the same bar if it is gated to the checks predicted to accept in full.
- The only way left is running the draft inside the check's under-filled kernels on a second GPU queue. Owner records
  show MLX's second GPU stream barely overlaps here:
  - D-COMMIT design D (the draft on a second stream beside the commit replay) hid 0.06 ms per cycle (KILL, 11-03).
  - NAX-OCC (two expert calls on two streams): KILL, "no overlap lever" (11-04).
  - Neither ran beside the dense pass itself (~60 GB/s, bubble-bound).
- Re-price: draft-ahead counts 0 on FLAT. Leap's 'with the draft' line is the pieces alone (~42-48 before the bench).
  On the real cycle with SSD waits it could still give up to h x 9.1 = -3.1..-3.9 ms per code check, if >= 9 ms of true
  GPU idle sits in the waits; BigMoE's svc 19.3 ms median is an upper bound. That idle shrinks as NEVER-WAIT and the
  flat cuts land.
- Built but not installed, exact: BP-9's screened draft head (R57_DSUB; PASS 11-04, propose -1.0 ms with D8-NAX, kept
  code x1.0000) exists only as Leap's data/r57_session_vocab.diff. FLAT-2's arms (R57_D7B=code + R57_D7BNAX=1, and the 45
  preset doesn't set it) drafted without it: ~-1 ms on FLAT if it carries to D7b-NAX. NewIdeas / BigMoE decide.

### T16-HE: T16 on HumanEval, engine A/B on v1' + wired: pre-registration (Sat 11-07 14:0x, before any run; NewIdeas PLAN 220 item 6)

- **Question.** Does T16 make the 45 GB release faster on HumanEval code? T16 is GOAL's guarded comment / docstring
  acceptance: a comment or docstring row whose draft is not the 125B's argmax is kept only if the 125B ranks the draft in
  its top 3 at p >= 0.1 p1. Module: BigMoE scripts/inert_span.py "T16" (4f09d77). On MBPP12 it measured K x1.035 median /
  x1.105 pooled (Sat 00:xx, BETWEEN); it was never run on HumanEval.
- **Arms.** BigMoE's engine at C 256, the 12 HumanEval of S1 / S6W, one process per arm, quiet lock, NO-SWAP >= 47.2 GB.
  - B = v1' + wired: data/stack/final_env.txt (dd7ad0f) + R57_WIRED=1, QOM sets humaneval. This is S6W's W arm (181.2
    tok/s whole decode).
  - T = B without R57_RELAX_SPAN / R57_RELAX_SPAN_RULE, plus R57_INERT=T16 and R57_INERT_LOG. Span B has to go because
    the screen asserts that INERT and SPAN are separate arms (r57_screen_fast.py:190).
  - Order B / T / T / B.
- **Caveat.** T also drops span B. Span B made 0 relaxed accepts on these 12 prompts last night (stackb), so dropping it
  only removes its host cost. The verdict is on the T config, the release candidate, not on T16 alone.
- **Line** (pair-mean geo WHOLE DECODE tok/s, T vs B, over the 12):
  - PASS: >= +3 % AND T's lifetime peak <= 45.0 GB AND median tokens per task T / B <= 1.05 AND no T prompt runs to the
    cap where B ended by itself. Then EvalSuite's S4-style gate vs NAX6-N (same scores in both orders + long code +
    MBPP), started by Mon 12:00.
  - KILL: < +1 %, OR T's peak > 45.0 GB, OR a runaway (a T prompt at the cap where B ended by itself).
  - BETWEEN otherwise (+1..+3 %): no gate by Tuesday.
- **Beside** (not the gate): median cycle (ms), full period, check 0, kept per check (geo and pooled), tokens per task, rows
  kept by T16's rule (INERT log), and tokens identical per prompt T vs B next to the floor B1 vs B2.
- **Prediction [P].** Kept x1.03-1.08: GOAL's HumanEval replay says x1.08 (x1.04 with the gain halved), and MATCH g2
  already takes the copied docstrings. Cycle +0..+1 ms of host work. So whole decode +1..+6 %; PASS is roughly a coin toss.
- **Cost.** ~25 min of quiet GPU (4 arms at C 256, MEM ~43 GB). No download.
- **Runner.** BigMoE: its engine, its data folder, and its freeze + commit for EvalSuite's provenance. MTP Head reads the
  result, writes the RESULT here and the TRIED line, and hands a PASS to EvalSuite and NewIdeas.

#### RESULT (Sat 11-07 14:31-14:37, BigMoE scripts/t16he_ab.sh 7bc901e on S5's snapshot d8eb2e4, data a03e25d): KILL. T16's +3 % kept is eaten by +1.9 ms of cycle

| arm | whole decode tok/s (passes) | full period | median cycle | kept per check, geo / pooled | tokens per task | peak |
|---|---|---|---|---|---|---|
| B = v1' + wired | 173.2 (172.3 / 173.8) | 168.0 | 63.6 ms | 10.56 / 10.75 | 246 | 42.9 GB |
| T = B - span B + T16 | 173.2 (169.2 / 176.6) | 167.8 | 65.5 ms | 10.89 / 11.11 | 246 | 42.9 GB |

- **Line:** T vs B whole decode -0.0 %, so KILL (< +1 %). My re-derivation from the per-prompt logs gives -0.01 %, with 6 of
  12 prompts faster. Tokens per task T / B 1.000 (geo 0.999), no runaway, peak 42.9 GB. Full period -0.2 %.
- **Beside:** T16 kept 47 rows by its rule over both T passes; 2,912 rows sat in comments or docstrings. Tokens identical T
  vs B: 3 and 4 of 12, against a B1-vs-B2 floor of 8.
- **Reading.** Kept x1.031 geo / x1.033 pooled, at the low end of the prediction (x1.03-1.08). But the median cycle grew
  +1.9 ms (+3.0 %), so the net is zero.
- **Where T16's host time goes** (r57_screen_fast.py 1127 / 1358-1404 + inert_span.py; read, not profiled):
  - an extra GPU sync per check, for the draft's logit margin and rank on all 16 rows (a 16 x vocab compare);
  - two Python scans per check: pre_block before the verify, classify after it. Each copies the CodeScan and feeds the
    draft text one character at a time;
  - three rounds of per-token detokenization.
- **A cheaper tagger with the same decisions** could remove most of that:
  - run classify first, and sync only when a row has a rule mode;
  - use an id -> text table;
  - skip the scan when the state is code and no draft row holds '#' or a quote.
- **Its ceiling is the kept gain, ~+3 %.** That is borderline on this line, and the arm is still lossy (an S4-style gate
  plus a CERT top-up). Not worth it by Tuesday unless the exact-safe levers fall short by <= 3 %.

### CHAT150: what drafting can do for chat (Sat 11-07 23:4x-23:5x; CPU, sticky/chat_ngram.py -> data/chat150_read.txt; the orchestrator 23:4x + NewIdeas PLAN CHAT 150 items CH3 / CH4)

**Bottom line.** Nothing on the drafting side prices >= x1.5 on chat. The best exact stack by Tuesday is ~x1.10-1.15
tok/s. And 150 tok/s on chat is a check-cost problem, not a drafting one: with today's chat check costs, a drafter
that never misses would reach only ~114 tok/s at 16 rows with the serial MTP draft, or ~150 with a one-pass draft.

- **Chat's check cost** (BigMoE N45, Sat 23:25, 45 GB C 256, 12 IFEval x 2 passes, checks >= 2, the engine's own
  per-check records): the wall is fixed cost + verify per row + the serial MTP draft (2.59 ms per row).
  - Exact config: median wall 61.0 ms at a median 3 rows. Fit: 47.0 ms + 3.25 ms per row + the draft. One row alone
    costs 56.2 ms.
  - Lossy config: median 48.5 ms. Fit: 33.1 ms + 4.35 ms per row + the draft. One row costs 45.1 ms.
  - Kept / rows = 0.80 in both, so the 0.3 cut already spends rows well. The fixed part is 75-85 % of a chat check.
- **Replays.** Head b5 (the shipped chat head), teacher-forced on the 30 IFEval + 24 multi held-out answers (D-24's
  data), walked with D-17's loop. The shipped rule (K15, cut 0.3) gives IFEval 2.34 / multi 2.54 kept per check, and
  36.2 / 38.3 tok/s [P] (the engine measured 39.9 on its 12 IFEval).

| lever | IFEval kept / check | IFEval tok/s [P] | multi tok/s [P] | status |
|---|---|---|---|---|
| shipped MTP chain, cut 0.3 | 2.34 | 36.2 | 38.3 | today |
| cut re-tuned to 0.2 | 2.48 (x1.06) | x1.015 | x1.009 | exact, a constant |
| CH3: head 2x faster, best cut 0.15 | 2.57 (x1.09) | x1.095 | x1.083 | exact; needs a faster MTP step (kernels) |
| CH3: head 4x faster, best cut 0.15 | 2.57 | x1.138 | x1.128 | exact |
| prompt lookup alone (match >= 1-3) | 1.11-1.19 | x0.48-0.59 | x0.44-0.52 | KILL |
| prompt lookup + MTP hybrid (match >= 2..6, caps 4 / 8 / 15) | 2.18-2.34 | x0.90-0.996 | x0.89-0.994 | KILL |
| the same, other answers as a corpus (SuffixDecoding-style) | 2.29-2.32 | x0.95-0.98 | x0.93-0.96 | KILL |
| oracle pick of lookup vs MTP per check | 2.37 | x1.012 | x1.000-1.004 | ceiling of any selector |
| tree leaves (rank-1 / 2 at kept depths with p < 0.8), today's head | 2.50-2.92 | x1.009-1.044 | x0.98-1.02 | needs a GDN tree verify (not built) |
| tree leaves, head 2x faster | 2.50-2.92 | x1.07-1.11 | x1.05-1.09 | same |

- **Relaxed acceptance on chat** (lossy; chat runs exact today). EvalSuite's RELAX (11-05): the chat arm E kept +7.2 %
  (2.67 vs 2.49), at IFEval 87 vs 90 (BETWEEN). The stronger F / G rules failed IFEval (-5 / -6). So relaxing adds
  ~x1.07 at most within quality.
- **Retuning the head on chat** (D-22, [K]): b4, with twice the chat windows, moved IFEval 2.43 -> 2.51 (n.s.). AUF (b5)
  gave +0.3 %. The wall is the head's depth-1 accuracy: IFEval 0.73-0.76, multi 0.78-0.80, code 0.98.
- **CH4: an EAGLE-3-style head (target features + one trained layer) by Tuesday: no.**
  - Its features must come from the 125B. Free cloud can't hold it, and the Mac captures at prefill speed: 98-124 tok/s
    (N45), i.e. ~2.5 h of quiet GPU per 1M tokens.
  - The 5 taps take 25.6 KB per token, i.e. 25.6 GB per 1M tokens. The disk is at 88 GB free, under its 100 GB floor,
    so only online training (capture and train in one process) would fit.
  - One night is therefore ~3-4M tokens, a few % of the chat data EAGLE-3-class heads train on. It would start from a
    target-trained one-layer head (the native MTP) that a 2x chat-data tune moved by 1 %.
  - [G] kept +0-5 % for a night of work, not the 3+ that CH4 asks for.
- **What 150 needs.** At K 3 (the realistic best: the cut re-tune + trees + CH3) the check must cost ~20 ms, against
  48.5-61 today. At NewIdeas' K 3.75 it must cost ~25 ms, and the serial MTP draft at today's speed would take ~11 ms of
  that. So 150 needs the check's fixed part cut ~2.5x and a head ~4x faster, both by Tuesday.
- **Data:** data/chat150_read.txt; script sticky/chat_ngram.py (inputs data/d24/{ifeval,multi}_b5.npz, BigMoE
  data/r57/nie*/nil*_256_checks.json).

### F2A feasibility: an EAGLE-3-style chat head on the 125B's own features (Sun 11-08 00:xx; CPU, sticky/f2a_headroom.py -> data/f2a_headroom.txt + desk; NewIdeas PROGRAM CHAT 150, 00:2x)

**Answer.** F2a can be built by Tue-Wed, but the realistic gain is x1.1-1.3 in kept tokens on IFEval [G], not the x1.8
that 150 needs (K 2.5 -> 4.5).
- With F1 (NewIdeas' ~30 ms check), a modest better head gives ~90 tok/s on chat.
- A head that fixed 80 % of the misses where the 125B is sure gives ~105.
- 150 needs a near-perfect head (depth-1 0.95 on IFEval) plus F1, and even that lands at ~133-140 [P].

**Headroom.** Simulated better heads on D-24's teacher-forced chain (head b5; 30 IFEval + 24 multi held-out answers).
Each miss flips to a hit with a probability set by how sure the 125B is of that token (its top-1 p at that position,
data/d02a), and a flipped token's head p is raised to >= 0.9. Walk and costs as in CHAT150. F1 = 30 ms at 3 rows +
3.25 ms per extra row + a fused 0.6 ms per drafted row.

| head | IFEval depth-1 | kept / check, cut 0.3 (cut 0.15) | tok/s, today's check | tok/s with F1 |
|---|---|---|---|---|
| today (b5) | 0.756 | 2.34 (2.57) | 36.2 | 74.0 |
| modest: fixes 50 / 25 / 10 % of misses at the 125B's p1 >= 0.8 / 0.5-0.8 / < 0.5 | 0.827 | 2.93 (3.34) | 43.2 | 89.0 |
| strong: 80 / 50 / 20 % | 0.886 | 3.56 (4.24) | 50.2 | 104.5 |
| all sure tokens: 100 / 50 / 20 % | 0.906 | 3.90 (4.75) | 53.8 | 114.1 |
| near-perfect: 100 / 80 / 50 % | 0.950 | 4.65 (5.88) | 61.3 | 133.1 |

Multi runs 5-10 % higher on every row.
- **Where today's head misses** (depth 1, IFEval): 30 % of misses are at tokens the 125B gives p1 < 0.5, 30 % at
  0.5-0.8, 16 % at 0.8-0.95, and 24 % at >= 0.95. A head that hit every token the 125B is >= 0.8 sure of would reach
  0.854 (multi 0.890).
- **Caveat:** the stored answers are not the 125B's own greedy text. Its top-1 matches 82 % of IFEval tokens, 93 % multi
  and 77 % GSM8K. So hit rates are against the answer text. The replay's 2.34 kept is close to the engine's 2.38 (N45,
  12 IFEval).

**Expected p for F2a [G]: depth-1 0.78-0.83 on IFEval**, i.e. the modest row, x1.15-1.25 kept.
- The native MTP layer is already a target-trained one-layer head. Every chat tune so far moved it <= 3 % (D-22 b3-b5;
  twice the chat windows gave +1 %), and the DFlash2 chat tunes (D-15a, E1, E5) were null.
- F2a's new ingredient is volume: ~3-10M tokens in a night vs ~0.1M before, 5-7 doublings. At +1-4 % per doubling that
  is +5-30 %.

**Build, if NewIdeas keeps it:**
- **Head.** Initialize from the native MTP layer (one decoder layer, width 2560). Add an EAGLE-3 fusion layer over 3 taps
  (low / mid / high) and train it with training-time test (D-22's TTT code).
- **Vocabulary.** The 125B's vocab is 248,320, so its output layer is 636M parameters.
  - That layer likely dominates today's 2.6 ms draft step and the training cost.
  - The fused <= 0.6 ms step of NewIdeas' F1c needs a draft vocab of roughly <= 96K at 4 bits [P].
  - A 32K vocab covered only 53 % of multilingual answer tokens (FR-Spec desk KILL), so check coverage first.
- **Features.** 3 taps x 2560 fp16 = 15.4 KB per token, plus the 125B's top-32 (192 B).
  - At PFX's 2M tokens/hour that is ~31 GB/hour. The disk has 88 GB free, under its 100 GB floor, so nothing can be
    stored.
  - So training must be ONLINE: one MLX process holds the 125B's prefill path (~42 GB) and the head trainer (~3-5 GB),
    alternating prefill chunks and training steps, and keeps only checkpoints.
  - That is ~47-50 GB, a quiet whole-Mac job.
- **Throughput [P].** Training costs ~25-30 GFLOP per token: the output layer, times the TTT steps. At an effective
  ~5 TFLOPS that is ~0.6M tokens/hour of training. Sharing the GPU with the 2M/hour capture gives ~0.3-0.7M tokens/hour
  end to end, so ~3-7M tokens in a 10-hour night.
- **Data.** Local chat text is far short: IFEval prompts (541, no answers), MGSM, the drafter-data log's 175 multilingual
  answers (on-policy) and D-22's 147 windows.
  - A chat corpus has to be downloaded (e.g. UltraChat-200k or ShareGPT, ~1-2 GB), which needs Kamen's OK via the
    orchestrator.
  - Plan: teacher-force the corpus's answers, with labels = the 125B's own argmax per position (off-policy text,
    on-policy labels).
- **Timeline.** Sun: build + a 1-hour smoke test (the loss falls; held-out depth-1 on D-24's IFEval / multi). Sun night:
  the ~10-hour online run. Mon: read it on D-24's replay, then an engine A/B. Tue: decide.
- **Risks:** memory (two big things in one process), GPU sharing, off-policy text, and re-tuning the cut (D-23's lesson).

**F2b** (a small same-tokenizer dense drafter): my data agrees with NewIdeas' x0.9-1.15. DFlash2 (5 layers) keeps
0.59-0.65 per token on IFEval vs the native MTP head's 0.756, and small independent models start below the native head
on chat.

### D1: the chat draft step, measured (Sun 11-08 00:0x; quiet locks, sticky/mtp_step_bench.py -> data/mtp_step_bench{,_p4}.txt; shared, sticky/d1_variants.py -> data/d1_variants.txt; CHAT150 strike)

**Bottom line.** The cheapest D1 by Monday puts the head's trained parts at 4 bits and drops the per-step sync. Each
drafted row goes 2.63 -> 2.14 ms (-19 %), with chat hits unchanged: about -1.5 ms per chat check, +3-4 % chat tok/s
[P]. The -5..-6 ms the board asks for would need a cheaper 248K output layer, and every cheaper output layer I tried
loses chat hits.

- **Step anatomy** (head b5 as the engine installs it, the engine's shapes, 512 context rows, 40 reps; drafts identical
  between loops):
  - Shipped loop: 2.63-2.82 ms per drafted row.
  - Without the per-step host sync (the K steps as one graph): 2.43 ms, x1.08.
  - The 4-bit 248K output layer (with the hyper-connection mixer): 1.49 ms. Re-quantized to 3 bits 1.11 ms, to 2 bits
    0.85 ms.
  - The MTP layer itself: ~1.1 ms.
  - 4-bit trained parts: the 19 bf16 modules mtp125.install puts in (fc, q / k / v / o, router, shared expert) at 4 bits,
    group 64. The shipped loop drops to 2.34 ms, and to 2.14 ms without the sync.
- **Chat hits per variant** (teacher-forced chain, 30 IFEval + 24 multi held-out answers, depths 1-6; kept with the
  shipped cut, K6):

| variant | IFEval depth-1 | IFEval kept (rows) | multi depth-1 | multi kept (rows) |
|---|---|---|---|---|
| shipped (bf16 trained parts, 4-bit output) | 0.756 | 2.301 (2.88) | 0.784 | 2.507 (3.18) |
| output layer 3-bit | 0.728 | 2.190 (2.74) | 0.762 | 2.402 (3.03) |
| output layer 2-bit | 0.661 | 1.989 (2.62) | 0.700 | 2.130 (2.80) |
| trained parts 4-bit | 0.756 | 2.427 (3.33) | 0.781 | 2.630 (3.75) |
| trained parts 4-bit + output 3-bit | 0.736 | 2.343 (3.18) | 0.760 | 2.529 (3.59) |

- **Reading.**
  - The output layer can't be cut. 3-bit loses 5 % of kept for 0.4 ms a row, and 2-bit loses 14 %.
  - A 64K vocabulary plus the prompt and answer tokens covers 97 % of IFEval tokens but only 67 % of multi tokens (CPU
    check on the same answers), so a reduced vocabulary doesn't work for multilingual chat either.
  - 4-bit trained parts keep the hits. They also raise the head's confidence a little, so the 0.3 cut lets more rows
    through: kept +5 % at +0.45 rows.
  - With chat's fitted check cost, 4-bit parts alone come to +2.8 % tok/s [P], and +3.9 % without the sync.
  - Drafts can't change chat outputs: chat verifies exactly.
- **Engine change** (BigMoE's screen, behind a flag): after `_M.install(...)`, quantize the head's bf16 Linear modules
  at 4 bits, group 64 (the predicate in sticky/mtp_step_bench.py, D1_P4). Optionally, run the chain's steps
  without the per-step `.item()`: evaluate pairs of steps and cut post hoc.

### F2A night: pre-registration (Sun 11-08 00:2x, before any run; NewIdeas' quiet whole-Mac window Sun 23:00 -> Mon 07:00, QUEUE c16f510)

- **Goal.** A chat MTP head that keeps more per check on chat: b5 continued on ~1.5-2M tokens of the 125B's chat features.
- **Data.**
  - Text: one shard of UltraChat-200k (MIT; English multi-turn) + Aya (Apache-2.0; 17 languages incl. the multi set's
    bg de fr es ru zh ja ar), ~2 : 1 in tokens. Rendered with the 125B's chat template, thinking off
    (sticky/f2a_data.py -> data/f2a/seqs.npz).
  - Labels: the 125B's own top-32 at every position (sticky/f2a_capture.py). So the labels are on-policy and only the
    text is off-policy.
  - On-policy eval: 24 of the 125B's own greedy IFEval answers (EvalSuite reference decodes), captured first.
- **Recipe:** D-22's trainer from b5, depth 6, AUF as b5, lr = b5's x 0.5, warm-up 20 + cosine, one pass over the ring
  (sticky/f2a_train.py).
- **Line** (f2a vs b5; kept per check with the shipped cut t 0.3, K6; teacher-forced chain; sticky/f2a_eval.py):
  - PASS: held-out kept >= x1.08 (geo of D-24's 30 IFEval and 24 multi, which neither head trained on) AND on-policy
    IFEval kept >= x1.05. A PASS goes to BigMoE's engine A/B on Monday (12 IFEval, chat whole decode, exact).
  - KILL: held-out < x1.03.
  - BETWEEN otherwise.
- **Prediction [G]:** held-out x1.05-1.20 (the modest row of 'F2A feasibility'). PASS ~35 %.
- **Phases and guards** (turns/f2a_night.sh):
  - Capture until the ring holds 34 GB or 03:45. The expert pool is C 256 if NO-SWAP >= 46 GB, else C 192 / 128.
  - Train until 06:10. Evaluate, commit, and release the lock by 06:50.
  - Each phase stops cleanly (keeping its checkpoint) on swap +2 GB or free disk < 25 GB (Kamen keeps 20 GB for
    swap). The scripts wait while PAUSE exists.
- **Engine A/B (BigMoE chs2, Sun 00:5x-01:0x, 12 IFEval at C 256, one pass each; data/r57/x2*_256_*):**
  - On the exact config E, v1 is exact (12/12 equal to the AR reference) and neutral: drf 8.7 -> 8.4 ms per check, rows
    3.14 -> 3.41, kept 2.53 -> 2.57, whole decode 36.6 -> 36.9 tok/s.
  - On the lossy config L, v1 is a KILL: drf 9.1 -> 16.2 ms per check, kept 2.62 -> 2.55.
  - Cause (BigMoE's split, confirmed in the code): my predicate also put 8 non-trained bf16 Linears at 4 bits. These are
    the head's hyper-connection mixers (GatedResidual input_mix down / up + block_inject, and the output mixer). L's HC8
    read path takes another route when they carry scales (qwen4_exp.py GatedResidual, line 1166).
  - v1b quantizes only the 11 trained modules (predicate + `"hyper_connection" not in path`). It goes to BigMoE's chs4
    on L after CERT. Expected: E's numbers, i.e. whole decode +0.5-1 % [P].
  - A no-sync / pairwise chain isn't worth it: a chain cut at an odd step wastes a ~2.3 ms step, more than the ~0.2 ms
    saved per sync.
- **Amendment 1** (Sun 01:2x, before any night run; pipeline smoke 01:16-01:18, data/f2a/smoke/):
  - **On-policy eval set changed.** All 24 planned eval answers (EvalSuite's reference IFEval decodes) are in b5's
    training windows: 92 of the 95 decodes are in data/d22c. So the night instead first generates the 125B's own greedy
    answers (<= 192 tokens) for 24 IFEval prompts that neither data/d22c nor dec_ifeval holds, and uses those as the
    on-policy eval. The line is unchanged.
  - **Smoke results.**
    - Capture: 112-143 tok/s at a 16-slot pool, peak 20.9 GB.
    - Training: 0.3 s per step after its first, peak 22.8 GB. The lr is b5's 1e-5 x 0.5 = 5e-6; 69.1M trainable.
    - Eval: b5 reproduces D1's numbers (IFEval kept 2.301, multi 2.507). The template matches EvalSuite's IFEval prompts,
      and tokenizers 0.23.3 is in both venvs.
  - **Runner rebalanced, since capture, not training, is the bottleneck.**
    - Capture to 05:15 or a 50 GB ring (~2.4M tokens), then 2 epochs of training.
    - The pool is sized from the measured peak (~21 GB + 0.118 GB per slot, gated at peak + 4): C 256 at NO-SWAP >= 54,
      C 192 >= 46, C 128 >= 38, C 64 >= 31, else C 16.
- **Amendment 2** (Sun 11-08 08:2x, before any night run; runner v2 re-armed at 08:24):
  - **Disk.** Free disk is 54 GB (Sun 08:14). So a 50 GB ring can't fit above Kamen's 20 GB swap reserve.
    - The v1 runner also had a flaw. Once capture filled the disk to the 25 GB guard, the same guard would have
      stopped the training and eval phases within 15 s, because the ring stays on disk.
    - Fix: capture now stops itself at 28 GB free (F2A_DISK_MIN), 3 GB above the guard.
    - Ring 1 = ~26 GB = ~1.2-1.3M tokens, if the disk stays at 54 GB free.
  - **Stage 1 = the pre-registered run.** Capture to 02:00 or the disk floor; 2 epochs to 03:00; eval by 03:15;
    commit. The line and the verdict are unchanged: f2a vs b5, data/f2a/f2a_eval.txt.
  - **Stage 2 (new, secondary).** Ring 1 fills by about 02:00, so stage 1 ends near 02:45 and the window would idle
    for 4 h. If stage 1 ends by 03:30:
    - delete ring 1, capture ring 2 (the next corpus sequences) to 05:10;
    - train f2a for 2 more epochs on ring 2 (same lr, schedule restarted) -> f2a2;
    - eval f2a2 vs b5 and vs f2a -> data/f2a/f2a2_eval.txt.
  - **Stage 2 lines.**
    - f2a2 vs b5: the same PASS / KILL line.
    - Data scaling, f2a2 vs f2a on held-out kept (geo): >= x1.02 = "more data pays", < x1.005 = "flat", else "small".
    - "More data pays" makes faster capture worth building for the next night and for Qwen 4, e.g. a packed form of
      BigMoE's layer-major prefill (PF-A runs one sequence per call; ours are ~470 tokens).
    - Monday's engine A/B gets whichever head passes with the higher held-out kept.
  - **Guards.** A ring under 200 windows isn't trained on (NO DATA). The eval skips an empty window set instead of
    failing.
  - **Checked.** A dry run of the runner with stub programs (both stages, and the NO DATA path). The three Python
    scripts parse. Unchanged since the smoke: the capture, training and eval code paths, apart from the disk floor, the
    init name and the output name.

### D1 v1b in the engine: KILL, and D1 is closed (Sun 11-08 08:2x; BigMoE CHS4 07:42, data/chs/chs4_report.txt arms y0 / y0p4)
- Setup: 12 IFEval prompts at C 256, one pass per arm, quiet.
- On the lossy config L, v1b (only the 11 trained modules at 4 bits) is a KILL:

  | | whole decode tok/s | kept per check | rows | drf ms per check | cycle ms |
  |---|---|---|---|---|---|
  | L (y0) | 43.8 | 2.54 | 3.22 | 8.7 | 57.6 |
  | L + v1b (y0p4) | 40.2 (-8.2 %) | 2.57 | 3.45 | 14.9 | 63.2 |

- So leaving the hyper-connection mixers in bf16 didn't fix L.
  - Kept and rows rose, as the replay predicted.
  - But L's draft costs +6.2 ms per check.
- Untested hypothesis: L's draft path takes a slower route for 4-bit weights in these modules, for example a
  dequantize every step. On E the same weights were neutral.
- Not worth chasing: the whole D1 lever was +0.5-1 % at best.
- **D1 closed.** The draft step stays bf16 in both configs.
- The CHAT150 picture is unchanged. Chat acceptance is the wall: BigMoE CHS5 08:13 puts 150 tok/s at ~6 kept per check
  at a 40 ms cycle, against 2.4-2.5 today.

### PAPER SPEED: whole request and first token for the live drafter A/Bs (Sun 11-08 16:5x; CPU on the existing rows, no new runs; sticky/paper_speed.py -> data/paper_speed.txt)
- Why: the papers' rule 2 asks for whole decode, whole request and first token with the lifetime peak.
- Definitions:
  - whole decode = tokens / decode seconds, the rows' own tok_s;
  - whole request = tokens / (decode + prefill seconds);
  - first token = the median prefill seconds.
- Each selection reproduces its read-out's decode numbers: D-20b 15.95 / 20.30, D-23 29.43 / 28.65, D-23d chat 23.11 /
  28.78.

| A/B | arm | whole decode | whole request | first token (median) | lifetime peak |
|---|---|---|---|---|---|
| D-23d chat (19) | block drafter | 23.11 | 18.56 | 1.91 s | 37.9 GB (logs/d23d.out) |
| D-23d chat (19) | MTP K15 + cut | 28.78 (+24.5 %) | 21.95 (+18.3 %) | 1.88 s | same run |
| D-20b chat (6) | block drafter / MTP K7 | 11.38 / 15.07 | 9.18 / 11.61 (+26.5 %) | 3.87 / 3.93 s | not logged |
| D-20b whole set (17) | block drafter / switch | 15.95 / 20.30 | 7.11 / 6.42 | 3.94 / 3.94 s | not logged |
| D-23 whole set (18) | block drafter / switch | 29.43 / 28.65 | 7.09 / 7.04 | 3.80 / 3.82 s | 42.4 GB |

- On whole sets the agent prompts' first tokens (median 62-89 s, prefill of long agent contexts at the time) dominate the
  whole-request time, so whole request barely sees the drafter there. D-20b's switch reads lower on whole request
  (6.42 vs 7.11) only because its agent rows' prefills ran longer; the prefill doesn't use the drafter.
- On chat (short prompts) the drafter's decode gain carries to whole request at about three quarters of its size.
- **Amendment 3** (Sun 11-08 19:2x, before any night run; ~/Papers/MONDAY.md moves the release to Monday evening; runner v3
  re-armed at 19:23):
  - **Stage 1 only tonight**, 23:00 to about 03:15. Its line and verdict are unchanged.
  - **Stage 2** (amendment 2) moves to after the release. Tonight it runs only if data/f2a/FULL_NIGHT exists when stage 1
    ends, which would be Kamen's call relayed by 22:45.
  - **End of stage 1:**
    - peer unlock;
    - then data/f2a/STAGE1_DONE (time + verdict);
    - then "F2a night DONE" as the last line of data/f2a/night.log.
    - EvalSuite's chain starts on either.
    - Every exit after 23:00 does the same, through a trap (TERM, NO LOCK, NO DATA, the normal end).
  - **Checked:** dry runs with stub programs: stage 1 only (3 phases), the FULL_NIGHT flag (6 phases), NO DATA, and a TERM
    mid-phase (the live phase stopped, unlock, DONE, last line).
- **Amendment 4** (Sun 11-08 19:5x, before any night run; Kamen's go via the orchestrator + the queue order; re-armed at 19:52):
  - The engine's 12-min PF-D run goes first: its lock is queued at 23:00:00, after the apps close.
  - F2a requests its quiet lock at 23:01, queues behind that ticket, and starts when it unlocks (~23:15).
  - Stage-1 deadlines +15 min: capture 02:15, train 03:15, eval 03:30. STAGE1_DONE and the closing line are unchanged, at
    ~03:30 at the latest.
  - The line and the verdict are unchanged.

#### F2A night RESULT (stage 1, Sun 11-08 23:01-23:48; runner v3; data/f2a/f2a_eval.txt, capture.log, train.log, night.log): KILL by its line, at 0.10M tokens, 8 % of the planned ring. The swap guard stopped capture after 27 min.
- **Run.**
  - Quiet lock 23:01:58. Capture at frac 0.25 (C 128) from NO-SWAP 45 GB, while the apps were closing.
  - The 24 on-policy eval windows took 334 s.
  - The swap guard fired at 23:29: swap 5,295 -> 7,408 MB, MLX peak 37.4 GB. The ring then held 241 windows,
    101,070 tokens, 1.9 GB.
  - Training: 2 epochs = 482 steps in 1,076 s, peak 27.4 GB. Eval: 45 s.
  - Lock released 23:48:18, then STAGE1_DONE and the closing line.

| set (kept per check, cut 0.3, K6, teacher-forced) | b5 | f2a | ratio | depth-1 hit b5 -> f2a |
|---|---|---|---|---|
| held-out IFEval (30) | 2.301 | 2.334 | x1.014 | 0.756 -> 0.758 |
| held-out multilingual (24) | 2.507 | 2.540 | x1.013 | 0.784 -> 0.790 |
| on-policy IFEval (24 fresh greedy answers) | 2.444 | 2.450 | x1.002 | 0.770 -> 0.777 |

- **Verdict, by the pre-registered line: KILL.** Held-out geo is x1.014, under the KILL line of x1.03; on-policy is
  x1.002.
- **What it does and doesn't say.**
  - It trained on 0.10M tokens: ~8 % of amendment 2's ~1.2-1.3M, and ~5 % of the pre-registration's 1.5-2M.
  - So it kills F2a at 0.1M tokens. It is not a test of the data-rich plan.
  - The sign is small and positive on held-out chat (+1.3-1.4 %, depth-1 +0.2..0.6 points) and flat on-policy.
  - Prediction was x1.05-1.20 at full data, PASS ~35 %.
- **The capture rate is the planning number.**
  - 76-80 tok/s at C 128 over 241 windows. The smoke's 112-143 came from its first 2 windows only.
  - So 1.3M tokens need ~4.6 h of quiet capture. Even without the guard, 02:15 would have ended the ring near ~0.9M.
- **Memory lesson.** The gate (measured peak + 4 GB under NO-SWAP) was too thin while the apps were still closing:
  peak 37.4 against 45, and swap +2 GB in 27 min. A re-run should gate at peak + 8, or use C 64.
- **Next, after the release, not before.**
  - A full-window capture at a swap-safe pool (~4.6 h for 1.3M tokens), then the same line.
  - Then stage 2 (data scaling), as in amendment 2.
- **The head.** ckpt/d22_f2a.safetensors, md5 4fa8f7216edb946ab9e22e18a9199388. Kept for the record, not for
  shipping: nothing ships.

### NUMBER CHECK notes (Mon 11-09 07:4x; the papers' number check of DRAFTER v1.1, re-derived from raw rows)
- **CHAT150's chat check-cost fit.** The estimator was least squares of (cycle wall - the measured draft time) on rows,
  over checks >= 2 (sticky/chat_ngram.py line 41): 47.0 + 3.25 ms per row.
  - A plain least-squares fit of the check time alone, on the same 1,280 checks (first 3 per prompt skipped), gives
    44.9 + 4.00 ms per row. Its 16-row ceiling is ~106 tok/s.
  - So the never-miss ceiling is ~105-115 tok/s with the serial draft (~135-150 one-pass) [P]. Both fits extrapolate
    from checks of mostly 2-4 rows.
  - The fixed share (~74-85 % at 2-4 rows) holds under both.
- **'D1 v1b in the engine' / the D1 engine A/B.**
  - The draft times there (v1 9.1 -> 16.2, v1b 8.7 -> 14.9 ms) are the run report's means over prompts. Pooled per check
    they are 8.6 -> 14.4 and 7.9 -> 13.4 ms (the same +68-70 %).
  - The peaks 42.40 / 42.30 GB are the report's 0.1-GB rounding of 42.37 (L) / 42.28 (L + v1b) GB.
- **'D-20b SWITCH45 confirmation'** (number check, 07:51): its per-type gains (chat +26.0 %, math +18.3 %, agent +27.0 %)
  are means of per-prompt rates. Pooled from data/d20_runs.jsonl (C 224, the 17 prompts with both arms, as in PAPER
  SPEED) they are chat +32.4 %, math +19.8 %, agent +42.0 % and code -14.7 %. The whole-set +27.3 % is pooled in both.
- **Withdrawn-headline line, "180.3 tok/s"** (number check, 10:36): DRAFTER section 4 said the registered steady metric
  "is 142.1 on whole decode". Re-derived from BigMoE's raw records (BM:data/r57/s1n1_256_runs.json + s1n2_256_runs.json,
  24 runs, gate+t, block drafter, n 256): pooled whole decode sum(n) / sum(n / tps) = 142.77 tok/s; the geo mean of the
  per-prompt rates is 142.10 (the correction note's 142.1); the geo mean of the steady rates is 180.25 (the 180.3).
  DRAFTER now says "142.8 on whole decode, pooled (... the correction note's 142.1 is a geo mean)", as MAIN and KERNELS do.
- **Appendix C row 19, MATCH** (number check, 12:05): the engine's x1.08 / x1.115 is kept per check, not speed. Re-derived
  from BM:data/r57/st{b,bb}_224_runs.json (MATCH on: copy.chg 422) vs st{s,sb}_224_runs.json (off), 12 HumanEval x 2
  passes: all tokens / all checks 11.158 vs 10.013 = x1.1144 (K_SCOREBOARD's x1.115 divides the rounded 11.16 / 10.01);
  geo over prompts x1.081. Whole decode pooled 125.19 vs 116.23 tok/s = x1.077 (geo x1.056 = the run report's +5.6 %).
  DRAFTER 3.5 and row 19 now say kept per check x1.08 geo / x1.11 pooled, and whole decode x1.077 pooled.

### F2A amendment 5 = F2b, the data-rich re-run (Wed 11-11 11:23, before any F2b run; runner turns/f2b.sh, DRY-tested on both paths)
- Why: F2a stage 1 was a KILL at 0.10M tokens, 8 % of the planned ~1.3M, so it did not test the data-rich plan
  ('F2A night RESULT').
- Same recipe and lines as the pre-registration: b5 continued (D-22 trainer, depth 6, lr b5 x 0.5, warm-up 20 + cosine),
  2 epochs over the ring. f2b vs b5 on D-24's held-out 30 IFEval + 24 multi, and on the 24 on-policy IFEval windows
  captured Sunday. PASS: held-out kept >= x1.08 AND on-policy >= x1.05. KILL: held-out < x1.03. A PASS goes to an engine
  A/B (12 IFEval, chat whole decode, exact) and notes/F2A_SHIP.md. Also reported: data scaling f2b vs f2a (more data
  pays >= x1.02; flat < x1.005).
- Data: Sunday's ring (241 windows, 0.10M tokens) is kept and the capture continues from cursor.json; ring budget 25 GiB
  (~1.3M tokens); --eval 0.
- Validity: a ring under 1,200 windows (~0.5M tokens, 5x Sunday's) is NOT A TEST: no training.
- Memory: the expert pool is gated at the measured peak + 8 GB. C 128 needs NO-SWAP >= 46 (Sunday's C 128 at NO-SWAP 45
  swapped +2.1 GB at 23:29); C 64 needs >= 37.
- Time: capture to END - 4 h 30 min; train to END - 25 min (2 epochs of a full ring take ~3.9 h at Sunday's 2.23 s per
  window-step, data/f2a/train.log); eval to END - 5 min. Needs a quiet whole-Mac window of >= ~6 h for a minimal test
  and ~9-10.5 h for the full ring (capture ran 76-80 tok/s at C 128; C 64 is not measured).
- Prediction [G]: held-out x1.02-1.08; PASS ~20 % (lowered from ~35 % after Sunday's x1.014 at 0.10M tokens).
