> (a quote from the project owner, kept private)
# breakthrough in the field to get that"). Started 21:33. Today's tables are ROUTE CAPS, not verdicts.
visible tok/s = 1/(1-F) x S x A / T_pass. Today (best measured): F ~0.27 (x1.36), S 1.0, A ~4 agentic / 6.8 code (MTP / adapted
drafter), T_pass ~38 ms (4-row, in RAM) ... ~100 ms (24 GB streamed). Byte floor per pass ~20-25 ms (dense ~3 GB + experts).

## The assumptions every published route shares, and how each could break
| # | shared assumption (what it caps) | a way to break it | prior art | cheapest decisive test | status |
|---|---|---|---|---|---|
| A1 | GREEDY argmax matching decides acceptance (A capped ~4-7 on agentic by near-ties) | lossless speculative SAMPLING (accept with min(1, p/q)): at near-ties acceptance = 1 - TV(p, q), not argmax agreement; Qwen's own card advises against greedy (loops) | speculative sampling is published (Leviathan, Chen 2023); "our agentic ceiling is a greedy artifact" is our claim to test | CPU: on stored top-K distributions of the 125B and a near-copy / the MTP head at the same positions, compare 1 - TV with argmax agreement on agentic + IFEval text | PROPOSED (needs distributions per position) |
| A2 | ONE sequential stream (S = 1) | Kamen's idea: write in the cheapest order, then sort; chunk form = skeleton, then parallel blocks sharing expert reads | SoT (2023), APAR, PASTA, Multiverse, Hogwild!; diffusion LMs | CPU on Kamen's 104 sessions (scripts/chunk_parallel.py) | **KILLED as a 350 route 21:33**: S_eff 1.27 corpus-wide (kill 1.3); file writes 2.1-2.2x (19 % of output) = a +10-20 % lever on Write-heavy turns |
| A3 | Every pass reads all dense weights (~3 GB) for 4-16 rows | amortize dense over MORE accepted tokens per pass: needs A up (A1, A6) or S up (A2) | wide verify (SpecExec, Sequoia) | - | coupled to A1 / A6 |
| A4 | Every pass loads each row's own experts (union grows with rows) | share experts across the pass's rows (budget), gated by confidence (UGB) | MoE-Spec, AcceptMoE, XShare / OEA (published); UGB gate (new) | TrainLab turn17: drafter-confidence gate under the strict rules | **IN FLIGHT** (oracle / post gates PASS; drafter gate decides) |
| A5 | Fixed weights: the drafter imitates the target; the target never changes | a draft-friendly target: fine-tune the 125B's router / MTP jointly so the target becomes more predictable at near-ties, without moving answers | self-distillation for speculation (DistillSpec, FastMTP) | local only; out of reach for the full 125B; the MTP head was tried (-4.6 %) | parked |
| A6 | One verify pass per drafted block, verified by the full 48 layers | verify easy blocks with fewer layers (early exit) or fewer experts and only re-verify fully where unsure | LayerSkip, Draft & Verify, CALM (non-exact) | same tf machinery as UGB: per-position KLD of a 24-layer early exit vs confidence | PROPOSED after UGB |
| A7 | GPU only; CPU / Neural Engine idle while the GPU runs | run the drafter on the Neural Engine / CPU in parallel with the GPU's verify (PEARL-style pipelining), so drafting is free and deeper | PEARL (parallel drafting), Apple ANE studies | measure MTP head time share of a pass (if < 10 %, kill) | PROPOSED, likely small |
| A8 | Text is generated token by token in reading order | (Kamen) any-order / diffusion generation | Mercury, Gemini Diffusion, LLaDA, Dream, RND1 (Qwen3-30B-A3B -> diffusion, 500B tokens), Nemotron-TwoTower (2.42x at 98.7 %), DiffusionGemma, DFlash | see Kamen's idea below | (a)-(d) closed on paper; (e) = A2 killed |

> (a quote from the project owner, kept private)
- (a) An any-order / diffusion version of this model or a sibling: none for Qwen3.8-Flash-Next. Siblings with open weights: RND1
  (from Qwen3-30B-A3B, 500B tokens of continued training), Nemotron-TwoTower (2.42x throughput at 98.7 % quality), DiffusionGemma
  (Gemma 4 26B MoE). Converting the 125B needs hundreds of billions of training tokens: not possible locally. A sibling as a
  drafter for the 125B would be a weaker model than the MTP head on our text.
- (b) A locally trained any-order block drafter = DFlash's design (block diffusion). Our retarget reached 6.83 tokens per block on
  code and 2.06 on agentic; any drafter is capped by the agentic ceiling under greedy matching (A1 is the way around it).
- (c) Jacobi / Lookahead parallel fill-in with the 125B itself (exact, no training): published gains 1.5-2.3x tokens per step on
  dense models, below our MTP's 3.5-4 per pass, and each wide Jacobi window loads more experts. Closed on paper.
- (d) Fill order chosen by memory cost: needs an any-order model at the token level; for parallel streams, grouping by shared
  experts is published (Saber, multi-node clustering), and S itself is small (A2).
- (e) Chunk-level form (skeleton, then parallel blocks): measured tonight, S_eff 1.27 (kill line 1.3). Keep as a Write-file lever.

## Bets in flight (tag in every summary)
1. UGB (uncertainty-gated budget): decisive drafter-gate screen = TrainLab turn17.
2. A1 (greedy is the hidden cap) = LEAP's FIRST BET (account-2, owns it end-to-end): data = turn17's exact__tgt_top_* /
   exact__mtp_top_* (top-16). Kill if lossless sampling acceptance (1 - TV) <= argmax agreement + 0.02 on IFEval / agentic.
