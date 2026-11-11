# Leap, Sat 2026-10-24 01:52: the DFlash2 tree bound (code / math) and bet L4 (DFlash2 trained on the 125B's own
# agentic outputs). Pre-registered before any run
Assigned by NewIdeas (00:18). The orchestrator relayed Kamen's go for the whole night (01:40).

## Found while preparing: my tau counting was low
- dfl2_eval.py's `tau_of` counted a fully accepted block as 7 tokens. The official loop
  (third_party/dflash_model_mlx.py, `_stream_generate`) yields the 7 drafts plus the bonus, `t_list[7]`: 8 tokens.
- So every tau I reported (L2, L2b, L3; DFlash2 and MTP alike) is low by the share of fully accepted blocks.
  - The per-depth acceptance figures (P(first j right)) and the hybrid policies were counted correctly.
- Implied corrected values: DFlash2 code 6.15 -> ~6.93, GSM8K 5.26 -> ~5.74, agentic 2.86 -> ~2.97. MTP agentic:
  3.54 plus its own share, to be measured.
- The kill verdicts compared both drafters under the same counting. They are re-checked below with the fix.
- From now on: the fixed counting decides; the old one is printed beside it.

## Part 1: the tree bound for DFlash2 on code / math
NewIdeas' line: tree-16 >= 8.0 tokens per pass gives code-350 a second lever; else park trees.

**Known before any run** (the drafter's config and L2's measured coverage):
- DFlash2 is trained with block_size 8: a block drafts depths 1..7, so one pass holds at most 8 tokens (7 drafts + the
  bonus), whatever the tree. A mean of 8.0 would need every block fully accepted.
- Ceiling of ANY tree over the top-16 candidates (unlimited rows), from L2's per-depth top-16 coverage:
  - code: 1.00 .99 .98 .97 .95 .94 .93 -> 7.38-7.76 tokens per pass (union bound .. min bound);
  - GSM8K: 1.00 .99 .97 .95 .92 .90 .87 -> 7.02-7.60.
- **Verdict on the line, by construction: tree-16 < 8.0 -> park trees.**
- What a tree could still add over the corrected chain-8 (~6.93 code): at most ~0.5-0.8 tokens per pass. A 16-row
  pass costs more than an 8-row one.

**Measurement (in the batch below; GPU ~4 min, then CPU):**
- At every anchor of L2's code (24 HumanEval+) and GSM8K (60) sequences, the original drafter's lattice:
  - its top-16 candidates and unary scores per depth;
  - the selector's 16 x 16 transition scores between consecutive depths (depth 1: from the anchor).
- Replays with exact greedy verify, tokens per pass counted correctly:
  - chain-8: the selector's greedy path. It must reproduce the official replay's accepted lengths.
  - tree-16 / tree-32 (rows = anchor + 15 / 31 nodes): best-first by cumulative selector probability (softmax over
    the 16 candidates given the parent), as in EAGLE-2.
  - The top-16 ceiling (any tree).
- Predictions:

| | chain-8 | tree-16 | tree-32 | ceiling |
|---|---|---|---|---|
| code | 6.9 | 7.1-7.3 | 7.3-7.5 | 7.5-7.7 |
| GSM8K | 5.7 | 6.0-6.3 | 6.3-6.7 | 7.0-7.4 |

## Part 2: bet L4, DFlash2 trained on the 125B's own agentic outputs
**Question.** L3 learned Kamen's text but did not transfer to the 125B's own output. Does the same fine-tune on the
125B's OWN agentic outputs (self-distillation) beat MTP by 10 % on the 12 held-out agentic turns?

**Prior art (tag: published method, untested here).** Training a drafter on the target's own generations is standard:
EAGLE-3, Medusa-2 self-distillation, the SpecForge / NeMo recipes; DFlash's drafters are trained on target-generated
data. Nothing new is claimed.

**Data** (stays on this Mac, never committed or uploaded, deleted after the verdict):
- Turns sampled from Kamen's sessions exactly as SpecDecode built the eval:
  - extract.load_session / context_before / classify: kinds edit / write / bash / text;
  - render.render: the forced assistant prefix up to the bulk argument.
  - Mixed like the 12 eval turns: 4 edit : 2 write : 3 bash : 3 text.
  - At most 2 turns per session, round-robin over projects.
- Excluded, as in L3: every session holding a route4 sample's assistant message or target start, or a piece of the
  125B's outputs on the 12 eval turns. Also SpecDecode's own sessions.
- Context: CTX_CHARS 12,000 (~3.5K tokens; the eval's is 28,000). The drafter sees 2,047 rows, and this halves the
  prefill cost.
- Generation: the 125B, exact greedy (spec125.generate, MTP drafts verified, C = 51 experts resident, as L3's
  extraction).
  - Up to 768 tokens per turn, stopping at the end of the turn.
  - Stops at ~36K generated tokens or 45 min, whichever comes first.
- Labels and features: dfl2c_extract on prompt + output. Labels are the 125B's top-32; features are kept only for the
  last 2,047 prompt rows + the output.
  - The on-path rate on its own output is a check (expected >= .97; spec verify vs prefill numerics).
- Split: 90 / 10 by session.

**Training: identical to L3.** One run, no sweep.
- Frozen fc / hidden_norm / the 125B's head. Trained: the 5 layers + norm + selector.
- Loss: soft CE to the top-32 + the selector's CE.
- AdamW, lr 2e-5, cosine. 2 epochs if they fit 30 min, else 1.
- Micro-batches of 2 groups, MLX cache capped at 2 GB.

**Evaluation.**
- The L2 replay at window 2,048 on the 12 held-out agentic turns (features regenerated), original vs fine-tuned in the
  same batch.
- The hybrid bound. The code guard (HumanEval+).
- Fixed counting, with the old counting printed beside it.

**Lines.**
- KILL (NewIdeas): held-out agentic tau8 < 1.10 x MTP K=7, on the same tokens with the same counting.
  - NewIdeas' 3.9 is 1.10 x 3.54 in the old counting. With the fix, the line is 1.10 x MTP's corrected tau, and that
    decides. Measured, not projected.
- Code guard: a code tau drop > 10 % means an agentic-only drafter (two drafters).
- Checks:
  - the original reproduces L3's per-turn tau (old counting);
  - the on-path rate on generated tokens is >= .97;
  - the fine-tuned weights load fully (79 / 79).

**Predictions.**
- Validation (held-out sessions of the 125B's own output), depth-1 on-path top-1: ~.78 -> ~.85.
- Held-out agentic tau8, old counting: 2.86 -> 3.5 (3.1-4.0). First token .64 -> .70.
- P(>= 1.10 x MTP) ~ 20 %.
  - Why not higher: L3's in-distribution gain, carried over at the same size, reaches roughly MTP parity, not +10 %.
  - What could beat it: every depth now has labels (on-path ~1, against L3's .75^(j-1)).
- Code guard: 6.15 -> ~6.0 (old counting).

**Cost: one unattended quiet lock of <= 2 h.**
- Data (CPU) 2 min.
- Eval / code / math features ~8 min.
- Tree lattices + baseline evals ~5 min.
- Generation <= 45 min.
- Extraction ~15 min.
- Training <= 30 min.
- Fine-tuned eval ~4 min.
- Peak ~22 GB, under the swap and HEALTH guards.
- Disk: ~1 GB of features (compressed; only the rows the drafter sees) + ~1 GB eval / code / math + 1.1 GB weights.
  127 GB free; floor ~100.
