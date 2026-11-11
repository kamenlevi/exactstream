# L4 result (QUEUE L1-05): DFlash2 fine-tuned on the 125B's OWN agentic outputs. Leap, Sat 2026-10-24 07:48
Pre-registration: DFL2D_PREREG.md Part 2 (01:52). The run was L4b: the 03:32 generation died in a quiet-window freeze;
the rerun resumed at 17 turns, pause-safe, generating at C=192.

## Verdict: KILLED. Agentic tau8 2.96, the same as the original (line 4.07 = 1.10 x MTP's 3.70)

| held-out (fixed counting) | original | fine-tuned |
|---|---|---|
| 12 agentic turns, tau8 | 2.96 | **2.96** |
| P(first j right), agentic | 0.64 0.41 0.31 0.22 0.17 0.14 0.11 | 0.61 0.43 0.31 0.23 0.17 0.15 0.11 |
| MTP K=7 on the same tokens | 3.70 | 3.70 |
| cross-check / oracle hybrid | 3.48 / 4.04 | 3.53 / 4.11 |
| code guard (24 HumanEval+) | 6.87 | 6.73 (-2 %, guard holds) |

- Training data: 60 windows of the 125B's greedy outputs (18,518 tokens, 17,019 anchors) on Kamen's session prompts,
  with L3's exclusions. 2 epochs = 144 steps, 6.2 min, lr 2e-5, as L3.
- Validation on held-out windows of the same kind did not move. Depth-1 on-path top-1 0.801 -> 0.796 (predicted
  ~.78 -> ~.85), and every depth stayed flat. The drafter learned nothing measurable from it.
- Predictions: agentic 2.86 -> 3.5 (old counting) was wrong; measured 2.86 -> 2.85. P(line) ~20 %.
- Checks:
  - original per-turn tau reproduced in the old counting (the L4 batch, 02:09);
  - on-path rate of the extraction 0.996-1.000;
  - weights loaded (eval ran with them).

## What it means
- DFlash2 on agentic text is stuck at ~3 tokens per pass, whatever is tried here:
  - longer context (L2b);
  - Kamen's text as labels (L3);
  - the 125B's own outputs (L4).
- MTP stays at 3.70. The agentic drafter gap is not a distribution mismatch that a short local fine-tune closes.
- Caveat: small data. The published recipe uses ~800K regenerated samples; this used 60 windows. A run 10-100x larger
  is beyond a "really small local model" job unless it is spread over many nights. NewIdeas' call.
- Agentic stays on MTP + lookup. The block drafters belong to code / math.

## Operations
- Generation at C=192 peaked at 36.0 GB, and swap grew 9.87 -> 11.63 GB (+1.75 GB, under the 2 GB stop). C=192 is at
  the memory edge on this Mac; use C<=128 for long-context generation.
- Data deleted after the verdict (07:48, private session-derived): features, windows, samples, fine-tuned weights
  (~2.3 GB). Also the L5 weights (1.1 GB). Freed ~3.4 GB; disk 123 GB free.
  - Kept: the public code / math features (data/dfl2d_code, dfl2d_math, 0.76 GB) for quick drafter evals.
