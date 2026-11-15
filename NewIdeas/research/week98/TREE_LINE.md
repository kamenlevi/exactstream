> SUPERSEDED by EvalSuite's CERTIFY98 amendment 6 (c915da9, Sun 15 Nov 14:49): delta-method LB (CR1, t(K-1)); each method under its OWN walk over the same windows; coverage in certify98's walk convention (HE+ depth 0 93.33 -> 98.23, MBPP 89.72 -> 97.61). Scorer: EvalSuite tools/certify98_tree.py. Default candidate checkpoint: ~/datasets/deepdraft/ckpt/dd_t30b.safetensors, sha256 790c42db393ad158f0f61e64962c868967995268babb816b27cbf8f4e215d502 (CONFIRMED by the orchestrator, Sun 15 Nov 14:5x; final unless it writes otherwise before Wed evening). Amendment 6a (598d5fb): Leap's measured c16 = 1.025, so PASS = LB >= 1.025 AND T >= 1.05575 (strict 3 % net, agreed); HE+ power 0.72, MBPP 1.00.

# CERTIFY98: the tree line (draft v1 for EvalSuite)

NewIdeas, Sun 15 Nov 2026, 14:4x (`date`). This follows the orchestrator's 14:5x request. The dev numbers come from
`tree_coverage.py` / `tree_coverage.txt`, run on BigMoE's real t30b tree dumps in S9c's layout and walk.

## The candidate
- **t30b + EAGLE-2-16:** best-first by path probability, top-8 children, 15 nodes plus the bonus, verified in one exact
  16-row tree check.
- A better drafter replaces it if one lands by Thursday.

## The statistic T (per fresh set)
- **T = tree-16 kept per check ÷ chain-16 kept per check.**
  - Both are measured under the decode walk (certify98's `kept_walk` logic, R0 = 1), pooled over windows, on the same
    anchors.
  - The data is BigMoE's post-capture TOPK + TREE dump: head only, `on_true` per node.
- **Uncertainty:** a window-clustered bootstrap (>= 2,000 resamples of windows), giving a one-sided 95 % lower bound
  LB(T).

## The line (gated on HE+ 140 and MBPP 300; everyday 150 reported only, since there is no dev tree data on chat)
- **PASS:** LB(T) >= 1.021 AND point T >= 1.052.
  - 1.021 is Leap's measured tree-verify overhead ceiling c16. The 3 % net margin over it is S9c's pre-registered
    line.
- **BETWEEN:** point >= 1.052 but LB < 1.021 (under-powered).
- **FAIL:** point < 1.052.

## Power (dev SE scaled to the fresh window counts)
| set | dev T | dev SE (windows) | fresh SE (windows) | P(LB >= 1.021) | P(point >= 1.052) |
|---|---|---|---|---|---|
| MBPP | 1.136 | 0.0105 (100) | 0.0060 (300) | 1.00 | 1.00 |
| HE+ | 1.060 | 0.0170 (24) | 0.0071 (140) | 1.00 | 0.87 |
| HE+ at dev's 95 % low end (1.027) | | | 0.0071 | 0.20 | 0.00 |

HE+'s 3 % net margin is the binding part of the line, because HE+'s tree gain is narrow.

## What "98 % exact" means with trees
- **Definition: walk-strict COVERAGE per position.** At each depth d, P(the 125B's word is among the VERIFIED
  candidates at d | all earlier positions were covered), measured at the walk's check starts.
  - For a chain, coverage is exactly the per-guess (top-1) walk-strict accuracy that CERTIFY98 already gates, so the
    two metrics nest.
  - For a tree, coverage counts any verified sibling.
- **Dev coverage by band (chain-16 top-1 -> tree-16):**

| set | 0 | 1-2 | 3-6 | 7-10 | 11-14 |
|---|---|---|---|---|---|
| HE+ | 93.80 -> **98.58** | 93.72 -> 96.42 | 94.47 -> 93.70 | 95.83 -> 95.00 | 97.10 -> 96.24 |
| MBPP | 90.03 -> **97.79** | 86.40 -> 92.11 | 86.08 -> 86.11 | 84.40 -> 81.04 | 86.78 -> 80.38 |

- **Read:**
  - Trees meet 98 % where the wall is: the first guess after a miss, at depth 0, the check start.
  - Deeper, the 16-node budget leaves few siblings, so coverage at 3-14 equals or trails the chain's top-1.
  - So a tree certificate should state coverage per band, paired with T for throughput, and not make a single
    "98 % to 30" claim.
- **Recommended wording:** "Exact. The verified tree covers the 125B's word at x % per position (per band), and keeps
  T x more tokens per check than the chain at the same check cost."
