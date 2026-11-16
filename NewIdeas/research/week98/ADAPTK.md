# ADAPTIVE K: choose the check's size per check (NewIdeas, Tue 17 Nov 01:0x)

**Verdict: PASS (screen), exact, a lead on copy-heavy code.** A one-parameter rule chooses the tree size per check.
Best-first drafting keeps popping nodes while the next one's path probability is >= tau. The rows are then rounded
up to an available check size (16 or 32 rows today).
- With today's two kernels, at a 0.40-0.92 ms per-level draft cost (counted), HE+ gains **+8.0-10.3 %** over the
  best fixed K, using the other set's tau. MBPP gains ~0 % and never loses (-0.2 to +0.2 %).
- With one-pass check forms up to 128 rows: HE+ +10.3-15.6 %.
- The hindsight ceiling (oracle DP): HE+ +17-34 %, MBPP +9.5-12.4 %.

## Why it works

- path_p is calibrated. The expected kept, 1 + the sum of path_p over the tree's pops, matches the realised mean:
  - HE+ 12.83 vs 12.82 at 16 rows, 21.25 vs 21.01 at 32;
  - MBPP 8.22 vs 8.18 at 16 rows.
- path_p is non-increasing in pop order on 100 % of anchors. So "include a node iff its path_p >= lam x the marginal
  cost of its row" is a threshold rule.
- HE+ has long confident stretches (docstring copies, where K* = 32 already tied). There the rule takes 32 rows, and
  16 elsewhere. MBPP's drafter is rarely confident past 15 pops (32 rows on 12 of 2,080 checks).

## Numbers (exact engine costs from E14; tok/s)

Setup: t30b 127-node dumps (n127k8), certify98 walk, overhead 4 ms; draft cost = L ms per tree level, counted.

| variant | L | HE+ fixed best | HE+ policy (cross tau) | MBPP fixed best | MBPP policy (cross tau) |
|---|---|---|---|---|---|
| KERN2 (16 / 32 rows, today's kernels) | 0.40 | 154.2 | 170.0 +10.3 % (168.3 +9.2 %) | 103.7 | 103.9 +0.2 % (103.7 -0.1 %) |
| KERN2 | 0.92 | 141.6 | 153.9 +8.7 % (153.0 +8.0 %) | 96.6 | 96.7 +0.1 % (96.4 -0.2 %) |
| MULTIPASS (> 32 rows = repeated passes) | 0.40 | 154.2 | 164.0 +6.4 % (same) | 103.7 | 103.8 +0.1 % |
| ONEPASS (Leap's v2 growth, rescaled to E14) [est] | 0.40 | 154.2 | 178.2 +15.6 % (175.8 +14.0 %) | 103.7 | 103.9 +0.2 % |
| ONEPASS [est] | 0.92 | 141.6 | 159.3 +12.5 % (156.1 +10.3 %) | 96.6 | 96.7 +0.2 % |

- With drafting hidden (L 0): ONEPASS HE+ +17.4 % (oracle +35.9 %).
- Lossy byte-floor model, ONEPASS L 0.40: HE+ +12.3 % (oracle +34.0 %), MBPP +0.4 % (oracle +17.8 %).
- Oracle (hindsight DP), KERN2 L 0.40: HE+ +18.9 %, MBPP +9.5 %. The gap to the policy is per-check signal that
  path_p misses, mostly on MBPP.

## Caveats

- **Dev data.** heplus24 + mbppbank, the same windows the K* tables use. The gain is HE+-specific (copy-heavy). A
  fresh-cells re-read on Wednesday's capture is the clean test.
- **Engine.** The policy needs a 32-row TREE verify path. Today's TREE16 path is 16 rows (M4 tests it Tue night).
  C32one exists as a chain check. c32 = 1.04 is a prior.
- **Drafting.** The draft cost per level is an estimate (0.40 for a smaller head; 0.66-0.92 for today's 177.6 MB per
  level). A draft forward is counted per tree level, including the spare-slot pops.
- **Prior art.** The live 45 GB loop truncates rows by running confidence (S45c t 0.3, ADOPTED; raising it was a
  KILL, S45d). This rule is the other direction: grow past 16 when confident. Outside the project, OPT-Tree /
  Sequoia-style size choice. The new part here is the measured calibration and gain on this drafter and engine.

## Cheapest next kill

On fresh cells (the Wednesday capture's dumps), re-read the policy at the frozen tau (0.9 = MBPP's, cross). It is
KILLED if HE+ falls under +3 % at L 0.66 with KERN2. Then one engine A/B (adaptive vs fixed-16, C T T C, like M4)
once a 32-row tree verify exists.

Files: adaptk_oracle.py / .txt (stage 1, drafting hidden), adaptk_draft.py / .txt (draft cost + policy),
adaptk_sens.py / .txt (kernel variants), costs from costs_e14.json.
