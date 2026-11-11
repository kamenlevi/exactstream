# Breakthrough round 3 result (draft 2026-10-21 10:20; final after the Eval suite's frugal decode verdict and today's speed runs)

Goal (STATE.md): Qwen3.8-Flash-Next 125B with Q4-like "similar intelligence" in 6-12 GB at 100-150 tok/s on the M5 Pro.

## 1. Frugal routing: fast and small, with a Q3-class drift
Wait only for each token's top-2 experts, skip missing lower ranks (no SSD reads for them), MTP/lookup speculation:
| residency | peak RAM | code / GSM8K / prose tok/s | GSM8K-150 (exact 147) | teacher-forced KLD (ar4 0.0096) |
|---|---|---|---|---|
| 10 % | 14.3 GB (16.3 over GSM8K) | 39.8 / 38.1 / 30.3 | 146, 146 answers equal | 0.057 suite; 0.100 ifeval+multi, 0.027 gsm+code |
| 5 % | 10.9 GB (13.0) | 34.4 / 35.1 / - | 144, 144 equal | not measured (higher) |
Frugal reads ~56-82 MB/token from the SSD. Its KLD sits between llama.cpp's Q3_K_M (~0.037) and Q2_K (~0.082): the Eval
suite's decode rows (math, IFEval, HumanEval; running 10-21 morning) decide whether that is "similar intelligence".

**OFFICIAL verdict (Eval suite, 10-21 13:25): frugal 10 % FAIL 5/8 (fidelity: KLD x5.97 ar4, top-1 -3.5 pt; decode IFEval -2.0 and HumanEval -1.7 pass); frugal 5 % FAIL 4/8 (KLD x9.9, IFEval -7.0 pt). The fidelity gate is per corpus part (KLD <= 1.10 x ar4); easy text binds; only wait-to-85 % (no renorm) passes every part on our subsets.** Earlier preliminary reading: math n=50 frugal 90.0 % vs reference
86.0 % / floor 88.0 %; IFEval n=100 88.0 % vs 90.0 % / 88.0 % (-2.0 [-6.7, +2.7] pt, McNemar p 0.73); answer agreement with
the reference 92 % / 91 % vs the floor's 98 % / 97 %; HumanEval and the agreement-vs-ar4 rule pending in the official report.
Why tasks hold despite the KLD: frugal's KLD sits at high-entropy positions (86 % of it at entropy > 0.5), where it picks a
different but similarly plausible token; confident positions never flip (0 of 21,712 at entropy <= 0.1). Gap to close
before adopting it: its largest drift is on multilingual text (teacher-forced KLD 0.092 vs 0.015 on code) and none of the
decode tests is multilingual: a Bulgarian / multilingual generation check is needed.

## 2. Why frugal drifts, and what fixes cost (all measured on the Eval suite's own scorer)
- The 125B's routing is flat (rank 0 = 21 % of the top-10 gate mass, 90 % needs 8.8 ranks) and its routed experts are
  mutually orthogonal (best |cos| between two experts' outputs on the same input: median 0.024). A skipped expert's
  contribution is a lost independent direction: no other expert, shared expert or mean expert approximates it, and
  renormalising the rest makes it worse. KLD_excess ~ (missed gate energy)^1.5, validated to 1 %.
- Every coverage policy we tried lands on one KLD-versus-SSD-bytes curve per kind of text (wait-by-mass, adaptive
  thresholds, confidence-gated exact re-runs, hybrids, low-bit tail reads). Routing statistics are the same for code,
  chat, math and Bulgarian; ifeval/multilingual text costs more KLD only because its predictions are more sensitive.
- Q4-like KLD at 10 % residency needs ~200-300 MB/token of SSD reads (wait until 85 % of the gate mass: 0.0094 on the
  hard subset / 0.0021 on gsm+code at ~107 reads/token), only ~30 % fewer reads than exact decoding at that cache size.
  Trellis-coded 1-2-bit tail copies would save at most ~20 %.

## 3. What this means for the goals
- Q4-like quality at 6-12 GB is, in cost, "the main track's exact engine with a small cache": every main-track speedup
  (load/compute overlap, 3-bit tier, records3) carries over; the SSD caps it at ~50-100 tok/s on the M5 Pro with perfect
  overlap (today's engine: ~20-25 tok/s at 14 GB, speed runs pending), ~10-20 tok/s on the base Mac mini (3.2 GB/s).
- 100-150 tok/s at 6-12 GB is reachable only (a) with frugal-level reads (~60-80 MB/token) if its drift is acceptable,
  plus engine work, or (b) with fewer active experts per token via distillation (Edge0-style; needs rented GPUs; the
  cheap router-only re-fit cannot work because the experts are orthogonal) - Kamen's decision.
- 4 GB: below the measured 7.5 GB floor (dense trunk read every pass); not reachable with this model by inference tricks.

Details: SCOREBOARD.md (sections from "Frugal's quality mechanism" on), research/skipping_principles.md,
research/speed_path.md, research/training_route.md, IDEAS.md D1b-D1d.
