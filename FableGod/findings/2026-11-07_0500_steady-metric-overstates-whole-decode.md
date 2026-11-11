# CORRECTION: tonight's 45 GB FAST numbers used the registered "steady" metric, which leaves out host work and each prompt's first check; on whole decode the v1 config runs 142.1 tok/s, not 180.3
- Date: 2026-11-07 05:0x EET. Sessions: NewIdeas found it (~/NewIdeas/scripts/s1_fullperiod.py, data/s1/). The orchestrator
  recomputed every arm from BigMoE's raw run records (this note).
- The metric: the registered "geo steady" tok/s is tpsw = kept / (c2 - c0) over checks >= 2. It times only part of each
  cycle and skips each prompt's first check. It leaves out:
  - ~2.35 ms of host work per check in the base, and ~4.7 ms with NW v3. NW's per-check counter graph (a sort and 8 sums
    per layer) runs at the readback after c2, outside the timed window. Both arms wait on the commit replay (NewIdeas
    05:1x; BigMoE's own re-derivation: full period +4.3-4.8 %, whole decode +3.3 %);
  - the first check of each prompt (~300 ms, ~20 % of check + draft time on these 12 prompts).
- Whole decode = n / decode wall time (prefill excluded). Below: the geometric mean over the 12 HumanEval prompts per pass,
  then the geometric mean of the passes. From ~/BigMoE/data/r57/<tag>_runs.json (fields n, tps, tpsw). This aggregation
  differs from the reports' by <= 0.5 tok/s.

| arm (12 HumanEval, FAST) | registered steady | whole decode | steady / whole |
|---|---|---|---|
| qall, C 224 | 99.0 | 90.5 | 1.09 |
| stack, C 224 | 132.6 | 116.9 | 1.13 |
| stackb, C 224 | 140.1 | 123.4 | 1.14 |
| stack, C 256 | 136.5 | 116.8 | 1.17 |
| stackb, C 256 | 144.9 | 123.5 | 1.17 |
| S2 A/B: stackb -> + S2 | 142.2 -> 146.6 (+3.1 %) | 119.5 -> 124.8 (+4.4 %) | |
| KEEP A/B: stackb + S2 -> + KEEP | 149.0 -> 164.0 (+10.1 %) | 126.0 -> 135.3 (+7.4 %) | |
| S1 A/B: + KEEP -> + NW v3 | 164.8 -> 180.2 (+9.3 %) | 137.5 -> 142.1 (+3.4 %) | 1.27 |

- What changes:
  - The 45 GB FAST v1 config (stackb + S2 + KEEP + NW v3, C 256) runs **142.1 tok/s whole decode**. Its full-period steady
    (all wall time after each prompt's first check, NewIdeas) is 165.4, and the registered steady is 180.3.
  - NW v3: +3.1 to +3.4 % on whole decode and +4.3 % full period. That is BETWEEN on S1's +8 % line, not PASS.
    - A re-test is queued (NewIdeas): NW with its per-check readback off, B / NW / NW / B at C 256, scored on full
      period.
    - If it fails, v1 = stackb + S2 + KEEP (137.5 whole decode).
  - stackb at C 256 vs C 224: +3.4 % on the registered steady (its PASS) but +0.1 % on whole decode.
  - S2 (+4.4 %) and KEEP (+7.4 %) hold on whole decode.
  - The gap grows as checks get faster (1.09 for qall, 1.27 for NW): the fixed per-prompt and per-check costs weigh more.
- Earlier notes in the same metric carry a CORRECTION pointing here (2026-11-07_0025, 2026-11-07_0205). Older "steady"
  figures used the same metric. Example: the 11-02 exact default, 63.6 steady vs 56.4 whole run.
- From now on (orchestrator): headlines use whole decode, with prefill reported separately. Full-period steady is the
  steady-state figure. The registered tpsw is information only.
- Against 200 tok/s: x1.41 on whole decode, x1.21 on full-period steady.
- How: ~/BigMoE/data/r57/<tag>_runs.json for the tags in data/{sat,s2,s2c,s1}/*_report.txt;
  ~/NewIdeas/data/s1/s1_fullperiod.txt and s1_rederive.txt.
