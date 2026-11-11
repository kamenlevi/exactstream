# Plan: optional fast modes on the 24 GB track, judged by task results (2026-10-23)

Kamen's bar (Fri ~08:00, via the orchestrator): an optional fast mode may give slightly different answers if it scores
the same on math, coding and instruction tests; exact mode stays the default. First candidate: b16 (shared-expert-set
per-pass budget + budget-aware JIT, +47 % geomean at 24 GB per TrainLab); if it fails, b18, then b20.

## Protocol (the same as the small-RAM verdict)

- Decode sets: core = math 50 (MATH-500 L3-5 + GSM8K), IFEval 100, HumanEval+ 60, **two runs**, pooled per item; plus
  MGSM 80 (10 languages, scored), multilingual 24 (agreement with ref vs the floor's), agentic 29 real Claude Code turns
  (next 128 tokens, agreement vs the floor's), one run each (two if the GPU allows).
- Anchors already decoded: ref = the exact 4-bit model, plain AR greedy; floor = the same weights, other summation
  order. Also shown: plan4433333333 (the 3-bit tier plan, core sets) as the "v3 exact mode" column, if that is the exact
  mode the fast mode is measured against.
- Guard: OFF on both sides (like-for-like with the guard-off anchors; R1 is the main-track default, so the fast-mode
  runs set BIGMOE_LOOP_GUARD=0).
- Rules: the pre-registered ones (IFEval / HumanEval+ capability drop > -5 pt with McNemar; length ratio; hesitation;
  truncations / loops <= ref + 1), read on the pooled two-run deltas with 90 % CIs; math and MGSM without a rule.
- GPU: ~2.5-3 h per mode (2 x ~55 min core at MTP speed + ~40 min new rows), ~24 GB + MTP; queued after B12L's two
  core runs (tools/day_runner10.sh), then b18 / b20 only if b16 fails.

## What the hook needs to give (BigMoE / TrainLab)

1. A loader with the bt_b12 contract: `load_<mode>(...) -> (rt, st)` (st.gate.check() available), the pinned commit.
2. The decode call as it ships: MTPLX generate_ar, or the MTP path (which function, k / draft settings), greedy,
   stopping at EOS; the prefill as it ships (chunk sizes).
3. The env switches it sets, and which of the suite's refused speed flags it needs (allowed for this variant only).
4. RAM footprint for the memory gate.
5. Whether outputs depend on history across rows (budget / JIT state): two runs either way.
6. Which exact mode the speedup is against (4-bit reference or the v3 tier plan), so the verdict table shows the right
   anchor.

## Hook received (BigMoE 08:05, pinned 36c9cc5; engine = af0cdcb)

- `~/BigMoE/scripts/fast24.py`: `load_fast24(0.25, mtp=True, budget="b16" | "b18" | "b20" | "")`, `generate_fast24` =
  spec125 MTP with FIXED widths 3:3:0 (deterministic per prompt), chunk-major 2,048-token prefill through the exact
  grouped tier prefill, guard off, ~24-25.5 GB (gate 26 GB). No output-relevant state across rows.
- Variants: **f24b16** (the budget mode) and **f24nat** (budget "" = the v3 tier plan's natural routing, the speed-up's
  baseline, exact-to-v3); f24b18 / f24b20 only if b16 fails. The suite applies none of its own patches or forced flags
  (EVALSUITE_FAST24=1; FAST24_ENV set from process start by cli.gpu, which also checks the pin).
- Tables: f24b16 vs the 4-bit ref (+ floor) and **f24b16 vs f24nat** (`report --ref f24nat`), which answers Kamen's
  question ("scores the same as exact mode"); f24nat vs ref shows the v3 plan itself.
- Running: tools/day_runner11.sh after B12L's two core runs: f24b16 core, f24nat core, their MGSM / multilingual /
  agentic rows, f24b16's second core run (also a determinism check: fixed widths should reproduce it).
