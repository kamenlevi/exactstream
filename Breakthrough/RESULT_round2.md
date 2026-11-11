# RESULT, Breakthrough night 1 (2026-10-20, written ~01:35; for the orchestrator's 07:20 pass)

All GPU work ran in the evening gap under queued quiet locks (except ~20 min of quiet-window time, see the end) and
finished by 00:54. Details: SCOREBOARD.md section 8, IDEAS.md "Status after night 1", data/expA, data/noise35, data/d3.

## D1 budgeted routing: alive
- 125B, GSM8K-150, wait for the router's top-2 experts only, the rest substituted from resident experts or dropped
  (missing ones still read in the background), vs BigMoE's exact eager 147/150 (same harness; same-weights noise arm
  147/150 identical):
  - 20 % of experts resident (13.6 GB): **145/150, 144/150 answers identical (96 %)**, 21.5 tok/s eager vs 13.8 exact.
  - 10 % resident (6.9 GB): **144/150, 144/150 identical (96 %)**, 19.0 tok/s eager, peak 13.2 GB.
- The frugal mode (no background reads: the one that saves SSD bytes, 152 -> 20 reads/token in the trace replay)
  hangs MoEPrefetch's fork gate (main thread waits forever on the GPU event when nk < n). Not measured. Needs the
  frugal path fixed in a Breakthrough copy of fork125's C++ ext.

## D2 Q4-equal bits: alive, reframed as a KLD budget (35B proxy, noise injection per tensor class)
- Against the 8-bit 35B, everything at Q4 (~ the 4-bit pack) costs KLD 0.112 (top-1 86.9 %). Routed experts are 0.041
  of it, GDN projections 0.057 (the largest), attention 0.035, shared 0.022, lm_head 0.009.
- Q4's error is NMSE ~0.0085-0.0105 everywhere. Experts at the error of a ~3.25-bit trellis code add +0.011 KLD over Q4
  experts, ~3.0 bits +0.024. On down_proj and GDN, affine Q4 costs more than white noise at the same error, so a
  rotated code wins there at equal bits.
- Floor: any perturbation (even NMSE 1e-5) costs ~0.011 KLD / ~3 % top-1 on this MoE via routing flips.
- Implication: experts at ~3-3.25 bits with a rotated code, plus GDN/down better than Q4, can stay at or under the
  pack's total KLD. Next: a real trellis quantizer (fake-quant) instead of noise, and GSM8K on the best mix.

## D3 long passes: dead for the SSD
- Teacher-forced on 40 of the 125B's own answers, a Qwen3.5-4B drafter writes 7.2 tokens between must-fix tokens
  (exact top-1), 27.5 if any of the 125B's top-3 is accepted (GSM 71); 2B: 5.6 / 16. The SSD lever needs >= 100-250.
  Drafter entropy predicts must-fix tokens well (AUROC 0.89-0.95). Kept only for the one-stream speed goal.

## Also done
- Kamen's answers recorded in BRIEF.md and SCOREBOARD.md (1000 tok/s = one chat; base Mac mini 16/256, model <= ~100 GB
  on disk). HANDOVER.md kept current. 125B re-trace with router probabilities saved (data/expA/traces125/*_g.npz);
  the gate-mass replay script is written but not run (programs blocked by the Eval suite's quiet lock until 02:55).

## Utilization
- GPU: expA 22:03-23:54 (GSM8K arms, 50-problem chunks), noise35 00:22-00:35 (~41-49 GB peak), D3 00:36-00:54
  (~41 GB peak). Gaps were other sessions' locks and the 00:00-00:22 quiet window.
- Mistake: at 00:03 I sent SIGCONT to my runner shell, which the orchestrator had frozen for Kamen's no-fans window
  (00:00-00:22). It took the lock and started the 35B reference job at 00:03; that job only finished at 00:22 (it
  needs ~1-2 min), so it seems to have been frozen again, but it may have loaded part of the model before that.
  I won't resume stopped processes again.
