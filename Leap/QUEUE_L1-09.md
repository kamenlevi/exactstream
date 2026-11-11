# QUEUE L1-09, early-exit verify of confident passes: Leap, Sat 2026-10-24 06:04 (GPU screen in the L5 batch)
Line (NewIdeas, QUEUE.md): kill if top-1 agreement < 0.98 at layer 40 on code positions where the full model has
p >= 0.9.

**How.** scripts/dfl2_exit.py, teacher-forced prefill over 96 sequences: EvalSuite's tf_code and tf_gsm items, plus
the 12 held-out agentic turns.
- An exit applies the model's own final hyper-connection mixer and LM head to the widened stream after zero-based layer
  35 or 39 (36 or 40 of 48 layers), as the text model does after layer 47.
- Sanity: the full model's top-1 equals the stored greedy tokens on 0.988-1.000 of the positions.

**Result: KILLED by a wide margin** (data/exit_result.txt).

| set | positions | agree, exit after 36 | agree, exit after 40 | at full p >= 0.9 (share of positions) |
|---|---|---|---|---|
| tf_code | 6,375 | 0.002 | 0.057 | 0.056 (93 %) |
| tf_gsm | 14,320 | 0.002 | 0.060 | 0.059 (88 %) |
| agentic | 1,920 | 0.001 | 0.033 | 0.029 (71 %) |

- The exits are never confident: their own p >= 0.9 on 0 % of positions.
- So the model's readout does not decode intermediate streams. With hyper-connections the final mixer seems tuned to
  the last layer's stream. Possibly also the per-layer n-gram embedding (PLE) feeds in late. Not checked.
- A trained readout per exit (tuned lens, LayerSkip-style) would have to go from 0.06 to >= 0.98 on confident
  positions. That is not credible without training the 125B itself.
- **Control PASSED (Sat 06:31, first GPU step of L4b; data/exit_control.txt; 6 sequences per set).** The exit
  after the last layer agrees 1.000 on every set, so the pipeline is right and the kill stands. The prediction forms
  only in the last few layers:

  | exit after layer | code | GSM8K | agentic |
  |---|---|---|---|
  | 44 | 0.324 | 0.290 | 0.239 |
  | 46 | 0.678 | 0.680 | 0.506 |
  | 47 | 0.925 | 0.928 | 0.739 |
  | 48 (full model) | 1.000 | 1.000 | 1.000 |

  - Even skipping only the last layer misses the line: at full p >= 0.9, layer 47 agrees 0.950 on code.
  - Gated by the exit's own p >= 0.99, layer 47 is right 0.997 on 62 % of code positions. That saves 1/48 of a pass:
    nothing.
