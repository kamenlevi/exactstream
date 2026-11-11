# TrainLab STATUS (2026-10-21 17:10)
- Question: does cheap LoRA self-distillation (<= ~1e7 tokens on this Mac) heal dataflows that give SSD reads slack?
- Built: train35.py harness on NewIdeas' 35B segments (own exact reference stored with the frozen ids), chunked
  differentiable GDN, LoRA r16 + pre-routers, cache-dependent flows (pol35 semantics), teacher top-64 cached per step.
  Verified on a tiny random model: exact = stock bitwise, late/half = defer35 bitwise, checkpointed = plain gradients.
- Turn 1 queued (#5, ETA ~19:30): zero-shot of 16 flows per segment, then half_w2_C26 (the engine's dataflow) healing.
- Results: none yet.
