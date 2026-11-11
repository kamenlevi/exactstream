# 45 GB exact prefill 1.9-3.4x faster with bitwise-identical logits: layer-major prefill reads each expert once (430 -> 68 GB of SSD per 7.9K prompt), 441 tok/s at 7.9K; the 45.10 GB peak needs a 0.1 GB trim
- Date: 2026-11-08 07:43-07:5x EET (BigMoE PFX, data/pfx/pfx_report.txt, WED_TESTS e52e687, snapshot 94a6f02).
  Orchestrator: this note, checked against that report.
- Setup: the 45 GB exact config, C 256, three agent prompts (1.6K / 4.7K / 7.9K tokens), quiet lock. TTFT = prefill + setup
  + the first token.
- Result [K], today's prefill (1,024-row chunks) -> PF-A (R57_PFLM=1, layer-major, full expert scratch):
  - 1.6K: 15.2 s = 103 tok/s -> 8.0 s = 195 tok/s (x1.90).
  - 4.7K: 37.3 s = 127 tok/s -> 12.9 s = 368 tok/s (x2.90).
  - 7.9K: 61.9 s = 128 tok/s -> 18.0 s = 441 tok/s (x3.44).
  - Exact: decode tokens equal and the logits hash matches on every check (8 / 8, 28 / 28, 22 / 22).
  - Prefill SSD reads: 109 / 265 / 430 GB -> 64 / 67 / 68 GB (about one pass over the expert store).
  - Lifetime peak 45.10 GB vs 44.50 GB today: 0.1 GB over the 45.0 GB line. It needs a trim (a smaller C or freeing
    the scratch earlier) before it counts at 45 GB.
- PF-A with a compact copy in today's call shape (R57_PFLM=2): 138 / 224 / 256 tok/s, exact, peak 45.90 GB.
- Larger chunks alone (2,048-8,192 rows) are OUT. They change the logits (hash 0 / N), the decode tokens differ at
  4.7K / 7.9K, and peaks reach 46.5-53.8 GB. The pre-registered chunk-size verdict is BETWEEN.
- CORRECTED 08:2x: this note first compared PF-A with the 12 GB engine's prefill (191 / 378 / 461 tok/s,
  2026-11-08_0515). Those figures are VALID (Fit12's GPU proof 09:54; re-measured 192 / 373 / 459 tok/s). An MLX 0.32.2 bug (Leap, a521e91) leaves expert rows
  unwritten in sorted-gather calls over 32,767 rows that are not a multiple of 64. Fit12's engine has no such call:
  its largest is 15,872 rows and every call is a multiple of 64. PF-A is not hit either:
  its 1,024-token calls are at most 10,240 rows, and its logits match bitwise. BigMoE's pe4096 / pe8192 arms were hit;
  pe2048 (20,480 rows) was not and still changes the logits, so the "larger chunks OUT" line stands.
- Leap's PF-B, priced and exact with the pad-to-64 fix: one MoE call per layer is x1.9 on the prefill experts at 4.7K /
  7.9K. Adding the read overlap that the call split allows projects ~590 / ~750 tok/s at 4.7K / 7.9K [P].
- What limits it now:
  - The long prompts are compute-bound: SSD use is 3.8-5.2 GB/s at 7.9K / 4.7K (8.0 GB/s at 1.6K).
  - The levers toward 500-1000 are faster prefill compute (Leap's PF-B) and overlapping SSD with compute.
  - 36-57 % of the expert layers are already in memory when prefill starts, yet today's prefill reads them again.
    Reusing them would cut the 1.6K prompt's 64 GB.
