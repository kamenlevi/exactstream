# Each prompt's first check re-wired the 32 GB expert pools (~250 ms); one mx.set_wired_limit call removes it, bitwise in the exact mode (+18.8 % [P] whole decode on the 45 GB FAST v1')
- Date: 2026-11-07 07:0x-08:37 EET. Sessions: NewIdeas found the cost (firstcheck_price.py: 15.8 % of v1's decode wall);
  BigMoE's kill test (data/fc, 3ea8e2e) showed a one-time wait, not host work; Leap named and fixed it (FCP, 038bfa1,
  TABLE_350 "FC"); orchestrator (this note, from Leap's table).
- Cause: IOGPU's wired-memory collector unwires the expert pools during the prefill. The first check then wires
  +31.8 GB again, which blocks the main thread with no CPU work.
- Fix: one call after model load, mx.set_wired_limit(max_recommended_working_set_size) = 55.66 GB on this Mac. This is
  published practice: mlx-lm's generate.py wires model memory this way during a generation. Our screens never called it.
  The footprint is unchanged: the limit caps what MLX may wire.
- Numbers [K] (C 224; medians over 3 prompts: HumanEval 68, 16, ifeval/1220; CAP 64):

| arm | check 0 wall ms | checks 1-3 ms | check 0 excess | check 0 wires | lifetime peak |
|---|---|---|---|---|---|
| v1' (FAST) | 345.2 | 86.9 / 99.7 / 91.9 | +253.3 | +31.8 GB | 43.0 GB |
| v1' + wired | 89.6 | 85.5 / 99.2 / 90.2 | -0.7 | +0.2 GB | 43.0 GB |
| NAX6-N exact | 287.5 | 159.4 / 213.9 / 191.1 | +96.4 | +30.9 GB | 38.2 GB |
| NAX6-N exact + wired | 178.0 | 153.8 / 207.4 / 186.1 | -8.2 | +0.2 GB | 38.2 GB |

- Exactness: in the exact env, the verify-logits hash equals the unwired run's on all 43 checks of the 3 prompts, and the
  tokens are equal 3 / 3. FAST v1' tokens are equal 3 / 3. Residency changes no kernel, input or order.
- Gain MEASURED (BigMoE S6W 09:13): +27.4 % whole decode on v1' at C 256 (142.2 -> 181.2); see 2026-11-07_0945.
  NewIdeas' [P] was +18.8 %. The same call applies to the EXACT default and to the 12 GB and Breakthrough engines.
- Resolved (BigMoE 14:1x): the volatile tier (R57_VTIER) uses its own newBuffer allocations, not MLX's allocator, so the wired limit does not wire it; it stays purgeable. It
  filled 0.00 GB in these runs.
