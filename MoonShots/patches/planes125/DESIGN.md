# Plane-granular expert streaming for the 125B (prototype design, round 18 #1)

Goal: BigMoE's streamer with each routed expert stored as two planes, cached as separate units. Deterministic: the gate
waits for every plane a use needs. No never-stall fallback (that makes outputs depend on SSD timing). Quality =
mix4-msb2m: top-4 of 10 gate ranks exact (MSB + LSB), ranks 4–9 MSB plane only.

Sim (planes_cache_sim.py, 125B traces, equal bytes): demand MB/pass 155.5 → 92.4 (−41 %) at 34 GB. With MSB files only (top
ranks read the original 4-bit files): −24 %. GSM8K: 35B mix4 equal within noise on 450 problems; 125B 6/6 so far, the
full 150 is queued.

## Planes (per expert, per projection; groups of 64; s, b = the 4-bit scales and biases; q = 4m + l)
- MSB plane: m packed 2-bit (2 bits/weight) + scales 4s + bias b + s·off, where off = per-group mean of l
  (bf16, 0.5 bits/weight) → 1.54 MB per expert. Alone it is the msb2m expert.
- LSB plane: l packed 2-bit + off as uint8 per group (0.125 bits/weight) → 1.31 MB. The exact weight is
  MSB term + (s·l − s·off), so the LSB term runs gather_qmm with scale s = MSB scale / 4 and bias −s·off, both computed
  per call from the gathered slots (tiny).
- Files: one MSB file + one LSB file per layer, expert-major, written once by a converter from ~/BigMoE/models/bare
  (read-only source). ~70 GB in total; **leaves ~192 GB free, below the 200 GB floor** (needs an OK, or MSB files only).

## Runtime: two virtual layers per real layer, BigMoE's C++ ext unchanged
- `ext.add_layer(C, E, fds, offs, pers, pools)` is called twice per real layer L: virtual 2L = MSB planes (C_m slots),
  2L+1 = LSB planes (C_l slots), each with its own pools, files, LRU and slot table. `Gate(96)`.
- Decode, MoE block for layer L (rank-sorted top-10 ids, as in planes125.py):
  `sm = gate.route_and_gate(2L, ids[:10])` and `sl = gate.route_and_gate(2L+1, ids[:4])`;
  y = gather_qmm(MSB pool, sm) for all 10, plus gather_qmm(LSB pool, sl) for ranks 0–3 (same rows); weighted sum as before.
- Budget split per layer: C_m·1.54 + C_l·1.31 MB = the per-layer byte budget. Start at C_m/C_l from the sim's LRU mix,
  then tune.
- Prefetch: BigMoE's look-ahead lists feed both virtual layers (MSB for all predicted ids, LSB for the predicted top-4).
- Prefill: unchanged for the first prototype (the 4-bit files and the current path), i.e. exact prefill and plane decode.

## Measure
tok/s at C equivalent to 0.50 (34 GB) vs today's 4-bit streamer at the same GB, on BigMoE's code prompt (eager decode, MTP
off), plus stalled layers/token. Then GSM8K-150 with the prototype, to confirm it matches planes125.py's answers.

## Hooks wanted from BigMoE (one message)
1. Is `ext.add_layer` safe to call for 96 virtual layers with different piece shapes (2-bit codes, uint8 offsets)?
2. `route_and_gate(L, ids)` with fewer than K ids (the LSB layer gets 4): any assumption that the count equals K?
3. Where do the look-ahead predictions attach, so both virtual layers of a real layer get them?
