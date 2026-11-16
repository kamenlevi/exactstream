# IDEAS_1116: the lossy 32-row check at <= 25 ms, measured by Friday (Fable B, THINK_1116 problem 3, Mon 16 Nov 12:0x-12:5x; desk)

[K] measured by its owner, [P] derived from measured numbers, [G] my estimate (halved in sums). Budget terms: check ms.

## 0. Where the 32-row lossy check stands, and what 25 ms needs [K Fit12 LOSSY32, Leap part B, Fit12 bound]

| part | today (C5, two 16-row calls) | floor | why it is above the floor |
|---|---|---|---|
| FAST dense | 23.8-26.6 ms | 7.6 | ~2,900 dispatches at 1.6 us GPU each, narrow K-serial GEMVs, ~5 ms of attention / GDN cores |
| experts | 20.7-21.4 | 14.5-15 | 16-row tiles with ~2.9 rows per expert (110 experts, 320 uses), the one-call's instruction mix |
| 45 GB misses | 5.3 | 0 | SSD reads on the critical path |
| CPU build + encode + commit | 27.9 (a bound) | - | 4,022 dispatches x ~7 us; 52 % of them MLX glue |
| best case, all at once | ~29.8 GPU | 21.9 (6.12 GB) | |

- **The first wall is the CPU, not bytes.** 27.9 ms of host time cannot hide under a 25 ms GPU; today it hides under
  ~61. Any 25 ms check needs <= ~3,000 dispatches (CPU <= ~21 ms, overlapped by B5b's async build) and ideally <= ~1,700.
- **The second wall is the dense pass at 3x its floor.** Its bytes (2.1 GB) are 7.6 ms; its ~2,900 launches and the
  narrow kernels are the other ~17 ms. Bytes cuts on the dense side are worth at most ~1-2 ms (3-bit dense is closed,
  measured flips 3.86 % vs the 1.34 % line [K]).
- So the 4-day work is launches first, then expert efficiency and misses, then bytes. The honest ladder is in section 6.

## 1. DISPATCH DIET THAT EXISTS: ~-1,500 launches in 4 days  [tried in parts; exact except where noted]

- Mechanism. Replace the glue around the check's custom kernels with kernels already drafted or built, by count per
  layer [K part B counts at 16 rows: GDN-type layer 44, attention-type 116; a one-pass 32-row check has no pads]:
  1. attention: att_single or fs4 instead of the 12-chunk SDPA fallback: -56 per attention layer = **-672** [K R-62:
     61 -> 5 kernels per call; FLAT -4.06 / -4.18 ms]. In the lossy stack already? If not, it is the biggest single cut.
  2. expert id / mask glue as one kernel (`kernels/eprep.py`, compiles, logic bitwise on 4,000 cases): -8..-10 per
     layer = **-400..-480**.
  3. GDN body as one dispatch (Leap's GDNB, PASS, bitwise, -0.71 ms at 16 rows): -4 per GDN layer = **-144**; the a|b
     fold into in_proj (bitwise, in DN-KS v2's queue) -2 per layer = -72; k|v into q -2 x 12 = -24.
  4. router + shared gate|up + shared gate as one call stream (W_mlp of DENSE_REBUILD v2 section 3; row-parallel,
     can be bitwise): -3 per layer = **-144**.
  5. HC read: k1c + k2b + k3c stay (k2 folds were KILLed); its 3-6 glue ops per read (pending-write handling, casts)
     are the next -100..-200, by inspection of Leap's dot graphs (part B, dot_*/).
- Numbers [P]. 4,022 -> ~2,400-2,600 dispatches: CPU 27.9 -> ~17-18 ms (under the GPU, so the bound lifts); GPU
  -2.3..-2.6 ms of waits plus the narrow-kernel time the merged calls remove (-1.6 a|b, -1.1 k|v, from part B).
  With Leap's Python specialization (-1.5..-2 us per dispatch, 3-5 days) the CPU reaches ~12-13 ms.
- Tag: tried in parts (R-62 PASS, GDNB PASS, DN-KS PASS, eprep / W_mlp drafted, not run). Exact: 1-4 bitwise or
  FAST-by-record; 5 bitwise.
- Kills it: att_single already in the lossy stack (then item 1 is 0); the glue count per layer at 32 rows being lower
  than part B's 16-row count (then the total is ~-1,000, CPU ~21).
- Test (<= 15 min, Leap, by day): count dispatches of one GDN-type and one attention-type layer at 32 rows with the
  diet items on (part B's wrapper), and time the CPU build of a 4-layer run. PASS: <= 55 per GDN layer and <= 70 per
  attention layer, CPU build <= 20 ms per check.
- Gain in the budget: check -3..-5 ms GPU; the CPU bound from 27.9 to ~17 (a precondition for everything below).

## 2. ROW-DENSE EXPERT TILES at 32 rows: experts 1.43x -> ~1.15x the floor  [tried at 16 rows (L1-24); new at 32; exact]

- Mechanism. The one-call expert kernel runs 16-row matrix tiles. At 32 rows an expert serves 2.9 rows on average
  (320 uses over 110 experts per layer), so ~82 % of each tile's MACs are padding and the kernel is instruction-bound
  [K NAX-ABL: 83 % of the time is decode + op + epilogue, super-additive]. Gather each expert's rows into one tile pass
  (two tiles only when an expert serves > 16 rows), with the row order fixed per expert so the combine stays
  batch-invariant (L1-24 gmr24 was bitwise). Fewer tile passes = fewer instructions per byte.
- Numbers [P]. Today's expert time 20.7-21.4 ms at 1.43x a 14.5-15 ms floor. L1-24 at 16 rows gained 6 % with ~2.2
  rows per expert; at 32 rows with 2.9 the padding is the same order, so the structural gain is in the tile count:
  rows padded per expert 16 -> 16 (no change!) unless the tile is split by expert PAIRS sharing a tile (8 + 8 rows of
  two experts whose row sets are small). That halves the tile passes for ~70 % of experts [G]. Expected -2..-4 ms, I
  count -1.5.
- Tag: tried (L1-24 KILL at 6 %, bitwise); new form (two experts per tile).
- Exact: yes (per-row arithmetic unchanged; the combine order fixed).
- Kills it: the A-operand marshalling of two experts per tile costing more than the saved op.runs (NAX-ABL: A
  marshalling + op = 27 %); or NAX6-N's hot loop having "no cheap cut left" (NAX-MICRO) also covering this form.
- Test (<= 15 min, Leap, share lock): NAX6-N one-call on layer 20's real 32-row routing, tiles packed by expert pairs
  vs today; bits == today; time per call. PASS <= 0.85x, KILL > 0.95x.
- Gain: check -1.5..-4 ms.

## 3. DEAD-ROW DROP WITH EXACT OUTPUT (early death without repair)  [new; output exact, kept lossy]

- Mechanism. A row whose parent token is rejected can never be accepted. If a probe says a row is dead at layer L, the
  row leaves the batch for layers L+1..48. If the probe is wrong, the run simply ends one row earlier (the bonus token
  comes from the last computed row): the output is still the 125B's own argmax sequence, only kept drops. So no
  repair pass, and the probe needs only a good precision / recall trade, not perfection. This is the lossy form of
  IDEAS_1115 idea 1.
- Numbers [P on K unions 73 / 110 at 16 / 32 rows]. Chain-32 keeps ~16, so ~16 rows are dead on average. Dropping them
  at layer 16 cuts the expert bytes from 110 x 48 to 16 x 110 + 32 x ~80 = 4,320 expert-layers: -18 %, i.e.
  -2.7..-3.8 ms of the 15-21 ms expert part. At layer 8: -27 %, -4..-5.6 ms. On a 64-node tree the dead share is
  larger (157 -> ~85 union after the drop): -30..-40 % of expert bytes.
- Predictor: for 4 days, the drafter's own cumulative acceptance probability (free, before the check: p1 >= 0.95 is
  right 99.5-100 % [K]) plus the 125B's layer-L hidden state of the parent position once Tuesday night's capture
  exists (layers 8 / 16 / 24, confirmed by NewIdeas 10:1x). A cf-only drop at layer 0 is just a shorter check, which
  K* already prices; the in-pass drop is the new part.
- Tag: new (M-22 TWOSTAGE and early exit are different mechanisms, see IDEAS_1115).
- Exact: output yes; kept no (EvalSuite sees no change in tokens, only speed).
- Kills it: a probe that cannot beat the drafter's cf (then the drop is a shorter K, already priced); the one-call
  expert kernel not taking a per-layer row mask cheaply (it takes QOM's zero rows today, so the mask exists).
- Test (<= 15 min, CPU, Wednesday on the capture): precision / recall of "row survives" from (cf, layer-L state) at
  layers 8 / 16 on dev; PASS: >= 60 % of dead rows found at layer 8 with <= 3 % live rows dropped.
- Gain: check -2.7..-5.6 ms (chain-32), more on trees; kept -0.1..-0.3 at 3 % false drops.

## 4. MISSES TO ZERO: never-wait or pool policy at 45 GB  [tried (FAR NW-BUILD, PASS forms on record); lossy]

- Mechanism. 5.3 ms of the lossy check is SSD misses on the critical path. FAR's never-wait writes a late demand-miss
  term through the next layer's inject gates (built, EvalSuite-gated forms: NW-SURE keeps 76-89 % of the gain); with
  1c-tiers the bytes per expert fall, so the pools hold more experts at the same 45 GB.
- Numbers [K]: 5.3 ms today; NW-SURE priced at -4 ms of it; FAR-G1 PASS / BETWEEN lines on record.
- Tag: tried (FAR NW-BUILD, DEFER-ALL KILL). Lossy (EvalSuite).
- Kills it: the quality gate (never-wait changes tokens) or the misses rising with the 127-node dumps' wider unions.
- Test: already defined by FAR; one 16-row A/B on the lossy stack with NW-SURE, 10 min.
- Gain: check -3..-5 ms.

## 5. HC mixers at 4 bits  [new on the dense side; lossy; 1-day]

- Mechanism. The dense pass reads 0.63 GB of 8-bit HC mixers per check (96 reads x 6.6 MB). At 4 bits that is 0.32 GB:
  -1.1 ms at 280 GB/s. The HC8-1P kernel's decode becomes the 4-bit code path qmm16n already has.
- Numbers [K]: HC8 passed its same-scores gate as a FAST read; 4 bits on the mixers is untested; the mixers are
  low-rank (320) and feed sigmoid gates, so errors are bounded by the gate's range.
- Tag: new. Lossy (EvalSuite).
- Kills it: flips on code above the 1.34 % line in a 16-row A/B.
- Test (<= 15 min): quantize the 96 mixers to 4-bit g64 offline, run EvalSuite's map screen on the 24 code windows.
- Gain: check -1.1 ms.

## 6. The honest ladder to Friday, and what 25 ms really needs

| step | check ms [P] | who |
|---|---|---|
| today, C5 two 16-row calls | 61-64 [K] | |
| one-pass 32-row form (Fit12's price) | 50-55 | Leap |
| + dispatch diet (idea 1): CPU bound lifted, GPU -3..-5 | 45-50 | Leap, with eprep / W_mlp / folds |
| + misses to zero (idea 4) | 41-45 | FAR / BigMoE |
| + row-dense expert tiles (idea 2) | 38-43 | Leap |
| + dead-row drop (idea 3), HC4 (idea 5) | 33-38 | Leap + capture |
| **measured by Friday, realistic** | **~35-40** | |

- 25 ms needs two more things that do not fit in 4 days: the dense pass at <= 1.3x its floor (~400 launches per check,
  i.e. the fused-layer kernels of DENSE_REBUILD v1, parked for the exact 16-row check but exactly right for a 32-row
  lossy one: 2-3 weeks), and experts within 1.1x of 15 ms. With both, 25-27 ms. Without them, no byte cut on the table
  (3-bit dense closed; lambda 1.8e-2 -0.35 GB) reaches 25.
- What I would change in the plan today: put the launch count on the scoreboard next to the bytes. It is the number
  that decides whether anything under 30 ms is measurable at all.
