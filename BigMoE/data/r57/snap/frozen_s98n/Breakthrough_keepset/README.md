# QOM + DRAFT17 keep-set rule (C's map rule), for BigMoE's tier call

Breakthrough, Wed 2026-11-04 (the orchestrator 15:1x). Import read-only; numpy plus MLX, no engine files.

## Files

- `qom_keep.py` holds the rule in two forms:
  - a numpy reference: `k0_from_cf`, `loads`, `keep`, `bits`, `kept_sets`, `t3_mask`, `touch_order`, `block`;
  - `qom_keep_mx`, the same rule on MLX arrays. It does no host sync and takes leading batch dimensions.
- `test_qom_keep.py` tests the rule:
  - it replays the decision map's masks, with the simulation ported from NewIdeas' `dm_oea_masks.py` and the rule taken
    from `qom_keep` only;
  - it checks bit-equality with the stored masks of C (`qom_0_live_bq_c147_t3`) and screen 6 (`qom_0_live_ball_c147_t3`)
    on all 36 windows;
  - it writes `qom_vectors.npz` and checks `qom_keep_mx` against the numpy rule on those vectors.
- `qom_vectors.npz` holds test vectors: inputs plus expected outputs, to check your wiring bitwise.

## The rule (per verify block, per MoE layer)

A block is the set of rows the engine computes together: the drafted rows and the last row.

1. **Sure row.** A row is sure if it verifies a draft whose confidence is `cf >= thr`. C uses `thr = 0`, so every row
   that verifies a draft is sure. Under DRAFT17 the last row verifies the 17th draft, so it is sure by the same rule. A
   row with no draft is exact.
2. **Loads.** Every row's ranks `< k0` go into the block's loads (its union), read from the SSD on a pool miss. `k0` is
   2 on a sure row and 10 on an exact row.
3. **Keep.**
   - An exact row keeps all 10 experts.
   - A sure row keeps ranks 0-1.
   - A sure row also keeps each rank 2-9 expert that is among the block's loads or resident in the pool. The others are
     the ones that would miss, and they are dropped.
4. **Weights.** The router's normalised top-10 weights times `keep`. Dropped ranks are zeroed, with no renormalisation.
   The map ran C with `renorm=False`.
5. **Bits** (C's tested form, the default):
   - a sure row's ranks 0-1 at **3 bits** (records3's MSB-3 part);
   - its kept ranks 2-9 at **4 bits**;
   - an exact row's 10 experts at 4 bits.

6. **Split (optional, off by default; Thu 2026-11-05, Fable C's audit test 1).** With `split=T`, a sure row whose
   confidence is `>= T` is *strict*. It keeps ranks 0-1 plus the block's loads (union members), and NO resident expert.
   Every other row is unchanged. Strict rows are bounded per block like the band tables; the other rows keep QOM's resident
   keep.
   - In the map's pool replay, a strict row touches only the experts it keeps. The engine does this naturally: it
     refreshes what it computes.
   - Under DRAFT17 the last row with no next draft carries cf 1.0 in the map's convention, so it is strict for any T.
   - Tested values: T = 0.9 and T = 0.99. On code both PASS on the map: 1.31 % / 0.92 % all-row flips vs 0.51 % for
     QOM-KEEP + DRAFT17, line 1.34 %.

Prompt (prefill) rows are outside this rule; they stay exact.

## Calling it from the engine

```python
import sys; sys.path.insert(0, "/Users/kamen/Breakthrough/keepset")
import qom_keep as Q
keep, weights, bits = Q.qom_keep_mx(ids, scores, conf, resident)
```

| argument | shape and type | meaning |
|---|---|---|
| `ids` | `[..., R, 10]` int | the block's routed experts per row |
| `scores` | `[..., R, 10]` | the router's normalised top-10 weights |
| `conf` | `[..., R]` float | the confidence of the draft each row verifies; NaN or `< thr` means no draft (exact) |
| `resident` | `[..., R, 10]` bool | the (row, rank) expert is in the layer's pool for this block |
| `valid` | `[..., R]` bool, optional | `False` marks padded rows: they load nothing and stay all-kept |
| `ranked` | default `True` | columns in rank order (rank 0 first). Use `False` for argpartition order: ranks then come from `argsort(-scores as float32)`, as the map's harness ranks them, and the outputs keep the input order |

Defaults: `thr=0.0`, `ks=2`, `top_bits=3`, `tail_bits=4`, `exact_bits=4`, `split=None` (off). For the split arms:
`qom_keep_mx(ids, scores, conf, resident, split=0.9)` (or `0.99`). The numpy side has `strict_rows(cf, k0, split)`, then
`keep(..., strict)` and `touch_order(..., strict, loads)`; `block(..., split=T)` does all of it.

Returns `keep [..., R, 10]` bool, `weights = scores * keep`, and `bits [..., R, 10]` int8 (0 means dropped). The cost is
one `[R, 10, R, 10]` equality test per block (28,900 booleans at R = 17).

## Answers to BigMoE's questions (15:2x)

- **Which confidence?** The drafter's confidence for the draft the row verifies: the HPD-0 walk's cf, which is the
  candidate selector's softmax at each pick, the quantity the running-confidence cut multiplies. At `thr = 0` only
  whether a draft exists matters. So pass any value `>= 0` for drafted rows (and the DRAFT17 last row) and NaN for rows
  without a draft. A separate row kind is not needed.
- **Residency.** Pass a bool per (row, rank).
  - The map's meaning is the layer's LRU pool *after* the block's loads, so an expert evicted by the block's own loads is
    not kept.
  - The engine's pre-lookup (slot or -1 per id), with the loader protecting what the block uses, keeps those too. So the
    engine keeps slightly more than the map: quality is at least the map's, SSD bytes are the same.
  - For a bitwise check against the vectors, pass the vectors' `resident`.
- **3 bits for every rank (no LSB reads)?** That is **not** C's tested form.
  - C served the kept ranks 2-9 at 4 bits (the map's stock path). In the engine that needs the LSB part of each kept
    resident.
  - `tail_bits=3` gives every kept expert of a sure row at 3 bits, with no LSB read on sure rows. Its map quality is arm
    `s7_ref3` in tonight's batch, every kept expert at MSB-3, but at the 239-slot pool, not 147.
  - **Decided: `tail_bits=3`** (NewIdeas 18:4x, d07577a).
    - FAST certifies against REF2-T, which already serves ranks 2-9 at 3 bits, so 3-bit tails are the like-for-like
      choice (NewIdeas 7fac438).
    - On the map at 239 slots, s7_ref3 vs s7_ref = +0.30 / +1.14 / +1.01 points of flips (code / IFEval / multi). That is
      REF2-T's own cost against the 4-bit model, not a FAST cost (Breakthrough ba53ede).
    - The module's defaults stay C's tested form (the test vectors' bits), so pass `tail_bits=3`.
- **Parameters.** `thr = 0` (every drafted row is sure); ranks 0-1 are always kept (`ks = 2`); DRAFT17's last row is sure
  by the same rule.
- **Rank order.** Rank 0 is the highest router weight. Your tier path holds argpartition order, so pass `ranked=False`.

## Test vectors (`qom_vectors.npz`)

Every block × 48 layers of the first window per domain, from the replay of screen 6's arm (`qom_0_live_ball_c147_t3`,
147-slot pools, DRAFT17), padded to 17 rows. Expected outputs apply to valid rows only; the expected weights are
`scores * keep`.

| key | contents |
|---|---|
| `ids` | `[V, 17, 10]` int32; -1 on padded rows |
| `scores` | `[V, 17, 10]` float32, synthetic, descending per row |
| `conf` | `[V, 17]` float32; -1 = exact row, NaN = padded |
| `resident` | `[V, 17, 10]` bool, the map's pool after the block's loads |
| `valid` | `[V, 17]` bool |
| `keep` | `[V, 17, 10]` bool, expected |
| `bits` | `[V, 17, 10]` int8, expected (C's form) |
| `layer`, `domain`, `window` | metadata |

## Results [K] (Wed 15:2x, test_run_1520.log; CPU, 50 s, 0.7 GB)

- C, `qom_0_live_bq_c147_t3`: **36/36 windows bit-equal** to NewIdeas' stored masks.
- Screen 6, `qom_0_live_ball_c147_t3` (C + DRAFT17): **36/36 windows bit-equal**.
- `qom_keep_mx` on the 14,256 test vectors:
  - keep and bits equal the numpy rule;
  - weights equal `scores * keep` exactly;
  - with columns shuffled per row and `ranked=False`, the results are the same.
- What the rule keeps, per sure row-layer (C's form, 147 slots):

  | domain | 3-bit (ranks 0-1) | kept 4-bit (ranks 2-9) | dropped | exact row-layers (C / C + DRAFT17) |
  |---|---|---|---|---|
  | code | 2 | 6.07 (5.89 with DRAFT17) | 1.93 (2.11) | 3,984 / 0 |
  | IFEval | 2 | 6.32 | 1.68 | 48 / 0 |
  | multi | 2 | 6.43 | 1.57 | 0 / 0 |

  C's exact row-layers are last rows with no next draft; DRAFT17 gates them all.

## Results [K], the split (Thu 11-05 10:39-10:43, test_run_split_1039.log; CPU, 4.5 min)

The replay now also takes the 45 GB pool: per-layer sizes from `~/MoonShots/data/laysz45_256.json`'s `C_m`, mean 295 of
512, as NewIdeas' builder uses. Each arm below is bit-equal on 36/36 windows to NewIdeas' stored masks.

| arm | what it is | windows bit-equal |
|---|---|---|
| `qom_0_live_bq_c147_t3` | C, the regression | 36/36 |
| `qom_0_live_ball_c147_t3` | screen 6, the regression | 36/36 |
| `qom_0_live_ball` | QOM-KEEP + DRAFT17 at 45 GB, the split arms' twin | 36/36 |
| `qom_0_split0.9_live_ball` | split T = 0.9 | 36/36 |
| `qom_0_split0.99_live_ball` | split T = 0.99 | 36/36 |
| `qom_0_split0.9_live_bq`, `qom_0_split0.99_live_bq` | the bq forms | 36/36 each |

`qom_keep_mx(split=0.9)` and `qom_keep_mx(split=0.99)` each equal the numpy rule with strict rows on 14,256 / 14,256 test
vectors (35,136 and 22,560 strict row-layers). The vector file is unchanged in content (regenerated by the same replay).
