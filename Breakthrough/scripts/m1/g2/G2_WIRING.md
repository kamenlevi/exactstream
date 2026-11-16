# G2 wiring: V1 (R57_LMSL) and V3 (R57_HC4) for BigMoE's g2a snapshot

Breakthrough, Mon 16 Nov 2026 13:xx. This is for BigMoE's g2a snapshot, which takes EvalSuite's G2 (lossygate
amendment 5). It mirrors how int8d is wired in g1a (`~/BigMoE/data/r57/snap/g1a/r57_screen_g1a.py`, "S:" line numbers
below).
- Installers: `~/Breakthrough/scripts/m1/g2/lmsl.py` (V1) and `hc4.py` (V3).
- V3's read: Leap's `~/Leap/scripts/week98/hc4i.py` + `hc4r.py` (7c3e149).
- Copy all of them into the snapshot's frozen tree, as int8d.py was.

## Env

- `R57_LMSL=1`, `R57_LMSL_SETS=humaneval,humanevalx,mbpp`, and optionally `R57_LMSL_IDS=<npy>`. The default is
  `~/Breakthrough/data/m1/lmsl_code32k.npy`: 32,768 ids, sha256 0aa829881dbe, out of sample of HumanEval / MBPP
  (docs/CHECK_BYTES_900.md section 3).
- `R57_HC4=1`, `R57_HC4_SETS=humaneval,humanevalx,mbpp`.
- Both are code-scoped. Off the sets, every call is the g1a path bit for bit, so IFEval keeps token identity.

## Load time

1. **V3 copies, just before `hc8r.hc8_install(model)` (S:945).** The bf16 originals still exist there, and the DNR
   conversion (S:941-944) has run.
   ```python
   if HC4_ON:
       import hc4 as HC4
       c4, mb4 = HC4.prepare(model)        # {'mix': 194, 'inject': 96, 'ple': 2} on the 125B; count mb4 in MEM
       say(f"HC4: 4-bit copies {c4}, +{mb4:.0f} MB beside HC8")
   ```
   `prepare()` asserts that the targets are still the bf16 `nn.Linear`. If it runs after HC8, it stops.
2. **Routers and the head list, in the int8d block (S:2139), BEFORE `I8.install()`.** SVH (S:985-991) is up by then.
   ```python
   if HC4_ON:
       import hc4i
       ref_on(True); hc4i.install(); HC4.install(); ref_on(False)
   if LMSL_ON:
       import lmsl as LM
       ref_on(True); n_, mb_ = LM.install(model.language_model.lm_head, os.environ.get("R57_LMSL_IDS") or None); ref_on(False)
       say(f"LMSL: {n_} code ids, +{mb_:.0f} MB")
   ```

## Per item, in run()

- **Before the prefill, beside `I8.ON[0] = False` (S:1152-1153):** `LM.ON[0] = False` and `HC4.engage(False)`.
- **After the prefill, beside `i8_on_` (S:1238-1242):**
  ```python
  lm_on_ = LM is not None and fset in LMSL_SETS;  LM.ON[0] = lm_on_
  hc4_on_ = HC4 is not None and fset in HC4_SETS; HC4.engage(hc4_on_)
  ```
- **Each check, right before `with ref2.verify_scope(model):` (S:1529):** `if lm_on_: LM.add_drafts(dr)`.
  - This puts the check's drafts in the list's session rows. SVH never adds drafts.
  - Cost: one small gather + eval per check.
- **lossy_check (S:1792):** merge AFTER t1c_, so the real counts replace its zero placeholders:
  ```python
  lossy_check={**(t1c_ if t1c_on_ else {}), **(_i8_take(i8c0_) if i8_on_ else {}),
               **(LM.take() if lm_on_ else {}), **(HC4.take() if hc4_on_ else {})} or None
  ```
  - `LM.take()` -> `{lmsl_rows, lmsl_fallback}`, and turns ON off.
  - `HC4.take()` -> `{hc4: 1}`, and swaps the originals back.
  - The warm-up run has no set, so it never engages.

## What each piece does

- **V1:**
  - A second SVH state over the 32K code list, plus the session ids SVH's own feed adds, plus this prompt's drafts.
  - It applies on the trunk head's <= 16-row calls in the verify phase. Logits are -inf off the union.
  - `lmsl_fallback` counts the rows whose top-2 over the union are a tie (p1 - p2 <= 0.1). It is counted on the GPU
    only; there is no acting fallback in v1.
  - Forced misses on the map's 24 code windows: 0.14 % of rows (no drafts).
- **V3:**
  - Per item, the 194 mixers, 96 injects and 2 PLE projections swap to 4-bit copies made from the bf16 originals,
    exactly as the map's bt_mix4.
  - hc4i routes the 4-bit-mixer reads to hc4r. The PLE copies are DNR. The final mixer takes MTPLX's stock eager read.
  - ref2's and hc4i's per-module caches check weight identity, so a swap only rebuilds small dicts.
- **MEM:** + the V3 copies (`mb4`, ~0.3-0.4 GB) + the V1 list state (~53 MB).

## Tests (Breakthrough's)

- `test_hc4.py` (CPU), ALL PASS:
  - copies == dm_d2._requant bit for bit;
  - only the PLE copies are DNR;
  - swap / restore by identity;
  - prepare() refuses after HC8.
- `test_lmsl.py` (GPU, real lm_head):
  - the unset path bitwise;
  - engaged: finite exactly on the union, values within 2e-2 of a float32 reference, the argmax right;
  - the feed and drafts;
  - the counters;
  - `_qlin` on the head;
  - remove().
- Leap's 4-layer slice: the hc4i routing with this switch (Leap offered, Tue).
- BigMoE's real-loop key test on g2a stays the gate before EvalSuite arms G2.
