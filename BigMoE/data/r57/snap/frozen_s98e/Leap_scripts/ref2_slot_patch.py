"""R-50 SLOTCALL's adoption into REF2 (prepared Thu 2026-10-29 20:25; to be merged into ref2.py after STEP 3v and the
STEP 3y-fix retry have run, so their pre-registered harness doesn't change under them). Merge plan:
  1. paste _sync_slots below _switch_sorted in ref2.py
  2. in apply()'s BigMoE block:  _ORIG["sync"] = BS._sync_call; BS._sync_call = _sync_slots; _BS[0] = BS
  3. COUNTS gains slot_calls; the module docstring gains one line.
R-50 (data/r50_screen.txt): -384 ms per 16-row check, bit-equal. BigMoE's plain _sync_call gathers each layer's experts
into a fresh scratch pool (9 mx.take + concatenate + clear_cache, ~330 MB per call) to keep MLX's sorted gather on the
expert ORDER. REF2's expert kernel computes every (row, expert) pair on its own, so a <= 16-row call can run on the
pool itself with slot ids, as the gated path already does. Under PF1 (EvalSuite's env) the prefill's staging is dropped
first, as the engine's first decode call does; that also keeps checks off PF1's stale-staging path (3y)."""
import numpy as np
import mlx.core as mx

_BS = [None]

def _sync_slots(cls, self, x, indices, bm, *a, **kw):
    from ref2 import _ON, CFG, R, COUNTS, _ORIG, _rows
    BS = _BS[0]
    lead, rows = _rows(x)
    if not _ON[0] or not CFG["experts"] or rows > R or "tier" in bm or bm.get("mtp"):
        return _ORIG["sync"](cls, self, x, indices, bm, *a, **kw)    # > 16 rows: stock MLX needs the expert order
    ids = np.array(indices)                                           # the shipped path's id sync
    uniq = np.unique(ids)
    if len(uniq) > int(bm["C"] * 0.9):                                # the shipped path would go in groups
        return _ORIG["sync"](cls, self, x, indices, bm, *a, **kw)
    if BS._STAGE["bufs"] is not None:                                 # PF1's prefill staging: drop it, as the
        BS._stage_drop()                                              # engine's first gated call does
    BS.SYNC["calls"] += 1; BS.SYNC["rows"] += ids.size // ids.shape[-1]
    slots = np.array(BS.ext.ensure(bm["L"], mx.array(uniq.astype(np.int32))))
    sl = mx.array(slots[np.searchsorted(uniq, ids)].astype(np.int32)).astype(indices.dtype)
    y = cls.__call__(self, x, sl, *a, **kw)                           # -> _switch_base -> REF2's experts_split
    mx.eval(y)                                                        # before a later ensure of L can evict slots
    COUNTS["slot_calls"] = COUNTS.get("slot_calls", 0) + 1
    return y
