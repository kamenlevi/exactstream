"""lmsl.py (Breakthrough, Mon 16 Nov 2026 13:xx; V1 = R57_LMSL for EvalSuite's G2, lossygate amendment 5; CHECK BYTES,
~/Breakthrough/docs/CHECK_BYTES_900.md): the CODE lm_head shortlist for the verify check, as an opt-in installer on top of
Leap's SVH (ref2 untouched), in the form of Leap's int8d.py.

What changes (FAST, lossy; EvalSuite gates it; adopting it is Kamen's call):
  - install(lm_head) builds a SECOND screened-head state over the code list (default ~/Breakthrough/data/m1/
    lmsl_code32k.npy: the Magicoder bank's top-1 tokens + FR-Spec fill, 32,768 ids, out of sample of HumanEval / MBPP),
    in ref2's SVH layout (base rows gathered once + `cap` session rows), and registers it in ref2._SH under its own key.
    SVH's own feed (ref2.screened_head_add / _reset via SessionVocab: the prompt, the first token, committed ids) loops
    over every _SH state, so the session ids reach this state too.
  - ref2._screened is wrapped: while ON[0] and in the verify phase (mtplx 'decode_verify': the check's forward and its
    commit replay; never the prefill or the drafter), the SVH call on the trunk's head runs over THIS state: logits on
    (code list U session ids U this prompt's drafts), -inf elsewhere, so the argmax is over that union.
  - add_drafts(ids): each check's drafted tokens join this state's session rows before the check (SVH never adds drafts,
    so a draft outside its set gets -inf); call it with the check's draft list right before the verify.
  - COUNTS / take(): lmsl_rows = rows scored over the list; lmsl_fallback = of those, rows whose top-2 over the union are
    a tie by the map's rule (p1 - p2 <= 0.1): where a fallback to the full head would act. Counted on the GPU (lazy, no
    host sync per check); v1 has no acting fallback (Leap 13:1x: a margin test would need a host sync or a second head pass).
Unchanged: every call with ON[0] off (SVH's own state, bit for bit), any call outside the verify phase, > 16-row calls
(SVH's own rule: they take the full head).
Memory: the list's head rows (32,768 x 2,560 4-bit + scales / biases: ~47 MB) + cap session rows (~5.9 MB at 4,096).
Scope: BigMoE's screen sets ON[0] per item (R57_LMSL_SETS, like R57_INT8D_SETS), after the prefill.
  install(lm_head) after SessionVocab(svh=True); ON[0]; add_drafts(dr) before each check; take(); remove()"""
import os
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
DEFAULT_IDS = os.path.expanduser("~/Breakthrough/data/m1/lmsl_code32k.npy")
ON = [False]
COUNTS = dict(lmsl_rows=0, calls=0, drafts_added=0, passthru=0)
_FB = [mx.array(0, mx.int32)]                    # lmsl_fallback since the last take (a GPU scalar, read at take())
_TAKEN = dict(lmsl_rows=0)
_ORIG = {}
_ST = {}
_PH = [None]
R2 = [None]


def _verify_phase():
    if _PH[0] is None:
        from mtplx.attention_context import current_attention_phase
        _PH[0] = current_attention_phase
    return _PH[0]() == "decode_verify"


def _add(e, token_ids):
    """ref2.screened_head_add's body for ONE state: ids not in the state's set join its session rows (up to cap)"""
    new = [t for t in dict.fromkeys(int(t) for t in np.asarray(token_ids).reshape(-1)) if t not in e["set"]]
    new = new[:max(0, e["cap"] - len(e["extra"]))]
    if not new:
        return 0
    a, n = len(e["extra"]), len(new)
    idx = mx.array(np.asarray(new, np.int32))
    e["xids"][a:a + n] = idx
    e["xw"][a:a + n] = mx.take(e["w"], idx, axis=0)
    e["xs"][a:a + n] = mx.take(e["s"], idx, axis=0)
    e["xb"][a:a + n] = mx.take(e["b"], idx, axis=0)
    mx.eval(e["xids"], e["xw"], e["xs"], e["xb"])
    e["extra"] += new
    e["set"].update(new)
    return n


def _screened(e, x, scales, biases):
    st = _ST
    if ON[0] and st and e is st["e"] and _verify_phase():
        y = _ORIG["screened"](st["e2"], x, scales, biases)
        if y is not None:
            yf = y.reshape(-1, y.shape[-1]).astype(mx.float32)
            t2 = mx.topk(yf, 2, axis=-1)
            p = mx.exp(t2 - mx.logsumexp(yf, axis=-1, keepdims=True))
            _FB[0] = _FB[0] + mx.sum(mx.abs(p[:, 0] - p[:, 1]) <= 0.1).astype(mx.int32)
            mx.async_eval(_FB[0])
            COUNTS["lmsl_rows"] += int(yf.shape[0])
            COUNTS["calls"] += 1
            return y
    COUNTS["passthru"] += 1
    return _ORIG["screened"](e, x, scales, biases)


def install(lm_head, ids_path=None, ref2_module=None):
    """after SessionVocab(svh=True) registered SVH on lm_head: build and register the list's state, wrap ref2._screened"""
    if ref2_module is None:
        import ref2 as ref2_module
    r2 = R2[0] = ref2_module
    w, sc, bi = lm_head["weight"], lm_head["scales"], lm_head["biases"]
    e = r2._SH.get(id(w))
    assert e is not None and e["w"] is w, "R57_LMSL: SVH first (SessionVocab(svh=True) on this lm_head)"
    ids = np.load(ids_path or DEFAULT_IDS).astype(np.int32)
    assert ids.ndim == 1 and ids.size % 32 == 0 and np.all(np.diff(ids) > 0) and 0 <= ids[0] and ids[-1] < w.shape[0], \
        "R57_LMSL: the list must be sorted unique ids, a multiple of 32, inside the vocabulary"
    base = mx.array(ids)
    cap = int(e["cap"])
    e2 = dict(w=w, s=sc, b=bi, V=int(e["V"]), ids=base, set=set(ids.tolist()), extra=[], cap=cap,
              wb=mx.contiguous(mx.take(w, base, axis=0)), sb=mx.contiguous(mx.take(sc, base, axis=0)),
              bb=mx.contiguous(mx.take(bi, base, axis=0)),
              xids=mx.zeros((cap,), mx.int32), xw=mx.zeros((cap, w.shape[1]), w.dtype),
              xs=mx.zeros((cap, sc.shape[1]), sc.dtype), xb=mx.zeros((cap, bi.shape[1]), bi.dtype))
    mx.eval(e2["ids"], e2["wb"], e2["sb"], e2["bb"], e2["xids"], e2["xw"], e2["xs"], e2["xb"])
    key = ("lmsl", id(w))
    r2._SH[key] = e2
    if e["extra"]:                                   # installed after a prompt was fed: carry SVH's session ids over
        _add(e2, e["extra"])
    _ST.clear()
    _ST.update(e=e, e2=e2, key=key)
    if not _ORIG:
        _ORIG["screened"] = r2._screened
        r2._screened = _screened
    return int(ids.size), sum(e2[k].nbytes for k in ("wb", "sb", "bb", "xw", "xs", "xb")) / 1e6


def add_drafts(token_ids):
    """this check's drafted tokens join the list state's session rows (before the check); no-op while ON[0] is off"""
    if not (ON[0] and _ST):
        return 0
    n = _add(_ST["e2"], token_ids)
    COUNTS["drafts_added"] += n
    return n


def take():
    """this item's counters since the last take (one host read of the GPU fallback count), then ON[0] off"""
    ON[0] = False
    fb = int(_FB[0].item())
    _FB[0] = mx.array(0, mx.int32)
    rows = COUNTS["lmsl_rows"] - _TAKEN["lmsl_rows"]
    _TAKEN["lmsl_rows"] = COUNTS["lmsl_rows"]
    return dict(lmsl_rows=rows, lmsl_fallback=fb)


def remove():
    r2 = R2[0]
    if _ORIG and r2 is not None:
        r2._screened = _ORIG["screened"]
        _ORIG.clear()
    if _ST and r2 is not None:
        r2._SH.pop(_ST["key"], None)
    _ST.clear()
    ON[0] = False
