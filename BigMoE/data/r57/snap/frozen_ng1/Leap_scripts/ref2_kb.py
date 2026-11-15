"""REF2 key buckets (R-40 STEP 4c; Leap, Fri 2026-10-30): past ~512 keys MLX 0.32.2's full (NAX) attention kernel gives a
row of a 16-row call one of two bit patterns depending on the call's key length N (STEP 4b, real data: the pattern
flips irregularly over N = 536..1100, and a row is equal whenever both calls run at the same N). REF2's AR (one real
row, N = p + 16) and a check (N = start + 16) then disagree on rows >= 2 (STEP 4a). Fix: every 16-row QSA attention
call with N >= 512 keys runs at N_eff = the next multiple of 512 above N (k / v zero-padded, mask False); rows whose
AR step falls in another class are recomputed in that class (ref2's STEP 3m split, generalized). Classes: N < 128;
128 <= N < 512 (both N-invariant, measured); then one per 512-key bucket. Below 512 keys nothing changes; past it
REF2's AR bits change (a new reference past 512 keys).
Usage: import ref2, ref2_kb; ref2_kb.install() before ref2.apply() (ref2.apply then installs this wrapper; ref2.remove
restores MTPLX's own). KB["on"] = False falls back to ref2's 128-key split alone (the control)."""
import mlx.core as mx
import ref2

KB = {"on": True, "from": 512, "step": 512}
COUNTS = dict(kb_pad=0, kb_split=0)
_R2 = {}

def key(N):
    """the run class of a 16-row call with N keys: 0 (N < 128), 1 (128 <= N < 512), else the bucket's run length"""
    if N < 128:
        return 0
    if N < KB["from"]:
        return 1
    return (N // KB["step"] + 1) * KB["step"]

def _at(q, k, v, mask, scale, n):
    """MTPLX's own SDPA with the keys zero-padded to n and the mask extended (False / -inf)"""
    orig = ref2._ORIG["vsdpa"]
    N = k.shape[2]
    if n == N:
        return orig(q, k, v, scale=scale, mask=mask)
    pad = n - N
    kp = mx.concatenate([k, mx.zeros(k.shape[:2] + (pad, k.shape[3]), k.dtype)], axis=2)
    vp = mx.concatenate([v, mx.zeros(v.shape[:2] + (pad, v.shape[3]), v.dtype)], axis=2)
    fill = mx.zeros(mask.shape[:-1] + (pad,), mx.bool_) if mask.dtype == mx.bool_ else \
        mx.full(mask.shape[:-1] + (pad,), -float("inf"), mask.dtype)
    return orig(q, kp, vp, scale=scale, mask=mx.concatenate([mask, fill], axis=-1))

def _verify_sdpa(q, k, v, *, scale, mask):
    """ref2's _verify_sdpa with key buckets past 512 keys (see the module doc)"""
    if not KB["on"]:
        return _R2["fn"](q, k, v, scale=scale, mask=mask)
    S, N = q.shape[2], k.shape[2]
    if not ref2._ON[0] or not ref2.CFG["sdpa_classes"] or S != ref2.R or mask is None:
        return ref2._ORIG["vsdpa"](q, k, v, scale=scale, mask=mask)
    run = lambda c: N if c == 0 else max(N, 128) if c == 1 else c
    c0 = key(N)
    cls = [key(N + j) for j in range(S)]
    out = _at(q, k, v, mask, scale, run(c0))
    if c0 > 1:
        COUNTS["kb_pad"] += 1
    for c in sorted(set(cls) - {c0}):
        yp = _at(q, k, v, mask, scale, run(c))
        sel = mx.array([cls[j] == c for j in range(S)])[None, None, :, None]
        out = mx.where(sel, yp, out)
        ref2.COUNTS["sdpa_split"] += 1
        COUNTS["kb_split"] += 1
    return out

def install():
    """make ref2.apply() install this wrapper (call before ref2.apply())"""
    if "fn" not in _R2:
        _R2["fn"] = ref2._verify_sdpa
    ref2._verify_sdpa = _verify_sdpa
