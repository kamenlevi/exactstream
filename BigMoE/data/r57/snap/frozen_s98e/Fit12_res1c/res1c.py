"""res1c.py (Fit12, Mon 16 Nov 2026 13:5x; WEEK98 pillar 2; NewIdeas 13:4x GO; pre-registered in
/Users/kamen/Fit12/PLAN_12GB_90_fit12.md "RES1C INSTALLER"). ONE opt-in installer with two switches, on top of BigMoE's
1c-tiers (R57_TIERS=1c, screen t1c1) and Leap's ref2 (b43efc4). Neither of their files is edited.

  COMQOM  On a 1c-tiers check, rows tagged by lossy-45's causal CodeScan ('#' comments; docstrings too on the sets in
          R57_COMQOM_DOC_SETS) run QOM cf 0 instead of the tiers: their rank 0-1 uses read the base's precision
          (LSB + MSB), their rank 2-9 uses are bit-21 OPTIONAL (served only when resident), and no tier drop applies to
          them.
  RES     Untagged rows' MSB-3-tier uses are bit-21 optional: an MSB-3-tier expert is read only when resident.

Mechanics
  1. ref2._ooo_1c_call is replaced by a source-transformed copy (md5-guarded on the original function's source):
     - it admits t["qom"] next to a 2-D t["nolsb"] only while ref2.F12_QOM2D[0] is set (Leap's guard is otherwise
       unchanged);
     - it counts, per call, the optional uses posted and those the loader resolved to -2 (ref2.F12_CNT, lazily).
     With the switch off and F12_CNT None, it runs the original's code path line for line.
  2. The screen's _t1c (BigMoE, md5-guarded) is replaced. With both switches off it calls the original (bitwise); on,
     it computes the same tiers (tiers_mx) and applies masks() below.
  3. Per verify check the caller passes the rows' tags: check_begin(scan, dtexts) runs a copy of the run's CodeScan over
     the drafts (R57_RELAX_SPAN's convention: row j is tagged by the state before draft j; the last row after all
     drafts) and returns FG["crow"]. check_end() folds the counters; prompt_counts() returns them for runs.json
     lossy_check.
Scope by set (comma lists, empty = off): R57_COMQOM_SETS, R57_COMQOM_DOC_SETS, R57_RES_SETS. set_prompt(fset) per item.

Usage, in a screen after ref2.apply() and the 1c wiring:
    import res1c; res1c.install(globals(), ref2)
    per prompt:       res1c.set_prompt(fset)
    per verify check: FG["crow"] = res1c.check_begin(sscan, dtexts)   (before the model call)
                      res1c.check_end()                               (after the check's readback)
    per prompt end:   lossy_check.update(res1c.prompt_counts())
"""
import hashlib
import inspect
import os

import mlx.core as mx

REF2_FN_MD5 = "19d399d8a01890735a3f6c907046ea3c"      # ref2._ooo_1c_call, Leap b43efc4 (frozen in BigMoE's t1c1)
T1C_FN_MD5 = "12a26f5a0d1b350ad67549323fc08a76"       # BigMoE's screen _t1c (t1c1, 68274ff)
GUARD_OLD = 't.get("qom") is not None or _NATIVE_FLAGS'
GUARD_NEW = '(t.get("qom") is not None and not F12_QOM2D[0]) or _NATIVE_FLAGS'
KERN_LINE = "    y, rs_m = _kern(xin, pre_m, pre_l, sc)\n"
KERN_ADD = ("    if F12_CNT is not None and qm is not None and nl2d:   # Fit12 res1c: optional uses posted / resolved -2\n"
            "        F12_CNT.append(mx.stack([qm[0].reshape(M, K).sum().astype(mx.int32),\n"
            "                                 ((rs_m.reshape(M, K) == -2) & qm[0].reshape(M, K)).sum().astype(mx.int32)]))\n")


def _sets(name):
    return {x for x in os.environ.get(name, "").split(",") if x}


SETS = dict(com=_sets("R57_COMQOM_SETS"), doc=_sets("R57_COMQOM_DOC_SETS"), res=_sets("R57_RES_SETS"))
ON = dict(com=False, doc=False, res=False)
_S = dict(installed=False, ref2=None, orig_t1c=None, FG=None, lazy=[], tot=None)
_ZERO = dict(f12_checks=0, com_rows=0, com_opt=0, res_opt=0, opt_post=0, opt_drop=0)


def masks(tier, com, res):
    """the switches' rule, shared by the screen path and the bit tests.
    tier [M, K] int (0 4-bit / 1 MSB-3 / 2 drop, BigMoE's tiers_mx), com [M] bool (tagged rows), res bool ->
    (km [M, K]: the use is not tier-dropped; nolsb [M, 2]: the rank 0-1 use reads MSB-3 only; opt [M, K]: bit-21
    optional, or None when nothing is optional)"""
    M_, K_ = tier.shape
    c2 = com.reshape(M_, 1)
    rk = mx.arange(K_).reshape(1, K_)
    km = (tier != 2) | c2                              # a tagged row keeps every use (QOM cf 0 drops only by residency)
    nl = (tier[:, :2] != 0) & ~c2                      # a tagged row's rank 0-1 uses read LSB + MSB (base precision)
    opt = c2 & (rk >= 2)                               # a tagged row's rank 2-9 uses: served only when resident
    if res:
        opt = opt | ((tier == 1) & ~c2)                # RES: an untagged row's MSB-3-tier use: only when resident
    opt = mx.where(km, opt, opt[:, :1])                # a tier-dropped use is posted as its row's rank-0 id (weight 0):
    return km, nl, opt                                 # it takes that slot's optionality, so it forces no extra read


def _transform_ref2(ref2):
    if getattr(ref2, "_F12_ORIG_OOO1C", None) is not None:
        return
    fn = ref2._ooo_1c_call
    src = inspect.getsource(fn)
    md5 = hashlib.md5(src.encode()).hexdigest()
    assert md5 == REF2_FN_MD5, f"res1c: ref2._ooo_1c_call changed (md5 {md5}, expected {REF2_FN_MD5}): re-derive the transform"
    assert src.count(GUARD_OLD) == 1 and src.count(KERN_LINE) == 1, "res1c: the transform's anchors moved"
    new = src.replace(GUARD_OLD, GUARD_NEW).replace(KERN_LINE, KERN_LINE + KERN_ADD)
    ref2.F12_QOM2D = [False]
    ref2.F12_CNT = None
    ns = {}
    exec(compile(new, f"{ref2.__file__} [Fit12 res1c transform of _ooo_1c_call]", "exec"), ref2.__dict__, ns)
    ref2._F12_ORIG_OOO1C = fn
    ref2._ooo_1c_call = ns["_ooo_1c_call"]


def _install_t1c(g):
    orig = g["_t1c"]
    src = inspect.getsource(orig)
    md5 = hashlib.md5(src.encode()).hexdigest()
    assert md5 == T1C_FN_MD5, f"res1c: the screen's _t1c changed (md5 {md5}, expected {T1C_FN_MD5})"
    FG, T1C, s_, lam_ = g["FG"], g["T1C"], g["TIERS_S"], g["TIERS_LAM"]

    def _t1c_f12(self_, indices):
        if not (ON["com"] or ON["res"]):
            return orig(self_, indices)                # both switches off: BigMoE's own function, bit for bit
        K_ = indices.shape[-1]; ids2 = indices.reshape(-1, K_)
        M_ = ids2.shape[0]
        t_ = self_._bm["tier"]
        assert t_.get("scores") is not None, "R57_TIERS=1c needs the fused combine's scores (BIGMOE_TIER_COMBINE)"
        tier = T1C.tiers_mx(ids2, t_["scores"].reshape(-1, K_), s=s_, lam=lam_)
        texp = mx.full((T1C.N_EXPERTS,), 3, mx.int32).at[ids2.reshape(-1)].minimum(tier.reshape(-1).astype(mx.int32))
        FG["t1cn"].append(mx.stack([(texp == 0).sum(), (texp == 1).sum(), (texp == 2).sum()]))   # as BigMoE counts
        cr = FG.get("crow")
        com = cr if (ON["com"] and cr is not None and cr.shape[0] == M_) else mx.zeros((M_,), mx.bool_)
        km, nl, opt = masks(tier, com, ON["res"])
        t_["nolsb"] = (nl, None)
        t_["qom"] = (opt, mx.ones((M_,), mx.bool_), None)
        c2 = com.reshape(M_, 1)
        _S["lazy"].append(mx.stack([(opt & c2).sum().astype(mx.int32), (opt & ~c2).sum().astype(mx.int32)]))
        return km.reshape(indices.shape)

    _S["orig_t1c"] = orig
    g["_t1c"] = _t1c_f12


def install(screen_globals, ref2):
    """after ref2.apply() and the screen's 1c wiring (its module-level _t1c, FG, T1C, TIERS_S, TIERS_LAM)"""
    if _S["installed"]:
        return info()
    _transform_ref2(ref2)
    _install_t1c(screen_globals)
    _S.update(installed=True, ref2=ref2, FG=screen_globals["FG"], tot=dict(_ZERO))
    return info()


def info():
    return dict(installed=_S["installed"], sets={k: sorted(v) for k, v in SETS.items()}, ref2_fn_md5=REF2_FN_MD5,
                t1c_fn_md5=T1C_FN_MD5)


def set_prompt(fset):
    """per item: the switches for this prompt's set (COMQOM / docstrings / RES), and ref2's 2-D + qom admission"""
    ON["com"] = fset in SETS["com"]
    ON["doc"] = ON["com"] and fset in SETS["doc"]
    ON["res"] = fset in SETS["res"]
    if _S["ref2"] is not None:
        _S["ref2"].F12_QOM2D[0] = ON["com"] or ON["res"]
    _S["tot"] = dict(_ZERO)
    return dict(ON)


def row_tags(scan, dtexts, doc=False):
    """the check's rows (the current token + each draft): row j by the tagger's state before draft j, the last row after
    every draft (R57_RELAX_SPAN's convention; scan = the run's CodeScan after the committed text, not changed here)"""
    s = scan.copy()
    tag = lambda: bool(s.in_comment() or (doc and s.code and s.doc))
    out = []
    for x in dtexts:
        out.append(tag())
        s.feed(x)
    out.append(tag())
    return out


def check_begin(scan, dtexts):
    """before a verify check: the rows' tags (or None) for FG["crow"], and the one-call counters armed"""
    if not (ON["com"] or ON["res"]):
        return None
    _S["ref2"].F12_CNT = []
    _S["lazy"] = []
    _S["tot"]["f12_checks"] += 1
    if not ON["com"] or scan is None:
        return None
    tags = row_tags(scan, dtexts, ON["doc"])
    _S["tot"]["com_rows"] += sum(tags)
    return mx.array(tags, dtype=mx.bool_)


def check_end():
    """after the check's readback: fold this check's lazy counters into the prompt's totals"""
    r2 = _S["ref2"]
    if r2 is None or r2.F12_CNT is None:
        return
    cnt, lz = r2.F12_CNT, _S["lazy"]
    r2.F12_CNT, _S["lazy"] = None, []
    if lz:
        a = mx.stack(lz).sum(axis=0)
        mx.eval(a)
        _S["tot"]["com_opt"] += int(a[0].item()); _S["tot"]["res_opt"] += int(a[1].item())
    if cnt:
        b = mx.stack(cnt).sum(axis=0)
        mx.eval(b)
        _S["tot"]["opt_post"] += int(b[0].item()); _S["tot"]["opt_drop"] += int(b[1].item())


def prompt_counts():
    """the prompt's counters for runs.json lossy_check (all 0 when the switches were off on this prompt)"""
    t = dict(_S["tot"] or _ZERO)
    _S["tot"] = dict(_ZERO)
    return t
