"""patch_c25_t1cf.py (Leap, Mon 16 Nov 2026 12:5x): adds the fused 1c-tiers kernel (scripts/week98/t1cf.py; == BigMoE's
tiers_mx + glue on 384,000 / 384,000 slots, t1cf_check.txt) to c25_loop.py as an opt-in (C25_T1CF[0], set per arm by
c25_engine): _t1c returns None and leaves the km'd ids in FG["t1c_ids"]; _fast_call then skips its where lines and
passes those ids on. Anchored; each anchor must match once."""
import os
P = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts/week98/c25/c25_loop.py")
s = open(P).read()
R = [
("import os as _c25os, sys as _c25sys\n",
 "import os as _c25os, sys as _c25sys\nC25_T1CF = [False]                                     # C25: the fused 1c-tiers kernel (c25_engine sets it per arm)\n"),
('''        K_ = indices.shape[-1]; ids2 = indices.reshape(-1, K_)
        t_ = self_._bm["tier"]
        assert t_.get("scores") is not None, "R57_TIERS=1c needs the fused combine's scores (BIGMOE_TIER_COMBINE)"
''', '''        K_ = indices.shape[-1]; ids2 = indices.reshape(-1, K_)
        t_ = self_._bm["tier"]
        assert t_.get("scores") is not None, "R57_TIERS=1c needs the fused combine's scores (BIGMOE_TIER_COMBINE)"
        if C25_T1CF[0]:                                  # C25: ONE dispatch for the tiers, counter, 2-D nolsb and km path
            _sys0 = _c25os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts/week98")
            if _sys0 not in sys.path:
                sys.path.insert(1, _sys0)
            import t1cf as _T1CF
            sc_, ids_, nl_, cnt_, _tr = _T1CF.fused(indices, t_["scores"], s=TIERS_S, lam=TIERS_LAM)
            t_["scores"] = sc_
            t_["nolsb"] = (nl_, None)
            FG["t1cn"].append(cnt_)
            FG["t1c_ids"] = ids_
            return None
'''),
('''        try:
            return _sw_call(self, x, indices, *a, **kw)
        finally:
            self._bm["tier"].pop("nolsb", None)
''', '''        if FG.get("t1c_ids") is not None:                # C25: the fused 1c kernel's km'd ids (its where lines are in it)
            indices = FG.pop("t1c_ids")
        try:
            return _sw_call(self, x, indices, *a, **kw)
        finally:
            self._bm["tier"].pop("nolsb", None)
'''),
]
for i, (a, b) in enumerate(R):
    n = s.count(a)
    assert n == 1, f"anchor {i}: {n} matches"
    s = s.replace(a, b)
compile(s, P, "exec")
open(P, "w").write(s)
print("patched", len(R), "anchors")
