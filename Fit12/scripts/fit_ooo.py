"""Fit12 O1 (2026-10-24): out-of-order experts on B12LL's verify rows (MoonShots' OoO as BigMoE adopted it, OOO1; v3f
tier._ooo_call), patched into bt_b12._install inside this process (Breakthrough's file is read, never edited).
On a multi-row call whose used ranks are all waited (NF >= K, i.e. B12_LATE3=2,10: B12LL's verify rows), the pairs
resident at post time (the post kernel's partial-hit row; ranks 0-1 need both record parts) run before the loader spin,
with the shared expert; the missed pairs after it; then combine_phase (== tier_glu(..., scores=) bitwise for any hit mask,
tierglu.py). 1-row calls (late parts), PLAN4 and the pre-routed path are unchanged.
F12_OOO=1 turns it on; apply(B) must run before B.load_b12. CALLS[0] counts patched block calls. F12_OOO_COUNT=1
(diagnostic arms only): CNT keeps each call's lazy count of phase-1 pairs, read by hit_share() after the run."""
import inspect, os

ON = os.environ.get("F12_OOO") == "1"
CALLS = [0]
CNT = [] if os.environ.get("F12_OOO_COUNT") == "1" else None
_PL = 'plan="".join(T.PLAN)'
_REPL = [
    ("    TG = T._tier_glu()\n",
     "    TG = T._tier_glu()\n"
     "    TGM = T._tier_glu_mod()                          # [Fit12 O1] tier_glu_phase / combine_phase\n"),
    ("        sm = gate.spin(dep_m, L, n_m).reshape(KC, S).T                # [S, 20] base (MSB-3) slots, -1 = absent / late\n",
     "        oo = not one and not PLAN4 and NF >= K                        # [Fit12 O1] every used rank is waited\n"
     "        if oo:                                                        # pairs resident at post time, before the spin\n"
     "            _F12_OOO[0] += 1\n"
     "            pre_m = dep_m[1:1 + n_m].reshape(KC, S).T[:, :K]\n"
     "            pre_l = dep_l[1:1 + n_l].reshape(2, S).T\n"
     "            ok = (pre_m >= 0) & mx.concatenate([pre_l >= 0, mx.ones((S, K - 2), mx.bool_)], axis=-1)\n"
     "            ph = mx.where(ok, 1, 2).astype(mx.int32)\n"
     "            if _F12_OOO_CNT is not None:\n"
     "                _F12_OOO_CNT.append(((ph == 1).sum(), ph.size))\n"
     f"            y1 = TGM.tier_glu_phase(x2, t[\"base\"], t[\"sideP\"], pre_m, mx.maximum(pre_l, 0), ph, 1, {_PL})\n"
     "            dep_m, dep_l = mx.depends(dep_m, [y1, shared]), mx.depends(dep_l, [y1, shared])\n"
     "        sm = gate.spin(dep_m, L, n_m).reshape(KC, S).T                # [S, 20] base (MSB-3) slots, -1 = absent / late\n"),
    ("        y = TG(x2, t[\"base\"], t[\"sideP\"], s10, l2, plan=\"\".join(T.PLAN), scores=w_now)   # [S, D] combined in-kernel\n",
     "        if oo:                                                        # [Fit12 O1] the missed pairs, then the combine\n"
     f"            y2 = TGM.tier_glu_phase(x2, t[\"base\"], t[\"sideP\"], s10, l2, ph, 2, {_PL})\n"
     "            y = TGM.combine_phase(y1, y2, ph, w_now)\n"
     "        else:\n"
     "            y = TG(x2, t[\"base\"], t[\"sideP\"], s10, l2, plan=\"\".join(T.PLAN), scores=w_now)   # [S, D] combined in-kernel\n"),
]


def apply(B):
    if not ON:
        return False
    src = inspect.getsource(B._install)
    for old, new in _REPL:
        assert src.count(old) == 1, ("fit_ooo: bt_b12 changed", old[:70])
        src = src.replace(old, new)
    B.__dict__["_F12_OOO"] = CALLS
    B.__dict__["_F12_OOO_CNT"] = CNT
    exec(compile(src, B.__file__ + " [fit_ooo]", "exec"), B.__dict__)
    return True


def hit_share():
    """(phase-1 pairs, all pairs) over the counted calls; evaluates the lazy counts (diagnostic arms only)."""
    import mlx.core as mx
    if not CNT:
        return 0, 0
    mx.eval([a for a, _ in CNT])
    return sum(int(a.item()) for a, _ in CNT), sum(n for _, n in CNT)
