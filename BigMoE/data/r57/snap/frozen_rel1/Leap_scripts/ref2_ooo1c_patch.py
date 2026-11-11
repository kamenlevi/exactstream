# OOO-1C wiring for /Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts/ref2.py (Leap, Wed 2026-11-04; TABLE_350 'OOO-1C'). Default OFF and inert:
# CFG["ooo_1c"] / REF2_OOO_1C=1: OOO's two expert calls per MoE layer become ONE (scripts/ooo1c.py) on
#  - BigMoE's 3c path (today's exact default): ref2._ooo_sel -> _ooo_3c_1c (== tier_glu with scores == _ooo_call's two
#    phases + combine_phase, bitwise);
#  - the NAX path (C1, CFG tier_nax_ooo): _ooo_nax6 -> _ooo_nax6_1c (== _ooo_nax6's two phases, bitwise);
# when the gate runs in spin mode (release regions exist). Native records (3c-N / NAX6-N) and nax_hybrid keep two phases.
# Exact replacements; backup -> data/ref2_before_ooo1c.py; atomic write. Usage: python ref2_ooo1c_patch.py [--dry]
import os, sys, shutil, ast
p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts/ref2.py")
DRY = "--dry" in sys.argv
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)

rep("""    4.97 vs 5.27 ms per 96-read chain (quiet); v2 is what EvalSuite's snap/leap-c805fd0 runs.
""", """    4.97 vs 5.27 ms per 96-read chain (quiet); v2 is what EvalSuite's snap/leap-c805fd0 runs.
  - OOO-1C (Wed 2026-11-04, exact, off by default; CFG["ooo_1c"] or REF2_OOO_1C=1): BigMoE's OOO in ONE expert call per
    MoE layer (scripts/ooo1c.py): the late pairs go last in the grid and resolve their slots inside kernel A from the
    loader's release region (coherent(system) polls of BigMoE's coherent-spin layout; no ext change). On BigMoE's 3c path
    (== tier_glu with scores == its two phases + combine_phase, bitwise) and on C1's NAX path (== _ooo_nax6, bitwise);
    needs the gate in spin mode; native records and nax_hybrid keep the two phases. Harness, ~4 % late pairs: -3.2 (3c)
    / -4.1 (NAX) ms per check. COUNTS ooo_1c.
""")
rep('''       "hc8_v": int(os.environ.get("REF2_HC8_V", "4"))}''',
    '''       "hc8_v": int(os.environ.get("REF2_HC8_V", "4")),
       "ooo_1c": os.environ.get("REF2_OOO_1C") == "1"}''')
rep("""fatt_r8=0, dense_native=0, dense_native_repack=0, hc8_fast=0, dense_native_unpack=0)""",
    """fatt_r8=0, dense_native=0, dense_native_repack=0, hc8_fast=0, dense_native_unpack=0, ooo_1c=0)""")
rep('''def _ooo_nax6(x, ids, bm, t, post, sc, shared):''', '''def _ooo1c():
    """OOO-1C: scripts/ooo1c.py, loaded on first use"""
    import ooo1c
    return ooo1c

def _ooo1c_offs(bm, t, R4):
    """OOO-1C: [base region offset, sidecar region offset, seq] of this pass for the gate's release regions (the ext's
    ((seq * nl + L) % reg_n) * pairs * rs), or None when the gate has no regions (event mode)"""
    import bigmoe.gated as BG
    gate = bm["gate"]
    if getattr(BG, "MODE", "spin") != "spin" or getattr(gate, "region", None) is None:
        return None
    rs = 32 * (2 + BG.XLINES)
    off = lambda Lx: ((gate.seq * gate.nl + Lx) % gate.nreg) * BG.PAIRS * rs
    return mx.array([off(bm["L"]), off(t["side"]) if R4 else 0, gate.seq], mx.int32)

def _ooo_1c_call(x, ids, bm, t, post, sc, shared, nax):
    """OOO-1C: the post as today, then ONE expert call (3c or NAX6) whose late pairs resolve their slots in kernel A;
    None when not applicable (the caller then runs its two phases)"""
    import bigmoe.tier as BT
    PLAN = BT.PLAN
    K = ids.shape[-1]
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    lo4 = [s_ for s_ in t["segs"] if s_[2] == 4]
    R4 = lo4[-1][1] if lo4 else 0
    assert not lo4 or lo4[0][0] == 0, t["segs"]
    offs = _ooo1c_offs(bm, t, R4)
    if offs is None:
        return None
    gate, L = bm["gate"], bm["L"]
    BT.OOO_STATS["calls"] += 1
    COUNTS["ooo_1c"] += 1
    dep_m, n_m = post(L, ids)
    pre_m = dep_m[1:1 + n_m].reshape(M, K)
    if R4:
        dep_l, n_l = post(t["side"], ids[..., :R4], after=dep_m)
        pre_l = dep_l[1:1 + n_l].reshape(M, R4)
    else:
        pre_l = mx.zeros((M, 1), mx.int32)
    xin = mx.depends(x2, [shared]) if shared is not None else x2   # the shared expert first, as today's spins wait after it
    OC = _ooo1c()
    if nax:
        TG = BT._tier_glu_mod()
        y, _, _ = OC.tier_glu_nax6_1c(xin, t["base"], t["sideP"], pre_m, pre_l, ids.reshape(M, K), PLAN, gate.region, offs,
                                      gate.status)
        COUNTS["nax6_ooo"] += 1
        return TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32),
                                sc.reshape(-1, K)).reshape(*x.shape[:-1], -1)
    y, _, _ = OC.tier_glu_3c_1c(xin, t["base"], t["sideP"], pre_m, pre_l, PLAN, gate.region, offs, gate.status,
                                sc.reshape(-1, K))
    return y.reshape(*x.shape[:-1], -1)

def _ooo_nax6(x, ids, bm, t, post, sc, shared):''')
rep('''    import bigmoe.tier as BT
    N6 = _nax6mod()
    TG = BT._tier_glu_mod()
    from tierglu_nax import rowsum64
    BT.OOO_STATS["calls"] += 1
    COUNTS["nax6_ooo"] += 1''', '''    if CFG["ooo_1c"] and not (CFG["nax_native"] or CFG["nax_hybrid"] or CFG["nax_repack_p1"]):
        y1c = _ooo_1c_call(x, ids, bm, t, post, sc, shared, nax=True)   # OOO-1C: one NAX6 call
        if y1c is not None:
            return y1c
    import bigmoe.tier as BT
    N6 = _nax6mod()
    TG = BT._tier_glu_mod()
    from tierglu_nax import rowsum64
    BT.OOO_STATS["calls"] += 1
    COUNTS["nax6_ooo"] += 1''')
rep('''    elif _layout(t) in (1, 2):
        return _ooo_3cn(x, ids, bm, t, post, sc, shared)
    return _ORIG["ooo_call"](x, ids, bm, t, post, sc, shared)''', '''        if CFG["ooo_1c"]:                                # OOO-1C: BigMoE's 3c in one call
            y1c = _ooo_1c_call(x, ids, bm, t, post, sc, shared, nax=False)
            if y1c is not None:
                return y1c
    elif _layout(t) in (1, 2):
        return _ooo_3cn(x, ids, bm, t, post, sc, shared)
    return _ORIG["ooo_call"](x, ids, bm, t, post, sc, shared)''')
ast.parse(s)
if DRY:
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "ref2_ooo1c_dry.py")
    open(out, "w").write(s)
    print("dry: applies and parses ->", out)
else:
    shutil.copy(p, os.path.expanduser("~/Leap/data/ref2_before_ooo1c.py"))
    tmp = p + ".tmp_ooo1c"
    open(tmp, "w").write(s)
    os.replace(tmp, p)
    print("applied (backup data/ref2_before_ooo1c.py); now run ref2.self_test and ooo1c_wire_check.py")
