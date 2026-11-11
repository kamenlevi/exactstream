# N7-15 wiring for /Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts/ref2.py (apply ONLY after BigMoE's night ticket, then ref2.self_test). Exact replacements.
import os, shutil
p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts/ref2.py")
scratch = os.path.expanduser("~/Leap/data")
shutil.copy(p, os.path.join(scratch, "ref2_before_n715.py"))
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep("""    tierglu_nax6n.repack (3c cannot read them). With nax_hybrid too: NAX6-NH (== NAX6-H bitwise, +23.3 % at S 16).
""", """    tierglu_nax6n.repack (3c cannot read them). With nax_hybrid too: NAX6-NH (== NAX6-H bitwise, +23.3 % at S 16).
  - N7-15 (Tue 2026-11-03, BigMoE's load path; inert without registered flags): native_flags(base, dirty_m, side,
    dirty_l) registers a tier layer's per-slot dirty flags (int32; 1 = an in-service demand load left the slot row-major).
    With CFG nax_native every NAX6-N call first rewrites its dirty slots in place (tierglu_nax6n.repack_slots): in
    _nax6_glu, in _ooo_nax6 before phase 2, and before phase 1 too with CFG nax_repack_p1 (off: BigMoE gathers every
    other load native on the CPU, so phase 1 never meets a dirty slot).
""")
rep('''       "nax_hybrid": False, "nax_native": False}''', '''       "nax_hybrid": False, "nax_native": False, "nax_repack_p1": False}''')
rep("hc_b=0, nax_hybrid=0, nax_native=0)", "hc_b=0, nax_hybrid=0, nax_native=0, nax_repack=0)")
rep('''def _nax6_glu(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", scores=None, **kw):''',
    '''_NATIVE_FLAGS = {}
def native_flags(base, dirty_m, side=None, dirty_l=None):
    """N7-15 (BigMoE's ext, BIGMOE_NAX_NATIVE=1): register one tier layer's per-slot dirty flags, int32 MLX arrays the ext
    owns (it sets 1 after an in-service demand load wrote a slot row-major, before the slot's ready status; only the GPU
    repack clears them). Call once per tier layer after install_sidecars with t["base"] / t["sideP"]."""
    _NATIVE_FLAGS[id(base)] = (base, dirty_m, side, dirty_l)

def _native_repack(msb, lsb, msb_slot, lsb_slot):
    """N7-15: the dirty slots among a call's pairs -> native, in place, before a NAX6-N call reads them; returns the slot
    arrays the call must take (pools without registered flags, or nax_native off: unchanged)"""
    f = _NATIVE_FLAGS.get(id(msb)) if CFG["nax_native"] else None
    if f is None:
        return msb_slot, lsb_slot
    import tierglu_nax6n as NN
    COUNTS["nax_repack"] += 1
    side = f[3] is not None and lsb_slot is not None
    return NN.repack_slots(msb, lsb if side else None, msb_slot, lsb_slot if side else None, f[1], f[3] if side else None)

def _nax6_glu(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", scores=None, **kw):''')
rep('''    M, R = x.shape[0], len(plan)
    y = _nax_glu_fn()(x, msb, lsb, msb_slot, lsb_slot, plan=plan, b2=kw.get("b2"), grp=N6.gpu_grp(msb_slot, lsb_slot, plan))
    COUNTS["nax6"] += 1''', '''    M, R = x.shape[0], len(plan)
    msb_slot, lsb_slot = _native_repack(msb, lsb, msb_slot, lsb_slot)
    y = _nax_glu_fn()(x, msb, lsb, msb_slot, lsb_slot, plan=plan, b2=kw.get("b2"), grp=N6.gpu_grp(msb_slot, lsb_slot, plan))
    COUNTS["nax6"] += 1''')
rep('''    xs = rowsum64(x2)
    sl1 = mx.maximum(pre_l, 0) if R4 else None''', '''    xs = rowsum64(x2)
    if CFG["nax_repack_p1"]:                       # N7-15 option (c): phase 1's slots too (negative = not resident: skipped)
        pre_m, pre_l = _native_repack(t["base"], t["sideP"], pre_m, pre_l if R4 else None)
    sl1 = mx.maximum(pre_l, 0) if R4 else None''')
rep('''    sl = gate.spin(mx.depends(dep_l, after), t["side"], n_l).reshape(M, R4) if R4 else None
    y2 = glu(''', '''    sl = gate.spin(mx.depends(dep_l, after), t["side"], n_l).reshape(M, R4) if R4 else None
    sm, sl = _native_repack(t["base"], t["sideP"], sm, sl)   # N7-15: the spin's demand loads -> native before phase 2
    y2 = glu(''')
open(p + ".n715tmp", "w").write(s)
os.replace(p + ".n715tmp", p)          # atomic: live importers never see a partial file
print("ref2 patched (backup: ~/Leap/data/ref2_before_n715.py)")
