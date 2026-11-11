# SMALL-N wiring (Leap, Wed 2026-11-04; TABLE_350 'SMALL-N'). FAST, default OFF and inert: CFG["small_native"] /
# REF2_SMALL_NATIVE=1 runs the small 4-bit g64 calls that take MLX's 16-row call today (N < dense_min_n or N % 64 != 0:
# R-74's shared gate | up | gate concat, the k / v projections, R-83's b | a) on qmm16n, rows zero-padded to a multiple
# of 32, for weights marked as resident (nn.QuantizedLinear calls, R-83's concats, R-74's concat via glue_fused).
# Bits differ from MLX's call (FAST); row-invariant (16 padded rows). Patches ref2.py (backup data/ref2_before_sn.py)
# and glue_fused.py (marks R-74's concat; backup data/glue_fused_before_sn.py), atomic writes.
# Usage: python ref2_sn_patch.py [--dry]
import os, sys, shutil, ast
DRY = "--dry" in sys.argv
p = os.path.expanduser("~/Leap/scripts/ref2.py")
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
rep("""    (Friday's converted store) take the native forms (3c-N / NAX6-N, bf16 native or p11 pools); layers with N7-15's
    registered dirty flags keep the two phases.
""", """    (Friday's converted store) take the native forms (3c-N / NAX6-N, bf16 native or p11 pools); layers with N7-15's
    registered dirty flags keep the two phases.
  - SMALL-N (Wed 2026-11-04, FAST, off by default; CFG["small_native"] or REF2_SMALL_NATIVE=1): the small 4-bit g64 calls
    that take MLX's 16-row call (N < dense_min_n or N % 64 != 0: R-74's shared gate | up | gate, k / v, R-83's b | a)
    run on qmm16n with rows zero-padded to a multiple of 32, for marked resident weights (QuantizedLinear calls, R-83's
    concats, R-74's concat). Bits differ from MLX's call; 2.94 -> 1.80 ms per check on the default's shapes. COUNTS
    small_native. remove() drops the padded copies.
""")
rep('''       "ooo_1c": os.environ.get("REF2_OOO_1C") == "1"}''',
    '''       "ooo_1c": os.environ.get("REF2_OOO_1C") == "1", "small_native": os.environ.get("REF2_SMALL_NATIVE") == "1"}''')
rep("""hc8_fast=0, dense_native_unpack=0, ooo_1c=0)""", """hc8_fast=0, dense_native_unpack=0, ooo_1c=0, small_native=0)""")
rep('''def _qlin(self, x):
    """DENSE-N: nn.QuantizedLinear's call with its weight marked as resident for _qmm (inert unless dense_native)"""
    if not (_ON[0] and CFG["dense_native"]):''', '''_SN = {}                                             # SMALL-N: id(weight) -> (weight, wn, sn, bn, padded N)

def _sn_get(w, scales, biases):
    """SMALL-N: the padded native copy of a marked resident small weight (built at its first marked call), or None"""
    e = _SN.get(id(w))
    if e is not None and e[0] is w:
        return e
    if w is not _DN_SRC[0] or w.ndim != 2:
        return None
    N = int(w.shape[0])
    Np = (N + 31) // 32 * 32
    pad = lambda a: mx.concatenate([a, mx.zeros((Np - N,) + tuple(a.shape[1:]), a.dtype)]) if Np > N else a
    e = _SN[id(w)] = (w,) + _dn_repack(pad(w), pad(scales), pad(biases)) + (Np,)
    return e

def _qlin(self, x):
    """DENSE-N / SMALL-N: nn.QuantizedLinear's call with its weight marked as resident for _qmm (inert unless one of
    the switches is on)"""
    if not (_ON[0] and (CFG["dense_native"] or CFG["small_native"])):''')
rep('''        else:
            y = qmm16c(x16, w, scales, biases)
            COUNTS["qmm16c"] += 1
    else:
        y = orig(x16, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode)
        COUNTS["mlx16"] += 1''', '''        else:
            y = qmm16c(x16, w, scales, biases)
            COUNTS["qmm16c"] += 1
    else:
        e = (_sn_get(w, scales, biases) if (CFG["small_native"] and bits == 4 and group_size == 64 and K_ % 64 == 0
                                           and scales is not None and scales.dtype == mx.bfloat16) else None)
        if e is not None:                                # SMALL-N (FAST): qmm16n on the padded native copy
            y = _qmm16n()(x16, e[1], e[2], e[3], e[4], 1, 1)[:, :N]
            COUNTS["small_native"] += 1
        else:
            y = orig(x16, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode)
            COUNTS["mlx16"] += 1''')
rep('''            yba = q(self.ba)''', '''            prev, _DN_SRC[0] = _DN_SRC[0], self.ba[0]   # SMALL-N: b | a is resident too (inert unless small_native)
            try:
                yba = q(self.ba)
            finally:
                _DN_SRC[0] = prev''')
rep('''    _HC8.clear()                                     # HC8-1P: drop the cached parameter views
''', '''    _HC8.clear()                                     # HC8-1P: drop the cached parameter views
    _SN.clear()                                      # SMALL-N: drop the padded copies
''')
ast.parse(s)
g = os.path.expanduser("~/Leap/scripts/glue_fused.py")
gs = open(g).read()
a = '''    cat, n = k
    y = mx.quantized_matmul(x, *cat, transpose=True, group_size=64, bits=4)'''
assert gs.count(a) == 1
gs = gs.replace(a, '''    cat, n = k
    try:                                                 # SMALL-N: mark the resident concat for ref2 (inert unless
        import ref2 as _r2                               # CFG small_native: ref2 then runs it on qmm16n, FAST)
        prev = _r2._DN_SRC[0]
        _r2._DN_SRC[0] = cat[0]
    except Exception:
        _r2 = None
    try:
        y = mx.quantized_matmul(x, *cat, transpose=True, group_size=64, bits=4)
    finally:
        if _r2 is not None:
            _r2._DN_SRC[0] = prev''')
ast.parse(gs)
if DRY:
    d = os.environ.get("TMPDIR", "/tmp")
    open(os.path.join(d, "ref2_sn_dry.py"), "w").write(s)
    open(os.path.join(d, "glue_fused_sn_dry.py"), "w").write(gs)
    print("dry: both apply and parse ->", d)
else:
    shutil.copy(p, os.path.expanduser("~/Leap/data/ref2_before_sn.py"))
    shutil.copy(g, os.path.expanduser("~/Leap/data/glue_fused_before_sn.py"))
    for path, text in ((p, s), (g, gs)):
        tmp = path + ".tmp_sn"
        open(tmp, "w").write(text)
        os.replace(tmp, path)
    print("applied (backups data/ref2_before_sn.py, data/glue_fused_before_sn.py); now self_test + sn_wire_check.py")
