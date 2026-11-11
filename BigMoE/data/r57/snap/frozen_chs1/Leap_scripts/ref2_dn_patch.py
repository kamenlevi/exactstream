# DENSE-N wiring for /Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts/ref2.py (Leap, Wed 2026-11-04; TABLE_350 'DENSE-N'). Apply ONLY when no other
# session's run imports ref2 (BigMoE's quiet arms start fresh processes), then ref2.self_test + dn_wire_check.py.
# Exact replacements; backup -> data/ref2_before_dn.py; atomic write (temp + os.replace). Default OFF and inert.
# Usage: python ref2_dn_patch.py [--dry]   (--dry: apply to a copy in the scratch dir and only parse it)
import os, sys, shutil, ast
p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts/ref2.py")
DRY = "--dry" in sys.argv
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)

rep("""    (8-row checks, AR steps) runs fs4's R 8 form on its first 8 rows: bitwise == F1's rows (B6 INV (e)); COUNTS fatt_r8.
""", """    (8-row checks, AR steps) runs fs4's R 8 form on its first 8 rows: bitwise == F1's rows (B6 INV (e)); COUNTS fatt_r8.
  - DENSE-N (Wed 2026-11-04, exact-safe; CFG["dense_native"] or REF2_DENSE_NATIVE=1, off by default): the 4-bit dense
    calls that take qmm16c run L1-26's fragment-native layout instead (qmm26.qmm16n at ks 1: bitwise == qmm16c). A
    module's resident weight is repacked on the GPU at its first switched call (_dn_repack, MLX ops, no sync) and kept
    in _DN (+~1.5 GB for the 125B's dense set; MLX's layout stays for prefill); only weights marked by the patched
    nn.QuantizedLinear call (or R-83's concat) are repacked, any other array (a slice) keeps qmm16c. CFG
    dense_native_nt (REF2_DENSE_NATIVE_NT, default 1) only groups tiles per threadgroup (bits unchanged). COUNTS
    dense_native / dense_native_repack; remove() drops the copies. qmm26 is imported on the first switched call.
""")
rep('''       "fatt_verify": os.environ.get("REF2_FATT_VERIFY") == "1", "fatt_r8": os.environ.get("REF2_FATT_R8") == "1"}''',
    '''       "fatt_verify": os.environ.get("REF2_FATT_VERIFY") == "1", "fatt_r8": os.environ.get("REF2_FATT_R8") == "1",
       "dense_native": os.environ.get("REF2_DENSE_NATIVE") == "1",
       "dense_native_nt": int(os.environ.get("REF2_DENSE_NATIVE_NT", "1"))}''')
rep("""              fatt_verify=0, verify_old=0, verify_old_eligible=0, fatt_r8=0)""",
    """              fatt_verify=0, verify_old=0, verify_old_eligible=0, fatt_r8=0, dense_native=0, dense_native_repack=0)""")
rep("""def _qmm(x, w, *args, **kw):
    orig = _ORIG["qmm"]""", """_DN = {}                                             # DENSE-N: id(weight) -> (weight, wn, sn, bn), native copies
_DN_SRC = [None]                                     # DENSE-N: the resident weight of the call being built (_qlin)
_DN_MAPS = []

def _qmm16n():
    \"\"\"DENSE-N: qmm26.qmm16n, loaded on the first switched call only\"\"\"
    from qmm26 import qmm16n
    return qmm16n

def _dn_repack(w, scales, biases):
    \"\"\"DENSE-N: qmm26.repack's layout built with MLX ops on the GPU (lazy, no host sync): wn [N/32, K/64, 32, 8]
    uint32 (lane l's word 2 st + p = rows 16 p + fm(l) (low half) and 16 p + fm(l) + 8 (high half), codes of step st),
    sn / bn [N/32, K/64, 32]\"\"\"
    if not _DN_MAPS:
        lane = np.arange(32)
        qid = lane >> 2
        fm, fq = (qid & 4) | ((lane >> 1) & 3), (qid & 2) | (lane & 1)
        i0, i1, sh = (np.zeros((32, 8), np.int64) for _ in range(3))
        for l in range(32):
            for st in range(4):
                for pp in range(2):
                    ws = 2 * int(fq[l]) + (st >> 1)
                    i0[l, 2 * st + pp] = (16 * pp + int(fm[l])) * 8 + ws
                    i1[l, 2 * st + pp] = (16 * pp + int(fm[l]) + 8) * 8 + ws
                    sh[l, 2 * st + pp] = 16 * (st & 1)
        _DN_MAPS.extend([mx.array(i0.reshape(-1).astype(np.int32)), mx.array(i1.reshape(-1).astype(np.int32)),
                         mx.array(sh.reshape(-1).astype(np.uint32))])
    i0, i1, sh = _DN_MAPS
    N, KW = w.shape
    T, KG = N // 32, KW // 8
    w3 = w.reshape(T, 32, KG, 8).transpose(0, 2, 1, 3).reshape(T, KG, 256)          # [tile, group, row * 8 + word]
    lo = mx.bitwise_and(mx.right_shift(mx.take(w3, i0, axis=2), sh), 0xFFFF)
    hi = mx.bitwise_and(mx.right_shift(mx.take(w3, i1, axis=2), sh), 0xFFFF)
    wn = mx.contiguous(mx.bitwise_or(lo, mx.left_shift(hi, 16)).reshape(T, KG, 32, 8))
    sn = mx.contiguous(scales.reshape(T, 32, KG).transpose(0, 2, 1))
    bn = mx.contiguous(biases.reshape(T, 32, KG).transpose(0, 2, 1))
    return wn, sn, bn

def _dn_get(w, scales, biases):
    \"\"\"DENSE-N: the native copy of a resident weight (repacked at its first marked call), or None (keep qmm16c)\"\"\"
    e = _DN.get(id(w))
    if e is not None and e[0] is w:
        return e
    if w is not _DN_SRC[0] or w.shape[0] % 32 != 0:
        return None
    e = _DN[id(w)] = (w,) + _dn_repack(w, scales, biases)
    COUNTS["dense_native_repack"] += 1
    return e

def _qlin(self, x):
    \"\"\"DENSE-N: nn.QuantizedLinear's call with its weight marked as resident for _qmm (inert unless dense_native)\"\"\"
    if not (_ON[0] and CFG["dense_native"]):
        return _ORIG["qlin"](self, x)
    prev = _DN_SRC[0]
    _DN_SRC[0] = self.get("weight")
    try:
        return _ORIG["qlin"](self, x)
    finally:
        _DN_SRC[0] = prev

def _qmm(x, w, *args, **kw):
    orig = _ORIG["qmm"]""")
rep("""    if bits == 4 and group_size == 64 and N >= CFG["dense_min_n"] and N % 64 == 0 and K_ % 64 == 0:
        y = qmm16c(x16, w, scales, biases)
        COUNTS["qmm16c"] += 1""", """    if bits == 4 and group_size == 64 and N >= CFG["dense_min_n"] and N % 64 == 0 and K_ % 64 == 0:
        e = _dn_get(w, scales, biases) if CFG["dense_native"] else None
        if e is not None:                                # DENSE-N: bitwise == qmm16c (ks 1); nt only groups tiles
            nt = CFG["dense_native_nt"] if N % (32 * CFG["dense_native_nt"]) == 0 else 1
            y = _qmm16n()(x16, e[1], e[2], e[3], N, 1, nt)
            COUNTS["dense_native"] += 1
        else:
            y = qmm16c(x16, w, scales, biases)
            COUNTS["qmm16c"] += 1""")
rep("""            COUNTS["gdn_inproj_x"] += 1
            yqz, yba = q(self.qz), q(self.ba)""", """            COUNTS["gdn_inproj_x"] += 1
            prev, _DN_SRC[0] = _DN_SRC[0], self.qz[0]   # DENSE-N: the concat is resident (inert unless dense_native)
            try:
                yqz = q(self.qz)
            finally:
                _DN_SRC[0] = prev
            yba = q(self.ba)""")
rep("""        mx.quantized_matmul = _qmm
        nn.Linear.__call__ = _lin
""", """        mx.quantized_matmul = _qmm
        nn.Linear.__call__ = _lin
        _ORIG["qlin"] = nn.QuantizedLinear.__call__      # DENSE-N: marks resident weights (inert unless dense_native)
        nn.QuantizedLinear.__call__ = _qlin
""")
rep("""def remove():
    _ON[0] = False
""", """def remove():
    _ON[0] = False
    _DN.clear()                                      # DENSE-N: drop the native copies (and their weight references)
""")
ast.parse(s)
if DRY:
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "ref2_dn_dry.py")
    open(out, "w").write(s)
    print("dry: applies and parses ->", out)
else:
    shutil.copy(p, os.path.expanduser("~/Leap/data/ref2_before_dn.py"))
    tmp = p + ".tmp_dn"
    open(tmp, "w").write(s)
    os.replace(tmp, p)
    print("applied (backup data/ref2_before_dn.py); now run ref2.self_test and dn_wire_check.py")
