# HC8-1P FAST wiring for /Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts/ref2.py (Leap, Wed 2026-11-04; TABLE_350 'HC8-1P'). Default OFF and inert:
# - CFG["hc8_fast"] (REF2_HC8_FAST=1) + CFG["hc8_np"] (REF2_HC8_NP, default 20): a <= 16-row HC read of a GatedResidual
#   whose mixers are 8-bit (hc8r.hc8_install, the FAST arm's model change) runs hc8r.read (v2), padded to 16 rows.
# - Without the switch, 8-bit mixers take MTPLX's eager chain at 16 padded rows (never the bf16 S4 path, which would
#   misread quantized weights). bf16 mixers: unchanged (the exact S4 / kernel-B path).
# Exact replacements; backup -> data/ref2_before_hc8.py; atomic write. Then ref2.self_test + hc8_wire_check.py.
# Usage: python ref2_hc8_patch.py [--dry]
import os, sys, shutil, ast
p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts/ref2.py")
DRY = "--dry" in sys.argv
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)

rep("""    dense_native / dense_native_repack; remove() drops the copies. qmm26 is imported on the first switched call.
""", """    dense_native / dense_native_repack; remove() drops the copies. qmm26 is imported on the first switched call.
  - HC8-1P (Wed 2026-11-04, FAST, NOT exact; CFG["hc8_fast"] or REF2_HC8_FAST=1, off by default; CFG hc8_np /
    REF2_HC8_NP, default 20): a <= 16-row HC read of a GatedResidual whose mixers are 8-bit g64 nn.QuantizedLinear
    (scripts/hc8r.py hc8_install, the FAST arm's model change, the HC8 recipe) runs hc8r.read v2 at 16 padded rows: one
    pass over the streams, the norm after the GEMM, fp32 gates (6.69e-3 rel rms vs the exact read = HC8-eager's
    6.70e-3; 5.22 vs 9.67 ms per 96-read chain, quiet). Row-invariant (16 padded rows), so SD == AR. Without the
    switch, 8-bit mixers take MTPLX's eager chain at 16 padded rows. COUNTS hc8_fast. hc8r is imported on first use.
""")
rep('''       "dense_native": os.environ.get("REF2_DENSE_NATIVE") == "1",
       "dense_native_nt": int(os.environ.get("REF2_DENSE_NATIVE_NT", "1"))}''',
    '''       "dense_native": os.environ.get("REF2_DENSE_NATIVE") == "1",
       "dense_native_nt": int(os.environ.get("REF2_DENSE_NATIVE_NT", "1")),
       "hc8_fast": os.environ.get("REF2_HC8_FAST") == "1", "hc8_np": int(os.environ.get("REF2_HC8_NP", "20"))}''')
rep("""fatt_r8=0, dense_native=0, dense_native_repack=0)""", """fatt_r8=0, dense_native=0, dense_native_repack=0, hc8_fast=0)""")
rep('''def _gr(self, hyper_input, pending=None):
    orig = _ORIG["gr"]''', '''_HC8 = {}
def _hc8r():
    """HC8-1P: scripts/hc8r.py, loaded on first use"""
    import hc8r
    return hc8r

def _hc8_params(g):
    """HC8-1P: the 8-bit mixers' arrays of a GatedResidual as hc8r.read's P (cached per module), or None when its mixers
    are not 8-bit g64 bf16-scaled nn.QuantizedLinear"""
    d, u = g.input_mix_weight_down, g.input_mix_weight_up
    if not (isinstance(d, nn.QuantizedLinear) and isinstance(u, nn.QuantizedLinear)):
        return None
    e = _HC8.get(id(g))
    if e is not None and e[0] is d["weight"] and e[1] is u["weight"]:
        return e[2]
    if not (d.bits == 8 and u.bits == 8 and d.group_size == 64 and u.group_size == 64 and "biases" in d and "biases" in u
            and d["scales"].dtype == mx.bfloat16 and u["scales"].dtype == mx.bfloat16 and "block_inject_weight" in g):
        return None
    P = dict(wq=d["weight"], ws=d["scales"], wb=d["biases"], uq=u["weight"], us=u["scales"], ub=u["biases"],
             wi=g.block_inject_weight.weight.astype(mx.bfloat16), wn=g.hc_norm.weight.astype(mx.bfloat16),
             hc=int(g.hc_count), d=int(g.hidden_size), lr=int(d["weight"].shape[0]), eps=float(g.hc_norm.eps))
    _HC8[id(g)] = (d["weight"], u["weight"], P)
    return P

def _gr(self, hyper_input, pending=None):
    orig = _ORIG["gr"]''')
rep('''    if "block_inject_weight" in self:
        if id(self) not in _WCAT:''', '''    q8 = _hc8_params(self) if "block_inject_weight" in self else None
    if q8 is not None:                               # HC8-1P: 8-bit mixers (the FAST arm) never take the bf16 S4 path
        bf = mx.bfloat16
        if CFG["hc8_fast"] and h16.dtype == bf and (p16 is None or all(t.dtype == bf for t in p16)):
            bo, inj = (None, None) if p16 is None else (p16[0].reshape(R, -1), p16[1].reshape(R, -1))
            mixed, written, inject = _hc8r().read(q8, h16.reshape(R, -1), bo, inj, NP=CFG["hc8_np"], v=2)
            COUNTS["hc8_fast"] += 1
            return (mixed[:rows].reshape(*lead, -1), written[:rows].reshape(hyper_input.shape),
                    inject[:rows].reshape(*lead, -1))
        m_, w_, i_ = orig(self, h16, p16)
        COUNTS["hc_eager16"] += 1
        return (m_[0, :rows].reshape(*lead, -1), w_[0, :rows].reshape(hyper_input.shape),
                i_[0, :rows].reshape(*lead, -1))
    if "block_inject_weight" in self:
        if id(self) not in _WCAT:''')
rep("""    _DN.clear()                                      # DENSE-N: drop the native copies (and their weight references)
""", """    _DN.clear()                                      # DENSE-N: drop the native copies (and their weight references)
    _HC8.clear()                                     # HC8-1P: drop the cached parameter views
""")
ast.parse(s)
if DRY:
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "ref2_hc8_dry.py")
    open(out, "w").write(s)
    print("dry: applies and parses ->", out)
else:
    shutil.copy(p, os.path.expanduser("~/Leap/data/ref2_before_hc8.py"))
    tmp = p + ".tmp_hc8"
    open(tmp, "w").write(s)
    os.replace(tmp, p)
    print("applied (backup data/ref2_before_hc8.py); now run ref2.self_test and hc8_wire_check.py")
