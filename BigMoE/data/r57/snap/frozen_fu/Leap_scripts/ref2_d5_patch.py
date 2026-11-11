# D5 wiring (Leap, Wed 2026-11-04 16:4x; TABLE_350 'D5'; FableGod HC_READ.md D5). EXACT, default OFF and inert:
# CFG["hc_pend_mlp"] / REF2_HC_PEND=1 hands a layer's MLP-side residual write to the next layer's attention-side HC read
# as its pending write, as DecoderLayer already hands the attention write to the MLP read. ref2's write (_hc_write, the
# patched Q._hyper_residual_write) stashes (output, hyper, block_out, inject) in one carry slot; ref2's HC read (_gr)
# whose input IS that output (identity) reads (hyper, pending) instead, folding the write into its norm pass (leapk's
# hc_read / HC8: the same bf16 ops). The write's own kernel then runs only if something else consumes its output (taps,
# a mid-loop async_eval). One slot + the identity check: no stale match. Patches ref2.py (backup
# data/ref2_before_d5.py), atomic write.
# Usage: python ref2_d5_patch.py [--dry]
import os, sys, shutil, ast
DRY = "--dry" in sys.argv
p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts/ref2.py")
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
rep("""  - SMALL-N (Wed 2026-11-04, FAST, off by default; CFG["small_native"] or REF2_SMALL_NATIVE=1): the small 4-bit g64 calls
""", """  - D5 (Wed 2026-11-04, EXACT, off by default; CFG["hc_pend_mlp"] or REF2_HC_PEND=1): a layer's MLP-side residual write
    goes to the next layer's attention-side HC read as pending (as the attention write goes to the MLP read). The
    write (_hc_write) stashes (output, hyper, block_out, inject) in one carry slot; an HC read whose input IS that
    output reads (hyper, pending) and folds the write into its norm pass, bitwise. The write's kernel then runs only
    for other consumers (taps, mid-loop evals). COUNTS hc_pend.
  - SMALL-N (Wed 2026-11-04, FAST, off by default; CFG["small_native"] or REF2_SMALL_NATIVE=1): the small 4-bit g64 calls
""")
rep('''"small_native": os.environ.get("REF2_SMALL_NATIVE") == "1"}''',
    '''"small_native": os.environ.get("REF2_SMALL_NATIVE") == "1",
       "hc_pend_mlp": os.environ.get("REF2_HC_PEND") == "1"}''')
rep("""ooo_1c=0, small_native=0)""", """ooo_1c=0, small_native=0, hc_pend=0)""")
rep('''def _hc_write(hyper, block_out, inject):
    """R-73 (exact-safe, CFG hc_write): hyper + block_out x inject as ONE compiled kernel on <= 16-row bf16 calls
    (bitwise == the stock Multiply + Add: R-73, layer 3's real weights, 6 scales)"""
    lead, rows = _rows(hyper)
    if _ON[0] and CFG["hc_write"] and rows <= R and hyper.dtype == mx.bfloat16:
        COUNTS["hc_write"] += 1
        return _hc_write_c(hyper, block_out, inject)
    return _ORIG["hc_write"](hyper, block_out, inject)''', '''_CARRY = [None]                                      # D5: (output, hyper, block_out, inject) of the last write

def _hc_write(hyper, block_out, inject):
    """R-73 (exact-safe, CFG hc_write): hyper + block_out x inject as ONE compiled kernel on <= 16-row bf16 calls
    (bitwise == the stock Multiply + Add: R-73, layer 3's real weights, 6 scales). D5 (CFG hc_pend_mlp): every write's
    operands are stashed for the next HC read of its output"""
    lead, rows = _rows(hyper)
    if _ON[0] and CFG["hc_write"] and rows <= R and hyper.dtype == mx.bfloat16:
        COUNTS["hc_write"] += 1
        y = _hc_write_c(hyper, block_out, inject)
    else:
        y = _ORIG["hc_write"](hyper, block_out, inject)
    if _ON[0] and CFG["hc_pend_mlp"]:
        _CARRY[0] = (y, hyper, block_out, inject)
    return y''')
rep('''def _gr(self, hyper_input, pending=None):
    orig = _ORIG["gr"]
    if not _ON[0]:
        return orig(self, hyper_input, pending)
''', '''def _gr(self, hyper_input, pending=None):
    orig = _ORIG["gr"]
    if not _ON[0]:
        return orig(self, hyper_input, pending)
    c = _CARRY[0]
    if pending is None and c is not None and c[0] is hyper_input and CFG["hc_pend_mlp"] and "block_inject_weight" in self:
        _CARRY[0] = None                                 # D5: read the last write's operands, the write folded in
        hyper_input, pending = c[1], (c[2], c[3])
        COUNTS["hc_pend"] += 1
''')
rep('''    _SN.clear()                                      # SMALL-N: drop the padded copies
''', '''    _SN.clear()                                      # SMALL-N: drop the padded copies
    _CARRY[0] = None                                 # D5: no carried write across apply / remove
''')
ast.parse(s)
if DRY:
    d = os.environ.get("TMPDIR", "/tmp")
    open(os.path.join(d, "ref2_d5_dry.py"), "w").write(s)
    print("dry: applies and parses ->", d)
else:
    shutil.copy(p, os.path.expanduser("~/Leap/data/ref2_before_d5.py"))
    tmp = p + ".tmp_d5"
    open(tmp, "w").write(s)
    os.replace(tmp, p)
    print("applied (backup data/ref2_before_d5.py); now self_test + d5_check.py")
