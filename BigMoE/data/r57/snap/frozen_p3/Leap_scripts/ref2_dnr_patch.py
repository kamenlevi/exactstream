# DENSE-N-RAM wiring for /Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts/ref2.py (Leap, Wed 2026-11-04; TABLE_350 'DENSE-N-RAM'). Inert until a harness
# calls ref2.dense_native_convert(model): each eligible 4-bit dense weight is then replaced IN PLACE by its native copy
# (4-D on purpose), so DENSE-N costs no RAM. Exact: <= 16-row calls run qmm16n (== qmm16c), every other call rebuilds
# MLX's layout for MLX's own kernel (stock bits). Exact replacements; backup -> data/ref2_before_dnr.py; atomic write.
# Usage: python ref2_dnr_patch.py [--dry]
import os, sys, shutil, ast
p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts/ref2.py")
DRY = "--dry" in sys.argv
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)

rep("""    switch, 8-bit mixers take MTPLX's eager chain at 16 padded rows. COUNTS hc8_fast. hc8r is imported on first use.
""", """    switch, 8-bit mixers take MTPLX's eager chain at 16 padded rows. COUNTS hc8_fast. hc8r is imported on first use.
  - DENSE-N-RAM (Wed 2026-11-04, exact, inert until called): dense_native_convert(model, head=False) replaces every
    eligible 4-bit dense weight (QuantizedLinear, bits 4, group 64, N >= dense_min_n; R-83's qkv | z concat too) IN PLACE
    by its native copy, 4-D on purpose so that a direct reader fails loudly; one copy, no RAM cost. _qmm: a converted
    weight takes qmm16n at <= 16 rows with ref2 on (bitwise == qmm16c); any other call (prefill, ref2 off) rebuilds MLX's
    layout per call (_dn_unpack, the exact inverse) for MLX's own kernel (stock bits). The 4-bit lm_head stays unless
    head=True. COUNTS dense_native / dense_native_unpack. dense_native_unconvert(model) restores MLX's layout.
""")
rep("""fatt_r8=0, dense_native=0, dense_native_repack=0, hc8_fast=0)""",
    """fatt_r8=0, dense_native=0, dense_native_repack=0, hc8_fast=0, dense_native_unpack=0)""")
rep('''def _dn_get(w, scales, biases):''', '''_DNR = {}                                            # DENSE-N-RAM: id(native weight) -> (native weight, N, K)
_DNR_ON = [False]
_DN_IMAPS = []

def _dn_unpack(wn, sn, bn, N, K):
    """DENSE-N-RAM: _dn_repack's exact inverse (MLX ops): native [N/32, K/64, 32, 8] + [N/32, K/64, 32] ->
    MLX's [N, K/8] uint32 + [N, K/64] scales / biases"""
    if not _DN_IMAPS:
        lane = np.arange(32)
        qid = lane >> 2
        fm, fq = (qid & 4) | ((lane >> 1) & 3), (qid & 2) | (lane & 1)
        lane_of = {(int(fm[l]), int(fq[l])): l for l in range(32)}
        ilo, ihi, sh = (np.zeros(256, np.int64) for _ in range(3))
        for row in range(32):
            pp, r16 = row >> 4, row & 15
            for ws in range(8):
                l = lane_of[(r16 & 7, ws >> 1)]
                stb = 2 * (ws & 1)
                ilo[row * 8 + ws] = l * 8 + 2 * stb + pp
                ihi[row * 8 + ws] = l * 8 + 2 * (stb + 1) + pp
                sh[row * 8 + ws] = 16 * int(r16 >= 8)
        _DN_IMAPS.extend([mx.array(ilo.astype(np.int32)), mx.array(ihi.astype(np.int32)), mx.array(sh.astype(np.uint32))])
    ilo, ihi, sh = _DN_IMAPS
    T, KG = N // 32, K // 64
    n3 = wn.reshape(T, KG, 256)                                        # [tile, group, lane * 8 + word]
    lo = mx.bitwise_and(mx.right_shift(mx.take(n3, ilo, axis=2), sh), 0xFFFF)
    hi = mx.bitwise_and(mx.right_shift(mx.take(n3, ihi, axis=2), sh), 0xFFFF)
    w = mx.bitwise_or(lo, mx.left_shift(hi, 16)).reshape(T, KG, 32, 8).transpose(0, 2, 1, 3).reshape(N, K // 8)
    return (w, sn.transpose(0, 2, 1).reshape(N, KG), bn.transpose(0, 2, 1).reshape(N, KG))

def _dnr_arrays(w, scales, biases):
    """DENSE-N-RAM: the native copy, evaluated (so the caller can drop the originals right away)"""
    wn, sn, bn = _dn_repack(w, scales, biases)
    mx.eval(wn, sn, bn)
    return wn, sn, bn

def _dn_mlx(m):
    """DENSE-N-RAM: a QuantizedLinear's (weight, scales, biases) in MLX's layout (unpacked if converted)"""
    w = m["weight"]
    e = _DNR.get(id(w))
    if e is not None and e[0] is w:
        return _dn_unpack(w, m["scales"], m["biases"], e[1], e[2])
    return w, m["scales"], m["biases"]

def dense_native_convert(model, head=False):
    """DENSE-N-RAM: replace every eligible 4-bit dense weight of model (and R-83's built qkv | z concats) by its native
    copy in place; returns (weights converted, MB). Call after load, outside mx.compile; ref2.apply() first."""
    n, mb = 0, 0.0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear) and "biases" in m and m.bits == 4 and m.group_size == 64 \\
                and getattr(m, "mode", "affine") == "affine":
            w = m["weight"]
            e = _DNR.get(id(w))
            if w.ndim != 2 or (e is not None and e[0] is w) or m["scales"].dtype != mx.bfloat16:
                continue
            N, K = int(w.shape[0]), int(w.shape[1]) * 8
            if N < CFG["dense_min_n"] or N % 64 or K % 64 or (N > 65536 and not head):
                continue
            wn, sn, bn = _dnr_arrays(w, m["scales"], m["biases"])
            m["weight"], m["scales"], m["biases"] = wn, sn, bn
            _DNR[id(wn)] = (wn, N, K)
            n += 1
            mb += (wn.nbytes + sn.nbytes + bn.nbytes) / 1e6
        f = getattr(m, "__dict__", {}).get("in_proj_fused")
        if isinstance(f, _InProjExact) and f._convert():
            n += 1
            mb += sum(t.nbytes for t in f.qz) / 1e6
    _DNR_ON[0] = True
    return n, mb

def dense_native_unconvert(model):
    """DENSE-N-RAM: MLX's layout back (every converted module and R-83 concat of model); returns the count"""
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            w = m["weight"]
            e = _DNR.pop(id(w), None)
            if e is not None and e[0] is w:
                wm, sm, bm = _dn_unpack(w, m["scales"], m["biases"], e[1], e[2])
                mx.eval(wm, sm, bm)
                m["weight"], m["scales"], m["biases"] = wm, sm, bm
                n += 1
        f = getattr(m, "__dict__", {}).get("in_proj_fused")
        if isinstance(f, _InProjExact):
            e = _DNR.pop(id(f.qz[0]), None)
            if e is not None and e[0] is f.qz[0]:
                f.qz = _dn_unpack(*f.qz, e[1], e[2])
                mx.eval(f.qz)
                n += 1
    _DNR_ON[0] = False
    return n

def _dnr_qmm(x, w, orig, args, kw):
    """DENSE-N-RAM: a call on a converted weight"""
    scales = args[0] if len(args) > 0 else kw.get("scales")
    biases = args[1] if len(args) > 1 else kw.get("biases")
    transpose = args[2] if len(args) > 2 else kw.get("transpose", True)
    group_size = args[3] if len(args) > 3 else kw.get("group_size")
    bits = args[4] if len(args) > 4 else kw.get("bits")
    mode = args[5] if len(args) > 5 else kw.get("mode", "affine")
    _, N, K = _DNR[id(w)]
    lead, rows = _rows(x)
    if (_ON[0] and transpose and mode == "affine" and bits == 4 and group_size == 64 and rows <= R
            and x.dtype == mx.bfloat16 and x.shape[-1] == K and scales.dtype == mx.bfloat16):
        nt = CFG["dense_native_nt"] if N % (32 * CFG["dense_native_nt"]) == 0 else 1
        y = _qmm16n()(_pad16(x.reshape(rows, K)), w, scales, biases, N, 1, nt)
        COUNTS["dense_native"] += 1
        return y[:rows].reshape(*lead, N)
    wm, sm, bm = _dn_unpack(w, scales, biases, N, K)
    COUNTS["dense_native_unpack"] += 1
    extra = {k: v for k, v in kw.items() if k == "stream"}
    return orig(x, wm, sm, bm, transpose=transpose, group_size=group_size, bits=bits, mode=mode, **extra)

def _dn_get(w, scales, biases):''')
rep('''def _qmm(x, w, *args, **kw):
    orig = _ORIG["qmm"]
    if not _ON[0]:''', '''def _qmm(x, w, *args, **kw):
    orig = _ORIG["qmm"]
    if _DNR:
        e = _DNR.get(id(w))
        if e is not None and e[0] is w:                  # DENSE-N-RAM: a converted weight (any phase, ref2 on or off)
            return _dnr_qmm(x, w, orig, args, kw)
    if not _ON[0]:''')
rep('''        self.gs, self.bits, self.mode = ps[0].group_size, ps[0].bits, getattr(ps[0], "mode", "affine")
        self.n = [p.weight.shape[0] for p in ps]
        assert (self.n[0] + self.n[1]) % 64 == 0 and self.n[0] + self.n[1] >= CFG["dense_min_n"], self.n
        assert self.n[2] + self.n[3] < CFG["dense_min_n"], self.n
        cat = lambda a, b: tuple(mx.concatenate([a[k], b[k]]) for k in ("weight", "scales", "biases"))
        self.qz, self.ba = cat(ps[0], ps[1]), cat(ps[2], ps[3])
        mx.eval(self.qz, self.ba)
''', '''        self.gs, self.bits, self.mode = ps[0].group_size, ps[0].bits, getattr(ps[0], "mode", "affine")
        mw = [_dn_mlx(p) for p in ps]                    # DENSE-N-RAM: converted modules unpacked (transient)
        self.n = [int(a[0].shape[0]) for a in mw]
        assert (self.n[0] + self.n[1]) % 64 == 0 and self.n[0] + self.n[1] >= CFG["dense_min_n"], self.n
        assert self.n[2] + self.n[3] < CFG["dense_min_n"], self.n
        cat = lambda a, b: tuple(mx.concatenate([a[i], b[i]]) for i in range(3))
        self.qz, self.ba = cat(mw[0], mw[1]), cat(mw[2], mw[3])
        mx.eval(self.qz, self.ba)
        if _DNR_ON[0]:
            self._convert()

    def _convert(self):
        """DENSE-N-RAM: the qkv | z concat replaced by its native copy (registered); False if already converted"""
        w = self.qz[0]
        e = _DNR.get(id(w))
        if (e is not None and e[0] is w) or w.ndim != 2:
            return False
        N, K = int(w.shape[0]), int(w.shape[1]) * 8
        wn, sn, bn = _dnr_arrays(*self.qz)
        _DNR[id(wn)] = (wn, N, K)
        self.qz = (wn, sn, bn)
        return True

    def _qz_mlx(self):
        e = _DNR.get(id(self.qz[0]))
        return _dn_unpack(*self.qz, e[1], e[2]) if e is not None and e[0] is self.qz[0] else self.qz
''')
rep('''        sl = lambda w, a, e: tuple(t[a:e] for t in w)                    # off: the four stock calls on row slices
        return (q(sl(self.qz, 0, n0)), q(sl(self.qz, n0, n0 + self.n[1])), q(sl(self.ba, 0, n2)),
                q(sl(self.ba, n2, n2 + self.n[3])))''', '''        sl = lambda w, a, e: tuple(t[a:e] for t in w)                    # off: the four stock calls on row slices
        qz = self._qz_mlx()                                              # DENSE-N-RAM: MLX's layout for the slices
        return (q(sl(qz, 0, n0)), q(sl(qz, n0, n0 + self.n[1])), q(sl(self.ba, 0, n2)),
                q(sl(self.ba, n2, n2 + self.n[3])))''')
ast.parse(s)
if DRY:
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "ref2_dnr_dry.py")
    open(out, "w").write(s)
    print("dry: applies and parses ->", out)
else:
    shutil.copy(p, os.path.expanduser("~/Leap/data/ref2_before_dnr.py"))
    tmp = p + ".tmp_dnr"
    open(tmp, "w").write(s)
    os.replace(tmp, p)
    print("applied (backup data/ref2_before_dnr.py); now run ref2.self_test and dn_ram_check.py")
