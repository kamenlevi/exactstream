"""Fit12 (Fri 6 Nov 09:4x; 12 GB plan lever 3, the orchestrator 09:3x): Leap's Friday kernel switches, ported onto the
12 GB T3 path. That path (pinned bt_b12 8db71cc537 + the v3p tier engine + MTPLX's own dense and attention calls + the
d7b_live drafter on MLX's quantized_matmul) reads none of Leap's switches as they are: it loads no ref2, no d7b_nax, no r57.
Leap's kernels come from pinned copies in /Users/kamen/Fit12/pinned/leap_1106 (MANIFEST.txt: sha256; copied 09:38 from
~/Leap/scripts, read-only; in the copies only the live-path inserts are removed, so the pinned qmm16 / qmm32 load).
Switches, set per arm by set_arm() (all off = the T3 path unchanged; install() alone changes nothing):
  dnax  D7BNAX_32=2's <= 16-row rule: the drafter's 4-bit layer projections (d7b_nax.install) run qmm16c with the per-shape
        K split. Drafts change by fp32 accumulation order only, and every draft is verified: EXACT for the target's tokens.
  att1  R-62 att_single, pass 1 only: a 16-row verify SDPA as ONE fused call instead of MTPLX's 12 head chunks, while
        T2._P1 is set. Bitwise == the chunks below 512 keys (Leap R-62), different past them. Pass 1 is the FAST part.
  dn1   DN-KS, pass 1 only: 4-bit g64 dense matmuls of <= 16 rows with N >= 1024 (ref2's dense_min_n) on qmm16c with the
        per-shape K split, while T2._P1 is set. (ref2's DN-KS rides qmm16n on a native copy, +~1.5 GB, which 12 GB cannot
        hold; qmm16c takes the same split on MLX's own layout.) FAST: exact <= 4-row calls keep MLX's kernels.
FAST-NAX (R57_NAX / REF2_NAX_1CQ) has no counterpart: the 12 GB experts run v3p's tier GLU (gather_qmm) on records3m."""
import os, sys

PIN = "/Users/kamen/Fit12/pinned/leap_1106"
ON = {"dnax": False, "att1": False, "dn1": False}
CNT = dict(att1=0, dn1=0, dn1_ks=0, p1_qmm_mlx=0)
_S = {}


def _ks(N, K):
    """d7b_nax's per-shape K split (Leap Fri 08:2x): the smallest ks dividing K / 64, <= 10, giving >= 320 simdgroups"""
    g, tiles, ks = K // 64, N // 32, 1
    for k in (1, 2, 4, 5, 8, 10):
        if g % k == 0:
            ks = k
            if tiles * k >= 320:
                break
    return ks


def _pinned():
    sys.path.insert(0, PIN)
    try:
        import qmm16, qmm32, d7b_nax
    finally:
        sys.path.remove(PIN)
    for m in (qmm16, qmm32, d7b_nax):
        assert os.path.dirname(os.path.abspath(m.__file__)) == PIN, f"{m.__name__} from {m.__file__}, not the pinned copy"
    return qmm16, d7b_nax


def install(model, draft):
    """wrap MTPLX's _verify_sdpa and mx.quantized_matmul (inert until ON says so) and load the pinned kernels. Call after
    T3.apply(B), B.load_b12 and T3.load_draft."""
    import mlx.core as mx
    from mtplx.models import qwen4_exp as Q
    import t2_twopass as T2
    qmm16, d7b_nax = _pinned()
    d7b_nax.D32[0] = 2                                   # D7BNAX_32=2: per-shape ks on the <= 16-row qmm16c calls
    qmm16c = qmm16.qmm16c
    vs0, qm0 = Q._verify_sdpa, mx.quantized_matmul
    _S.update(vs0=vs0, qm0=qm0, nax=d7b_nax, draft=draft)

    def _vsdpa(q, k, v, *, scale, mask):
        if ON["att1"] and T2._P1[0] and q.shape[2] == 16:
            CNT["att1"] += 1
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        return vs0(q, k, v, scale=scale, mask=mask)

    def _qmm(x, w, *a, **kw):
        if ON["dn1"] and T2._P1[0]:
            sc = a[0] if len(a) > 0 else kw.get("scales")
            bi = a[1] if len(a) > 1 else kw.get("biases")
            tr = a[2] if len(a) > 2 else kw.get("transpose", True)
            gs = a[3] if len(a) > 3 else kw.get("group_size")
            bt = a[4] if len(a) > 4 else kw.get("bits")
            md = a[5] if len(a) > 5 else kw.get("mode", "affine")
            K, N = x.shape[-1], w.shape[0]
            rows = x.size // K
            if (tr and md == "affine" and gs == 64 and bt == 4 and bi is not None and sc is not None and rows <= 16
                    and x.dtype == mx.bfloat16 and sc.dtype == mx.bfloat16 and w.dtype == mx.uint32
                    and w.shape[1] * 8 == K and N >= 1024 and N % 32 == 0 and K % 64 == 0):
                ks = _ks(N, K)
                CNT["dn1"] += 1
                CNT["dn1_ks"] += ks > 1
                return qmm16c(x.reshape(rows, K), w, sc, bi, ks).reshape(*x.shape[:-1], N)
            CNT["p1_qmm_mlx"] += 1
        return qm0(x, w, *a, **kw)

    Q._verify_sdpa = _vsdpa
    mx.quantized_matmul = _qmm
    return d7b_nax


def set_arm(dnax=False, att1=False, dn1=False):
    """switch the ports for one arm; returns the number of drafter projections on NaxQL (0 when dnax is off)"""
    ON.update(dnax=bool(dnax), att1=bool(att1), dn1=bool(dn1))
    nax, draft = _S["nax"], _S["draft"]
    nax.uninstall(draft)
    return nax.install(draft) if dnax else 0


def counts():
    """the engagement counters so far: the pass-1 ports' and the drafter's (d7b_nax.COUNTS)"""
    return dict(CNT, **{f"nax_{k}": v for k, v in _S["nax"].COUNTS.items()})


def selftest():
    """GPU, seconds: qmm16c with the per-shape ks vs MLX's quantized_matmul on random 4-bit weights of the 125B's dense
    shapes (16 rows), and the single SDPA vs MTPLX's head chunks (16 rows, GQA 12, 300 and 700 keys); prints max rel err"""
    import mlx.core as mx
    from mtplx.models import qwen4_exp as Q
    qmm16, _ = _pinned()
    mx.random.seed(0)
    out = []
    for N, K in ((2560, 4096), (1024, 2560), (16384, 2560), (2560, 2560), (5120, 2560), (2560, 1280)):
        w = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.02
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        x = mx.random.normal((16, K)).astype(mx.bfloat16)
        y0 = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
        y1 = qmm16.qmm16c(x, wq, s, b, _ks(N, K))
        e = (mx.abs(y1.astype(mx.float32) - y0.astype(mx.float32)).max() / mx.abs(y0.astype(mx.float32)).max()).item()
        out.append(f"qmm16c N {N} K {K} ks {_ks(N, K)}: max rel err {e:.2e}")
    nq, nkv, D = 24, 2, 256
    for T in (300, 700):
        q = mx.random.normal((1, nq, 16, D)).astype(mx.bfloat16)
        k = mx.random.normal((1, nkv, T, D)).astype(mx.bfloat16)
        v = mx.random.normal((1, nkv, T, D)).astype(mx.bfloat16)
        qpos = (T - 16) + mx.arange(16)
        mask = (mx.arange(T)[None, :] <= qpos[:, None])[None, None]
        y0 = Q._verify_sdpa(q, k, v, scale=D ** -0.5, mask=mask)
        y1 = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask=mask)
        d = mx.abs(y1.astype(mx.float32) - y0.astype(mx.float32))
        out.append(f"single SDPA vs head chunks, {T} keys: max abs diff {d.max().item():.2e}, "
                   f"bitwise {bool(mx.array_equal(y0, y1).item())}")
    print("\n".join(out), flush=True)
    return out
