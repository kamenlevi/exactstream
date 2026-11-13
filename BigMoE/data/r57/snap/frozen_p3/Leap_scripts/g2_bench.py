"""G2-PRICE (Leap, Sun 2026-11-08 00:4x; CHAT-150 strike piece G2, experts at 1-4 rows): is a GEMV-form expert call faster
than L's NAX one-call at chat's row counts? FAST v1' (L) runs every <= 16-row expert call on NAX6-N's FAST one-call
(ooo1cq.tier_glu_nax6n_1cq + the combine): 16-row neural-accelerator tiles, so a 1-4-row call pays the tile compute of 16
rows. 3c-N's one-call (ooo1c.tier_glu_3cn_1c, MoonShots' 3c on Friday's converted store: one GEMV per (row, rank) pair,
MLX's qmv arithmetic, the combine fused in kernel B) needs no tile. Both exist and are wired in ref2 (the non-NAX branch
of _ooo_1c_call), so if 3c-N is faster at <= 4 rows the lever is a row-count switch (FAST only, IFEval-gated: the two
kernels' sums differ in order), no new kernel.
Real pools of layers 20 / 44 (native+p11+pad, all 512 resident, C 512), PLAN 4433333333, top-10, no U3 (every row's
4-bit ranks read their sidecar: BIGMOE_POM_LP1 would change the sidecar loader), skewed routing (Zipf 0.9).
Dependent chains of 12 calls alternating the two pools (cold), calls minus the glue-only chain, median of 7, arms
interleaved. MB per call = the union of records read (base slots + unsure rows' sidecar slots).
usage: [BigMoE venv] g2_bench.py [M ...] -> ~/Leap/data/g2_bench.txt"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Breakthrough_keepset"))
assert os.environ.get("BIGMOE_POM_LP1") != "1", "LP1 changes the sidecar loader (set_lp1): run without it"
import mlx.core as mx
import bigmoe.gated as BG
import bigmoe.tier as T
import ooo1c as OC
import ooo1cq as OQ
ext = BG.ext
TG = T._tier_glu_mod()

OUT = os.path.expanduser("~/Leap/data/g2_bench.txt")


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


PLAN, K, R4 = "4433333333", 10, 2
T.PLAN = PLAN
assert getattr(BG, "MODE", "spin") == "spin"
if BG.XLINES < 4:
    BG.XLINES = 4
RS = 32 * (2 + BG.XLINES)
BARE = os.path.expanduser("~/BigMoE/models/bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
C = 512
Ms = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 8, 16]
say(f"\nG2-PRICE {time.strftime('%a %H:%M:%S')} ({os.environ.get('G2_COND', 'share lock')}; MLX {mx.__version__}); M {Ms}")
d_, ix = T.load_index(BARE, layout_ok=True)
nxt = [0]
P = {}


def nbytes(tree):
    if isinstance(tree, mx.array):
        return tree.nbytes
    if isinstance(tree, dict):
        return sum(nbytes(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(nbytes(v) for v in tree)
    return 0


for Lm in (20, 44):
    r = ix["layers"][str(Lm)]
    assert r.get("layout") == "native+p11+pad", r.get("layout")
    L = nxt[0]
    base, _ = T.make_pools(ext, L, 512, C, SH4, d_, r)
    Ls, side, _ = T.make_sidecar(ext, 512, C, SH4, d_, r)
    nxt[0] = Ls + 1
    allids = np.arange(512, dtype=np.int32)
    for j in range(0, 512, 64):
        ext.ensure(L, mx.array(allids[j:j + 64])); ext.ensure(Ls, mx.array(allids[j:j + 64]))
    sm_of = np.array(ext.lookup(L, allids.tolist()), np.int32)
    sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
    assert (sm_of >= 0).all() and (sl_of >= 0).all(), "an expert did not load"
    P[Lm] = (L, Ls, base, side, sm_of, sl_of)
BB, SB = nbytes(P[20][2]) / C, nbytes(P[20][3]) / C
say(f"pools built: layers 20 / 44, 512 experts each, native+p11+pad, C {C}; base record {BB / 1e6:.3f} MB, sidecar "
    f"record {SB / 1e6:.3f} MB")


class Gate:
    pass


gate = Gate()
gate.region = mx.zeros((64 * BG.PAIRS * RS,), mx.int32)
gate.status = mx.zeros((64,), mx.int32)
mx.eval(gate.region, gate.status)
rng = np.random.default_rng(78)
clock = time.perf_counter


def case(Lm, Mx):
    Lb, Ls, base, side, sm_of, sl_of = P[Lm]
    pop = 1.0 / np.arange(1, 513) ** 0.9
    pop = pop[rng.permutation(512)]
    pop /= pop.sum()
    ids = np.stack([rng.choice(512, K, replace=False, p=pop) for _ in range(Mx)]).astype(np.int32)
    sr = np.zeros(Mx, bool)                     # no U3 sure rows: both kernels read every 4-bit rank's sidecar
    pm = sm_of[ids].astype(np.int32)
    pl = np.where(sr[:, None], -2, sl_of[ids[:, :R4]]).astype(np.int32)
    sc = rng.uniform(0.02, 0.3, (Mx, K)).astype(np.float32)
    x = (mx.random.normal((Mx, 2560)) * 0.5).astype(mx.bfloat16)
    c = dict(x=x, base=base, side=side, pre_m=mx.array(pm), pre_l=mx.array(pl), ids=mx.array(ids),
             sc=mx.array(sc).astype(mx.bfloat16), offs=mx.array([0, 0, 1], mx.int32),
             mb=(len(np.unique(pm)) * BB + len(np.unique(pl[pl >= 0])) * SB) / 1e6, groups=len(np.unique(pm)))
    mx.eval(c["x"], c["pre_m"], c["pre_l"], c["ids"], c["sc"], c["offs"])
    return c


def nax1cq(c, x):
    M_ = x.shape[0]
    y, _, _ = OQ.tier_glu_nax6n_1cq(x, c["base"], c["side"], c["pre_m"], c["pre_l"], c["ids"], PLAN, gate.region, c["offs"],
                                   gate.status)
    return TG.combine_phase(y.reshape(M_ * K, -1), y.reshape(M_ * K, -1), mx.ones((M_, K), mx.int32), c["sc"].reshape(-1, K))


def c3n1c(c, x):
    y, _, _ = OC.tier_glu_3cn_1c(x, c["base"], c["side"], c["pre_m"], c["pre_l"], PLAN, gate.region, c["offs"], gate.status,
                                c["sc"].reshape(-1, K))
    return y


ARMS = {"nax1cq": nax1cq, "3cn1c": c3n1c}


def chain(Cs, fn):
    tt = []
    for calls in (False, True):
        prev = mx.zeros((1,), mx.float32)
        for c in Cs:
            x = c["x"] + (prev * 0).astype(mx.bfloat16)
            prev = (fn(c, x) if calls else x).reshape(-1)[:1].astype(mx.float32)
        t0 = clock(); mx.eval(prev); tt.append(clock() - t0)
    return 1e6 * (tt[1] - tt[0]) / len(Cs)


for Mx in Ms:
    Cs = [case(20 if i % 2 == 0 else 44, Mx) for i in range(12)]
    ys = {a: f(Cs[0], Cs[0]["x"]).astype(mx.float32) for a, f in ARMS.items()}
    mx.eval(list(ys.values()))
    err = float(mx.max(mx.abs(ys["3cn1c"] - ys["nax1cq"])).item() / max(1e-30, mx.max(mx.abs(ys["nax1cq"])).item()))
    for a in ARMS:
        chain(Cs, ARMS[a])
    acc = {a: [] for a in ARMS}
    names = list(ARMS)
    for r in range(7):
        for a in (names if r % 2 == 0 else names[::-1]):
            acc[a].append(chain(Cs, ARMS[a]))
    t = {a: float(np.median(v)) for a, v in acc.items()}
    mb = float(np.mean([c["mb"] for c in Cs]))
    g = float(np.mean([c["groups"] for c in Cs]))
    say(f"  M {Mx:2d} (~{g:.0f} base groups, {mb:.1f} MB per call): "
        + " | ".join(f"{a} {t[a]:.1f} us ({mb / t[a] * 1e3:.0f} GB/s)" for a in ARMS)
        + f" | 3cn1c / nax1cq {t['3cn1c'] / t['nax1cq']:.3f}; x 48 layers: {48 * (t['3cn1c'] - t['nax1cq']) / 1e3:+.2f} ms per check"
        + f"; max rel diff {err:.1e}")
say("G2-PRICE done")
