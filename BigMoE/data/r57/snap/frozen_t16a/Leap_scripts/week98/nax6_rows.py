"""NAX6-ROWS (WEEK98 pillar 2; Leap, Sun 15 Nov 2026 02:5x): is the exact-45 expert kernel (NAX6, REF2-T tier layout
4433333333) row-invariant at 32 rows? One 32-row call vs the same rows as two 16-row calls (same ids, same pools):
per-(row, rank) outputs bitwise, then combine_phase (the per-row k-sum with scores) at 32 vs 16 + 16.
Pools are random in NAX6's layout (E 512), so the bits test the kernel's arithmetic, not the model. Routing has the
32-row window's overlap: ids drawn so the window's union is ~108 distinct (BigMoE step 0's mean), each row 10 distinct,
in gate-rank order. Three windows, each also at 17 / 24 / 31 rows (rows 0-15 vs the rest).
Line: all bitwise -> NAX6 is not where a 32-row drift comes from; any differing pair -> the kernel's grouping depends on
the call's rows (then: which groups, and the fix is a row-invariant grouping).
usage: [venv] nax6_rows.py (MEM ~3 GB; share lock, bits only)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_t16a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
import tierglu as TG
import tierglu_nax6 as N6
PLAN, RK, R4, H = "4433333333", 10, 2, 2560
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
E = 512


def say(s):
    print(s, flush=True)


def rand_pool():
    msb, lsb = {}, {}
    for pr in TG.PROJ:
        o, i = SH[pr]
        sc = mx.random.uniform(0.001, 0.003, (E, o, i // 64)).astype(mx.bfloat16)
        msb[pr] = {"w3": mx.random.randint(0, 2 ** 31, (E, o, i * 3 // 32)).astype(mx.uint32), "s3": sc,
                   "b3": (-4.0 * sc.astype(mx.float32)).astype(mx.bfloat16)}
        lsb[pr] = {"l1": mx.random.randint(0, 2 ** 31, (E, o, i // 32)).astype(mx.uint32),
                   "b": mx.random.uniform(-1e-3, 1e-3, (E, o, i // 64)).astype(mx.bfloat16)}
    mx.eval([v for dd in (msb, lsb) for pr in dd for v in dd[pr].values()])
    return msb, lsb


def window(rng, M, union=108):
    pool = rng.choice(E, size=union, replace=False)
    ids = np.stack([rng.choice(pool, size=RK, replace=False) for _ in range(M)]).astype(np.int32)
    return ids


def call(x, msb, lsb, ids):
    sm, sl = mx.array(ids), mx.array(np.ascontiguousarray(ids[:, :R4]))
    y = N6.tier_glu_nax6(x, msb, lsb, sm, sl, plan=PLAN, grp=N6.gpu_grp(sm, sl, PLAN))
    return y.reshape(ids.shape[0] * RK, -1)


def comb(y, sc, M):
    ones = mx.ones((M, RK), mx.int32)
    return TG.combine_phase(y, y, ones, sc)


mx.random.seed(3)
t0 = time.time()
msb, lsb = rand_pool()
say(f"NAX6-ROWS {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}; pools {E} experts in {time.time() - t0:.0f} s")
rng = np.random.default_rng(11)
allok = True
for w in range(3):
    ids32 = window(rng, 32)
    x32 = (mx.random.normal((32, H)) * 0.5).astype(mx.bfloat16)
    sc32 = mx.softmax(mx.random.normal((32, RK)), axis=-1).astype(mx.bfloat16)
    for M in (32, 17, 24, 31):
        ids, x, sc = ids32[:M], x32[:M], sc32[:M]
        y_one = call(x, msb, lsb, ids)
        y_two = mx.concatenate([call(x[:16], msb, lsb, ids[:16]), call(x[16:], msb, lsb, ids[16:])], axis=0)
        c_one = comb(y_one, sc, M)
        c_two = mx.concatenate([comb(y_two[:16 * RK], sc[:16], 16), comb(y_two[16 * RK:], sc[16:], M - 16)], axis=0)
        eq_p = mx.all(y_one == y_two, axis=-1)
        eq_c = mx.all(c_one == c_two, axis=-1)
        d = mx.max(mx.abs(y_one.astype(mx.float32) - y_two.astype(mx.float32)))
        mx.eval(eq_p, eq_c, d)
        np_, nc_ = int(np.array(eq_p).sum()), int(np.array(eq_c).sum())
        ok = np_ == M * RK and nc_ == M
        allok &= ok
        say(f"  window {w}, M {M} (union {len(np.unique(ids))}): pairs bitwise {np_} / {M * RK}, combined rows {nc_} / {M}, "
            f"max |pair diff| {float(d.item()):.3g}")
say(f"NAX6-ROWS verdict: {'ALL BITWISE -> NAX6 + combine_phase are row-invariant at 17-32 rows' if allok else 'NOT BITWISE -> the grouping depends on the rows'}")
