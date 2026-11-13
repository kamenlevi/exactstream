"""NATIVE-FRI step 1 (Leap, Wed 2026-11-04 15:0x; TABLE_350 'NATIVE-FRI'; the orchestrator's option A): the converted-store
path end to end at layer level, with the GPU kernels. BigMoE's live package (tier.load_index / make_pools / make_sidecar /
ext.ensure with its layout patch; read-only use) loads the same 64 experts of layers 20 and 47 from
  (i)  my scratch store data/scratch_fri (APFS clones of records3 L20 / L47 converted in place by convert_records_p11.py
       --pad: native+p11+pad, 4 KB-aligned pieces, direct-split LSB reads), and
  (ii) today's row-major models/bare/records3,
then, on 3 random routings over the loaded experts at S 16 / 8 / 1 (PLAN 4433333333):
  3c-N(engine-loaded converted pools) == 3c(engine-loaded row-major pools) bitwise: BigMoE's tier_glu (REF2-T's kernel),
    with scores, and the two phases + combine_phase;
  NAX6-N(converted) == NAX6(row-major) bitwise.
BigMoE's scripts/test_layout_engine.py already checks the bytes (every slot == its record's part, the prefill rebuild).
Usage: [BigMoE venv] nat_engine_check.py -> ~/Leap/data/nat_engine_check.txt (GPU ~1 GB; share lock)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
import mlx.core as mx
import ref2
ref2.apply()
import bigmoe.gated as BG                                   # the ext the engine loads
import bigmoe.tier as T
import tierglu_3cn as C3
import tierglu_nax6 as N6
import tierglu_nax6n as NN
ext = BG.ext
TG = T._tier_glu_mod()

OUT = os.path.expanduser("~/Leap/data/nat_engine_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nNATIVE-FRI engine-path check {time.strftime('%a %H:%M:%S')} (ext {os.path.basename(os.path.dirname(ext.__file__))}, "
    f"set_direct_split {hasattr(ext, 'set_direct_split')}, layouts {sorted(T.LAYOUTS)})")
SCR = os.path.expanduser("~/Leap/data/scratch_fri")
BARE = os.path.expanduser("~/BigMoE/models/bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
PLAN, C = "4433333333", 64
d_c, ix_c = T.load_index(SCR, layout_ok=True)
d_r, ix_r = T.load_index(BARE)
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
nxt = [0]
def pools(d, r):
    L = nxt[0]
    base, _ = T.make_pools(ext, L, 512, C, SH4, d, r)
    Ls, side, _ = T.make_sidecar(ext, 512, C, SH4, d, r)
    nxt[0] = Ls + 1
    return L, Ls, base, side

ok_all = True
mx.random.seed(20)
for Lm in (20, 47):
    rc, rr = ix_c["layers"][str(Lm)], ix_r["layers"][str(Lm)]
    assert rc.get("layout") == "native+p11+pad" and rr.get("layout", "row-major") == "row-major", (rc.get("layout"), rr.get("layout"))
    Lc, Lcs, msb_c, lsb_c = pools(d_c, rc)
    Lr, Lrs, msb_r, lsb_r = pools(d_r, rr)
    ids = np.random.default_rng(Lm).permutation(512)[:C].astype(np.int32)
    for L_ in (Lc, Lcs, Lr, Lrs):
        ext.ensure(L_, mx.array(ids))
    slot = {L_: dict(zip(ids.tolist(), ext.lookup(L_, ids.tolist()))) for L_ in (Lc, Lcs, Lr, Lrs)}
    assert all(s >= 0 for m in slot.values() for s in m.values()), "an expert did not load"
    same = slot[Lc] == slot[Lr] and slot[Lcs] == slot[Lrs]
    rng = np.random.default_rng(100 + Lm)
    cnt = {}
    for S in (16, 8, 1):
        for _ in range(3):
            e = np.stack([rng.choice(ids, 10, replace=False) for _ in range(S)]).astype(np.int32)
            sl = lambda L_, a: mx.array(np.vectorize(slot[L_].get)(a).astype(np.int32))
            ms_c, ls_c, ms_r, ls_r = sl(Lc, e), sl(Lcs, e[:, :2].copy()), sl(Lr, e), sl(Lrs, e[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, 10)), axis=-1).astype(mx.bfloat16)
            ph = mx.array(rng.integers(1, 3, (S, 10)).astype(np.int32))
            pairs = {
                "3c": (lambda: TG.tier_glu(x, msb_r, lsb_r, ms_r, ls_r, plan=PLAN),
                       lambda: C3.tier_glu_n(x, msb_c, lsb_c, ms_c, ls_c, plan=PLAN)),
                "3c+scores": (lambda: TG.tier_glu(x, msb_r, lsb_r, ms_r, ls_r, plan=PLAN, scores=sc),
                              lambda: C3.tier_glu_n(x, msb_c, lsb_c, ms_c, ls_c, plan=PLAN, scores=sc)),
                "3c phases": (lambda: TG.combine_phase(TG.tier_glu_phase(x, msb_r, lsb_r, ms_r, ls_r, ph, 1, plan=PLAN),
                                                       TG.tier_glu_phase(x, msb_r, lsb_r, ms_r, ls_r, ph, 2, plan=PLAN), ph, sc),
                              lambda: TG.combine_phase(C3.tier_glu_phase_n(x, msb_c, lsb_c, ms_c, ls_c, ph, 1, plan=PLAN),
                                                       C3.tier_glu_phase_n(x, msb_c, lsb_c, ms_c, ls_c, ph, 2, plan=PLAN), ph, sc)),
                "NAX6": (lambda: N6.tier_glu_nax6(x, msb_r, lsb_r, ms_r, ls_r, plan=PLAN, grp=N6.gpu_grp(ms_r, ls_r, PLAN)),
                         lambda: NN.tier_glu_nax6n(x, msb_c, lsb_c, ms_c, ls_c, plan=PLAN, grp=N6.gpu_grp(ms_c, ls_c, PLAN)))}
            for k, (fa, fb) in pairs.items():
                a_, b_ = fa(), fb()
                mx.eval(a_, b_)
                c_ = cnt.setdefault(k, [0, 0])
                c_[0] += eqb(a_, b_)
                c_[1] += 1
    ok = all(v[0] == v[1] for v in cnt.values())
    ok_all &= ok
    say(f"  layer {Lm}: 64 experts loaded through the engine from both stores (same slots: {same}); "
        + ", ".join(f"{k} converted == row-major {v[0]}/{v[1]}" for k, v in cnt.items()) + f" -> {'PASS' if ok else 'FAIL'}")
st = ref2.self_test()
st_ok = all(bool(v) for v in st.values())
say(f"NATIVE-FRI engine-path check: {'PASS' if ok_all and st_ok else 'FAIL'} (ref2.self_test {'PASS' if st_ok else 'FAIL'})")
