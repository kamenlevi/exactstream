"""ref2 CFG nax_hybrid / nax_native check (Leap, Mon 2026-11-02 23:1x; nax_native added 23:3x): through ref2._nax6_glu on the real layer-20 pools with real
routing (code trace, S 16 and S 1, with scores): switch off == tier_glu_nax6 + combine_phase (bitwise, the shipped C1
path), switch on == tier_glu_nax6h + combine_phase (bitwise), COUNTS nax_hybrid counts the hybrid calls; then
ref2.self_test with the switch off.
Usage: [BigMoE venv] nax_hybrid_check.py -> ~/Leap/data/nax_hybrid_check.txt"""
import json, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
import ref2
ref2.apply()
N6 = ref2._nax6mod()
import tierglu_nax6h as NH
import tierglu_nax6n as NN
import bigmoe.tier as BT
TG = BT._tier_glu_mod()
from tierglu import PROJ
OUT = os.path.expanduser("~/Leap/data/nax_hybrid_check.txt")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
LAYER, PLAN = 20, "4433333333"
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
E, REC = idx["E"], idx["recsz"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
msb, lsb = {}, {}
for p, pr in enumerate(PROJ):
    o, i = SH[pr]
    ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
    msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
               "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
               "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
    lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
               "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
del raw
mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
eq = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/code_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
mx.random.seed(0)
ok = True
for S in (16, 1):
    ids = tr[200:200 + S, :10].copy()
    ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
    x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
    sc = mx.softmax(mx.random.normal((S, 10)), axis=-1).astype(mx.bfloat16)
    one = mx.ones((S, 10), mx.int32)
    grp = N6.gpu_grp(ms_, ls_, PLAN)
    want_off = TG.combine_phase(*(lambda y: (y, y))(N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp).reshape(S * 10, -1)), one, sc)
    want_on = TG.combine_phase(*(lambda y: (y, y))(NH.tier_glu_nax6h(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp).reshape(S * 10, -1)), one, sc)
    ref2.CFG["nax_hybrid"] = False
    c0 = ref2.counts()["nax_hybrid"]
    got_off = ref2._nax6_glu(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc)
    ref2.CFG["nax_hybrid"] = True
    got_on = ref2._nax6_glu(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc)
    ref2.CFG["nax_hybrid"] = False
    mx.eval(want_off, want_on, got_off, got_on)
    c1 = ref2.counts()["nax_hybrid"]
    r = (eq(got_off, want_off), eq(got_on, want_on), c1 - c0 == 1, eq(got_on, got_off))
    ok &= r[0] and r[1] and r[2]
    say(f"S {S}: off == NAX6 {r[0]}; on == NAX6-H {r[1]}; hybrid counted once {r[2]}; on == off {r[3]} (FAST: may differ)")
msb_n, lsb_n = NN.repack(msb, lsb)
mx.eval([v for d in (msb_n, lsb_n) for pr in d for v in d[pr].values()])
for S in (16, 1):
    ids = tr[300:300 + S, :10].copy()
    ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
    x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
    sc = mx.softmax(mx.random.normal((S, 10)), axis=-1).astype(mx.bfloat16)
    one = mx.ones((S, 10), mx.int32)
    grp = N6.gpu_grp(ms_, ls_, PLAN)
    w6 = TG.combine_phase(*(lambda y: (y, y))(N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp).reshape(S * 10, -1)), one, sc)
    wh = TG.combine_phase(*(lambda y: (y, y))(NH.tier_glu_nax6h(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp).reshape(S * 10, -1)), one, sc)
    c0 = ref2.counts()["nax_native"]
    ref2.CFG["nax_native"] = True
    g_n = ref2._nax6_glu(x, msb_n, lsb_n, ms_, ls_, plan=PLAN, scores=sc)
    ref2.CFG["nax_hybrid"] = True
    g_nh = ref2._nax6_glu(x, msb_n, lsb_n, ms_, ls_, plan=PLAN, scores=sc)
    ref2.CFG["nax_hybrid"] = False
    ref2.CFG["nax_native"] = False
    mx.eval(w6, wh, g_n, g_nh)
    r = (eq(g_n, w6), eq(g_nh, wh), ref2.counts()["nax_native"] - c0 == 2)
    ok &= all(r)
    say(f"S {S}: nax_native (native pools) == NAX6 {r[0]}; nax_native + nax_hybrid == NAX6-H {r[1]}; counted {r[2]}")
st = ref2.self_test()
say(f"ref2.self_test (switch off): {'PASS' if all(st.values()) else 'FAIL'} {st}")
say(f"nax_hybrid / nax_native check: {'PASS' if ok and all(st.values()) else 'FAIL'}")
open(OUT, "w").write("\n".join(lines) + "\n")
