"""N3C check (Leap, Tue 2026-11-03 16:0x; the orchestrator's routing 15:4x): the conversion prep, default off, bitwise on
data/native_sample (layers 20 and 44; compact pools of the experts the windows use, from records3 and from the sample):
  a. tierglu_3cn's pair kernels: tier_glu_pairs_n == TG.tier_glu_pairs, tier_glu_split_n == TG.tier_glu_split;
  b. ref2 CFG native_3c through ref2's own selector: _tier_glu_sel() -> _tier_glu_3cn (native) == TG.tier_glu
     (row-major), with and without scores; COUNTS native_3c counts it;
  c. ref2._ooo_3cn (native) == BigMoE's original _ooo_call (row-major), with a stand-in gate (post / spin) and random
     residency;
  d. switch off: _tier_glu_sel() is BigMoE's own 3c and the counter stays 0; the guards raise on native_3c + tier_nax4
     and on native_3c + tier_nax_ooo without nax_native;
  e. ref2.self_test with the switch off.
Usage: [BigMoE venv] native3c_check.py -> ~/Leap/data/native3c_check.txt (GPU, small)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
import ref2
ref2.apply()
import bigmoe.tier as BT
TG = BT._tier_glu_mod()
import tierglu_3cn as C3
from tierglu import PROJ

OUT = os.path.expanduser("~/Leap/data/native3c_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"N3C check {time.strftime('%a %H:%M')}")
PLAN = "4433333333"
BT.PLAN = PLAN
R = len(PLAN)
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
ND = os.path.expanduser("~/Leap/data/native_sample")
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
idx = json.load(open(f"{RD}/index.json"))["layers"]
def pools(path, r, sel):
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(r["E"], r["recsz"]))
    sub = np.ascontiguousarray(raw[sel])
    del raw
    C = len(sel)
    msb, lsb = {}, {}
    for p, pr in enumerate(PROJ):
        o, i = SH[pr]
        ow, os_, ob = r["msb_piece_off"][3 * p:3 * p + 3]
        lw, lb = r["lsb_piece_off"][2 * p:2 * p + 2]
        msb[pr] = {"w3": mx.array(np.ascontiguousarray(sub[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(C, o, i * 3 // 32)),
                   "s3": mx.array(np.ascontiguousarray(sub[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(C, o, i // 64)).view(mx.bfloat16),
                   "b3": mx.array(np.ascontiguousarray(sub[:, ob:ob + o * i // 32]).view(np.uint16).reshape(C, o, i // 64)).view(mx.bfloat16)}
        lsb[pr] = {"l1": mx.array(np.ascontiguousarray(sub[:, lw:lw + o * i // 8]).view(np.uint32).reshape(C, o, i // 32)),
                   "b": mx.array(np.ascontiguousarray(sub[:, lb:lb + o * i // 32]).view(np.uint16).reshape(C, o, i // 64)).view(mx.bfloat16)}
    mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    return msb, lsb
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
class Gate:                                                      # stand-in: the spin returns the final slots
    def __init__(self, full):
        self.full = full
    def spin(self, dep, L, n):
        return mx.depends(self.full[L], dep)
def make_post(pre):
    def post(L, ids, after=None):
        dep = mx.concatenate([mx.zeros((1,), mx.int32), pre[L].reshape(-1)])
        return (mx.depends(dep, after) if after is not None else dep), pre[L].size
    return post
rng = np.random.default_rng(7)
mx.random.seed(7)
ok_all = True
for L in (20, 44):
    TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{L}"].astype(np.int32)
          for t in ("code", "chat", "math", "bulgarian")}
    wins = [(t, S, int(t0)) for t in TR for S in (1, 8, 16) for t0 in np.linspace(0, TR[t].shape[0] - S, 2).astype(int)]
    sel = np.unique(np.concatenate([TR[t][t0:t0 + S, :10].reshape(-1) for t, S, t0 in wins]))
    pos = {int(e): i for i, e in enumerate(sel)}
    r = idx[str(L)]
    msb, lsb = pools(f"{RD}/{r['file']}", r, sel)
    msb_n, lsb_n = pools(f"{ND}/{r['file']}", r, sel)
    cnt = {k: [0, 0] for k in ("pairs", "split", "ref2 tier_glu", "ref2 tier_glu+scores", "ref2 ooo")}
    for t, S, t0 in wins:
        ids = np.vectorize(pos.get)(TR[t][t0:t0 + S, :10]).astype(np.int32)
        ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
        x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
        pidx = mx.array(np.sort(rng.choice(S * R, max(1, S * R // 3), replace=False)).astype(np.int32))
        hit = rng.random((S, R)) < 0.6
        out = {"pairs": (TG.tier_glu_pairs(x, msb, lsb, ms_, ls_, pidx, plan=PLAN),
                         C3.tier_glu_pairs_n(x, msb_n, lsb_n, ms_, ls_, pidx, plan=PLAN)),
               "split": (TG.tier_glu_split(x, msb, lsb, ms_, ls_, hit, sc, plan=PLAN),
                         C3.tier_glu_split_n(x, msb_n, lsb_n, ms_, ls_, hit, sc, plan=PLAN))}
        ref2.CFG["native_3c"] = True
        f = ref2._tier_glu_sel()
        assert f is ref2._tier_glu_3cn, f
        out["ref2 tier_glu"] = (TG.tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN), f(x, msb_n, lsb_n, ms_, ls_, plan=PLAN))
        out["ref2 tier_glu+scores"] = (TG.tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc),
                                       f(x, msb_n, lsb_n, ms_, ls_, plan=PLAN, scores=sc))
        # OOO: pre-route slots with ~40 % of pairs not resident (-1), the spin returns the real slots
        pre_m = np.where(rng.random(ids.shape) < 0.6, ids, -1).astype(np.int32)
        pre_l = np.where(rng.random(ids[:, :2].shape) < 0.6, ids[:, :2], -1).astype(np.int32)
        gate = Gate({0: mx.array(ids.reshape(-1)), 1: mx.array(ids[:, :2].reshape(-1).copy())})
        post = make_post({0: mx.array(pre_m), 1: mx.array(pre_l)})
        t_rm = {"segs": BT.segments(PLAN), "base": msb, "sideP": lsb, "side": 1}
        t_nt = {"segs": BT.segments(PLAN), "base": msb_n, "sideP": lsb_n, "side": 1}
        bm = {"gate": gate, "L": 0}
        xin = x.reshape(1, S, 2560)
        out["ref2 ooo"] = (ref2._ORIG["ooo_call"](xin, mx.array(ids).reshape(1, S, R), bm, t_rm, post, sc, None),
                           ref2._ooo_sel(xin, mx.array(ids).reshape(1, S, R), bm, t_nt, post, sc, None))
        ref2.CFG["native_3c"] = False
        mx.eval([v for pq in out.values() for v in pq])
        for k, (a, b) in out.items():
            cnt[k][0] += eqb(a, b)
            cnt[k][1] += 1
    ok = all(c[0] == c[1] for c in cnt.values())
    ok_all &= ok
    say(f"layer {L} ({len(sel)} experts): " + ", ".join(f"{k} {c[0]} / {c[1]}" for k, c in cnt.items()) + f" -> {'PASS' if ok else 'FAIL'}")
    del msb, lsb, msb_n, lsb_n
    mx.clear_cache()
say(f"COUNTS native_3c after the switched calls: {ref2.COUNTS['native_3c']} (expected > 0)")
# d. off is BigMoE's own path; guards
ref2.reset_counts()
ref2.CFG["native_3c"] = False
off_ok = ref2._tier_glu_sel() is TG.tier_glu and ref2.COUNTS["native_3c"] == 0
guards = []
for cfg in ({"native_3c": True, "tier_nax4": True}, {"native_3c": True, "tier_nax_ooo": True}):
    saved = {k: ref2.CFG[k] for k in cfg}
    ref2.CFG.update(cfg)
    try:
        ref2._tier_glu_sel()
        guards.append(False)
    except RuntimeError:
        guards.append(True)
    ref2.CFG.update(saved)
say(f"d. switch off -> BigMoE's own 3c: {off_ok}; guards raise (native_3c + tier_nax4, native_3c + tier_nax_ooo w/o nax_native): {guards}")
st = ref2.self_test()
st_ok = all(bool(v) for v in st.values()) if isinstance(st, dict) else bool(st)
say(f"e. ref2.self_test with the switch off: {'PASS' if st_ok else 'FAIL'} {st}")
ok_all &= off_ok and all(guards) and st_ok and ref2.COUNTS is not None
say(f"N3C check: {'PASS' if ok_all else 'FAIL'}")
