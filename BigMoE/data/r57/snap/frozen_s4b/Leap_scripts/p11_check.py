"""P11 check (Leap, Tue 2026-11-03 17:2x; TABLE_350 'P11'; M-27b's gate): 3c-N and NAX6-N reading 11-bit packed scales
(scripts/p11.py pieces) == the same kernels on bf16 native pieces, bit for bit, on EVERY piece of full layers.
Per layer (20, 44): records3 (row-major) -> p11.to_p11 in RAM (chunks) -> p11 pools (s3 / b3 / b uint32 [512, 8804]) and
bf16 native pools (the same records, native order, unpacked). Routing: 16 calls of 16 rows x 10 ranks in which every
expert is a 4-bit rank (0 / 1: s3 + LSB b) in one call and a 3-bit rank (2-9: s3 + b3) in about four, so every s3 / b3 /
b piece of gate / up / down is read. Compared, y bitwise:
  3c-N: tier_glu_n (A + B), + scores (A + BC), tier_glu_phase_n x 2 + combine (AP + BP), tier_glu_pairs_n (AS + BS),
        tier_glu_split_n; through ref2's native_3c switch: _tier_glu_sel() and _ooo_sel();
  NAX6-N: tier_glu_nax6n.
Speed (rough, shared; layer 20, real code routing, S 16 chains of 16): each kernel on p11 vs bf16 pools.
Then ref2.self_test (these modules are imported by ref2 live).
Usage: [BigMoE venv] p11_check.py -> ~/Leap/data/p11_check.txt (GPU ~4 GB; share lock)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import ref2
ref2.apply()
import bigmoe.tier as BT
TG = BT._tier_glu_mod()
import tierglu_3cn as C3
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import p11 as P11
from tierglu import PROJ

OUT = os.path.expanduser("~/Leap/data/p11_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nP11 check {time.strftime('%a %H:%M')} ({os.environ.get('P11_COND', 'share lock')})")
PLAN = "4433333333"
BT.PLAN = PLAN
R = len(PLAN)
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
idx = json.load(open(f"{RD}/index.json"))["layers"]
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def pools(r):
    """native bf16 pools and native p11 pools of all experts of a layer (the same records)"""
    E = r["E"]
    raw = np.memmap(f"{RD}/{r['file']}", dtype=np.uint8, mode="r", shape=(E, r["recsz"]))
    q = P11.layout_p11(r)
    new = np.empty((E, q["recsz"]), np.uint8)
    for e0 in range(0, E, 64):
        new[e0:e0 + 64] = P11.to_p11(np.asarray(raw[e0:e0 + 64]), r)[0]
    del raw
    nat, pk = {"msb": {}, "lsb": {}}, {"msb": {}, "lsb": {}}
    for nm, part, pr, off, sz, dt, per, packed in P11.pieces(q):
        o, i = SH[pr]
        n = o * (i // 64) * per
        if packed:
            w = np.ascontiguousarray(new[:, off:off + sz]).view(np.uint32)
            pk[part].setdefault(pr, {})[nm] = mx.array(w)
            nat[part].setdefault(pr, {})[nm] = mx.array(P11.unpack(w, n).reshape(E, o, i // 64)).view(mx.bfloat16)
        else:
            a = np.ascontiguousarray(new[:, off:off + n * np.dtype(dt).itemsize]).view(dt)
            shape = (E, o, i * 3 // 32) if nm == "w3" else (E, o, i // 32)
            arr = mx.array(a.reshape(shape))
            nat[part].setdefault(pr, {})[nm] = arr
            pk[part].setdefault(pr, {})[nm] = arr
    del new
    mx.eval([v for d in (nat, pk) for part in d.values() for pr in part.values() for v in pr.values()])
    return nat["msb"], nat["lsb"], pk["msb"], pk["lsb"]


def cover_routing(rng, E=512, calls=16, S=16):
    """calls x [S, 10]: call c's 4-bit ranks hold experts 32c .. 32c + 31; 3-bit ranks a shuffled cover of all experts"""
    perm = rng.permutation(np.tile(np.arange(E), calls * S * 8 // E + 1))
    out, k = [], 0
    for c in range(calls):
        ids = np.empty((S, 10), np.int32)
        for s in range(S):
            ids[s, :2] = (32 * c + 2 * s, 32 * c + 2 * s + 1)
            row = []
            while len(row) < 8:
                e = int(perm[k % perm.size]); k += 1
                if e not in row and e not in ids[s, :2]:
                    row.append(e)
            ids[s, 2:] = row
        out.append(ids)
    return out


class Gate:                                                      # stand-in for the OOO gate: the spin returns final slots
    def __init__(self, full):
        self.full = full
    def spin(self, dep, L, n):
        return mx.depends(self.full[L], dep)
def make_post(pre):
    def post(L, ids, after=None):
        dep = mx.concatenate([mx.zeros((1,), mx.int32), pre[L].reshape(-1)])
        return (mx.depends(dep, after) if after is not None else dep), pre[L].size
    return post


ok_all = True
mx.random.seed(11)
rng = np.random.default_rng(11)
for L in (20, 44):
    t0 = time.time()
    msb, lsb, msb_p, lsb_p = pools(idx[str(L)])
    tb = time.time() - t0
    cnt = {}
    def tally(k, a, b):
        c = cnt.setdefault(k, [0, 0]); c[0] += eqb(a, b); c[1] += 1
    seen_l, seen_m = set(), set()
    for ids in cover_routing(rng):
        seen_l |= set(ids[:, :2].reshape(-1).tolist()); seen_m |= set(ids[:, 2:].reshape(-1).tolist())
        S = ids.shape[0]
        ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
        x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
        ph = mx.array(rng.integers(1, 3, (S, R)).astype(np.int32))
        pidx = mx.array(np.sort(rng.choice(S * R, S * R // 3, replace=False)).astype(np.int32))
        hit = rng.random((S, R)) < 0.6
        for nm, f in (("tier_glu_n", lambda m, l: C3.tier_glu_n(x, m, l, ms_, ls_, plan=PLAN)),
                      ("tier_glu_n+scores", lambda m, l: C3.tier_glu_n(x, m, l, ms_, ls_, plan=PLAN, scores=sc)),
                      ("phases", lambda m, l: TG.combine_phase(C3.tier_glu_phase_n(x, m, l, ms_, ls_, ph, 1, plan=PLAN),
                                                              C3.tier_glu_phase_n(x, m, l, ms_, ls_, ph, 2, plan=PLAN), ph, sc)),
                      ("pairs", lambda m, l: C3.tier_glu_pairs_n(x, m, l, ms_, ls_, pidx, plan=PLAN)),
                      ("split", lambda m, l: C3.tier_glu_split_n(x, m, l, ms_, ls_, hit, sc, plan=PLAN)),
                      ("NAX6-N", lambda m, l: NN.tier_glu_nax6n(x, m, l, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)))):
            a, b = f(msb, lsb), f(msb_p, lsb_p)
            mx.eval(a, b)
            tally(nm, a, b)
        # ref2's switch with p11 pools: the selector and the OOO path
        ref2.CFG["native_3c"] = True
        f = ref2._tier_glu_sel()
        a, b = f(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc), f(x, msb_p, lsb_p, ms_, ls_, plan=PLAN, scores=sc)
        pre_m = np.where(rng.random(ids.shape) < 0.6, ids, -1).astype(np.int32)
        pre_l = np.where(rng.random(ids[:, :2].shape) < 0.6, ids[:, :2], -1).astype(np.int32)
        gate = Gate({0: mx.array(ids.reshape(-1)), 1: mx.array(ids[:, :2].reshape(-1).copy())})
        post = make_post({0: mx.array(pre_m), 1: mx.array(pre_l)})
        bm = {"gate": gate, "L": 0}
        xin, idr = x.reshape(1, S, 2560), mx.array(ids).reshape(1, S, R)
        oa = ref2._ooo_sel(xin, idr, bm, {"segs": BT.segments(PLAN), "base": msb, "sideP": lsb, "side": 1}, post, sc, None)
        ob = ref2._ooo_sel(xin, idr, bm, {"segs": BT.segments(PLAN), "base": msb_p, "sideP": lsb_p, "side": 1}, post, sc, None)
        ref2.CFG["native_3c"] = False
        mx.eval(a, b, oa, ob)
        tally("ref2 native_3c tier_glu", a, b)
        tally("ref2 native_3c ooo", oa, ob)
    ok = all(c[0] == c[1] for c in cnt.values()) and len(seen_l) == 512 and len(seen_m) == 512
    ok_all &= ok
    say(f"layer {L} (pools built in {tb:.0f} s; experts seen at 4-bit ranks {len(seen_l)}, at 3-bit ranks {len(seen_m)}): " +
        ", ".join(f"{k} {c[0]} / {c[1]}" for k, c in cnt.items()) + f" -> {'PASS' if ok else 'FAIL'}")
    if L == 20:                                                    # speed, rough
        tr = np.load(os.path.expanduser("~/Breakthrough/data/expA/traces125/code_800_g.npz"))["L20"].astype(np.int32)
        acc = {}
        for t0_ in np.linspace(0, tr.shape[0] - 16, 4).astype(int):
            ids = tr[t0_:t0_ + 16, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x0 = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((16, R)), axis=-1).astype(mx.bfloat16)
            mx.eval(x0, sc)
            arms = {"3c-N bf16": lambda x: C3.tier_glu_n(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc),
                    "3c-N p11": lambda x: C3.tier_glu_n(x, msb_p, lsb_p, ms_, ls_, plan=PLAN, scores=sc),
                    "NAX6-N bf16": lambda x: NN.tier_glu_nax6n(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).sum(1),
                    "NAX6-N p11": lambda x: NN.tier_glu_nax6n(x, msb_p, lsb_p, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).sum(1)}
            def chain(f):
                out = None
                for _ in range(16):
                    out = f(x0 if out is None else mx.depends(x0, out))
                return out
            for f in arms.values():
                mx.eval(chain(f))
            ts = {k: [] for k in arms}
            for _ in range(7):
                for k, f in arms.items():
                    mx.synchronize(); t1 = time.perf_counter(); mx.eval(chain(f)); ts[k].append((time.perf_counter() - t1) / 16)
            for k, v in ts.items():
                acc.setdefault(k, []).append(float(np.median(v)))
        m = {k: 1e3 * float(np.mean(v)) for k, v in acc.items()}
        say(f"  speed layer 20, S 16 (rough): 3c-N {m['3c-N bf16']:.3f} -> p11 {m['3c-N p11']:.3f} ms "
            f"({100 * (m['3c-N p11'] / m['3c-N bf16'] - 1):+.1f} %); NAX6-N {m['NAX6-N bf16']:.3f} -> p11 {m['NAX6-N p11']:.3f} ms "
            f"({100 * (m['NAX6-N p11'] / m['NAX6-N bf16'] - 1):+.1f} %)")
    del msb, lsb, msb_p, lsb_p
    mx.clear_cache()
st = ref2.self_test()
st_ok = all(bool(v) for v in st.values()) if isinstance(st, dict) else bool(st)
say(f"ref2.self_test: {'PASS' if st_ok else 'FAIL'}")
say(f"P11 check: {'PASS' if ok_all and st_ok else 'FAIL'}")
