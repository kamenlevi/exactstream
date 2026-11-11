"""COH check (Leap, Tue 2026-11-03 22:5x; TABLE_350 'COH', pre-registered 22:38): the expert kernels with coherent(system)
pool reads (tierglu_coh.py; forms cn / cv) vs the plain kernels, on layer 20's real pools (native bf16, built from
records3 as p11_check does) and real code routing (Breakthrough's code_800_g trace, 4 windows of 16 rows):
 (1) bits: NAX6-N's y and 3c-N's y bitwise == plain, every window, both forms;
 (2) speed: ms per call in dependent chains of 16 (p11_check's method), median of 7 per window, mean over windows.
Lines: PASS = (1) AND the better form <= +3 % on both kernels; KILL = not bitwise or > +15 % on either; else BETWEEN.
Usage: [BigMoE venv] coh_check.py -> ~/Leap/data/coh_check.txt (GPU ~3 GB; share lock)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
import ref2
ref2.apply()
import tierglu_3cn as C3
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import tierglu_coh as CO
import p11 as P11

OUT = os.path.expanduser("~/Leap/data/coh_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nCOH check {time.strftime('%a %H:%M')} ({os.environ.get('COH_COND', 'share lock')})")
PLAN = "4433333333"
R = len(PLAN)
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def native_pools(r):
    """p11_check's pools(), the native bf16 half only: records3 (row-major) -> native order via p11's transform"""
    E = r["E"]
    raw = np.memmap(f"{RD}/{r['file']}", dtype=np.uint8, mode="r", shape=(E, r["recsz"]))
    q = P11.layout_p11(r)
    new = np.empty((E, q["recsz"]), np.uint8)
    for e0 in range(0, E, 64):
        new[e0:e0 + 64] = P11.to_p11(np.asarray(raw[e0:e0 + 64]), r)[0]
    del raw
    nat = {"msb": {}, "lsb": {}}
    for nm, part, pr, off, sz, dt, per, packed in P11.pieces(q):
        o, i = SH[pr]
        n = o * (i // 64) * per
        if packed:
            w = np.ascontiguousarray(new[:, off:off + sz]).view(np.uint32)
            nat[part].setdefault(pr, {})[nm] = mx.array(P11.unpack(w, n).reshape(E, o, i // 64)).view(mx.bfloat16)
        else:
            a = np.ascontiguousarray(new[:, off:off + n * np.dtype(dt).itemsize]).view(dt)
            nat[part].setdefault(pr, {})[nm] = mx.array(a.reshape((E, o, i * 3 // 32) if nm == "w3" else (E, o, i // 32)))
    del new
    mx.eval([v for part in nat.values() for pr in part.values() for v in pr.values()])
    return nat["msb"], nat["lsb"]


idx = json.load(open(f"{RD}/index.json"))["layers"]
t0 = time.time()
msb, lsb = native_pools(idx["20"])
say(f"layer 20 native pools built in {time.time() - t0:.0f} s")
tr = np.load(os.path.expanduser("~/Breakthrough/data/expA/traces125/code_800_g.npz"))["L20"].astype(np.int32)
ok_bits = True
acc = {}
DROPPED = {}
for t0_ in np.linspace(0, tr.shape[0] - 16, 4).astype(int):
    ids = tr[t0_:t0_ + 16, :10].copy()
    ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
    mx.random.seed(int(t0_))
    x0 = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
    sc = mx.softmax(mx.random.normal((16, R)), axis=-1).astype(mx.bfloat16)
    mx.eval(x0, sc)
    grp = N6.gpu_grp(ms_, ls_, PLAN)
    arms = {"NAX6-N plain": lambda x: NN.tier_glu_nax6n(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp),
            "NAX6-N cn": lambda x: CO.tier_glu_nax6n_coh(x, msb, lsb, ms_, ls_, form="cn", plan=PLAN, grp=grp),
            "NAX6-N cv": lambda x: CO.tier_glu_nax6n_coh(x, msb, lsb, ms_, ls_, form="cv", plan=PLAN, grp=grp),
            "3c-N plain": lambda x: C3.tier_glu_n(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc),
            "3c-N cn": lambda x: CO.tier_glu_n_coh(x, msb, lsb, ms_, ls_, form="cn", plan=PLAN, scores=sc),
            "3c-N cv": lambda x: CO.tier_glu_n_coh(x, msb, lsb, ms_, ls_, form="cv", plan=PLAN, scores=sc)}
    ys = {}
    for k in list(arms):                                   # a form that does not build is dropped and reported
        try:
            ys[k] = arms[k](x0); mx.eval(ys[k])
        except Exception as e:
            if k not in DROPPED:
                DROPPED[k] = str(e).split("error:")[1].split("\n")[0].strip() if "error:" in str(e) else str(e)[:120]
                say(f"  {k}: does not build ({DROPPED[k]}); dropped")
            del arms[k]
    for k in arms:
        if not k.endswith("plain"):
            same = eqb(ys[k], ys[k.split()[0] + " plain"])
            ok_bits &= same
            if not same:
                say(f"  window {t0_}: {k} NOT bitwise == plain")
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
cost = {k: 100 * (m[k] / m[k.split()[0] + " plain"] - 1) for k in m if not k.endswith("plain")}
for k in DROPPED:
    cost[k] = float("inf")
say("(1) bits: " + ("every window, both kernels, both forms bitwise == plain" if ok_bits else "NOT bitwise (see above)"))
say("(2) ms per call (S 16, layer 20, 4 windows): " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
say("    cost vs plain: " + ", ".join(f"{k} {v:+.1f} %" for k, v in cost.items()))
best = {kern: min(cost[f"{kern} cn"], cost[f"{kern} cv"]) for kern in ("NAX6-N", "3c-N")}
v = "KILL" if not ok_bits or max(best.values()) > 15 else "PASS" if max(best.values()) <= 3 else "BETWEEN"
say(f"COH verdict: {v} (best form's cost: NAX6-N {best['NAX6-N']:+.1f} %, 3c-N {best['3c-N']:+.1f} %)")
st = ref2.self_test()
say(f"ref2.self_test: {'PASS' if all(st.values()) else 'FAIL'}")
