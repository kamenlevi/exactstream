"""s2_bench.py (Leap, Tue 17 Nov 2026 08:4x; BREAK98b, ZERO's screen S2 in ~/FableGod/zero/IDEAS_1117b_cheapest_row.md):
the FAST-NAX expert one-call (ref2._ooo_sel -> ooo1cq, as exp32_bench.py: layer 20's converted pools, all 512 resident, real
1c-tiered code routing) at 32 rows vs 64 rows where rows 32-63 are OFF-SPINE rows routed INSIDE the spine's loaded union.
  - Spine = a real 32-row block (exp32's construction: kept uses, 1c tiers, nolsb, the km convention for drops).
  - Off-spine = the next 32 real rows of the same window. A use whose expert is in the spine's kept union keeps its id and
    weight at MSB-3 (nolsb: no LSB bytes); any other use is dropped onto the row's first in-union id at weight 0 (the km
    convention), so NO expert enters the call that the spine did not load: zero new bytes, more pairs per segment.
  - Reported: us per call at 32 and 64 rows (interleaved runs, median), the ratio, the in-union uses per off-spine row
    (ZERO assumes ~9 of 10 for real tree rows; later chain rows of the same text are a harder case), segments, the
    largest segment, and 16-row tiles by expert id (the grouping keys on id + tier + sidecar state: an approximation).
  - ZERO's lines: PASS <= 1.10x, KILL > 1.4x; the orchestrator's gate for S1: <= ~1.2x at 64 rows.
  - 128 rows is NOT run: ooo1cq.gpu_grp_1cq groups pairs in ONE threadgroup (P <= 1024); 128 x 10 = 1280 needs it widened.
usage: [BigMoE venv] s2_bench.py   (MEM ~3 GB; a share lock; ratio robust, absolute us rough)"""
import json
import os
import sys
import time

os.environ["BIGMOE_MAXK"] = "320"
if "prof" in sys.argv[1:]:                               # libmtlprof before mlx: kernels per call
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_third_party/mtlprof"))
    import mtlprof as MP
    MP.load(2)
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts"))
import mlx.core as mx
import ref2
ref2.apply()
import bigmoe.gated as BG
import bigmoe.tier as T
import ooo1c as OC
import ooo1cq as OQ

OUT = os.path.expanduser("~/Leap/data/week98/check25/s2_bench.txt")
L_ = []


def say(s):
    print(s, flush=True)
    L_.append(s)


ext = BG.ext
PLAN, K, R4, NL, D, LM = "4433333333", 10, 2, 96, 2560, 20
T.PLAN = PLAN
BARE = os.path.expanduser("~/BigMoE/models/bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
d_, ix = T.load_index(BARE, layout_ok=True)
r = ix["layers"][str(LM)]
base, _ = T.make_pools(ext, 0, 512, 512, SH4, d_, r)
Ls, side, _ = T.make_sidecar(ext, 512, 512, SH4, d_, r)
allids = np.arange(512, dtype=np.int32)
for j in range(0, 512, 64):
    ext.ensure(0, mx.array(allids[j:j + 64])); ext.ensure(Ls, mx.array(allids[j:j + 64]))
sm_of = np.array(ext.lookup(0, allids.tolist()), np.int32)
sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
assert (sm_of >= 0).all() and (sl_of >= 0).all()
RS = 32 * (2 + BG.XLINES)
MSB_MB = sum(r["msb_piece_size"]) / 1e6                  # the records3 pieces the one-call reads per expert
LSB_MB = sum(r["lsb_piece_size"]) / 1e6


class Gate:
    def __init__(self):
        self.nl, self.nreg, self.seq = NL, 128, 9
        self.region = mx.zeros((self.nreg * BG.PAIRS * RS,), mx.int32)
        self.status = mx.zeros((4096,), mx.int32)
        mx.eval(self.region, self.status)
        self.REG = np.frombuffer(memoryview(self.region), dtype=np.int32)
        self.REG.setflags(write=True)


GATE = Gate()


CUR = [None]


def post(Lx, ids_, after=None):
    """sync-free: the block's slots were precomputed on the host (the loader's posts are async in the engine; a
    np.array(ids_) here would sync the GPU on every call, as the first q3 run did: ~2.6x the floor, a harness artifact)"""
    pb, ps, nb, ns = CUR[0]
    return (pb, nb) if Lx == 0 else (ps, ns)


def call(blk):
    x, ids, sc, nl, pb, ps, nb, ns = blk
    CUR[0] = (pb, ps, nb, ns)
    GATE.seq += 1
    t = {"base": base, "sideP": side, "segs": [(0, 2, 4), (2, 10, 3)], "side": NL // 2 + LM, "layout": 2,
         "nolsb": (nl, None)}
    return ref2._ooo_sel(x, ids, {"gate": GATE, "L": 0}, t, post, sc, None)



MAP = os.path.expanduser("~/FableGod/decision_map/out")
basej = json.load(open(os.path.join(MAP, "baseline.json")))
code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
MSP, MT = 32, 64
VAR = os.environ.get("S2_VAR", "chain")   # chain | fill | sib
assert VAR in ("chain", "fill", "sib"), VAR


def mk(z, tz, a, b0, m_tot):
    rows = slice(b0, b0 + m_tot)
    ids = z["ids"][rows, LM].astype(np.int64)
    w = z["w"][rows, LM].astype(np.float32)
    k4 = tz["keep"][a + b0:a + b0 + m_tot, LM].astype(bool)
    k3 = tz["keep3"][a + b0:a + b0 + m_tot, LM].astype(bool)
    kept = k4 | k3
    idk = np.where(kept, ids, ids[:, :1])
    wk = np.where(kept, w, 0.0)
    nl = ~k4[:, :R4]
    sk = kept[:MSP]
    U = np.unique(idk[:MSP][sk]) if sk.any() else np.unique(idk[:MSP, :1])
    st = {}
    if m_tot > MSP and VAR == "sib":                      # siblings: the spine's own routing, tiers and weights (zero new bytes)
        idk[MSP:], wk[MSP:], nl[MSP:] = idk[:MSP], wk[:MSP], nl[:MSP]
        st["in_union"] = 10.0
    elif m_tot > MSP:
        oi, ow = ids[MSP:], w[MSP:]
        inu = np.isin(oi, U)
        st["in_union"] = float(inu.sum(1).mean())
        first = np.where(inu.any(1), oi[np.arange(len(oi)), inu.argmax(1)], U[0])
        idk[MSP:] = np.where(inu, oi, first[:, None])
        wk[MSP:] = np.where(inu, ow, 0.0)
        nl[MSP:] = True
        if VAR == "fill":                                  # weight-0 pairs into the free padding rows of tier-3 segments
            cnt = {}
            for r_ in range(m_tot):
                for j_ in range(idk.shape[1]):
                    if r_ >= MSP and not inu[r_ - MSP, j_]:
                        continue
                    k_ = (int(idk[r_, j_]), 4 if (j_ < R4 and not nl[r_, j_]) else 3)
                    cnt[k_] = cnt.get(k_, 0) + 1
            for r_, j_ in zip(*np.nonzero(~inu)):
                fbest, ebest = max(((-cnt.get((int(e_), 3), 0)) % 16, int(e_)) for e_ in U)
                if fbest == 0:                              # no tier-3 tile has room: the expert with the largest count
                    ebest = int(max(U, key=lambda e_: cnt.get((int(e_), 3), 0)))
                idk[MSP + r_, j_] = ebest
                cnt[(ebest, 3)] = cnt.get((ebest, 3), 0) + 1
        assert np.isin(np.unique(idk[MSP:]), U).all()        # zero new experts: off-spine ids inside the spine's union
    m = m_tot
    nlm, idm = nl[:m], idk[:m]
    fl = np.flatnonzero(~nlm.reshape(-1))
    dpk = idm[:, :R4].reshape(-1)[fl[0]] if len(fl) else idm[0, 0]
    ids_s = np.where(nlm, dpk, idm[:, :R4])
    pb = mx.array(np.concatenate([[0], sm_of[idm.reshape(-1)]]).astype(np.int32))
    ps = mx.array(np.concatenate([[0], sl_of[ids_s.reshape(-1)]]).astype(np.int32))
    blk = (mx.array(z["x"][rows, LM][:m].astype(np.float32)).astype(mx.bfloat16), mx.array(idm.astype(np.int32)),
           mx.array(wk[:m]).astype(mx.bfloat16), mx.array(nlm), pb, ps, int(idm.size), int(ids_s.size))
    u_, c_ = np.unique(idm.reshape(-1), return_counts=True)
    tk = {}
    for r_ in range(m):
        for j_ in range(idm.shape[1]):
            k_ = (int(idm[r_, j_]), 4 if (j_ < R4 and not nlm[r_, j_]) else 3)
            tk[k_] = tk.get(k_, 0) + 1
    st.update(segs=len(u_), max_seg=int(c_.max()), tiles16=int(np.ceil(c_ / 16).sum()), union=len(U),
              tiles16t=int(sum(-(-v // 16) for v in tk.values())))
    return blk, st


B32, B64, S32, S64 = [], [], [], []
for f in code:
    z = np.load(os.path.join(MAP, "xcap", f))
    tz = np.load(os.path.join(MAP, "bt", "tiers_1c", f))
    a = int(z["a"])
    n = z["ids"].shape[0]
    for b0 in range(0, n - MT + 1, MT):
        b_, s_ = mk(z, tz, a, b0, MSP); B32.append(b_); S32.append(s_)
        b_, s_ = mk(z, tz, a, b0, MT); B64.append(b_); S64.append(s_)
mx.eval([t_ for blk in B32 + B64 for t_ in blk if isinstance(t_, mx.array)])
NB = min(24, len(B32))
ref2.CFG.update(ooo_1c=True, tier_nax_ooo=True, nax_1cq=True, nax_native=True)
mean = lambda S, k: float(np.mean([s[k] for s in S[:NB]]))
say(f"\nS2 BENCH [{VAR}] {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; layer {LM}; {len(code)} code windows -> {len(B32)} "
    f"64-row chunks (timing {NB}); spine union {mean(S32, 'union'):.1f} experts; off-spine uses inside it {mean(S64, 'in_union'):.2f}"
    f" of 10 per row; segments 32 / 64 rows {mean(S32, 'segs'):.1f} / {mean(S64, 'segs'):.1f}; largest segment "
    f"{mean(S32, 'max_seg'):.1f} / {mean(S64, 'max_seg'):.1f} pairs; 16-row tiles by id {mean(S32, 'tiles16'):.1f} / {mean(S64, 'tiles16'):.1f}; by (id, tier) {mean(S32, 'tiles16t'):.1f} / {mean(S64, 'tiles16t'):.1f}")


def run(blocks, rounds=7):
    ts = []
    for r_ in range(rounds):
        mx.synchronize()
        t0 = time.perf_counter()
        ys = [call(b) for b in blocks]
        mx.eval(ys)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(ts[1:])) / len(blocks)


y32 = call(B32[0]); y64 = call(B64[0]); mx.eval(y32, y64)
same = bool(mx.all(y64[:MSP] == y32).item()) if y64.shape[0] == 2 * y32.shape[0] or y64.shape[0] >= MSP else None
say(f"  spine rows of the 64-row call == the 32-row call (values): {same} (shapes {tuple(y32.shape)} / {tuple(y64.shape)})")
res = []
for coh in (False, True):
    ref2.CFG["nax_coh"] = coh
    t32, t64 = [], []
    for rep in range(3):                                   # interleaved: 32, 64, 32, 64, ...
        t32.append(run(B32[:NB])); t64.append(run(B64[:NB]))
    u32, u64 = float(np.median(t32)), float(np.median(t64))
    say(f"  coh {int(coh)}: 32 rows {u32:7.1f} us/call, 64 rows {u64:7.1f} us/call -> {u64 / u32:.3f}x "
        f"(runs 32: {', '.join(f'{v:.0f}' for v in t32)}; 64: {', '.join(f'{v:.0f}' for v in t64)})")
    res.append(dict(coh=coh, us32=u32, us64=u64, ratio=u64 / u32, t32=t32, t64=t64))
r = max(x["ratio"] for x in res)
say(f"  [{VAR}] S2 at 64 rows (the worse COH setting): {r:.3f}x -> {'PASS (<= 1.10x)' if r <= 1.10 else 'KILL (> 1.4x)' if r > 1.4 else 'BETWEEN'}; "
    f"the orchestrator's S1 gate (<= ~1.2x): {'OPEN' if r <= 1.2 else 'CLOSED'}")
say(f"done; peak {mx.get_peak_memory() / 1e9:.1f} GB")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "a").write("\n".join(L_) + "\n")
json.dump(dict(var=VAR, res=res, s32=S32[:NB], s64=S64[:NB]), open(OUT.replace(".txt", f"_{VAR}.json"), "w"), indent=1)
