"""xfront_check.py (Breakthrough, Tue 17 Nov 2026 21:3x; MOE BLOCK front fold, STACK FORM; gate G1 of docs/MOEBLOCK.md
"FRONT FOLD, STACK FORM (mode tgl16ef)", pre-registered 1b71659): xfront.gpu_front == ooo1cq.gpu_grp_1cq + rowsum64,
bitwise, for each grouping variant: pidx (all P), nseg, seg[0..nseg] (the kernels read no further) and xs (fp32 bits).
  (a) 2,000 random synthetic calls: M 1..102 (P <= 1020), plans 4433333333 (the stack's) and 3333333333 (no sidecar),
      base slots with -1 (pending) / -2 (QOM: per pair and per expert), sidecars with -1 / -2, expert pools from 10 to
      512 (key runs > 16 and > 32), some rows with repeated ids, x bf16 at scales 1e-3 .. 1e3, hidden 2560 / 4096 / 64.
  (b) real routing ids + x from the decision map's code captures (the stack's form: raw top-10 ids, no tier marks),
      layers 20 and 44, 12 blocks x 32 rows and their first 16 rows; slots = a fixed random permutation (grouping reads
      only the slot numbers; G2 runs the real pools).
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/xfront_check.py [vars, default v,r,s] [n_synth]   (share lock, < 1 GB)
Writes data/xrw/xfront_check.txt."""
import json
import os
import sys

os.environ.setdefault("BIGMOE_MAXK", "320")
for _p in ("~/BigMoE", "~/Leap/third_party/ms_gdn", "~/Leap/scripts", "~/Leap/scripts/week98"):
    _p = os.path.expanduser(_p)
    if _p not in sys.path:
        sys.path.append(_p)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402
import ooo1cq as OQ                                         # noqa: E402
from tierglu_nax import rowsum64                            # noqa: E402
import xfront as XF                                         # noqa: E402

MAP = os.path.expanduser("~/FableGod/decision_map/out")
OUT = os.path.join(HERE, "..", "..", "data", "xrw", "xfront_check.txt")
L_ = []


def say(s=""):
    print(s, flush=True)
    L_.append(s)


def real_blocks(Lm, nb=12):
    """the decision map's code captures, 32-row blocks: (name, b0, x [32, 2560] fp32, ids [32, 10]); no tier marks"""
    basej = json.load(open(os.path.join(MAP, "baseline.json")))
    out = []
    for f in sorted(os.listdir(os.path.join(MAP, "xcap"))):
        if basej[f]["dom"] != "code":
            continue
        z = np.load(os.path.join(MAP, "xcap", f))
        n = z["ids"].shape[0]
        for b0 in range(0, n - 31, 32):
            out.append((f, b0, z["x"][b0:b0 + 32, Lm].astype(np.float32), z["ids"][b0:b0 + 32, Lm].astype(np.int32)))
            if len(out) >= nb:
                return out
    return out


def same(ref, got):
    """(pidx, seg, nseg, P), xs  vs  the same from the fold -> dict of bools"""
    (p0, s0, n0, P0), x0 = ref
    (p1, s1, n1, P1), x1 = got
    mx.eval(p0, s0, n0, x0, p1, s1, n1, x1)
    a, b = int(np.array(n0)[0]), int(np.array(n1)[0])
    return dict(P=P0 == P1, pidx=bool(np.array_equal(np.array(p0), np.array(p1))), nseg=a == b,
                seg=a == b and bool(np.array_equal(np.array(s0)[:a + 1], np.array(s1)[:b + 1])),
                xs=x0.shape == x1.shape and bool(np.array_equal(np.array(x0).view(np.uint32), np.array(x1).view(np.uint32))))


def synth(rng):
    M = int(rng.integers(1, 103))
    plan = "4433333333" if rng.random() < 0.9 else "3333333333"
    r4 = 2 if plan[0] == "4" else 0
    L = int(rng.choice([2560] * 8 + [4096, 64]))
    npool = int(rng.choice([10, 11, 12, 14, 20, 40, 512]))
    pool = rng.choice(512, size=npool, replace=False)
    rep = rng.random() < 0.1
    ids = np.stack([rng.choice(pool, 10, replace=rep) for _ in range(M)]).astype(np.int32)
    slot_of = rng.permutation(1024)[:512].astype(np.int32)
    side_of = rng.permutation(1024)[:512].astype(np.int32)
    pm = slot_of[ids]
    p_late, p_q = rng.choice([0.0, 0.05, 0.3]), rng.choice([0.0, 0.05, 0.3])
    pm = np.where(rng.random(pm.shape) < p_late, -1, pm)
    pm = np.where(rng.random(pm.shape) < p_q, -2, pm)
    if rng.random() < 0.3:                                  # a whole expert dropped (QOM is per expert)
        pm = np.where(ids == pool[int(rng.integers(npool))], -2, pm)
    if r4:
        pl = side_of[ids[:, :r4]]
        pl = np.where(rng.random(pl.shape) < rng.choice([0.0, 0.1, 0.4]), -1, pl)
        pl = np.where(rng.random(pl.shape) < rng.choice([0.0, 0.1, 0.4]), -2, pl)
    else:
        pl = np.zeros((M, 1), np.int32)
    sc = float(rng.choice([1e-3, 0.5, 30.0, 1e3]))
    x = mx.array((rng.standard_normal((M, L)) * sc).astype(np.float32)).astype(mx.bfloat16)
    return M, plan, mx.array(pm.astype(np.int32)), mx.array(pl.astype(np.int32)), mx.array(ids), x


def main():
    vars_ = (sys.argv[1] if len(sys.argv) > 1 else "v,r,s").split(",")
    nsyn = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    say(f"XFRONT CHECK (G1): gpu_front vs ooo1cq.gpu_grp_1cq + rowsum64, bitwise; variants {vars_}; src_ok "
        f"{XF.src_ok()}")
    ok = {v: True for v in vars_}
    bad = {v: [] for v in vars_}
    rng = np.random.default_rng(1117)
    st = {"M": [], "runs16": 0}
    for c in range(nsyn):
        M, plan, pm, pl, ids, x = synth(rng)
        ref = (OQ.gpu_grp_1cq(pm, pl, ids, plan), rowsum64(x))
        for v in vars_:
            r = same(ref, XF.gpu_front(pm, pl, ids, plan, x, v))
            if not all(r.values()):
                ok[v] = False
                if len(bad[v]) < 5:
                    bad[v].append((c, M, plan, {k: b for k, b in r.items() if not b}))
        st["M"].append(M)
        seg = np.array(ref[0][1])[:int(np.array(ref[0][2])[0]) + 1]
        st["runs16"] += int((np.diff(seg) == 16).sum() > 0)
    say(f"  (a) {nsyn} synthetic calls (M {min(st['M'])}..{max(st['M'])}; {st['runs16']} with a full 16-pair "
        f"segment): " + "; ".join(f"{v} {'PASS' if ok[v] else 'FAIL ' + str(bad[v])}" for v in vars_))
    for Lm in (20, 44):
        bl = real_blocks(Lm, 12)
        rr = np.random.default_rng(Lm)
        slot_of = rr.permutation(512).astype(np.int32)
        side_of = rr.permutation(512).astype(np.int32)
        for M in (32, 16):
            okr = {v: True for v in vars_}
            for (f, b0, xf, idr) in bl:
                idm = idr[:M]
                pm, pl = mx.array(slot_of[idm]), mx.array(side_of[idm[:, :2]])
                ids = mx.array(idm)
                x = mx.array(xf[:M]).astype(mx.bfloat16)
                ref = (OQ.gpu_grp_1cq(pm, pl, ids, "4433333333"), rowsum64(x))
                for v in vars_:
                    r = same(ref, XF.gpu_front(pm, pl, ids, "4433333333", x, v))
                    if not all(r.values()):
                        okr[v] = ok[v] = False
                        if len(bad[v]) < 10:
                            bad[v].append((f, b0, M, {k: b for k, b in r.items() if not b}))
            say(f"  (b) real routing L{Lm}, {M} rows, {len(bl)} blocks: " +
                "; ".join(f"{v} {'PASS' if okr[v] else 'FAIL'}" for v in vars_))
    neg, nn = {v: 0 for v in vars_}, 0                      # negative control: the check must see a changed input
    for c in range(20):
        M, plan, pm, pl, ids, x = synth(rng)
        ref = (OQ.gpu_grp_1cq(pm, pl, ids, plan), rowsum64(x))
        pm2 = mx.concatenate([mx.array([[2047]], mx.int32) * mx.ones((1, 1), mx.int32),
                              pm.reshape(1, -1)[:, 1:]], axis=1).reshape(pm.shape)   # pair 0's key moves past every other
        x2 = x * 2
        nn += 1
        for v in vars_:
            r = same(ref, XF.gpu_front(pm2, pl, ids, plan, x2, v))
            neg[v] += int((not r["pidx"] or not r["seg"] or not r["nseg"]) and not r["xs"])
    say(f"  negative control ({nn} calls, pair 0's slot and x changed): detected " +
        "; ".join(f"{v} {neg[v]}/{nn}" for v in vars_))
    say("XFRONT CHECK (G1): " + "; ".join(f"{v} {'PASS' if ok[v] else 'FAIL ' + str(bad[v])}" for v in vars_))
    open(OUT, "w").write("\n".join(L_) + "\n")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
