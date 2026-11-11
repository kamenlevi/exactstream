"""P11PAD check (Leap, Tue 2026-11-03 22:5x; TABLE_350 'P11PAD'; native+p11+pad agreed with BigMoE 22:4x): runs with the
CANDIDATE modules first on sys.path (P11PAD_DIR: p11.py, tierglu_3cn.py, tierglu_nax6n.py with the pools' strides read
from their shapes), before they replace the live ones.
(A) layout, layers 20 and 44: layout_p11(r, pad=True) keeps recsz / msb_sz / lsb_sz; every MSB piece and LSB l1 plane is
    4 KB-aligned; to_p11(pad=True) -> from_p11 == the row-major records bitwise; each packed piece == the unpadded one +
    zero padding.
(B) kernels, layers 20 and 44 (the first P11PAD_EXPERTS = 128 experts: ~2 GB, by day), p11_check's cover routing
    (every one of them at a 4-bit and at 3-bit ranks): 3c-N tier_glu_n
    (+ scores), phases, pairs, split, and NAX6-N on PADDED p11 pools (s3 / b3 [C, 9216], b [C, 8804]) == UNPADDED p11
    pools == bf16 native pools, bitwise.
(C) ref2.self_test with the candidate modules loaded.
Usage: [BigMoE venv] P11PAD_DIR=<dir> p11pad_check.py -> ~/Leap/data/p11pad_check.txt (GPU ~5 GB; share lock)"""
import json, os, sys, time
import numpy as np
CAND = os.environ["P11PAD_DIR"]
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
import importlib.util
import mlx.core as mx
def _load(name):                                            # the candidates, registered before anything imports the live ones
    spec = importlib.util.spec_from_file_location(name, os.path.join(CAND, name + ".py"))  # (live modules put /Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts first)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m
for _n in ("p11", "tierglu_3cn", "tierglu_nax6n"):
    _load(_n)
import ref2
ref2.apply()
import bigmoe.tier as BT
TG = BT._tier_glu_mod()
import p11 as P11
import tierglu_3cn as C3
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import tierglu_nax6h as NH                                  # NAX6-NH (Wed 11-04: its p11 path), the live module
for m in (P11, C3, NN):
    assert os.path.dirname(os.path.abspath(m.__file__)) == os.path.abspath(CAND), m.__file__

OUT = os.path.expanduser("~/Leap/data/p11pad_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nP11PAD check {time.strftime('%a %H:%M')} (candidates in {CAND})")
PLAN = "4433333333"
EP = int(os.environ.get("P11PAD_EXPERTS", "128"))
BT.PLAN = PLAN
R = len(PLAN)
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
idx = json.load(open(f"{RD}/index.json"))["layers"]
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def pools3(r):
    """bf16 native, unpadded p11 and padded p11 pools of a layer (the padded ones read from padded records)"""
    E = EP                                                   # the first EP experts: ~2 GB by day (one big job at a time)
    raw = np.memmap(f"{RD}/{r['file']}", dtype=np.uint8, mode="r", shape=(r["E"], r["recsz"]))[:EP]
    q0, q1 = P11.layout_p11(r), P11.layout_p11(r, pad=True)
    n0 = np.empty((E, q0["recsz"]), np.uint8)
    n1 = np.empty((E, q1["recsz"]), np.uint8)
    rt_ok = True
    for e0 in range(0, E, 64):
        chunk = np.asarray(raw[e0:e0 + 64])
        n0[e0:e0 + 64] = P11.to_p11(chunk, r)[0]
        n1[e0:e0 + 64] = P11.to_p11(chunk, r, pad=True)[0]
        rt_ok &= bool(np.array_equal(P11.from_p11(n1[e0:e0 + 64], q1, r), chunk))
    del raw
    lay_ok = (q1["recsz"] == q0["recsz"] and q1["msb_sz"] == q0["msb_sz"] and q1["lsb_sz"] == q0["lsb_sz"]
              and q1["recsz"] % 4096 == 0 and all(o % 4096 == 0 for o in q1["msb_piece_off"])
              and all(q1["lsb_piece_off"][2 * p] % 4096 == 0 for p in range(3)))
    piece_ok = True
    nat, p0, p1 = {"msb": {}, "lsb": {}}, {"msb": {}, "lsb": {}}, {"msb": {}, "lsb": {}}
    for (nm, part, pr, off0, sz0, dt, per, packed), (_, _, _, off1, sz1, _, _, _) in zip(P11.pieces(q0), P11.pieces(q1)):
        o, i = SH[pr]
        n = o * (i // 64) * per
        if packed:
            w0 = np.ascontiguousarray(n0[:, off0:off0 + sz0]).view(np.uint32)
            w1 = np.ascontiguousarray(n1[:, off1:off1 + sz1]).view(np.uint32)
            piece_ok &= bool(np.array_equal(w1[:, :w0.shape[1]], w0) and not w1[:, w0.shape[1]:].any())
            p0[part].setdefault(pr, {})[nm] = mx.array(w0)
            p1[part].setdefault(pr, {})[nm] = mx.array(w1)
            nat[part].setdefault(pr, {})[nm] = mx.array(P11.unpack(w0, n).reshape(E, o, i // 64)).view(mx.bfloat16)
        else:
            a = np.ascontiguousarray(n0[:, off0:off0 + n * np.dtype(dt).itemsize]).view(dt)
            b = np.ascontiguousarray(n1[:, off1:off1 + n * np.dtype(dt).itemsize]).view(dt)
            piece_ok &= bool(np.array_equal(a, b))
            arr = mx.array(a.reshape((E, o, i * 3 // 32) if nm == "w3" else (E, o, i // 32)))
            for d in (nat, p0, p1):
                d[part].setdefault(pr, {})[nm] = arr
    del n0, n1
    mx.eval([v for d in (nat, p0, p1) for part in d.values() for pr in part.values() for v in pr.values()])
    return nat, p0, p1, lay_ok, rt_ok, piece_ok, q1


def cover_routing(rng, E=512, calls=16, S=16):
    """p11_check's: call c's 4-bit ranks hold experts 32c .. 32c + 31; 3-bit ranks a shuffled cover of all experts"""
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


ok_all = True
mx.random.seed(11)
rng = np.random.default_rng(11)
for L in (20, 44):
    t0 = time.time()
    nat, p0, p1, lay_ok, rt_ok, piece_ok, q1 = pools3(idx[str(L)])
    say(f"layer {L}: (A) layout keeps sizes + 4 KB-aligned {lay_ok}, padded round trip == row-major {rt_ok}, "
        f"pieces == unpadded + zero padding {piece_ok} (built in {time.time() - t0:.0f} s; MSB offsets {q1['msb_piece_off']}, "
        f"LSB offsets {q1['lsb_piece_off']})")
    ok_all &= lay_ok and rt_ok and piece_ok
    cnt = {}
    for ids in cover_routing(rng, E=EP, calls=EP // 32):
        S = ids.shape[0]
        ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
        x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
        ph = mx.array(rng.integers(1, 3, (S, R)).astype(np.int32))
        pidx = mx.array(np.sort(rng.choice(S * R, S * R // 3, replace=False)).astype(np.int32))
        hit = rng.random((S, R)) < 0.6
        for nm, f in (("tier_glu_n", lambda d: C3.tier_glu_n(x, d["msb"], d["lsb"], ms_, ls_, plan=PLAN)),
                      ("tier_glu_n+scores", lambda d: C3.tier_glu_n(x, d["msb"], d["lsb"], ms_, ls_, plan=PLAN, scores=sc)),
                      ("phases", lambda d: TG.combine_phase(C3.tier_glu_phase_n(x, d["msb"], d["lsb"], ms_, ls_, ph, 1, plan=PLAN),
                                                           C3.tier_glu_phase_n(x, d["msb"], d["lsb"], ms_, ls_, ph, 2, plan=PLAN), ph, sc)),
                      ("pairs", lambda d: C3.tier_glu_pairs_n(x, d["msb"], d["lsb"], ms_, ls_, pidx, plan=PLAN)),
                      ("split", lambda d: C3.tier_glu_split_n(x, d["msb"], d["lsb"], ms_, ls_, hit, sc, plan=PLAN)),
                      ("NAX6-N", lambda d: NN.tier_glu_nax6n(x, d["msb"], d["lsb"], ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN))),
                      ("NAX6-NH", lambda d: NH.tier_glu_nax6nh(x, d["msb"], d["lsb"], ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)))):
            a, b, c = f(p1), f(p0), f(nat)
            mx.eval(a, b, c)
            k = cnt.setdefault(nm, [0, 0, 0])
            k[0] += eqb(a, b); k[1] += eqb(a, c); k[2] += 1
    ok = all(v[0] == v[1] == v[2] for v in cnt.values())
    ok_all &= ok
    say(f"layer {L}: (B) padded == unpadded p11 / == bf16, calls: " + ", ".join(f"{k} {v[0]}/{v[1]}/{v[2]}" for k, v in cnt.items())
        + f" -> {'PASS' if ok else 'FAIL'}")
    del nat, p0, p1
    mx.clear_cache()
st = ref2.self_test()
st_ok = all(bool(v) for v in st.values())
say(f"(C) ref2.self_test with the candidates: {'PASS' if st_ok else 'FAIL'}")
say(f"P11PAD check: {'PASS' if ok_all and st_ok else 'FAIL'}")
