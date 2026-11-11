"""MSB2-JUDGE (MoonShots, Thu 2026-11-05; the orchestrator 07:1x: re-derive NewIdeas' 'data-aware codes need a KL screen'):
expert-output errors of MSB-3, MSB-2 and the screen-7 TRL store against the 4-bit model, on the decision map's own answer
rows at screen 7's layers. Every weight comes from a source the Thursday records3 conversion does not touch:
  W4    = the original 4-bit MLX model (~/BigMoE/models/bare/*.safetensors, switch_mlp, group 64): w = s q + b;
  MSB-3 = records3's rule (~/BigMoE/DESIGN_3bit.md): 2s (q >> 1) + bf16(b + s mean(q & 1)) per group of 64;
  MSB-2 = the map's s7_cal rule (~/MoonShots/scripts/to2bit.py, bias 'mid'): s2 = 2 s3 = 4s, codes (q >> 1) >> 1,
          b2 = bf16(b3 + 0.5 s3);
  TRL   = Fit12's 8-bit view of the store (~/Fit12/scripts/fit_trl_judge.view, copied verbatim: no records3 read at all).
Rows: Breakthrough's xcapture (~/FableGod/decision_map/out/xcap, 36 windows: the MoE input x per layer, rank-sorted ids,
normalised router weights). Per (row, rank) pair: Fit12's whole-expert error ||y(W) - y(W4)||^2 with all three
projections replaced, and the same x w^2 (router-weighted: the expert's share of the MoE output error); pooled sums per
layer, domain and rank. Check: TRL / MSB-3 must reproduce Fit12's judge_map.json (built from records3).
CPU (numpy), <= 4 threads, ~2 GB; reads ~5 GB of expert weights.
usage: [BigMoE venv] OMP_NUM_THREADS=4 python msb2_judge.py [layers, default 7,18,30,41] -> data/msb2_judge.json + stdout"""
import glob, json, os, struct, sys, time
import numpy as np

H = os.path.expanduser("~")
MD = f"{H}/BigMoE/models/bare"
XCAP = f"{H}/FableGod/decision_map/out/xcap"
OUT = f"{H}/MoonShots/data/msb2_judge.json"
DOMS = ("code", "ifeval", "multi")
PROJ = ("gate", "up", "down")
STORE = f"{H}/Fit12/data/trl"
KS = {"gate": 2, "up": 2, "down": 3}


class J:
    """Fit12's fit_trl_judge.view / silu, copied verbatim (importing that module would read records3's index through
    fit_q at import; this script must not touch records3 during Thursday's in-place conversion)"""

    @staticmethod
    def silu(z):
        return z / (1.0 + np.exp(-z))

    @staticmethod
    def view(L, e, p):
        """the store's 8-bit view -> fp32 [o, i], exactly s * q + b (MLX packs element j of a uint32 in bits 8j..8j+7)."""
        f = os.path.join(STORE, f"L{L}", f"e{e:03d}_{p}_k{KS[p]}.safetensors")
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
            buf = fh.read()
        t = {}
        for k in ("w", "s", "b"):
            a, b = h[k]["data_offsets"]
            t[k] = np.frombuffer(buf[a:b], np.uint32 if k == "w" else np.uint16).reshape(h[k]["shape"])
        q = t["w"].view(np.uint8).astype(np.float32)                 # [o, i]
        s = (t["s"].astype(np.uint32) << 16).view(np.float32)
        b = (t["b"].astype(np.uint32) << 16).view(np.float32)
        return np.repeat(s, 64, 1) * q + np.repeat(b, 64, 1)


_WM = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
_HDR = {}


def tensor(name):
    """a read-only memmap of one safetensors tensor"""
    f = os.path.join(MD, _WM[name])
    if f not in _HDR:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _HDR[f] = (8 + n, json.loads(fh.read(n)))
    base, h = _HDR[f]
    t = h[name]
    dt = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}[t["dtype"]]
    a, b = t["data_offsets"]
    return np.memmap(f, dtype=dt, mode="r", offset=base + a, shape=tuple(t["shape"]))


def bf16f(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def bf16r(x):
    """round fp32 to bf16 (nearest even), returned as fp32"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


_SH4 = np.arange(0, 32, 4, dtype=np.uint32)


def views(L, e, T):
    """W4 and the three error matrices (MSB-3, MSB-2, TRL minus W4) per projection, fp32 [out, in]"""
    W4, D = {}, {"m3": {}, "m2": {}, "trl": {}}
    for p in PROJ:
        w, s, b = T[p]
        wq = np.asarray(w[e])                                         # [out, in / 8] uint32
        q = ((wq[..., None] >> _SH4) & 15).reshape(wq.shape[0], -1).astype(np.float32)   # [out, in]
        sf, bf = bf16f(s[e]), bf16f(b[e])                             # [out, in / 64]
        rep = lambda a: np.repeat(a, 64, axis=1)
        w4 = rep(sf) * q + rep(bf)
        m = np.floor(q / 2)
        lb = q - 2 * m
        b3 = bf16r(bf + sf * lb.reshape(lb.shape[0], -1, 64).mean(-1))
        s3 = 2 * sf
        w3 = rep(s3) * m + rep(b3)
        b2 = bf16r(b3 + 0.5 * s3)
        w2 = rep(2 * s3) * np.floor(m / 2) + rep(b2)
        W4[p] = w4
        D["m3"][p] = w3 - w4
        D["m2"][p] = w2 - w4
        D["trl"][p] = J.view(L, e, p).astype(np.float32) - w4
    return W4, D


def main():
    layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "7,18,30,41").split(",")]
    files = sorted(glob.glob(os.path.join(XCAP, "*.npz")))
    assert len(files) == 36, len(files)
    t0 = time.time()
    X, I, Wt, D = {L: [] for L in layers}, {L: [] for L in layers}, {L: [] for L in layers}, []
    for f in files:
        z = np.load(f)
        d = DOMS.index(os.path.basename(f).split("_")[1])
        for L in layers:
            X[L].append(z["x"][:, L, :].astype(np.float32))
            I[L].append(z["ids"][:, L, :].astype(np.int64))
            Wt[L].append(z["w"][:, L, :].astype(np.float32))
        D.append(np.full(z["x"].shape[0], d, np.int64))
    D = np.concatenate(D)
    print(f"[msb2-judge] 36 windows, {D.shape[0]} answer rows, read in {time.time() - t0:.0f} s", flush=True)
    res = {"layers": {}, "rows": {dn: int((D == i).sum()) for i, dn in enumerate(DOMS)},
           "views": ["m3", "m2", "trl"], "note": "sums [view][domain][rank]: e = whole-expert ||dy||^2, ew = e x w^2"}
    for L in layers:
        t1 = time.time()
        pre = f"language_model.model.layers.{L}.mlp.switch_mlp."
        T = {p: (tensor(pre + f"{p}_proj.weight"), tensor(pre + f"{p}_proj.scales"), tensor(pre + f"{p}_proj.biases"))
             for p in PROJ}
        x_all, ids, wt = np.concatenate(X[L]), np.concatenate(I[L]), np.concatenate(Wt[L])
        flat = ids.ravel()
        order = np.argsort(flat, kind="stable")
        rows, ranks = order // 10, order % 10
        bounds = np.searchsorted(flat[order], np.arange(513))
        acc = np.zeros((3, 2, len(DOMS), 10))                          # [view, (e, e w^2), domain, rank]
        cnt = np.zeros((len(DOMS), 10), np.int64)
        for e in range(512):
            r, k = rows[bounds[e]:bounds[e + 1]], ranks[bounds[e]:bounds[e + 1]]
            if len(r) == 0:
                continue
            W4, Dv = views(L, e, T)
            x, d, w = x_all[r], D[r], wt[r, k]
            g4, u4 = x @ W4["gate"].T, x @ W4["up"].T
            y4 = (J.silu(g4) * u4) @ W4["down"].T
            for vi, v in enumerate(("m3", "m2", "trl")):
                dW = Dv[v]
                dy = (J.silu(g4 + x @ dW["gate"].T) * (u4 + x @ dW["up"].T)) @ (W4["down"] + dW["down"]).T - y4
                ey = (dy * dy).sum(1)
                np.add.at(acc[vi, 0], (d, k), ey)
                np.add.at(acc[vi, 1], (d, k), ey * w * w)
            np.add.at(cnt, (d, k), 1)
        res["layers"][str(L)] = {"sums": acc.tolist(), "pairs": cnt.tolist()}
        g = lambda vi, wi, di, rk: acc[vi, wi, di][rk].sum()
        msg = []
        for di, dn in enumerate(DOMS):
            for gn, rk in (("r0-1", [0, 1]), ("r2-9", list(range(2, 10)))):
                msg.append(f"{dn} {gn}: m2/m3 {g(1, 0, di, rk) / g(0, 0, di, rk):.3f} (w {g(1, 1, di, rk) / g(0, 1, di, rk):.3f}), "
                           f"trl/m3 {g(2, 0, di, rk) / g(0, 0, di, rk):.3f} (w {g(2, 1, di, rk) / g(0, 1, di, rk):.3f})")
        print(f"[msb2-judge] L{L} ({time.time() - t1:.0f} s): " + "; ".join(msg), flush=True)
    json.dump(res, open(OUT, "w"))
    print(f"[msb2-judge] wrote {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
