"""Fit12 (Thu 5 Nov 18:5x; the orchestrator 18:5x): an independent check of Fable A's HALF-K numbers (~/FableGod/LOSSY350_A.md
section 1; Fable A's own replay ~/FableGod/lossy350a/arms.py on 40 experts): the whole-expert output error, in MSB-3 units,
of magnitude-gated half experts at 0.51 / 0.59 / 0.68 of the weights touched (x32 h32 / x40 h32 / x48 h32), held out.
Mechanism as Fable A states it: per (row, layer) c = V^T x for the layer's 32 dominant input directions, x_perp = x - V c;
per 64-wide quant group of gate / up keep the c_x members with the largest |x_perp| (one list per row, shared by its
experts; the response to V c is exact); per 64-wide group of down keep the c_h units with the largest |h|. Plain arms: the
same masks on |x|, no deflation. Weights touched = (c_x / 64) 2/3 + (c_h / 64) 1/3 (+ 0.83 % for the two [640 x 32]
responses when deflated). V here = the top-32 eigenvectors of the uncentered input second moment over B10-r's fit rows
R0-R7 at that layer (exact-duplicate rows removed), so every row judged below is held out from V.
Independent of Fable A's code and data:
  - my own loader: the 4-bit experts read with pread from ~/BigMoE/models/bare safetensors (read-only); MSB-3 rebuilt as
    records3 held it (m = q >> 1, s3 = 2 s, b3 = bf16(b + s mean(l)) per group: the identity fit_q.py checked on records3),
    validated by --validate, which reproduces Wed's records3-based judge ratios for L41 e0 / L7 e5;
  - ALL 512 experts per layer, layers 2 / 12 / 24 / 36 / 46;
  - two held-out row sets: B10-r's R8 + R9 (top-10 sets) and the map's own answer rows (Breakthrough's xcapture,
    /Users/kamen/FableGod/decision_map/out/xcap, read-only; rank-sorted, so ranks 0-1 and 2-9 are reported apart).
Metric: whole expert, sum ||y~ - y4||^2 over the held-out routed pairs / the same sum for the MSB-3 expert ("x MSB-3");
also MSB-3's own sum ||y3 - y4||^2 / sum ||y4||^2. numpy fp32 (no MLX matmuls, so no TF32). CPU only, 4 threads, one
process, < 2 GB; stops if swap grows > 1 GB (EvalSuite T0's guard). Writes /Users/kamen/Fit12/data/halfk/halfk.json.
Usage: [BigMoE venv] python -I -B fit_halfk_check.py [LAYERS 2,12,24,36,46] | --validate"""
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import glob, json, struct, subprocess, sys, time
import numpy as np

MD = "/Users/kamen/BigMoE/models/bare"
CAP = "/Users/kamen/Fit12/data/b10rcap"
XCAP = "/Users/kamen/FableGod/decision_map/out/xcap"
OUT = "/Users/kamen/Fit12/data/halfk"
WMAP = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
ARMS = [("defl", 64, 32), ("defl", 48, 32), ("defl", 40, 32), ("defl", 32, 32),
        ("plain", 48, 32), ("plain", 40, 32), ("plain", 32, 32)]
_H, _FD = {}, {}


def swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0])


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def rbf16(x):                                    # fp32 -> nearest-even bf16, back as fp32 (as fit_q.py)
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).view(np.float32)


def slab(name, e):
    f = os.path.join(MD, WMAP[name])
    if f not in _H:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _H[f] = (8 + n, json.loads(fh.read(n)))
        _FD[f] = os.open(f, os.O_RDONLY)
    base, h = _H[f]
    t = h[name]
    dt = {"U32": np.uint32, "BF16": np.uint16}[t["dtype"]]
    per = int(np.prod(t["shape"][1:])) * np.dtype(dt).itemsize
    assert t["data_offsets"][1] - t["data_offsets"][0] == per * t["shape"][0]
    return np.frombuffer(os.pread(_FD[f], per, base + t["data_offsets"][0] + e * per), dt).reshape(t["shape"][1:])


def expert(L, e):
    """-> W4 (fp32) and D3 = MSB-3 - W4 (formed in fp64) per projection, [out, in]."""
    W4, D3 = {}, {}
    for p in ("gate", "up", "down"):
        pre = f"language_model.model.layers.{L}.mlp.switch_mlp.{p}_proj"
        w = slab(pre + ".weight", e)
        s, b = bf16(slab(pre + ".scales", e)), bf16(slab(pre + ".biases", e))
        o, K = w.shape[0], w.shape[1] * 8
        q = ((w[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).astype(np.uint8).reshape(o, K // 64, 64)
        w4 = s[..., None].astype(np.float64) * q + b[..., None]
        b3 = rbf16(b + s * (q & 1).mean(-1, dtype=np.float32))
        w3 = (2.0 * s[..., None].astype(np.float64)) * (q >> 1) + b3[..., None]
        W4[p] = w4.reshape(o, K).astype(np.float32)
        D3[p] = (w3 - w4).reshape(o, K).astype(np.float32)
    return W4, D3


def silu(z):
    return z / (1.0 + np.exp(-z))


def topk(a, k):                                  # keep the k largest of each 64-wide group (along the last axis)
    n, D = a.shape
    if k >= 64:
        return np.ones((n, D), bool)
    g = a.reshape(n, D // 64, 64)
    thr = np.partition(g, 64 - k, axis=-1)[..., 64 - k:65 - k]
    return (g >= thr).reshape(n, D)


def dedupe(X):
    v = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
    _, i = np.unique(v, return_index=True)
    return X[np.sort(i)]


def validate():
    """Wed's records3-based judge: L41 e0 / L7 e5, TRL view vs MSB-3 on all fit rows (deduplicated by fit_b3.dedupe):
    gate 1.5544 / 1.4905 / down 0.6458 (L41 e0) and 1.3180 / 1.3182 / 0.5410 (L7 e5). Same numbers from this loader =
    its W4 and MSB-3 are records3's."""
    sys.path.insert(0, "/Users/kamen/Fit12/scripts")
    import fit_b3 as B
    import fit_trl_judge as J
    for L, e, ref in ((41, 0, (1.5544, 1.4905, 0.6458)), (7, 5, (1.3180, 1.3182, 0.5410))):
        XF = np.concatenate([B.dedupe(np.load(os.path.join(CAP, f"R{i}", f"x_L{L}.npy")))[0] for i in range(8)]).astype(np.float32)
        W4, D3 = expert(L, e)
        h = silu(XF @ W4["gate"].T) * (XF @ W4["up"].T)
        got = []
        for p in ("gate", "up", "down"):
            x = h if p == "down" else XF
            dT = (J.view(L, e, p).astype(np.float64) - W4[p]).astype(np.float32)
            a, b = x @ dT.T, x @ D3[p].T
            got.append((a * a).sum() / (b * b).sum())
        print(f"[halfk] validate L{L} e{e}: " + " | ".join(f"{p} {g:.4f} (Wed {r:.4f})" for p, g, r in
              zip(("gate", "up", "down"), got, ref)), flush=True)
        assert all(abs(g - r) < 6e-4 for g, r in zip(got, ref)), "loader does not reproduce the records3-based judge"
    print("[halfk] validate OK", flush=True)


def main():
    if "--validate" in sys.argv:
        return validate()
    layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2,12,24,36,46").split(",")]
    os.makedirs(OUT, exist_ok=True)
    sw0, t0 = swap_mb(), time.time()
    files = sorted(glob.glob(os.path.join(XCAP, "*.npz")))
    assert len(files) == 36
    MX = {L: [] for L in layers}; MI = {L: [] for L in layers}
    for f in files:
        z = np.load(f)
        for L in layers:
            MX[L].append(z["x"][:, L, :]); MI[L].append(z["ids"][:, L, :].astype(np.int64))
    print(f"[halfk] map rows read ({sum(a.shape[0] for a in MX[layers[0]])} answer rows), {time.time() - t0:.0f} s", flush=True)
    res = {"layers": {}, "arms": [f"{d} x{cx} h{ch}" for d, cx, ch in ARMS]}
    for L in layers:
        t1 = time.time()
        XF = np.concatenate([np.load(os.path.join(CAP, f"R{i}", f"x_L{L}.npy")) for i in range(8)])
        XF = dedupe(XF).astype(np.float32)
        C = np.zeros((2560, 2560))
        for i in range(0, XF.shape[0], 4096):
            c = XF[i:i + 4096].astype(np.float64)
            C += c.T @ c
        ev, U = np.linalg.eigh(C)
        V = U[:, ::-1][:, :32].astype(np.float32)
        top = float(ev[::-1][:32].sum() / ev.sum())
        del XF, C
        sets = {}
        xv = np.concatenate([np.load(os.path.join(CAP, f"R{i}", f"x_L{L}.npy")) for i in (8, 9)]).astype(np.float32)
        iv = np.concatenate([np.load(os.path.join(CAP, f"R{i}", f"ids_L{L}.npy")) for i in (8, 9)]).astype(np.int64)
        sets["val"] = (xv, iv, False)
        sets["map"] = (np.concatenate(MX[L]).astype(np.float32), np.concatenate(MI[L]), True)
        idx = {}
        for s, (X, I, ranked) in sets.items():
            flat = I.ravel()
            order = np.argsort(flat, kind="stable")
            idx[s] = (order // 10, order % 10, np.searchsorted(flat[order], np.arange(513)))
        # sums[set][group] = [MSB-3 err, |y4|^2, arm errs...]; groups: all, r0-1, r2-9 (map only)
        acc = {s: {g: np.zeros(2 + len(ARMS)) for g in ("all", "r0-1", "r2-9")} for s in sets}
        npairs = {s: {g: 0 for g in ("all", "r0-1", "r2-9")} for s in sets}
        for e in range(512):
            W4, D3 = expert(L, e)
            W3 = {p: W4[p] + D3[p] for p in W4}
            for s, (X, I, ranked) in sets.items():
                rows, ranks, bd = idx[s]
                r, k = rows[bd[e]:bd[e + 1]], ranks[bd[e]:bd[e + 1]]
                if len(r) == 0:
                    continue
                x = X[r]
                y4 = (silu(x @ W4["gate"].T) * (x @ W4["up"].T)) @ W4["down"].T
                y3 = (silu(x @ W3["gate"].T) * (x @ W3["up"].T)) @ W3["down"].T
                cols = [((y3 - y4) ** 2).sum(1), (y4 * y4).sum(1)]
                c = x @ V
                xp = x - c @ V.T
                for d, cx, ch in ARMS:
                    base = xp if d == "defl" else x
                    xt = base * topk(np.abs(base), cx)
                    if d == "defl":
                        xt = xt + c @ V.T
                    h = silu(xt @ W4["gate"].T) * (xt @ W4["up"].T)
                    yt = (h * topk(np.abs(h), ch)) @ W4["down"].T
                    cols.append(((yt - y4) ** 2).sum(1))
                cols = np.stack(cols, 1)
                groups = [("all", np.ones(len(r), bool))]
                if ranked:
                    groups += [("r0-1", k < 2), ("r2-9", k >= 2)]
                for g, m in groups:
                    acc[s][g] += cols[m].sum(0)
                    npairs[s][g] += int(m.sum())
            if e % 128 == 127:
                if swap_mb() - sw0 > 1024:
                    print(f"[halfk] STOP: swap grew {swap_mb() - sw0:.0f} MB", flush=True)
                    sys.exit(4)
                print(f"[halfk] L{L} e{e}: {time.time() - t1:.0f} s", flush=True)
        lay = {"V32_energy_share_fit": top}
        for s in sets:
            for g in acc[s]:
                if npairs[s][g] == 0:
                    continue
                a = acc[s][g]
                lay[f"{s} {g}"] = {"pairs": npairs[s][g], "msb3_rel": float(a[0] / a[1]),
                                   **{res["arms"][j]: float(a[2 + j] / a[0]) for j in range(len(ARMS))}}
        res["layers"][str(L)] = lay
        print(f"[halfk] L{L}: V32 holds {100 * top:.1f} % of the fit rows' energy; {time.time() - t1:.0f} s", flush=True)
        for key in [k for k in lay if k != "V32_energy_share_fit"]:
            v = lay[key]
            print(f"[halfk] L{L} {key:9s} pairs {v['pairs']:6d} | MSB-3 rel {v['msb3_rel']:.4f} | " +
                  " | ".join(f"{a} {v[a]:.2f}" for a in res["arms"]), flush=True)
        json.dump(res, open(os.path.join(OUT, "halfk.json"), "w"), indent=1)
    print(f"[halfk] done, {time.time() - t0:.0f} s, swap {swap_mb() - sw0:+.0f} MB", flush=True)


if __name__ == "__main__":
    main()
