"""Fit12 (Wed 4 Nov 22:4x; NewIdeas 22:33 for the orchestrator's QUEUE 22:3x, pre-registered in 5fa0f47): the screen-7 TRL
store (/Users/kamen/Fit12/data/trl, the 2.40 b mix: gate / up TRL-2, down TRL-3, as 8-bit views) judged against MSB-3 on
ROUTED inputs per domain. Question: is screen 7's miss (KILL, d37b3ef: the mix did the damage of an error ratio of 2.72 /
2.55 / 3.15 vs MSB-3 on code / IFEval / multi) the inputs' distribution? The store was fit on a broad 7-domain mix.
Inputs: B10-r's capture /Users/kamen/Fit12/data/b10rcap (MoE input x as fp16 + each row's top-10 experts by router logit;
row r = token r of the part, checked: rows 0-2 identical across the 10 parts, as their first 3 tokens are):
  val = R8 + R9 (held out from the fit, 8,190 rows), fit = R0..R7 (the store's own fit rows: in-sample), U = English
  literary prose (held out, a domain absent from the fit). Per-token domains from the parts' round-robin layout
  (/Users/kamen/Fit12/scripts/fit_dom_labels.py -> b10rcap/domain_labels.npz; == tokens_by_domain in all 10 parts).
Per layer, per expert (all 512), on the rows routed to that expert, per set and domain, errors against W4 (the 4-bit model):
  gate / up: ||(W - W4) x||^2;  down: ||(W - W4) h4||^2, h4 = silu(W4g x) * (W4u x) (the expert's own W4 hidden);
  whole expert: ||y(W) - y(W4)||^2, y = Wd (silu(Wg x) * (Wu x)) with all three projections replaced (no router weight).
  W = the store's 8-bit view (what the map ran; dequantized exactly in fp32 numpy, checked against mx.dequantize) or MSB-3
  (records3's m with s3 / b3, as fit_trl_store.py builds it).
Ratio = the sum over experts and routed rows of the store's error / the same sum for MSB-3 (traffic-weighted, pooled); also
the median of per-expert ratios over experts with >= 8 routed rows in that cell.
CPU only (numpy, 4 threads), < 2 GB. Writes /Users/kamen/Fit12/data/trl_judge/judge_L{L}.json + prints the tables.
Usage: [BigMoE venv] python -B fit_trl_judge.py [LAYERS 7,18,30,41] [E0-E1]"""
import json, os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_q as F

CAP = "/Users/kamen/Fit12/data/b10rcap"
STORE = "/Users/kamen/Fit12/data/trl"
OUT = "/Users/kamen/Fit12/data/trl_judge"
DOM = ("math", "code", "instr", "multi", "wiki", "know", "agent", "prose")
SETS = {"val": [f"R{i}" for i in (8, 9)], "fit": [f"R{i}" for i in range(8)], "U": ["U"]}
KS = {"gate": 2, "up": 2, "down": 3}
Q = ("gate", "up", "down", "whole")


def silu(z):
    return z / (1.0 + np.exp(-z))


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
    q = t["w"].view(np.uint8).astype(np.float32)                     # [o, i]
    s = (t["s"].astype(np.uint32) << 16).view(np.float32)
    b = (t["b"].astype(np.uint32) << 16).view(np.float32)
    return np.repeat(s, 64, 1) * q + np.repeat(b, 64, 1)


def load_set(L, parts, labels):
    X, I, D = [], [], []
    for p in parts:
        X.append(np.load(os.path.join(CAP, p, f"x_L{L}.npy")).astype(np.float32))
        I.append(np.load(os.path.join(CAP, p, f"ids_L{L}.npy")).astype(np.int64))
        n = X[-1].shape[0]
        D.append(labels[p][:n].astype(np.int64) if p in labels else np.full(n, DOM.index("prose"), np.int64))
    X, I, D = np.concatenate(X), np.concatenate(I), np.concatenate(D)
    assert I.shape == (X.shape[0], 10) and (D >= 0).all()
    order = np.argsort(I.ravel(), kind="stable")                      # rows routed to each expert
    rows = (order // 10)
    bounds = np.searchsorted(I.ravel()[order], np.arange(513))
    return X, D, rows, bounds


def main():
    layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "7,18,30,41").split(",")]
    e0, e1 = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0-511").split("-"))
    os.makedirs(OUT, exist_ok=True)
    labels = dict(np.load(os.path.join(CAP, "domain_labels.npz")))
    nd = len(DOM)
    for L in layers:
        t0 = time.time()
        S = {s: load_set(L, parts, labels) for s, parts in SETS.items()}
        acc = {s: np.zeros((2, len(Q), nd)) for s in S}               # [store / msb3, quantity, domain] error sums
        cnt = {s: np.zeros(nd, np.int64) for s in S}                  # routed (row, expert) pairs per domain
        per = {s: [[[] for _ in range(nd + 1)] for _ in Q] for s in S}   # per-expert ratios (domains + all)
        for e in range(e0, e1 + 1):
            rec = F.read(L, e)
            W4, W3, WT = {}, {}, {}
            for p in F.PROJ:
                r = rec[p]
                s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
                w4 = np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)
                w3 = np.repeat(s3, 64, 1) * r["m"] + np.repeat(b3, 64, 1)
                W4[p] = w4.astype(np.float32)
                W3[p] = (w3 - w4).astype(np.float32)                   # deltas, formed in float64
                WT[p] = (view(L, e, p).astype(np.float64) - w4).astype(np.float32)
            for s, (X, D, rows, bounds) in S.items():
                idx = rows[bounds[e]:bounds[e + 1]]
                if len(idx) == 0:
                    continue
                x, d = X[idx], D[idx]
                g4, u4 = x @ W4["gate"].T, x @ W4["up"].T
                h4 = silu(g4) * u4
                y4 = h4 @ W4["down"].T
                errs = np.zeros((2, len(Q), len(idx)))
                for v, dW in enumerate((WT, W3)):
                    dg, du = x @ dW["gate"].T, x @ dW["up"].T
                    errs[v, 0] = (dg * dg).sum(1)
                    errs[v, 1] = (du * du).sum(1)
                    dd = h4 @ dW["down"].T
                    errs[v, 2] = (dd * dd).sum(1)
                    h = silu(g4 + dg) * (u4 + du)
                    dy = h @ (W4["down"] + dW["down"]).T - y4
                    errs[v, 3] = (dy * dy).sum(1)
                for v in range(2):
                    for qi in range(len(Q)):
                        acc[s][v, qi] += np.bincount(d, errs[v, qi], nd)
                cnt[s] += np.bincount(d, minlength=nd)
                for qi in range(len(Q)):
                    a, b = np.bincount(d, errs[0, qi], nd), np.bincount(d, errs[1, qi], nd)
                    n = np.bincount(d, minlength=nd)
                    for di in range(nd):
                        if n[di] >= 8 and b[di] > 0:
                            per[s][qi][di].append(a[di] / b[di])
                    if len(idx) >= 8:
                        per[s][qi][nd].append(errs[0, qi].sum() / errs[1, qi].sum())
            if e % 64 == 63 or e == e1:
                print(f"[judge] L{L} e{e}: {time.time() - t0:.0f} s", flush=True)
        res = {"layer": L, "experts": [e0, e1], "sets": {}}
        for s in S:
            tab = {}
            for di, dn in list(enumerate(DOM)) + [(nd, "all")]:
                sl = slice(None) if dn == "all" else slice(di, di + 1)
                pairs = int(cnt[s][sl].sum())
                if pairs == 0:
                    continue
                row = {"pairs": pairs}
                for qi, qn in enumerate(Q):
                    st, ms = acc[s][0, qi, sl].sum(), acc[s][1, qi, sl].sum()
                    pe = per[s][qi][di]
                    row[qn] = st / ms
                    row[qn + "_median"] = float(np.median(pe)) if pe else None
                    row[qn + "_experts"] = len(pe)
                tab[dn] = row
            res["sets"][s] = tab
        res["seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(os.path.join(OUT, f"judge_L{L}.json"), "w"), indent=1)
        print(f"[judge] L{L} store / MSB-3 error, pooled over routed rows (median of per-expert ratios), {res['seconds']} s")
        for s, tab in res["sets"].items():
            for dn, r in tab.items():
                print(f"[judge] L{L} {s:3s} {dn:6s} pairs {r['pairs']:7d} | " + " | ".join(
                    f"{qn} {r[qn]:.3f}x ({r[qn + '_median']:.3f})" if r[qn + "_median"] is not None else f"{qn} {r[qn]:.3f}x (-)"
                    for qn in Q), flush=True)


if __name__ == "__main__":
    main()
