"""Fit12 (Wed 4 Nov 22:5x; NewIdeas 22:33 for the orchestrator's QUEUE 22:3x): the screen-7 TRL store vs MSB-3 on the MAP'S
OWN MoE inputs. Data (read-only): Breakthrough's xcapture, /Users/kamen/FableGod/decision_map/out/xcap/<window>.npz (run_map.py
"xcapture", Tue 3 Nov 02:0x): per answer row of the map's 36 windows (12 each of code / ifeval / multi), the MoE input x at
every layer (f16), the rank-sorted router ids and normalised router weights.
Same errors as /Users/kamen/Fit12/scripts/fit_trl_judge.py, against W4 (the 4-bit model): gate / up ||(W - W4) x||^2, down
||(W - W4) h4||^2 on the expert's own W4 hidden, whole expert ||y(W) - y(W4)||^2 with all three projections replaced; W = the
store's 8-bit view or MSB-3. Per layer 7 / 18 / 30 / 41 and domain, by rank group: ranks 2-9 (the experts s7_trl served from
the store), ranks 0-1, all 10. Plus whole_w = the whole-expert error weighted by the router weight squared (the expert's
share of the MoE output error). Ratio = pooled store error / pooled MSB-3 error over the (row, expert) pairs.
CPU only (numpy, 4 threads), < 2 GB. Writes /Users/kamen/Fit12/data/trl_judge/judge_map.json + prints the tables.
Usage: [BigMoE venv] python -B fit_trl_judge_map.py [LAYERS 7,18,30,41]"""
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_q as F
import fit_trl_judge as J

XCAP = "/Users/kamen/FableGod/decision_map/out/xcap"
OUT = "/Users/kamen/Fit12/data/trl_judge"
DOMS = ("code", "ifeval", "multi")
GROUPS = {"r2-9": list(range(2, 10)), "r0-1": [0, 1], "all": list(range(10))}
Q = ("gate", "up", "down", "whole", "whole_w")


def main():
    layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "7,18,30,41").split(",")]
    files = sorted(glob.glob(os.path.join(XCAP, "*.npz")))
    assert len(files) == 36, len(files)
    dom = [DOMS.index(os.path.basename(f).split("_")[1]) for f in files]
    t0 = time.time()
    X, I, Wt, D = {L: [] for L in layers}, {L: [] for L in layers}, {L: [] for L in layers}, []
    for f, d in zip(files, dom):                                  # one read per window, all layers at once
        z = np.load(f)
        x, ids, w = z["x"], z["ids"], z["w"]
        for L in layers:
            X[L].append(x[:, L, :].astype(np.float32))
            I[L].append(ids[:, L, :].astype(np.int64))
            Wt[L].append(w[:, L, :].astype(np.float32))
        D.append(np.full(x.shape[0], d, np.int64))
    D = np.concatenate(D)
    print(f"[map-judge] {len(files)} windows, {D.shape[0]} answer rows (code {int((D == 0).sum())}, ifeval "
          f"{int((D == 1).sum())}, multi {int((D == 2).sum())}), read in {time.time() - t0:.0f} s", flush=True)
    res = {"layers": {}, "rows": {dn: int((D == i).sum()) for i, dn in enumerate(DOMS)}}
    for L in layers:
        t1 = time.time()
        x_all, ids, wt = np.concatenate(X[L]), np.concatenate(I[L]), np.concatenate(Wt[L])
        flat = ids.ravel()
        order = np.argsort(flat, kind="stable")
        rows, ranks = order // 10, order % 10
        bounds = np.searchsorted(flat[order], np.arange(513))
        acc = np.zeros((2, len(Q), len(DOMS), 10))                # [store / msb3, quantity, domain, rank]
        cnt = np.zeros((len(DOMS), 10), np.int64)
        for e in range(512):
            r, k = rows[bounds[e]:bounds[e + 1]], ranks[bounds[e]:bounds[e + 1]]
            if len(r) == 0:
                continue
            rec = F.read(L, e)
            W4, D3, DT = {}, {}, {}
            for p in F.PROJ:
                q = rec[p]
                s3, b3, bb = F.bf16(q["s3u"]), F.bf16(q["b3u"]), F.bf16(q["bu"])
                w4 = np.repeat(s3 / 2, 64, 1) * (2 * q["m"] + q["l"]) + np.repeat(bb, 64, 1)
                W4[p] = w4.astype(np.float32)
                D3[p] = (np.repeat(s3, 64, 1) * q["m"] + np.repeat(b3, 64, 1) - w4).astype(np.float32)
                DT[p] = (J.view(L, e, p).astype(np.float64) - w4).astype(np.float32)
            x, d, w = x_all[r], D[r], wt[r, k]
            g4, u4 = x @ W4["gate"].T, x @ W4["up"].T
            h4 = J.silu(g4) * u4
            y4 = h4 @ W4["down"].T
            for v, dW in enumerate((DT, D3)):
                dg, du = x @ dW["gate"].T, x @ dW["up"].T
                dd = h4 @ dW["down"].T
                dy = (J.silu(g4 + dg) * (u4 + du)) @ (W4["down"] + dW["down"]).T - y4
                ey = (dy * dy).sum(1)
                for qi, val in enumerate(((dg * dg).sum(1), (du * du).sum(1), (dd * dd).sum(1), ey, ey * w * w)):
                    np.add.at(acc[v, qi], (d, k), val)
            np.add.at(cnt, (d, k), 1)
        lay = {}
        for di, dn in enumerate(DOMS + ("all",)):
            dsl = slice(None) if dn == "all" else slice(di, di + 1)
            for gn, gk in GROUPS.items():
                row = {"pairs": int(cnt[dsl][:, gk].sum())}
                for qi, qn in enumerate(Q):
                    row[qn] = float(acc[0, qi][dsl][:, gk].sum() / acc[1, qi][dsl][:, gk].sum())
                lay[f"{dn} {gn}"] = row
        res["layers"][str(L)] = lay
        print(f"[map-judge] L{L} store / MSB-3 error on the map's answer rows, pooled ({time.time() - t1:.0f} s)", flush=True)
        for key, row in lay.items():
            print(f"[map-judge] L{L} {key:12s} pairs {row['pairs']:6d} | " +
                  " | ".join(f"{qn} {row[qn]:.3f}x" for qn in Q), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "judge_map.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
