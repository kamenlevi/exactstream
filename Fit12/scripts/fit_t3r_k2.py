"""Fit12 T3R K2 (2026-10-25; NewIdeas' bet T3R 19:58, ~/NewIdeas/QUEUE.md row T3R; pre-registered in PLAN.md 'T3R'): the
quality of a TRL-3 3INST base + a k_r = 1 trellis residual (R1) for rank 0-1 uses, on B10-r's 36 experts (seed 2, 9
layers) and captures (fit R0..R7, validation R8..R9; judges B all / B routed / U all / U routed; fit_b3.py main_b10r).
Per matrix, in B10-r's rotated frame:
  base  TRL-3 with the 3INST table at H damping 0.1 / 0.3; kept: the lowest VALIDATION error
  R1    r0 = W4' - base, its own bf16 row scale, BlockLDLQ k = 1 with the 3INST table at 0.1 / 0.3 (the same fit H);
        kept: the lowest validation error of base + R1.
  (Trimmed after the 1-expert smoke, 20:15: damping 0.01 (B10-r chose it for 1 of 108 matrices) and the D2-LUT residual
  side arm (= 3INST within 0.001 E4) dropped, so the run ends before the 21:41 GPU-queue wakes.)
Errors in E4 units as B10a (per matrix err / E4, E4 = W4's own rounding error s4^2 / 12 per weight, H-diagonal-weighted;
mean over experts per projection, then the mean of the three) and vs MSB-3 as B10-r. KILL (NewIdeas): base + R1 > 1.0 E4
on B routed. Bits per weight from the format (fit_t3r_k1.py): base 3 + 12/256 + 16/K, R1 1 + 12/256 + 16/K.
Usage: B3_THREADS=2 VECLIB_MAXIMUM_THREADS=2 ~/BigMoE/.venv/bin/python -B fit_t3r_k2.py   (CPU; F12_T3R_SMOKE=1: 1 expert)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_b3 as B

F, Q2 = B.F, B.Q2
DAMPS = (0.1, 0.3)
SMOKE = os.environ.get("F12_T3R_SMOKE") == "1"
OUT = os.path.expanduser("~/Fit12/data/t3r")
ARMS = ("TRL-3 3INST base", "base + R1 3INST")


def main(ne=4):
    t0 = time.time()
    q2 = np.random.default_rng(0)
    q2sel = {Ly: set(q2.choice(512, B.NE, replace=False).tolist()) for Ly in B.LAYERS}
    rng = np.random.default_rng(2)
    res, tim = {}, {}
    silu = lambda a: a / (1 + np.exp(-a))
    bits = {p: (3 + 12 / 256 + 16 / K, 1 + 12 / 256 + 16 / K) for p, K in (("gate", 2560), ("up", 2560), ("down", 640))}

    def add(nm, p, v):
        res.setdefault(nm, {q: [] for q in F.PROJ})[p].append(v)

    def e4(s3u, H):
        s4 = np.repeat(F.bf16(s3u) / 2, 64, 1).astype(np.float64)
        return float(((s4 * s4 / 12.0) * np.diag(H)[None, :]).sum())

    def ld(d, Ly, ids=False):
        X, I = B.dedupe(np.load(os.path.join(d, f"x_L{Ly}.npy")), np.load(os.path.join(d, f"ids_L{Ly}.npy")) if ids else None)
        return (X, I) if ids else X

    def row(nm, m):
        return f"{nm} | {m['gate']:.3f} | {m['up']:.3f} | {m['down']:.3f} | {np.mean(list(m.values())):.3f}"

    def e4u(tn, arm):                                       # B10a's E4 units: mean over experts of the per-matrix ratios
        return {p: float(np.mean(res[f"[{tn}] {arm} / E4"][p])) for p in F.PROJ}

    def report(fh):
        tns = [tn for tn in ("val", "B all", "B routed", "U all", "U routed") if f"[{tn}] MSB-3" in res]
        for tn in tns:
            n = len(res[f"[{tn}] MSB-3"]["gate"])
            print(f"judged on {tn} ({n} experts) | gate | up | down | mean", file=fh)
            print(row("MSB-3 / E4 (sanity ~3)", e4u(tn, "MSB-3")), file=fh)
            for arm in ARMS:
                print(row(f"{arm} / E4", e4u(tn, arm)), file=fh)
            for arm in ARMS:
                print(row(f"{arm} vs MSB-3 (x)", {p: np.mean(res[f"[{tn}] {arm}"][p]) / np.mean(res[f"[{tn}] MSB-3"][p])
                                                  for p in F.PROJ}), file=fh)
            print(row("base + R1 3INST, mean err / mean E4 (side)", {p: np.mean(res[f"[{tn}] base + R1 3INST"][p]) /
                                                                     np.mean(res[f"[{tn}] E4"][p]) for p in F.PROJ}), file=fh)
        for tn in ("B routed", "B all", "U all", "U routed"):
            if f"[{tn}] MSB-3" in res:
                v = float(np.mean(list(e4u(tn, "base + R1 3INST").values())))
                b = float(np.mean(list(e4u(tn, "TRL-3 3INST base").values())))
                tag = f"(kill > 1.0) -> {'KILL' if v > 1.0 else 'PASS'}" if tn == "B routed" else "(level)"
                print(f"{'DECISIVE' if tn == 'B routed' else 'level'} K2 base + R1 (3INST, val-best d) on {tn}: {v:.3f} E4 "
                      f"{tag}; TRL-3 base alone {b:.3f} E4", file=fh)
        for k in ("base", "R1"):
            dm = res[f"val-best damping {k}"]
            print(f"val-best damping {k}: " + ", ".join(f"{p} " + "/".join(str(sum(x == d for x in dm[p])) for d in DAMPS)
                  for p in F.PROJ) + f" (counts of {DAMPS})", file=fh)
        for k in ("B", "U"):
            r = res[f"routed rows {k}"]["gate"]
            print(f"routed rows per expert in {k}: mean {np.mean(r):.0f}, min {min(r):.0f}, < 8 (no routed test) "
                  f"{sum(x < 8 for x in r)} of {len(r)}", file=fh)
        nw = {"gate": 640 * 2560, "up": 640 * 2560, "down": 2560 * 640}
        for i, k in enumerate(("TRL-3 base", "R1")):
            tot = sum(bits[p][i] * nw[p] for p in F.PROJ) / 8
            print(f"bits / weight {k}: gate {bits['gate'][i]:.4f}, up {bits['up'][i]:.4f}, down {bits['down'][i]:.4f}; "
                  f"{tot:,.0f} B per expert ({8 * tot / sum(nw.values()):.4f} b/w)", file=fh)
        for k, v in tim.items():
            print(f"encode {k}: median {np.median(v):.2f} s per matrix (B3_THREADS={B.NTHR}, {len(v)} encodes)", file=fh)
        fh.flush()

    for Ly in (B.LAYERS9[:1] if SMOKE else B.LAYERS9):
        XF = np.concatenate([ld(os.path.join(B.B10R, f"R{i}"), Ly) for i in range(8)])
        XV = np.concatenate([ld(os.path.join(B.B10R, f"R{i}"), Ly) for i in (8, 9)])
        XB, idB = ld(os.path.join(B.CAPD, "B"), Ly, True)
        XU, idU = ld(os.path.join(B.B10R, "U"), Ly, True)
        Rx = B.rot(2560, Ly)
        HFr = Rx.T @ (XF.T @ XF) @ Rx
        Ux = {}
        for d in DAMPS:
            Ux[d], dead = Q2.prep(HFr, d)
            assert not dead.any()
        HX = {tn: Z.T @ Z for tn, Z in (("val", XV), ("B all", XB), ("U all", XU))}
        HXr = {tn: Rx.T @ H @ Rx for tn, H in HX.items()}
        pool = [e for e in range(512) if e not in q2sel.get(Ly, ())]
        es = sorted(rng.choice(pool, ne, replace=False).tolist())[:1 if SMOKE else ne]
        print(f"[t3r] layer {Ly}: fit rows {XF.shape[0]}, val {XV.shape[0]}, B {XB.shape[0]}, U {XU.shape[0]}, experts {es}", flush=True)
        for e in es:
            B.step_aside()
            rec = F.read(Ly, e)
            W4, W3 = {}, {}
            for p in F.PROJ:
                r = rec[p]
                s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
                W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
                W3[p] = (np.repeat(s3, 64, 1) * r["m"] + np.repeat(b3, 64, 1)).astype(np.float64)
            hs = lambda Z: silu(Z @ W4["gate"].T) * (Z @ W4["up"].T)
            hF = hs(XF)
            Rh = B.rot(640, 1000 * Ly + e)
            HhFr = Rh.T @ (hF.T @ hF) @ Rh
            Uh = {}
            for d in DAMPS:
                Uh[d], dead = Q2.prep(HhFr, d)
                assert not dead.any()
            tests = {"val": XV, "B all": XB, "U all": XU}
            mB, mU = (idB == e).any(1), (idU == e).any(1)
            add("routed rows B", "gate", float(mB.sum()))
            add("routed rows U", "gate", float(mU.sum()))
            if mB.sum() >= 8:
                tests["B routed"] = XB[mB]
            if mU.sum() >= 8:
                tests["U routed"] = XU[mU]
            Hs = {}
            for tn, Z in tests.items():
                hz = hs(Z)
                Hx = HX[tn] if tn in HX else Z.T @ Z
                Hs[tn] = {"x": (Hx, HXr[tn] if tn in HXr else Rx.T @ Hx @ Rx), "h": (hz.T @ hz, Rh.T @ (hz.T @ hz) @ Rh)}
            for p in F.PROJ:
                R, U, key = (Rh, Uh, "h") if p == "down" else (Rx, Ux, "x")
                Wr = W4[p] @ R
                sc = F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
                Hvr = Hs["val"][key][1]
                base = {}
                for d in DAMPS:
                    t1 = time.time()
                    base[d] = B.trl(Wr, U[d], 3, sc, B.LUT3)
                    tim.setdefault("TRL-3 3INST (k = 3)", []).append(time.time() - t1)
                db = min(DAMPS, key=lambda d: Q2.err(base[d], Wr, Hvr))
                qb = base[db]
                r0 = Wr - qb
                sr = F.rbf16(np.sqrt((r0 * r0).mean(1)).astype(np.float32)).astype(np.float64)
                full = {}
                for d in DAMPS:
                    t1 = time.time()
                    full[d] = qb + B.trl(r0, U[d], 1, sr, B.LUT3)
                    tim.setdefault("R1 3INST (k = 1)", []).append(time.time() - t1)
                dr = min(DAMPS, key=lambda d: Q2.err(full[d], Wr, Hvr))
                add("val-best damping base", p, db)
                add("val-best damping R1", p, dr)
                q = {"TRL-3 3INST base": qb, "base + R1 3INST": full[dr]}
                for tn in tests:
                    Hj, Hjr = Hs[tn][key]
                    den = float(((W4[p] @ Hj) * W4[p]).sum())
                    E4 = e4(rec[p]["s3u"], Hj) / den
                    msb = Q2.err(W3[p], W4[p], Hj)
                    add(f"[{tn}] E4", p, E4)
                    add(f"[{tn}] MSB-3", p, msb)
                    add(f"[{tn}] MSB-3 / E4", p, msb / E4)
                    for nm, Q in q.items():
                        v = Q2.err(Q, Wr, Hjr)
                        add(f"[{tn}] {nm}", p, v)
                        add(f"[{tn}] {nm} / E4", p, v / E4)
            print(f"[t3r] layer {Ly} expert {e} done ({time.time() - t0:.0f} s)", flush=True)
        with open(os.path.join(OUT, "k2_smoke_partial.txt" if SMOKE else "k2_partial.txt"), "w") as fh:   # kept if the run stops early
            print(f"partial: layers through {Ly} ({time.time() - t0:.0f} s)", file=fh)
            report(fh)
    report(sys.stdout)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
