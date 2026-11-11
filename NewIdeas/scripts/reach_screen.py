"""REACH-CUT screen (NewIdeas, Tue 3 Nov 2026; CPU only, writes nothing). Idea (adapted from AcceptMoE's offline per-position
commitment weights, arXiv 2608.02989, made per row and online): on top of Q1-UNION 0.9, an UNSURE row (cf < 0.9) whose
chance of being reached R_j = prod_{i < j} cf_i is low (0.3 <= R_j < tau; the live default already ends the check
below 0.3) keeps only its top-2 + the check's union members, because such rows are rarely reached.
Numbers per domain, on the map's 36 windows and the HPD-0 walks (checks cut at R < 0.3 as the live engine would):
- union per check (unique experts summed over layers) vs exact, for Q1-UNION 0.9 alone and with the REACH band;
- realized rows (A .. first miss) that fall in the band, and their flips in p1_oea_k2 (top-2 + union on every row, an
  all-rows run: a pessimistic proxy for the band's own flips) added to q1u_cf09's all-row flips, vs Q1 at 0.99's line.
Usage (BigMoE venv): python reach_screen.py"""
import json, os, sys
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import dm_gate  # noqa: E402
OUT = os.path.join(DM, "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
P2 = json.load(open(os.path.join(OUT, "p1_oea_k2.json"))); QU = json.load(open(os.path.join(OUT, "q1u_cf09.json")))
LINE = {"code": 1.34, "ifeval": 5.42, "multi": 3.69}
TAUS = (0.4, 0.5, 0.7, 1.01)


def flipvec(R, nm):
    b = base[nm]; m = min(len(b["tk"]), len(R[nm]["tk"]))
    return b["a"], np.r_[np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0], np.zeros(len(b["tk"]) - m, bool)]


res = {}
for dom, wf in dm_gate.WALK.items():
    Z = np.load(os.path.join(dm_gate.TD, wf))
    for i, nm in enumerate(str(x) for x in Z["names"]):
        if nm not in base: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids)
        a, fq = flipvec(QU, nm); _, f2 = flipvec(P2, nm)
        for (k, m, A), dc in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            dc = np.array(dc, float); Rr = np.cumprod(dc)
            L = int(np.searchsorted(-Rr, -0.3, side="right"))     # drafts kept while R >= 0.3 (live cut): rows A .. A + L
            rows, cfs, Rs = [], [], []
            fm = min(A + k, N - 1)
            for j in range(min(L, m) + 1):
                p = A + j
                if p > N - 1: break
                r = p if p <= fm else (p + 97) % N              # rows after the first miss: wrong-draft stand-ins
                rows.append(r); cfs.append(dc[j] if j < m else -1.0)
                Rs.append(1.0 if j == 0 else Rr[j - 1])        # P(reach row A + j) = prod of the drafts before it
            cfs, Rs = np.array(cfs), np.array(Rs)
            kq = np.where(cfs >= 0.9, 2, 10); sub = ids[rows]                        # [rows, 48, 10]
            ux = sum(len(np.unique(sub[:, l, :])) for l in range(48))
            uq = sum(len(np.unique(np.concatenate([sub[j, l, :kq[j]] for j in range(len(rows))]))) for l in range(48))
            for tau in TAUS:
                d = res.setdefault((dom, tau), dict(u_ex=0, u_q=0, u_r=0, band_real=0, band_flip=0, real=0))
                kr = np.where((cfs < 0.9) & (Rs < tau), 2, kq)
                d["u_ex"] += ux; d["u_q"] += uq
                d["u_r"] += uq if (kr == kq).all() else sum(
                    len(np.unique(np.concatenate([sub[j, l, :kr[j]] for j in range(len(rows))]))) for l in range(48))
                for j, r in enumerate(rows):
                    if r != A + j or r > fm: continue          # realized rows only
                    if r < a: continue                         # measured rows start at a
                    d["real"] += 1
                    if kr[j] == 2 and kq[j] == 10:
                        d["band_real"] += 1; d["band_flip"] += int(f2[r - a]) if r - a < len(f2) else 0
for dom in ("code", "ifeval", "multi"):
    n = sum(len(b["tk"]) for b in base.values() if b["dom"] == dom)
    fq = sum(int(flipvec(QU, nm)[1].sum()) for nm, b in base.items() if b["dom"] == dom)
    for tau in TAUS:
        d = res[(dom, tau)]
        est = 100 * (fq + d["band_flip"]) / n
        print(f"{dom} tau {tau}: union x{d['u_q'] / d['u_ex']:.3f} (Q1-UNION 0.9, live-cut checks) -> x{d['u_r'] / d['u_ex']:.3f} with "
              f"the band; band realized rows {d['band_real']} of {d['real']} ({100 * d['band_real'] / max(d['real'], 1):.1f} %), "
              f"their p1_oea_k2 flips {d['band_flip']}; all-row flips q1u_cf09 {100 * fq / n:.2f} % -> est. {est:.2f} % "
              f"(line {LINE[dom]})", flush=True)
