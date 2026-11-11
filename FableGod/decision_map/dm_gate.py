"""The REALIZABLE gate for the decision map (Breakthrough, Mon 2026-11-02; NewIdeas 07:3x, DAY9 Q1 / Q2): per row, the
drafter's confidence for the draft that row verifies, as the live engine sees it before the check.
Source: MTP Head's HPD-0 walk of DFlash2 L1-20 (~/TrainLab/data/d02b_walk_hpd0_<dom>.npz, TL_WALK_CF): per check the
anchor A, the accepted count k, and the 15 drafts' cf (the candidate selector's softmax at each pick, the quantity the
default's running-confidence cut multiplies). Its checks equal the map's window walks (d02b_walk_code /
d02b_walk_d02a_{ifeval,multi}) on all 36 windows. Row A + j (input ids[A + j], predicting ids[A + j + 1]) verifies
draft j + 1, so its gate is cf[j], for j = 0 .. k (row A + k is the bonus row, gated by the drafter's own first-miss
draft); after 15 accepted drafts the bonus row has no draft (NaN = exact). Prompt rows are NaN (exact prefill).
Caveat: the walk does not apply the t 0.3 cut; the cut ends a check before its first miss on 4-8 % of checks (2-4 % of
kept rows), where the live engine would start a new check instead.
Usage (BigMoE venv): python dm_gate.py -> out/gate_dfl2.json {name: [gate of rows 0 .. n-2, null = none]} and the
agreement table (drafter bands x the 125B's own class) on the measured rows."""
import json, os
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
OUT = os.path.expanduser(os.environ.get("DM_OUT", "~/FableGod/decision_map/out"))
WALK = {"code": "d02b_walk_hpd0_code.npz", "ifeval": "d02b_walk_hpd0_ifeval.npz", "multi": "d02b_walk_hpd0_multi.npz"}
MAPW = {"code": "d02b_walk_code.npz", "ifeval": "d02b_walk_d02a_ifeval.npz", "multi": "d02b_walk_d02a_multi.npz"}
BANDS = [(0.999, 2.0, ">=0.999"), (0.99, 0.999, "0.99-0.999"), (0.9, 0.99, "0.9-0.99"), (-1.0, 0.9, "<0.9")]

def build():
    G = {}
    for dom, wf in WALK.items():
        Z, M = np.load(os.path.join(TD, wf)), np.load(os.path.join(TD, MAPW[dom]))
        zn = [str(x) for x in Z["names"]]
        for s, nm in enumerate([str(x) for x in M["names"]][:int(os.environ.get("DM_MAX", "12"))]):   # FLOOR night: DM_MAX 30
            i = zn.index(nm); ids = Z[f"{i}_ids"]; n = len(ids)
            assert np.array_equal(ids, M[f"{s}_ids"]) and np.array_equal(Z[f"{i}_checks"], M[f"{s}_checks"]), nm
            g = np.full(n - 1, np.nan)
            for (k, m, A), cf in zip(Z[f"{i}_checks"], Z[f"{i}_dcf"]):
                for j in range(k + 1):                   # accepted rows A .. A + k - 1 and the bonus row A + k
                    if A + j >= n - 1: break
                    g[A + j] = cf[j] if j < m else np.nan  # j == m: every checkable draft kept, no draft for the bonus
            G[nm] = g
    return G

def load():
    return {nm: np.array([np.nan if v is None else v for v in g]) for nm, g in json.load(open(os.path.join(OUT, "gate_dfl2.json"))).items()}

if __name__ == "__main__":
    G = build()
    json.dump({nm: [None if np.isnan(v) else round(float(v), 6) for v in g] for nm, g in G.items()},
              open(os.path.join(OUT, "gate_dfl2.json"), "w"))
    base = json.load(open(os.path.join(OUT, "baseline.json")))
    lines = ["DFlash2 gate (HPD-0 walk cf) vs the 125B's own class on the measured rows: share of rows per band, and per band "
             "the oracle split sure / tie / mid (% of the band's rows)"]
    for dom in ("code", "ifeval", "multi"):
        tab, nrow = {}, 0
        for nm, b in base.items():
            if b["dom"] != dom: continue
            a, lp = b["a"], np.array(b["lp"]); p1, p2 = np.exp(lp[:, 0]), np.exp(lp[:, 1])
            c = np.where(p1 >= 0.99, 0, np.where(p1 - p2 <= 0.1, 1, 2))
            g = G[nm][a:a + len(c)]; nrow += len(c)
            assert not np.isnan(g).all(), nm
            for lo, hi, lab in BANDS + [(None, None, "none")]:
                k = np.isnan(g) if lo is None else (g >= lo) & (g < hi)
                t = tab.setdefault(lab, np.zeros(3, int))
                for x in range(3): t[x] += int((k & (c == x)).sum())
        lines.append(f"{dom} ({nrow} rows): " + "; ".join(
            f"{lab} {100 * t.sum() / nrow:.1f} % [{' / '.join(f'{100 * v / max(t.sum(), 1):.0f}' for v in t)}]"
            for lab, t in tab.items()))
        s = sum(tab[l] for l in (">=0.999", "0.99-0.999")); allc = sum(tab.values())
        lines.append(f"  drafter-sure (cf >= 0.99): {100 * s.sum() / nrow:.1f} % of rows; oracle-sure {100 * allc[0] / nrow:.1f} %; "
                     f"of drafter-sure rows {100 * s[0] / max(s.sum(), 1):.1f} % oracle-sure; of oracle-sure rows "
                     f"{100 * s[0] / max(allc[0], 1):.1f} % drafter-sure")
    txt = "\n".join(lines); open(os.path.join(OUT, "gate_dfl2_agreement.txt"), "w").write(txt + "\n"); print(txt)
