"""Decision map night 2, stage 1 extras (Breakthrough, Mon 2026-11-02): for the chained run, the flip rate on the rows
that skipped (baseline-sure rows) by STALENESS = rows since the last computed row (a non-sure answer row, or the last
prompt row), per domain; and the share of answer rows that skip. Plain numpy (system-independent; no model).
Usage (BigMoE venv): python stage1_extra.py [chain_run_name]  -> out/extra_<name>.txt (stage 1's first output went to
out/stage1_extra.txt). A bounded chain (chain<S> in the name) applies the same refresh rule as dm_common.bound_runs."""
import json, os, re, sys
import numpy as np
OUT = os.path.expanduser(os.environ.get("DM_OUT", "~/FableGod/decision_map/out"))
name = sys.argv[1] if len(sys.argv) > 1 else "moe_reuse_chain_L12_35"
_m = re.search(r"chain(\d*)_", name); S = int(_m.group(1) or 0) if _m else 0   # bounded chain (stage 2)
_d = re.search(r"_dcf(0\d+)$", name); THR = float("0." + _d.group(1)[1:]) if _d else None   # DAY9: the drafter gate
if THR is not None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dm_gate; GATE = dm_gate.load()

def bound_runs(skip, S):                               # = dm_common.bound_runs
    if not S: return skip
    r = np.zeros(len(skip), int)
    for j in range(len(skip)): r[j] = (r[j - 1] + 1 if j else 1) if skip[j] else 0
    return skip & (r % (S + 1) != 0)

def classes(lp):                                       # = dm_common.classes: 0 sure, 1 tie, 2 mid
    p1, p2 = np.exp(lp[:, 0]), np.exp(lp[:, 1])
    return np.where(p1 >= 0.99, 0, np.where(p1 - p2 <= 0.1, 1, 2))

base = json.load(open(os.path.join(OUT, "baseline.json")))
run = json.load(open(os.path.join(OUT, name + ".json")))
B = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 10 ** 6)]
st_tab, cls_tab, skip_tab = {}, {}, {}
for nm, b in base.items():
    btk, blp, rtk = np.array(b["tk"]), np.array(b["lp"]), np.array(run[nm]["tk"])
    m = min(len(btk), len(rtk)); btk, blp, rtk = btk[:m], blp[:m], rtk[:m]
    c = classes(blp); f = rtk[:, 0] != btk[:, 0]
    s = bound_runs(c == 0 if THR is None else np.nan_to_num(GATE[nm][b["a"]:b["a"] + m], nan=-1) >= THR, S)
    st = np.zeros(m, int); last = -1                    # answer index -1 = the last prompt row (computed)
    for i in range(m):
        if s[i]: st[i] = i - last
        else: last = i
    dom = b["dom"]
    for lo, hi in B:
        k = s & (st >= lo) & (st <= hi)
        d = st_tab.setdefault(dom, {}).setdefault((lo, hi), [0, 0]); d[0] += int(f[k].sum()); d[1] += int(k.sum())
    d = cls_tab.setdefault(dom, {}).setdefault('refresh', [0, 0]); k = (c == 0) & ~s; d[0] += int(f[k].sum()); d[1] += int(k.sum())
    for k in range(3):
        d = cls_tab.setdefault(dom, {}).setdefault(k, [0, 0]); d[0] += int(f[c == k].sum()); d[1] += int((c == k).sum())
    d = skip_tab.setdefault(dom, [0, 0]); d[0] += int(s.sum()); d[1] += m
lines = [f"{name}: flip % on SKIPPED rows ({'drafter cf >= ' + str(THR) if THR else 'baseline-sure'}) by staleness = rows since the last computed row (flips / rows)"]
lines.append("| domain | skip share | " + " | ".join(f"{lo}-{hi if hi < 10 ** 6 else 'inf'}" for lo, hi in B)
             + " | computed rows: tie / mid flip % | sure rows that refresh (computed) |")
lines.append("|---" * (len(B) + 4) + "|")
for dom in ("code", "ifeval", "multi"):
    if dom not in st_tab: continue
    cells = []
    for b_ in B:
        fl, n = st_tab[dom][b_]
        cells.append(f"{100 * fl / n:.2f} ({fl}/{n})" if n else "-")
    t, md = cls_tab[dom][1], cls_tab[dom][2]
    sk = skip_tab[dom]
    lines.append(f"| {dom} | {100 * sk[0] / sk[1]:.1f} % | " + " | ".join(cells)
                 + f" | {100 * t[0] / max(t[1], 1):.1f} / {100 * md[0] / max(md[1], 1):.1f} | "
                 + (f"{100 * rf[0] / rf[1]:.2f} ({rf[0]}/{rf[1]})" if (rf := cls_tab[dom]['refresh'])[1] else "-") + " |")
txt = "\n".join(lines)
open(os.path.join(OUT, f"extra_{name}.txt"), "w").write(txt + "\n"); print(txt)
