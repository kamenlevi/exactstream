"""WALL 1105: every measured map arm on disk, read by the 125B's own row class (floor_rows.json: sure / mid / tie, tie =
p1 - p2 <= 0.1) on the 36 map windows: all-row flips, non-tie flips, and flips on tie rows. CPU, seconds.
Purpose: how much of the map's all-row flip rate is the near-tie rows, which flip under ANY perturbation.
Output: wall_class_flips.out."""
import ast, glob, gzip, json, os, sys
B = os.path.expanduser("~/Breakthrough/data"); D = os.path.expanduser("~/FableGod/decision_map/out")
fr = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))
CLS = {(r["nm"], int(r["i"])): r["cls"] for r in fr}; DOM = {r["nm"]: r["dom"] for r in fr}
def load(p): return json.load(gzip.open(p)) if p.endswith(".gz") else json.load(open(p))
SETS = [(B + "/floor1", B + "/floor1/baseline.json", ["oea_k8", "gate_top5_cf099", "gate_top2_cf0999", "gate_bands", "hyb_0.9_u8_live_t3"]),
        (B + "/floor1_36", B + "/floor1_36/baseline.json", ["gate_top4_cf099", "gate_top0_cf0999_L08_35", "gate_bands+moe_reuse_chain1_L08_35_dcf0999"]),
        (B + "/dm_s7/results_wed", D + "/baseline.json", None), (B + "/dm_s7/results_thu", D + "/baseline.json", None)]
lines = ["| arm | all-row flips code / IFEval / multi | non-tie rows | tie rows | sure rows | mid rows |", "|---|---|---|---|---|---|"]
for d, bp, arms in SETS:
    if not os.path.isdir(d): continue
    base = load(bp)
    files = [os.path.join(d, a + ".json") for a in arms] if arms else sorted(glob.glob(d + "/*.json.gz") + glob.glob(d + "/*.json"))
    for f in files:
        nm_ = os.path.basename(f).replace(".json.gz", "").replace(".json", "")
        if nm_ in ("baseline", "capture") or nm_.startswith("s7_") or not os.path.exists(f): continue
        try: A = load(f)
        except Exception as ex: continue
        acc = {}
        for w, v in A.items():
            if w not in base or w not in DOM or not isinstance(v, dict) or "tk" not in v: continue
            ta, tb = v["tk"], base[w]["tk"]
            if len(ta) != len(tb): continue
            a0 = int(base[w]["a"])
            for j in range(len(ta)):
                c = CLS.get((w, a0 + j))
                if c is None: continue
                fl = int(ta[j][0] != tb[j][0])
                for key in ("all", c, "nontie" if c != "tie" else None):
                    if key: x = acc.setdefault((DOM[w], key), [0, 0]); x[0] += fl; x[1] += 1
        if not acc: continue
        g = lambda k: " / ".join((f"{100 * acc[(dm, k)][0] / acc[(dm, k)][1]:.2f}" if (dm, k) in acc else "-") for dm in ("code", "ifeval", "multi"))
        lines.append(f"| {nm_} | {g('all')} | {g('nontie')} | {g('tie')} | {g('sure')} | {g('mid')} |")
# the uniform small-perturbation arms of floor_rows itself
for k in ("att_q4", "gdn_q4", "hc_q4", "topk5", "topk2"):
    acc = {}
    for r in fr:
        fl = r["flip"] if isinstance(r["flip"], dict) else ast.literal_eval(r["flip"]); c = r["cls"]
        for key in ("all", c, "nontie" if c != "tie" else None):
            if key: x = acc.setdefault((r["dom"], key), [0, 0]); x[0] += bool(fl[k]); x[1] += 1
    g = lambda kk: " / ".join(f"{100 * acc[(dm, kk)][0] / acc[(dm, kk)][1]:.2f}" for dm in ("code", "ifeval", "multi"))
    lines.append(f"| uniform {k} (floor_rows) | {g('all')} | {g('nontie')} | {g('tie')} | {g('sure')} | {g('mid')} |")
out = "\n".join(lines); print(out); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wall_class_flips.out"), "w").write(out + "\n")
