"""WEEK98 (MoonShots, Sun 15 Nov 2026): the FROZEN audit rule (tau 4 / 2 / 1 / 0.5 by depth, not re-tuned) with a different
auditor's per-row picks (rows<tag>_<set>.npz from agree_model.py), on MBPP 100 (BigMoE's cells, all 100 files) and HE+
(MTP Head's cells). Reports the bands, accepted per anchor at P = 1 / 2, and the oracle bound (head OR auditor right).
usage: ~/BigMoE/.venv/bin/python -I join_auditor.py 4b"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98"))
import chooser as CH                                      # noqa: E402

tag = sys.argv[1]
out = {"auditor": tag}
for name, cells in (("mbpp", CH.mbpp_cells()), ("heplus", CH.heplus_cells())):
    c = CH.attach_2b(cells, H(f"~/MoonShots/week98/rows{tag}_{name}.npz"), lambda t: 5)
    sel = np.ones(len(c["depth"]), bool)
    tau = np.array([CH.SCHED[int(d)] for d in c["depth"]], np.float32)
    frozen = np.where(c["m2b"] >= tau, c["r2b"], c["right"])
    oracle = c["right"] | c["r2b"]
    r = {}
    for k, ok, P in (("head", c["right"], 0), ("frozen", frozen, 1), ("oracle", oracle, None)):
        r[k] = dict(bands=CH.bands(c, ok, sel), acc_P1=CH.walk(c, sel, c["right"], ok, 1 if P != 0 else 0),
                    acc_P2=CH.walk(c, sel, c["right"], ok, 2 if P != 0 else 0))
        print(f"{name} {k:7s} {r[k]['bands']}  accepted P1 {r[k]['acc_P1']} P2 {r[k]['acc_P2']}", flush=True)
    out[name] = r
json.dump(out, open(H(f"~/MoonShots/week98/join_auditor_{tag}.json"), "w"), indent=1)
