"""CA run 3 report: data/ca_diff.json (12-layer slice, 9 GDN + 3 QSA) scaled to 48 layers: per-layer parts x4; L0
(embedding, head, acceptance read, sync) and the PLE layer once. Walls exclude each check's PLE n-gram staging (CPU,
before any GPU work, 0.7-3.5 ms from the file-backed table: it halves the paired noise), reported on its own.
Usage: [BigMoE venv] ca_diff_report.py -> data/ca_diff_report.txt"""
import json, os
import numpy as np
D = os.path.expanduser("~/Leap/data")
J = json.load(open(os.path.join(D, "ca_diff.json")))
LAY = ["experts", "hc", "dense_gdn", "attn", "dense_moe", "gdn_rec", "gather"]
out = []
def say(s):
    print(s); out.append(s)
say("48-layer estimates (ms) per exact resident check: MTPLX kernels, 4-bit experts at top-10, walls less PLE staging")
say("ctx rows | engine A | D | parts sum | D-all | commit | PLE stage | L0 | " + " | ".join(LAY)
    + " | dense_qsa (in attn) | head (in L0) | ple layer | resid | paired sd (12 layers)")
for o in J["out"]:
    raw = o["raw"]
    w = lambda v: np.array([c["wall"] - c["ple"] for c in raw[v]])
    d = {v: float(np.median(w("D") - w(v))) for v in LAY + ["dense_qsa", "head", "ple", "all"]}
    L0, Dm, Am = (float(np.median(w(v))) for v in ("L0", "D", "A"))
    x48 = lambda t: L0 + 4 * (t - L0)
    parts = L0 + 4 * sum(d[c] for c in LAY) + d["ple"]
    resid = x48(Dm) - parts
    sd = np.median([np.std(w("D") - w(v)) for v in LAY])
    say(f"{o['ctx']:5d} {o['rows']:2d} | {x48(Am):6.1f} | {x48(Dm):6.1f} | {parts:6.1f} | {4 * d['all']:6.1f} | "
        f"{4 * np.median([c['commit'] for c in raw['D']]):5.1f} | {np.median([c['ple'] for c in raw['D']]):4.2f} | {L0:4.2f} | "
        + " | ".join(f"{4 * d[c]:5.1f}" for c in LAY)
        + f" | {4 * d['dense_qsa']:5.1f} | {d['head']:4.2f} | {d['ple']:5.2f} | {resid:5.1f} | {sd:.2f}")
open(os.path.join(D, "ca_diff_report.txt"), "w").write("\n".join(out) + "\n")
