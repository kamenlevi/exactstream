"""CA report (Leap, Mon 2026-10-26; CPU only): the anatomy of an exact resident check from ca_anat.py's runs.
Inputs: data/ca_intervals.xml (xctrace export of the metal-gpu-intervals table of the prof run; one interval per
command buffer), data/ca_labels.json (prof: the labels of every command buffer, in order), data/ca_time.json (time:
A / B walls per config, the drafter's 1-step draft ms).
Each prof block (one config x B2 / P) is the run of GPU intervals between two >= 0.2 s gaps whose count matches the
block's labels (prefill runs never match). P's buffers are assigned to labels in order. Per check: GPU ms per part
class; B2: GPU busy per check. The 12-layer slice (9 GDN + 3 QSA) scales to 48 layers x4, except the one PLE layer, the
embedding, the head and the acceptance read (x1).
Usage: [BigMoE venv] ca_report.py [intervals.xml] -> ~/Leap/data/ca_report.txt"""
import json, os, sys, collections
import xml.etree.ElementTree as ET
import numpy as np

D = os.path.expanduser("~/Leap/data")
lab = json.load(open(f"{D}/ca_labels.json"))
tim = json.load(open(f"{D}/ca_time.json")) if os.path.exists(f"{D}/ca_time.json") else None
PID, KINDS = str(lab["pid"]), lab["kinds"]
out = []
def say(s):
    print(s, flush=True); out.append(s)

ids, iv = {}, []
XML = sys.argv[1] if len(sys.argv) > 1 else f"{D}/ca_intervals.xml"
for row in ET.parse(XML).getroot().iter("row"):
    vals = []
    for el in row:
        for sub in el.iter():
            if "id" in sub.attrib:
                ids[sub.attrib["id"]] = sub
        vals.append(ids[el.attrib["ref"]] if "ref" in el.attrib else el)
    proc = vals[10]
    pid = proc.find("pid")
    if pid is None and "ref" in proc.attrib:
        pid = ids[proc.attrib["ref"]].find("pid")
    pid = ids[pid.attrib["ref"]] if pid is not None and "ref" in pid.attrib else pid
    if vals[2].text == "Compute" and vals[5].text == "0" and pid is not None and pid.text == PID:
        a = int(vals[0].text)
        iv.append((a, a + int(vals[1].text)))
iv.sort()
runs, cur = [], [iv[0]]
for x in iv[1:]:
    if x[0] - cur[-1][1] > 2e8:
        runs.append(cur); cur = []
    cur.append(x)
runs.append(cur)
say(f"CA report: {len(iv)} GPU intervals of pid {PID} in {len(runs)} runs (split at >= 0.2 s gaps); run sizes "
    f"{[len(r) for r in runs]}")

def classify(layer, part, label):
    if layer < 0:
        return {"commit": "commit replay", "embed": "embed", "accept": "accept read"}.get(part, "head")
    if part == "ple":
        return "dense" if label.startswith("qmm") else "PLE"
    if part.startswith("hc"):
        return "dense" if label.startswith("qmm") else "HC mixers"
    if part == "moe":
        if label in ("exp_gu", "exp_down", "pre_exp_down") or label.startswith("gqmm"):
            return "experts"
        if label in ("pre_route_sort", "route_sort", "pre_exp_gu"):
            return "routing"
        if label.startswith("qmm"):
            return "dense"
        return "MoE combine"
    if label.startswith("qmm"):
        return "dense"
    return "attention core" if part == "qsa" else "GDN core"

ONCE = {"embed", "head", "accept read", "PLE"}          # not per layer: x1 at 48 layers (the rest x4)
CLASSES = ["dense", "experts", "HC mixers", "PLE", "attention core", "GDN core", "routing", "MoE combine",
           "commit replay", "embed", "head", "accept read"]

ri, res = 0, {}
for blk in lab["blocks"]:
    labs, nw = blk["labels"], blk["n_warm"]
    per = [len(l) for l in labs] if blk["variant"] == "P" else [3] * len(labs)
    n = sum(per)
    while ri < len(runs) and len(runs[ri]) not in (n, n + 1):
        ri += 1
    key = (blk["ctx"], blk["rows"], blk["variant"])
    if ri == len(runs):
        say(f"  {key}: NO MATCHING RUN ({n} buffers expected)"); ri = 0; continue
    run, ri, k0 = runs[ri], ri + 1, 0
    checks = []
    for k, m in enumerate(per):
        bufs = run[k0:k0 + m]; k0 += m
        if k < nw:
            continue
        dur = [(z - a) / 1e6 for a, z in bufs]
        c = dict(busy=sum(dur), span=(bufs[-1][1] - bufs[0][0]) / 1e6, n=m)
        if blk["variant"] == "P":
            cls, seg = collections.Counter(), []
            for (layer, part, label), d in zip(labs[k], dur):
                cl = classify(layer, part, label)
                cls[cl] += d
                seg.append(d)
            c["cls"], c["seg"] = cls, seg
        checks.append(c)
    res[key] = dict(checks=checks, wall=float(np.median(blk["walls"][nw:])), ple=blk.get("ple_ms", 0.0))

for P in sorted({k[0] for k in res}):
    for R in sorted({k[1] for k in res if k[0] == P}):
        b2, p = res.get((P, R, "B2")), res.get((P, R, "P"))
        if not (b2 and p):
            continue
        busy_b2 = float(np.median([c["busy"] for c in b2["checks"]]))
        busy_p = float(np.median([c["busy"] for c in p["checks"]]))
        segs = np.concatenate([c["seg"] for c in p["checks"]])
        floor = float(np.median(np.sort(segs)[:max(10, len(segs) // 20)]))   # ~a marker-only buffer
        nb = float(np.median([c["n"] for c in p["checks"]]))
        busy_pc = busy_p - (nb - 3) * floor
        cls = {cl: float(np.mean([c["cls"].get(cl, 0.0) for c in p["checks"]])) for cl in CLASSES}
        # marker-only floor spread over each class by its buffer count is small; report raw class times
        once = sum(cls[cl] for cl in ONCE)
        s48 = {cl: cls[cl] * (1 if cl in ONCE else 4) for cl in CLASSES}
        busy48 = 4 * (busy_b2 - once) + once
        t = next((x for x in (tim or {}).get("checks", []) if x["ctx"] == P and x["rows"] == R), None)
        say(f"\nctx {P}, {R} rows (12-layer slice; medians over {len(p['checks'])} checks):")
        say(f"  B2 (async every 4 layers, 3 buffers): GPU busy {busy_b2:.2f} ms, wall {b2['wall']:.2f} ms")
        say(f"  P (one buffer per part, {nb:.0f} buffers): GPU busy {busy_p:.2f} ms raw, {busy_pc:.2f} ms less "
            f"{nb - 3:.0f} x the {1e3 * floor:.0f} us marker floor; wall {p['wall']:.2f} ms; parts / B2 busy = "
            f"{busy_pc / busy_b2:.3f} (raw {busy_p / busy_b2:.3f}); PLE staging (CPU) {p['ple']:.2f} ms")
        if t:
            say(f"  time run: A eager wall {t['A']:.2f} ms, B wall {t['B']:.2f} ms (tokens B == A {t['same']}); PLE "
                f"staging {t['ple_A']:.2f} ms")
        say(f"  48 layers: GPU busy {busy48:.1f} ms (B2, slice x4 with the per-check parts once)")
        if t:
            ple_cpu = t["ple_A"]
            a48 = 4 * (t["A"] - ple_cpu - once) + ple_cpu + once
            bb48 = 4 * (t["B"] - ple_cpu - once) + ple_cpu + once
            say(f"  48 layers: A eager wall {a48:.1f} ms -> GPU idle {a48 - busy48:.1f} ms; B wall {bb48:.1f} ms -> idle "
                f"{bb48 - busy48:.1f} ms")
        tot = sum(s48.values())
        say(f"  48-layer parts (GPU ms, P's buffers; sum {tot:.1f}):")
        for cl in CLASSES:
            say(f"    {cl:15s} {s48[cl]:6.2f} ms  ({100 * s48[cl] / tot:4.1f} %)")
        other = sum(s48[cl] for cl in ("attention core", "GDN core", "routing", "MoE combine", "commit replay",
                                       "embed", "accept read"))
        say(f"  'other' on the GPU (attention core + GDN core + routing + combine + commit + embed + accept): "
            f"{other:.1f} ms; dense + HC + PLE + head: {s48['dense'] + s48['HC mixers'] + s48['PLE'] + s48['head']:.1f} "
            f"ms; experts {s48['experts']:.1f} ms")
if tim and tim.get("draft"):
    say("\ndrafting (DFlash2 L1-20, 1-step, refine19's path incl. the per-position selector loop): " +
        ", ".join(f"block {k} {v:.2f} ms" for k, v in sorted(tim["draft"].items(), key=lambda x: int(x[0]))))
open(f"{D}/ca_report.txt", "w").write("\n".join(out) + "\n")
