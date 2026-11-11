import re, sys, csv
src = sys.argv[1]; out_tsv = sys.argv[2]; out_field = sys.argv[3]
KW = re.compile(r"union|top-?k|top-?[0-9]|\bskip|budget|capacity|\bsure\b|\bQOM|\bQ1\b|\bOEA\b|reach|sibling|\btree|P-frame|reuse|prun|merg|shared expert|router|pre-?rout|runahead|affinit|substitut|frugal|fewer expert|expert count|experts per|cold|rerout|adaptive-?k|renormal|certain|confiden|MoE-Spec|AcceptMoE|Lynx|budget|distinct", re.I)
sec = ""; rows = []
for line in open(src, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("## "):
        sec = line[3:].split(" (")[0]; continue
    if not line.startswith("- "): continue
    parts = [p.strip() for p in line[2:].split(" | ")]
    d = {"section": sec, "idea": parts[0]}
    for p in parts[1:]:
        m = re.match(r"(wall|line|exact|verdict|date|source|novelty):\s*(.*)", p)
        if m: d[m.group(1)] = m.group(2)
        elif re.match(r"\[.*\]$", p): d["sessions"] = p
        elif re.match(r"\d{4}-\d\d-\d\d", p): d["date"] = p
        elif "number" not in d and "verdict" in d: d["number"] = p
    rows.append(d)
cols = ["section","idea","wall","line","exact","verdict","number","date","source","sessions","novelty"]
with open(out_tsv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore"); w.writeheader()
    for d in rows: w.writerow(d)
sel = [d for d in rows if KW.search(d["idea"]) or KW.search(d.get("number",""))]
with open(out_field, "w") as f:
    cur = None
    for d in sel:
        if d["section"] != cur:
            cur = d["section"]; f.write(f"\n## {cur}\n")
        f.write(f"- {d['idea'][:230]} || {d.get('verdict','?')[:40]} || {d.get('number','')[:160]} || {d.get('line','')[:12]} || {d.get('date','')}\n")
from collections import Counter
print("rows parsed:", len(rows), " field-matching:", len(sel))
print("verdicts (field):", Counter((d.get("verdict","?").split(" ")[0].split("(")[0]) for d in sel).most_common(8))
print("sections (field):", Counter(d["section"] for d in sel).most_common())
