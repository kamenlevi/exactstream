import csv, re, sys
KW = re.compile(r"union|top-?k|top-?[0-9]|\bskip|budget|\bsure\b|\bQOM|\bQ1\b|\bOEA\b|reach|sibling|\btree|P-frame|reuse|prun|merg|shared expert|router|pre-?rout|runahead|affinit|substitut|frugal|fewer expert|expert count|experts per|cold|rerout|adaptive-?k|renormal|certain|confiden|MoE-Spec|AcceptMoE|Lynx|distinct|wait-by-mass|mass|bonus|wasted|rejected", re.I)
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
sel = [d for d in rows if KW.search(d["idea"]) or KW.search(d.get("number") or "")]
out = open(sys.argv[2], "w")
out.write(f"\n## Appendix A. TRIED lines in this field ({len(sel)} of {len(rows)} parsed lines; keyword match on the idea or its number; source FableGod/TRIED/TRIED.md)\n")
out.write("Format: idea (ID first) | verdict | the number that decided it | line | date. Truncated to one line each; the full text is in TRIED.md.\n")
cur = None
for d in sel:
    if d["section"] != cur:
        cur = d["section"]; out.write(f"\n### {cur}\n")
    out.write(f"- {d['idea'][:150]} | **{(d.get('verdict') or '?')[:28]}** | {(d.get('number') or '')[:140]} | {(d.get('line') or '')[:10]} | {d.get('date') or ''}\n")
print(len(sel))
