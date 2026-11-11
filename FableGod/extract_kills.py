import re, sys, collections
src = "/Users/kamen/TRIED/TRIED.md"
wall = None
rows = []
for ln in open(src, encoding="utf-8"):
    if ln.startswith("## "):
        wall = ln[3:].split(" (")[0].strip(); continue
    if not ln.startswith("- "): continue
    parts = [p.strip() for p in ln[2:].split(" | ")]
    d = {"wall": wall, "idea": parts[0], "raw": ln.rstrip("\n")}
    for p in parts[1:]:
        if ":" in p:
            k, v = p.split(":", 1); d.setdefault(k.strip(), v.strip())
    rows.append(d)
def verdict(d): return (d.get("verdict") or "")
def line(d): return (d.get("line") or "")
def date(d):
    m = re.search(r"(2026-)?(\d{2})-(\d{2})", d.get("raw", ""))
    return m.group(0) if m else ""
old = [d for d in rows if re.search(r"KILL|PARKED", verdict(d)) and (re.search(r"\b(24|12)\b", line(d)) or "other" in line(d) or line(d)=="all")]
early45 = [d for d in rows if re.search(r"KILL|PARKED", verdict(d)) and re.search(r"\b45\b", line(d))]
def dump(fn, lst):
    with open(fn, "w", encoding="utf-8") as f:
        for d in lst: f.write(d["raw"] + "\n")
dump("kills_old_lines.txt", old); dump("kills_45.txt", early45)
adopted45 = [d for d in rows if "ADOPTED" in verdict(d) and re.search(r"\b45\b", line(d))]
dump("adopted_45.txt", adopted45)
c = collections.Counter(d["wall"] for d in old); c2 = collections.Counter(d["wall"] for d in early45)
print("total rows", len(rows)); print("old-line KILL/PARKED", len(old), dict(c)); print("45-line KILL/PARKED", len(early45), dict(c2)); print("adopted45", len(adopted45))
