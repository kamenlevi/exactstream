"""NUMBER CHECK helper (NewIdeas, Mon 9 Nov): did a relabelled table lose credit to published work?
For every Markdown table row in the chosen section of the old and new paper, collect the credit tokens of the row: names in
parentheses, capitalised names, arXiv ids, PR / issue numbers, URLs. A token present in an old row but absent from the
whole new section (and from the new paper's References) is reported as possibly lost credit.
usage: /usr/bin/python3 -I credit_check.py <old.md> <new.md> <section heading regex, e.g. '^## 8\\.'> [...more headings]"""
import re, sys

STOP = {"The", "A", "An", "On", "In", "Of", "Our", "Ours", "We", "It", "Its", "If", "And", "Or", "For", "To", "With", "As",
        "At", "By", "Not", "New", "Yes", "No", "Live", "Exact", "Lossy", "PASS", "KILL", "BETWEEN", "K", "P", "G", "MD",
        "SD", "MAC", "AGPU", "GB", "MB", "SSD", "GPU", "CPU", "MoE", "HC", "GDN", "MTP", "LRU", "OPT", "This", "That"}


def section(text, rx):
    m = re.search(rx, text, re.M)
    if not m:
        return ""
    nxt = re.search(r"^## ", text[m.end():], re.M)
    return text[m.start(): m.end() + (nxt.start() if nxt else len(text))]


def tokens(s):
    out = set()
    out |= set(re.findall(r"\b\d{4}\.\d{4,5}\b", s))                       # arXiv ids
    out |= set(re.findall(r"(?:PR|pull request|issue)\s*#\s*\d+", s, re.I))
    out |= set(re.findall(r"https?://\S+", s))
    for w in re.findall(r"\b[A-Z][A-Za-z0-9+\-]*(?:\s+[A-Z][A-Za-z0-9+\-]*)*", s):
        for part in [w] + w.split():
            if part not in STOP and len(part) > 2:
                out.add(part)
    return out


old, new = open(sys.argv[1], encoding="utf-8").read(), open(sys.argv[2], encoding="utf-8").read()
refs_new = section(new, r"^## References")
for rx in sys.argv[3:]:
    so, sn = section(old, rx), section(new, rx)
    rows_old = [l for l in so.split("\n") if l.startswith("|") and not set(l) <= set("|-: ")]
    tok_new = tokens(sn) | tokens(refs_new)
    lost = []
    for r in rows_old:
        key = [c.strip() for c in r.split("|")[1:2]]
        for t in sorted(tokens(r)):
            if t not in tok_new and t not in sn and t not in refs_new:
                lost.append((key[0][:50] if key else "?", t))
    print(f"== {rx}: old rows {len(rows_old)}; tokens not found in the new section or References: {len(lost)}")
    for k, t in lost[:60]:
        print(f"   row '{k}': {t!r}")
