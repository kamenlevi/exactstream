"""NUMBER CHECK helper (NewIdeas, Mon 9 Nov): for chosen sections of a restructured paper, list every number that does not
occur anywhere in an earlier version (the editor's rule: new text uses only numbers already in the paper).
usage: /usr/bin/python3 -I newtext_nums.py <new.md> <old.md> <line ranges, e.g. 1-11,59-170,547-581,693-712>"""
import re, sys
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")
new = open(sys.argv[1], encoding="utf-8").read().split("\n")
old = open(sys.argv[2], encoding="utf-8").read()
old_set = {m.group(0).lstrip("+-").rstrip(",") for m in NUM.finditer(old)}
old_ratio = set(re.findall(r"x\s?(\d+(?:\.\d+)?)", old))
rng = [tuple(int(x) for x in r.split("-")) for r in sys.argv[3].split(",")]
miss = []
for a, b in rng:
    for i in range(a, b + 1):
        l = new[i - 1]
        for m in NUM.finditer(re.sub(r"(?<![A-Za-z])x(?=\d)", "x ", l)):
            v = m.group(0).lstrip("+-").rstrip(",")
            if v not in old_set and v not in old_ratio:
                miss.append((i, v, l.strip()[:150]))
print(f"numbers in the chosen lines not found anywhere in the old version: {len(miss)}")
for i, v, l in miss:
    print(f"  l{i}: {v!r}   | {l}")
