"""Merge the sessions' TRIED parts (~/TRIED/<session>.md) into ~/TRIED/TRIED.md (Kamen, Sun 1 Nov 2026).
One idea per line in the README shape: idea | wall | line | exact | verdict | number | date | source.
Ideas with the same ID (the first word before ':') from DIFFERENT sessions are merged into one line that keeps every
session's verdict and source. A session's own 2nd, 3rd ... row under one ID (sub-ideas such as M-29's four results) stays
its own line, keyed ID#2, ID#3 ... (the key novelty.tsv uses for it).
Usage: python3 -I tried_merge.py [~/TRIED]"""
import collections, os, re, sys

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/TRIED")
SESSIONS = ["leap", "bigmoe", "evalsuite", "breakthrough", "moonshots", "fit12", "newideas", "mtphead", "orchestrator"]
WALLS = ["SSD", "bytes", "fixed-cost", "drafter", "quality", "memory", "prefill", "exactness", "other"]
VERDICTS = ["ADOPTED", "PASS", "BETWEEN", "PARKED", "KILL"]
SYN = {"ssd": "SSD", "bytes": "bytes", "fixed-cost": "fixed-cost", "fixed": "fixed-cost", "check": "fixed-cost",
       "kernel": "fixed-cost", "kernels": "fixed-cost", "drafter": "drafter", "tokens": "drafter", "quality": "quality",
       "memory": "memory", "prefill": "prefill", "exactness": "exactness", "other": "other"}


def wall_of(f):
    w = re.sub(r"^\s*wall:\s*", "", f, flags=re.I).strip().lower()
    w = re.split(r"[ /,;(+]", w)[0] if w else ""
    return SYN.get(w, "other")


def verdict_of(f):
    up = f.upper()
    for v in ["ADOPTED", "BETWEEN", "PARKED", "PASS", "KILL"]:
        if v in up:
            return v
    return "?"


rows, counts, bad = [], collections.Counter(), collections.Counter()
seen = collections.Counter()  # (session, ID) -> rows so far: a session's 2nd+ row under one ID is a distinct sub-idea
for s in SESSIONS:
    p = os.path.join(D, f"{s}.md")
    if not os.path.exists(p):
        continue
    for n, line in enumerate(open(p, encoding="utf-8"), 1):
        if not line.startswith("- "):
            continue
        parts = [x.strip() for x in line[2:].rstrip("\n").split(" | ")]
        if len(parts) < 6:
            bad[s] += 1
            continue
        parts += [""] * (8 - len(parts))
        idea = parts[0]
        key = re.split(r"[:\s]", idea, 1)[0].upper().rstrip(".,")
        seen[(s, key)] += 1
        if seen[(s, key)] > 1:  # (Tue 3 Nov fix: 270 rows under 97 shared IDs had collapsed into their first row's text)
            key = f"{key}#{seen[(s, key)]}"
        rows.append(dict(session=s, n=n, idea=idea, key=key, wall=wall_of(parts[1]), fields=parts[1:8],
                         verdict=verdict_of(parts[4])))
        counts[s] += 1

groups = collections.OrderedDict()
for r in rows:
    groups.setdefault(r["key"], []).append(r)

NOV = {}  # novelty.tsv (novelty_tags.py): key -> "tag (prior art)" for ADOPTED / PASS levers
npath = os.path.join(D, "novelty.tsv")
if os.path.exists(npath):
    for line in open(npath, encoding="utf-8"):
        if line.startswith("# ") or "\t" not in line:  # keys like "#18" are data, not comments
            continue
        k, t, note = (line.rstrip("\n").split("\t") + ["", ""])[:3]
        NOV[k] = t + (f" ({note})" if note else "")

merged = []
for key, rs in groups.items():
    first = rs[0]
    f = list(first["fields"])
    if len(rs) > 1:
        verdicts = sorted({r["verdict"] for r in rs})
        if len(verdicts) > 1:
            f[3] = f[3] + " (also " + "; ".join(f"{r['session']}: {r['fields'][3]}" for r in rs[1:]) + ")"
        f[6] = "; ".join(r["fields"][6] or f"~/TRIED/{r['session']}.md:{r['n']}" for r in rs)
    tag = "+".join(sorted({r["session"] for r in rs}))
    nov = f" | novelty: {NOV[key]}" if key in NOV else ""
    merged.append(dict(wall=first["wall"], verdict=first["verdict"], key=key,
                       text=f"- {first['idea']} | " + " | ".join(f[:7]) + f" | [{tag}]" + nov))

vorder = {v: i for i, v in enumerate(VERDICTS + ["?"])}
out = ["# TRIED: every idea this project has tried, merged from the sessions' parts (NewIdeas keeps it current)",
       "Check here before proposing or screening an idea. Line shape: idea | wall | line | exact | verdict | the number "
       "that decided it | date | source | [sessions]. Same-ID ideas from several sessions are merged; every source kept.",
       "Parts: " + ", ".join(f"{s} {counts[s]}" for s in SESSIONS if counts[s]) + f"; merged ideas {len(merged)}.",
       "ADOPTED / PASS levers end in 'novelty: published / adapted / new / engineering (prior art)' from novelty.tsv; "
       "the checks and sources are in NOVELTY.md.", ""]
for w in WALLS:
    ws = sorted([m for m in merged if m["wall"] == w], key=lambda m: (vorder.get(m["verdict"], 9), m["key"]))
    if not ws:
        continue
    vc = collections.Counter(m["verdict"] for m in ws)
    out.append(f"## {w} ({len(ws)}: " + ", ".join(f"{v} {vc[v]}" for v in VERDICTS + ['?'] if vc[v]) + ")")
    out += [m["text"] for m in ws]
    out.append("")
open(os.path.join(D, "TRIED.md"), "w", encoding="utf-8").write("\n".join(out))

vc = collections.Counter(m["verdict"] for m in merged)
wc = collections.Counter(m["wall"] for m in merged)
print("parts:", dict(counts), "unparsed:", dict(bad))
print("merged ideas:", len(merged), "from", len(rows), "lines")
print("verdicts:", dict(vc))
print("walls:", dict(wc))
print("missing parts:", [s for s in SESSIONS if not os.path.exists(os.path.join(D, f"{s}.md"))])
