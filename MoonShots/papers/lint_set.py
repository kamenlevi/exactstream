"""lint_set.py - STYLE.md checks over the paper set (read-only). usage: python3 -I lint_set.py [paper.md ...]"""
import re, sys, pathlib
P = pathlib.Path.home()/"Papers"
FILES = sys.argv[1:] or [str(P/f) for f in ("MAIN.md","KERNELS.md","SMALL_RAM.md","SMALL_RAM_parts_breakthrough.md",
                                            "DRAFTER.md","EVALUATION.md","RELEASE.md")]
CHECKS = [
 ("never-word", r"(?i)\b(lossless|near[- ]exact|same quality|fastest|world'?s first|state[- ]of[- ]the[- ]art|SOTA|first to)\b"),
 ("4-bit", r"(?i)\b4[- ]bit\b"),
 # a folder name inside a path into the public snapshot (BigMoE/data/...) is allowed (orchestrator 14:3x)
> (a quote from the project owner, kept private)
 ("home path", r"~/[A-Za-z]|/Users/"),
 ("TBD", r"\[\[TBD"),
 ("info-only metric", r"(?i)\b(steady|tpsw|full[- ]period)\b"),
 ("personal / told", r"(?i)(we were told|zl@|blackpeak|\bhe said\b)"),
 ("MLX version", r"(?i)\b0\.32\.\d\b|#4632"),
 ("stale date", r"(?i)\b(Tue(sday)? \d|Wed(nesday)?\b|Mon 22:00|Tue 12:00|Tue 20:00)"),
 ("GiB", r"\bGiB\b"),
]
# Public edition (Mon 9 Nov 13:07 decision): the author block comes from ~/Papers/AUTHORS.txt (kept out of this repo);
# the author's own lines (the block and companion-paper references) are exempt from the session/account-name check.
AUTH = (P/"AUTHORS.txt").read_text().strip().split("\n") if (P/"AUTHORS.txt").exists() else []
CHECKS.append(("placeholder", r"\[\[(?!TBD)"))
for f in FILES:
    L = pathlib.Path(f).read_text().splitlines()
    txt = "\n".join(L)
    print(f"\n### {pathlib.Path(f).name}: {len(L)} lines")
    own = lambda l: bool(AUTH) and (l.strip() == AUTH[0] or l.startswith("- [") and f"] {AUTH[0]}. " in l)
    for name, pat in CHECKS:
        hits = [(i+1, l) for i, l in enumerate(L) if re.search(pat, l) and not (name == "session/account name" and own(l))]
        if not hits: continue
        ln = ",".join(str(i) for i, _ in hits[:25]) + (" ..." if len(hits) > 25 else "")
        print(f"- {name}: {len(hits)} lines: {ln}")
        if name in ("never-word", "personal / told", "stale date", "info-only metric", "MLX version", "4-bit"):
            for i, l in hits[:6]:
                m = re.search(pat, l); s = max(0, m.start()-60)
                print(f"    l{i}: ...{l[s:m.end()+60]}...")
    req = {"left-on-table section": r"(?i)what we left on the table",
           "GB = 10^9 stated": r"10\^9|10⁹|1e9|10\*\*9",
           "self-reported label": r"(?i)self-reported",
           "Strata credit (MIT + URL)": r"MIT.{0,40}github\.com/Niko1221/Strata|github\.com/Niko1221/Strata.{0,40}MIT",
           }
    if AUTH:
        req["author block"] = re.escape(AUTH[0]) + r"\n\n" + re.escape(AUTH[1] if len(AUTH) > 1 else "")
        req["Use of AI tools section"] = r"(?m)^## Use of AI tools$"
        req["References section, last"] = r"(?ms)^## References$(?!.*^## )"
    else:
        req["[[AUTHORS]]"] = r"\[\[AUTHORS\]\]"
    miss = [k for k, pat in req.items() if not re.search(pat, txt)]
    if "Strata" not in txt: miss = [m for m in miss if not m.startswith("Strata")]
    if not re.search(r"(?i)splash|kapitsa|slotstream|strata", txt): miss = [m for m in miss if m != "self-reported label"]
    print("- MISSING:", ", ".join(miss) if miss else "none")
