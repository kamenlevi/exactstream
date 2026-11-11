"""SMALL_RAM release, set-wide part (staged 10 Nov 2026; run only on Kamen's go). SMALL_RAM itself is already filled.
1. MAIN and EVALUATION stop calling [SMALL-RAM] forthcoming (5 phrases).
2. Every paper's generated tail is rebuilt by make_refs with no FORTHCOMING entry, so [SMALL-RAM]'s reference reads
   "2026. A companion paper of this set." like the others.
3. With --write on the real folder, refs.py's FORTHCOMING is emptied too, so later make_refs runs agree.
The build's publish set (~/Papers/public/src/publish_set.txt) must also list SMALL_RAM; that file is the build's.
usage: python3 -I release_smallram.py [--dir DIR] [--write]     (DIR defaults to ~/Papers; a copy gives a dry run)
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refs, make_refs

args = sys.argv[1:]
D = os.path.join(os.path.abspath(os.path.expanduser(args[args.index("--dir") + 1])), "") if "--dir" in args \
    else os.path.expanduser("~/Papers/")
WRITE = "--write" in args
REAL = os.path.realpath(D) == os.path.realpath(os.path.expanduser("~/Papers"))

TEXT = {
    "MAIN": [
        ("the subject of a forthcoming companion paper ([SMALL-RAM], by 11 November 2026); no number in this paper depends on it.",
         "the subject of a companion paper ([SMALL-RAM]); no number in this paper depends on it."),
        ("The 24 GB line's rows and its lossy chat route are in the forthcoming [SMALL-RAM].",
         "The 24 GB line's rows, among them a gated lossy configuration, and its lossy chat route are in [SMALL-RAM]."),
        ("(details in the forthcoming [SMALL-RAM])", "(details in [SMALL-RAM])"),
        ("- **11 November 2026 (v1.1.0).** PF-B's difference", "- **11 November 2026 (v1.1.0).** [SMALL-RAM] is published. PF-B's difference"),
    ],
    "EVALUATION": [
        ("a forthcoming\n  companion paper ([SMALL-RAM], by 11 November 2026) for the 24 / 12 GB loops.",
         "a\n  companion paper ([SMALL-RAM]) for the 24 / 12 GB loops."),
        ("(described in the forthcoming [SMALL-RAM])", "(described in [SMALL-RAM])"),
    ],
}

bad = 0
for name, reps in TEXT.items():
    path = D + name + ".md"
    t = open(path).read()
    for o, w in reps:
        c = t.count(o)
        if c != 1:
            bad += 1; print(f"BAD x{c} {name}: {o[:80]!r}"); continue
        t = t.replace(o, w)
    left = [l for l in t.split("\n") if re.search(r"(?i)forthcoming", l) and "SMALL-RAM" in l]
    print(f"{name}: {len(reps)} phrase(s); 'forthcoming' + [SMALL-RAM] left outside the tail: "
          f"{len([l for l in left if not l.startswith('- [SMALL-RAM]')])}")
    if WRITE and not bad:
        open(path, "w").write(t)

if bad:
    sys.exit(f"{bad} phrase(s) did not match; nothing regenerated")
make_refs.P = D
refs.FORTHCOMING = {}
for n in make_refs.FILES:
    old, new, rep = make_refs.build(n)
    print(n, "tail", "changed" if new != old else "same", {k: rep[k] for k in ("companions", "unknown", "uncovered")})
    if WRITE and new != old:
        open(D + n + ".md", "w").write(new)
if WRITE and REAL:
    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs.py")
    s = open(rp).read()
    s2 = s.replace('FORTHCOMING = {"SMALL-RAM": "by 11 November 2026"}', "FORTHCOMING = {}  # SMALL_RAM released")
    if s2 != s:
        open(rp, "w").write(s2); print("refs.py: FORTHCOMING emptied")
print("written to " + D if WRITE else "dry run (no --write)")
