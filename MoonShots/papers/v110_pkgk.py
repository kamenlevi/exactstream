"""v1.1.0: RELEASE names the build its 1.1.0 checks ran on (proposed to the orchestrator Wed 11 Nov 12:09; apply only on
its yes). The checks (cc110g: clean-clone run, idle defaults, decode at 32 GB) ran on candidate build pkg_110g; 1.1.0 ships
pkg_110k (Leap 94b6f75). Re-derived by the editor (Wed 12:0x): diff -rq of ~/Leap/release/out/pkg_110g and pkg_110k =
INSTALL.md, LIMITS.md, README.md, RESULTS.md, MANIFEST.json (leap_commit + those files' hashes) and the memory comment
lines in configs/chat-24.env and lossy-24.env; no code file; chat.py and serve.py byte-equal in both and to
release/src at d37fb68. Leap's record: LP:data/w24_pkg/candidate_110k/diff_vs_110g.txt.
Orchestrator 12:13: yes to all three, apply. 12:15: 1.1.0 is now pkg_110l (package commit fd1e02e; Leap release/src
a31794a; record Leap 567398b): only LIMITS.md (line 22, the papers' two 24 GB qualifiers) and MANIFEST.json differ from
110k; cite LP:data/w24_pkg/candidate_110l/diff_vs_110k.txt beside the 110k file. Editor's re-check (12:1x): diff -rq
pkg_110k / pkg_110l = LIMITS.md + MANIFEST.json; pkg_110g / pkg_110l = the five files above + the two config comments;
chat.py and serve.py byte-equal. Also: pkg_w24 (tested 10 Nov 10:18) / pkg_110l = CITATION.cff, CONFIGS.md, INSTALL,
LIMITS, MANIFEST, README, RESULTS, run.py (the help string), tools/clean_clone_test.sh, chat.py + serve.py (new) and the
memory comment in chat-24.env / lossy-24.env (comment only: LP:data/w24_pkg.md, pkg_110i / pkg_110j), so section 4's
"differs from that build in its documents, one help string and two new programs" gains the config comments.
usage: python3 -I v110_pkgk.py [--write]   (backups: ~/Papers/.v110_pkgk_bak_<time>/)"""
import pathlib
import re
import sys
import time

P = pathlib.Path.home() / "Papers"
R = {"RELEASE.md": []}

R["RELEASE.md"] += [
    ("  and 1.5-1.6 % of one with it, at 24 and 32 GB [K, LP:data/agent_cc_1111.md; LP:data/idle_ms_1111.md]; the shipped\n"
     "  build, with its own defaults, idled at 1.6 % (serve.py) and 1.5 % (chat.py) at 24 GB [K, LP:data/w24_pkg.md]. Whole\n",
     "  and 1.5-1.6 % of one with it, at 24 and 32 GB [K, LP:data/agent_cc_1111.md; LP:data/idle_ms_1111.md]; the\n"
     "  candidate build 110g, whose code ships unchanged in package 1.1.0, idled with its own defaults at 1.6 % (serve.py) and\n"
     "  1.5 % (chat.py) at 24 GB [K, LP:data/w24_pkg.md]. Whole\n"),
    ("- Checks: in the clean-clone test of the shipped build (11 Nov 03:52-03:54), lossy-24 and chat-24 gave their\n",
     "- Checks: in build 110g's clean-clone test (11 Nov 03:52-03:54), lossy-24 and chat-24 gave their\n"),
    ("(LP d37fb68); the shipped\n"
     "  build's checks LP:data/w24_pkg.md and LP:data/w24_pkg/candidate_110g/checks_0345/ (the clean-clone run, the eight\n",
     "(LP d37fb68); the checks\n"
     "  of build 110g, whose code ships unchanged (the final package, build 110l, differs from it only in documents, its\n"
     "  manifest and config comments: LP:data/w24_pkg/candidate_110k/diff_vs_110g.txt and\n"
     "  LP:data/w24_pkg/candidate_110l/diff_vs_110k.txt), LP:data/w24_pkg.md and\n"
     "  LP:data/w24_pkg/candidate_110g/checks_0345/ (the clean-clone run, the eight\n"),
    ("The 1.1.0 package differs from that build in its documents,\n"
     "one help string and two new programs, chat.py and serve.py,",
     "The 1.1.0 package differs from that build in its documents,\n"
     "the memory comment in its two 24 GB configs, one help string and two new programs, chat.py and serve.py,"),
]

MARKER = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s")
bad, out = 0, {}
for f, reps in R.items():
    t = (P / f).read_text()
    for o, w in reps:
        c = t.count(o)
        if c != 1:
            bad += 1; print(f"BAD x{c} {f}: {o.splitlines()[0][:90]!r}"); continue
        for ln in w.split("\n")[1:]:
            if ln.strip() and MARKER.match(ln) and not re.match(r"^\s*- ", ln) and not ln.startswith("| "):
                bad += 1; print(f"BAD marker {f}: {ln[:90]!r}")
        t = t.replace(o, w); print(f"OK  {f}: {o.splitlines()[0][:90]}")
    out[f] = t
for f, t in out.items():                                   # what the build's lint would flag, and leftovers
    for i, ln in enumerate(t.split("\n"), 1):
        if re.search(r"Anthropic|BIGMOE|\bLeap [0-9a-f]|shipped build|shipped\s*$", ln):
            print(f"CHECK {f}:{i}: {ln.strip()[:110]}")
if "--write" in sys.argv and not bad:
    bak = P / f".v110_pkgk_bak_{time.strftime('%H%M')}"; bak.mkdir(exist_ok=True)
    for f, t in out.items():
        (bak / f).write_text((P / f).read_text()); (P / f).write_text(t)
    print("written; backups in", bak)
print("bad", bad)
