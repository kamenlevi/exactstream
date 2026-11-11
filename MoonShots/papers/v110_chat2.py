"""v1.1.0, after v110_chat.py: the shipped build's own real-engine check ran (Leap cc110g.sh, Wed 03:45-03:57; recorded in
Leap 1692627: data/w24_pkg.md, data/idle_ms_1111.md, data/w24_pkg/candidate_110g/checks_0345/), so RELEASE's chat.py /
serve.py part names it instead of the first build's check. Re-derived by the editor from those raw files:
- clean_clone_run.log (pkg_110g clone v110g): lossy-24 and chat-24 2/2 keys prefix-equal, closure 0 / 0; chat.py --mem 24
  rc 0, 71 tokens, lifetime peak 23.63 GB; serve.py --mem 24: OpenAI [DONE] 1 + finish_reason 1, messages stream
  message_stop 1 + 4 text deltas, stopped on SIGTERM; CLEAN-CLONE PASS 03:54:15.
- runner.log: the packaged defaults (no variable set) idle at 1.6 % (serve.py) and 1.5 % (chat.py) of one core at 24 GB.
- dec_32_{1B..8B}.out: 128.8, 129.3, 5.6, 135.1, 131.5, 133.8, 135.0, 134.2 tok/s; the tokens files they name hash equal
  to each other and to the 01:5x runs' (7f8ad667a246). Run 3 (arm A) overlaps fanwatch's heat freeze 03:48:21-03:49:21
  (fanwatch_0347.txt) and is left out. Pooled with the 01:5x runs: A 130.0, 131.9, 129.3, 133.8, 135.0 (mean 132.0);
  B 128.6, 129.7, 128.8, 135.1, 131.5, 134.2 (mean 131.3); B/A 0.995.
usage: python3 -I v110_chat2.py [--write]   (backups: ~/Papers/.v110_chat2_bak_<time>/)"""
import pathlib
import re
import sys
import time

P = pathlib.Path.home() / "Papers"
R = {"RELEASE.md": [
    ("  and 1.5-1.6 % of one with it, at 24 and 32 GB [K, LP:data/agent_cc_1111.md; LP:data/idle_ms_1111.md]. Decode speed\n"
     "  changed by x1.002 at 24 GB and x0.986 at 32 GB, within the spread between two runs of the same setting (up to 0.4\n"
     "  and 1.5 %), and the tokens stayed the same (three HumanEval prompts, four runs per line) [K,\n"
     "  LP:data/idle_ms_1111.md].\n",
     "  and 1.5-1.6 % of one with it, at 24 and 32 GB [K, LP:data/agent_cc_1111.md; LP:data/idle_ms_1111.md]; the shipped\n"
     "  build, with its own defaults, idled at 1.6 % (serve.py) and 1.5 % (chat.py) at 24 GB [K, LP:data/w24_pkg.md]. Decode\n"
     "  speed did not measurably change: x1.002 at 24 GB (two runs per arm) and x0.995 at 32 GB (five and six runs; one more\n"
     "  ran through a heat pause and is left out), with the same tokens in every run (three HumanEval prompts) [K,\n"
     "  LP:data/idle_ms_1111.md].\n"),
    ("- Checks: in the clean-clone test of the first build that held them (10 Nov 20:57-20:59), chat.py answered a message at\n"
     "  24 GB (lifetime peak 23.63 GB), and serve.py at 24 GB answered a streamed request in each format and stopped on\n"
     "  SIGTERM [K, LP:data/w24_pkg/candidate_110c/report.txt]. chat.py's later changes (the paste handling, `/file`, the\n"
     "  system line, the window title) passed a terminal test against a stand-in for the engine (22 of 22 checks) and two\n"
     "  runs on the real engine at 32 GB [K, LP:data/chatpy_fixes_1110.md].\n",
     "- Checks: in the clean-clone test of the shipped build (11 Nov 03:52-03:54), lossy-24 and chat-24 gave their\n"
     "  references' tokens with every module and opened file inside the allowed folders, chat.py answered a message at\n"
     "  24 GB (lifetime peak 23.63 GB), and serve.py at 24 GB answered a streamed request in each format and stopped on\n"
     "  SIGTERM [K, LP:data/w24_pkg.md; LP:data/w24_pkg/candidate_110g/checks_0345/clean_clone_run.log].\n"
     "  chat.py's interactive parts (pastes, `/file`, the window title) passed a terminal test against a stand-in for the\n"
     "  engine, 22 checks of 22, and chat.py ran twice on the real engine at 32 GB [K, LP:data/chatpy_fixes_1110.md].\n"),
    ("- chat.py and serve.py (1.1.0): the code LP:release/src/chat.py and LP:release/src/serve.py (LP d37fb68); the first\n"
     "  build's clean-clone check LP:data/w24_pkg/candidate_110c/report.txt; chat.py's later changes and their tests\n",
     "- chat.py and serve.py (1.1.0): the code LP:release/src/chat.py and LP:release/src/serve.py (LP d37fb68); the shipped\n"
     "  build's checks LP:data/w24_pkg.md and LP:data/w24_pkg/candidate_110g/checks_0345/ (the clean-clone run, the eight\n"
     "  decode runs at 32 GB, the idle defaults and the heat-pause log); chat.py's changes and their tests\n"),
]}

MARKER = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s")
bad, out = 0, {}
for f, reps in R.items():
    t = (P / f).read_text()
    for o, w in reps:
        c = t.count(o)
        if c != 1:
            bad += 1; print(f"BAD x{c} {f}: {o.splitlines()[0][:90]!r}"); continue
        for ln in w.split("\n")[1:]:
            if ln.strip() and MARKER.match(ln) and not ln.startswith("- "):
                bad += 1; print(f"BAD marker {f}: {ln[:90]!r}")
        t = t.replace(o, w); print(f"OK  {f}: {o.splitlines()[0][:90]}")
    out[f] = t
for f, t in out.items():
    for w in ("Anthropic", "BIGMOE", "Leap "):
        for i, ln in enumerate(t.split("\n"), 1):
            if w in ln and not ln.startswith("- Leap"):
                print(f"LINT? {f}:{i}: {ln.strip()[:100]}")
if "--write" in sys.argv and not bad:
    bak = P / f".v110_chat2_bak_{time.strftime('%H%M')}"; bak.mkdir(exist_ok=True)
    for f, t in out.items():
        (bak / f).write_text((P / f).read_text()); (P / f).write_text(t)
    print("written; backups in", bak)
print("bad", bad)
