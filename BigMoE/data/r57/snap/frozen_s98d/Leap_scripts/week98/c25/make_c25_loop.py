"""make_c25_loop.py (Leap, Mon 16 Nov 2026 12:4x; CHECK25): builds c25_loop.py = a COPY of BigMoE's r57_screen_fast.py (read
only; md5 pinned below) with E14's test-mode hooks pointed at c25_engine: OUT is ~/Leap/data/week98/c25, and under
R57_G32TEST + R57_G32TIME each window goes to c25_engine.window() (the lossy 32-row arms) instead of the screen's own
bitwise / time mode; c25_engine.finish() at the end. Anchored replacements, each must match exactly once.
NOTE: the pinned source md5 bd954d13 was BigMoE's UNCOMMITTED working copy at Mon 12:4x (it changed minutes later:
the R57_INT8D wiring), so c25_loop.py as committed is the build of record; a rebuild needs a new pin and an anchor check."""
import hashlib, os
SRC = os.path.expanduser("~/BigMoE/scripts/r57_screen_fast.py")
DST = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts/week98/c25/c25_loop.py")
MD5 = "bd954d13dae28108a62eb288c41113dc"
s = open(SRC).read()
md = hashlib.md5(s.encode()).hexdigest()
assert md == MD5, f"r57_screen_fast.py changed ({md}): re-check the anchors"
R = []
R.append(("import ctypes, json, os, sys, threading, time\n",
          "# C25 (Leap, Mon 16 Nov 2026 12:4x; CHECK25, ~/Leap/data/week98/check25_plan.md): a COPY of BigMoE's\n"
          f"# ~/BigMoE/scripts/r57_screen_fast.py (md5 {MD5}, read-only source) with E14's test-mode hooks: OUT is\n"
          "# ~/Leap/data/week98/c25, and under R57_G32TEST + R57_G32TIME each window goes to Leap's c25_engine (the lossy\n"
          "# 32-row arms). Nothing else differs from the source (scripts/week98/c25/make_c25_loop.py builds it).\n"
          "import os as _c25os, sys as _c25sys\n"
          "_c25sys.path.insert(0, _c25os.path.expanduser(\"/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts/week98/c25\"))\n"
          "_c25sys.path.insert(1, _c25os.path.expanduser(\"~/BigMoE/scripts\"))\n"
          "import ctypes, json, os, sys, threading, time\n"))
R.append(('OUT = os.path.expanduser("~/BigMoE/data/r57"); os.makedirs(OUT, exist_ok=True)\n',
          'OUT = os.path.expanduser("~/Leap/data/week98/c25"); os.makedirs(OUT, exist_ok=True)   # C25: never ~/BigMoE\n'))
R.append(('''        S0 = CS.snapshot_untrimmable_cache_lazy(P)
        say(f"G32TEST {key}: prompt {gen}, gate_req {G}")
''', '''        S0 = CS.snapshot_untrimmable_cache_lazy(P)
        say(f"G32TEST {key}: prompt {gen}, gate_req {G}")
        if int(os.environ.get("R57_G32TIME", "0") or 0):    # C25: the lossy 32-row arms replace the screen's own modes
            import c25_engine
            c25_engine.window(globals(), key, P, S0, gids, gen, G, first=(key == os.environ["R57_G32TEST"].split(",")[0]))
            del P, S0
            mx.clear_cache()
            continue
'''))
R.append(('''        del P, S0
        mx.clear_cache()
    say(f"G32TEST verdict: ''', '''        del P, S0
        mx.clear_cache()
    if int(os.environ.get("R57_G32TIME", "0") or 0):    # C25: the report
        import c25_engine
        c25_engine.finish(globals())
    say(f"G32TEST verdict: '''))
out = s
for i, (a, b) in enumerate(R):
    n = out.count(a)
    assert n == 1, f"anchor {i}: {n} matches"
    out = out.replace(a, b)
# the source's comments cite private planning notes in places: mask every match of the private leak pattern (read from
# the gitignored release/private_patterns.json, never written here) in comment text, so the copy can be committed
import json, re
rx = re.compile(json.load(open(os.path.expanduser("~/Leap/release/private_patterns.json")))["leak"], re.I)
lines, nm = out.split("\n"), 0
for j, ln in enumerate(lines):
    if rx.search(ln):
        k = ln.find("#")
        assert k >= 0 and rx.search(ln[:k]) is None, f"line {j + 1}: a private match outside a comment"
        lines[j] = ln[:k] + rx.sub("<private notes>", ln[k:])
        nm += 1
out = "\n".join(lines)
assert rx.search(out) is None
print(f"masked {nm} comment line(s)")
compile(out, DST, "exec")
open(DST, "w").write(out)
print(f"c25_loop.py written ({len(R)} anchors), md5 {hashlib.md5(out.encode()).hexdigest()} from source {MD5}")
