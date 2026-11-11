#!/bin/bash
# RELAX wiring (docs/PREREG_relax_2026-11-02.md): move pending/relax.py, replay.py, relax_core.py into evalsuite/, add
# the `relax` / `relax-replay` stages to evalsuite/stages.py, and put "MEM <n> GB" in cli.py's share-lock reason
# (RULES.md). Run only once runner 38 has ended: its pin hashes evalsuite/*.py (a new file there stops it, exit 7).
# Idempotent: a second run finds everything in place and says so.
cd "$HOME/EvalSuite" || exit 1
if pgrep -f "tools/day_runner38.sh" >/dev/null; then echo "runner 38 is alive: not wiring"; exit 2; fi
for f in relax.py replay.py relax_core.py; do
  if [ -f "pending/$f" ]; then git mv "pending/$f" "evalsuite/$f" || exit 3; fi
  [ -f "evalsuite/$f" ] || { echo "evalsuite/$f missing"; exit 3; }
done
python3 - <<'PY'
import os, sys
p = os.path.expanduser("~/EvalSuite/evalsuite/stages.py")
s = open(p).read()
if '"relax"' in s:
    print("stages.py: relax stages already in"); sys.exit(0)
a = '''    ap.add_argument("stage", choices=["gen-tf", "prefill", "tfstep", "decode", "live16", "live16diag", "live16b"])'''
b = '''    ap.add_argument("stage", choices=["gen-tf", "prefill", "tfstep", "decode", "live16", "live16diag", "live16b", "relax",
                                      "relax-replay"])'''
c = '''    elif a.stage == "live16b":                    # R-56 on REF2b: every token + every kept row vs REF2b's AR (live16b.py)'''
d = '''    elif a.stage in ("relax", "relax-replay"):    # RELAX: quality-defined modes (relax.py) and their exact replay (replay.py)
        from . import relax, replay
        if (relax if a.stage == "relax" else replay).main(rt, st, a, log) == "yield":
            st.gate.check()
            sys.exit(75)                          # yielded to a quiet ticket (cli.gpu --share asks again later)
''' + c
e = '''    ap.add_argument("--arm", default="", help="live16 (R-56): ref2 or stock")'''
f = '''    ap.add_argument("--arm", default="", help="live16 (R-56): ref2 or stock; relax: control, A, D, C or B")'''
for x in (a, c, e):
    assert s.count(x) == 1, x[:60]
s = s.replace(a, b).replace(c, d).replace(e, f)
g = '''  live16b ref --arm A  R-56b-e'''
if s.count(g) == 1:
    s = s.replace(g, '''  relax ref --arm A    RELAX (docs/PREREG_relax_2026-11-02.md): the gate / R1 arm A on the live loop (relax.py);
  relax-replay ref --arm A   REF2-T replays arm A's logged checks (replay.py)
''' + g)
open(p, "w").write(s)
print("stages.py: relax and relax-replay added")
PY
python3 - <<'PY'
import os
# RULES.md (Ultra, Mon 20:5x): a share lock's reason carries "MEM <n> GB" (n = the stage's measured peak, --need)
p = os.path.expanduser("~/EvalSuite/evalsuite/cli.py")
s = open(p).read()
if "MEM {need_gb} GB" in s:
    print("cli.py: MEM tag already in")
else:
    a = '''"--gpu", "60", f"evalsuite: {why}"],'''
    assert s.count(a) == 1, a
    open(p, "w").write(s.replace(a, '''"--gpu", "60", f"evalsuite: MEM {need_gb} GB: {why}"],'''))
    print("cli.py: share-lock reason carries MEM <need> GB")
PY
~/BigMoE/.venv/bin/python -c "import ast; [ast.parse(open(f).read()) for f in ('evalsuite/stages.py', 'evalsuite/relax.py', 'evalsuite/replay.py')]" \
  && echo "parse ok" && git status --short evalsuite pending | head
