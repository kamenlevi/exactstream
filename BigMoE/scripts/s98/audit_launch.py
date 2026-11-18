"""run scripts/kt_launch.py in this process; at exit print which bundle modules loaded from where + r57_draft's PK counters"""
import atexit, os, runpy, sys
B = "/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/MoonShots_draft"
NAMES = sorted(f[:-3] for f in os.listdir(B) if f.endswith(".py"))
def audit():
    got = {n: getattr(sys.modules[n], "__file__", "") for n in NAMES if n in sys.modules}
    other = {n: f for n, f in got.items() if not os.path.realpath(f or "").startswith(B)}
    rd = sys.modules.get("r57_draft"); st = getattr(rd, "ST", {}) if rd else {}
    print(f"[audit] bundle names loaded {len(got)}: {sorted(got)}; from elsewhere: {other}; "
          f"r57_draft fmt {st.get('fmt')} pk_trees {st.get('pk_trees', 0)} pk_fallback {st.get('pk_fallback', 0)}", flush=True)
atexit.register(audit)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
