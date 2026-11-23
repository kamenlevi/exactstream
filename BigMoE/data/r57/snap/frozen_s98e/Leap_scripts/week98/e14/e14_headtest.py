"""e14_headtest.py (Leap, Sun 15 Nov 2026 20:5x; from Breakthrough's c55_headtest.py): head test of e14_loop.py with the run's
exact env (e14_env.sh), through the real loop up to its model load (evalsuite.model.load patched to print a marker and exit
97; the flagcheck pattern), then an import check of what the G32 section and e14_engine import later (ref2, v32, v32one,
refc, bofi, ck, bof, moe32, glue_fused, router_fused, e14_engine). Usage: [venv] python -I -B e14_headtest.py with the env
set; rc 0 = PASS. Light: no model load, no GPU work."""
import importlib.abc, importlib.util, os, runpy, subprocess, sys
MARK = "E14 HEADTEST: reached EM.load"
HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1 and sys.argv[1] == "child":
    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "evalsuite.model":
                return None
            sys.meta_path.remove(self)
            s = importlib.util.find_spec(name)
            orig = s.loader.exec_module

            def exec_module(mod):
                orig(mod)

                def load(*a, **k):
                    print(MARK, flush=True)
                    os._exit(97)
                mod.load = load
            s.loader.exec_module = exec_module
            return s
    sys.meta_path.insert(0, Hook())
    script = os.path.join(HERE, "e14_loop.py")
    sys.argv = [script, "160", "256"]
    runpy.run_path(script, run_name="__main__")
    print("E14 HEADTEST: the loop ended without loading the model", flush=True)
    sys.exit(3)
p = subprocess.run([sys.executable, "-I", "-B", os.path.abspath(__file__), "child"], capture_output=True, text=True, timeout=300)
ok1 = p.returncode == 97 and MARK in p.stdout
print("loop head:", "PASS" if ok1 else f"FAIL rc {p.returncode}: " + (p.stdout + p.stderr)[-1500:])
code = ("import os, sys; sys.path[:0] = [os.path.expanduser(d) for d in ('/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts/week98/e14', '/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts', "
        "'/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts/week98', '~/BigMoE', '~/EvalSuite')]; import mlx.core; "
        "import ref2, v32, v32one, refc, bofi, ck, bof, moe32, glue_fused, router_fused, e14_engine; "
        "sys.path.append(os.path.expanduser('~/Breakthrough/scripts/m1/check55')); import moe_dense; "
        "print('imports ok; seam_hi', ref2.CFG.get('seam_hi'), 'v32one.ON', v32one.ON, 'refc.ON', refc.ON, 'bofi.ON', bofi.ON, "
        "'arms', e14_engine.ARMS, 'moe_dense', len(getattr(moe_dense, 'SOURCES', {})), 'sources')")
q = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=300)
ok2 = q.returncode == 0
print("imports:", (q.stdout.strip().splitlines() or ["?"])[-1] if ok2 else "FAIL " + (q.stdout + q.stderr)[-1500:])
sys.exit(0 if ok1 and ok2 else 1)
