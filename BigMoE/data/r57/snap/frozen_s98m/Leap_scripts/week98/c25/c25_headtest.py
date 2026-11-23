"""c25_headtest.py (Leap, Mon 16 Nov 2026 12:5x; from e14_headtest.py): head test of c25_loop.py with the run's env
(c25_env.py), through the real loop up to its model load (evalsuite.model.load patched to print a marker and exit 97; the
flagcheck pattern), then an import check of what the G32 section and c25_engine import later (ref2, v32, f32one, int8d,
dn32, qmm32n, one32, glue_fused, router_fused, c25_engine). Usage: eval "$(python3 c25_env.py)"; [venv] python -I -B
c25_headtest.py; rc 0 = PASS. Light: no model load, no GPU work."""
import importlib.abc, importlib.util, os, runpy, subprocess, sys
MARK = "C25 HEADTEST: reached EM.load"
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
    script = os.path.join(HERE, "c25_loop.py")
    sys.argv = [script, "256", "2048"]
    runpy.run_path(script, run_name="__main__")
    print("C25 HEADTEST: the loop ended without loading the model", flush=True)
    sys.exit(3)
p = subprocess.run([sys.executable, "-I", "-B", os.path.abspath(__file__), "child"], capture_output=True, text=True, timeout=300)
ok1 = p.returncode == 97 and MARK in p.stdout
print("loop head:", "PASS" if ok1 else f"FAIL rc {p.returncode}: " + (p.stdout + p.stderr)[-1500:])
code = ("import os, sys; sys.path[:0] = [os.path.expanduser(d) for d in ('/Users/kamen/BigMoE/data/r57/snap/frozen_s98m/Leap_scripts/week98/c25', '/Users/kamen/BigMoE/data/r57/snap/frozen_s98m/Leap_scripts', "
        "'/Users/kamen/BigMoE/data/r57/snap/frozen_s98m/Leap_scripts/week98', '~/BigMoE', '~/EvalSuite')]; import mlx.core; "
        "import ref2, v32, f32one, int8d, dn32, qmm32n, one32, glue_fused, router_fused, c25_engine; "
        
        "print('imports ok; f32one.ON', f32one.ON, 'int8d.ON', int8d.ON, "
        "'arms', c25_engine.ARMS)")
q = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=300)
ok2 = q.returncode == 0
print("imports:", (q.stdout.strip().splitlines() or ["?"])[-1] if ok2 else "FAIL " + (q.stdout + q.stderr)[-1500:])
sys.exit(0 if ok1 and ok2 else 1)
