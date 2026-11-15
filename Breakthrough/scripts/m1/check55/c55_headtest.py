"""c55_headtest.py (Breakthrough): head test of r57_c55.py with the run's exact env, through the real loop up to its model load
(evalsuite.model.load patched to print a marker and exit 97; Leap's flagcheck pattern), then an import check of what the
G32 section imports later (ref2, v32, v32one, glue_fused, router_fused, c55_engine). Usage: [venv] python -I -B c55_headtest.py
with the env set; rc 0 = PASS."""
import importlib.abc, importlib.util, os, runpy, subprocess, sys
MARK = "C55 HEADTEST: reached EM.load"
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
    script = os.path.join(HERE, "r57_c55.py")
    sys.argv = [script, "160", "256"]
    runpy.run_path(script, run_name="__main__")
    print("C55 HEADTEST: the loop ended without loading the model", flush=True)
    sys.exit(3)
p = subprocess.run([sys.executable, "-I", "-B", os.path.abspath(__file__), "child"], capture_output=True, text=True, timeout=300)
ok1 = p.returncode == 97 and MARK in p.stdout
print("loop head:", "PASS" if ok1 else f"FAIL rc {p.returncode}: " + (p.stdout + p.stderr)[-1500:])
code = ("import os, sys; sys.path[:0] = [os.path.expanduser(d) for d in ('~/Breakthrough/scripts/m1/check55', '~/Leap/scripts', "
        "'~/Leap/scripts/week98', '~/BigMoE', '~/EvalSuite')]; import mtlprof; mtlprof.load(2); import mlx.core; "
        "import ref2, v32, v32one, glue_fused, router_fused, c55_engine; print('imports ok; seam_hi', ref2.CFG.get('seam_hi'), "
        "'v32one.ON', v32one.ON)")
q = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=300)
ok2 = q.returncode == 0
print("imports:", (q.stdout.strip().splitlines() or ["?"])[-1] if ok2 else "FAIL " + (q.stdout + q.stderr)[-1500:])
sys.exit(0 if ok1 and ok2 else 1)
