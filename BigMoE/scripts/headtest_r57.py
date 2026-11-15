"""Head-test for r57_screen_fast.py: everything before its model load (env parsing, module-level asserts) with the run's
exact env and args; EM.load prints a marker and exits 97 (no weights, no GPU work). usage: python headtest_r57.py <script> <args>"""
import os, runpy, sys
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env  # noqa: F401  (the screen imports it first too)
import evalsuite.model as EMm


def _stop(*a, **k):
    print("HEADTEST: reached the model load", flush=True)
    os._exit(97)


EMm.load = _stop
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
