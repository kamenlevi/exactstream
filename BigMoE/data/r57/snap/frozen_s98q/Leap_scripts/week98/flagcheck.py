"""flagcheck.py (Leap, Sat 14 Nov 2026 21:2x): a runner's head test against the REAL decode loop's own flag checks (the
orchestrator 21:0x: "head-test runners against the real loop's flag checks, not fakes"; screen 3's lossy-24 runs died at
start-up on w24p2:159, R57_ROWCAP vs R57_WARMSPLIT, which a fake never sees).

It runs the clean clone's own run.py main() with the run's exact arguments: the same config read, the same environment
build (base env, the config's line, every --env), the same spec file. Only the engine child differs: it is this script,
which calls run.py's own engine_child() but makes evalsuite.model.load (the loop's model load, EM.load) print a marker and
exit 97. So everything the loop does before loading the model runs with the run's environment (its imports, its
environment parsing and every module-level assert), and nothing heavy does: no weights, no GPU work, no pool.
Checks the loop makes AFTER the model load (per-arm asserts inside run(), the probe's install) are not covered.

Usage (as the runner would run run.py: same python, HOME, HF_HOME, cwd):
  <clone>/.venv/bin/python flagcheck.py <clone> <run.py args ...>
Prints PASS (the loop reached the model load) or FAIL (the child's last lines); rc 0 / 1."""
import importlib.abc
import importlib.util
import os
import subprocess
import sys

MARK = "FLAGCHECK: reached the model load"


def _runcli(clone):
    spec = importlib.util.spec_from_file_location("runcli", os.path.join(clone, "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def child(clone, sp):
    class Hook(importlib.abc.MetaPathFinder):        # the same pattern as run.py's own _Hook: patch right after import
        def find_spec(self, name, path, target=None):
            if name != "evalsuite.model":
                return None
            sys.meta_path.remove(self)
            s = importlib.util.find_spec(name)
            if s is None:
                return None
            orig = s.loader.exec_module

            def exec_module(mod):
                orig(mod)

                def load(*a, **k):
                    print(f"{MARK} (evalsuite.model.load); every check before it passed", flush=True)
                    os._exit(97)
                mod.load = load
            s.loader.exec_module = exec_module
            return s
    sys.meta_path.insert(0, Hook())
    _runcli(clone).engine_child(sp)
    print("FLAGCHECK: the loop ended without loading the model", flush=True)
    sys.exit(3)


def parent(clone, args):
    m = _runcli(clone)
    real = subprocess.run
    seen = {}

    def run(cmd, env=None, stdout=None, stderr=None, **kw):   # run.py: [python, run.py, "--_engine", spec]
        assert cmd[-2] == "--_engine", cmd
        r = real([cmd[0], os.path.abspath(__file__), "--child", clone, cmd[-1]], env=env, stdout=stdout, stderr=stderr,
                 timeout=300)
        seen["rc"] = r.returncode
        seen["log"] = getattr(stdout, "name", None)
        return r
    subprocess.run = run
    sys.argv = [os.path.join(clone, "run.py")] + args
    try:
        m.main()
    finally:
        subprocess.run = real
    log = open(seen["log"]).read() if seen.get("log") and os.path.exists(seen["log"]) else ""
    ok = seen.get("rc") == 97 and MARK in log
    print(("PASS" if ok else "FAIL") + f": flag check of run.py {' '.join(args)} (child rc {seen.get('rc')})")
    if not ok:
        print("".join(log.splitlines(True)[-15:]))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    sys.exit(parent(sys.argv[1], sys.argv[2:]))
