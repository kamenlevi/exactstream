"""Head test for runner 64's arms (the orchestrator 11-14: head-test runners against the REAL loop, not fakes). It runs an
arm's screen with the arm's exact environment, as runner 64's fast() builds it (its unsets, the env file's last line,
R57_PRESET=45, R57_REPS=1, R57_WARM=1, R57_TAG, R57_KEYS; argv C CAP). It stops at evalsuite.model.load, so the screen's
imports, env parsing and module-level asserts all see the run's environment, in the screen's own import order (an import
hook patches load as the screen imports evalsuite.model). Checks after the model load are not covered. A watchdog kills
the child above 3 GB or after 120 s, in case anything starts a real load. CPU only, a few seconds, no model.
  python3 tools/r57_headtest.py <env file> <tag> [K=V ...]      (extra env, e.g. R57_REPLAY=...)
Last line: `HEADTEST <tag>: PASS (reached the model load) | FAIL (...)`; exit 0 / 1."""
import json, os, re, subprocess, sys, time

BM = os.path.expanduser("~/BigMoE")
ES = os.path.expanduser("~/EvalSuite")
UNSETS = ("EVALSUITE_BIGMOE_CODE", "EVALSUITE_RELAX_VARIANT", "EVALSUITE_STAGE_ENV", "R57_KEYS", "R57_TAG", "BIGMOE_TIER",
          "BIGMOE_TIER_GLU", "BIGMOE_TIER_CL", "BIGMOE_TIER_COMBINE", "BIGMOE_OOO", "BIGMOE_MAXK")   # runner 64's UNSETS
CHILD = r'''
import importlib.abc, importlib.machinery, os, runpy, sys
class Stop(BaseException):
    pass
def load(*a, **k):
    raise Stop()
class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "evalsuite.model":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None:
            ex = spec.loader.exec_module
            def exec_module(m):
                ex(m); m.load = load
            spec.loader.exec_module = exec_module
        return spec
sys.meta_path.insert(0, Hook())
screen = sys.argv[1]; sys.argv = [screen] + sys.argv[2:]
try:
    runpy.run_path(screen, run_name="__main__")
except Stop:
    print("HEADTEST-REACHED-LOAD", flush=True); sys.exit(0)
print("HEADTEST-NO-LOAD: the screen ended before evalsuite.model.load", flush=True); sys.exit(3)
'''


def env_pairs(path):
    """runner 64's parse_env: the last non-comment line, comma-split; a token without '=' continues the previous value"""
    line = [l for l in open(path).read().splitlines() if l and not l.startswith("#")][-1]
    out = []
    for t in line.split(","):
        if "=" in t or not out:
            out.append(t)
        else:
            out[-1] += "," + t
    assert all("=" in t for t in out), "the env line does not parse as K=V pairs"
    return [t.split("=", 1) for t in out]


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    fenv, tag, extra = argv[0], argv[1], [a.split("=", 1) for a in argv[2:]]
    comment = " ".join(l for l in open(fenv).read().splitlines() if l.startswith("#"))
    snap = re.search(r"data/r57/snap/([A-Za-z0-9_.-]+/)*r57_screen[A-Za-z0-9_.-]*\.py", comment)
    screen = os.path.join(BM, snap.group(0)) if snap else None
    if not screen or not os.path.isfile(screen):
        print(f"HEADTEST {tag}: FAIL (no snapshot screen named in the env comment)"); return 1
    c = re.search(r"(Sat|Sun|Mon|C) ([12][0-9][0-9])", comment)
    key = json.load(open(os.path.join(ES, "tools", "lossy_panel.json")))["tier0_code"][0]
    env = {k: v for k, v in os.environ.items() if k not in UNSETS}
    env.update(dict(env_pairs(fenv)))
    env.update(dict(extra))
    env.update(R57_PRESET="45", R57_REPS="1", R57_WARM="1", R57_TAG=f"headtest_{tag}", R57_KEYS=key)
    py = os.path.join(BM, ".venv", "bin", "python")
    t0 = time.time()
    p = subprocess.Popen([py, "-c", CHILD, screen, c.group(2) if c else "256", "2048"], env=env, cwd=ES,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak, why = 0, ""
    while p.poll() is None:
        rss = int((subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout or "0").strip() or 0)
        peak = max(peak, rss)
        if rss > 3 * 1024 * 1024 or time.time() - t0 > 120:
            why = f"watchdog: {rss / 1048576:.1f} GB after {time.time() - t0:.0f} s"
            p.kill(); break
        time.sleep(0.2)
    out = p.communicate()[0]
    tail = [l for l in out.strip().splitlines() if l.strip()][-6:]
    ok = p.returncode == 0 and "HEADTEST-REACHED-LOAD" in out and not why
    print(f"  screen {os.path.relpath(screen, BM)}; C {c.group(2) if c else 256}; {len(env_pairs(fenv))} env pairs"
          f"{'; extra ' + ','.join(k for k, _ in extra) if extra else ''}; {time.time() - t0:.1f} s; peak {peak / 1024:.0f} MB")
    if not ok:
        for l in tail:
            print(f"  | {l[:200]}")
    print(f"HEADTEST {tag}: {'PASS (reached the model load)' if ok else 'FAIL (' + (why or (tail[-1][:160] if tail else 'no output')) + ')'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
