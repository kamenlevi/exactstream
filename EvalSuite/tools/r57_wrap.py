"""Run BigMoE's scripts/r57_screen_fast.py as its own runner would, then record every module it loaded (path + sha256) for
the run's provenance (docs/PREREG_agentfast_2026-11-06.md; the runner pins BigMoE's tree to the commit before each run).
  [BigMoE venv] python tools/r57_wrap.py <script> <manifest.json> <C> <CAP>"""
import hashlib, json, os, runpy, sys, time

script, manifest = sys.argv[1], sys.argv[2]
sys.argv = [script] + sys.argv[3:]
t0 = time.strftime("%H:%M:%S")


def dump():
    mods = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if f and os.path.isfile(f) and f.endswith((".py", ".so")) and "/site-packages/" not in f and "/lib/python3." not in f:
            mods[name] = [f, hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]]
    env = {k: v for k, v in os.environ.items() if k.startswith(("R57_", "REF2_", "BIGMOE_", "MTPLX_", "EVALSUITE_"))}
    json.dump(dict(script=script, argv=sys.argv, start=t0, end=time.strftime("%H:%M:%S"), env=env, modules=mods),
              open(manifest, "w"), indent=0, sort_keys=True)


try:
    runpy.run_path(script, run_name="__main__")
finally:
    dump()
