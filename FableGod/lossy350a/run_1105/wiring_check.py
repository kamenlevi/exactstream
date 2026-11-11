"""CPU check before any GPU time: with the runner's environment, which files will the map process import?
Mirrors dm_common's own sys.path inserts; uses find_spec only (nothing is imported or executed)."""
import importlib.util, os, sys
R = os.path.dirname(os.path.abspath(__file__)); DM = os.path.expanduser("~/FableGod/decision_map")
assert sys.path[0] == R, sys.path[:3]
for i, p in enumerate(("~/BigMoE", "~/Leap/scripts", "~/TrainLab/sticky")): sys.path.insert(i, os.path.expanduser(p))
want = {"dm_common": R, "dm_ablations": R, "dm_xs": R, "dm_t3": DM, "pausewait": os.path.expanduser("~/Leap/scripts")}
ok = True
for m, d in want.items():
    s = importlib.util.find_spec(m); o = os.path.dirname(os.path.realpath(s.origin)) if s and s.origin else None
    good = o == os.path.realpath(d); ok &= good
    print(f"  {m:13s} -> {s.origin if s else None} {'OK' if good else 'WRONG, want ' + d}")
env = {k: os.environ.get(k) for k in ("DM_XS", "BIGMOE_EXT_BUILD", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH", "DM_OUT", "DM_SETS")}
print("  env:", env)
ok &= env["DM_XS"] == "1" and env["BIGMOE_EXT_BUILD"] == "build_nat" and env["PYTHONDONTWRITEBYTECODE"] == "1" and env["PYTHONPATH"] == DM
ok &= bool(env["DM_OUT"]) and os.path.realpath(env["DM_OUT"]).startswith(os.path.realpath(R)) and os.path.exists(os.path.join(env["DM_OUT"], "baseline.json"))
print("WIRING OK" if ok else "WIRING WRONG"); sys.exit(0 if ok else 1)
