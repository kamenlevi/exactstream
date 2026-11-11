"""Process environment for GPU runs of the 125B. Import this module before mlx, mtplx or bigmoe.

The reference numerics are BigMoE's plain exact streamer as MoonShots' GSM8K harness ran it (gsm_planes125.py /
BigMoE waittop125 "exact": no speed fusions, MTP off, compile off), plus BigMoE's PF1 staging with the full-E scratch
(BIGMOE_PF_PREFILL=1, BIGMOE_PF_FULLSCR=1), which the layer-major driver needs and which is bit-identical to the
chunk-major prefill (PrefillKernels / MoonShots lmp)."""
import os, sys

BIGMOE = os.path.expanduser("~/BigMoE")
REF_MODEL = os.path.join(BIGMOE, "models", "bare")
BIGMOE_CODE = os.environ.get("EVALSUITE_BIGMOE_CODE") or BIGMOE   # a frozen copy of BigMoE's bigmoe/ + ext/build/*.so
                                                  # (snap/<name>/: a verdict that runs for hours while BigMoE changes daily)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULTS = {
    "MTPLX_NGRAM_PREWARM": "off",
    "MTPLX_COMPILED_VERIFY": "0",
    "MTPLX_QWEN4EXP_COMPILE": "0",
    "MTPLX_COMPILE_AR_FORWARD": "0",
    "MTPLX_COMPILED_TARGET_PREFIX": "0",
}
_FORCED = {
    "BIGMOE_SPARE": "0",
    "BIGMOE_PF_NOPUBLISH": "1",
    "BIGMOE_PF_PREFILL": "1",
    "BIGMOE_PF_FULLSCR": "1",
}
# speed switches that change numerics or the expert files read; a run refuses to start with any of them set
_REFUSED = ("BIGMOE_RECORDS", "BIGMOE_COMPILE_HC", "BIGMOE_DUPROW", "BIGMOE_HC_VREAD", "BIGMOE_HC_PACK", "BIGMOE_HC_PIN",
            "BIGMOE_SORT_VERIFY", "BIGMOE_FRSPEC", "BIGMOE_SCOUT_BUNDLE", "BIGMOE_PKD1", "BIGMOE_PF_GROUP",
            "BIGMOE_MTP_STREAM", "BIGMOE_PREFETCH")

FRUGAL = os.environ.get("EVALSUITE_FRUGAL") == "1"  # Breakthrough's frugal runtime (bt_frugal.py): MoEPrefetch's fork125
FAST24 = os.environ.get("EVALSUITE_FAST24") == "1"  # BigMoE's 24 GB fast mode (scripts/fast24.py): load_fast24 sets its own
#                                                    env (FAST24_ENV), so none of the suite's forced / refused flags apply
SHIP = os.environ.get("EVALSUITE_SHIP") == "1"      # Breakthrough's small-RAM configs (bt_b12.py, v3f) with MTP as they
if FRUGAL or SHIP:                                  # ship (cli.SHIP_ENV); both run without PF1 staging, as their own
    for k in ("BIGMOE_PF_PREFILL", "BIGMOE_PF_FULLSCR"):  # drivers ran them
        _FORCED.pop(k)
        os.environ.pop(k, None)
if FAST24:
    for k in _FORCED:
        os.environ.pop(k, None)
    _FORCED, _REFUSED = {}, ()
if SHIP:                                            # MTP_STREAM is exact; FRSPEC shapes the drafts: what ships
    _REFUSED = tuple(k for k in _REFUSED if k not in ("BIGMOE_MTP_STREAM", "BIGMOE_FRSPEC", "BIGMOE_PF_GROUP"))
for k, v in _DEFAULTS.items():
    os.environ.setdefault(k, v)
os.environ.update(_FORCED)
_bad = [k for k in _REFUSED if os.environ.get(k) not in (None, "", "0")]
if _bad:
    raise SystemExit(f"evalsuite: unset {_bad}: the suite measures the plain exact reference path")
if BIGMOE_CODE not in sys.path and not SHIP:        # SHIP: spec125 must not find ~/BigMoE's bigmoe (repguard, bigmoe_ext)
    sys.path.insert(0, BIGMOE_CODE)


def provenance():
    """git heads of the code this run depends on (BigMoE changes daily)."""
    import subprocess
    out = {}
    repos = [("bigmoe" if BIGMOE_CODE == BIGMOE else "bigmoe_live", BIGMOE), ("evalsuite", ROOT)]
    if BIGMOE_CODE != BIGMOE:                     # the frozen copy this run imports (its SOURCE line: commit, tree, .so sha)
        out["bigmoe"] = "snap: " + open(os.path.join(BIGMOE_CODE, "SOURCE")).read().strip()
        m = sys.modules.get("bigmoe")
        out["bigmoe_imported_from"] = os.path.dirname(os.path.dirname(m.__file__)) if m else None
    if FRUGAL:
        repos += [("moeprefetch", os.path.expanduser("~/MoEPrefetch")), ("breakthrough", os.path.expanduser("~/Breakthrough"))]
    if SHIP:
        repos += [("breakthrough", os.path.expanduser("~/Breakthrough")), ("specdecode", os.path.expanduser("~/SpecDecode"))]
    if FAST24:
        repos += [("specdecode", os.path.expanduser("~/SpecDecode"))]
    if SHIP:                                      # ~/Fit12 is no git repo: hash its hook files instead
        import glob, hashlib
        bt = os.path.expanduser("~/Breakthrough")    # and the live bt_b12 + the engine build (ext/build is not in git)
        eng = os.environ.get("B12_ENGINE")
        for f in [os.path.join(bt, "scripts", "bt_b12.py")] + (glob.glob(os.path.join(bt, eng, "ext", "build", "*.so")) if eng else []):
            out["sha/" + os.path.relpath(f, bt)] = hashlib.sha256(open(f, "rb").read()).hexdigest()[:10]
        for f in ("fit_hook.py", "fit_pr1.py", "fit_budget.py", "fit_topk.py"):
            p = os.path.expanduser(f"~/Fit12/scripts/{f}")
            if os.path.exists(p):
                out["fit12/" + f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:10]
    if "ref2" in sys.modules:                     # R-40 REF2: Leap's patch module and its kernels, hashed where imported
        import hashlib                            # from (model.REF2_CODE: a frozen copy; its SOURCE names the commit)
        d = os.path.dirname(os.path.abspath(sys.modules["ref2"].__file__))
        repos += [("leap_live", os.path.expanduser("~/Leap"))]
        out["ref2_imported_from"] = d
        if os.path.exists(os.path.join(d, "SOURCE")):
            out["ref2_source"] = open(os.path.join(d, "SOURCE")).read().strip()
        for f in ("ref2.py", "leapk.py", "qmm16.py", "qmmg25.py", "gmr24.py"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                out["sha/leap/" + f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:10]
        g = os.path.join(d, "ms_gdn")             # REF2c snapshots: MoonShots' exact GDN files (model.ref2_import)
        for f in sorted(os.listdir(g)) if os.path.isdir(g) else ():
            if f.endswith(".py"):
                out["sha/leap/ms_gdn/" + f] = hashlib.sha256(open(os.path.join(g, f), "rb").read()).hexdigest()[:10]
        out["ref2_cfg"] = dict(sys.modules["ref2"].CFG)   # its switches as run (a REF2c variant's ref2_cfg sets some)
    for name, path in repos:
        try:
            r = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
            d = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True)
            out[name] = r.stdout.strip() + ("+dirty" if d.stdout.strip() else "")
        except Exception:
            out[name] = "?"
    try:
        import mlx.core as mx, mtplx
        out["mlx"] = mx.__version__
        out["mtplx"] = getattr(mtplx, "__version__", "?")
    except Exception:
        pass
    return out
