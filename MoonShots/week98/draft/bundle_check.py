"""Run r57_draft_test.py (the harness, live) against the bundle or the live tree, digesting every draft() output
(all fields but 'ms') in call order, so a bundle run and a live run compare bit for bit (MoonShots, Wed 18 Nov 2026).
--bundle: the bundle dir comes first on sys.path, a meta-path finder serves every bundle module name from it (so the
harness's own ~/ inserts cannot shadow them), the live MoonShots / BigMoE scripts / TrainLab / Leap dirs are removed,
and ~/BigMoE is appended for the harness's model load only (mtp125.load -> bigmoe.loader). Ends with a module audit:
where every loaded module came from (bundle / R57_LEAP_SCRIPTS / venv / other).
usage: ~/BigMoE/.venv/bin/python -I -B bundle_check.py (--bundle <dir> | --live) -- <r57_draft_test args>"""
import hashlib
import importlib.abc
import importlib.util
import os
import runpy
import sys

H = os.path.expanduser
HERE = os.path.dirname(os.path.abspath(__file__))
i = sys.argv.index("--")
opts, targs = sys.argv[1:i], sys.argv[i + 1:]
BUNDLE = os.path.realpath(opts[1]) if opts[0] == "--bundle" else None
NAMES = set()
if BUNDLE:
    live = [H("~/MoonShots"), H("~/BigMoE/scripts"), H("~/TrainLab"), H("~/Leap"), H("~/SpecDecode")]
    sys.path = [p for p in sys.path if os.path.realpath(p or ".") != HERE
                and not any(os.path.realpath(p or ".").startswith(r) for r in live)]
    NAMES = {f[:-3] for f in os.listdir(BUNDLE) if f.endswith(".py")}

    class _Bundle(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name in NAMES:
                return importlib.util.spec_from_file_location(name, os.path.join(BUNDLE, name + ".py"))
            return None
    sys.meta_path.insert(0, _Bundle())
    sys.path.insert(0, BUNDLE)
    sys.path.append(H("~/BigMoE"))
else:                                                    # live: the harness's own inserts (python -I adds no script dir)
    sys.path.insert(0, H("~/MoonShots/week98/draft"))
    sys.path.insert(0, H("~/BigMoE/scripts"))
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import r57_draft as R  # noqa: E402

D, N = hashlib.sha256(), [0]


def _dig(v):
    if isinstance(v, mx.array):
        a = np.array(v.view(mx.uint16) if v.dtype == mx.bfloat16 else v)
        return f"mx{v.dtype}{a.shape}".encode() + a.tobytes()
    if isinstance(v, np.ndarray):
        return f"np{v.dtype}{v.shape}".encode() + v.tobytes()
    if isinstance(v, dict):
        return b"{" + b"".join(str(k).encode() + b"=" + _dig(v[k]) for k in sorted(v)) + b"}"
    if isinstance(v, (list, tuple)):
        return b"[" + b"".join(_dig(x) for x in v) + b"]"
    return repr(v).encode()


_draft = R.draft


def draft(*a, **k):
    o = _draft(*a, **k)
    if o is not None:
        D.update(_dig({kk: vv for kk, vv in o.items() if kk != "ms"}))
        N[0] += 1
    return o


R.draft = draft


def audit():
    pre = [os.path.realpath(p) + "/" for p in {sys.prefix, sys.base_prefix, sys.exec_prefix}]
    leap = os.environ.get("R57_LEAP_SCRIPTS")
    c, other = {"bundle": 0, "leap_env": 0, "venv": 0}, []
    for name, m in sorted(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        f = os.path.realpath(f)
        if BUNDLE and f.startswith(BUNDLE + "/"):
            c["bundle"] += 1
        elif BUNDLE and leap and f.startswith(os.path.realpath(leap) + "/"):
            c["leap_env"] += 1
        elif any(f.startswith(p) for p in pre):
            c["venv"] += 1
        else:
            other.append(f"{name}:{f.replace(H('~'), '~')}")
    bad = [n for n in NAMES if n in sys.modules and not os.path.realpath(sys.modules[n].__file__).startswith(BUNDLE + "/")]
    print(f"[bundle_check] {'bundle ' + os.path.basename(BUNDLE) if BUNDLE else 'live'}: drafts {N[0]} digest "
          f"{D.hexdigest()[:16]}; modules {c}; bundle names loaded elsewhere {bad}; other {other}", flush=True)


sys.argv = [os.path.join(HERE, "r57_draft_test.py")] + targs
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    audit()
