"""Install check of the NEVER-WAIT arms (Fable FAR, Thu 5 Nov 2026; run by install_nw.sh on the patched scratch copies,
and usable afterwards on the installed map). CPU (MLX's CPU device), seconds, reads only.
Per arm: dm_ablations resolves it (alone and with +code) to the expected PERT; its snapshot folder holds 36 windows whose
files have the build's md5s (MD5SUMS) and the routing capture's shapes; for a never-wait arm every window's `late` goes
through dm_common._nw_late under the arm's own PERT (shape, late within keep, the tail resolved for its lateness).
Usage (BigMoE venv): python -I -B check_install.py <dir with the patched dm_common.py / dm_ablations.py> <snapshot dir>
Exit 0 = ALL OK."""
import hashlib, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PD, SNAP = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.environ.setdefault("DM_OUT", os.path.join(os.environ.get("NW_TEST_SCRATCH", os.path.join(HERE, "scratch")), "test_out"))
os.makedirs(os.environ["DM_OUT"], exist_ok=True)
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
sys.path.insert(0, PD)
import dm_common as C
import dm_ablations as A
assert os.path.dirname(os.path.abspath(C.__file__)) == PD and os.path.dirname(os.path.abspath(A.__file__)) == PD, (C.__file__, A.__file__)
ROUT = os.path.expanduser("~/FableGod/decision_map/out/routing")
MD5 = {}
for line in open(os.path.join(HERE, "MD5SUMS")):
    h, p = line.split(None, 1); MD5[p.strip()] = h
# arm -> (mask folder, lateness or None for a twin, late flags forced off)
ARMS = {"qom_0_live_ball_c208": ("qom_0_live_ball_c208", None, False),
        "qom_0_live_ball_c208_nw1": ("qom_0_live_ball_c208_nw1", 1, False),
        "qom_0_live_ball_c208_nw1r1": ("qom_0_live_ball_c208_nw1r1", 1, False),
        "qom_0_live_ball_nw1": ("qom_0_live_ball_nw1", 1, False),
        "qom_0_split0.99_live_ball_nw1": ("qom_0_split0.99_live_ball_nw1", 1, False),
        "qom_0_live_ball_c266": ("qom_0_live_ball_c266", None, False),
        "qom_0_live_ball_c266_nw1": ("qom_0_live_ball_c266_nw1", 1, False),
        "qom_0_live_ball_c208_nwz": ("qom_0_live_ball_c208", 1, True),
        "qom_0_live_ball_c208_nw0": ("qom_0_live_ball_c208_nw0", 0, False),
        "qom_0_live_ball_c208_nw2": ("qom_0_live_ball_c208_nw2", 2, False),
        "qom_0_live_ball_c208_nw1s": ("qom_0_live_ball_c208_nw1s", 1, False),
        "qom_0_split0.99_live_ball_c208": ("qom_0_split0.99_live_ball_c208", None, False),
        "qom_0_split0.99_live_ball_c208_nw1": ("qom_0_split0.99_live_ball_c208_nw1", 1, False),
        "qom_0_split0.9_live_ball_nw1": ("qom_0_split0.9_live_ball_nw1", 1, False)}
bad = []
def ok(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c: bad.append(m)
print(f"[ci] dm_common / dm_ablations from {PD}; snapshots in {SNAP}", flush=True)
ok(hasattr(C, "NW") and hasattr(A, "nw_mask"), "the never-wait patch is in both files")
src = open(os.path.join(PD, "dm_ablations.py")).read()
for arm, (folder, d, zero) in ARMS.items():
    want = dict(rmask=os.path.join(SNAP, folder), renorm=False)
    if d is not None: want["nw"] = d
    if zero: want["nw_zero"] = True
    C.clear(); A.ablation(arm)(); p1 = dict(C.PERT); C.clear(); A.ablation(arm + "+code")(); p2 = dict(C.PERT)
    res = p1 == want and p2 == want and f'"{arm}"' in src
    dd = os.path.join(SNAP, folder); names = sorted(f for f in os.listdir(dd) if f.endswith(".npz")) if os.path.isdir(dd) else []
    n_md5 = n_shape = n_late = 0; flags = 0; err = ""
    C.PERT.clear(); C.PERT.update(want)                          # the arm's own PERT, as measure() sees it
    for n in names:
        f = os.path.join(dd, n)
        n_md5 += hashlib.md5(open(f, "rb").read()).hexdigest() == MD5.get(f"masks/{folder}/{n}")
        rm = np.load(f)["keep"]; n_shape += rm.shape == np.load(os.path.join(ROUT, n))["ids"].shape and rm.dtype == np.bool_
        if d is not None:
            try:
                lm = C._nw_late(os.path.join(C.OUT, want["rmask"], n), rm); n_late += lm.shape == rm.shape; flags += int(lm.sum())
            except Exception as e: err = f" ({type(e).__name__}: {str(e)[-90:]})"
    C.clear()
    okk = res and len(names) == 36 and n_md5 == 36 and n_shape == 36 and (d is None or (n_late == 36 and (flags == 0) == zero))
    ok(okk, f"{arm}: PERT {'ok' if res else 'WRONG ' + str(p1)}; {len(names)}/36 windows, md5 {n_md5}/36, routing shapes {n_shape}/36"
       + ("" if d is None else f", late loads {n_late}/36 at lateness {d} ({flags} late flags{', forced off' if zero else ''}){err}"))
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
