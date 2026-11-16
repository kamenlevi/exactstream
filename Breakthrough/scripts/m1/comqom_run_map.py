#!/usr/bin/env python3
"""Optional comment-QOM map arms (Breakthrough, Mon 16 Nov 2026 13:4x; NewIdeas 13:4x, information only; pre-registered
docs/CHECK_BYTES_900.md amendment (f)). One process on the ORIGINAL 12-window map (DM_OUT=~/FableGod/decision_map/out,
DM_SETS=code, --max 12), where Fit12's masks and the residency exist. The arms are registered IN MEMORY here (the armed
G1 / G2 runner and dm_bt.py are untouched):
  bt_cq_ctrl = 1c tiers (out/bt/tiers_1c)              + hc8w_mix + int8 (answer rows)   (the matched control)
  bt_cq      = 1c with comment rows on QOM (out/bt/cq_comqom, a copy of Fit12's masks_1c_comqom) + hc8w_mix + int8
  bt_cq_res  = the same + RES (out/bt/cq_comqom_res, a copy of masks_1c_comqom_res)          + hc8w_mix + int8
Usage (BigMoE venv, cwd ~/Breakthrough):  python -I -B scripts/m1/comqom_run_map.py
Prints '[cq] PEAK FOOTPRINT ...' at any exit; BG_PEAK_MAX (GB, default 22) self-guard as g1g2_run_map.py."""
import atexit, ctypes, os, signal, sys, threading, time
sys.dont_write_bytecode = True
T0 = time.time()
LIM = float(os.environ.get("BG_PEAK_MAX", "22"))
os.environ["DM_OUT"] = os.path.expanduser("~/FableGod/decision_map/out")
os.environ["DM_SETS"] = "code"
ARMS = ["bt_cq_ctrl", "bt_cq", "bt_cq_res"]


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def lifetime_gb():
    ru = _RU()
    return ru.u28 / 1e9 if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


def _peak_line():
    try:
        import mlx.core as mx
        mlxp = f"{mx.get_peak_memory() / 1e9:.2f}"
    except Exception:
        mlxp = "n/a"
    print(f"[cq] PEAK FOOTPRINT {lifetime_gb():.3f} GB (lifetime max phys_footprint), MLX peak active {mlxp} GB, "
          f"wall {time.time() - T0:.0f} s", flush=True)


def _guard():
    while True:
        time.sleep(0.5)
        if lifetime_gb() > LIM:
            print(f"[cq] SELF-GUARD: lifetime footprint > {LIM} GB: ending the run (exit 7)", flush=True)
            _peak_line()
            os._exit(7)


atexit.register(_peak_line)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
threading.Thread(target=_guard, daemon=True).start()

DM = os.path.expanduser("~/FableGod/decision_map")
sys.path.insert(0, DM)
import dm_ablations as A  # noqa: E402
import dm_bt as BT        # noqa: E402  (tiers(), the int8 arm, _three: read-only use)

for nm, d in (("bt_cq_ctrl_t", "bt/tiers_1c"), ("bt_cq_t", "bt/cq_comqom"), ("bt_cq_res_t", "bt/cq_comqom_res")):
    n = len([f for f in os.listdir(os.path.join(os.environ["DM_OUT"], d)) if f.endswith(".npz")])
    assert n == 36, f"{n} mask files in {d}"
    A.ABLATIONS[nm] = BT.tiers(d)
A.ABLATIONS["bt_cq_ctrl"] = BT._three("bt_cq_ctrl_t", "hc8w_mix", "bt_int8")
A.ABLATIONS["bt_cq"] = BT._three("bt_cq_t", "hc8w_mix", "bt_int8")
A.ABLATIONS["bt_cq_res"] = BT._three("bt_cq_res_t", "hc8w_mix", "bt_int8")
src = os.path.join(DM, "run_map.py")
sys.argv = [src] + ARMS + ["--max", "12", "--skip-done"]
print(f"[cq] running {src} {' '.join(sys.argv[1:])} (DM_OUT {os.environ['DM_OUT']}, DM_SETS code, self-guard {LIM} GB)",
      flush=True)
exec(compile(open(src).read(), src, "exec"), {"__name__": "__main__", "__file__": src})
