#!/usr/bin/env python3
"""The G1 / G2 map pre-screens in one process (Breakthrough, Mon 16 Nov 2026 13:xx; pre-registered
docs/CHECK_BYTES_900.md section 2, 9ac2850). Usage (BigMoE venv, cwd ~/Breakthrough, never inside ~/FableGod):
    python -I -B scripts/m1/g1g2_run_map.py
- DM_OUT=~/FableGod/decision_map/out_code24 and DM_SETS=code (the 24 code windows of f755d70), then run_map.py's own code
  as `run_map.py <arms> --max 24 --skip-done`, arms = bt_g1h8 bt_dense3 bt_l20g2 bt_g1 bt_restore bt_l20comb bt_c6comb
  (amendments (a), (b): the decision arms first; G1G2_ARMS overrides; the arms live in ~/FableGod/decision_map/dm_bt.py).
  Results: out_code24/<arm>.json, one per finished arm.
- The file name holds 'run_map.py', so other runners' `pgrep -f run_map.py` gates see this process as a map process.
- Prints '[g1g2] PEAK FOOTPRINT <GB> GB (lifetime max phys_footprint), MLX peak active, wall' at any exit (a TERM too).
- BG_PEAK_MAX (env, GB; default 22): a daemon thread ends the process (exit 7) when the lifetime footprint passes it.
"""
import atexit, ctypes, os, signal, sys, threading, time
sys.dont_write_bytecode = True
T0 = time.time()
LIM = float(os.environ.get("BG_PEAK_MAX", "22"))
os.environ["DM_OUT"] = os.path.expanduser("~/FableGod/decision_map/out_code24")
os.environ["DM_SETS"] = "code"
ARMS = os.environ.get("G1G2_ARMS", "bt_g1h8 bt_dense3 bt_l20g2 bt_g1h8_res bt_g1 bt_restore bt_l20comb bt_c6comb").split()   # (a), (b), (e)


class _RU(ctypes.Structure):   # rusage_info_v4: uuid, then 64-bit counters (u28 = lifetime max phys_footprint)
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
    print(f"[g1g2] PEAK FOOTPRINT {lifetime_gb():.3f} GB (lifetime max phys_footprint), MLX peak active {mlxp} GB, "
          f"wall {time.time() - T0:.0f} s (DM_C {os.environ.get('DM_C', '51')})", flush=True)


def _guard():
    while True:
        time.sleep(0.5)
        g = lifetime_gb()
        if g > LIM:
            print(f"[g1g2] SELF-GUARD: lifetime footprint {g:.3f} GB > {LIM} GB: ending the run (exit 7)", flush=True)
            _peak_line()
            os._exit(7)


atexit.register(_peak_line)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
threading.Thread(target=_guard, daemon=True).start()

DM = os.path.expanduser("~/FableGod/decision_map")
sys.path.insert(0, DM)
_cj = os.path.join(os.environ["DM_OUT"], "bt_int8_census.jsonl")   # a fresh census unless an int8 arm already finished
if os.path.exists(_cj) and not any(os.path.exists(os.path.join(os.environ["DM_OUT"], a + ".json"))
                                   for a in ("bt_g1h8", "bt_l20g2", "bt_g1")):
    os.remove(_cj)
src = os.path.join(DM, "run_map.py")
sys.argv = [src] + ARMS + ["--max", "24", "--skip-done"]
print(f"[g1g2] running {src} {' '.join(sys.argv[1:])} (DM_OUT {os.environ['DM_OUT']}, DM_SETS code, "
      f"DM_C {os.environ.get('DM_C', '51')}, self-guard {LIM} GB)", flush=True)
exec(compile(open(src).read(), src, "exec"), {"__name__": "__main__", "__file__": src})
