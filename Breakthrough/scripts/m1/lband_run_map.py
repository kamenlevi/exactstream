#!/usr/bin/env python3
"""Per-layer lambda band screen (Breakthrough, Mon 16 Nov 2026 14:0x; pre-registered docs/CHECK_BYTES_900.md section 6;
NewIdeas 13:5x GO). One process on the 24 code windows (DM_OUT=~/FableGod/decision_map/out_code24, DM_SETS=code,
--max 24), experts only (the exact dense path). Arms registered IN MEMORY here (dm_bt.py and the armed runners untouched):
  bt_c6tiers  C6 everywhere (lambda 1.8e-2; dm_bt's own entry)            the reference
  bt_lb0      lambda 3.2e-2 on layers 0-15, C6 elsewhere (out_code24/bt/tiers_lb0)
  bt_lb1      lambda 3.2e-2 on layers 16-31                   (tiers_lb1)
  bt_lb2      lambda 3.2e-2 on layers 32-47                   (tiers_lb2)
Usage (BigMoE venv, cwd ~/Breakthrough):  python -I -B scripts/m1/lband_run_map.py
Prints '[lb] PEAK FOOTPRINT ...' at any exit; BG_PEAK_MAX (GB, default 22) self-guard."""
import atexit, ctypes, os, signal, sys, threading, time
sys.dont_write_bytecode = True
T0 = time.time()
LIM = float(os.environ.get("BG_PEAK_MAX", "22"))
os.environ["DM_OUT"] = os.path.expanduser("~/FableGod/decision_map/out_code24")
os.environ["DM_SETS"] = "code"
ARMS = ["bt_c6tiers", "bt_lb0", "bt_lb1", "bt_lb2"]


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
    print(f"[lb] PEAK FOOTPRINT {lifetime_gb():.3f} GB (lifetime max phys_footprint), MLX peak active {mlxp} GB, "
          f"wall {time.time() - T0:.0f} s", flush=True)


def _guard():
    while True:
        time.sleep(0.5)
        if lifetime_gb() > LIM:
            print(f"[lb] SELF-GUARD: lifetime footprint > {LIM} GB: ending the run (exit 7)", flush=True)
            _peak_line()
            os._exit(7)


atexit.register(_peak_line)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
threading.Thread(target=_guard, daemon=True).start()

DM = os.path.expanduser("~/FableGod/decision_map")
sys.path.insert(0, DM)
import dm_ablations as A  # noqa: E402
import dm_bt as BT        # noqa: E402  (tiers(): read-only use; bt_c6tiers is its own entry)

for i in range(3):
    d = f"bt/tiers_lb{i}"
    n = len([f for f in os.listdir(os.path.join(os.environ["DM_OUT"], d)) if f.endswith(".npz")])
    assert n == 24, f"{n} mask files in {d}"
    A.ABLATIONS[f"bt_lb{i}"] = BT.tiers(d)
assert "bt_c6tiers" in A.ABLATIONS
src = os.path.join(DM, "run_map.py")
sys.argv = [src] + ARMS + ["--max", "24", "--skip-done"]
print(f"[lb] running {src} {' '.join(sys.argv[1:])} (DM_OUT {os.environ['DM_OUT']}, DM_SETS code, self-guard {LIM} GB)",
      flush=True)
exec(compile(open(src).read(), src, "exec"), {"__name__": "__main__", "__file__": src})
