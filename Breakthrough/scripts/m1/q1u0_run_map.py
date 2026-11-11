#!/usr/bin/env python3
"""Map arm q1u_0_live (= Fit12's S1u) without editing the decision map's files (Breakthrough, Mon 9 Nov 2026 21:4x).
Registers the arm in memory, dm_ablations.q1_union on the masks NewIdeas' builder wrote to
~/Breakthrough/data/dm_masks/q1u_0_live (an absolute rmask dir resolves as the harness documents), then runs run_map.py's
own code as: run_map.py q1u_0_live --max 12 --skip-done. The result lands where every map arm's does:
~/FableGod/decision_map/out/q1u_0_live.json. This file's name holds 'run_map.py', so other runners' `pgrep -f run_map.py`
gates see this process as a map process. Start it with cwd ~/Breakthrough (never inside ~/FableGod): the thermostat and
fanwatch's heat freeze recognise research jobs by working folder or command line.
Usage (BigMoE venv): cd ~/Breakthrough && python -I -B scripts/m1/q1u0_run_map.py
"""
import atexit, ctypes, os, signal, sys, time
sys.dont_write_bytecode = True
T0 = time.time()


class _RU(ctypes.Structure):   # rusage_info_v4: uuid, then 64-bit counters (as Fit12's bt_b12.py reads it; u28 = lifetime max)
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def _peak_line():
    """23:2x: the measured peak NewIdeas asked for (a 12 GB slot beside the w24 gate needs <= 12 GB MEASURED): lifetime max
    phys_footprint (the 12 GB lines' metric), MLX peak active, wall time. Printed at any exit, a TERM included."""
    ru = _RU()
    ok = ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0
    try:
        import mlx.core as mx
        mlxp = f"{mx.get_peak_memory() / 1e9:.2f}"
    except Exception:
        mlxp = "n/a"
    print(f"[q1u0] PEAK FOOTPRINT {(ru.u28 / 1e9) if ok else float('nan'):.3f} GB (lifetime max phys_footprint), "
          f"MLX peak active {mlxp} GB, wall {time.time() - T0:.0f} s", flush=True)


atexit.register(_peak_line)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))   # a stop runs atexit too (KILL 20 s later still frees memory)
DM = os.path.expanduser("~/FableGod/decision_map")
MASKS = os.path.expanduser("~/Breakthrough/data/dm_masks/q1u_0_live")
ARM = "q1u_0_live"
sys.path.insert(0, DM)
import dm_ablations as A  # noqa: E402

n = len([f for f in os.listdir(MASKS) if f.endswith(".npz")])
assert n == 36, f"{n} mask files in {MASKS}"
if ARM in A.ABLATIONS:
    print(f"[q1u0] {ARM} is already registered in dm_ablations.py; using that entry", flush=True)
else:
    A.ABLATIONS[ARM] = A.q1_union(MASKS)
src = os.path.join(DM, "run_map.py")
sys.argv = [src, ARM, "--max", "12", "--skip-done"]
print(f"[q1u0] running {src} {' '.join(sys.argv[1:])} (masks {MASKS})", flush=True)
exec(compile(open(src).read(), src, "exec"), {"__name__": "__main__", "__file__": src})
