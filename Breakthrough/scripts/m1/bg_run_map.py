#!/usr/bin/env python3
"""One decision-map arm on masks outside the decision map's files, with a measured-peak print and a memory self-guard
(Breakthrough, Mon 9 Nov 2026 23:4x; for the 12 GB neighbour slot beside EvalSuite's w24 card gate, NewIdeas 23:30).
Usage (BigMoE venv, cwd ~/Breakthrough, never inside ~/FableGod):  python -I -B scripts/m1/bg_run_map.py <arm> <mask dir>
- Registers <arm> in memory as dm_ablations.q1_union(<mask dir>) (36 window files expected; an absolute rmask dir resolves as
  the harness documents) unless dm_ablations.py already has it, then runs run_map.py's own code as
  `run_map.py <arm> --max 12 --skip-done`. The result lands where every arm's does: ~/FableGod/decision_map/out/<arm>.json.
  The file name holds 'run_map.py', so other runners' `pgrep -f run_map.py` gates see this process as a map process.
- DM_C (env, the harness's expert slots per layer; default 51) sets the cache: same flips, other speed and memory
  (0.104 GB per slot = 48 layers x 2.16 MB).
- Prints '[bg] PEAK FOOTPRINT <GB> GB (lifetime max phys_footprint), MLX peak active, wall' at any exit, a TERM included.
- BG_PEAK_MAX (env, GB; default 11.9): a daemon thread reads the lifetime max footprint every 0.5 s and ends the process
  (exit 7) when it passes the limit, so the arm measures <= 12.0 GB (the neighbour rule).
"""
import atexit, ctypes, os, signal, sys, threading, time
sys.dont_write_bytecode = True
T0 = time.time()
ARM, MASKS = sys.argv[1], os.path.abspath(os.path.expanduser(sys.argv[2]))
LIM = float(os.environ.get("BG_PEAK_MAX", "11.9"))


class _RU(ctypes.Structure):   # rusage_info_v4: uuid, then 64-bit counters (as Fit12's bt_b12.py reads it; u28 = lifetime max)
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
    print(f"[bg] {ARM} PEAK FOOTPRINT {lifetime_gb():.3f} GB (lifetime max phys_footprint), MLX peak active {mlxp} GB, "
          f"wall {time.time() - T0:.0f} s (DM_C {os.environ.get('DM_C', '51')})", flush=True)


def _guard():
    while True:
        time.sleep(0.5)
        g = lifetime_gb()
        if g > LIM:
            print(f"[bg] SELF-GUARD: lifetime footprint {g:.3f} GB > {LIM} GB: ending the arm (exit 7)", flush=True)
            _peak_line()
            os._exit(7)


atexit.register(_peak_line)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))   # a stop runs atexit too (the runner's KILL 20 s later frees memory)
threading.Thread(target=_guard, daemon=True).start()

DM = os.path.expanduser("~/FableGod/decision_map")
sys.path.insert(0, DM)
import dm_ablations as A  # noqa: E402

n = len([f for f in os.listdir(MASKS) if f.endswith(".npz")])
assert n == 36, f"{n} mask files in {MASKS}"
if ARM in A.ABLATIONS:
    print(f"[bg] {ARM} is already registered in dm_ablations.py; using that entry", flush=True)
else:
    A.ABLATIONS[ARM] = A.q1_union(MASKS)
src = os.path.join(DM, "run_map.py")
sys.argv = [src, ARM, "--max", "12", "--skip-done"]
print(f"[bg] running {src} {' '.join(sys.argv[1:])} (masks {MASKS}, DM_C {os.environ.get('DM_C', '51')}, "
      f"self-guard {LIM} GB)", flush=True)
exec(compile(open(src).read(), src, "exec"), {"__name__": "__main__", "__file__": src})
