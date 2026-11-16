"""mtlrec.py (Breakthrough, Mon 16 Nov 2026 21:0x; MODULE 1a, docs/MODULE1A.md): ctypes wrapper for libmtlrec.dylib.
Call load() BEFORE `import mlx.core` (the swizzles must be in place before MLX creates its encoders)."""
import ctypes
import os

_L = None


def load():
    global _L
    _L = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmtlrec.dylib"))
    _L.rec_counts.argtypes = [ctypes.POINTER(ctypes.c_long)]
    _L.rec_replay.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    _L.rec_replay.restype = ctypes.c_int
    return _L


def start():
    _L.rec_start()


def stop():
    _L.rec_stop()


def clear():
    _L.rec_clear()


def counts():
    a = (ctypes.c_long * 6)()
    _L.rec_counts(a)
    return dict(ops=a[0], encoders=a[1], dispatches=a[2], blits=a[3], commits=a[4], set_bytes_total=a[5])


def replay(serial=0):
    o = (ctypes.c_double * 5)()
    rc = _L.rec_replay(int(serial), o)
    return rc, dict(cpu_encode_us=o[0] / 1e3, cpu_commit_wait_us=o[1] / 1e3, gpu_us=o[2] / 1e3, status=int(o[3]),
                    dispatches=int(o[4]))
