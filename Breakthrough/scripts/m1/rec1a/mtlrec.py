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


OPN = {1: "ENC", 2: "BLIT", 3: "END", 4: "COMMIT", 5: "PSO", 6: "BUF", 7: "BUFOFF", 8: "BYTES", 9: "TGMEM", 10: "DTG",
       11: "DT", 12: "BAR", 13: "BARR", 14: "FWAIT", 15: "FUPD", 16: "USE", 17: "BCOPY", 18: "BFILL"}


def opcounts():
    _L.rec_opcounts.argtypes = [ctypes.POINTER(ctypes.c_long)]
    a = (ctypes.c_long * 20)()
    _L.rec_opcounts(a)
    return {OPN.get(k, str(k)): a[k] for k in range(20) if a[k]}


def build_icb():
    _L.rec_build_icb.restype = ctypes.c_long
    _L.rec_icb_error.restype = ctypes.c_char_p
    n = _L.rec_build_icb()
    return n, (_L.rec_icb_error() or b"").decode()


def replay_icb():
    _L.rec_replay_icb.argtypes = [ctypes.POINTER(ctypes.c_double)]
    _L.rec_replay_icb.restype = ctypes.c_int
    o = (ctypes.c_double * 5)()
    rc = _L.rec_replay_icb(o)
    return rc, dict(cpu_encode_us=o[0] / 1e3, cpu_commit_wait_us=o[1] / 1e3, gpu_us=o[2] / 1e3, status=int(o[3]),
                    dispatches=int(o[4]))


def build_rset():
    _L.rec_build_rset.restype = ctypes.c_long
    return _L.rec_build_rset()


def icb_nres():
    _L.rec_icb_nres.restype = ctypes.c_long
    return _L.rec_icb_nres()


def replay_icb2(use_rs):
    _L.rec_replay_icb2.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    _L.rec_replay_icb2.restype = ctypes.c_int
    o = (ctypes.c_double * 9)()
    rc = _L.rec_replay_icb2(int(use_rs), o)
    return rc, dict(cpu_encode_us=o[0] / 1e3, cpu_commit_wait_us=o[1] / 1e3, gpu_us=o[2] / 1e3, status=int(o[3]),
                    dispatches=int(o[4]), cb_us=o[5] / 1e3, enc_us=o[6] / 1e3, use_us=o[7] / 1e3, exec_us=o[8] / 1e3)


# ---- setBytes inspection and per-check patching (milestone 2)
def bytes_dump():
    """Every setBytes op of the record: a list of dicts (k, length, bind index, first dispatch ordinal, payload bytes)."""
    _L.rec_bytes_n.restype = ctypes.c_long
    n = _L.rec_bytes_n()
    tot = counts()["set_bytes_total"]
    meta = (ctypes.c_long * (4 * max(n, 1)))()
    buf = ctypes.create_string_buffer(max(tot, 1))
    _L.rec_bytes_meta.argtypes = [ctypes.POINTER(ctypes.c_long), ctypes.c_long, ctypes.c_char_p, ctypes.c_long]
    _L.rec_bytes_meta.restype = ctypes.c_long
    got = _L.rec_bytes_meta(meta, n, buf, max(tot, 1))
    if got != n:
        raise RuntimeError(f"rec_bytes_meta returned {got}, expected {n}")
    raw = buf.raw
    return [dict(k=k, length=meta[4 * k], bind=meta[4 * k + 1], disp=meta[4 * k + 2],
                 data=raw[meta[4 * k + 3]:meta[4 * k + 3] + meta[4 * k]]) for k in range(n)]


def disp_names():
    _L.rec_disp_names.argtypes = [ctypes.c_char_p, ctypes.c_long]
    _L.rec_disp_names.restype = ctypes.c_long
    cap = 1 << 16
    while True:
        buf = ctypes.create_string_buffer(cap)
        m = _L.rec_disp_names(buf, cap)
        if m >= 0:
            return buf.raw[:m].decode().splitlines()
        cap *= 4


def bytes_patch(k, off, data):
    _L.rec_bytes_patch.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_char_p, ctypes.c_long]
    _L.rec_bytes_patch.restype = ctypes.c_int
    return _L.rec_bytes_patch(k, off, bytes(data), len(data))


RULE_T = {"int32": 0, "float32": 1, "int64": 2}


def rule_add(k, off, type="int32", a=1, b=0, m=0, s=1, c=0):
    """Word `off` of setBytes op k gets ((seq * a + b) mod m) * s + c per check (no mod when m <= 0)."""
    _L.rec_rule_add.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_int] + [ctypes.c_long] * 5
    _L.rec_rule_add.restype = ctypes.c_long
    return _L.rec_rule_add(k, off, RULE_T[type], a, b, m, s, c)


def rules_clear():
    _L.rec_rules_clear()


def rules_apply(seq):
    _L.rec_rules_apply.argtypes = [ctypes.c_long]
    _L.rec_rules_apply.restype = ctypes.c_long
    return _L.rec_rules_apply(seq)
