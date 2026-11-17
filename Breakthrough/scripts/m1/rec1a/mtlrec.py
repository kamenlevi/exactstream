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


TARGET_T = {"int32": 0, "float32": 1, "int64": 2, "bufoff": 3, "bufword32": 4, "bufword64": 5}


def rule_add2(type, k, off=0, var=0, a=1, b=0, d=0, m=0, s=1, c=0):
    """value = (((vars[var] * a + b) div d) mod m) * s + c (floor div; none when d <= 0; no mod when m <= 0).
    type int32 / float32 / int64: word `off` of setBytes op k; type bufoff: the byte offset of BUF op k (record index)."""
    _L.rec_rule_add2.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int] + [ctypes.c_long] * 6
    _L.rec_rule_add2.restype = ctypes.c_long
    return _L.rec_rule_add2(TARGET_T[type], k, off, var, a, b, d, m, s, c)


def rules_apply_v(vals):
    _L.rec_rules_apply_v.argtypes = [ctypes.POINTER(ctypes.c_long), ctypes.c_long]
    _L.rec_rules_apply_v.restype = ctypes.c_long
    arr = (ctypes.c_long * len(vals))(*vals)
    return _L.rec_rules_apply_v(arr, len(vals))


OPS_COLS = ["op", "i0", "n0", "n1", "gw", "gh", "gd", "tw", "th", "td", "o0", "o1", "nb"]


def ops_dump():
    """Every op of the record as an int64 array (n, 13); columns OPS_COLS."""
    import numpy as np
    n = counts()["ops"]
    buf = (ctypes.c_long * (13 * max(n, 1)))()
    _L.rec_ops_dump.argtypes = [ctypes.POINTER(ctypes.c_long), ctypes.c_long]
    _L.rec_ops_dump.restype = ctypes.c_long
    got = _L.rec_ops_dump(buf, n)
    if got != n:
        raise RuntimeError(f"rec_ops_dump returned {got}, expected {n}")
    return np.ctypeslib.as_array(buf).reshape(-1, 13)[:n].copy()


def icb_nseg():
    _L.rec_icb_nseg.restype = ctypes.c_long
    return _L.rec_icb_nseg()


def replay_icb_seg(use_rs=0):
    """One command buffer per recorded command buffer (segments at the recorded commits), committed back to back."""
    _L.rec_replay_icb_seg.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    _L.rec_replay_icb_seg.restype = ctypes.c_int
    o = (ctypes.c_double * 6)()
    rc = _L.rec_replay_icb_seg(int(use_rs), o)
    return rc, dict(cpu_encode_commit_us=o[0] / 1e3, wait_us=o[1] / 1e3, gpu_span_us=o[2] / 1e3, status=int(o[3]),
                    segments=int(o[4]), commands=int(o[5]))


def toyloader_start(row_addr, rel_addr, mul=3, add=1):
    _L.rec_toyloader_start.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
    _L.rec_toyloader_start.restype = ctypes.c_int
    return _L.rec_toyloader_start(row_addr, rel_addr, mul, add)


def toyloader_stop():
    _L.rec_toyloader_stop.restype = ctypes.c_long
    return _L.rec_toyloader_stop()


def toyloader_count():
    _L.rec_toyloader_count.restype = ctypes.c_long
    return _L.rec_toyloader_count()


def buf_len(op):
    _L.rec_buf_len.argtypes = [ctypes.c_long]
    _L.rec_buf_len.restype = ctypes.c_long
    return _L.rec_buf_len(op)


def buf_read(op, off=0, n=None):
    """The current contents of the buffer bound by BUF op `op` (shared storage), bytes [off, off + n)."""
    L = buf_len(op)
    if L < 0:
        return None
    n = L - off if n is None else n
    b = ctypes.create_string_buffer(max(n, 1))
    _L.rec_buf_read.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_char_p, ctypes.c_long]
    _L.rec_buf_read.restype = ctypes.c_long
    return b.raw[:n] if _L.rec_buf_read(op, off, b, n) == n else None


def buf_write(op, data, off=0):
    """Write bytes into the buffer bound by BUF op `op` (shared storage) at byte `off`."""
    _L.rec_buf_write.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_char_p, ctypes.c_long]
    _L.rec_buf_write.restype = ctypes.c_long
    data = bytes(data)
    return _L.rec_buf_write(op, off, data, len(data))


def buf_snap(op):
    """The bind-time snapshot (contents at the buffer's first binding in the record, before the GPU ran) of the buffer
    bound by BUF op `op`, for shared buffers <= 4 KB; None if there is none."""
    b = ctypes.create_string_buffer(4096)
    _L.rec_buf_snap.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_long]
    _L.rec_buf_snap.restype = ctypes.c_long
    n = _L.rec_buf_snap(op, b, 4096)
    return b.raw[:n] if n >= 0 else None
