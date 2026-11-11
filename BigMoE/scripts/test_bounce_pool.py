"""Bounce pool fix (NewIdeas 10-29 GO, after Fit12's R-45T): footprint across idle gaps, ext only (no GPU work).
One records3m base pool (layer 23, M-01 MSB part, 64 slots). Each cycle ensure()s 64 experts that are not resident
(one dispatch_apply over 64 read jobs, so about every GCD worker reads one), then idles GAP s, longer than GCD's
idle-worker reap time. Prints the process's phys_footprint now, its thread count and the ext's bounce pool after each
cycle. Old ext (thread_local bounce): the footprint grows by about (new workers) x 2.05 MB per gap. New ext (shared
pool): flat after the first cycle. The loaded slots are compared bitwise with the same experts read by the first cycle's
pass (the same bytes whatever buffer carried them).
usage: BIGMOE_M01=1 BIGMOE_EXT_BUILD=build_bp test_bounce_pool.py [cycles=6] [gap_s=12]"""
import ctypes, json, os, sys, time
import numpy as np
assert os.environ.get("BIGMOE_M01"), "set BIGMOE_M01=1"
sys.path.insert(0, os.path.expanduser("~/BigMoE/ext/" + os.environ.get("BIGMOE_EXT_BUILD", "build")))
import mlx.core as mx
import bigmoe_ext as ext

N_CYC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
MD = os.path.expanduser("~/BigMoE/models/bare")
L, SLOTS, B = 23, 64, 64                                      # 64 slots: each cycle replaces them all


class _RU4(ctypes.Structure):                                  # rusage_info_v4: u7 = phys_footprint now
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(35)]


class _TI(ctypes.Structure):                                   # proc_taskinfo (PROC_PIDTASKINFO = 4)
    _fields_ = [(f, ctypes.c_uint64) for f in ("vsz", "rss", "tu", "ts", "thu", "ths")] + [
        (f, ctypes.c_int32) for f in ("policy", "faults", "pageins", "cow", "msent", "mrecv", "mach", "unix", "csw",
                                      "threads", "running", "prio")]


libc = ctypes.CDLL(None)


def fp_mb():
    ru = _RU4()
    return ru.u7 / 1e6 if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


def threads():
    ti = _TI()
    return ti.threads if libc.proc_pidinfo(os.getpid(), 4, ctypes.c_uint64(0), ctypes.byref(ti), ctypes.sizeof(ti)) > 0 else -1


r3 = json.load(open(os.path.join(MD, "records3", "index.json")))["layers"][str(L)]
assert r3.get("layout", "row-major") == "row-major", "records3 is converted (docs/PLAN_records3_convert.md): this script reads row-major records only"
rm = json.load(open(os.path.join(MD, "records3m", "index.json")))["layers"][str(L)]
sizes = r3["msb_piece_size"]
P = [mx.zeros((SLOTS, n // 2), mx.uint16) for n in sizes]; mx.eval(*P)
Lx = ext.add_layer(SLOTS, r3["E"], [0] * 9, [0] * 9, list(sizes), P)
ext.set_records_part(Lx, os.path.join(MD, "records3m", rm["file"]), rm["recsz"], 0, rm["msb_sz"])
bs = (lambda: ext.bounce_stats()) if hasattr(ext, "bounce_stats") else (lambda: None)
rng = np.random.default_rng(7)
order = rng.permutation(r3["E"])
first = {}                                                     # expert -> its 9 slot pieces from cycle 0 (bytes)
ok, rows = True, []
print(f"ext {os.environ.get('BIGMOE_EXT_BUILD', 'build')}: {N_CYC} cycles of {B} reads (layer {L}, M-01 MSB "
      f"{rm['msb_sz'] / 1e6:.2f} MB each), {GAP:.0f} s idle gaps; threads at start {threads()}", flush=True)
for c in range(N_CYC):
    ids = [int(e) for e in order[(c % 2) * B:(c % 2) * B + B]]   # alternate two sets: every cycle's 64 are misses
    t0 = time.perf_counter()
    sl = np.array(ext.ensure(Lx, mx.array(ids, dtype=mx.int32)))
    dt = time.perf_counter() - t0
    for e, s in zip(ids, sl):
        got = [np.array(a[int(s)]).tobytes() for a in P]
        if e in first:
            ok &= got == first[e]
        else:
            first[e] = got
    th_busy, f_after = threads(), fp_mb()
    time.sleep(GAP)
    rows.append(dict(cycle=c, load_s=round(dt, 3), fp_after_mb=round(f_after, 1), threads_after=th_busy,
                     threads_after_gap=threads(), fp_after_gap_mb=round(fp_mb(), 1), bounce=bs()))
    print(json.dumps(rows[-1]), flush=True)
grow = rows[-1]["fp_after_mb"] - rows[0]["fp_after_mb"]
print(f"footprint after the last cycle - after the first: {grow:+.1f} MB over {N_CYC - 1} gaps "
      f"({grow / max(N_CYC - 1, 1):+.1f} MB per gap); re-read experts bitwise equal: {ok}", flush=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
