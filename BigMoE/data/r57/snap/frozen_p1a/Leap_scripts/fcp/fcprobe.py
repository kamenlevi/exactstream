"""FCP (Leap, Sat 2026-11-07 08:3x; the orchestrator 08:1x: scope FIRSTCHK). An in-process probe of each prompt's check 0
on a copy of BigMoE's v1' screen (fcp_run.sh patches the copy: install() after `import mlx.core as mx`, mark() at the
check's c1 / cb / ce stamps). Nothing here changes a computed value; FCP_NOCLEAR=1 is the one intervention (below).
Per check window (checks 0-3 of every prompt), at c1 (build start), cb (graph built) and ce (eval done):
  - wall, main-thread CPU (time.thread_time), process CPU (getrusage), every thread's CPU (proc_pidinfo, with names)
  - this task's events (faults, pageins, mach messages, syscalls, context switches) and the host's VM counters
    (zero fills, pageins, decompressions, purges, wired pages)
  - MTLCompilerService CPU (a Metal pipeline build runs there, outside this process), MLX active / cache memory
  - timed calls inside the window: mx.eval / async_eval / clear_cache, bigmoe tier.prefill_done / _reuse_drop,
    stream._stage_drop, Gate.new_pass (wall, thread CPU, caller line)
  - a 1 kHz sampler of the main thread's Python stack (where the wall goes, line by line; a gap = the GIL held in C)
FCP_NOCLEAR=1: tier._reuse_drop() drops R-52's kept prefill buffers without mx.clear_cache() (prefill_done runs inside
check 0's graph build, at the first MoE layer). Memory management only: the same kernels on the same inputs.
-> FCP_OUT/<tag>_fcp.json (fcp_read.py reads it)."""
import atexit
import ctypes
import json
import os
import resource
import sys
import threading
import time

OUT = os.environ.get("FCP_OUT", os.path.expanduser("~/Leap/data/fcp"))
TAG = os.environ.get("R57_TAG", "fcp")
NCHK = int(os.environ.get("FCP_NCHK", "4"))
HZ = float(os.environ.get("FCP_HZ", "1000"))
NOCLEAR = os.environ.get("FCP_NOCLEAR") == "1"
WIRED = os.environ.get("FCP_WIRED") == "1"
COMMIT = os.environ.get("FCP_COMMIT") == "1"   # PLAN 220: split each commit (ref2 lazy pre / MTPLX commit + the PLE forward / post)
CSYNC = os.environ.get("FCP_CSYNC") == "1"
SEAMS = os.environ.get("FCP_SEAMS") == "1"     # PLAN 220 (4): every eval / async_eval + the cycle marks, for the GPU-idle seams
SEAM_RUN0 = int(os.environ.get("FCP_SEAM_RUN0", "3"))
SEAMREC, MARKS = [], []     # + an eval of the cache states right after it: the commit's GPU time
CREC = []   # the fix candidate: MLX's residency set (mx.set_wired_limit) keeps buffers wired
S = {"on": False, "run": -1, "chk": -1, "phase": None, "wrapped2": False, "cur": None, "mtl": []}
REC, CALLS, SAMP = [], [], []

_c = ctypes.CDLL(None)
_p = ctypes.CDLL("/usr/lib/libproc.dylib")
_p.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
_p.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
_p.proc_name.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
_p.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_c.mach_host_self.restype = ctypes.c_uint32
_c.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_c.task_info.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_HOST = _c.mach_host_self()
_TASK = ctypes.c_uint32.in_dll(_c, "mach_task_self_").value
_U32, _U64 = ctypes.c_uint32, ctypes.c_uint64


class _PTI(ctypes.Structure):                          # struct proc_threadinfo
    _fields_ = [("ut", _U64), ("st", _U64)] + [(f"i{k}", ctypes.c_int32) for k in range(8)] + [("name", ctypes.c_char * 64)]


class _VM(ctypes.Structure):                           # struct vm_statistics64 (HOST_VM_INFO64 = 4)
    _fields_ = [("free", _U32), ("active", _U32), ("inactive", _U32), ("wire", _U32), ("zero_fill", _U64),
                ("reactivations", _U64), ("pageins", _U64), ("pageouts", _U64), ("faults", _U64), ("cow_faults", _U64),
                ("lookups", _U64), ("hits", _U64), ("purges", _U64), ("purgeable", _U32), ("speculative", _U32),
                ("decompressions", _U64), ("compressions", _U64), ("swapins", _U64), ("swapouts", _U64),
                ("compressor", _U32), ("throttled", _U32), ("external", _U32), ("internal", _U32), ("uncomp", _U64)]


class _RU0(ctypes.Structure):                          # struct rusage_info_v0
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("ut", _U64), ("st", _U64)] + [(f"u{k}", _U64) for k in range(8)]


def _threads():
    pid, buf = os.getpid(), (_U64 * 1024)()
    n = max(0, _p.proc_pidinfo(pid, 6, 0, buf, ctypes.sizeof(buf))) // 8     # PROC_PIDLISTTHREADS
    out = {}
    for h in buf[:n]:
        ti = _PTI()
        if _p.proc_pidinfo(pid, 5, h, ctypes.byref(ti), ctypes.sizeof(ti)) == ctypes.sizeof(ti):   # PROC_PIDTHREADINFO
            out[str(h)] = [ti.ut + ti.st, ti.name.decode(errors="replace")]
    return out


def _vm():
    v, n = _VM(), _U32(ctypes.sizeof(_VM) // 4)
    _c.host_statistics64(_HOST, 4, ctypes.byref(v), ctypes.byref(n))
    return {k: int(getattr(v, k)) for k in ("zero_fill", "pageins", "faults", "cow_faults", "purges", "decompressions",
                                            "compressions", "swapins", "pageouts", "wire", "free", "purgeable")}


def _task_events():
    v, n = (ctypes.c_int32 * 8)(), _U32(8)
    _c.task_info(_TASK, 2, v, ctypes.byref(n))                                 # TASK_EVENTS_INFO
    return dict(zip(("faults", "pageins", "cow_faults", "msg_sent", "msg_recv", "sys_mach", "sys_unix", "csw"), list(v)))


def _ru(pid):
    r = _RU0()
    return int(r.ut + r.st) if _p.proc_pid_rusage(pid, 0, ctypes.byref(r)) == 0 else -1


def _mtl_pids():
    buf = (ctypes.c_int * 8192)()
    n = max(0, _p.proc_listallpids(buf, ctypes.sizeof(buf)))
    out, nm = [], ctypes.create_string_buffer(256)
    for pid in buf[:n]:
        if pid > 0 and _p.proc_name(pid, nm, 256) > 0 and b"MTLCompiler" in nm.value:
            out.append(int(pid))
    return out


def _snap(full):
    ru = resource.getrusage(resource.RUSAGE_SELF)
    d = dict(t=time.perf_counter(), tt=time.thread_time(), proc=ru.ru_utime + ru.ru_stime, ev=_task_events(), vm=_vm(),
             ru_self=_ru(os.getpid()), mtl={str(p): _ru(p) for p in S["mtl"]})
    try:
        import mlx.core as mx
        d["mlx"] = [mx.get_active_memory(), mx.get_cache_memory(), mx.get_peak_memory()]
    except Exception:
        d["mlx"] = None
    if full:
        d["thr"] = _threads()
    return d


def _caller(depth=2):
    f = sys._getframe(depth)
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"


def _wrap(kind, fn):
    def w(*a, **k):
        if SEAMS and S["run"] >= SEAM_RUN0 and kind in ("mx.eval", "mx.async_eval"):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                SEAMREC.append((kind, t0, time.perf_counter(), _caller(), S["run"], S["chk"]))
        if not S["on"]:
            return fn(*a, **k)
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return fn(*a, **k)
        finally:
            CALLS.append((S["run"], S["chk"], S["phase"], kind, t0, 1e3 * (time.perf_counter() - t0),
                          1e3 * (time.thread_time() - c0), _caller()))
    w.__wrapped__ = fn
    return w


def _sampler():
    main = threading.main_thread().ident
    dt = 1.0 / HZ
    while True:
        time.sleep(dt)
        if not S["on"] and not (SEAMS and S["run"] >= SEAM_RUN0):
            continue
        f = sys._current_frames().get(main)
        key = []
        while f is not None and len(key) < 5:
            key.append(f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}:{f.f_code.co_name}")
            f = f.f_back
        SAMP.append((time.perf_counter(), S["run"], S["chk"], S["phase"], " < ".join(key)))


def _cwrap(kind, fn):                                  # a commit piece: always recorded, tagged with (keep, verified)
    def w(*a, **k):
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return fn(*a, **k)
        finally:
            CREC.append((S["run"], S["chk"], kind, S.get("ck"), t0, 1e3 * (time.perf_counter() - t0),
                         1e3 * (time.thread_time() - c0), None))
    w.__wrapped__ = fn
    return w


def _layer_wrap(fn):                                   # a decoder layer's forward, recorded only inside the commit
    def w(self_, *a, **k):
        if not S.get("in_commit"):
            return fn(self_, *a, **k)
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return fn(self_, *a, **k)
        finally:
            CREC.append((S["run"], S["chk"], "layer_fwd", S.get("ck"), t0, 1e3 * (time.perf_counter() - t0),
                         1e3 * (time.thread_time() - c0), None))
    w.__wrapped__ = fn
    return w


def _commit_top(fn):
    def w(self, cache, snapshot_states, *, keep_tokens, verified_tokens):
        if not S.get("lw"):
            S["lw"] = True
            for T_ in {type(l) for l in self.layers if getattr(l, "ple", None) is not None}:
                T_.__call__ = _layer_wrap(T_.__call__)
        S["ck"] = (int(keep_tokens), int(verified_tokens))
        t0, c0 = time.perf_counter(), time.thread_time()
        S["in_commit"] = True
        try:
            ok = fn(self, cache, snapshot_states, keep_tokens=keep_tokens, verified_tokens=verified_tokens)
        finally:
            S["in_commit"] = False
        t1, c1_ = time.perf_counter(), time.thread_time()
        sync = None
        if CSYNC:
            import mlx.core as mx
            mx.eval([c.state for c in cache if c is not None and hasattr(c, "state")])
            sync = 1e3 * (time.perf_counter() - t1)
        CREC.append((S["run"], S["chk"], "commit", S["ck"], t0, 1e3 * (t1 - t0), 1e3 * (c1_ - c0), sync))
        return ok
    w.__wrapped__ = fn
    return w


def _wrap_late():                                      # bigmoe's modules exist by the first check
    S["wrapped2"] = True
    tier, stream, gated = (sys.modules.get(f"bigmoe.{m}") for m in ("tier", "stream", "gated"))
    if tier is not None:
        if NOCLEAR:
            def _reuse_drop_noclear(_t=tier):          # FCP_NOCLEAR: the buffers go, MLX's buffer cache stays
                if _t._REUSE:
                    _t._REUSE.clear()
            tier._reuse_drop = _reuse_drop_noclear
            print("[fcp] FCP_NOCLEAR: tier._reuse_drop() keeps MLX's buffer cache (no mx.clear_cache())", flush=True)
        tier._reuse_drop = _wrap("tier._reuse_drop", tier._reuse_drop)
        tier.prefill_done = _wrap("tier.prefill_done", tier.prefill_done)
    if stream is not None:
        stream._stage_drop = _wrap("stream._stage_drop", stream._stage_drop)
    if gated is not None and hasattr(gated, "Gate"):
        gated.Gate.new_pass = _wrap("Gate.new_pass", gated.Gate.new_pass)
    if COMMIT:
        r2, Qm = sys.modules.get("ref2"), sys.modules.get("mtplx.models.qwen4_exp")
        if r2 is not None and Qm is not None:
            r2._lz_commit_pre = _cwrap("lz_pre", r2._lz_commit_pre)
            r2._lz_commit_post = _cwrap("lz_post", r2._lz_commit_post)
            if "commit" in r2._ORIG:
                r2._ORIG["commit"] = _cwrap("mtplx_commit", r2._ORIG["commit"])
            Qm.Qwen4ExpTextModel.commit_verified_window = _commit_top(Qm.Qwen4ExpTextModel.commit_verified_window)
        print(f"[fcp] FCP_COMMIT: ref2 {r2 is not None}, qwen4_exp {Qm is not None}, CSYNC {CSYNC}", flush=True)
    print(f"[fcp] wrapped: tier {tier is not None}, stream {stream is not None}, gated {gated is not None}", flush=True)


def install():
    import mlx.core as mx
    for k in ("eval", "async_eval", "clear_cache"):
        setattr(mx, k, _wrap(f"mx.{k}", getattr(mx, k)))
    S["mtl"] = _mtl_pids()
    if WIRED:                                          # FCP_WIRED=1 (the fix candidate): mlx-lm's wired_limit, for the whole run
        lim = int(mx.device_info()["max_recommended_working_set_size"])
        print(f"[fcp] FCP_WIRED: mx.set_wired_limit({lim / 1e9:.2f} GB), was {mx.set_wired_limit(lim) / 1e9:.2f} GB", flush=True)
    threading.Thread(target=_sampler, name="fcp-sampler", daemon=True).start()
    atexit.register(dump)
    print(f"[fcp] installed: {NCHK} checks per prompt, sampler {HZ:.0f} Hz, MTLCompilerService pids {S['mtl']}, "
          f"NOCLEAR {NOCLEAR}", flush=True)


def mark(kind, chk=None):
    """c1: check `chk` starts (chk 0 = a new prompt); cb: graph built; ce: eval done; c0 / c2: cycle start / commit sent."""
    if not S["wrapped2"]:
        _wrap_late()
    if SEAMS:
        MARKS.append((kind, time.perf_counter(), S["run"] + (1 if kind == "c1" and chk == 0 else 0), chk if kind == "c1" else S["chk"]))
    if kind in ("c0", "c2", "rb0", "rb1"):
        return
    if kind == "c1":
        if chk == 0:
            S["run"] += 1
        S["chk"] = chk
        if chk >= NCHK:
            return
        if chk == 0:
            S["mtl"] = _mtl_pids()
        S["cur"] = dict(run=S["run"], chk=chk, c1=_snap(True))
        S["phase"], S["on"] = "build", True
    elif S["cur"] is not None and S["chk"] < NCHK:
        if kind == "cb":
            S["cur"]["cb"] = _snap(False)
            S["phase"] = "wait"
        elif kind == "ce":
            S["on"] = False
            S["mtl"] = sorted(set(S["mtl"]) | set(_mtl_pids()))   # a compiler service started inside the window counts from 0
            S["cur"]["ce"] = _snap(True)
            REC.append(S["cur"])
            S["cur"], S["phase"] = None, None


def dump():
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{TAG}_fcp.json")
    json.dump(dict(rec=REC, calls=CALLS, samp=SAMP, noclear=NOCLEAR, crec=CREC, csync=CSYNC, seams=SEAMREC, marks=MARKS), open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print(f"[fcp] {len(REC)} check windows, {len(CALLS)} calls, {len(SAMP)} samples -> {p}", flush=True)
    r2 = sys.modules.get("ref2")
    if r2 is not None:
        print(f"[fcp] ref2 COUNTS: gdn_tg {r2.COUNTS.get('gdn_tg')}, eprep {r2.COUNTS.get('eprep')}, ooo_1c {r2.COUNTS.get('ooo_1c')}, "
              f"ooo_1c_qom {r2.COUNTS.get('ooo_1c_qom')}; CFG gdn_tg {r2.CFG.get('gdn_tg')}, eprep {r2.CFG.get('eprep')}", flush=True)
