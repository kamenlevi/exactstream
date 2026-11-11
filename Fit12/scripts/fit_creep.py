"""Fit12 R-45 CREEPMAP (Wed 28 Oct 2026; NewIdeas QUEUE R-45, design Breakthrough; PLAN.md 'R-45', written before any run):
which suspect holds the per-row footprint creep of EvalSuite's fold-exact math process (R-41C: row starts 10.12 -> 10.95 GB
over rows 2-9)? One process as the suite's b12llfx decode stage runs it: bt_b12 c3f957cba6 (read-only, unpatched,
load_b12(0.10, mtp=True)), the suite's process env (cli.gpu + evalsuite/env.py for SHIP runs), rows decoded as
decode._gen_mtp does. The first F12_NROWS (8) math rows of R-41C's fold-exact process, in its order. Per row: tokens vs the
suite's row, footprint (start / max / after), MLX active / cache, the n-gram hot-row caches (rows, counted bytes, real bytes
= the fetch batches the cached rows are views into), then gc.collect() with a census of the garbage and the footprint
again; swap / compressor; vmmap --summary after the first and last rows. Usage: [BigMoE venv] fit_creep.py TAG"""
import ctypes, gc, hashlib, json, os, subprocess, sys, threading, time

POP = ("EVALSUITE_FRUGAL", "EVALSUITE_SHIP", "EVALSUITE_FAST24", "B12_LATE3", "B12_W", "B12_NORM", "BIGMOE_LOOP_GUARD", "B12_CM",
       "B12_CL", "B12_ENGINE", "B12_TRIMS", "B12_PFR", "B12_ALLWAIT", "B12_CACHE_MB", "B12_FOLD", "B12_FOLD_CM", "B12_FOLD_ROWS",
       "B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_PFR", "B12_FOLD_MOV", "BIGMOE_PF_GROUP", "BIGMOE_ASYNC_EVERY",
       "BIGMOE_PF_PREFILL", "BIGMOE_PF_FULLSCR", "MTPLX_NGRAM_HOT_MB")
ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "EVALSUITE_SHIP": "1",  # SHIP_AR
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1",              # B12_ENV b12ll
       "B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0", "B12_CACHE_MB": "64",                               # FOLD_SHIP
       "B12_ENGINE": "v3m", "B12_FOLD": "1", "B12_PFR_GRID": "keep",                                             # b12llfx
       "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80", "BIGMOE_FRSPEC": "1",                # SHIP_MTP
       "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0",
       "MTPLX_NGRAM_PREWARM": "off", "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0",                # env.py
       "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"}
ARM = os.environ.get("F12_ARM", "fx")             # fx = b12llfx (fold-exact, R-45); ff = b12llff (fold-fast, R-45F)
if ARM == "ff":                                  # cli.B12_ENV["b12llff"]
    ENV.update(B12_FOLD_AW="1", B12_FOLD_CM="1", B12_PFR_GRID="balanced")
if ARM == "ff3":                                 # M-5b: cli.B12_ENV["b12llff3"] (decision 5's measured config, on 677c6558b6)
    ENV.update(B12_ENGINE="v3o", B12_FOLD_AW="1", B12_FOLD_CM="2700:60,8600:60,16500:56,32800:47", B12_PFR_GRID="balanced",
               B12_FRS_NOCYCLE="1", B12_PFR_MIN="512")
for k in POP:
    os.environ.pop(k, None)
ENV["B12_ENGINE"] = os.environ.get("F12_ENGINE", ENV["B12_ENGINE"])   # R-46 (30 Oct): v3n with bt_b12 23b42972ef
if ARM == "live":                                # M-5 (Sun 1 Nov): the live file's own mode defaults (B12_MODE fast: engine, fold
    for k in ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID"):   # pieces, grid) in the suite's process env, as R-60 ran it
        ENV.pop(k, None)
os.environ.update(ENV)
if os.environ.get("F12_HOT_MB"):                 # R-46 HOTCAP: the n-gram hot-row cap (MTPLX_NGRAM_HOT_MB, counted at 100 B / row)
    os.environ["MTPLX_NGRAM_HOT_MB"] = os.environ["F12_HOT_MB"]
for _kv in filter(None, os.environ.get("F12_SETENV", "").split(",")):   # M-5b: one piece changed (B12_ENGINE=v3o, B12_MODE=exact)
    os.environ[_kv.split("=", 1)[0]] = _kv.split("=", 1)[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "creep"
NR = int(os.environ.get("F12_NROWS", "8"))
CAPT = int(os.environ.get("F12_CAP", "0"))           # smoke tests only: cap the tokens, compare the prefix
OUTD = os.environ.get("F12_OUT", os.path.expanduser("~/Fit12/data/scr/creep"))
ES = os.path.expanduser("~/EvalSuite")
BT = os.environ.get("F12_BT", os.path.expanduser("~/Breakthrough/scripts/bt_b12.py"))   # R-45T: a pinned ce6d4c63d2 copy
EOS = {248044, 248046}                                # = evalsuite/decode.EOS


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def fp():                                            # (now, lifetime peak) physical footprint, GB
    ru = _RU()
    return (ru.u7 / 1e9, ru.u28 / 1e9) if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else (float("nan"),) * 2


PEAK = [0.0]
PK = [0.0, float("nan"), float("nan")]             # M-5b: (time, MLX active GB, MLX cache GB) at PEAK's last rise


def _sampler():
    while True:
        f = fp()[0]
        if f > PEAK[0]:
            PEAK[0] = f
            m, mlx = sys.modules.get("mlx.core"), [float("nan")] * 2
            try:                                     # M-5b I (12:34): mlx.core sits in sys.modules while it is still
                mlx = [m.get_active_memory() / 1e9, m.get_cache_memory() / 1e9]   # importing: never let that kill the
            except Exception:                        # sampler (it voided arm I's row maxima)
                pass
            PK[:] = [time.perf_counter()] + mlx
        time.sleep(0.02)


def pressure():                                      # (swap used GB, compressor GB): the Mac's, to read drops by
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()
        used = float(sw[sw.index("used") + 2].rstrip("M")) / 1e3
        vs = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines()
        ps = int(vs[0].split("page size of")[1].split()[0])
        comp = next(int(l.split(":")[1].strip().rstrip(".")) for l in vs if l.startswith("Pages occupied by compressor"))
        return used, comp * ps / 1e9
    except Exception:
        return float("nan"), float("nan")


sha = hashlib.sha256(open(BT, "rb").read()).hexdigest()[:10]
assert sha in ("c3f957cba6", "ce6d4c63d2", "23b42972ef", "677c6558b6", "cf8bd7840a", "31e4333196", "8db71cc537"), f"bt_b12.py is {sha}: R-41C rows on c3f957cba6 (ce6d4c63d2 + R-45 fix; 23b42972ef + v3n M-01; 677c6558b6 = live v3o, R-46; cf8bd7840a = live fold-fast default, M-5; 31e4333196 = cf8b with default v3o again; 8db71cc537 = v3p + F12_R53A=0, M-5d)"
REF = [json.loads(l) for l in open(os.path.join(ES, "results", "b12llff" if ARM == "ff" else "b12llfx", "decode.jsonl"))]
SET = os.environ.get("F12_SET", "math")             # M-5: math (R-41C's 16) or agent (its 29); live agent refs = b12llff3
if ARM == "live" and SET == "agent":
    REF = [json.loads(l) for l in open(os.path.join(ES, "results", "b12llff3", "decode.jsonl"))]
if os.environ.get("F12_REFARM"):                 # HT1: the suite arm whose rows are the token refs (b12llff4 = live 8db7)
    REF = [json.loads(l) for l in open(os.path.join(ES, "results", os.environ["F12_REFARM"], "decode.jsonl"))]
REF = [r for r in REF if r["set"] == SET]
ROWS = [int(x) for x in os.environ.get("F12_ROWS", "").split(",") if x]   # R-45S: suite math rows by number (1-based), repeats allowed
KEYS = [x for x in os.environ.get("F12_KEYS", "").split(",") if x]   # HT1: rows by key suffix, in this order
SEL = ([next(r for r in REF if r["key"].endswith("/" + q)) for q in KEYS] if KEYS else
       [REF[i - 1] for i in ROWS] if ROWS else REF[:NR])
NR = len(SEL)
VMALL = os.environ.get("F12_VMALL") == "1"         # R-45S: vmmap after every row
MHIST = os.environ.get("F12_MHIST") == "1"         # R-45T: malloc_history -allBySize after the first and last rows
MHROWS = {int(x) for x in os.environ.get("F12_MHIST_ROWS", "").split(",") if x}   # R-45U: after these rows instead
TOOLENV = {k: v for k, v in os.environ.items() if not k.startswith("MallocStackLogging")}   # the tools don't log themselves
ITEMS = {it["key"]: it for it in map(json.loads, open(os.path.join(ES, "data", "items", f"dec_{SET}.jsonl")))}   # M-5: the set's items
for r in SEL:
    assert len(ITEMS[r["key"]]["prompt_ids"]) == r["n_prompt"], r["key"]
SUITE = [] if (ROWS or KEYS) else [r["mem"]["start"] for r in REF[:NR + 1]]   # R-41C's row starts: rows 2..NR+1 start after rows 1..NR

threading.Thread(target=_sampler, daemon=True).start()
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, os.path.dirname(BT))
import bt_b12 as B
assert os.path.realpath(B.__file__) == os.path.realpath(BT), B.__file__
print(f"[{TAG}] bt_b12 {sha} from {BT}, MallocStackLogging {os.environ.get('MallocStackLogging', 'off')}", flush=True)
V3F = os.environ.get("F12_V3F")                    # SG12: an engine tree outside ~/Breakthrough (as fit_scr53.py)
if V3F:
    B.V3F = os.path.expanduser(V3F)
rt, st = B.load_b12(0.10, mtp=True)
if V3F:
    import bigmoe_v3f
    assert os.path.realpath(bigmoe_v3f.__file__).startswith(os.path.realpath(B.V3F) + "/"), bigmoe_v3f.__file__
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model = rt.model
MT = os.environ.get("F12_MEMTRACE") == "1"         # HT1 (Mon 2 Nov): MLX peak per prefill interval (counters only, no syncs)
EV, T0 = [], [0.0]
if MT:
    from bigmoe_v3f import tier as _TT, stream as _SS

    def _mark(kind, L=-1, reset=True, **kw):
        EV.append(dict(t=round(time.perf_counter() - T0[0], 4), kind=kind, L=L, act=mx.get_active_memory() / 1e9,
                       peak=mx.get_peak_memory() / 1e9, cache=mx.get_cache_memory() / 1e9, **kw))
        if reset:                                    # the peak counter restarts per interval (the engine's own
            mx.reset_peak_memory()                   # _mem_in / _mem_out reset it at every trunk MoE call already)

    def _bm_L(f):
        bm = f.f_locals.get("bm")
        return bm["L"] if isinstance(bm, dict) and "L" in bm else -1

    _mi0, _mo0 = _TT._mem_in, _TT._mem_out

    def _mem_in():                                   # the interval before layer L's MoE call (PLE, HC, attention, router)
        _mark("pre", _bm_L(sys._getframe(1)), reset=False)
        _mi0()                                       # the engine reads its stats, then resets the peak counter

    def _mem_out():                                  # layer L's MoE call
        f = sys._getframe(1)
        u, i_ = f.f_locals.get("uniq"), f.f_locals.get("ids")
        _mark("moe", _bm_L(f), reset=False, uniq=int(len(u)) if u is not None else -1, n=int(i_.size) if i_ is not None else -1,
              grouped=int(_TT.PF.get("grouped", 0)), single=int(_TT.PF.get("single", 0)))
        _mo0()
    _TT._mem_in, _TT._mem_out = _mem_in, _mem_out
    _sc0 = _SS._sync_call

    def _sync_call(cls, self, x, indices, bm, *a, **kw):   # non-tier layers' prefill MoE: the MTP head's prompt pass
        if "tier" in bm:
            return _sc0(cls, self, x, indices, bm, *a, **kw)
        _mark("sync_in", bm.get("L", -1))
        try:
            return _sc0(cls, self, x, indices, bm, *a, **kw)
        finally:
            _mark("sync_out", bm.get("L", -1))
    _SS._sync_call = _sync_call
    _ha0 = spec125._MTPHead.append

    def _append(self, hidden, next_ids):             # the head's history (lazy; feed() evaluates its logits after)
        _mark("head_in")
        try:
            return _ha0(self, hidden, next_ids)
        finally:
            _mark("head_out")
    spec125._MTPHead.append = _append
    _bp0 = B.b12_prefill

    def _b12_prefill(*a, **kw):
        _mark("pf_in")
        try:
            return _bp0(*a, **kw)
        finally:
            _mark("pf_out")
    B.b12_prefill = _b12_prefill


def memtrace(k, key, a0, pkt):                       # HT1: intervals ranked by MLX peak; the one holding the sampled max
    lab = lambda e: e["kind"] + (f" L{e['L']}" if e["L"] >= 0 else "")
    prev, rows = None, []
    for e in EV:
        path = ""
        if e["kind"] == "moe":
            path = "grouped" if prev and e["grouped"] > prev["grouped"] else "single" if prev and e["single"] > prev["single"] else "?"
            prev = e
        rows.append((e["peak"], lab(e) + (f" {path} uniq {e['uniq']} n {e['n']}" if e["kind"] == "moe" else ""), e["t"]))
    top = sorted(rows, key=lambda r: -r[0])[:6]
    at = next((r for r in rows if r[2] >= pkt), None)
    bykind = {}
    for e in EV:
        bykind[e["kind"]] = max(bykind.get(e["kind"], 0.0), e["peak"])
    json.dump(EV, open(os.path.join(OUTD, f"{TAG}_events_r{k}.json"), "w"))
    print(f"[{TAG}] {key} MEMTRACE row {k} | row start MLX active {a0:.3f} GB | top MLX peaks: " + "; ".join(
          f"{lb} {pk:.3f} (+{pk - a0:.3f}) at {t:.2f} s" for pk, lb, t in top) + f" | sampled max at {pkt:.2f} s: in "
          + (f"{at[1]} (closes {at[2]:.2f} s, peak {at[0]:.3f})" if at else "the decode or after the last mark")
          + " | max by kind: " + ", ".join(f"{kd} {v:.3f}" for kd, v in bykind.items()), flush=True)


def _tables(m, seen, out):                           # every NGramTable in the module tree (mlx Modules are dicts)
    if id(m) in seen:
        return
    seen.add(id(m))
    if type(m).__name__ == "NGramTable":
        out.append(m)
    for v in (m.values() if isinstance(m, dict) else m if isinstance(m, (list, tuple)) else ()):
        if isinstance(v, (dict, list, tuple)):
            _tables(v, seen, out)


TABS = []
_tables(model, set(), TABS)
SCS = list({id(t._sidecar): t._sidecar for t in TABS if getattr(t, "_sidecar", None) is not None}.values())


def hot():                                           # the hot-row caches: rows, counted GB, data GB (unique fetch batches),
    rows = acc = hits = miss = objb = 0              # full GB (+ the dict, keys, tuples and views: rows are 100 B, their
    bases = {}                                       # Python objects ~5x that; smoke test 28 Oct 23:33)
    for sc in SCS:
        h = sc._hot
        rows += len(h)
        acc += len(h) * sc._hot_row_bytes
        hits += sc.hot_hits
        miss += sc.hot_misses
        objb += sys.getsizeof(h)
        for key, tup in h.items():
            objb += sys.getsizeof(key) + sys.getsizeof(tup)
            for a in tup:
                if a.base is None:                   # a row that owns its bytes
                    bases.setdefault(id(a), sys.getsizeof(a))
                    continue
                objb += sys.getsizeof(a)             # a view: its object; the batch it pins once below
                b = a.base
                while isinstance(b.base, np.ndarray):
                    b = b.base
                bases.setdefault(id(b), sys.getsizeof(b) if b.base is None else b.nbytes)
    data = sum(bases.values())
    return dict(rows=rows, acc=acc / 1e9, real=data / 1e9, full=(data + objb) / 1e9, batches=len(bases), hits=hits, miss=miss)


def gc_census():                                     # collect with DEBUG_SAVEALL, look at the garbage, then free it
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        n = gc.collect()
    finally:
        gc.set_debug(0)
    g = list(gc.garbage)
    gc.garbage.clear()
    seen, npb, mxb, fb, types, fns = set(), 0, 0, 0, {}, {}
    for o in g:
        types[type(o).__name__] = types.get(type(o).__name__, 0) + 1
        if callable(o) and hasattr(o, "__qualname__"):     # whose closures the cycles are (for the fix)
            q = f"{getattr(o, '__module__', '?')}.{o.__qualname__}"
            fns[q] = fns.get(q, 0) + 1
        for x in [o] + gc.get_referents(o):
            if isinstance(x, np.ndarray):
                b = x
                while isinstance(b.base, np.ndarray):
                    b = b.base
                if id(b) not in seen:
                    seen.add(id(b))
                    if b.base is None or isinstance(b.base, (bytes, bytearray)):
                        npb += b.nbytes
                    else:
                        fb += b.nbytes                       # mmap-backed: file pages, not footprint
            elif isinstance(x, mx.array) and id(x) not in seen:
                seen.add(id(x))
                mxb += x.nbytes
    top = ",".join(f"{k}:{v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1])[:4])
    if fns:
        top += "; " + ",".join(f"{k}:{v}" for k, v in sorted(fns.items(), key=lambda kv: -kv[1])[:4])
    del g
    o = x = b = None                                 # the loop's last objects must not outlive the collect
    return n, gc.collect(), npb / 1e9, mxb / 1e9, fb / 1e9, top


def vmmap(k):
    p = os.path.join(OUTD, f"{TAG}_vmmap_r{k}.txt")
    try:
        r = subprocess.run(["vmmap", "--summary", str(os.getpid())], capture_output=True, text=True, timeout=90, env=TOOLENV)
        open(p, "w").write(r.stdout or r.stderr)
    except Exception as e:
        open(p, "w").write(f"vmmap failed: {e}\n")


def mhist(k):                                        # live allocations by call stack, biggest first
    p, t = os.path.join(OUTD, f"{TAG}_mhist_r{k}.txt"), time.perf_counter()
    try:
        r = subprocess.run(["malloc_history", str(os.getpid()), "-allBySize", "-fullStacks", "-q"], capture_output=True,
                           text=True, timeout=420, env=TOOLENV)
        open(p, "w").write(r.stdout or r.stderr)
    except Exception as e:
        open(p, "w").write(f"malloc_history failed: {e}\n")
    print(f"[{TAG}] MHIST row {k}: {os.path.getsize(p) / 1e6:.1f} MB written in {time.perf_counter() - t:.0f} s", flush=True)


os.makedirs(OUTD, exist_ok=True)
h0 = hot()
CFG = getattr(B, "CFG", {})
print(f"[{TAG}] bt_b12 {sha}, engine {getattr(B, 'V3F', '?')}, PFR grid {os.environ.get('B12_PFR_GRID', 'default')}, C_m {CFG.get('cm')} "
      f"C_l {CFG.get('cl')}; suite rows {ROWS or f'1-{NR}'}: {sum(len(r['completion_ids']) for r in SEL)} tokens in "
      f"{sum(r['seconds'] for r in SEL):.0f} s; {len(TABS)} n-gram tables, {len(SCS)} sidecars, row bytes "
      f"{sorted({sc._hot_row_bytes for sc in SCS})}, cap rows {sorted({sc._hot_cap_rows for sc in SCS})}; after load: footprint "
      f"{fp()[0]:.3f} GB, MLX active {mx.get_active_memory() / 1e9:.3f} GB, hot rows {h0['rows']}", flush=True)
_T = sys.modules.get("bigmoe_v3f.tier")          # M-5d: the R-53 parts the engine actually runs with (v3p only)
print(f"[{TAG}] R-53 parts in the engine: a {getattr(_T, '_R53A', 'n/a')} b {getattr(_T, '_R53B', 'n/a')}, "
      f"F12_R53A env {os.environ.get('F12_R53A', 'unset')}", flush=True)
TRACE = os.environ.get("F12_TRACE") == "1"         # R-45R: tracemalloc (1 frame) snapshots after row 1 and the last row
if TRACE:
    import tracemalloc
    tracemalloc.start(1)


def tsum():                                          # traced bytes by file:line; the snapshot itself is dropped
    snap = tracemalloc.take_snapshot()
    st = snap.statistics("lineno")
    d = {(x.traceback[0].filename, x.traceback[0].lineno): (x.size, x.count) for x in st}
    tot = sum(x.size for x in st)
    del snap, st
    return d, tot


FPOST, D, H, SAME = [], [], [], []
for k, r in enumerate(SEL, 1):
    ids, cap = ITEMS[r["key"]]["prompt_ids"], ITEMS[r["key"]]["max_tokens"]
    cap = min(cap, CAPT) if CAPT else cap
    f0 = fp()[0]
    PEAK[0] = f0
    t0 = time.perf_counter()
    PK[:] = [t0, float("nan"), float("nan")]
    a0 = mx.get_active_memory() / 1e9
    mx.reset_peak_memory()
    if MT:
        EV.clear(); T0[0] = t0
    reset_generation_state(model)                    # = decode._gen_mtp
    B.STASH.clear(); B.PENDH.clear()
    state = B.b12_prefill_state(model, list(ids), 512, 4096)
    tp, pkp, mxp = time.perf_counter(), PEAK[0], mx.get_peak_memory() / 1e9   # M-5b: the prefill phase's peaks
    if MT:
        _mark("state")                               # [pf_out .. b12_prefill_state's return]
    toks, sst = spec125.generate(model, list(ids), cap, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512,
                                 prefilled=state, stop_ids=EOS)
    te, pkt, pka, pkc, mxd = time.perf_counter(), PK[0], PK[1], PK[2], mx.get_peak_memory() / 1e9
    if MT:
        _mark("decode")                              # the whole decode as one interval
    toks = [int(x) for x in toks]
    fin = "length"
    for j, x in enumerate(toks):
        if x in EOS:
            toks, fin = toks[:j + 1], "eos"
            break
    else:
        toks = toks[:cap]
    secs = time.perf_counter() - t0
    del state, sst
    pk, fend = PEAK[0], fp()[0]                      # ~ the suite's next row start
    act, cch = mx.get_active_memory() / 1e9, mx.get_cache_memory() / 1e9
    t1 = time.perf_counter()
    h = hot()
    time.sleep(1.0)                                  # the footprint settles after a row (smoke: -0.11 GB with nothing to collect)
    fpre, actp = fp()[0], mx.get_active_memory() / 1e9
    n, n2, npb, mxb, fb, top = gc_census()
    fpost, life = fp()
    act2 = mx.get_active_memory() / 1e9
    nobj = len(gc.get_objects())
    sw, comp = pressure()
    same = toks == (r["completion_ids"][:len(toks)] if CAPT else r["completion_ids"])
    FPOST.append(fpost); D.append(fpre - fpost); H.append(h["full"]); SAME.append(same)
    print(f"[{TAG}] {r['key']} RESULT row {k} | n {len(ids)} | tok {len(toks)} {fin} | sha {hashlib.sha1(str(toks).encode()).hexdigest()[:10]} "
          f"| suite tokens equal {same} | footprint start {f0:.3f} max {pk:.3f} end {fend:.3f} GB | MLX active {act:.3f} cache {cch:.3f} "
          f"| hot rows {h['rows']} counted {h['acc']:.3f} data {h['real']:.3f} full {h['full']:.3f} GB in {h['batches']} batches, hits "
          f"{h['hits']} misses {h['miss']} | settled {fpre:.3f} GB (S {fend - fpre:+.3f}), MLX active {actp:.3f} | gc {n} objects ({n2} "
          f"more), garbage numpy {npb:.3f} mx {mxb:.3f} file {fb:.3f} GB [{top}] | after gc: footprint {fpost:.3f} GB (D {fpre - fpost:+.3f}), "
          f"MLX active {act2:.3f}, {nobj} tracked objects | lifetime peak {life:.3f} | swap {sw:.2f} GB, compressor {comp:.2f} GB | "
          f"{secs:.2f} s decode = {len(toks) / secs:.2f} tok/s, {time.perf_counter() - t1:.1f} s measure", flush=True)
    print(f"[{TAG}] {r['key']} PEAKAT row {k} | start {f0:.3f} GB, MLX active {a0:.3f} | prefill {tp - t0:.2f} s, max {pkp:.3f} GB, "
          f"MLX peak {mxp:.3f} | row max {pk:.3f} GB in the {'prefill' if pkt <= tp else 'decode'} at {pkt - t0:.2f} s of "
          f"{te - t0:.2f} s (decode {100 * max(0.0, pkt - tp) / max(1e-9, te - tp):.0f} % through); at that max: MLX active "
          f"{pka:.3f} cache {pkc:.3f} GB | row MLX peak {mxd:.3f} GB", flush=True)
    if MT:
        memtrace(k, r["key"], a0, pkt - t0)
        print(f"[{TAG}] {r['key']} SUITE mem row {k}: {r.get('mem')}", flush=True)
    if VMALL or k in (1, NR):
        vmmap(k)
    if MHIST and (k in MHROWS if MHROWS else k in (1, NR)):
        mhist(k)
    if TRACE and k == 1:
        S1, T1 = tsum()
        print(f"[{TAG}] TRACE row 1: traced {T1 / 1e6:.1f} MB, tracemalloc overhead {tracemalloc.get_tracemalloc_memory() / 1e6:.1f} MB", flush=True)
    if TRACE and k == NR and NR > 1:
        S2, T2 = tsum()
        rows = sorted(((S2.get(q, (0, 0))[0] - S1.get(q, (0, 0))[0], S2.get(q, (0, 0))[1] - S1.get(q, (0, 0))[1], q)
                       for q in set(S1) | set(S2)), key=lambda t: -t[0])
        short = lambda f: f.split("site-packages/")[-1].replace(os.path.expanduser("~") + "/", "~/")
        print(f"[{TAG}] TRACE rows 1 -> {NR}: traced {T1 / 1e6:.1f} -> {T2 / 1e6:.1f} MB (T {(T2 - T1) / 1e6:+.1f} MB), tracemalloc "
              f"overhead {tracemalloc.get_tracemalloc_memory() / 1e6:.1f} MB; top growth by line: " + "; ".join(
              f"{dz / 1e6:+.2f} MB {dc:+d} blocks {short(q[0])}:{q[1]}" for dz, dc, q in rows[:12]), flush=True)
        print(f"[{TAG}] TRACE top shrink: " + "; ".join(f"{dz / 1e6:+.2f} MB {short(q[0])}:{q[1]}" for dz, dc, q in rows[-4:]), flush=True)
if len(FPOST) == NR and NR >= 2:
    Bv = sum(D[1:])
    C = FPOST[-1] - FPOST[0] + Bv
    A = H[-1] - H[0]
    sc = SUITE[NR] - SUITE[1] if len(SUITE) > NR else float("nan")
    ok = all(SAME)
    if not ok or C < 0.40:
        v = "NO VERDICT (" + ("tokens differ from b12llfx" if not ok else f"C {C:.3f} < 0.40 GB") + ")"
    elif max(A, Bv) >= 0.8 * C:
        v = "PASS (" + ("a: hot-row cache" if A >= Bv else "b: cyclic garbage") + ")"
    elif A + Bv >= 0.8 * C:
        v = "BETWEEN (a + b together)"
    else:
        v = "KILL (neither)"
    print(f"[{TAG}] SUMMARY rows 1-{NR}: creep C {C:.3f} GB (F_post {FPOST[0]:.3f} -> {FPOST[-1]:.3f}, + B) vs the suite's "
          f"{sc:.3f} over the same row starts; A (hot cache, full) {A:+.3f} GB = {A / C * 100 if C else float('nan'):.0f} %; "
          f"B (gc drops, rows 2-{NR}) {Bv:+.3f} GB = {Bv / C * 100 if C else float('nan'):.0f} %; tokens equal on all rows {ok}; "
          f"verdict by the line: {v}", flush=True)
if hasattr(_T, "R53"):
    print(f"[{TAG}] R-53 counters: " + ", ".join(f"{k} {v}" for k, v in _T.R53.items() if k != "bufs"), flush=True)
if hasattr(_T, "SG12"):
    print(f"[{TAG}] SG12 on {_T._SG12} G {_T._SG12_G}: " + ", ".join(f"{k} {v}" for k, v in _T.SG12.items())
          + f"; PF single {_T.PF.get('single')} grouped {_T.PF.get('grouped')}", flush=True)
if hasattr(_T, "SGB"):                               # SGB (Mon 2 Nov): balanced prefill groups
    print(f"[{TAG}] SGB on {_T._SGB} G {_T._SGB_G}: " + ", ".join(f"{k} {v}" for k, v in _T.SGB.items())
          + f"; PF single {_T.PF.get('single')} grouped {_T.PF.get('grouped')}", flush=True)
print(f"[{TAG}] done", flush=True)
