"""Breakthrough R-45 v3o creep check (NewIdeas Thu 19:52 GO; QUEUE R-45), a copy of Fit12's scripts/fit_creep.py (R-45 /
R-45S / R-45T harness, read-only source) with: the Eval suite's agent rows by key (F12_SET=agent, F12_KEYS=k1,k2,...,
repeats allowed; tokens checked against b12llfx's rows), an idle gap of F12_GAP s after every row before the gc census (so
GCD retires its idle workers, which leaked v3n's bounce buffer: Fit12 R-45T), the v3o candidate's sha allowed, and a
per-request summary: G = (after-gc footprint after the last row - after the first row) / (rows - 1), plus vmmap's
MALLOC_LARGE (dirty + swapped, regions) after the first and last rows.
Lines (NewIdeas): PASS (with hashes = v3n on p2k / p8k / write16, cand_check) G <= 3 MB per request; KILL G >= 15 MB.
Arms: v3o = F12_BT=<v3o candidate fd405861fa> F12_ENGINE=v3o; v3n = F12_BT=scripts/bt_b12.py (23b42972ef) F12_ENGINE=v3n
(both M-01, records3m). Usage: [BigMoE venv] F12_SET=agent F12_KEYS=... F12_GAP=10 F12_OUT=data/v3o v3o_creep.py TAG"""
import ctypes, gc, hashlib, json, os, subprocess, sys, threading, time

POP = ("EVALSUITE_FRUGAL", "EVALSUITE_SHIP", "EVALSUITE_FAST24", "B12_LATE3", "B12_W", "B12_NORM", "BIGMOE_LOOP_GUARD", "B12_CM",
       "B12_CL", "B12_ENGINE", "B12_TRIMS", "B12_PFR", "B12_ALLWAIT", "B12_CACHE_MB", "B12_FOLD", "B12_FOLD_CM", "B12_FOLD_ROWS",
       "B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_PFR", "B12_FOLD_MOV", "BIGMOE_PF_GROUP", "BIGMOE_ASYNC_EVERY",
       "BIGMOE_PF_PREFILL", "BIGMOE_PF_FULLSCR", "MTPLX_NGRAM_HOT_MB", "B12_PFR_MIN", "B12_MODE")
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
for k in POP:
    os.environ.pop(k, None)
if ARM == "def":                                 # the candidate's own defaults (decision 5: fold-fast): no engine / fold keys
    for k in ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID"):
        ENV.pop(k)
else:
    ENV["B12_ENGINE"] = os.environ.get("F12_ENGINE", ENV["B12_ENGINE"])   # R-46 (30 Oct): v3n with bt_b12 23b42972ef
os.environ.update(ENV)
if os.environ.get("F12_HOT_MB"):                 # R-46 HOTCAP: the n-gram hot-row cap (MTPLX_NGRAM_HOT_MB, counted at 100 B / row)
    os.environ["MTPLX_NGRAM_HOT_MB"] = os.environ["F12_HOT_MB"]
TAG = sys.argv[1] if len(sys.argv) > 1 else "creep"
NR = int(os.environ.get("F12_NROWS", "8"))
GAP = float(os.environ.get("F12_GAP", "10"))
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


def _sampler():
    while True:
        PEAK[0] = max(PEAK[0], fp()[0])
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
assert sha in ("23b42972ef", "fd405861fa", "9c3057ce62", "b9d5b491ad", "677c6558b6", "cf8bd7840a", "9250688a7a"), f"bt_b12.py is {sha}: v3n 23b42972ef, v3o fd405861fa, PFR_MIN 9c3057ce62, refill b9d5b491ad"
REF = [json.loads(l) for l in open(os.path.join(ES, "results", os.environ.get("F12_REF") or ("b12llff" if ARM == "ff" else "b12llfx"), "decode.jsonl"))]
SET = os.environ.get("F12_SET", "math")
REF = [r for r in REF if r["set"] == SET]
KEYS = [x for x in os.environ.get("F12_KEYS", "").split(",") if x]
ROWS = [int(x) for x in os.environ.get("F12_ROWS", "").split(",") if x]   # R-45S: suite math rows by number (1-based), repeats allowed
SEL = [{r["key"]: r for r in REF}[k] for k in KEYS] if KEYS else [REF[i - 1] for i in ROWS] if ROWS else REF[:NR]
NR = len(SEL)
VMALL = os.environ.get("F12_VMALL") == "1"         # R-45S: vmmap after every row
MHIST = os.environ.get("F12_MHIST") == "1"         # R-45T: malloc_history -allBySize after the first and last rows
TOOLENV = {k: v for k, v in os.environ.items() if not k.startswith("MallocStackLogging")}   # the tools don't log themselves
ITEMS = {it["key"]: it for it in map(json.loads, open(os.path.join(ES, "data", "items", f"dec_{SET}.jsonl")))}
for r in SEL:
    assert len(ITEMS[r["key"]]["prompt_ids"]) == r["n_prompt"], r["key"]
SUITE = [] if ROWS or KEYS else [r["mem"]["start"] for r in REF[:NR + 1]]   # R-41C's row starts: rows 2..NR+1 start after rows 1..NR

threading.Thread(target=_sampler, daemon=True).start()
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, os.path.dirname(BT))
import bt_b12 as B
assert os.path.realpath(B.__file__) == os.path.realpath(BT), B.__file__
print(f"[{TAG}] bt_b12 {sha} from {BT}, MallocStackLogging {os.environ.get('MallocStackLogging', 'off')}", flush=True)
rt, st = B.load_b12(0.10, mtp=True)
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model = rt.model
from bigmoe_v3f.gated import ext as _EXT


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
print(f"[{TAG}] bt_b12 {sha}, engine {getattr(B, 'V3F', '?')}, mode {getattr(B, 'MODE', '-')}, PFR grid {os.environ.get('B12_PFR_GRID', getattr(B, '_PFR_GRID0', 'keep'))}, PFR_MIN {os.environ.get('B12_PFR_MIN', getattr(B, '_PFR_MIN0', '4096'))}, C_m {CFG.get('cm')} "
      f"C_l {CFG.get('cl')}; suite rows {ROWS or f'1-{NR}'}: {sum(len(r['completion_ids']) for r in SEL)} tokens in "
      f"{sum(r['seconds'] for r in SEL):.0f} s; {len(TABS)} n-gram tables, {len(SCS)} sidecars, row bytes "
      f"{sorted({sc._hot_row_bytes for sc in SCS})}, cap rows {sorted({sc._hot_cap_rows for sc in SCS})}; after load: footprint "
      f"{fp()[0]:.3f} GB, MLX active {mx.get_active_memory() / 1e9:.3f} GB, hot rows {h0['rows']}", flush=True)
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
    reset_generation_state(model)                    # = decode._gen_mtp
    B.STASH.clear(); B.PENDH.clear()
    npf, nrf, tp = len(B.PFR_LOG), len(getattr(B, "REFILL_LOG", [])), time.perf_counter()
    state = B.b12_prefill_state(model, list(ids), 512, 4096)
    mx.eval(state["logits"])
    tp, pfr = time.perf_counter() - tp, len(B.PFR_LOG) > npf
    rfl = getattr(B, "REFILL_LOG", [])[nrf:]
    pk_pf = PEAK[0]
    toks, sst = spec125.generate(model, list(ids), cap, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512,
                                 prefilled=state, stop_ids=EOS)
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
    time.sleep(GAP)                                  # idle gap: GCD retires idle workers (v3n leaked their bounce buffers)
    fpre, actp = fp()[0], mx.get_active_memory() / 1e9
    n, n2, npb, mxb, fb, top = gc_census()
    fpost, life = fp()
    act2 = mx.get_active_memory() / 1e9
    nobj = len(gc.get_objects())
    sw, comp = pressure()
    same = toks == (r["completion_ids"][:len(toks)] if CAPT else r["completion_ids"])
    FPOST.append(fpost); D.append(fpre - fpost); H.append(h["full"]); SAME.append(same)
    print(f"[{TAG}] {r['key']} RESULT row {k} | n {len(ids)} | tok {len(toks)} {fin} | sha {hashlib.sha1(str(toks).encode()).hexdigest()[:10]} "
          f"| suite tokens equal {same} | prefill {tp:.2f} s, PFR {'yes' if pfr else 'no'}, refill {list(rfl[0]) if rfl else 'no'}, decode {secs - tp:.2f} s = {len(toks) / (secs - tp):.2f} tok/s, prefill max {pk_pf:.3f} (rise {pk_pf - f0:+.3f}) | footprint start {f0:.3f} max {pk:.3f} (rise {pk - f0:+.3f}) end {fend:.3f} GB | MLX active {act:.3f} cache {cch:.3f} "
          f"| hot rows {h['rows']} counted {h['acc']:.3f} data {h['real']:.3f} full {h['full']:.3f} GB in {h['batches']} batches, hits "
          f"{h['hits']} misses {h['miss']} | settled {fpre:.3f} GB (S {fend - fpre:+.3f}), MLX active {actp:.3f} | gc {n} objects ({n2} "
          f"more), garbage numpy {npb:.3f} mx {mxb:.3f} file {fb:.3f} GB [{top}] | after gc: footprint {fpost:.3f} GB (D {fpre - fpost:+.3f}), "
          f"MLX active {act2:.3f}, {nobj} tracked objects | lifetime peak {life:.3f} | swap {sw:.2f} GB, compressor {comp:.2f} GB | "
          f"{secs:.2f} s decode = {len(toks) / secs:.2f} tok/s, {time.perf_counter() - t1:.1f} s measure"
          + (f" | bounce pool {_EXT.bounce_stats()}" if hasattr(_EXT, 'bounce_stats') else ''), flush=True)
    if VMALL or k in (1, NR):
        vmmap(k)
    if MHIST and k in (1, NR):
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
def mlarge(k):                                       # vmmap MALLOC_LARGE: (dirty + swapped MiB, regions)
    try:
        for ln in open(os.path.join(OUTD, f"{TAG}_vmmap_r{k}.txt")):
            if ln.startswith("MALLOC_LARGE "):
                f = ln.split()
                mb = lambda x: float(x[:-1]) * {"K": 1 / 1024, "M": 1, "G": 1024}[x[-1]]
                return mb(f[3]) + mb(f[4]), int(f[8])
    except Exception:
        pass
    return float("nan"), -1


if len(FPOST) == NR and NR >= 2:
    G = (FPOST[-1] - FPOST[0]) / (NR - 1) * 1e3
    (m1, c1), (m2, c2) = mlarge(1), mlarge(NR)
    ok = all(SAME)
    v = "PASS" if G <= 3 else "KILL" if G >= 15 else "BETWEEN"
    print(f"[{TAG}] SUMMARY {NR} requests ({','.join(r['key'].split('/')[-1] for r in SEL)}), {GAP:.0f} s idle gaps: after-gc "
          f"footprint {FPOST[0]:.3f} -> {FPOST[-1]:.3f} GB = G {G:+.1f} MB per request (per row: "
          + " ".join(f"{(b - a) * 1e3:+.0f}" for a, b in zip(FPOST, FPOST[1:])) + f" MB); MALLOC_LARGE {m1:.1f} MiB / {c1} regions "
          f"-> {m2:.1f} MiB / {c2} regions ({(m2 - m1) / (NR - 1):+.1f} MiB, {(c2 - c1) / (NR - 1):+.1f} regions per request); "
          f"tokens equal to b12llfx on all rows {ok}; by the creep line (PASS <= 3, KILL >= 15 MB): {v}", flush=True)
print(f"[{TAG}] done", flush=True)
