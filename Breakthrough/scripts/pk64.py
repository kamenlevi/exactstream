"""Queue L2-10 (Breakthrough, 2026-10-25): where the 64K prefill's footprint peak sits, and what cuts it.
One variant per process (footprint peaks are per process): B12LL, the decoupled prefill of the first PK_N tokens of
data/prompts/ctx64k.txt, then 8 greedy one-row decode steps (token sha + a hash of the first step's logits: exactness
across variants). Two samplers: (1) 20 Hz physical footprint, MLX active and bt_b12.PHASE; (2) every 3 s `footprint -p`
categories: 'IOAccelerator (graphics)' (mapped GPU buffers) and 'Owned physical footprint (unmapped) (graphics)' (GPU
memory already freed but not yet reclaimed; it still counts in the footprint: 10-25 10:35, kv8_long at 32K, it swung
0.07 -> 2.2 -> 0.07 GB within seconds of large frees).
Suspect (10-25): every s_attn chunk grows each QSA layer's stock KV by concatenation (a new history + 512 rows buffer, the
old one freed: at 64K ~1.5 GB freed per chunk over 12 layers), so freed-not-reclaimed memory piles up late in the prompt.
env: PK_N (65000), PK_ATTN (512), PK_MOE (4096), PK_TARGET (0 = stock growth; 1 = mtplx qsa_rows_target(N + PK_HEAD) around
     the prefill: every QSA buffer sized once, attention reads the same rows), PK_HEAD (1024), PK_MTP (1 = the MTP head's
     history appended per MoE chunk, as b12_prefill_state), PK_KV8 (1 = b12_kv8 8-bit KV), PK_LANE (1 = the prefill inside
     mtplx attention_phase("prefill"), as MTPLX's own generation runs it: chunks with >= 32K history then take the sparse
     selector + NAX flash consumer instead of the dense [S, T] mask (b12_prefill never set it: every chunk was dense;
     same visible set, not bit-exact)), B12_CACHE_MB, PK_TAG
     -> a line in data/pk64/results.txt, the series in data/pk64/<tag>.json"""
import hashlib, json, os, subprocess, sys, threading, time

os.environ.setdefault("B12_LATE3", "2,10")
os.environ.setdefault("B12_NORM", "mass")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~/Breakthrough")
OUT = os.path.join(HOME, "data/pk64")
N, SA, SM = int(os.environ.get("PK_N", "65000")), int(os.environ.get("PK_ATTN", "512")), int(os.environ.get("PK_MOE", "4096"))
TARGET, HEAD = os.environ.get("PK_TARGET", "0") == "1", int(os.environ.get("PK_HEAD", "1024"))
MTP, KV8 = os.environ.get("PK_MTP", "0") == "1", os.environ.get("PK_KV8", "0") == "1"
LANE = os.environ.get("PK_LANE", "0") == "1"
TAG = os.environ.get("PK_TAG") or (f"n{N}_a{SA}_m{SM}" + ("_tgt" if TARGET else "") + ("_lane" if LANE else "") +
                                   ("_mtp" if MTP else "") +
                                   ("_kv8" if KV8 else "") + f"_c{os.environ.get('B12_CACHE_MB', '-')}")
import bt_b12 as B
rt, st = B.load_b12(0.10, mtp=MTP)
import numpy as np
import mlx.core as mx
import mtplx.models.qwen4_exp as Q
from bigmoe_v3f.fuse import reset_generation_state
from bigmoe_v3f import tier as _T                             # PF counters + the L2-10r scatter flag, for the result line
if KV8:
    import b12_kv8 as K
    K.install(); K.ON[0] = True
model, tok = rt.model, rt.tokenizer
ids = tok.encode(open(os.path.join(HOME, "data/prompts/ctx64k.txt")).read())[:N + 1]
assert len(ids) == N + 1, len(ids)
os.makedirs(OUT, exist_ok=True)

import ctypes


class _RU(ctypes.Structure):                                 # rusage_info_v4 (as kv8_tf): u7 = physical footprint now
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


libc = ctypes.CDLL(None)


def fp_now():
    ru = _RU()
    return ru.u7 / 1e9 if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else 0.0


SER, CAT, STOP, T0 = [], [], threading.Event(), time.perf_counter()
CATS = {"IOAccelerator (graphics)": "ioacc", "Owned physical footprint (unmapped) (graphics)": "unmapped",
        "MALLOC_LARGE": "malloc_large", "untagged (VM_ALLOCATE)": "untagged"}


def sampler():
    while not STOP.is_set():
        try:
            p = B.PHASE.get("p")
            SER.append((round(time.perf_counter() - T0, 2), round(fp_now(), 3), round(mx.get_active_memory() / 1e9, 3),
                        list(p) if isinstance(p, tuple) else p))
        except Exception:
            pass
        STOP.wait(0.05)


def categories():                                            # footprint -p: which kind of memory makes the gap
    while not STOP.is_set():
        try:
            t, p = round(time.perf_counter() - T0, 2), B.PHASE.get("p")
            o = subprocess.run(["footprint", "-p", str(os.getpid()), "-f", "bytes"], capture_output=True, text=True,
                               timeout=20).stdout
            row = {"t": t, "p": list(p) if isinstance(p, tuple) else p}
            for ln in o.splitlines():
                for k, v in CATS.items():
                    if ln.rstrip().endswith(k):
                        row[v] = round(int(ln.split()[0]) / 1e9, 3)
                if "phys_footprint:" in ln:
                    row["fp"] = round(int(ln.split()[1]) / 1e9, 3)
            CAT.append(row)
        except Exception as e:
            CAT.append({"err": repr(e)[:80]})
        STOP.wait(3.0)


reset_generation_state(model)
B.STASH.clear(); B.PENDH.clear()
mx.clear_cache()
B.PHASE.clear()
th = [threading.Thread(target=f, daemon=True) for f in (sampler, categories)]
for t_ in th:
    t_.start()
time.sleep(3.5)                                              # a baseline sample of each kind before the prefill
base = (fp_now(), mx.get_active_memory() / 1e9)
mx.reset_peak_memory()
t0 = time.perf_counter()
R = ((N + HEAD + 255) // 256) * 256
cache = model.make_cache()
head = None
pids = mx.array([ids[:-1]])
from contextlib import nullcontext
from mtplx.attention_context import attention_phase
with (Q.qsa_rows_target(R) if TARGET else nullcontext()), (attention_phase("prefill") if LANE else nullcontext()):
    if MTP:                                                  # b12_prefill_state's path (D1I_HEAD_EACH=1), the head's
        if os.path.expanduser("~/SpecDecode") not in sys.path:   # appends tagged in PHASE
            sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
        import spec125
        head = spec125._MTPHead(model)

        def feed(pairs):
            for (a, b), h in pairs:
                B.PHASE["p"] = (a, "mtp")
                head.append(h, [int(v) for v in ids[a + 1:b + 1]])
                mx.eval(head.logits)
        B.b12_prefill(model, pids, cache, SA, SM, on_chunk=feed)
    else:
        B.b12_prefill(model, pids, cache, SA, SM)
    mx.eval([a for c in cache for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)])
tp = time.perf_counter() - t0
mlx_peak = mx.get_peak_memory() / 1e9
after_pf = (fp_now(), mx.get_active_memory() / 1e9)
B.PHASE["p"] = "decode"
y, out, lhash = ids[-1], [], None
for i in range(8):
    lg = model(mx.array([[y]]), cache=cache)
    lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1]
    if i == 0:
        lhash = hashlib.sha1(np.array(lg.astype(mx.float32)).tobytes()).hexdigest()[:10]
    y = int(mx.argmax(lg).item())
    out.append(y)
st.gate.check()
after_dec = (fp_now(), mx.get_active_memory() / 1e9)
DEC = {}
if os.environ.get("PK_DECAB", "0") == "1":                  # paired decode A/B on the same 64K cache: MTPLX_QSA_FLASH
    B.PHASE["p"] = "decab"                                   # (opt-in flash-skip decode lane; off = the dense [1, T]
    for r in range(3):                                       # mask over the whole KV in all 12 QSA layers), interleaved
        for fl in ("0", "1"):                                # rounds of 16 greedy steps, median ms per step per round
            os.environ["MTPLX_QSA_FLASH"] = fl
            ts = []
            for i in range(16):
                t1 = time.perf_counter()
                lg = model(mx.array([[y]]), cache=cache)
                lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1]
                y = int(mx.argmax(lg).item())
                ts.append(time.perf_counter() - t1)
            DEC.setdefault(fl, []).append(round(1e3 * float(np.median(ts[2:])), 1))
    os.environ.pop("MTPLX_QSA_FLASH", None)
    st.gate.check()
time.sleep(3.5)
STOP.set()
for t_ in th:
    t_.join(timeout=25)
pf = [r for r in SER if r[3] not in (None, "decode", "decab")]
top = max(pf, key=lambda r: r[1])
gap = max(pf, key=lambda r: r[1] - r[2])
act = max(pf, key=lambda r: r[2])
kinds = {}
for r in pf:
    k = r[3][-1] if isinstance(r[3], list) else str(r[3])
    if r[1] > kinds.get(k, (0,))[0]:
        kinds[k] = (r[1], r[2], r[3])
cpf = [c for c in CAT if "unmapped" in c and c["p"] not in (None, "decode", "decab")]
um = max(cpf, key=lambda c: c["unmapped"]) if cpf else {}
kvb = sum(c.kv.nbytes for c in cache if hasattr(c, "kv")) / 1e9
sha = hashlib.sha1(json.dumps(out).encode()).hexdigest()[:10]
f2 = lambda r: f"{r[1]:.2f} GB (act {r[2]:.2f}) at {tuple(r[3]) if isinstance(r[3], list) else r[3]}"
line = (f"pk64 {TAG}: N {N} attn/moe {SA}/{SM} target {R if TARGET else 0} lane {int(LANE)} mtp {int(MTP)} kv8 {int(KV8)} cache_mb "
        f"{os.environ.get('B12_CACHE_MB', '-')} | prefill {tp:.0f} s | before {base[0]:.2f}/{base[1]:.2f} | PEAK fp {f2(top)} | "
        f"widest fp-act {gap[1] - gap[2]:.2f} at {tuple(gap[3]) if isinstance(gap[3], list) else gap[3]} | max act {act[2]:.2f} | "
        f"MLX peak {mlx_peak:.2f} | after prefill fp/act {after_pf[0]:.2f}/{after_pf[1]:.2f}, after 8 decode "
        f"{after_dec[0]:.2f}/{after_dec[1]:.2f} | unmapped max {um.get('unmapped', float('nan')):.2f} GB at {um.get('p')} "
        f"(ioacc {um.get('ioacc', float('nan')):.2f}) | QSA KV {kvb:.3f} GB | per part " +
        " ".join(f"{k} {v[0]:.2f}" for k, v in sorted(kinds.items())) + f" | tiers {dict(getattr(Q, '_QSA_PREFILL_COUNTS', {}))}"
        f" | sha {sha} logits {lhash} | {tok.decode(out)!r}" +
        (f" | decode ms/step dense {DEC['0']} flash {DEC['1']}" if DEC else "") +
        f" | env TIER_PF_GROUP {os.environ.get('BIGMOE_TIER_PF_GROUP', '-')} PF_GROUP {os.environ.get('BIGMOE_PF_GROUP', '-')}"
        f" ASYNC_EVERY {os.environ.get('BIGMOE_ASYNC_EVERY', '-')}"
        f" | tier scatter {getattr(_T, '_SCATTER', '-')} grouped/single/fallback {_T.PF['grouped']}/{_T.PF['single']}/{_T.PF['fallback']}")
print(line, flush=True)
with open(os.path.join(OUT, "results.txt"), "a") as f:
    f.write(time.strftime("%m-%d %H:%M ") + line + "\n")
json.dump({"tag": TAG, "line": line, "series": SER, "categories": CAT}, open(os.path.join(OUT, TAG + ".json"), "w"))
