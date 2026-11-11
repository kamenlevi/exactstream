"""Where do MTPLX's 4.3 ms per decode-step PLE n-gram gathers go, and does a direct-pread gather fix it?
CPU only, read-only on ngram-table.safetensors (32 GB). Random row ids are cold: the table is far bigger than the page cache.
Compares, per call of 16 fresh rows (misses in the hot LRU):
  stock : _SidecarGather._rows_matrices (pool warm pass, then numpy fancy-index on the memmaps)
  warm  : the warm pass alone; index: the fancy index alone right after it
  direct: one pool task per row that preads each map's row bytes and returns them (no memmap touch)
and checks that `direct` returns identical bytes. Usage: bench_gather.py [rows_per_call] [calls]"""
import os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from direct_gather import direct_rows  # noqa: E402
from mtplx.models import qwen4_exp as Q  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 16
CALLS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
path = Path(os.path.expanduser("~/BigMoE/models/bare/ngram-table.safetensors"))
t = Q.NGramTable(rows=1, dim=1, sidecar=True)
t.attach_sidecar(path)
sg = t._sidecar
names = ("weight",) if sg.bits == 0 else ("weight", "scales", "biases")
nrows = sg._maps["weight"][0].shape[0]
print(f"table rows {nrows:,}, maps {names}, row bytes {[rb for _, rb in sg._row_meta]}, layout {sg.row_layout}")
rng = np.random.default_rng(1)

# Optional background load like BigMoE's expert loader: LOAD threads doing F_NOCACHE 2.77 MB preads at random offsets
import fcntl, glob, threading
LOAD = int(os.environ.get("LOAD", "0"))
_stop = threading.Event()
_bytes = [0]
def _loader(seed):
    files = sorted(glob.glob(os.path.expanduser("~/BigMoE/models/bare/model-000*.safetensors")))
    fds = []
    for f in files:
        fd = os.open(f, os.O_RDONLY); fcntl.fcntl(fd, fcntl.F_NOCACHE, 1); fds.append((fd, os.path.getsize(f)))
    r = np.random.default_rng(seed); n = 2_765_000
    while not _stop.is_set():
        fd, size = fds[r.integers(len(fds))]
        off = int(r.integers(0, size - n)) & ~16383
        _bytes[0] += len(os.pread(fd, n, off))
for i in range(LOAD):
    threading.Thread(target=_loader, args=(i,), daemon=True).start()
if LOAD:
    time.sleep(1.0)
    b0 = _bytes[0]; time.sleep(1.0); print(f"background load: {LOAD} threads, {(_bytes[0]-b0)/1e9:.1f} GB/s of 2.77 MB F_NOCACHE reads")


def fresh():
    return rng.integers(0, nrows, R, dtype=np.int64)


def med(ts):
    return sorted(ts)[len(ts) // 2] * 1e3


res = {k: [] for k in ("stock", "warm", "index", "direct")}
same = True
for _ in range(CALLS):
    ids = fresh()
    a = time.perf_counter(); out = sg._rows_matrices(ids, names); res["stock"].append(time.perf_counter() - a)
    ids2 = fresh()
    a = time.perf_counter(); sg._warm(np.unique(ids2)); b = time.perf_counter()
    [np.ascontiguousarray(sg._maps[n][0][np.unique(ids2)]) for n in names]; c = time.perf_counter()
    res["warm"].append(b - a); res["index"].append(c - b)
    ids3 = fresh()
    a = time.perf_counter(); d = direct_rows(sg, ids3, names); res["direct"].append(time.perf_counter() - a)
    ref = {n: np.ascontiguousarray(sg._maps[n][0][ids3]) for n in names}
    same &= all(np.array_equal(d[n], ref[n]) for n in names)
for k, v in res.items():
    print(f"{k:7s} median {med(v):6.2f} ms  p90 {sorted(v)[int(len(v)*0.9)]*1e3:6.2f} ms  ({R} rows/call)")
print("direct bytes identical to the memmap rows:", same)
_stop.set()
