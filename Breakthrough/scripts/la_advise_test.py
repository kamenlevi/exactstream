"""D1k mechanism check (CPU + SSD only, ~15 s): can the OS read-ahead (fcntl F_RDADVISE on a cached fd) stage records3
experts that v3's tier prefill then reads through its F_NOCACHE fd? If an advised set reads ~as fast as RAM, layer L+1's
predicted union can be advised while layer L computes, with the prefill path (and its outputs) unchanged.
Reads 16 threads x one pread per expert (as read_rebuilt / the v3 stream do), 2.93 MB records.
Usage: la_advise_test.py [LAYER] [N_EXPERTS] [WAIT_MS ...]"""
import fcntl, os, random, struct, sys, time
from concurrent.futures import ThreadPoolExecutor

F_RDADVISE = 44                                     # <sys/fcntl.h>; struct radvisory {off_t ra_offset; int ra_count;}
D = os.path.expanduser("~/BigMoE/models/bare/records3")
L = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WAITS = [int(v) for v in sys.argv[3:]] or [0, 30, 60, 250]
RS, E = 2932736, 512
path = os.path.join(D, f"L{L:02d}.bin")
fd_nc = os.open(path, os.O_RDONLY)
fcntl.fcntl(fd_nc, fcntl.F_NOCACHE, 1)               # as the ext's read_rebuilt and the v3 stream open records3
fd_ad = os.open(path, os.O_RDONLY)                    # cached fd for the advisories
pool = ThreadPoolExecutor(max_workers=16)
bufs = [bytearray(RS) for _ in range(16)]


def read(ids):
    t0 = time.perf_counter()
    futs = [pool.submit(os.preadv, fd_nc, [memoryview(bufs[k % 16])], e * RS) for k, e in enumerate(ids)]
    for f in futs:
        assert f.result() == RS
    dt = time.perf_counter() - t0
    return dt, len(ids) * RS / dt / 1e9


def advise(ids):
    t0 = time.perf_counter()
    for e in ids:
        fcntl.fcntl(fd_ad, F_RDADVISE, struct.pack("qi4x", e * RS, RS))
    return time.perf_counter() - t0


rng = random.Random(L)
order = list(range(E)); rng.shuffle(order)
sets = [order[i * N:(i + 1) * N] for i in range(E // N)]
need = len(WAITS) + 1
if len(sets) < need:                                  # fewer disjoint sets than rows: smaller sets
    n = E // need; sets = [order[i * n:(i + 1) * n] for i in range(need)]
dt, gbs = read(sets[0])
print(f"L{L:02d} cold, {len(sets[0])} experts: {dt * 1e3:.0f} ms ({gbs:.1f} GB/s)", flush=True)
dt, gbs = read(sets[0])
print(f"L{L:02d} same set again through F_NOCACHE: {dt * 1e3:.0f} ms ({gbs:.1f} GB/s)", flush=True)
for w, s in zip(WAITS, sets[1:]):
    ta = advise(s)
    time.sleep(w / 1e3)
    dt, gbs = read(s)
    print(f"L{L:02d} advised ({ta * 1e3:.1f} ms to advise {len(s)}), read after {w} ms: {dt * 1e3:.0f} ms "
          f"({gbs:.1f} GB/s)", flush=True)
