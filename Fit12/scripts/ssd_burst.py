"""Fit12 SS1 (2026-10-24): does the SSD serve one layer's demand burst faster with the same bytes in more, smaller aligned
reads? A burst = 12 random experts' MSB parts (records3, 2,162,688 B = 132 pages each; B12LL's ~12 base misses per layer),
read uncached (F_NOCACHE) into page-aligned buffers by a persistent thread pool, as 1 / 2 / 4 / 6 reads per part
(132 / 66 / 33 / 22 pages each), waiting for the whole burst like the loader's release. Read-only on ~/BigMoE's model
files; system python3 + ctypes pread (the GIL is released, no Python-side copy). Prints per split: median / p90 burst ms
and GB/s, interleaved rounds so drift hits every split alike. Usage: python3 ssd_burst.py [bursts_per_split] [parts]"""
import ctypes, fcntl, json, mmap, os, random, statistics as st, sys, time
from concurrent.futures import ThreadPoolExecutor, wait

D = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(os.path.join(D, "index.json")))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 150
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 12
libc = ctypes.CDLL(None, use_errno=True)
libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
libc.pread.restype = ctypes.c_ssize_t
F_NOCACHE = 48
lay = {}
for L in range(48):
    r = idx["layers"][str(L)]
    fd = os.open(os.path.join(D, r["file"]), os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    lay[L] = (fd, r["recsz"], r["msb_sz"], r["E"])
MSB = lay[0][2]
bufs = [mmap.mmap(-1, MSB) for _ in range(NP)]
addr = [ctypes.addressof(ctypes.c_char.from_buffer(b)) for b in bufs]
pool = ThreadPoolExecutor(max_workers=64)


def rd(fd, dst, n, off):
    done = 0
    while done < n:
        r = libc.pread(fd, dst + done, n - done, off + done)
        if r <= 0:
            raise OSError(ctypes.get_errno(), "pread")
        done += r


def burst(split, rng):
    L = rng.randrange(48)
    fd, recsz, msb, E = lay[L]
    es = rng.sample(range(E), NP)
    chunk = msb // split
    assert chunk % 16384 == 0 or split == 1
    t0 = time.perf_counter()
    fs = [pool.submit(rd, fd, addr[i] + c * chunk, chunk, e * recsz + c * chunk) for i, e in enumerate(es) for c in range(split)]
    wait(fs)
    for f in fs:
        f.result()
    return time.perf_counter() - t0


rng = random.Random(1)
splits = [s for s in (1, 2, 4, 6) if MSB % (s * 16384) == 0]
for s in splits:                       # warm the pool threads
    burst(s, rng)
res = {s: [] for s in splits}
for _ in range(NB):
    for s in splits:
        res[s].append(burst(s, rng))
gb = NP * MSB / 1e9
for s in splits:
    v = sorted(res[s])
    m, p90 = st.median(v), v[int(0.9 * len(v))]
    print(f"split {s} ({MSB // s // 16384} pages per read, {NP * s} reads per burst): median {1e3 * m:.2f} ms "
          f"({gb / m:.1f} GB/s), p90 {1e3 * p90:.2f} ms, mean {1e3 * st.mean(v):.2f} ms", flush=True)
