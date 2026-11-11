"""Drop files' pages from the page cache without root: mmap + msync(MS_INVALIDATE).
Verifies with a 1 GB F_NOCACHE sequential read before/after (cached reads >> SSD speed).
usage: evict.py FILE|DIR..."""
import ctypes, fcntl, mmap, os, sys, time
import numpy as np
libc = ctypes.CDLL(None)
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
MS_INVALIDATE = 0x0002


def rate(path):
    fd = os.open(path, os.O_RDONLY); fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    buf = bytearray(8 << 20); t = time.perf_counter(); n = 0
    for off in range(0, min(os.path.getsize(path), 1 << 30), 8 << 20):
        n += os.preadv(fd, [buf], off)
    os.close(fd)
    return n / (time.perf_counter() - t) / 1e9


import glob
paths = []
for a in sys.argv[1:]:
    paths += sorted(glob.glob(os.path.join(a, "*.safetensors"))) if os.path.isdir(a) else [a]
for p in paths:
    before = rate(p)
    fd = os.open(p, os.O_RDONLY); sz = os.path.getsize(p)
    mm = mmap.mmap(fd, sz, prot=mmap.PROT_READ)
    a = np.frombuffer(mm, dtype=np.uint8)
    rc = libc.msync(a.ctypes.data, sz, MS_INVALIDATE)
    del a; mm.close(); os.close(fd)
    after = rate(p)
    print(f"{os.path.basename(p)}: first-GB read {before:.1f} GB/s -> after MS_INVALIDATE {after:.1f} GB/s (rc {rc})", flush=True)
