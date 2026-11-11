"""How much of a set of files sits in the OS page cache (mincore over a read-only mmap). usage: mincore.py GLOB
(Breakthrough, 2026-10-20: checks whether F_NOCACHE expert reads could have been served from cached pages.)"""
import ctypes, ctypes.util, glob, mmap, os, sys
import numpy as np
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
PG = os.sysconf("SC_PAGE_SIZE")
tot = res = 0
for f in sorted(glob.glob(os.path.expanduser(sys.argv[1]))):
    sz = os.path.getsize(f)
    if sz == 0:
        continue
    fd = os.open(f, os.O_RDONLY)
    m = mmap.mmap(fd, sz, prot=mmap.PROT_READ)
    a = np.frombuffer(m, dtype=np.uint8)
    n = (sz + PG - 1) // PG
    vec = ctypes.create_string_buffer(n)
    if libc.mincore(ctypes.c_void_p(a.ctypes.data), ctypes.c_size_t(sz), vec) == 0:
        r = int((np.frombuffer(vec.raw, dtype=np.uint8) & 1).sum())
        tot += n; res += r
    del a
    m.close(); os.close(fd)
print(f"{sys.argv[1]}: {res * PG / 1e9:.1f} GB of {tot * PG / 1e9:.1f} GB in the page cache ({100 * res / max(1, tot):.1f} %)")
