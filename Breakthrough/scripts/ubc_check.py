"""R-48 pre-check (CPU only, reads nothing): how much of each records file is resident in the unified buffer cache, by
mincore on a read-only MAP_SHARED mapping (no page is touched). An A/B arm whose files are cached reads fewer SSD bytes
(BigMoE's note Thu 16:39: test reads without F_NOCACHE had left records3 pages cached).
--evict: msync(MS_INVALIDATE) on each file above the line (BigMoE's method: drops its clean cached pages; the file is
unchanged), then measures again.
Usage: [BigMoE venv] ubc_check.py [--evict] [--line PCT (default 1)] DIR...   (exit 1 if a dir stays above the line)"""
import ctypes, os, sys

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
PROT_READ, MAP_SHARED, MS_INVALIDATE, MAP_FAILED = 0x1, 0x1, 0x2, ctypes.c_void_p(-1).value
PG = os.sysconf("SC_PAGE_SIZE")
args = sys.argv[1:]
EVICT = "--evict" in args
LINE = float(args[args.index("--line") + 1]) if "--line" in args else 1.0
DIRS = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--line")]


def resident(path, evict=False):
    n = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        a = libc.mmap(None, n, PROT_READ, MAP_SHARED, fd, 0)
        assert a not in (None, MAP_FAILED), (path, ctypes.get_errno())
        try:
            if evict:
                assert libc.msync(a, n, MS_INVALIDATE) == 0, (path, ctypes.get_errno())
            vec = ctypes.create_string_buffer((n + PG - 1) // PG)
            assert libc.mincore(a, n, vec) == 0, (path, ctypes.get_errno())
            return sum(b & 1 for b in vec.raw) * PG, n
        finally:
            libc.munmap(a, n)
    finally:
        os.close(fd)


bad = False
for d in DIRS:
    d = os.path.expanduser(d)
    files = sorted(f for f in os.listdir(d) if f.endswith(".bin"))
    tot = res = 0
    hot = []
    for f in files:
        r, n = resident(os.path.join(d, f))
        if 100 * r / n > LINE:
            if EVICT:
                r2, _ = resident(os.path.join(d, f), evict=True)
                hot.append(f"{f} {100 * r / n:.1f}->{100 * r2 / n:.1f}%")
                r = r2
            else:
                hot.append(f"{f} {100 * r / n:.1f}%")
            bad |= 100 * r / n > LINE                  # the line is per file, after any eviction
        tot += n; res += r
    pct = 100 * res / tot if tot else 0.0
    print(f"{d}: {len(files)} files, {tot / 1e9:.1f} GB, resident {res / 1e9:.3f} GB = {pct:.2f} % (line {LINE} %)"
          + (f"; above the line: {', '.join(hot[:8])}{' ...' if len(hot) > 8 else ''}" if hot else ""), flush=True)
sys.exit(1 if bad else 0)
