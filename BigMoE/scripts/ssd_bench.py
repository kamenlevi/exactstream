"""Cold SSD random-read throughput (F_NOCACHE, pread, thread pool) for this model's
read shapes: one Q4 expert blob (2.77 MB), and 4 KB pages (n-gram table rows).
Reads only from big model files not in use; never writes. Results -> data/ssd.txt"""
import fcntl, glob, os, random, time, sys
from concurrent.futures import ThreadPoolExecutor
FILES = sorted(glob.glob(os.path.expanduser("~/.lmstudio/models/unsloth/*/model-*.safetensors"))
               + glob.glob(os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-6bit/*.safetensors")))
fds = []
for f in FILES:
    fd = os.open(f, os.O_RDONLY); fcntl.fcntl(fd, fcntl.F_NOCACHE, 1); fds.append((fd, os.path.getsize(f)))
rng = random.Random(1); used = set()
def offs(n, size, align=16384):
    out = []
    while len(out) < n:
        i = rng.randrange(len(fds)); fd, fsz = fds[i]
        o = rng.randrange(0, (fsz - size) // align) * align
        key = (i, o >> 22)
        if key in used: continue
        used.add(key); out.append((fd, o))
    return out
def run(size, qd, n):
    o = offs(n, size); lat = []
    def rd(a):
        t = time.perf_counter(); r = len(os.pread(a[0], size, a[1])); lat.append(time.perf_counter() - t); return r
    t0 = time.perf_counter()
    with ThreadPoolExecutor(qd) as ex: got = sum(ex.map(rd, o))
    dt = time.perf_counter() - t0; lat.sort()
    return got / dt / 1e9, n / dt, lat[len(lat)//2] * 1e3, lat[int(len(lat)*.99)] * 1e3
lines = [f"pool: {len(FILES)} files, {sum(s for _, s in fds)/1e9:.0f} GB"]
for size, lab, qds, n in [(2_764_800, "expert 2.77MB", (1, 4, 8, 16, 32), 600),
                          (4096, "4KB page", (1, 16, 64), 4000)]:
    for qd in qds:
        gbs, iops, p50, p99 = run(size, qd, n)
        lines.append(f"{lab:14s} QD{qd:<3d} {gbs:6.2f} GB/s  {iops:8.0f} IO/s  p50 {p50:.3f} ms  p99 {p99:.3f} ms")
open("data/ssd.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
