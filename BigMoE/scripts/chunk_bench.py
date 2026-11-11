"""Cold read latency of ONE expert (35B 4-bit, 9 pieces, 1.77 MB) when each piece is split into
chunks of at most CH bytes, all in flight at once (thread pool). Page cache evicted first.
-> data/chunk_bench.txt"""
import os, sys, time, fcntl, random
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from concurrent.futures import ThreadPoolExecutor
from bigmoe.stream import read_headers, PIECES
import numpy as np
MD = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
hdr = read_headers(MD); fds = {}
def fd(p):
    if p not in fds:
        f = os.open(p, os.O_RDONLY); fcntl.fcntl(f, fcntl.F_NOCACHE, 1); fds[p] = f
    return fds[p]
rows = []
for L in range(40):
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp"
    r = []
    for proj, fld in PIECES:
        path, off, dt, shape = hdr[f"{pre}.{proj}.{fld}"]
        r.append((path, off, int(np.prod(shape[1:])) * (4 if dt == "U32" else 2)))
    rows.append(r)
ex = ThreadPoolExecutor(64); rng = random.Random(1)
used = set()
def one(CH):
    while True:
        L, e = rng.randrange(40), rng.randrange(256)
        if (L, e) not in used: used.add((L, e)); break
    jobs = []
    for p, o, per in rows[L]:
        base = o + e * per
        for c in range(0, per, CH):
            jobs.append((fd(p), base + c, min(CH, per - c)))
    t = time.perf_counter()
    list(ex.map(lambda j: os.pread(j[0], j[2], j[1]), jobs))
    return time.perf_counter() - t, len(jobs)
out = []
for CH in (1 << 30, 512 << 10, 256 << 10, 128 << 10, 64 << 10):
    ts = []; nj = 0
    for _ in range(120):
        t, nj = one(CH); ts.append(t)
    ts.sort()
    lab = "whole pieces" if CH >= (1 << 29) else f"{CH >> 10} KB chunks"
    out.append(f"{lab:14s} ({nj:2d} reads): one cold expert p50 {ts[60]*1e3:.3f} ms, p90 {ts[108]*1e3:.3f} ms")
    print(out[-1], flush=True)
open(os.path.expanduser("~/BigMoE/data/chunk_bench.txt"), "w").write("\n".join(out) + "\n")
