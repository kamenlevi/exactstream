"""Cold read latency of one expert: 9 pieces (as stored in the checkpoint, all in flight) vs one
contiguous read of the same size, and 4 reads (3 weights + 1 repacked scales/biases blob).
Uses the 35B 4-bit checkpoint, page cache evicted first. -> data/piece_bench.txt"""
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
pieces = []
for L in range(40):
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp"
    row = []
    for proj, fld in PIECES:
        path, off, dt, shape = hdr[f"{pre}.{proj}.{fld}"]
        per = int(np.prod(shape[1:])) * (4 if dt == "U32" else 2)
        row.append((path, off, per))
    pieces.append(row)
ex = ThreadPoolExecutor(16); rng = random.Random(0)
def one(L, e, mode):
    row = pieces[L]
    if mode == "pieces":
        jobs = [(fd(p), o + e * per, per) for p, o, per in row]
    elif mode == "contig":
        tot = sum(per for _, _, per in row); p, o, per = row[0]
        jobs = [(fd(p), o + e * per, tot)]
    else:  # 3 weights + 1 combined small read
        w = [r for r, (proj, f) in zip(row, PIECES) if f == "weight"]
        small = sum(per for (p, o, per), (proj, f) in zip(row, PIECES) if f != "weight")
        jobs = [(fd(p), o + e * per, per) for p, o, per in w] + [(fd(row[1][0]), row[1][1] + e * small, small)]
    t = time.perf_counter()
    list(ex.map(lambda j: os.pread(j[0], j[2], j[1]), jobs))
    return time.perf_counter() - t
out = []
for mode in ("pieces", "contig", "4reads"):
    ts = []
    for _ in range(150):
        L = rng.randrange(40); e = rng.randrange(256)
        ts.append(one(L, e, mode))
    ts.sort(); out.append(f"{mode:7s}: one expert (1.77 MB) cold, all reads in flight: p50 {ts[75]*1e3:.3f} ms, p90 {ts[135]*1e3:.3f} ms")
    print(out[-1], flush=True)
open(os.path.expanduser("~/BigMoE/data/piece_bench.txt"), "w").write("\n".join(out) + "\n")
