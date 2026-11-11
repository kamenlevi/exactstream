"""Is BigMoE's ~1.0 ms per stalled layer SSD-intrinsic? Replays its loader's read pattern on the real 125B files:
for n missing experts of one layer (n = 1..4), all (expert x piece) preads in flight at once (F_NOCACHE), as
bigmoe_ext.cpp's load_jobs does with dispatch_apply; compared with the same bytes as n contiguous reads, and with
reads issued serially. Random cold experts from random layers. Prints wall ms per stall (median, p90).
Usage: stall_io.py   (CPU + SSD only; quiet lock for timing)"""
import fcntl, json, os, struct, time, random, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np

D = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
hdrs = {}


def meta(name):
    f = idx[name]
    if f not in hdrs:
        with open(f"{D}/{f}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdrs[f] = (json.loads(fh.read(n)), 8 + n)
    h, base = hdrs[f]
    m = h[name]
    return f, base + m["data_offsets"][0], m["data_offsets"][1] - m["data_offsets"][0], m["shape"][0]


fds = {}


def fd(f):
    if f not in fds:
        fds[f] = os.open(f"{D}/{f}", os.O_RDONLY)
        fcntl.fcntl(fds[f], fcntl.F_NOCACHE, 1)
    return fds[f]


PIECES = [f"mlp.switch_mlp.{p}.{q}" for p in ("gate_proj", "up_proj", "down_proj") for q in ("weight", "scales", "biases")]


def expert_reads(L, e):
    out = []
    for p in PIECES:
        f, off, size, E = meta(f"language_model.model.layers.{L}.{p}")
        per = size // E
        out.append((fd(f), off + e * per, per))
    return out


pool = ThreadPoolExecutor(64)
bufs = [bytearray(4 << 20) for _ in range(64)]


def rd(job):
    f, off, n = job[0]
    b = bufs[job[1]]
    mv = memoryview(b)[:n]
    done = 0
    while done < n:
        done += os.preadv(f, [mv[done:]], off + done)


def stall_parallel(jobs):
    list(pool.map(rd, [(j, i % 64) for i, j in enumerate(jobs)]))


def stall_serial(jobs):
    for i, j in enumerate(jobs):
        rd((j, i % 64))


rng = random.Random(0)
per_exp = sum(n for _, _, n in expert_reads(0, 0))
print(f"one expert = {len(PIECES)} pieces, {per_exp/1e6:.2f} MB")
for n in (1, 2, 3, 4):
    res = {"parallel pieces": [], "parallel contiguous": [], "serial pieces": []}
    for trial in range(40):
        L = rng.randrange(48)
        es = rng.sample(range(512), n)
        jobs = [r for e in es for r in expert_reads(L, e)]
        # contiguous stand-in: the same byte count as n single reads at random offsets of the down_proj tensor region
        f, off, size, E = meta(f"language_model.model.layers.{L}.mlp.switch_mlp.gate_proj.weight")
        cont = [(fd(f), off + (rng.randrange(size - per_exp) & ~16383), per_exp) for _ in range(n)]
        order = list(res)
        rng.shuffle(order)
        for k in order:
            L2 = rng.randrange(48); es2 = rng.sample(range(512), n)
            j2 = [r for e in es2 for r in expert_reads(L2, e)]
            t = time.perf_counter()
            if k == "parallel pieces":
                stall_parallel(j2)
            elif k == "parallel contiguous":
                f2, off2, size2, _ = meta(f"language_model.model.layers.{L2}.mlp.switch_mlp.up_proj.weight")
                stall_parallel([(fd(f2), off2 + (rng.randrange(size2 - per_exp) & ~16383), per_exp) for _ in range(n)])
            else:
                stall_serial(j2)
            res[k].append(time.perf_counter() - t)
    line = "  ".join(f"{k}: {np.median(v)*1e3:.2f} ms (p90 {np.percentile(v,90)*1e3:.2f})" for k, v in res.items())
    print(f"{n} missing experts ({n*per_exp/1e6:.1f} MB): {line}  -> {n*per_exp/np.median(res['parallel pieces'])/1e9:.1f} GB/s", flush=True)
