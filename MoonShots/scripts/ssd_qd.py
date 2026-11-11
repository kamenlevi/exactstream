"""SSD read bandwidth vs queue depth on this Mac (research-first build step 1 for the 24 GB goal; PRIOR_ART.md lever 1).
Cold random preads with F_NOCACHE from the 125B's expert tensors (~/BigMoE/models/bare, read-only); every (layer, expert)
is used once across the whole run, so neither the page cache nor a previous read helps. Two patterns:
  whole : one contiguous 2.77 MB read per expert-sized request (what an expert-major file layout allows)
  pieces: BigMoE's layout today: 9 preads per expert (gate/up/down codes ~0.82 MB each + 6 scale/bias reads ~51 KB),
          issued together, as its loader does with dispatch_apply
QD = concurrent preads (a thread pool; os.pread releases the GIL). Reports GB/s and experts/s per QD.
SlotStream measured an M5 Pro 2 TB SSD at 9.5 / 17.3 / 17.3 GB/s for cold 2.76 MB F_NOCACHE preads at QD 1 / 8 / 32.
usage: ssd_qd.py   (I/O heavy for ~1 min: run under a quiet lock)"""
import fcntl, json, os, random, struct, time
from concurrent.futures import ThreadPoolExecutor
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
fds, hdr = {}, {}


def meta(name):
    f = f"{MD}/{idx[name]}"
    if f not in hdr:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr[f] = (n, json.loads(fh.read(n)))
        fd = os.open(f, os.O_RDONLY); fcntl.fcntl(fd, fcntl.F_NOCACHE, 1); fds[f] = fd
    n, h = hdr[f]
    a, b = h[name]["data_offsets"]
    E = h[name]["shape"][0]
    return fds[f], 8 + n + a, (b - a) // E


PIECES = [(p, fl) for p in ("gate_proj", "up_proj", "down_proj") for fl in ("weight", "scales", "biases")]
LAYERS = 48
pieces = {L: [meta(f"language_model.model.layers.{L}.mlp.switch_mlp.{p}.{fl}") for p, fl in PIECES] for L in range(LAYERS)}
EXPERT = sum(per for _, _, per in pieces[0])
rng = random.Random(7)
pool = [(L, e) for L in range(LAYERS) for e in range(512)]
rng.shuffle(pool)
cursor = [0]


def take(n):
    out = pool[cursor[0]:cursor[0] + n]; cursor[0] += n
    assert len(out) == n, "ran out of cold experts"
    return out


def read(fd, off, size):
    got = 0
    while got < size:
        r = os.pread(fd, size - got, off + got)
        if not r:
            break
        got += len(r)
    return got


def run(pattern, qd, n_exp):
    exps = take(n_exp)
    if pattern == "whole":        # one EXPERT-sized contiguous read inside the expert's gate_proj codes region
        jobs = []
        for L, e in exps:
            fd, off, per = pieces[L][0]
            e0 = min(e, 512 - 4)  # 2.77 MB spans ~3.4 experts of gate codes; stays inside the tensor
            jobs.append((fd, off + e0 * per, EXPERT))
    else:
        jobs = [(fd, off + e * per, per) for L, e in exps for fd, off, per in pieces[L]]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=qd) as ex:
        total = sum(ex.map(lambda j: read(*j), jobs))
    dt = time.perf_counter() - t0
    return total / dt / 1e9, n_exp / dt


print(f"expert = {EXPERT / 1e6:.2f} MB in 9 pieces ({', '.join(f'{per / 1e3:.0f}K' for _, _, per in pieces[0])})", flush=True)
for pattern in ("whole", "pieces"):
    for qd in (1, 2, 4, 8, 16, 32, 64):
        gbs, eps = run(pattern, qd, 240 if qd >= 4 else 120)
        print(f"{pattern:6s} QD {qd:2d}: {gbs:5.2f} GB/s, {eps:6.0f} experts/s ({1e3 / eps * 10:.2f} ms per 10 experts)", flush=True)
