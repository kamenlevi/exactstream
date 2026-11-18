"""Prefill 1.5K: does bigmoe_ext_ov's read_rebuilt(_layout) (a copy of BigMoE's ext with the GIL released) give the same bytes
as the original, and does it let the main thread run Python while the read runs? CPU + SSD only, ~1.1 GB, no GPU work beyond
allocating the scratches. Layer L's first N experts (default 0, 192) are read by each binding into its own scratch (rows
0..N-1) and compared bit for bit; then each binding reads in a background thread while the main thread counts in a pure
Python loop: iterations per ms during the read vs alone. usage: gil_read_test.py <ext_ov build dir> [L] [N]"""
import importlib.util, json, os, struct, sys, threading, time
import numpy as np

CL = "/Users/kamen/Leap/release/cleanclone/cache/clone"
sys.path.insert(0, os.path.join(CL, "engine"))
sys.path.insert(0, os.path.join(CL, "engine", "ext", "build"))
import mlx.core as mx
from bigmoe import tier as _tier
from bigmoe.stream import PIECES
import bigmoe_ext as ext0

B = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N = int(sys.argv[3]) if len(sys.argv) > 3 else 192
spec = importlib.util.spec_from_file_location("bigmoe_ext_ov", [os.path.join(B, f) for f in os.listdir(B) if f.startswith("bigmoe_ext_ov") and f.endswith(".so")][0])
ext1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext1)

MD = os.path.join(CL, "engine", "models", "bare")
tier_dir, ix = _tier.load_index(MD, layout_ok=True)
r = ix["layers"][str(L)]
wm = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
DT = {"U32": (np.uint32, mx.uint32), "BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, mx.float16), "F32": (np.float32, mx.float32)}


def meta(name):
    with open(os.path.join(MD, wm[name]), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        m = json.loads(fh.read(n))[name]
    return m["shape"], m["dtype"]


pre = f"language_model.model.layers.{L}.mlp.switch_mlp."
sh4, dt4 = [], []
for proj, field in PIECES:
    shp, dt = meta(pre + f"{proj}.{field}")
    sh4.append(tuple(shp[1:])); dt4.append(DT[dt])
pers4 = [int(np.prod(sh)) * np.dtype(d[0]).itemsize for sh, d in zip(sh4, dt4)]
args = dict(path=os.path.join(tier_dir, r["file"]), recsz=r["recsz"], msb_sz=r["msb_sz"], pers4=pers4,
            layout=_tier.LAYOUTS[r.get("layout", "row-major")], rows3=[sh4[3 * p][0] for p in range(3)],
            offs=list(r.get("msb_piece_off", [])) + list(r.get("lsb_piece_off", [])))
print(f"layer {L}: {r['file']} recsz {r['recsz']} layout {args['layout']}; per-expert piece bytes {sum(pers4) / 1e6:.2f} MB; "
      f"reading experts 0..{N - 1} ({N * sum(pers4) / 1e9:.2f} GB per read)", flush=True)


def scratch():
    s = [mx.zeros((N,) + sh, d[1]) for sh, d in zip(sh4, dt4)]
    mx.eval(*s)
    return s


def read(ext, dst):
    ids = list(range(N))
    if args["layout"]:
        ext.read_rebuilt_layout(args["path"], args["recsz"], args["msb_sz"], args["pers4"], ids, ids, dst, args["layout"],
                                args["rows3"], args["offs"])
    else:
        ext.read_rebuilt(args["path"], args["recsz"], args["msb_sz"], args["pers4"], ids, ids, dst)


s0, s1 = scratch(), scratch()
t0 = time.perf_counter(); read(ext0, s0); ta = time.perf_counter() - t0
t0 = time.perf_counter(); read(ext1, s1); tb = time.perf_counter() - t0
same = all(np.array_equal(np.array(a.view(mx.uint8) if a.dtype != mx.uint32 else a), np.array(b.view(mx.uint8) if b.dtype != mx.uint32 else b))
           for a, b in zip(s0, s1))
print(f"bitwise: original vs ov scratches {'EQUAL' if same else 'DIFFER'} (read {ta * 1e3:.0f} / {tb * 1e3:.0f} ms)", flush=True)


def spin(stop):
    n = 0
    while not stop.is_set():
        n += 1
    return n


def loop_rate(seconds):
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        n += 1
    return n / ((time.perf_counter() - t0) * 1e3)


base = loop_rate(0.3)
for name, ext, dst in (("original (GIL held)", ext0, s0), ("ov (GIL released)", ext1, s1)):
    done = threading.Event()
    th = threading.Thread(target=lambda: (read(ext, dst), done.set()))
    n, t0 = 0, time.perf_counter()
    th.start()
    while not done.is_set():
        n += 1
    dt = time.perf_counter() - t0
    th.join()
    print(f"{name:22s}: read {dt * 1e3:6.0f} ms; main-thread Python loop {n / (dt * 1e3):8.0f} iterations/ms during the read "
          f"(alone: {base:.0f}/ms) -> {100 * n / (dt * 1e3) / base:5.1f} % of its speed", flush=True)
