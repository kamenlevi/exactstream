"""Can the pack4 down layout serve prefill too? Real 125B layer-0 down weights, a 2048-token prefill chunk routed with the real
125B traces (MoEPrefetch traces125: 800-token prompts stacked to 2048, top-10 -> 20480 (token, expert) pairs).
  mlx      : SwitchGLU's sorted path (pairs sorted by expert, gather_qmm(..., sorted_indices=True)) on the MLX layout
  down4    : the pack4 kernel with pairs pre-sorted and each pair's rank inside its expert precomputed (no in-kernel scan),
             PMAX pairs per owner (each owner re-reads the expert's weights)
  unpermute: cost of turning packed experts back into the MLX layout on the GPU (what a permuting PF1/PF2 scratch gather pays),
             per expert and for all experts of the chunk
Prints ms per call (median), determinism, and max rel diff vs mlx.   usage: prefill_down4.py   (GPU, quiet lock)"""
import glob, json, os, struct, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xkd
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]; dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = mx.array(np.array(np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,))).reshape(m["shape"]))
    return arr.view(mx.bfloat16) if m["dtype"] == "BF16" else arr


dn = tuple(tensor(f"language_model.model.layers.0.mlp.switch_mlp.down_proj.{f}") for f in ("weight", "scales", "biases"))
mx.eval(dn)
P4T = 64
packed4 = xkd.pack4(*dn, tpg=P4T); mx.eval(packed4)
E, N, F, KT, M = 512, 2560, 640, 10, 2048
tr = [np.load(f)["L0"][:, :KT].astype(np.int64) for f in sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))]
ids_np = np.concatenate(tr)[:M]                                     # [2048, 10] real routing
P = M * KT
flat = ids_np.reshape(-1)
order = np.argsort(flat, kind="stable")
sids = flat[order]
counts = np.bincount(sids, minlength=E)
first = np.concatenate([[0], np.cumsum(counts)[:-1]])
rank = np.arange(P) - first[sids]
print(f"chunk {M} tokens x top-{KT}: {P} pairs, {int((counts > 0).sum())} experts used, rows/expert mean {counts[counts > 0].mean():.1f} "
      f"max {counts.max()}", flush=True)
h = (mx.random.normal((P, F)) * 0.5).astype(mx.bfloat16)          # one row per pair, already in sorted order
sid = mx.array(sids.astype(np.uint32))


def tm(f, n=7):
    for _ in range(2): mx.eval(f())
    ts = []
    for _ in range(n):
        a = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - a)
    return sorted(ts)[n // 2] * 1e3


mlx_f = lambda: mx.gather_qmm(h[:, None, :], *dn, rhs_indices=sid, transpose=True, group_size=64, bits=4, sorted_indices=True)[:, 0]
t_mlx = tm(mlx_f)
ref = mlx_f().astype(mx.float32)
print(f"mlx sorted gather_qmm: {t_mlx:.2f} ms", flush=True)

# down4 with presorted pairs: owners are pairs whose rank % PMAX == 0 (kernel scan replaced by the rank input)
SRC = xkd.SRC_P4.replace("""    int jj = 0;
    for (int q = 0; q < p; q++) jj += ids[q] == e;
    if (jj % PMAX) return;
    int srv[PMAX]; int ns = 0;
    for (int q = p; q < P && ns < PMAX; q++) if (ids[q] == e) srv[ns++] = q;""", """    if (rk[p] % PMAX) return;
    int srv[PMAX]; int ns = 0;
    for (int q = p; q < P && ns < PMAX && ids[q] == e; q++) srv[ns++] = q;""")
assert SRC != xkd.SRC_P4
K4 = mx.fast.metal_kernel(name="xkd_p4_sorted", input_names=["h", "ids", "rk", "pw", "psb"], output_names=["y"],
                          header="#define SHAREDQ 0\n" + xkd.HDR, source=SRC)
rk = mx.array(rank.astype(np.int32)); ids32 = mx.array(sids.astype(np.int32))
for pm in (2, 4):
    f = lambda pm=pm: K4(inputs=[h, ids32, rk, *packed4], template=[("K", F), ("N", N), ("P", P), ("PMAX", pm), ("TPG", P4T)],
                         grid=(P * (N // (4 * P4T)) * P4T, 1, 1), threadgroup=(P4T, 1, 1),
                         output_shapes=[(P, N)], output_dtypes=[mx.bfloat16])[0]
    t = tm(f, 3)
    y = f(); det = mx.array_equal(y, f()).item()
    rel = (mx.abs(y.astype(mx.float32) - ref).max() / mx.abs(ref).max()).item()
    print(f"down4 presorted PMAX {pm}: {t:.2f} ms ({t/t_mlx:.1f}x mlx), deterministic {det}, rel diff {rel:.1e}", flush=True)

# un-permute packed -> MLX layout on the GPU: a gather with the inverse index map (one expert, then all used experts)
pw, psb = packed4
words = N * (F // 8)
inv = np.array(xkd.pack4(mx.arange(1 * words, dtype=mx.uint32).reshape(1, N, F // 8),
                         mx.zeros((1, N, F // 64), mx.bfloat16), mx.zeros((1, N, F // 64), mx.bfloat16), tpg=P4T)[0])
perm = np.empty(words, np.int64); perm[inv] = np.arange(words)       # packed position of each MLX-layout word
perm_m = mx.array(perm.astype(np.uint32))
used = np.nonzero(counts)[0]
for n_e in (1, len(used)):
    sel = mx.array(used[:n_e].astype(np.uint32))
    def unperm():
        base = (sel[:, None] * words).astype(mx.uint32) + perm_m[None]
        return mx.take(pw, base.reshape(-1))
    t = tm(unperm)
    print(f"unpermute {n_e} experts' down weights ({n_e * words * 4 / 1e6:.0f} MB): {t:.2f} ms "
          f"({n_e * words * 4 * 2 / t / 1e6:.0f} GB/s read+write)", flush=True)
