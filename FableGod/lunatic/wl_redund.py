# WALL_L test 1: is there shared structure between the 512 experts of a layer (a base + small deltas, a common
# subspace, near-duplicate neurons), and are the experts' OUTPUTS on real inputs orthogonal (re-routing / learned
# corrections)? Real weights: models/bare (MLX 4-bit affine g64). Real inputs: TrainLab d02a taps (if they fit).
# CPU, numpy, read-only; writes nothing but stdout.
import json, sys, os, struct, time
import numpy as np

BARE = os.path.expanduser('~/BigMoE/models/bare')
IDX = json.load(open(f'{BARE}/model.safetensors.index.json'))['weight_map']
rng = np.random.default_rng(1105)
_hdr = {}

def tensor(name):
    f = f"{BARE}/{IDX[name]}"
    if f not in _hdr:
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            _hdr[f] = (json.loads(fh.read(n)), 8 + n)
    h, base = _hdr[f]
    m = h[name]
    dt = {'U32': np.uint32, 'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32, 'U8': np.uint8}[m['dtype']]
    a, b = m['data_offsets']
    arr = np.memmap(f, dtype=dt, mode='r', offset=base + a, shape=tuple(m['shape']))
    return arr, m['dtype']

def f32(a, dt):
    a = np.asarray(a)
    if dt == 'BF16':
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)

def deq(prefix, sl, bits=4, gs=64):
    """dequantize experts sl of a stacked MLX affine weight -> float32 [n, out, in]"""
    w, _ = tensor(prefix + '.weight'); s, sd = tensor(prefix + '.scales'); b, bd = tensor(prefix + '.biases')
    w = np.asarray(w[sl]); s = f32(s[sl], sd); b = f32(b[sl], bd)
    per = 32 // bits; mask = (1 << bits) - 1
    q = np.stack([(w >> (bits * i)) & mask for i in range(per)], axis=-1).reshape(*w.shape[:-1], w.shape[-1] * per)
    q = q.astype(np.float32).reshape(*q.shape[:-1], -1, gs)
    out = q * s[..., None] + b[..., None]
    return out.reshape(*q.shape[:-2], -1)

def spectrum_frac(G, rs=(64, 256, 1024)):
    ev = np.linalg.eigvalsh(G.astype(np.float64))[::-1]
    ev = np.clip(ev, 0, None); tot = ev.sum()
    return [float(ev[:r].sum() / tot) for r in rs]

LAYERS = [int(x) for x in (sys.argv[1:] or ['3', '23', '43'])]
NG = 128                                    # experts sampled for the stacked spectra
t0 = time.time()
for L in LAYERS:
    P = f'language_model.model.layers.{L}.mlp.switch_mlp.'
    print(f"\n== layer {L}")
    samp = np.sort(rng.choice(512, NG, replace=False))
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        G = np.zeros((2560, 2560), np.float32); msum = None; e2 = 0.0
        for c in range(0, NG, 16):
            W = deq(P + proj, samp[c:c + 16])                     # [16, out, in]
            if proj == 'down_proj':                               # [16, 2560, 640]: output side = rows
                G += np.einsum('eoi,epi->op', W, W, optimize=True)
            else:                                                 # [16, 640, 2560]: input side = columns
                Wr = W.reshape(-1, 2560); G += Wr.T @ Wr
            msum = W.sum(0) if msum is None else msum + W.sum(0)
            e2 += float((W.astype(np.float64) ** 2).sum())
        mean = msum / NG
        base_frac = float((mean.astype(np.float64) ** 2).sum() * NG / e2)
        fr = spectrum_frac(G)
        side = 'output' if proj == 'down_proj' else 'input'
        print(f"  {proj:9s} index-aligned mean expert holds {base_frac * 100:5.2f} % of the energy (independent: {100 / NG:.2f} %);"
              f" stacked {side}-side spectrum top-64/256/1024 of 2560 dims: {fr[0] * 100:4.1f} / {fr[1] * 100:4.1f} / {fr[2] * 100:4.1f} %"
              f" (isotropic: 2.5 / 10.0 / 40.0 %)", flush=True)
    # single experts' own low rank (a small delta would need this) and near-duplicate neurons across experts
    Wg = deq(P + 'gate_proj', samp[:32])                          # [32, 640, 2560]
    lr = []
    for e in range(8):
        sv = np.linalg.svd(Wg[e], compute_uv=False) ** 2
        lr.append([sv[:r].sum() / sv.sum() for r in (16, 64, 160)])
    lr = np.mean(lr, 0)
    print(f"  one expert's gate (640 x 2560): top-16/64/160 singular directions hold {lr[0]*100:4.1f} / {lr[1]*100:4.1f} / {lr[2]*100:4.1f} % "
          f"(random 640 x 2560: ~4.5 / 16 / 37 %)")
    Wn = Wg / np.linalg.norm(Wg, axis=-1, keepdims=True)
    mx = []
    for e in range(32):
        others = np.concatenate([Wn[j] for j in range(32) if j != e])
        mx.append(np.abs(Wn[e] @ others.T).max(1))
    mx = np.concatenate(mx)
    print(f"  near-duplicate neurons: each gate row's best |cos| with 31 other experts' 19,840 rows: median {np.median(mx):.3f}, "
          f"p99 {np.quantile(mx, .99):.3f}, share > 0.5 {np.mean(mx > 0.5) * 100:.2f} % (independent ~0.09)", flush=True)
    del Wg, Wn
print(f"\n[{time.time() - t0:.0f} s]")
