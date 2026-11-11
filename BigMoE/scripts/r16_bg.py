"""R-16 overlap screen (b42) background load: MoonShots' W24-ec GPU rANS decode (~/MoonShots/scripts/w24ec.py) of real
MSB-3 codes (32 random experts each of layers 0 / 24 / 47, ~177 MB of codes per call), looped in its own process and paced
to RATE GB/s of decoded codes while the engine decodes in another process. The first decode is checked bit for bit
against the records3 bytes; the log gets the achieved rate and the decode time per call.
usage: r16_bg.py RATE_GBS SECONDS LOG   (stops early when LOG + '.stop' exists; prints 'ready' once encoded)"""
import os, sys, time
RATE, SECS, LOG = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
sys.argv = sys.argv[:1]                                      # w24ec reads its expert count from argv[1]
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import numpy as np
import mlx.core as mx
import w24ec as W


def say(s):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {s}\n")


rng = np.random.default_rng(16)                               # as W.main()
allw, allo, alls, allt, alltab, raw_all, base = [], [], [], [], [], [], 0
for li, L in enumerate(W.LAYERS):
    ex = sorted(rng.choice(512, W.NE, replace=False).tolist())
    raw = W.read_codes(L, ex)
    sym = W.unpack3(raw)
    f, c, slots, _ = W.table(sym)
    chunks = sym.reshape(-1, W.CH)
    words, off, st = W.encode(chunks, f, c)
    allw.append(words); allo.append(off[:-1].astype(np.int64) + base); base += len(words)
    alls.append(st); allt.append(np.full(len(chunks), li, np.uint32)); alltab.append((slots, f, c)); raw_all.append(raw)
A = [mx.array(np.concatenate(allw)), mx.array(np.concatenate(allo).astype(np.uint32)),
     mx.array(np.concatenate(alls).reshape(-1)), mx.array(np.concatenate([t[0] for t in alltab])),
     mx.array(np.concatenate([((t[1].astype(np.uint32) << 16) | t[2].astype(np.uint32)) for t in alltab])),
     mx.array(np.concatenate(allt))]
mx.eval(A)
NC = int(A[5].shape[0])
K = W.kernel()
gb = NC * W.CH * 3 / 8 / 1e9
y = W.gpu_decode(K, A, NC, 2)
mx.eval(y)
bad = int((np.array(y).reshape(-1) != np.concatenate([r.reshape(-1) for r in raw_all])).sum())
say(f"ready: {NC} chunks, {gb * 1e3:.1f} MB of codes per call, mismatching bytes {bad}, pacing to {RATE:.1f} GB/s")
print("ready", flush=True)
t0 = nxt = time.perf_counter()
n, busy = 0, 0.0
while time.perf_counter() - t0 < SECS and not os.path.exists(LOG + ".stop"):
    a = time.perf_counter()
    mx.eval(W.gpu_decode(K, A, NC, 2))
    busy += time.perf_counter() - a
    n += 1
    nxt += gb / RATE
    time.sleep(max(0.0, nxt - time.perf_counter()))
el = time.perf_counter() - t0
say(f"done: {n} calls in {el:.1f} s = {n * gb / max(el, 1e-9):.2f} GB/s of codes decoded; {1e3 * busy / max(n, 1):.2f} ms per "
    f"177 MB call alongside the engine ({n * gb / max(busy, 1e-9):.1f} GB/s while decoding)")
