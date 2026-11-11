"""Synthetic test of the sync-free exact gate: 48 fake layers, each with a pool of C slots of
2.77 MB "experts" backed by real file bytes. Per layer: fake router ids -> gate -> gather the
slots -> checksum (must equal the CPU checksum of the same bytes read from the file) -> a
20 MB quantized matmul as the layer's compute. -> data/test_gate.txt"""
import os, sys, time, fcntl, glob
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
from bigmoe.gated import Gate, ext

NL, E, C, K = 48, 512, 64, 10
PER = 524_288 // 4                                     # uint32 words per fake expert (512 KB; real ones are 2.77 MB)
f = sorted(glob.glob(os.path.expanduser("~/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MLX-8bit/model-*.safetensors")))[0]
fd = os.open(f, os.O_RDONLY); fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
base = lambda L: (L * E) * PER * 4 % (os.path.getsize(f) - E * PER * 4)
pools = [mx.zeros((C, PER), mx.uint32) for _ in range(NL)]; mx.eval(*pools)
for L in range(NL):
    ext.add_layer(C, E, [fd], [base(L)], [PER * 4], [pools[L]])
gate = Gate(NL)
D = 2560; R = int(20e6 / (D / 2 + D / 32))
W = mx.random.normal((R, D)).astype(mx.float16); Wq, Ws, Wb = mx.quantize(W, 64, 4); del W
x0 = mx.random.normal((1, D)).astype(mx.float16); mx.eval(Wq, Ws, Wb, x0)
rng = np.random.default_rng(0)
hot = [rng.choice(E, C - 20, replace=False) for _ in range(NL)]  # a working set that fits

def token(ids_per_layer, gated):
    h = x0; cks = []
    if gated: gate.new_pass()
    for L in range(NL):
        if gated or ids_per_layer == "ungated":
            slots = (gate.route_and_gate(L, mx.array(ids_per_layer[L], mx.int32)) if gated
                     else mx.array(np.arange(K, dtype=np.int32)))
            g = mx.take(pools[L], slots, axis=0)                     # [K, PER]
            ck = (g[:, ::4099].astype(mx.uint64)).sum(axis=1)          # per-expert checksum
            cks.append(ck)
            h = h + (ck.sum() % 7).astype(h.dtype) * 0.0
        y = mx.quantized_matmul(h, Wq, Ws, Wb, transpose=True, group_size=64, bits=4)
        h = h + y[:, :D] * 1e-3
    mx.eval(h, *cks)
    return cks

def cpu_ck(L, e):
    buf = bytearray(PER * 4); os.preadv(fd, [buf], base(L) + e * PER * 4)
    a = np.frombuffer(buf, np.uint32)[::4099].astype(np.uint64); return int(a.sum())

def ids_from(sets):  # K distinct experts per layer from each layer's set
    return [rng.choice(s, K, replace=False).astype(np.int32) for s in sets]

lines = []
# warm: load the hot sets
for _ in range(20): token(ids_from(hot), True)
ext.reset_stats()
def timed(fn, reps=15):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2] * 1e3
t_plain = timed(lambda: token("ungated", False))
t_hit = timed(lambda: token(ids_from(hot), True))
st = ext.stats(); lines.append(f"loader idle-wait per gated call {st['cpu_wait_s']/max(1,st['gated'])*1e6:.0f} us (incl. GPU compute between layers)"); lines.append(f"plain 48 layers: {t_plain:.2f} ms/token | gated, all hits: {t_hit:.2f} ms/token "
                               f"-> {(t_hit - t_plain) / NL * 1e3:.1f} us/layer (gated calls {st['gated']}, misses {st['misses']})")
# misses: on m layers per token, one expert outside the hot set
for m in (4, 12):
    ext.reset_stats(); ts = []
    for _ in range(12):
        ids = ids_from(hot)
        for L in rng.choice(NL, m, replace=False):
            cold = rng.choice(np.setdiff1d(np.arange(E), hot[L])); ids[L][0] = cold
        t = time.perf_counter(); token(ids, True); ts.append(time.perf_counter() - t)
    ts.sort(); tm = ts[len(ts) // 2] * 1e3; st = ext.stats()
    lines.append(f"gated, {m} stalled layers/token: {tm:.2f} ms/token -> {(tm - t_hit) / m * 1e3:.0f} us per stalled layer "
                 f"(misses/token {st['misses'] / 12:.1f}, io {st['io_s'] / 12 * 1e3:.2f} ms/token)")
# correctness: random ids, compare every gathered expert checksum with the file bytes
bad = tot = 0
for _ in range(5):
    ids = [rng.choice(E, K, replace=False).astype(np.int32) for _ in range(NL)]
    cks = token(ids, True)
    for L in range(NL):
        got = np.array(cks[L]).tolist()
        for j, e in enumerate(ids[L].tolist()):
            tot += 1; bad += int(got[j] != cpu_ck(L, e))
lines.append(f"delay {os.environ.get('BIGMOE_DELAY_US','0')} us: correctness: {bad} mismatches of {tot} gathered experts (random ids, many misses)")
gate.check(); gate.stop()
open(os.path.expanduser("~/BigMoE/data/test_gate.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
