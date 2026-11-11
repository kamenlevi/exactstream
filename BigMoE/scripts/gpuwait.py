"""Prototype: GPU-side demand wait for experts on Apple unified memory (no CPU<->GPU sync).
A tiny Metal kernel before each layer's expert matmul checks a residency flag in a shared
buffer. Hit: it falls through. Miss: it posts a request in shared memory and spins until
the CPU (polling, never blocking on the GPU) has pread() the expert into its slot and set
the flag. Measures: hit-path overhead per layer, miss stall vs a per-layer mx.eval sync,
and that the GPU sees the bytes the CPU wrote mid-command-buffer. -> data/gpuwait.txt"""
import fcntl, glob, os, sys, time
import numpy as np, mlx.core as mx, faulthandler
faulthandler.dump_traceback_later(int(os.environ.get("WD", 90)), exit=True)
def log(*a): print(*a, flush=True)
NL, EXP = 48, 2_764_800
LAYER_MB = float(os.environ.get("LAYER_MB", 60))        # Q4 weight bytes per fake layer (~0.3 ms)
src = r"""
uint tid = thread_position_in_grid.x;
int k = idx[0];
device atomic_int* f = reinterpret_cast<device atomic_int*>(const_cast<device int*>(flags));
device atomic_int* r = reinterpret_cast<device atomic_int*>(const_cast<device int*>(req));
if (tid == 0) {
  if (atomic_load_explicit(&f[k], memory_order_relaxed) == 0) {
    atomic_store_explicit(&r[k], 1, memory_order_relaxed);
    uint s = 0;
    #if VARIANT == 1
    while (atomic_fetch_add_explicit(&f[k], 0, memory_order_relaxed) == 0 && s < SPINCAP) s++;
#elif VARIANT == 2
    while (((volatile device int*)flags)[k] == 0 && s < SPINCAP) s++;
#else
    while (atomic_load_explicit(&f[k], memory_order_relaxed) == 0 && s < SPINCAP) s++;
#endif
    spins[0] = s;
  }
  #if CHECK
  // checksum a strided sample of the slot (proves the GPU sees the CPU's bytes)
  const device uint* p = reinterpret_cast<const device uint*>(slots) + (size_t)k * (EXPW);
  uint c = 0;
  for (uint i = 0; i < EXPW; i += 997) c = c * 31u + p[i];
  ck[0] = c;
  #endif
  out[0] = x[0];
} 
"""
kern = mx.fast.metal_kernel(name="expert_wait", input_names=["x", "flags", "req", "slots", "idx"],
                            output_names=["out", "ck", "spins"], source=src, header=f"#define EXPW {EXP//4}u\n#define VARIANT {os.environ.get('VARIANT','0')}\n#define CHECK {os.environ.get('CHECK','0')}\n#define SPINCAP {os.environ.get('SPINCAP','20000000')}u\n")
D = 2560; R = int(LAYER_MB * 1e6 / (D / 2 + D / 32))      # rows of a 4-bit [R, D] weight
W = mx.random.normal((R, D)).astype(mx.float16); Wq, Ws, Wb = mx.quantize(W, 64, 4); del W
flags = mx.zeros((16384,), mx.int32); req = mx.zeros((16384,), mx.int32)  # >4 KB: device memory, not a set_bytes copy
slots = mx.zeros((NL * EXP // 4,), mx.uint32)
idxs = [mx.array([k], mx.int32) for k in range(NL)]
mx.eval(Wq, Ws, Wb, flags, req, slots, *idxs)
F = np.frombuffer(memoryview(flags), np.int32); Q = np.frombuffer(memoryview(req), np.int32)
SL = np.frombuffer(memoryview(slots), np.uint8).reshape(NL, EXP)
assert not memoryview(flags).readonly
x0 = mx.random.normal((1, D)).astype(mx.float16); mx.eval(x0)
def layer(h):
    y = mx.quantized_matmul(h, Wq, Ws, Wb, transpose=True, group_size=64, bits=4)
    return h + y[:, :D] * 1e-3
cks = []; SP = []
def token(mode, sync=False):
    h = x0
    for L in range(NL):
        if mode == "wait":
            o, c, sp = kern(inputs=[h, flags, req, slots, idxs[L]], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                        output_shapes=[h.shape, (1,), (1,)], output_dtypes=[h.dtype, mx.uint32, mx.uint32], init_value=0)
            h = o; cks.append(c); SP.append(sp)
        if sync: mx.eval(h)
        h = layer(h)
    return h
files = sorted(glob.glob(os.path.expanduser("~/.lmstudio/models/unsloth/*/model-*.safetensors")))
fds = []
for f in files:
    fd = os.open(f, os.O_RDONLY); fcntl.fcntl(fd, fcntl.F_NOCACHE, 1); fds.append((fd, os.path.getsize(f)))
rng = np.random.default_rng(0); SEEN = {}; T0 = 0.0
def serve(h, miss_layers, do_io, deadline=5.0):
    """Run one token asynchronously; CPU polls requests and serves them (never waits on GPU)."""
    global T0; T0 = time.perf_counter(); mx.async_eval(h); served = 0
    pending = set(miss_layers)
    t_end = time.perf_counter() + deadline
    while pending and time.perf_counter() < t_end:
        for k in list(pending):
            if Q[k]:
                SEEN[k] = time.perf_counter() - T0
                if do_io:
                    fd, sz = fds[rng.integers(len(fds))]
                    os.preadv(fd, [SL[k]], int(rng.integers(0, (sz - EXP) // 16384)) * 16384)
                F[k] = 1; pending.discard(k); served += 1
    mx.eval(h)
    return served, pending
def timed(fn, reps=12):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2] * 1e3, ts[0] * 1e3
lines = [f"fake layer: Q4 [{R} x {D}] = {LAYER_MB:.0f} MB, {NL} layers/token"]
for _ in range(3): mx.eval(token("none"))
F[:] = 1
log("stage: baseline"); base = timed(lambda: mx.eval(token("none")))
log("stage: hits"); hit = timed(lambda: mx.eval(token("wait")))
log("stage: sync"); syncd = timed(lambda: mx.eval(token("none", sync=True)))
lines += [f"baseline, no checks          {base[0]:7.2f} ms/token (min {base[1]:.2f})",
          f"wait-kernel, all hits        {hit[0]:7.2f} ms/token  -> {(hit[0]-base[0])/NL*1e3:6.1f} us/layer",
          f"mx.eval sync every layer     {syncd[0]:7.2f} ms/token  -> {(syncd[0]-base[0])/NL*1e3:6.1f} us/layer"]
log("stage: misses")
for nmiss, io in [(4, False), (4, True), (12, True)]:
    ts = []
    for rep in range(10):
        ml = sorted(rng.choice(NL, nmiss, replace=False).tolist())
        F[:] = 1; F[ml] = 0; Q[:] = 0
        t = time.perf_counter(); n, pend = serve(token("wait"), ml, io); ts.append(time.perf_counter() - t)
        if pend: log("UNSERVED", sorted(pend), "GPU spins", [SP[-NL+k].item() for k in ml]); break
    ts.sort(); m = ts[len(ts) // 2] * 1e3
    log("  last run: layer, CPU saw request at ms, GPU spins:", [(k, round(SEEN.get(k, -1)*1e3, 2), SP[-NL+k].item()) for k in ml][:6])
    lines.append(f"{nmiss:2d} stalled layers, {'pread 2.77MB cold' if io else 'no IO (round trip)'}: "
                 f"{m:7.2f} ms/token -> {(m-hit[0])/nmiss*1e3:6.1f} us per stalled layer")
# correctness: last run used IO; compare GPU checksums with CPU recomputation of the slot bytes
mx.eval(*cks[-NL:])
bad = 0
for k in range(NL):
    p = SL[k].view(np.uint32); c = np.uint32(0)
    with np.errstate(over="ignore"):
        for i in range(0, EXP // 4, 997): c = np.uint32(c * np.uint32(31) + p[i])
    bad += int(c != np.uint32(cks[-NL + k].item()))
lines.append(f"variant {os.environ.get('VARIANT','0')} check {os.environ.get('CHECK','0')}"); lines.append(f"checksum mismatches GPU vs CPU over {NL} slots: {bad}")
os.makedirs(os.path.expanduser("~/BigMoE/data"), exist_ok=True)
open(os.path.expanduser("~/BigMoE/data/gpuwait.txt"), "a").write("\n".join(lines) + "\n\n"); print("\n".join(lines))
