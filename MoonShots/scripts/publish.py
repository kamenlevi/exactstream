"""Can a running Metal kernel publish a value to a spinning CPU thread mid-kernel, fast, if the GPU keeps streaming memory
(as a real layer does with its weights)? BigMoE found GPU writes reach the CPU only at command-buffer end or after a random
0.5–1000 ms *while the kernel only spins*, and that a 64 MB one-thread sweep published it (but slowly).
Here: thread 0 of threadgroup 0 stores req[0]=1 at kernel start, then ALL threadgroups stream X MB of a big buffer
(coalesced, like a matvec's weight reads) or, in the 'idle' variant, thread 0 just spins ~2 ms with no traffic.
The CPU spins on req[0] (zero-copy numpy view of the MLX buffer) and records when it sees it vs when the kernel completes.
publish delay ≈ seen(X) − seen(X=0). Fast mid-kernel publish ⇒ a persistent per-token kernel can hand routing ids to the
CPU loader without splitting command buffers. Small GPU job; needs a quiet lock."""
import time
import numpy as np
import mlx.core as mx

BIG = mx.random.randint(0, 2**31, ((512 << 20) // 4,)).astype(mx.uint32)   # 512 MB, random (no zero-page shortcut)
mx.eval(BIG)
G = 160                                                                       # threadgroups x 256 threads

SRC = """
    uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    device int* r = const_cast<device int*>(req);
    if (tg == 0 && t == 0) {
        *(volatile device int*)r = 1;
    }
    if (EXIT && tg == 0) return;                               // publisher threadgroup: store, then finish
    if (PUB && tg == 1) {                                      // late publisher: wait for a GPU-side phase flag, store, exit
        if (t == 0) {
            device atomic_int* ph = (device atomic_int*)(r + 4096);
            uint s = 0;
            while (atomic_load_explicit(ph, memory_order_relaxed) == 0 && s < 40000000u) s++;
            *(volatile device int*)(r + 8) = 1;
        }
        return;
    }
    uint acc = 0;
    if (IDLE) {
        if (tg == 0 && t == 0) { for (uint i = 0; i < 3000000u; ++i) acc += i * 2654435761u; }
    } else {
        const uint per = (uint)(XW / G);                       // words per threadgroup
        const device uint* p = big + (size_t)tg * per;
        for (uint i = t; i < per; i += 256) {
            acc ^= p[i];
            if (PUB && tg == 2 && i == t + 256u * 64u && t == 0)   // a worker raises the phase flag early in its stream
                atomic_store_explicit((device atomic_int*)(r + 4096), 1, memory_order_relaxed);
        }
    }
    out[tg * 256 + t] = acc;
"""
k = mx.fast.metal_kernel(name="publish", input_names=["req", "big"], output_names=["out"], source=SRC)


def run(xmb, idle=0, reps=7, exit_=0, pub=0):
    req = mx.zeros((16384,), mx.int32); mx.eval(req)
    Q = np.frombuffer(memoryview(req), np.int32)
    seen_l, done_l = [], []
    for _ in range(reps):
        Q[:] = 0
        o = k(inputs=[req, BIG], template=[("XW", (xmb << 20) // 4), ("G", G), ("IDLE", idle), ("EXIT", exit_), ("PUB", pub)],
              grid=(G * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(G * 256,)], output_dtypes=[mx.uint32])[0]
        t0 = time.perf_counter(); mx.async_eval(o); seen = None
        while time.perf_counter() - t0 < 2.0:
            if (Q[8] if pub else Q[0]):
                seen = time.perf_counter() - t0
                break
        mx.eval(o); done = time.perf_counter() - t0
        seen_l.append(seen if seen is not None else float("nan")); done_l.append(done)
    s = np.nanmedian(seen_l) * 1e3; d = np.median(done_l) * 1e3
    return s, d, sum(np.isnan(seen_l))


run(16)                                                      # warm-up / compile
print(f"{G} threadgroups x 256 threads; CPU-seen time of a store made at kernel start vs kernel completion (ms, medians)")
base = None
for xmb in (0, 1, 4, 16, 64, 256, 512):
    s, d, miss = run(xmb)
    base = s if base is None else base
    print(f"stream {xmb:>3} MB after the store: seen {s:7.3f}  done {d:7.3f}  -> publish delay ~{(s-base)*1e3:7.1f} µs "
          f"(store seen {d - s:6.3f} ms before kernel end){'  [not seen before end: ' + str(miss) + ']' if miss else ''}")
s, d, miss = run(0, idle=1)
print(f"idle spin (no memory traffic)  : seen {s:7.3f}  done {d:7.3f}  (store seen {d - s:6.3f} ms before kernel end)")

print("-- publisher threadgroups (does a finished threadgroup flush its stores?)")
for xmb in (64, 256, 512):
    s, d, miss = run(xmb, exit_=1)
    print(f"TG0 stores then exits, others stream {xmb:>3} MB: seen {s:7.3f}  done {d:7.3f}  (seen {d - s:6.3f} ms before end)")
for xmb in (64, 256, 512):
    s, d, miss = run(xmb, pub=1)
    print(f"TG1 waits for a GPU phase flag, stores, exits; stream {xmb:>3} MB: seen {s:7.3f}  done {d:7.3f}  (seen {d - s:6.3f} ms before end){'  MISS ' + str(miss) if miss else ''}")
