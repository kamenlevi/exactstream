"""Megakernel feasibility on M5: cost of a grid-wide barrier inside one dispatch vs a dependent kernel hop.
(a) hop: N tiny dependent kernels in one MLX graph (each reads the previous output), time / N.
(b) barrier: one dispatch of G threadgroups doing N phases; between phases every threadgroup bumps a device atomic
    counter and spins until it reaches phase*G. A bounded spin sets a fail flag instead of hanging (no co-residency
    guarantee on Metal), so too many threadgroups shows up as FAIL, not a GPU hang.
Prints µs per hop vs µs per barrier for several G. Tiny GPU job (< 1 s per config); needs a quiet lock for timing."""
import time
import mlx.core as mx

N = 400

HOP = """
    uint i = thread_position_in_grid.x;
    y[i] = x[i] + 1.0f;
"""
hop = mx.fast.metal_kernel(name="hop", input_names=["x"], output_names=["y"], source=HOP)

BAR = """
    uint tg = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    threadgroup uint stop;
    if (t == 0) stop = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 1; p <= NPH; ++p) {
        // phase work: a little device traffic per threadgroup
        atomic_fetch_add_explicit(&work[tg * 256 + t], 1.0f, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_device);
        if (t == 0) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
            atomic_fetch_add_explicit(&cnt[0], 1u, memory_order_relaxed);
            uint target = p * G;
            uint spins = 0;
            while (atomic_load_explicit(&cnt[0], memory_order_relaxed) < target) {
                if (++spins > 4000000u) { atomic_store_explicit(&fail[0], 1u, memory_order_relaxed); stop = 1; break; }
            }
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stop) return;
    }
"""
bar = mx.fast.metal_kernel(name="gridbar", input_names=["work_in"], output_names=["cnt", "fail", "work"],
                           source=BAR, atomic_outputs=True)


def t_hops():
    x = mx.zeros((4096,), mx.float32)
    def run():
        y = x
        for _ in range(N):
            y = hop(inputs=[y], grid=(4096, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(4096,)], output_dtypes=[mx.float32])[0]
        mx.eval(y)
    run()
    ts = []
    for _ in range(7):
        a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
    return sorted(ts)[3] / N * 1e6


def t_bar(G):
    w = mx.zeros((G * 256,), mx.float32)
    def run():
        cnt, fail, work = bar(inputs=[w], template=[("G", G), ("NPH", N)], grid=(G * 256, 1, 1),
                              threadgroup=(256, 1, 1), output_shapes=[(1,), (1,), (G * 256,)],
                              output_dtypes=[mx.uint32, mx.uint32, mx.float32], init_value=0)
        mx.eval(cnt, fail, work)
        return cnt.item(), fail.item()
    c, f = run()
    if f:
        return None, c
    ts = []
    for _ in range(7):
        a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
    return sorted(ts)[3] / N * 1e6, c


print(f"dependent kernel hop (MLX graph, {N} chained tiny kernels): {t_hops():.2f} µs/hop")
for G in (1, 8, 16, 20, 32, 40, 64, 128):
    us, c = t_bar(G)
    if us is None:
        print(f"grid barrier G={G:>3}: FAIL (counter reached {c} of {G*N}: threadgroups not co-resident)")
        break
    print(f"grid barrier G={G:>3} threadgroups x 256: {us:.2f} µs/barrier (counter {c} = {G*N} expected)")
