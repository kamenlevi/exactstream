"""hc_k2c.py (Fable B, DENSE_REBUILD v2): DRAFT of the HC8-1P gate kernel with its serial chains cut. NOT RUN ON A GPU.
Leap's k2b (hc8r.py _K2B) moves 0.45 MB but costs 9.4-10.5 us [K part B] against ~1.6 us for an empty wide kernel.
Its bytes are not the cost. Two serial chains are:
  - xsum: one thread adds 64 threadgroup values in a row (64 dependent adds after the barrier);
  - c: one thread adds NP = 16 products in a row.
k2c keeps k2b's inputs, outputs, grid and threadgroup (drop-in for hc8r's kernel 7) and changes only the order of sums:
  - c = sum over the 4 streams of rv[s] * (sum of that stream's NP / 4 partials): 4 short independent chains, 4 multiplies;
  - xsum: the 64 values of a quant group are added by one simdgroup, 2 per lane, then simd_sum (needs LR + HC >= 160
    threads per threadgroup: 352 here).
Bits: the sums are re-ordered, so mix1 / inject / xsum can differ from k2b in the last bit. HC8 is a FAST read already.
Expected [G]: ~4-5 us per call instead of ~10, i.e. -0.5 ms per check over 96 reads."""
HDR = ""
SRC = r"""
    constexpr int N = LR + HC, PPS = NP / HC, KG3 = LR / 64;
    const uint r = threadgroup_position_in_grid.x;
    const uint n = thread_index_in_threadgroup;
    threadgroup T ms[LR];
    if (n < N) {
        float rv[HC], cs[HC];
        for (int s = 0; s < HC; s++) {
            float ss = 0.0f, c = 0.0f;
            for (int p = s * PPS; p < (s + 1) * PPS; p++) {
                ss += ssq[p * 16 + r];
                c += part_in[((size_t)p * 16 + r) * NC + n];
            }
            rv[s] = metal::precise::rsqrt(ss / float(D) + EPS);
            cs[s] = c;
        }
        float c = (cs[0] * rv[0] + cs[1] * rv[1]) + (cs[2] * rv[2] + cs[3] * rv[3]);
        if (n < LR) {
            const float z = c / float(HC);
            const T m = static_cast<T>(z / (1.0f + metal::precise::exp(-z)));
            mix1[r * LR + n] = m;
            ms[n] = m;
        } else {
            inject[r * HC + (n - LR)] = static_cast<T>(2.0f / (1.0f + metal::precise::exp(-c / float(HC))));
        }
        if (n == 0) for (int s = 0; s < HC; s++) rinv[r * HC + s] = rv[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n < 32u * KG3) {                                       // simdgroup g adds quant group g: 2 values per lane
        const uint g = n / 32, l = n % 32;
        float xs = float(ms[g * 64 + 2 * l]) + float(ms[g * 64 + 2 * l + 1]);
        xs = simd_sum(xs);
        if (l == 0) xsum[r * KG3 + g] = xs;
    }
"""
IN = ["part_in", "ssq"]
OUT = ["mix1", "inject", "rinv", "xsum"]
TYPES = {"part_in": "float", "ssq": "float", "rinv": "float", "xsum": "float"}
