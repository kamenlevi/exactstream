"""S0 locator read-out (BigMoE Fri 23:4x; docs/ASYNC_POST.md section 2): where the Python thread spends its time, from a
cProfile of one 2-prompt r57 run. usage: .venv/bin/python -I scripts/s0_prof_read.py data/s0/prof.out [N]
Prints the top N functions by self time (the main thread only: cProfile's view) and, for the C-level calls that wait
on the GPU or the SSD (mx eval / array conversion / item / ext calls), who calls them, so the blocking call site is named."""
import io
import pstats
import sys

p = pstats.Stats(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
p.sort_stats("tottime")
buf = io.StringIO(); p.stream = buf
p.print_stats(n)
print("\n".join(ln for ln in buf.getvalue().splitlines() if ln.strip())[:12000])
WAIT = ("eval", "__array__", "item", "tolist", "ensure", "spin", "post_ids", "asarray", "array", "synchronize", "async_eval")
tops = [k for k in p.stats if any(w in k[2] for w in WAIT)]
tops.sort(key=lambda k: -p.stats[k][2])
print("\n== blocking candidates (self time) and their callers ==")
for k in tops[:8]:
    cc, nc, tt, ct, callers = p.stats[k]
    print(f"{tt:9.3f} s self, {nc:7d} calls  {k[2]}  ({k[0].split('/')[-1]}:{k[1]})")
    for c, v in sorted(callers.items(), key=lambda kv: -kv[1][2])[:4]:
        print(f"      <- {v[2]:8.3f} s  {c[2]} ({c[0].split('/')[-1]}:{c[1]})")
