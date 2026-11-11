"""Offline compile check of the sol_layer Metal kernels (no GPU work): builds one layer's graph lazily for M = 1 and 4 with
MLX's verbose source dump, then compiles every generated kernel with xcrun metal against MLX's shipped headers.
usage: compile_check.py [module with compile_graph(), default sol_layer]   (CPU only; runs during others' quiet locks via taskpolicy -b)"""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 2 and sys.argv[1] == "--dump":
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "..", "..", "engine"))
    sys.path.insert(0, os.path.join(HERE, "..", "planes_engine"))
    import mlx.core as mx
    mx.eval = lambda *a, **k: None                       # never run anything on the GPU
    mk = mx.fast.metal_kernel

    def _mk_verbose(**kw):
        k = mk(**kw)
        return lambda **a: k(**a, verbose=True)
    mx.fast.metal_kernel = _mk_verbose
    import importlib
    importlib.import_module(sys.argv[2]).compile_graph()
    sys.exit(0)

import importlib.util
inc = os.path.join(list(importlib.util.find_spec("mlx").submodule_search_locations)[0], "include")
mod = sys.argv[1] if len(sys.argv) > 1 else "sol_layer"
out = subprocess.run([sys.executable, __file__, "--dump", mod], capture_output=True, text=True)
if out.returncode:
    print(out.stdout[-3000:], out.stderr[-3000:]); sys.exit(1)
blocks = re.findall(r"Generated source code for `([^`]*)`:\n```\n(.*?)\n```", out.stdout, re.S)
print(f"{len(blocks)} kernel sources", flush=True)
tmp = tempfile.mkdtemp()
ok, seen = True, set()
for i, (name, src) in enumerate(blocks):
    if src in seen:
        continue
    seen.add(src)
    f = os.path.join(tmp, f"{name}_{i}.metal")
    open(f, "w").write("#include <mlx/backend/metal/kernels/utils.h>\n" + src)
    r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2", "-I", inc, "-c", f, "-o", f + ".air"],
                       capture_output=True, text=True)
    errs = [l for l in r.stderr.splitlines() if "error" in l]
    ok &= r.returncode == 0
    print(f"{name}: {'ok' if r.returncode == 0 else 'FAIL'}", flush=True)
    for l in errs[:8]:
        print("   ", l.replace(tmp + "/", ""))
print("ALL OK" if ok else "ERRORS", flush=True)
sys.exit(0 if ok else 1)
