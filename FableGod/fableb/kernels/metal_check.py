"""metal_check.py (Fable B): CPU-only compile of the draft kernels with Apple's offline Metal compiler (xcrun metal -c).
Nothing runs on the GPU. The harness stands in for what mx.fast.metal_kernel generates: the signature from the input /
output names, T = half in place of bfloat16. It proves syntax and types only. Run: python3 -I metal_check.py"""
import importlib.util, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = "#include <metal_stdlib>\nusing namespace metal;\n#define EPS 1e-6f\n"
ATTR = ["uint3 thread_position_in_grid [[thread_position_in_grid]]",
        "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]",
        "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
        "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def wrap(m, tmpl, inst, src=None):
    args = [f"const device {m.TYPES.get(n, 'T')}* {n} [[buffer({i})]]" for i, n in enumerate(m.IN)]
    args += [f"device {m.TYPES.get(n, 'T')}* {n} [[buffer({len(m.IN) + i})]]" for i, n in enumerate(m.OUT)]
    return (PRE + m.HDR + f"\ntemplate <{tmpl}>\n[[kernel]] void kfn(\n    " + ",\n    ".join(args + ATTR) + ")\n{\n"
            + (src or m.SRC) + "\n}\n" + f"template [[host_name(\"k_inst\")]] [[kernel]] decltype(kfn<{inst}>) kfn<{inst}>;\n")


ok = True
for name, tmpl, insts in (("gdn_w", "typename T, int S_C", ["half, 16", "half, 2"]),
                          ("hc_d", "typename T, int PEND", ["half, 1", "half, 0"]),
                          ("gdn_chunk", "typename T, int S_C, int NSG", ["half, 16, 16", "half, 2, 8", "half, 16, 32"]),
                          ("hc_k2c", "typename T, int LR, int HC, int NP, int NC, int D", ["half, 320, 4, 16, 352, 2560"]),
                          ("gdn_chunk2", "typename T, int S_C, int RB", ["half, 16, 8", "half, 16, 4", "half, 2, 8"]),
                          ("gdn_chunk2:TAB", "typename T, int S_C", ["half, 16"]),
                          ("gdn_step3", "typename T, int S_C", ["half, 16"]),
                          ("eprep", "typename TS, int K_C, int M_C, int R4, int HAS_Q, int HAS_SPLIT, int HAS_NL",
                           ["float, 10, 16, 2, 1, 1, 1", "half, 10, 2, 0, 0, 0, 0"])):
    if not os.path.exists(os.path.join(HERE, name.split(":")[0] + ".py")):
        continue
    tab = name.endswith(":TAB")
    m = load(name.split(":")[0])
    if tab:
        import types
        m = types.SimpleNamespace(HDR=m.HDR, SRC=m.TAB_SRC, IN=m.TAB_IN, OUT=m.TAB_OUT, TYPES={"dots": "float"})
    for inst in insts:
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "k.metal")
            variants = [(v, a, pf) for v in (0, 1, 2) for a in (0, 3) for pf in (0, 7)] + [(0, 0, 1), (0, 0, 2), (0, 0, 4)] if name == "gdn_step3" else [None]
            src = "".join(wrap(m, tmpl, inst, "#define VAR %d\n#define ABL %d\n#define PF %d\n" % va + m.SRC).replace("kfn", "kfn%d%d%d" % va).replace("k_inst", "k_inst%d%d%d" % va).replace(PRE + m.HDR, "" if i else PRE + m.HDR) + "#undef VAR\n#undef ABL\n#undef PF\n"
                          for i, va in enumerate(variants)) if variants[0] else wrap(m, tmpl, inst)
            open(f, "w").write(src)
            p = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", f, "-o", os.path.join(d, "k.air")],
                               capture_output=True, text=True)
            errs = [l for l in p.stderr.splitlines() if "error" in l]
            ok &= p.returncode == 0
            print("%-6s <%s>: %s (%d errors)%s" % (name, inst, "compiles" if p.returncode == 0 else "FAILS", len(errs),
                                                   "".join("\n    " + e[e.find("k.metal"):][:260] for e in errs[:8])))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
