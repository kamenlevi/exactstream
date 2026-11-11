"""metal_syntax_check.py (Fable B, build_1105): CPU-ONLY compile of the GDN body kernel sources with Apple's offline Metal
compiler (xcrun metal -c: no GPU, nothing runs). It checks that the lazy source DERIVED by gdnbody_lazy.patch's _lz_src()
is well-formed Metal, next to the unpatched source as the control (if the control does not compile, the harness is wrong,
not the kernel). It proves syntax and types only: bits and speed are for gdnb_lazy_check.py on the GPU.
The harness stands in for what MLX generates around a custom kernel: the function signature from the input / output names,
T = half in place of bfloat16 (same arithmetic shape), and a log1p (MLX's utils.h has its own).
Run: python -I metal_syntax_check.py   (needs the Metal toolchain: xcrun -f metal)"""
import ast, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LEAP = os.path.expanduser(os.environ.get("FTA_LEAP", "~/Leap"))


def literals(path, names):
    """module-level string assignments of a Python file, without importing it (no MLX)"""
    out = {}
    tree = ast.parse(open(path).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            ns = dict(out)
            out[n.targets[0].id] = eval(compile(ast.Expression(n.value), path, "eval"), ns)
    return out, tree


hdr = literals(os.path.join(LEAP, "third_party", "ms_gdn", "tierglu.py"), {"_HDR"})[0]["_HDR"]
src_path = os.path.join(HERE, "patched", "gdnbody.py")
text = open(src_path).read()
tree = ast.parse(text)
ns = {"_HDR": hdr}
for n in tree.body:                                   # _HDR_G, _SRC, PMAX and _lz_src, nothing else (no mlx import)
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("_HDR_G", "_SRC", "PMAX"):
        exec(compile(ast.Module(body=[n], type_ignores=[]), src_path, "exec"), ns)
    if isinstance(n, ast.FunctionDef) and n.name == "_lz_src":
        exec(compile(ast.Module(body=[n], type_ignores=[]), src_path, "exec"), ns)
SRC, LZ = ns["_SRC"], ns["_lz_src"]()
assert SRC == literals(os.path.join(LEAP, "scripts", "gdnbody.py"), {"_SRC"})[0]["_SRC"], "the patch changed _SRC"
open(os.path.join(HERE, "gdn_body_lazy.metal.txt"), "w").write(LZ)

IN = ["qkv", "state", "cw", "scale", "a", "b", "dtb", "alog", "dstate", "z", "w", "epsv"]
IN_LZ = IN + ["ppack", "pn"]
OUT = ["y", "state_out", "dstate_out", "q_out", "k_out", "v_out"]
OUT_LZ = OUT + ["pack_out"]
TY = {"dstate": "float", "epsv": "float", "dstate_out": "float", "pn": "uint"}
PRE = """#include <metal_stdlib>
using namespace metal;
template <typename T> inline T log1p(T x) { return metal::log(static_cast<T>(1) + x); }   // stand-in for MLX utils.h
"""


def wrap(src, ins, g3, lazy):
    args = [f"const device {TY.get(n, 'T')}* {n} [[buffer({i})]]" for i, n in enumerate(ins)]
    args += [f"device {TY.get(n, 'T')}* {n} [[buffer({len(ins) + i})]]" for i, n in enumerate(OUT_LZ if lazy else OUT)]
    args += ["uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
             "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]"]
    tp = "typename T, int S_C, int NSG" + (", int PMAX" if lazy else "")
    inst = "half, 16, 8" + (", 16" if lazy else "")
    return (PRE + ns["_HDR_G"] + f"\n#define GATES_IN 1\n#define G3_C {int(g3)}\ntemplate <{tp}>\n[[kernel]] void gdnb(\n    "
            + ",\n    ".join(args) + ")\n{\n" + src + "\n}\n"
            + f"template [[host_name(\"gdnb_inst\")]] [[kernel]] decltype(gdnb<{inst}>) gdnb<{inst}>;\n")


ok = True
for name, src, ins, lazy in (("today's body (control)", SRC, IN, False), ("lazy body", LZ, IN_LZ, True)):
    for g3 in (False, True):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "k.metal")
            open(f, "w").write(wrap(src, ins, g3, lazy))
            p = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-c", f, "-o", os.path.join(d, "k.air")],
                               capture_output=True, text=True)
            errs = [l for l in p.stderr.splitlines() if "error" in l]
            warns = [l for l in p.stderr.splitlines() if "warning" in l]
            ok &= p.returncode == 0
            print("%-22s G3 %d: %s (%d errors, %d warnings)%s" % (
                name, g3, "compiles" if p.returncode == 0 else "FAILS", len(errs), len(warns),
                "".join("\n    " + e[e.find("k.metal"):][:230] for e in errs[:6])))
print("lazy source written to gdn_body_lazy.metal.txt (%d lines; today's source %d lines)" % (LZ.count("\n"), SRC.count("\n")))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
