"""v3_import_check.py (Leap, Tue 17 Nov 2026 06:4x): CPU-only check of e15_loop.py's E15_V3TREE block, which the head test
cannot reach (it stops at EM.load, line ~354; the block sits before 'import ref2', line ~366). Builds the loop's sys.path as
at that point, runs the block's own lines (read from e15_loop.py), then v3's switch mechanics through ref2.apply():
  - ref2 / bofi / bof are the v3 copies; CFG bof / hc_b follow REF2_BOF / REF2_HC_B;
  - apply() with bof on: bofi wraps ref2's _qmm as the OUTERMOST mx.quantized_matmul; TABLE32 (2560, 6144) = (4, 1); ON;
  - bof off + apply(): bofi.ON False and _BOF_OWN False; on again: ON True and no double wrap (install idempotent).
No model, no GPU kernels. Usage: [venv] E15_V3TREE=... REF2_BOF=1 REF2_HC_B=1 python -I -B v3_import_check.py ; rc 0 = PASS."""
import os
import sys
E = os.path.expanduser
LOOP = E("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts/week98/e15/e15_loop.py")
sys.path.insert(0, E("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts/week98/e15")); sys.path.insert(1, E("~/BigMoE/scripts"))   # e15_loop.py lines 29-30
sys.path.insert(0, E("~/BigMoE")); sys.path.insert(0, E("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts")); sys.path.insert(0, E("~/EvalSuite"))
sys.path.insert(1, E("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_third_party"))                                                      # lines 105-108
src = open(LOOP).read().splitlines()
i = next(k for k, l in enumerate(src) if l.startswith('if os.environ.get("E15_V3TREE"):'))
j = next(k for k in range(i + 1, len(src)) if src[k] == "import ref2")
block = "\n".join(src[i:j])
exec(compile(block, LOOP + f":{i + 1}", "exec"))
import ref2
import bofi
import mlx.core as mx
v3 = os.environ["E15_V3TREE"]
ok = []
def chk(name, cond):
    ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, flush=True)
chk("modules are the v3 copies", all(m.__file__.startswith(v3) for m in (ref2, bofi, sys.modules["bof"])))
chk("CFG bof / hc_b on from the env", ref2.CFG.get("bof") is True and ref2.CFG.get("hc_b") is True)
ref2.apply()
chk("apply (on): bofi is the outermost quantized_matmul over ref2's _qmm",
    mx.quantized_matmul is bofi._qmmb and bofi._S.get("qmm") is ref2._qmm)
chk("apply (on): ON, TABLE32, _BOF_OWN", bofi.ON[0] and bofi.TABLE32.get((2560, 6144)) == (4, 1) and ref2._BOF_OWN[0])
ref2.CFG["bof"] = False
ref2.apply()
chk("apply (off): ON False, _BOF_OWN False, wrapper kept (inert)", not bofi.ON[0] and not ref2._BOF_OWN[0] and mx.quantized_matmul is bofi._qmmb)
ref2.CFG["bof"] = True
ref2.apply()
chk("apply (on again): ON True, no double wrap", bofi.ON[0] and bofi._S.get("qmm") is ref2._qmm and mx.quantized_matmul is bofi._qmmb)
ref2.remove()
chk("remove(): ref2 off (bofi's guard then falls through)", not ref2._ON[0])
print("V3 IMPORT CHECK:", "PASS" if all(ok) else "FAIL", flush=True)
sys.exit(0 if all(ok) else 1)
