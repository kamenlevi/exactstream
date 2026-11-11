"""Layout router check (Leap, Tue 2026-11-03 17:3x; TABLE_350 'P11'): ref2's per-layer routing by BigMoE's layout
(t["layout"] / bigmoe.tier._LAYOUT[0]: 0 row-major, 1 native, 2 native + p11), CPU only, no kernels run.
Usage: [BigMoE venv] layout_router_check.py"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/BigMoE')); sys.path.insert(0, os.path.expanduser('/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts'))
import ref2; ref2.apply()
import bigmoe.tier as BT
TG = BT._tier_glu_mod()
res = []
BT._LAYOUT = [0]; res.append(("layout 0 -> BigMoE 3c", ref2._tier_glu_sel() is TG.tier_glu))
BT._LAYOUT = [1]; res.append(("layout 1 -> 3c-N", ref2._tier_glu_sel() is ref2._tier_glu_3cn))
BT._LAYOUT = [2]; res.append(("layout 2 -> 3c-N", ref2._tier_glu_sel() is ref2._tier_glu_3cn))
res.append(("t layout beats the global", ref2._native({"layout": 0}) is False and ref2._native({"layout": 2}) is True))
for cfg, name in (({"native_3c": True}, "native_3c on layout 0 raises"), ({"tier_nax4": True}, "layout 2 + tier_nax4 raises")):
    BT._LAYOUT = [0] if "native_3c" in cfg else [2]
    ref2.CFG.update(cfg)
    try:
        ref2._tier_glu_sel(); res.append((name, False))
    except RuntimeError:
        res.append((name, True))
    ref2.CFG.update({k: False for k in cfg})
# REF2 removed (r57_screen toggles it around drafts): a converted layer still takes 3c-N, a row-major one BigMoE's 3c
ref2.remove()
BT._LAYOUT = [2]; res.append(("REF2 off, layout 2 -> 3c-N", ref2._tier_glu_sel() is ref2._tier_glu_3cn))
BT._LAYOUT = [1]; res.append(("REF2 off, layout 1 -> 3c-N", ref2._tier_glu_sel() is ref2._tier_glu_3cn))
BT._LAYOUT = [0]; res.append(("REF2 off, layout 0 -> BigMoE 3c", ref2._tier_glu_sel() is TG.tier_glu))
seen = []
orig_3cn, orig_ooo = ref2._ooo_3cn, ref2._ORIG["ooo_call"]
ref2._ooo_3cn = lambda *a: seen.append("3cn")
ref2._ORIG["ooo_call"] = lambda *a: seen.append("orig")
for lay in (2, 0):
    ref2._ooo_sel(None, None, None, {"layout": lay}, None, None, None)
ref2._ooo_3cn, ref2._ORIG["ooo_call"] = orig_3cn, orig_ooo
res.append(("REF2 off, OOO: layout 2 -> 3c-N, layout 0 -> BigMoE", seen == ["3cn", "orig"]))
ref2.apply()
del BT._LAYOUT
res.append(("no router -> CFG native_3c (off) -> BigMoE 3c", ref2._tier_glu_sel() is TG.tier_glu and ref2._layout() is None))
for k, v in res:
    print(f"{k}: {v}")
print("ROUTER CHECK:", "PASS" if all(v for _, v in res) else "FAIL")
