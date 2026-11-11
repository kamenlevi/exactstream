"""D1i (CPU only): tier.planes() == records3's stored MSB-3 / LSB-1 parts, byte for byte, for a few experts of a few
layers (the 4-bit source = the ext's read_rebuilt, i.e. what the prefill scratch holds)."""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/TrainLab/d1i/v3f"))
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
from bigmoe_v3f.gated import ext
from bigmoe_v3f import tier as T

MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(os.path.join(MD, "records3", "index.json")))
SH = [((640, 320), mx.uint32), ((640, 40), mx.bfloat16), ((640, 40), mx.bfloat16)] * 2 + \
     [((2560, 80), mx.uint32), ((2560, 10), mx.bfloat16), ((2560, 10), mx.bfloat16)]
pers4 = [int(np.prod(s)) * (4 if d == mx.uint32 else 2) for s, d in SH]
raw = lambda a: np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes()
bad = 0
for L, experts in ((0, [0, 7, 300, 511]), (23, [1, 128, 509]), (47, [2, 64, 400, 510])):
    r = idx["layers"][str(L)]
    path = os.path.join(MD, "records3", r["file"])
    scr = [mx.zeros((len(experts), *s), d) for s, d in SH]
    mx.eval(*scr)
    ext.read_rebuilt(path, r["recsz"], r["msb_sz"], pers4, experts, list(range(len(experts))), scr)
    msb, lsb = T.planes_expert(scr, list(range(len(experts))))
    mx.eval(*msb, *lsb)
    with open(path, "rb") as f:
        for k, e in enumerate(experts):
            f.seek(e * r["recsz"]); ms = f.read(r["msb_sz"])
            f.seek(e * r["recsz"] + r["msb_sz"]); ls = f.read(r["recsz"] - r["msb_sz"])
            mine_m = b"".join(raw(a[k]) for a in msb); mine_l = b"".join(raw(a[k]) for a in lsb)
            ok_m, ok_l = ms[:len(mine_m)] == mine_m, ls[:len(mine_l)] == mine_l
            if not (ok_m and ok_l):
                bad += 1
                for name, mine, ref in (("msb", msb, ms), ("lsb", lsb, ls)):
                    off = 0
                    for i, a in enumerate(mine):
                        b = raw(a[k]); n = len(b)
                        if b != ref[off:off + n]:
                            x = np.frombuffer(b, np.uint8); y = np.frombuffer(ref[off:off + n], np.uint8)
                            print(f"  L{L} e{e} {name} piece {i}: {int((x != y).sum())} of {n} bytes differ")
                        off += n
            print(f"L{L} expert {e}: MSB {'==' if ok_m else '!='} records3 ({len(mine_m)} B), LSB {'==' if ok_l else '!='} ({len(mine_l)} B)")
print("ALL EQUAL" if bad == 0 else f"{bad} experts differ")
