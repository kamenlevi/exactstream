"""NATIVE-3C sample (Leap, Tue 2026-11-03 15:4x; TABLE_350 'NATIVE-3C'): copies of records3's layers 3, 20 and 44 in NAX6-N's
fragment-native layout, into a scratch dir (~/Leap/data/native_sample/). The record layout and piece offsets stay; each
piece (w3 / s3 / b3 per projection, l1 / b of the LSB part) is permuted per expert by tierglu_nax6n.native_map; padding
bytes are copied. The real records are only read.
Usage: [BigMoE venv] native_sample.py [layers ...]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_scripts"))
from tierglu import PROJ
import tierglu_nax6n as NN

RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
OUTD = os.path.expanduser("~/Leap/data/native_sample")
os.makedirs(OUTD, exist_ok=True)
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
idx = json.load(open(f"{RD}/index.json"))
layers = [int(a) for a in sys.argv[1:]] or [3, 20, 44]
out_idx = {"layout": "native (tierglu_nax6n.native_map per piece, per expert)", "source": RD, "layers": {}}
for L in layers:
    r = idx["layers"][str(L)]
    E, REC = r["E"], r["recsz"]
    raw = np.fromfile(f"{RD}/{r['file']}", dtype=np.uint8).reshape(E, REC)
    nat = raw.copy()                                             # padding and anything unpermuted stays as is
    for p, pr in enumerate(PROJ):
        o, i = SH[pr]
        kg = i // 64
        pieces = [(r["msb_piece_off"][3 * p], np.uint32, 6), (r["msb_piece_off"][3 * p + 1], np.uint16, 1),
                  (r["msb_piece_off"][3 * p + 2], np.uint16, 1),
                  (r["lsb_piece_off"][2 * p], np.uint32, 2), (r["lsb_piece_off"][2 * p + 1], np.uint16, 1)]
        for off, dt, per in pieces:
            nbytes = o * kg * per * np.dtype(dt).itemsize
            a = np.ascontiguousarray(raw[:, off:off + nbytes]).view(dt).reshape(E, -1)
            m = NN.native_map(o, kg, per)
            nat[:, off:off + nbytes] = np.ascontiguousarray(a[:, m]).view(np.uint8).reshape(E, nbytes)
    nat.tofile(f"{OUTD}/{r['file']}")
    out_idx["layers"][str(L)] = dict(r)
    print(f"layer {L}: {E} records x {REC} B -> {OUTD}/{r['file']} ({nat.nbytes / 1e9:.2f} GB)", flush=True)
    del raw, nat
json.dump(out_idx, open(f"{OUTD}/index.json", "w"), indent=1)
