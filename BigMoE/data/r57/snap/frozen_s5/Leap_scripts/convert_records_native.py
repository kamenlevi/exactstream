"""Convert a records3-style directory to NAX6-N's native layout IN PLACE, layer by layer (Leap, Tue 2026-11-03 15:5x;
TABLE_350 'DISK-PLAN'; prepared for Kamen's decision, NOT run on the real records). Per layer:
  1. read L{n}.bin, permute every piece of every expert by tierglu_nax6n.native_map (or its inverse with --inverse);
  2. write L{n}.bin.native_tmp next to it, fsync, re-read it and compare with the in-RAM result (bitwise);
  3. atomic rename over L{n}.bin, then record the layer's layout in index.json (atomic write).
Extra disk at any time: one layer (~1.5 GB). Resumable (layers already marked are skipped), reversible (--inverse).
Refuses to start if free disk would fall under 112 GB, or if any process has a file of the directory open (lsof).
Usage: [BigMoE venv] convert_records_native.py DIR [--inverse] [--layers 0 1 ...] [--yes]   (without --yes: dry run)"""
import argparse, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
from tierglu import PROJ
import tierglu_nax6n as NN

SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
FLOOR = 112e9
ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--inverse", action="store_true", help="native -> row-major")
ap.add_argument("--layers", type=int, nargs="*")
ap.add_argument("--yes", action="store_true", help="actually convert (default: dry run)")
a = ap.parse_args()
D = os.path.abspath(os.path.expanduser(a.dir))
ix_path = os.path.join(D, "index.json")
ix = json.load(open(ix_path))
want = "row-major" if a.inverse else "native"
layers = a.layers if a.layers is not None else sorted(int(k) for k in ix["layers"])
todo = [L for L in layers if ix["layers"][str(L)].get("layout", "row-major") != want]
free = os.statvfs(D).f_bavail * os.statvfs(D).f_frsize
rec = ix["layers"][str(todo[0])] if todo else None
need = rec["E"] * rec["recsz"] if rec else 0
print(f"{D}: {len(todo)} of {len(layers)} layers to convert to {want}; free {free / 1e9:.1f} GB, one layer {need / 1e9:.2f} GB")
if free - need < FLOOR:
    sys.exit(f"refused: free disk would fall under {FLOOR / 1e9:.0f} GB")
busy = subprocess.run(["lsof", "+D", D], capture_output=True, text=True).stdout.strip()
if busy:
    sys.exit("refused: files of the directory are open:\n" + busy[:2000])
if not a.yes:
    sys.exit("dry run (pass --yes to convert)")

def permute(raw, r, inverse):
    out = raw.copy()
    for p, pr in enumerate(PROJ):
        o, i = SH[pr]
        kg = i // 64
        pieces = [(r["msb_piece_off"][3 * p], np.uint32, 6), (r["msb_piece_off"][3 * p + 1], np.uint16, 1),
                  (r["msb_piece_off"][3 * p + 2], np.uint16, 1),
                  (r["lsb_piece_off"][2 * p], np.uint32, 2), (r["lsb_piece_off"][2 * p + 1], np.uint16, 1)]
        for off, dt, per in pieces:
            nbytes = o * kg * per * np.dtype(dt).itemsize
            x = np.ascontiguousarray(raw[:, off:off + nbytes]).view(dt).reshape(raw.shape[0], -1)
            m = NN.native_map(o, kg, per)
            y = np.empty_like(x)
            if inverse:
                y[:, m] = x                                      # rowmajor[map[i]] = native[i]
            else:
                y = x[:, m]
            out[:, off:off + nbytes] = np.ascontiguousarray(y).view(np.uint8).reshape(raw.shape[0], nbytes)
    return out

for L in todo:
    t0 = time.time()
    r = ix["layers"][str(L)]
    path = os.path.join(D, r["file"])
    raw = np.fromfile(path, dtype=np.uint8).reshape(r["E"], r["recsz"])
    new = permute(raw, r, a.inverse)
    tmp = path + ".native_tmp"
    with open(tmp, "wb") as f:
        new.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    back = np.fromfile(tmp, dtype=np.uint8).reshape(r["E"], r["recsz"])
    if not np.array_equal(back, new):
        os.remove(tmp)
        sys.exit(f"layer {L}: the written file differs from the computed one; stopped (original untouched)")
    os.replace(tmp, path)
    ix["layers"][str(L)]["layout"] = want
    with open(ix_path + ".tmp", "w") as f:
        json.dump(ix, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(ix_path + ".tmp", ix_path)
    print(f"layer {L}: -> {want} in {time.time() - t0:.1f} s", flush=True)
    del raw, new, back
print("done")
