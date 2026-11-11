"""Convert a records3-style directory IN PLACE between "row-major" (bf16 scales) and "native+p11" (NAX6-N's native
order + 11-bit packed s3 / b3 / b), layer by layer (Leap, Tue 2026-11-03 17:1x; the orchestrator's routing after Kamen's
yes to the combined conversion; format: scripts/p11.py, pinned with BigMoE; plan: ~/BigMoE/docs/PLAN_records3_convert.md).
NOT to be run on the real records before Thu 15:00 (after RELAX) and a 1 h heads-up; scratch copies only until then.
Steps:
  --check     (default, read-only) every s3 / b3 / b piece of every layer to convert packs losslessly; free disk; lsof.
  --md5 FILE  write (forward) or verify (--inverse) the md5 of every row-major layer file.
  --yes       convert. Per layer, 64 records at a time: transform, write L.bin.p11_tmp, fsync; then re-read the temp
              file, transform it back and compare with the source bytes (bitwise); atomic rename over L.bin; then the
              layer's index.json entry (atomic rewrite): layout "native+p11" + the new sizes / offsets (forward), or the
              saved row-major entry (inverse).
Extra disk at any time: one layer (~1.43-1.50 GB). Resumable (layers already in the target layout are skipped), reversible
(--inverse; the md5s then match the originals). Refuses under 112 GB free or when lsof shows open files.
--pad (Tue 22:5x, agreed with BigMoE after MoonShots' M-29): forward to "native+p11+pad": the MSB scale pieces zero-padded
to 9216 words and the LSB part ordered l1, l1, l1, b, b, b, so every MSB piece and every l1 plane is 4 KB-aligned for
zero-copy slot reads; recsz / msb_sz / lsb_sz unchanged. --inverse takes either p11 layout back to row-major.
Usage: [BigMoE venv] convert_records_p11.py DIR [--inverse] [--pad] [--layers 0 1 ...] [--md5 FILE] [--yes]"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
import p11

FLOOR, CHUNK = 112e9, 64
ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--inverse", action="store_true", help="native+p11 (+pad) -> row-major")
ap.add_argument("--pad", action="store_true", help='forward to "native+p11+pad" (4 KB-aligned pieces; BigMoE + M-29, Tue 22:5x)')
ap.add_argument("--layers", type=int, nargs="*")
ap.add_argument("--md5", help="md5 list of the row-major files: written before a forward run, verified after --inverse")
ap.add_argument("--yes", action="store_true", help="actually convert (default: read-only check)")
a = ap.parse_args()
D = os.path.abspath(os.path.expanduser(a.dir))
ix_path = os.path.join(D, "index.json")
ix = json.load(open(ix_path))
want = "row-major" if a.inverse else ("native+p11+pad" if a.pad else "native+p11")
layers = a.layers if a.layers is not None else sorted(int(k) for k in ix["layers"])
lay = lambda L: ix["layers"][str(L)].get("layout", "row-major")
bad_src = [L for L in layers if lay(L) not in ("row-major", "native+p11", "native+p11+pad")
           or (not a.inverse and lay(L) not in ("row-major", want))]          # forward: from row-major only
if bad_src:
    sys.exit(f"refused: layers {bad_src} are in another layout ({lay(bad_src[0])}); convert them back first")
todo = [L for L in layers if lay(L) != want]
free = os.statvfs(D).f_bavail * os.statvfs(D).f_frsize
need = max((ix["layers"][str(L)]["E"] * ix["layers"][str(L)]["recsz"] for L in todo), default=0)
print(f"{D}: {len(todo)} of {len(layers)} layers to convert to {want}; free {free / 1e9:.1f} GB, one layer {need / 1e9:.2f} GB")
if free - need < FLOOR:
    sys.exit(f"refused: free disk would fall under {FLOOR / 1e9:.0f} GB")
busy = subprocess.run(["lsof", "+D", D], capture_output=True, text=True).stdout.strip()
if busy:
    sys.exit("refused: files of the directory are open:\n" + busy[:2000])


def rowmajor_entry(r):
    """The row-major entry a native+p11 entry came from (saved at conversion time)."""
    return r["rowmajor"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# read-only check: every scale piece of every forward layer packs at 11 bits
t0 = time.time()
if not a.inverse:
    nbad = 0
    for L in todo:
        r = ix["layers"][str(L)]
        raw = np.memmap(os.path.join(D, r["file"]), dtype=np.uint8, mode="r", shape=(r["E"], r["recsz"]))
        for nm, part, pr, off, sz, dt, per, _ in p11.pieces(r):
            if nm in ("s3", "b3", "b"):
                nbad += int((~p11.packable(np.ascontiguousarray(raw[:, off:off + sz]).view(np.uint16))).sum())
        del raw
    print(f"check: {nbad} scale pieces of {len(todo)} layers do not pack at 11 bits ({time.time() - t0:.0f} s)")
    if nbad:
        sys.exit("refused: some pieces do not pack; nothing written")
if a.md5 and not a.inverse:
    if not a.yes:
        print("(md5 list is written only with --yes)")
    else:
        with open(a.md5, "w") as f:
            for L in layers:
                if lay(L) == "row-major":
                    f.write(f"{md5(os.path.join(D, ix['layers'][str(L)]['file']))}  {ix['layers'][str(L)]['file']}\n")
        print(f"md5 of the row-major files -> {a.md5} ({time.time() - t0:.0f} s)")
if not a.yes:
    sys.exit("read-only check done (pass --yes to convert)")

for L in todo:
    t1 = time.time()
    r = ix["layers"][str(L)]
    src = os.path.join(D, r["file"])
    if a.inverse:
        q, rr = r, rowmajor_entry(r)
        fwd = lambda x: p11.from_p11(x, q, rr)
        back = lambda y: p11.to_p11(y, rr, pad=q.get("layout") == "native+p11+pad")[0]   # the check rebuilds q's own layout
        r_out, rec_in, rec_out = rr, q["recsz"], rr["recsz"]
    else:
        q = p11.layout_p11(r, pad=a.pad)
        q["rowmajor"] = {k: v for k, v in r.items() if k != "rowmajor"}
        fwd = lambda x: p11.to_p11(x, r, pad=a.pad)[0]
        back = lambda y: p11.from_p11(y, q, r)
        r_out, rec_in, rec_out = q, r["recsz"], q["recsz"]
    E = r["E"]
    raw = np.memmap(src, dtype=np.uint8, mode="r", shape=(E, rec_in))
    tmp = src + ".p11_tmp"
    with open(tmp, "wb") as f:
        for e0 in range(0, E, CHUNK):
            f.write(fwd(np.asarray(raw[e0:e0 + CHUNK])).tobytes())
        f.flush()
        os.fsync(f.fileno())
    new = np.memmap(tmp, dtype=np.uint8, mode="r", shape=(E, rec_out))
    ok = all(np.array_equal(back(np.asarray(new[e0:e0 + CHUNK])), np.asarray(raw[e0:e0 + CHUNK])) for e0 in range(0, E, CHUNK))
    del new, raw
    if not ok:
        os.remove(tmp)
        sys.exit(f"layer {L}: the written file does not transform back to the source bytes; stopped (layer {L} untouched)")
    os.replace(tmp, src)
    ix["layers"][str(L)] = r_out
    with open(ix_path + ".tmp", "w") as f:
        json.dump(ix, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(ix_path + ".tmp", ix_path)
    print(f"layer {L}: -> {want} in {time.time() - t1:.1f} s", flush=True)
if a.md5 and a.inverse:
    want_md5 = dict(reversed(ln.split()) for ln in open(a.md5) if ln.strip())
    got = {ix["layers"][str(L)]["file"]: md5(os.path.join(D, ix["layers"][str(L)]["file"])) for L in layers}
    bad = [f for f, h in got.items() if f in want_md5 and want_md5[f] != h]
    print(f"md5 after --inverse: {len(got) - len(bad)} of {len(got)} files match the originals" + (f"; MISMATCH {bad}" if bad else ""))
    if bad:
        sys.exit(1)
print("done")
