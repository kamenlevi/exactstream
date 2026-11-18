"""CERTIFY98's freeze check (docs/PREREG_certify98_2026-11-15.md amendment 10, section 3): the capture's files_sha256.json
must name the v2 manifest and exactly the capture list's 647 windows, and every placed file must have its listed sha256.
CPU only, streaming reads, one process.
  .venv/bin/python tools/certify98_freeze_check.py <files_sha256.json> [<root>]     # exit 0 valid, 1 not
The root defaults to the json's folder (amendment 5's layout: <root>/<set>/<name>.npz)."""
import hashlib, json, os, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V2 = os.path.join(ROOT, "results", "certify98", "fresh_manifest_v2_2026-11-18.json")
V2_SHA = "12e2bf9ad722fbda17176d664a930f4a21664d6677ce95421513fe7356473132"
LIST = os.path.join(ROOT, "data", "certify98", "capture_list_2026-11-18.jsonl")
LIST_SHA = "bb721ff97a338d5eee268d60b3c74ba1a75937540d8b3a8f120100fdf691f2e2"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv):
    fs = argv[0]
    root = argv[1] if len(argv) > 1 else os.path.dirname(os.path.abspath(fs))
    t0, bad = time.time(), []
    if sha256(V2) != V2_SHA:
        bad.append("the v2 manifest moved")
    if sha256(LIST) != LIST_SHA:
        bad.append("the capture list moved")
    want = {json.loads(l)["key"] + ".npz" for l in open(LIST)}
    F = json.load(open(fs))
    if F.get("manifest_sha256") != V2_SHA:
        bad.append(f"manifest_sha256 {F.get('manifest_sha256')!r} is not the v2 manifest's")
    got = dict(F.get("files") or {})
    if set(got) != want:
        bad.append(f"{len(got)} files listed; {len(want - set(got))} of the list's {len(want)} missing, "
                   f"{len(set(got) - want)} extra (e.g. {sorted(want - set(got))[:2]} {sorted(set(got) - want)[:2]})")
    else:
        nbytes = 0
        for name in sorted(got):
            p = os.path.join(root, name)
            if not os.path.isfile(p):
                bad.append(f"{name}: not placed"); continue
            nbytes += os.path.getsize(p)
            if sha256(p) != got[name]:
                bad.append(f"{name}: sha256 differs")
        print(f"  {len(got)} files re-hashed, {nbytes / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    print(f"FREEZE CHECK: {'PASS' if not bad else 'FAIL'} ({fs}; sha256 {sha256(fs)})"
          + "".join(f"\n  {b}" for b in bad[:10]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
