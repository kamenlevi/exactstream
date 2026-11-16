"""CERTIFY98's dump-manifest check (PREREG_certify98 amendments 7 / 7a): every input's sha256 and size equal its entry in
BigMoE's dumps_sha256.json, before any scoring. CPU only, streaming reads.
  .venv/bin/python tools/certify98_dumpcheck.py <dumps_sha256.json> <file> [<file> ...]
      [--gated <files_sha256.json>] [--pins <pins.json>] [--fresh <fresh_manifest.json>]
- An entry is the file's basename -> {"sha256", "bytes", ...} (BigMoE scripts/dd_manifest.py, 37744f27).
- Exit 0 only if every named file is in the manifest and matches it; otherwise 1, and nothing may be scored.
- Prints the manifest's own sha256 for the RESULT to quote; EvalSuite commits a copy in results/certify98/ before scoring.
- --gated (the fresh dumps) also needs:
  - amendment 7's run fields under "_run", each non-empty: files_sha256_json (== the sha256 of the named
    files_sha256.json), checkpoint_path, checkpoint_sha256, bigmoe_commit, dd_topk_sha256, dd_tree_sha256, args and
    modules ({path: sha256} of every local module the dumpers import, ALWAYS: ~/TrainLab/mtp125.py sits outside BigMoE's
    git), plus bigmoe_dirty as a bool;
  - on each named file, "script" (dd_topk or dd_tree) and "args", and those args among _run.args of that script.
- --pins (the proof-run amendment's pinned values, JSON {field: value}): each must equal _run's; "modules" must be the
  same {path: sha256} map exactly.
- --fresh: the files_sha256.json's manifest_sha256 must be the sha256 of the frozen fresh-set manifest."""
import hashlib, json, os, sys

RUN = ("files_sha256_json", "checkpoint_path", "checkpoint_sha256", "bigmoe_commit", "dd_topk_sha256", "dd_tree_sha256",
       "args", "modules")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def run_args_has(R, script, args):
    """the file's args appear in _run.args: dd_topk's dict, or one entry of dd_tree's list"""
    a = (R.get("args") or {}).get(script)
    return a == args if isinstance(a, dict) else isinstance(a, list) and args in a


def check(manifest, files, gated=None, pins=None, fresh=None):
    """(lines, ok): one line per file, then the run fields, the pins and the fresh manifest if asked"""
    M = json.load(open(manifest))
    out, ok = [f"manifest {manifest}: sha256 {sha256(manifest)}"], True
    R = M.get("_run") if isinstance(M.get("_run"), dict) else {}
    for p in files:
        e = M.get(os.path.basename(p))
        if not os.path.exists(p):
            out.append(f"  MISSING          {p}"); ok = False
        elif not isinstance(e, dict) or "sha256" not in e:
            out.append(f"  NOT IN MANIFEST  {p}"); ok = False
        else:
            n, s = os.path.getsize(p), sha256(p)
            good = s == e["sha256"] and ("bytes" not in e or n == int(e["bytes"]))
            out.append(f"  {'OK' if good else 'MISMATCH':<16} {p}: {n} bytes, sha256 {s}"
                       + ("" if good else f" (manifest: {e.get('bytes')} bytes, {e['sha256']})"))
            ok &= good
            if gated is not None:
                sc, a = e.get("script"), e.get("args")
                g = sc in ("dd_topk", "dd_tree") and isinstance(a, dict) and run_args_has(R, sc, a)
                out.append(f"    script {sc}, args {json.dumps(a, sort_keys=True)}: "
                           + ("in _run.args" if g else "MISSING or not in _run.args"))
                ok &= g
    if gated is not None:
        miss = [k for k in RUN if not R.get(k)] + ([] if isinstance(R.get("bigmoe_dirty"), bool) else ["bigmoe_dirty"])
        out.append(f"  run fields: {'all present' if not miss else 'MISSING ' + ', '.join(miss)}"
                   + (f" (bigmoe_dirty {R['bigmoe_dirty']}; {len(R.get('modules') or {})} modules)" if not miss else ""))
        fs = sha256(gated) if os.path.exists(gated) else None
        same = fs is not None and R.get("files_sha256_json") == fs
        out.append(f"  files_sha256_json {'==' if same else '!='} sha256 of {gated} ({fs})")
        ok &= not miss and same
        if fresh is not None:
            F = json.load(open(gated)) if fs else {}
            fr = sha256(fresh) if os.path.exists(fresh) else None
            g = fr is not None and F.get("manifest_sha256") == fr
            out.append(f"  files_sha256.json's manifest_sha256 {'==' if g else '!='} sha256 of {fresh} ({fr})")
            ok &= g
    if pins is not None:
        P = json.load(open(pins))
        for k in sorted(P):
            g = R.get(k) == P[k]
            out.append(f"  pin {k}: {'==' if g else '!= (manifest ' + json.dumps(R.get(k), sort_keys=True)[:160] + ')'}")
            ok &= g
    out.append(f"DUMP MANIFEST: {'PASS' if ok else 'FAIL: score nothing'} ({len(files)} files"
               + "".join(f"; {w}" for w, v in (("gated", gated), ("pinned", pins), ("fresh", fresh)) if v is not None) + ")")
    return out, ok


def main(argv):
    opt = {}
    for o in ("--gated", "--pins", "--fresh"):
        if o in argv:
            i = argv.index(o); opt[o[2:]] = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if len(argv) < 2 or (("pins" in opt or "fresh" in opt) and "gated" not in opt):
        print(__doc__); return 2
    lines, ok = check(argv[0], argv[1:], opt.get("gated"), opt.get("pins"), opt.get("fresh"))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
