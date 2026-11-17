"""CERTIFY98's dump-manifest check (PREREG_certify98 amendments 7 / 7a): every input's sha256 and size equal its entry in
BigMoE's dumps_sha256.json, before any scoring. CPU only, streaming reads.
  .venv/bin/python tools/certify98_dumpcheck.py <dumps_sha256.json> <file> [<file> ...]
      [--gated <files_sha256.json>] [--pins <pins.json>] [--fresh <fresh_manifest.json>] [--incapture <slim dir>]
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
- --fresh: the files_sha256.json's manifest_sha256 must be the sha256 of the frozen fresh-set manifest.
- --incapture <slim dir> (with --gated): variant (b) of the proof-run amendment, where the candidate's dumps were written
  in the capture sitting and wd was not stored (EvalSuite results/certify98/proofrun_amendment_b_DRAFT_2026-11-17.md):
  - _run.in_capture is true, and _run.capture_script_sha256 is set;
  - _run.wd_sha256 has exactly one 64-hex entry per file of files_sha256.json (the fp16 wd each file's passes read);
  - every file of files_sha256.json is a plain name, and is in <slim dir> with that sha256;
  - every dd_topk dump names only those files, and on every cell y == ids[row + 1] of its slim file (the dumps were
    computed on the stored token sequences);
  - every dd_tree dump has a dd_topk dump with the same file list, its anchors (a_fi, a_an) are exactly that dump's
    depth-0 cells (fi, row - 1), and on every node y == ids[an + depth + 2] of its slim file (-1 past the file's end);
  - every file of files_sha256.json is in exactly one named dd_topk dump and in at least one named dd_tree dump."""
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


def incapture(M, R, gated, slim, files):
    """(lines, ok): variant (b)'s checks (the docstring's --incapture list)"""
    import numpy as np
    out, ok = [], True
    ic = R.get("in_capture") is True and bool(R.get("capture_script_sha256"))
    out.append(f"  in-capture run: in_capture {R.get('in_capture')!r}, capture_script_sha256 "
               f"{R.get('capture_script_sha256') or 'MISSING'}")
    ok &= ic
    F = json.load(open(gated)) if os.path.exists(gated) else {}
    names = dict(F.get("files") or {})
    W = R.get("wd_sha256") if isinstance(R.get("wd_sha256"), dict) else {}
    hexok = all(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in W.values())
    g = set(W) == set(names) and hexok and bool(names)
    out.append(f"  wd_sha256: {len(W)} entries for {len(names)} slim files: " + ("one per file" if g else
               f"MISMATCH (missing {sorted(set(names) - set(W))[:3]}, extra {sorted(set(W) - set(names))[:3]}, 64-hex {hexok})"))
    ok &= g
    bad = [n for n, s in sorted(names.items()) if n != os.path.basename(n) or n in ("", ".", "..")
           or not os.path.isfile(os.path.join(slim, n)) or sha256(os.path.join(slim, n)) != s]
    out.append(f"  slim files in {slim}: {len(names) - len(bad)} of {len(names)} match files_sha256.json"
               + (f"; MISSING or DIFFERENT: {bad[:3]}" if bad else ""))
    ok &= not bad
    topk, cover_t, cover_tree = {}, {}, set()
    for p in files:
        e = M.get(os.path.basename(p)) or {}
        if e.get("script") != "dd_topk" or not os.path.exists(p):
            continue
        z = np.load(p)
        fl = [str(x) for x in z["files"]]
        miss = [f for f in fl if f not in names or not os.path.exists(os.path.join(slim, f))]
        if miss:
            out.append(f"    {os.path.basename(p)} (dd_topk): {len(miss)} of its files are not slim files of files_sha256.json: {miss[:3]}")
            ok = False
            continue
        ids = [np.asarray(np.load(os.path.join(slim, f))["ids"]) for f in fl]
        fi, row, y = np.asarray(z["fi"]), np.asarray(z["row"]), np.asarray(z["y"])
        nb = int(((fi < 0) | (fi >= len(fl))).sum())     # cells of no file count as mismatches
        for i in range(len(fl)):
            m = fi == i
            r = row[m] + 1
            inr = (r >= 0) & (r < len(ids[i]))
            nb += int((~inr).sum()) + int((ids[i][r[inr]] != y[m][inr]).sum())
        out.append(f"    {os.path.basename(p)} (dd_topk): {len(y)} cells; y == ids[row + 1] of its slim file on {len(y) - nb}"
                   + (f"; MISMATCH on {nb}" if nb else ""))
        ok &= nb == 0
        d0 = np.asarray(z["depth"]) == 0
        topk[tuple(fl)] = set(zip(fi[d0].tolist(), (row[d0] - 1).tolist()))
        for f in fl:
            cover_t[f] = cover_t.get(f, 0) + 1
    for p in files:
        e = M.get(os.path.basename(p)) or {}
        if e.get("script") != "dd_tree" or not os.path.exists(p):
            continue
        t = np.load(p)
        fl = tuple(str(x) for x in t["files"])
        na = len(t["a_fi"])
        an = set(zip(np.asarray(t["a_fi"]).tolist(), np.asarray(t["a_an"]).tolist()))
        g = fl in topk and len(an) == na and an == topk[fl]
        out.append(f"    {os.path.basename(p)} (dd_tree): {na} anchors" + (f" ({na - len(an)} repeated)" if len(an) != na else "") + " "
                   + ("== its TOPK dump's depth-0 cells (row - 1)" if g else
                      ("but no TOPK dump with the same file list" if fl not in topk else "!= its TOPK dump's depth-0 cells")))
        ok &= g
        if fl not in topk:
            continue
        cover_tree |= set(fl)
        ids = [np.asarray(np.load(os.path.join(slim, f))["ids"]) for f in fl]
        fi, y = np.asarray(t["fi"]), np.asarray(t["y"])
        r0 = np.asarray(t["an"]) + np.asarray(t["depth"]) + 2
        nb = int(((fi < 0) | (fi >= len(fl))).sum())
        for i in range(len(fl)):
            m = fi == i
            r, yi = r0[m], y[m]
            past = r >= len(ids[i])
            inr = (r >= 0) & ~past
            nb += int((ids[i][r[inr]] != yi[inr]).sum()) + int((yi[past] != -1).sum()) + int((r < 0).sum())
        out.append(f"      {len(y)} nodes; y == ids[an + depth + 2] of its slim file (-1 past its end) on {len(y) - nb}"
                   + (f"; MISMATCH on {nb}" if nb else ""))
        ok &= nb == 0
    once = sorted(n for n in names if cover_t.get(n) != 1)
    notree = sorted(set(names) - cover_tree)
    out.append(f"  coverage: {len(names) - len(once)} of {len(names)} slim files in exactly one TOPK dump"
               + (f" (not: {once[:3]})" if once else "")
               + f"; {len(names) - len(notree)} in a tree dump" + (f" (not: {notree[:3]})" if notree else ""))
    ok &= not once and not notree and bool(names)
    return out, ok


def check(manifest, files, gated=None, pins=None, fresh=None, slim=None):
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
    if slim is not None:
        lines_, ok_ = incapture(M, R, gated, slim, files)
        out += lines_; ok &= ok_
    out.append(f"DUMP MANIFEST: {'PASS' if ok else 'FAIL: score nothing'} ({len(files)} files"
               + "".join(f"; {w}" for w, v in (("gated", gated), ("pinned", pins), ("fresh", fresh), ("in-capture", slim))
                         if v is not None) + ")")
    return out, ok


def main(argv):
    opt = {}
    for o in ("--gated", "--pins", "--fresh", "--incapture"):
        if o in argv:
            i = argv.index(o); opt[o[2:]] = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if len(argv) < 2 or (("pins" in opt or "fresh" in opt or "incapture" in opt) and "gated" not in opt):
        print(__doc__); return 2
    lines, ok = check(argv[0], argv[1:], opt.get("gated"), opt.get("pins"), opt.get("fresh"), opt.get("incapture"))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
