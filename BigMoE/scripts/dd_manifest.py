"""The CERTIFY98 dump manifest dumps_sha256.json, written beside the dumps (Mon 16 Nov; the spec settled with EvalSuite
11:2x: PREREG_certify98 amendments 7 / 7a, ~/EvalSuite/tools/certify98_dumpcheck.py --gated):
  <file basename>: {sha256, bytes, written, script, args[, part]}       one entry per output file
  "_run": {files_sha256_json, checkpoint_path, checkpoint_sha256, bigmoe_commit, bigmoe_dirty, modules,
           dd_topk_sha256, dd_tree_sha256, args: {"dd_topk": {k, parts}, "dd_tree": [{k, nodes, parts}, ...]}}
- A file entry's args is exactly one of _run.args' entries; a part file also carries "part": "i/n" beside it.
- checkpoint_path / checkpoint_sha256: the weights file P.load_weights reads (~/datasets/deepdraft/ckpt/dd_<run>.safetensors).
- modules: {"~/<path>": sha256} for every local module the dumper imported (outside the venv and the Python install),
  always written (EvalSuite's --gated needs it when bigmoe_dirty is true), so the code is pinned even from a dirty copy.
- files_sha256_json: the sha256 of the frozen capture manifest (files_sha256.json), set only after verify_inputs checked
  every input file's sha256 against it; null on the dev banks, which have no frozen manifest.
- Several runs (TOPK, the T8 parts, their merge, ...) merge into one manifest. The pinned fields must agree:
  files_sha256_json, checkpoint_*, dd_topk_sha256, dd_tree_sha256, dd_topk's args, and every module both runs imported.
  Otherwise update() raises and leaves the manifest unchanged. bigmoe_commit stays the first run's; a later run at
  another commit (same pinned code) is listed in bigmoe_commits_seen; bigmoe_dirty is the OR over runs.
A run that loads no model (a merge) passes no checkpoint / files manifest and leaves those fields as they are."""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BM = os.path.dirname(HERE)
HOME = os.path.expanduser("~")
PINNED = ("files_sha256_json", "checkpoint_path", "checkpoint_sha256", "dd_topk_sha256", "dd_tree_sha256")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def tilde(p):
    p = os.path.realpath(p)
    return "~" + p[len(HOME):] if p.startswith(HOME + os.sep) else p


def local_modules():
    """{"~/path": sha256} of every imported module file under $HOME that is not inside the venv or a site-packages dir"""
    skip = {os.path.realpath(x) for x in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith((".py", ".so")):
            continue
        f = os.path.realpath(f)
        if not f.startswith(HOME + os.sep) or "site-packages" in f or any(f.startswith(s + os.sep) for s in skip):
            continue
        out[tilde(f)] = sha256_file(f)
    return dict(sorted(out.items()))


def git_state():
    c = subprocess.run(["git", "-C", BM, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    d = subprocess.run(["git", "-C", BM, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
    return c, bool(d.strip())


def verify_inputs(files_manifest, paths):
    """every input file's sha256 must equal its entry in the frozen files_sha256.json (keys: the path relative to that
    file's folder, e.g. "heplus/HumanEval_0.npz", or the basename; under "files" or at the top level; values a hex string
    or {"sha256": ...}). Raises on a missing entry or a mismatch; returns the sha256 of files_sha256.json itself."""
    FM = json.load(open(files_manifest))
    F = FM["files"] if isinstance(FM.get("files"), dict) else FM
    root = os.path.dirname(os.path.realpath(files_manifest))
    bad = []
    for p in paths:
        rel = os.path.relpath(os.path.realpath(p), root)
        e = F.get(rel, F.get(os.path.basename(p)))
        want = e.get("sha256") if isinstance(e, dict) else e
        got = sha256_file(p)
        if want != got:
            bad.append(f"{rel}: {'no entry' if want is None else 'sha256 ' + got[:12] + ' != ' + str(want)[:12]}")
    if bad:
        raise RuntimeError(f"capture files refused ({len(bad)} of {len(paths)}): " + "; ".join(bad[:5]))
    return sha256_file(files_manifest)


def update(mp, outputs, script, args, part=None, checkpoint=None, files_sha=None, modules=None, commit=None):
    """add the outputs to the manifest mp and merge this run's provenance into "_run" (see the module doc).
    script: "dd_topk" or "dd_tree"; args: {k, parts} or {k, nodes, parts}; modules / commit: test overrides only."""
    assert script in ("dd_topk", "dd_tree"), script
    c_, dirty = git_state()
    commit = commit or c_
    new = {"dd_topk_sha256": sha256_file(os.path.join(HERE, "dd_topk.py")),
           "dd_tree_sha256": sha256_file(os.path.join(HERE, "dd_tree.py"))}
    if checkpoint is not None:
        new["checkpoint_path"], new["checkpoint_sha256"] = tilde(checkpoint), sha256_file(checkpoint)
    if files_sha is not None:
        new["files_sha256_json"] = files_sha
    mods = local_modules() if modules is None else modules
    M = json.load(open(mp)) if os.path.exists(mp) else {}
    R = dict(M.get("_run", {}))
    bad = [f"{k}: {R[k]} != {v}" for k, v in new.items() if R.get(k) is not None and R[k] != v]
    bad += [f"module {p}: {R['modules'][p][:12]} != {s[:12]}" for p, s in mods.items() if R.get("modules", {}).get(p, s) != s]
    A = dict(R.get("args", {}))
    if script == "dd_topk" and A.get("dd_topk", args) != args:
        bad.append(f"dd_topk args {A['dd_topk']} != {args}")
    if bad:
        raise RuntimeError(f"{os.path.basename(mp)}: this run differs from the manifest's pinned fields: " + "; ".join(bad))
    for k in PINNED:
        if R.get(k) is None:
            R[k] = new.get(k)
    R["modules"] = dict(sorted({**R.get("modules", {}), **mods}.items()))
    if not R.get("bigmoe_commit"):
        R["bigmoe_commit"] = commit
    elif R["bigmoe_commit"] != commit:
        seen = R.get("bigmoe_commits_seen", [R["bigmoe_commit"]])
        R["bigmoe_commits_seen"] = seen + ([commit] if commit not in seen else [])
    R["bigmoe_dirty"] = bool(R.get("bigmoe_dirty")) or dirty
    if script == "dd_topk":
        A["dd_topk"] = args
    else:
        A["dd_tree"] = list(A.get("dd_tree", [])) + ([args] if args not in A.get("dd_tree", []) else [])
    R["args"] = A
    for p in outputs:
        e = {"sha256": sha256_file(p), "bytes": os.path.getsize(p), "script": script, "args": args,
             "written": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))}
        if part is not None:
            e["part"] = part
        M[os.path.basename(p)] = e
    M["_run"] = R
    tmp = mp + ".tmp"
    json.dump(M, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, mp)
    return M
