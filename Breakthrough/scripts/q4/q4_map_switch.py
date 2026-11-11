#!/usr/bin/env python3
"""Qwen 4 day-0 pre-work (NewIdeas' board, QWEN 4 DAY-0 PRE-WORK: "Breakthrough: the decision map's baseline capture
(top-8 rows, xcapture) with a model switch; done when today's capture reproduced"). Breakthrough, Sun 2026-11-08.

Anchored patcher for ~/FableGod/decision_map/dm_common.py (no git there: the decision-map patch recipe = patch a scratch
copy, test it, then move it in atomically with no map process running; the old file stays as dm_common.py.pre_q4).
What changes (each identical for today's model by construction):
  1. MD (the model store) from DM_MODEL, default ~/BigMoE/models/bare (today's).
  2. The architecture module from DM_ARCH, default mtplx.models.qwen4_exp (today's); a module without today's class
     names fails LOUDLY at import (AttributeError), never silently.
  3. The routing capture keeps the model's own top_k ranks (CAPK, read from the MoE block in discover_layers) instead of
     a hard-coded 10. Today top_k = 10.
  4. The capture write-outs (routing / scores / xcap) stack over the discovered MoE layers (MOEL) instead of range(48).
     Today MOEL = [0..47] in order, so the stacked arrays are the same bytes.
Usage: python q4_map_switch.py check | apply <src> <dst>   ('check' patches into a temp file and diffs it)
"""
import difflib, os, sys, tempfile

SRC = os.path.expanduser("~/FableGod/decision_map/dm_common.py")
EDITS = [
    ("from mtplx.models import qwen4_exp as Q\n",
     "import importlib as _imp                            # Qwen 4 day-0 (Breakthrough Sun 11-08): DM_ARCH switches the module\n"
     "Q = _imp.import_module(os.environ.get(\"DM_ARCH\", \"mtplx.models.qwen4_exp\"))   # (default = today's); same class names\n"),
    ('MD = os.path.expanduser("~/BigMoE/models/bare")\n',
     'MD = os.path.expanduser(os.environ.get("DM_MODEL", "~/BigMoE/models/bare"))   # Qwen 4 day-0: DM_MODEL switches the store\n'),
    ("LAYERS = {}   # id(module) -> (layer_index, kind)\n",
     "LAYERS = {}   # id(module) -> (layer_index, kind)\n"
     "MOEL = []     # Qwen 4 day-0: the MoE layer indices in order (today 0..47), set by discover_layers\n"
     "CAPK = [10]   # Qwen 4 day-0: the routing capture's ranks = the MoE block's top_k (today 10), set by discover_layers\n"),
    ('    print(f"[dm] MoE block: top_k {blk.top_k}, norm_topk_prob {blk.norm_topk_prob!r}", flush=True)\n',
     '    print(f"[dm] MoE block: top_k {blk.top_k}, norm_topk_prob {blk.norm_topk_prob!r}", flush=True)\n'
     '    MOEL[:] = sorted(i for i, k in LAYERS.values() if k == "moe"); CAPK[0] = int(blk.top_k)   # Qwen 4 day-0\n'
     '    print(f"[dm] model {MD}; MoE layers {len(MOEL)} ({MOEL[0]}..{MOEL[-1]}); capture ranks {CAPK[0]}", flush=True)\n'),
    ('CAPR["L"].setdefault(li, []).append(mx.argsort(-g.astype(mx.float32), axis=-1)[:, :10].astype(mx.int16))',
     'CAPR["L"].setdefault(li, []).append(mx.argsort(-g.astype(mx.float32), axis=-1)[:, :CAPK[0]].astype(mx.int16))'),
    ('o = mx.argsort(-gg, axis=-1)[:, :10]',
     'o = mx.argsort(-gg, axis=-1)[:, :CAPK[0]]'),
    ('R = [np.array(mx.concatenate(CAPR["L"][L])) for L in range(48)]',
     'R = [np.array(mx.concatenate(CAPR["L"][L])) for L in MOEL]   # Qwen 4 day-0: the discovered MoE layers'),
    ('S = np.stack([np.array(mx.concatenate(CAPR["W"][L])) for L in range(48)], axis=1)[:len(ids) - 1]',
     'S = np.stack([np.array(mx.concatenate(CAPR["W"][L])) for L in MOEL], axis=1)[:len(ids) - 1]'),
    ('X = np.stack([np.array(mx.concatenate(CAPR["X"][L]))[a:len(ids) - 1] for L in range(48)], axis=1)',
     'X = np.stack([np.array(mx.concatenate(CAPR["X"][L]))[a:len(ids) - 1] for L in MOEL], axis=1)'),
    ('W = np.stack([np.array(mx.concatenate(CAPR["W"][L]))[a:len(ids) - 1] for L in range(48)], axis=1)',
     'W = np.stack([np.array(mx.concatenate(CAPR["W"][L]))[a:len(ids) - 1] for L in MOEL], axis=1)'),
]


def patch(text):
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"anchor found {n} times (want 1): {old[:90]!r}")
        text = text.replace(old, new)
    return text


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    src = sys.argv[2] if len(sys.argv) > 2 else SRC
    old = open(src).read(); new = patch(old)
    compile(new, src, "exec")
    if mode == "check":
        sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "dm_common.py", "dm_common.py (q4)"))
    elif mode == "apply":
        dst = sys.argv[3]
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dst)), prefix=".q4_", suffix=".py")
        with os.fdopen(fd, "w") as f: f.write(new)
        os.replace(tmp, dst)
        print(f"patched {src} -> {dst}")


if __name__ == "__main__":
    main()
