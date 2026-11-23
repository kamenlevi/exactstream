"""Mid-step resume for Sandbox98 screens (orchestrator 01:2x: a run that yields to a quiet ticket picks up where it
stopped, with no re-run of finished parts). One state file per run, overwritten atomically: trainable parameters and
optimizer states as arrays, the loop's own variables as JSON in the safetensors metadata. A SIGTERM (guard_lib's yield)
sets STOP; the training loop saves and exits at its next step. The file is removed when the run finishes."""
import json, os, signal
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

STOP = {"f": False}


def arm_sigterm():
    signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__("f", True))


def save(path, modules, opts, meta):
    """-> bytes written"""
    arrs = {}
    for k, m in modules.items():
        arrs.update({f"{k}.p.{n}": v for n, v in tree_flatten(m.trainable_parameters())})
    for k, o in opts.items():
        arrs.update({f"{k}.o.{n}": v for n, v in tree_flatten(o.state)})
    tmp = path + ".tmp.safetensors"
    mx.save_safetensors(tmp, arrs, metadata={"meta": json.dumps(meta)})
    os.replace(tmp, path + ".safetensors")
    return sum(v.nbytes for v in arrs.values())


def load(path, modules, opts):
    """restore parameters + optimizer states; -> the saved meta dict, or None when there is no state"""
    f = path + ".safetensors"
    if not os.path.exists(f):
        return None
    arrs, md = mx.load(f, return_metadata=True)
    for k, m in modules.items():
        pre = f"{k}.p."
        m.update(tree_unflatten([(n[len(pre):], v) for n, v in arrs.items() if n.startswith(pre)]))
    for k, o in opts.items():
        pre = f"{k}.o."
        o.init(modules[k].trainable_parameters())
        o.state = tree_unflatten([(n[len(pre):], v) for n, v in arrs.items() if n.startswith(pre)])
    return json.loads(md["meta"])


def finish(path):
    for f in (path + ".safetensors", path + ".tmp.safetensors"):
        if os.path.exists(f):
            os.remove(f)
