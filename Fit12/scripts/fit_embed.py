"""Fit12 L2-03 (2026-10-24, NewIdeas QUEUE; handed over by Breakthrough 02:15): stream the untied 4-bit embed table.
The 125B's embed_tokens is an MLX QuantizedEmbedding (248,320 x 2560, 4-bit, group 64, untied from lm_head: 358 MB) that
only ever gathers the rows of the tokens in a call (decode: <= 8 per verify pass, 1 per MTP draft step; prefill: the prompt's
tokens). StreamedEmbedding gathers those rows from numpy memmaps over the safetensors shard (clean file pages, so not in the
physical footprint) and runs QuantizedEmbedding.__call__'s own mx.dequantize on them: the output is bitwise the resident
table's (dequantize is per group; the gathered words, scales and biases are the same bytes).
install(): patches bigmoe_v3f.stream.ExpertStreamer.install so the swap happens BEFORE the loader's mx.eval(model.parameters()):
the resident table is never read (load peak and steady state both drop by 358 MB). Call it after bt_b12 is imported and
before load_b12. check(model_dir): bitwise test against the resident math on random 1-D and 2-D id batches."""
import json, mmap, os, struct
import numpy as np
import mlx.core as mx
import mlx.nn as nn

PREFIX = "language_model.model.embed_tokens"
STATS = {"calls": 0, "rows": 0, "installed": 0}


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _view(x, dt):
    return x if dt is None else (x.view(dt) if hasattr(x, "view") else mx.view(x, dt))


class StreamedEmbedding(nn.Module):
    def __init__(self, model_dir, group_size=64, bits=4, mode="affine", prefix=PREFIX):
        super().__init__()
        wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        mm = {}
        for f in ("weight", "scales", "biases"):
            path = os.path.join(model_dir, wm[f"{prefix}.{f}"])
            hdr, start = _header(path)
            info = hdr[f"{prefix}.{f}"]
            dt = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16}[info["dtype"]]
            a = np.memmap(path, mode="r", dtype=dt, offset=start + info["data_offsets"][0], shape=tuple(info["shape"]))
            if os.environ.get("F12_EMBED_WILLNEED", "1") == "1":
                try:                                    # page-cache read-ahead (no user buffer, not in the footprint)
                    a._mmap.madvise(mmap.MADV_WILLNEED)
                except (AttributeError, OSError, ValueError):
                    pass
            mm[f] = (a, {"BF16": mx.bfloat16, "F16": mx.float16}.get(info["dtype"]))
        object.__setattr__(self, "_mm", mm)
        object.__setattr__(self, "_q", (group_size, bits, mode))

    def _rows(self, f, flat):
        a, dt = self._mm[f]
        return _view(mx.array(np.ascontiguousarray(a[flat])), dt)

    def __call__(self, x):
        ids = np.asarray(x).astype(np.int64)            # decode / prefill ids are host data already (no GPU sync)
        flat = ids.reshape(-1)
        STATS["calls"] += 1
        STATS["rows"] += int(flat.size)
        gs, bits, mode = self._q
        out = mx.dequantize(self._rows("weight", flat), scales=self._rows("scales", flat),
                            biases=self._rows("biases", flat), group_size=gs, bits=bits, mode=mode)
        return out.reshape(*ids.shape, out.shape[-1])


def install():
    """Swap embed_tokens inside ExpertStreamer.install (before the loader evaluates the remaining weights)."""
    import bigmoe_v3f.stream as BS
    if getattr(BS.ExpertStreamer.install, "_fit12_embed", False):
        return
    inst0 = BS.ExpertStreamer.install

    def inst(self, model, layers=None):
        r = inst0(self, model, layers)
        tm = model.language_model.model
        qe = tm.embed_tokens
        tm.embed_tokens = StreamedEmbedding(self.model_dir, qe.group_size, qe.bits, getattr(qe, "mode", "affine"))
        STATS["installed"] += 1
        print(f"fit12 embed: embed_tokens streamed from {self.model_dir} ({qe.bits}-bit, group {qe.group_size}); the "
              f"resident table is never evaluated", flush=True)
        return r
    inst._fit12_embed = True
    BS.ExpertStreamer.install = inst


def check(model_dir, n=4096, seed=0):
    """Bitwise: StreamedEmbedding vs QuantizedEmbedding.__call__'s math on the fully loaded table."""
    wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    W = {f: mx.load(os.path.join(model_dir, wm[f"{PREFIX}.{f}"]))[f"{PREFIX}.{f}"] for f in ("weight", "scales", "biases")}
    se = StreamedEmbedding(model_dir)
    rng = np.random.default_rng(seed)
    V = W["weight"].shape[0]
    res = []
    for ids in (rng.integers(0, V, n), rng.integers(0, V, (2, 37)), np.array([[0, V - 1, 7, 7, 7, 151643]]),
                rng.integers(0, 1000, (1, 513))):
        x = mx.array(ids.astype(np.int32))
        ref = mx.dequantize(W["weight"][x], scales=W["scales"][x], biases=W["biases"][x], group_size=64, bits=4)
        got = se(x)
        mx.eval(ref, got)
        same = ref.shape == got.shape and ref.dtype == got.dtype and bool(mx.array_equal(_view(ref, mx.uint16),
                                                                                          _view(got, mx.uint16)))
        res.append((tuple(ids.shape), same))
    return res


if __name__ == "__main__":
    md = os.path.expanduser("~/BigMoE/models/bare")
    r = check(md)
    print("fit12 embed check:", r, "->", "BITWISE EQUAL" if all(s for _, s in r) else "MISMATCH", flush=True)
