"""Breakthrough's WEEK98 goal-B screens (Sun 15 Nov 2026 01:2x), imported lazily by dm_ablations.ablation for names bt_*.
Pre-registered: ~/Breakthrough/docs/LOSSY_CHECK_PRICE.md (screens section). Exact experts unless the arm says otherwise.
- bt_mix4: the HC mixers (input_mix_weight_down / up), block_inject_weight and PLE key / value at 4 bits (requantized
  from bf16 with dm_d2's _requant, read-only use). Census: mix 194, inject 96, ple 2.
- bt_ga3: GDN (in_proj_a / b / qkv / z, out_proj) and attention (q / k / v / o) at 3 bits (from 4-bit, RTN). Census:
  gdn 180, att 48.
- bt_1c: the 1c expert tiers (acceptance-weighted, s 0.98, lambda 1.6e-2; 4-bit / MSB-3 / drop) from the masks in
  out/bt/tiers_1c/<window>.npz ('keep' = 4-bit ranks, 'keep3' = MSB-3 ranks, rank-sorted, from the exact routing) via
  dm_common's 'tiers' gate: the stock path serves 'keep', records3 at 3 bits serves 'keep3', the rest is dropped.
- bt_comb: bt_1c + bt_mix4 (the lm_head 16K shortlist is applied offline to the recorded top-8).
Every swap registers a TEARDOWN restore, so later names in one ticket see the original model."""
import mlx.core as mx
import mlx.nn as nn

import dm_common as C
import dm_d2
from dm_ablations import ABLATIONS


def swap(kinds, bits, want):
    def setup():
        TM = dm_d2._model()
        undo, cnt = [], {k: 0 for k in want}
        for name, mod in TM.named_modules():
            if name.startswith("mtp") or ".mtp" in name:
                continue
            for attr, kind in dm_d2.D2_SET.items():
                if kind not in kinds:
                    continue
                lin = mod.get(attr) if isinstance(mod, nn.Module) and attr in mod else None
                if not isinstance(lin, (nn.Linear, nn.QuantizedLinear)):
                    continue
                if kind == "gdn" and attr == "out_proj" and not isinstance(mod, C.Q.GatedDeltaNet):
                    continue
                q, gs = dm_d2._requant(lin, bits)
                setattr(mod, attr, q); undo.append((mod, attr, lin)); cnt[kind] += 1

        def restore():
            for mod, attr, lin in undo:
                setattr(mod, attr, lin)
            mx.clear_cache()
        C.TEARDOWN.append(restore)
        assert cnt == want, f"bt swap {sorted(kinds)} at {bits} bits swapped {cnt}, want {want}"
        print(f"[dm] bt swap: {sum(cnt.values())} linears at {bits} bits ({cnt})", flush=True)
    return setup


def tiers(maskdir):
    def setup():
        C.PERT.update(rmask=maskdir, gate=("tiers",), t3=3, t3k=10, renorm=False)
    return setup


ABLATIONS["bt_mix4"] = swap({"mix", "inject", "ple"}, 4, {"mix": 194, "inject": 96, "ple": 2})
ABLATIONS["bt_ga3"] = swap({"gdn", "att"}, 3, {"gdn": 180, "att": 48})
ABLATIONS["bt_1c"] = tiers("bt/tiers_1c")


def _comb():
    a, b = ABLATIONS["bt_1c"], ABLATIONS["bt_mix4"]

    def setup():
        a(); b()
    return setup


ABLATIONS["bt_comb"] = _comb()


# ---- AWQ-style 3-bit for GDN + attention (Sun 15 Nov 01:5x; pre-registered docs/LOSSY_CHECK_PRICE.md 544a1f1) ----
# bt_stats: E[x^2] per input channel of the 228 projections on an exact pass (run it with DM_SETS=ifeval: held out).
# bt_ga3awq: per linear the alpha in GRID minimising sum_j E[x_j^2] * ||W_:,j - Q3(W s)_:,j / s_j||^2, s = RMS(x)^alpha
# (normalised by sqrt(max * min)); y = Q3(W s)(x / s). bt_ga3w0: alpha 0 everywhere (the validity control vs bt_ga3).
# The wrappers are plain Modules, not QuantizedLinear subclasses, so dm's prefill lane (_linear) calls them, and a path
# that reads .weight directly fails loudly instead of skipping 1/s (the fused GDN out path is decode-only: not on the map).
import os
import numpy as np

STATS = {}
AWQ_PATH = os.path.join(C.OUT, "bt", "awq_stats.npz")


def _targets(TM):
    out = []
    for name, mod in TM.named_modules():
        if name.startswith("mtp") or ".mtp" in name:
            continue
        for attr, kind in dm_d2.D2_SET.items():
            if kind not in ("gdn", "att"):
                continue
            lin = mod.get(attr) if isinstance(mod, nn.Module) and attr in mod else None
            if not isinstance(lin, (nn.Linear, nn.QuantizedLinear)):
                continue
            if kind == "gdn" and attr == "out_proj" and not isinstance(mod, C.Q.GatedDeltaNet):
                continue
            out.append((f"{name}.{attr}", mod, attr, kind))
    return out


class _Rec(nn.Module):
    def __init__(self, inner, key):
        super().__init__()
        self.inner = inner
        self._key = key

    def __call__(self, x):
        xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        s = mx.sum(xf * xf, axis=0)
        mx.eval(s)
        acc = STATS.setdefault(self._key, [np.zeros(x.shape[-1]), 0])
        acc[0] += np.array(s, dtype=np.float64); acc[1] += xf.shape[0]
        return self.inner(x)


class _Awq(nn.Module):
    def __init__(self, inner, inv_s):
        super().__init__()
        self.inner = inner
        self.inv_s = inv_s

    def __call__(self, x):
        return self.inner(x * self.inv_s)


def stats(path):
    def setup():
        TM = dm_d2._model(); undo = []
        for key, mod, attr, kind in _targets(TM):
            lin = getattr(mod, attr); setattr(mod, attr, _Rec(lin, key)); undo.append((mod, attr, lin))
        assert len(undo) == 228, f"bt_stats wrapped {len(undo)}, want 228"

        def restore():
            for mod, attr, lin in undo:
                setattr(mod, attr, lin)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **{k: v[0] / max(v[1], 1) for k, v in STATS.items()})
            print(f"[dm] bt_stats: E[x^2] of {len(STATS)} linears -> {path}; rows per linear "
                  f"{min(v[1] for v in STATS.values())}-{max(v[1] for v in STATS.values())}", flush=True)
            STATS.clear()
        C.TEARDOWN.append(restore)
    return setup


def awq(path, bits=3, force_alpha=None, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    def setup():
        st = dict(np.load(path)); TM = dm_d2._model()
        undo, hist, cnt, miss = [], {}, {"gdn": 0, "att": 0}, 0
        for key, mod, attr, kind in _targets(TM):
            lin = getattr(mod, attr)
            if isinstance(lin, nn.QuantizedLinear):
                W = mx.dequantize(lin.weight, lin.scales, lin.get("biases"), lin.group_size, lin.bits).astype(mx.float32)
            else:
                W = lin.weight.astype(mx.float32)
            in_f = int(W.shape[1]); gs = 64 if in_f % 64 == 0 else 32
            if key in st:
                d = mx.array(st[key].astype(np.float32))
            else:
                d = mx.ones((in_f,)); miss += 1
            best = None
            for a in ((force_alpha,) if force_alpha is not None else grid):
                s = mx.power(mx.sqrt(d) + 1e-6, a); s = s / mx.sqrt(mx.max(s) * mx.min(s))
                q, sc, bi = mx.quantize((W * s[None, :]).astype(mx.bfloat16), group_size=gs, bits=bits)
                Wd = mx.dequantize(q, sc, bi, group_size=gs, bits=bits).astype(mx.float32) / s[None, :]
                err = mx.sum(mx.sum(mx.square(W - Wd), axis=0) * d)
                mx.eval(err)
                if best is None or float(err) < best[0]:
                    best = (float(err), a, s)
            _, a, s = best
            has_b = "bias" in lin
            tmp = nn.Linear(in_f, int(W.shape[0]), bias=has_b)
            tmp.weight = (W * s[None, :]).astype(mx.bfloat16)
            if has_b:
                tmp.bias = lin.bias
            q = nn.QuantizedLinear.from_linear(tmp, gs, bits); mx.eval(q.parameters())
            wrap = _Awq(q, (1.0 / s).astype(mx.bfloat16)); mx.eval(wrap.inv_s)
            setattr(mod, attr, wrap); undo.append((mod, attr, lin)); cnt[kind] += 1; hist[a] = hist.get(a, 0) + 1

        def restore():
            for mod, attr, lin in undo:
                setattr(mod, attr, lin)
            mx.clear_cache()
        C.TEARDOWN.append(restore)
        assert cnt == {"gdn": 180, "att": 48}, f"bt awq swapped {cnt}"
        print(f"[dm] bt awq {bits}-bit: {sum(cnt.values())} linears, alpha histogram {dict(sorted(hist.items()))}, "
              f"{miss} without stats", flush=True)
    return setup


ABLATIONS["bt_stats"] = stats(AWQ_PATH)
ABLATIONS["bt_ga3w0"] = awq(AWQ_PATH, 3, force_alpha=0.0)
ABLATIONS["bt_ga3awq"] = awq(AWQ_PATH, 3)


def _comb_awq():
    a, b, c = ABLATIONS["bt_1c"], ABLATIONS["bt_mix4"], ABLATIONS["bt_ga3awq"]

    def setup():
        a(); b(); c()
    return setup


ABLATIONS["bt_comb_awq"] = _comb_awq()
