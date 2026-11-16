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


# ---- G1 / G2 map pre-screens (Breakthrough, Mon 16 Nov 13:xx; pre-registered ~/Breakthrough/docs/CHECK_BYTES_900.md
# section 2, 9ac2850). Run in DM_OUT=out_code24 (masks out_code24/bt/tiers_c6, tiers_l20 from g1g2_map.py masks).
# - bt_c6comb: C6 tiers (s 0.98, lambda 1.8e-2) + bt_mix4; bt_l20comb: the same at lambda 2.0e-2.
# - bt_int8: Leap's int8d.py rule (~/Leap/scripts/week98/int8d.py, 7536125) as a fake-quant on the ANSWER rows only (the
#   verify phase; prompt rows stay bf16 on the baseline's own lane, _linear): per (row, 64-group) scale = max|x| / 127
#   (1 if all zero), round to nearest, clamp +-127; x_hat = q * scale in float32 feeds the module's own 4-bit codes with
#   float32 scales / biases (mx.quantized_matmul): y = sx (s sum q code + b sum q) as dn32q, with float32 sums.
#   Targets (Leap's list; 4-bit g64 affine only, int8d's own condition): GDN in_proj_qkv / z / b / a, out_proj; attention
#   q / k / v / o; the indexer's index_qk_proj; the shared expert (gate / up / down, or a fused gate|up + its down) and
#   shared_expert_gate; PLE key / value. Not the router, HC mixers / inject, head, experts or MTP.
#   Row positions: the text model's call is wrapped so each forward chunk sets its first position (reset per window by a
#   dm_common.run_window wrapper, which also sets the window's answer start a = gen + 1 - off from load_windows); a
#   chunk that straddles a runs both paths and picks per row.
# - bt_g1 = bt_c6comb + bt_int8; bt_dense3 = 11-04's "dense3" arm (Leap's dense3.py 894b0ee); bt_restore = nothing.
import json as _json

ABLATIONS["bt_c6tiers"] = tiers("bt/tiers_c6")
ABLATIONS["bt_l20tiers"] = tiers("bt/tiers_l20")


def _both(a_name, b_name):
    def setup():
        ABLATIONS[a_name](); ABLATIONS[b_name]()
    return setup


ABLATIONS["bt_c6comb"] = _both("bt_c6tiers", "bt_mix4")
ABLATIONS["bt_l20comb"] = _both("bt_l20tiers", "bt_mix4")
ABLATIONS["bt_dense3"] = ABLATIONS["dense3"]
ABLATIONS["bt_restore"] = lambda: None

I8_WANT = {"gdn": 180, "att": 48}                     # the census bt_ga3 / dense3 found (tests override it)
I8 = {"a": None, "p0": 0, "T": 0, "cpos": 0, "chunks": 0, "calls": {}, "rows_q": 0, "win": 0, "bad": [], "odd": 0,
      "rows_q_tot": 0, "rows_q_want_tot": 0}


def _fq64(x):
    """Leap's int8d rule on the last axis in 64-groups; returns x_hat (float32) = q * scale exactly"""
    K = x.shape[-1]
    xf = x.astype(mx.float32).reshape(-1, K // 64, 64)
    m = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    s = mx.where(m > 0, m / 127.0, mx.ones_like(m))
    q = mx.clip(mx.round(xf / s), -127, 127)
    return (q * s).reshape(x.shape)


def _qmm32(xh, w, sc, bi, gs, bits, bias=None):
    y = mx.quantized_matmul(xh, w, sc.astype(mx.float32), bi.astype(mx.float32), transpose=True, group_size=gs, bits=bits)
    return y if bias is None else y + bias.astype(mx.float32)


def _span(x, key):
    """(p0, n, a) for this call: the chunk's first position, the call's rows and the window's answer start"""
    n = 1
    for d in x.shape[:-1]:
        n *= d
    if n != I8["T"]:
        I8["odd"] += 1
    I8["calls"][key] = I8["calls"].get(key, 0) + 1
    return I8["p0"], n, I8["a"]


def _pick(x, p0, n, a, y8, ybf_fn):
    I8["rows_q"] += n - max(0, a - p0)
    if p0 >= a:
        return y8
    r = (mx.arange(n) + p0 >= a).reshape(*x.shape[:-1], 1)
    return mx.where(r, y8, ybf_fn())


class _I8Lin(nn.Module):
    """a 4-bit g64 QuantizedLinear with int8 activations on the answer rows"""

    def __init__(self, inner, key):
        super().__init__()
        self.inner = inner
        self._key = key

    def __call__(self, x):
        p0, n, a = _span(x, self._key)
        if p0 + n <= a:
            return C.Q._linear(self.inner, x)
        m = self.inner
        y8 = _qmm32(_fq64(x), m["weight"], m["scales"], m["biases"], m.group_size, m.bits,
                    m["bias"] if "bias" in m else None).astype(x.dtype)
        return _pick(x, p0, n, a, y8, lambda: C.Q._linear(self.inner, x))


class _I8FusedGU(nn.Module):
    """a shared expert built as _FusedGateUpMLP: int8 on gate|up's input and on down's input (answer rows)"""

    def __init__(self, inner, key, down_q4):
        super().__init__()
        self.inner = inner
        self._key = key
        self._down_q4 = down_q4

    def __call__(self, x):
        p0, n, a = _span(x, self._key)
        if p0 + n <= a:
            return self.inner(x)
        m = self.inner
        gu = _qmm32(_fq64(x), m.gu_weight, m.gu_scales, m.gu_biases, m.group_size, m.bits).astype(x.dtype)
        gate, up = mx.split(gu, 2, axis=-1)
        h = nn.silu(gate) * up
        d = m.down_proj
        if self._down_q4:
            y8 = _qmm32(_fq64(h), d["weight"], d["scales"], d["biases"], d.group_size, d.bits).astype(x.dtype)
        else:
            y8 = C.Q._linear(d, h)
        return _pick(x, p0, n, a, y8, lambda: self.inner(x))


def _q4(m):
    return (isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64
            and getattr(m, "mode", "affine") == "affine" and "biases" in m)


def int8():
    def setup():
        TM = dm_d2._model()
        undo, cen = [], {}

        def wrap(mod, attr, kind, mk):
            lin = getattr(mod, attr)
            setattr(mod, attr, mk(lin, f"{kind}:{len(undo)}"))
            undo.append((mod, attr, lin)); cen[kind] = cen.get(kind, 0) + 1

        for name, mod in TM.named_modules():
            if name.startswith("mtp") or ".mtp" in name or not isinstance(mod, nn.Module):
                continue
            for attr in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
                if attr in mod and _q4(mod[attr]):
                    wrap(mod, attr, "gdn", _I8Lin)
            if isinstance(mod, C.Q.GatedDeltaNet) and "out_proj" in mod and _q4(mod["out_proj"]):
                wrap(mod, "out_proj", "gdn", _I8Lin)
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if attr in mod and _q4(mod[attr]):
                    wrap(mod, attr, "att", _I8Lin)
            if "index_qk_proj" in mod and _q4(mod["index_qk_proj"]):
                wrap(mod, "index_qk_proj", "idx", _I8Lin)
            for attr in ("key_proj", "value_proj"):
                if attr in mod and _q4(mod[attr]):
                    wrap(mod, attr, "ple", _I8Lin)
            if "shared_expert_gate" in mod and _q4(mod["shared_expert_gate"]):
                wrap(mod, "shared_expert_gate", "shared_gate", _I8Lin)
            if "shared_expert" in mod:
                se = mod["shared_expert"]
                if isinstance(se, C.Q._FusedGateUpMLP):
                    if se.bits == 4 and se.group_size == 64 and se.mode == "affine":
                        dq = _q4(se["down_proj"])
                        wrap(mod, "shared_expert", "shared_fused_gu" + ("+down" if dq else ""),
                             lambda lin, key, dq=dq: _I8FusedGU(lin, key, dq))
                else:
                    for attr in ("gate_proj", "up_proj", "down_proj"):
                        if attr in se and _q4(se[attr]):
                            wrap(se, attr, "shared", _I8Lin)
        wins = {w[2].tobytes(): w[4] + 1 - w[3] for w in C.load_windows(10 ** 6)}
        orig_rw = C.run_window
        TMC = C.Q.Qwen4ExpTextModel
        prev_call = TMC.__call__

        def text_call(self, inputs, cache=None, input_embeddings=None):
            T = int(inputs.shape[-1]) if input_embeddings is None else int(input_embeddings.shape[-2])
            I8.update(p0=I8["cpos"], T=T, cpos=I8["cpos"] + T, chunks=I8["chunks"] + 1)
            return prev_call(self, inputs, cache, input_embeddings)

        def run_window(model, ids):
            a = wins[ids.tobytes()]
            I8.update(a=a, p0=0, T=0, cpos=0, chunks=0, calls={}, rows_q=0, odd=0)
            out = orig_rw(model, ids)
            I8["win"] += 1
            k = len(undo)
            ch = I8["chunks"]
            want = k * max(0, I8["cpos"] - a)
            ok = (I8["cpos"] in (len(ids) - 1, len(ids)) and len(I8["calls"]) == k
                  and set(I8["calls"].values()) == {ch} and I8["rows_q"] == want and I8["odd"] == 0)
            I8["rows_q_tot"] += I8["rows_q"]; I8["rows_q_want_tot"] += want
            if not ok:
                I8["bad"].append(dict(win=I8["win"], n=len(ids), a=a, cpos=I8["cpos"], chunks=ch, mods_called=len(I8["calls"]),
                                      mods=k, calls=sorted(set(I8["calls"].values())), rows_q=I8["rows_q"], want=want,
                                      odd=I8["odd"]))
                print(f"[dm] bt_int8 WARNING window {I8['win']}: {I8['bad'][-1]}", flush=True)
            return out
        C.run_window = run_window
        TMC.__call__ = text_call

        def restore():
            TMC.__call__ = prev_call
            C.run_window = orig_rw
            for mod, attr, lin in reversed(undo):
                setattr(mod, attr, lin)
            rep = dict(census=cen, windows=I8["win"], rows_q=I8["rows_q_tot"], rows_q_want=I8["rows_q_want_tot"],
                       bad=I8["bad"])
            p = os.path.join(C.OUT, "bt_int8_census.jsonl")                 # one line per int8 arm, in run order
            with open(p, "a") as fh:
                fh.write(_json.dumps(rep) + "\n")
            print(f"[dm] bt_int8: {I8['win']} windows; answer rows x modules quantized {I8['rows_q_tot']} (want "
                  f"{I8['rows_q_want_tot']}); windows with a mismatch {len(I8['bad'])}; census {cen} -> {p}", flush=True)
            I8.update(a=None, p0=0, T=0, cpos=0, chunks=0, calls={}, rows_q=0, win=0, bad=[], odd=0, rows_q_tot=0,
                      rows_q_want_tot=0)
            mx.clear_cache()
        C.TEARDOWN.append(restore)
        assert all(cen.get(k) == v for k, v in I8_WANT.items()), f"bt_int8 census {cen}, want {I8_WANT}"
        print(f"[dm] bt_int8: {sum(cen.values())} projections wrapped ({cen}); answer rows only", flush=True)
    return setup


ABLATIONS["bt_int8"] = int8()
ABLATIONS["bt_g1"] = _both("bt_c6comb", "bt_int8")


# Amendment (a), Mon 13:1x, before any run: G1 as BigMoE built it (data/stack/g1x45g1a_env.txt = lossy-45 + C6 tiers +
# int8; HC8, no V1 / V3, which are not built): C6 tiers + hc8w (HC mixers + PLE at 8-bit, lossy-45's HC8) + int8. With
# PLE at 8-bit, int8 does not touch it (int8d's 4-bit condition), as in the engine.
def _three(a_name, b_name, c_name):
    def setup():
        ABLATIONS[a_name](); ABLATIONS[b_name](); ABLATIONS[c_name]()
    return setup


# Amendment (c), Mon 13:2x, before any run: g1a's HC8 is hc8r.hc8_install(model) with its defaults (ple=False): the 194
# mixers at 8-bit, block_inject and PLE stay bf16 (snapshot r57_screen_g1a.py:945-951). So G1 as gated = hc8w_mix, not
# hc8w; with PLE bf16, int8 leaves it (as int8d can never touch it in g1a).
ABLATIONS["bt_g1h8"] = _three("bt_c6tiers", "hc8w_mix", "bt_int8")
# Amendment (b), Mon 13:1x, before any run (NewIdeas 13:0x: V1 + V3 ride with G2; if DENSE-3 fails, G2 = G1 + V1 + V3 +
# lambda 2.0e-2): G2's no-V4 route = lambda 2.0e-2 tiers + bt_mix4 (V3) + int8; V1 is applied offline in the read-out.
ABLATIONS["bt_l20g2"] = _three("bt_l20tiers", "bt_mix4", "bt_int8")
# Amendment (e), Mon 13:3x, before any run (NewIdeas 13:2x): G1 as gated + RES (Fit12's resident-only MSB-3 tier at
# C 256): masks out_code24/bt/tiers_c6res (g1g2_map.py resmasks; RES applied on the 12 windows with residency).
ABLATIONS["bt_c6restiers"] = tiers("bt/tiers_c6res")
ABLATIONS["bt_g1h8_res"] = _three("bt_c6restiers", "hc8w_mix", "bt_int8")
