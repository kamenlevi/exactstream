#!/usr/bin/env python3
"""CPU tests for hc4.py (V3 = R57_HC4; Breakthrough, Mon 16 Nov 2026 13:xx). MLX's CPU device; no GPU, no ref2 kernels.
  1. prepare(): the census on a toy trunk (GatedResiduals with and without inject, a PLE layer, an MTP child that must be
     skipped); every 4-bit copy == the map's bt_mix4 conversion (dm_d2._requant(lin, 4)) bit for bit; the PLE copies (and
     only those) go through dense_native_convert (a recording stub); the extra MB; prepare() after HC8 refuses.
  2. engage(): on swaps in exactly the copies (identity); off puts back the objects present at the first engage (HC8's
     8-bit mixers, the bf16 inject / PLE) by identity; a repeated engage is a no-op; take() reports hc4 1 then 0.
  3. the unset path: with the copies prepared and engage(False), the trunk's modules are the very objects HC8 put there.
Usage: ~/BigMoE/.venv/bin/python -I -B test_hc4.py
"""
import os
import sys
import tempfile

os.environ["DM_OUT"] = tempfile.mkdtemp(prefix="hc4test_")
sys.path.insert(0, os.path.expanduser("~/FableGod/decision_map"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx                                   # noqa: E402
import mlx.nn as nn                                     # noqa: E402
import numpy as np                                      # noqa: E402

mx.set_default_device(mx.cpu)
import dm_d2                                            # noqa: E402  (the map's _requant: the reference conversion)
import hc4                                              # noqa: E402

fails = []


def check(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails.append(msg)


rng = np.random.default_rng(1116)


def lin(N, K):
    m = nn.Linear(K, N, bias=False)
    m.weight = mx.array((rng.standard_normal((N, K)) * 0.05).astype(np.float32)).astype(mx.bfloat16)
    return m


class ToyGR(nn.Module):
    def __init__(self, inject=True):
        super().__init__()
        self.input_mix_weight_down = lin(32, 256)
        self.input_mix_weight_up = lin(256, 32)
        if inject:
            self.block_inject_weight = lin(4, 256)


class ToyPLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_proj, self.value_proj = lin(256, 64), lin(128, 64)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_hyper_connection, self.mlp_hyper_connection = ToyGR(), ToyGR()


class Trunk(nn.Module):
    def __init__(self, nl=3):
        super().__init__()
        self.layers = [Layer() for _ in range(nl)]
        self.hyper_connection_mixer = ToyGR(inject=False)
        self.ple = ToyPLE()
        self.mtp = Layer()                               # the MTP head's modules: never touched


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = Trunk()


class FakeRef2:
    def __init__(self):
        self.seen = []

    def dense_native_convert(self, m):
        for _, x in m.named_modules():
            if isinstance(x, nn.QuantizedLinear):
                self.seen.append(x)
        return len(self.seen), 0.0


M = Model()
T = M.language_model
orig = {}
for name, mod in T.named_modules():
    for a in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight", "key_proj", "value_proj"):
        if isinstance(mod, nn.Module) and a in mod:
            orig[(name, a)] = mod[a]
fr = FakeRef2()
cnt, mb = hc4.prepare(M, gr_cls=ToyGR, ple_cls=ToyPLE, ref2_module=fr)
check(cnt == {"mix": 2 * 7, "inject": 6, "ple": 2}, f"census: 7 GatedResiduals (6 with inject + the final mixer) x 2 "
      f"mixers, 6 injects, 2 PLE; the MTP layer skipped ({cnt})")
same = True
for mod, attr, q in hc4._T:
    ref, _ = dm_d2._requant(getattr(mod, attr), 4)
    same &= all(q[k].dtype == ref[k].dtype and bool(mx.array_equal(q[k], ref[k])) for k in ("weight", "scales", "biases"))
    same &= q.bits == ref.bits == 4 and q.group_size == ref.group_size
check(same and len(hc4._T) == 22, f"all {len(hc4._T)} copies == dm_d2._requant(lin, 4) bit for bit (the map's bt_mix4)")
ple_q = [q for mod, attr, q in hc4._T if attr in ("key_proj", "value_proj")]
check(len(fr.seen) == 2 and all(any(a is b for b in ple_q) for a in fr.seen), "dense_native_convert saw the 2 PLE copies only")
check(mb > 0 and all(T[k[0].split(".")[0]] is not None for k in orig) and
      all(getattr(mod, attr) is orig_obj for (name, attr), orig_obj in orig.items()
          for mod in [dict(T.named_modules())[name]]), f"prepare() changed nothing in the model; extra {mb:.3f} MB")
# HC8 as hc8r.hc8_install does it: the mixers -> 8-bit QuantizedLinear (inject / PLE stay bf16)
hc8 = {}
for name, mod in T.named_modules():
    if isinstance(mod, ToyGR) and not name.startswith("mtp"):
        for a in ("input_mix_weight_down", "input_mix_weight_up"):
            q8 = nn.QuantizedLinear.from_linear(mod[a], 64 if mod[a].weight.shape[1] % 64 == 0 else 32, 8)
            setattr(mod, a, q8)
            hc8[(name, a)] = q8
try:
    hc4.prepare(M, gr_cls=ToyGR, ple_cls=ToyPLE, ref2_module=FakeRef2())
    check(False, "prepare() after HC8 must refuse")
except AssertionError as e:
    check("before hc8r.hc8_install" in str(e), "prepare() after HC8 refuses (the bf16 originals are gone)")
    cnt, mb = hc4.prepare.__wrapped__(M) if hasattr(hc4.prepare, "__wrapped__") else (cnt, mb)
# the refused call cleared hc4._T: prepare again on a fresh model state is not possible (HC8 is in), so rebuild the
# copies from the saved originals the way the screen would have had them (prepare ran before HC8)
M2 = Model()
T2 = M2.language_model
for (name, a), o in orig.items():
    mod = dict(T2.named_modules())[name]
    setattr(mod, a, o)
cnt2, _ = hc4.prepare(M2, gr_cls=ToyGR, ple_cls=ToyPLE, ref2_module=FakeRef2())
for (name, a), q8 in hc8.items():
    setattr(dict(T2.named_modules())[name], a, q8)
n_on = hc4.engage(True)
on_ok = all(getattr(mod, attr) is q for mod, attr, q in hc4._T)
check(n_on == 22 and on_ok, "engage(True): every target is its 4-bit copy (identity)")
check(hc4.engage(True) == 0, "a repeated engage(True) is a no-op")
r1 = hc4.take()
off_ok = all(getattr(mod, attr) is o for mod, attr, o in hc4._OFF)
want = {**{(n, a): q for (n, a), q in hc8.items()},
        **{(n, a): o for (n, a), o in orig.items() if a in ("block_inject_weight", "key_proj", "value_proj")}}
mods2 = dict(T2.named_modules())
back_ok = all(getattr(mods2[n], a) is o for (n, a), o in want.items())
check(r1 == {"hc4": 1} and off_ok and back_ok, f"take(): hc4 1, and the HC8 8-bit mixers + bf16 inject / PLE are back "
      f"by identity ({r1})")
check(hc4.take() == {"hc4": 0}, "take() on a disengaged item: hc4 0")
hc4.engage(True)
hc4.engage(False)
check(all(getattr(mods2[n], a) is o for (n, a), o in want.items()), "engage(True) then engage(False): the unset path "
      "is the very objects HC8 put there")
mtp_untouched = all(getattr(mod, a) is orig_mtp for (name, a), orig_mtp in orig.items() if name.startswith("mtp")
                    for mod in [mods2[name]])
check(mtp_untouched, "the MTP layer's modules were never swapped")
print("ALL PASS" if not fails else f"{len(fails)} FAIL", flush=True)
sys.exit(1 if fails else 0)
