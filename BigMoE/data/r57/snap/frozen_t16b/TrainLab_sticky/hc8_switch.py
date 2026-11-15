"""S5 (NewIdeas Sun 09:45): HC8 per arm in one process. The pinned hc8.install (MoonShots 3055bf7 patches/hc8/hc8.py,
the same file as ~/Breakthrough/v3f/hc8.py; copied unchanged to sticky/hc8_3055bf7.py, md5 799347061a369cbd35f7a132b22922e8)
swaps the trunk's hyper-connection mixers (input_mix_weight_down / up of every GatedResidual, the MTP head's left alone)
and the PLE key / value projections for MLX affine 8-bit, group 64, in place. build() runs it once and keeps the bf16
modules beside the 8-bit ones; use(on) puts one set or the other back (the same module objects every time, so 'off' is
the loaded model bit for bit). The 8-bit mixers take MTPLX's eager HC read (its fused reads admit bf16 mixers only).
usage: n, mb_bf16, mb_8bit = build(model); use(True / False)"""
import hc8_3055bf7 as H8

ATTRS = ("input_mix_weight_down", "input_mix_weight_up", "key_proj", "value_proj")   # hc8.install's targets
SW = {"pairs": [], "on": False}


def build(model, **kw):
    TM = getattr(model, "language_model", model)
    before = [(m, a, m[a]) for _, m in TM.named_modules() for a in ATTRS if a in m]
    n, mb0, mb1 = H8.install(model, **kw)
    SW["pairs"] = [(m, a, lin, m[a]) for m, a, lin in before if m[a] is not lin]   # (module, attr, bf16, 8-bit)
    assert len(SW["pairs"]) == n, (len(SW["pairs"]), n)
    use(False)
    return n, mb0, mb1


def use(on):
    for m, a, lin, q in SW["pairs"]:
        setattr(m, a, q if on else lin)
    SW["on"] = bool(on)
