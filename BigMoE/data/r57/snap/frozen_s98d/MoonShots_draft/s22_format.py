"""S22 ENGINE FORMAT (MoonShots, Tue 17 Nov 2026; the orchestrator 15:3x; NewIdeas QUEUE M-47): the r57 per-level
loop's head formats as REAL packed modules, one table for sf_tree.py --real and BigMoE's loop (read-only import).

  q8h32  today's head: P's 10 Linears + the 6 HC mixes at 8-bit g64; top-10 experts; ScreenedHead 32K + extras
  c2     S22 / path A (= p3_dd_tree.CUTS['C2'], the cut TrainLab's QAT trains under): P's 9 Linears and the 6 HC
         mixes at 4-bit g64, the router at 8-bit g64; top-4 experts; ScreenedHead 8K (or 32K) + extras

As stored, in both: se_gatew, the norms, the routed experts (4-bit g32 SwitchGLU), the lm_head (the screen's rows).
Each module is mx.quantize(w.astype(bfloat16), group_size=64, bits) in QuantizedLinear layout: exactly the
quantization of p3_dd_tree.fq, so the dequantized module equals M3's fake-quant weight bit for bit (checked at load:
check_vs_fakequant). The forward runs the packed quantized matmuls (the engine's form).
S22's weights: pass its dd checkpoint (P's Linears, fp32) as weights=...; the format applies at load, as for t30b."""

FORMATS = {
    "q8h32": dict(lin=8, router=8, hc=8, top=10, screen=32768),
    "c2": dict(lin=4, router=8, hc=4, top=4, screen=8192),
}
LINEARS = ("q_proj", "o_proj", "k_proj", "v_proj", "se_gate", "se_up", "se_down", "fc_hidden", "fc_embedding", "router")
HCS = ("attn_hc", "mlp_hc", "mixer")


def _hc_groups(head):
    L0 = head.layers[0]
    return (("attn_hc", L0.attn_hyper_connection), ("mlp_hc", L0.mlp_hyper_connection),
            ("mixer", head.hyper_connection_mixer))


def _qlin(mx, nn, lin, bits):
    w = lin.weight.astype(mx.bfloat16)
    ql = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=64, bits=bits)
    ql.weight, ql.scales, ql.biases = mx.quantize(w, group_size=64, bits=bits)
    return ql


def apply_real(mx, nn, P, head, fmt):
    """replace P's Linears and the head's 6 HC mixes by packed QuantizedLinears per FORMATS[fmt]; set top-k.
    Returns (done, originals): originals = {name: the pre-quantization weight} for check_vs_fakequant."""
    f = FORMATS[fmt]
    done, orig = [], {}
    for name in LINEARS:
        bits = f["router"] if name == "router" else f["lin"]
        lin = getattr(P, name)
        orig[f"P.{name}"] = (lin.weight, bits)
        setattr(P, name, _qlin(mx, nn, lin, bits))
        done.append(f"P.{name} {bits}-bit")
    for gname, g in _hc_groups(head):
        for sub in ("input_mix_weight_down", "input_mix_weight_up"):
            lin = getattr(g, sub)
            orig[f"{gname}.{sub}"] = (lin.weight, f["hc"])
            setattr(g, sub, _qlin(mx, nn, lin, f["hc"]))
            done.append(f"{gname}.{sub} {f['hc']}-bit")
    head.layers[0].mlp.top_k = int(f["top"])
    done.append(f"experts top-{f['top']}")
    mx.eval(P.parameters(), head.parameters())
    return done, orig


def check_vs_fakequant(mx, P, head, orig, fq):
    """each packed module dequantized back to its original dtype == fq(original, bits) (M3's fake quant), bitwise"""
    mods = {f"P.{n}": getattr(P, n) for n in LINEARS}
    for gname, g in _hc_groups(head):
        for sub in ("input_mix_weight_down", "input_mix_weight_up"):
            mods[f"{gname}.{sub}"] = getattr(g, sub)
    ok, bad = 0, []
    for key, (w0, bits) in orig.items():
        m = mods[key]
        deq = mx.dequantize(m.weight, m.scales, m.biases, group_size=64, bits=bits).astype(w0.dtype)
        if bool(mx.array_equal(deq, fq(mx, w0, bits)).item()):
            ok += 1
        else:
            bad.append(key)
    return ok, len(orig), bad


def bytes_per_level(P, head, sh, top, expert_bytes):
    """dense modules (weights + scales + biases) + top x one routed expert + the screened head's rows"""
    n = 0
    for name in LINEARS:
        m = getattr(P, name)
        n += m.weight.nbytes + m.scales.nbytes + m.biases.nbytes
    for _, g in _hc_groups(head):
        for sub in ("input_mix_weight_down", "input_mix_weight_up"):
            m = getattr(g, sub)
            n += m.weight.nbytes + m.scales.nbytes + m.biases.nbytes
    head_rows = 0
    if sh is not None:
        for nm in ("w", "s", "b", "weight", "scales", "biases"):
            if hasattr(sh, nm) and hasattr(getattr(sh, nm), "nbytes"):
                head_rows += getattr(sh, nm).nbytes
    return n, top * expert_bytes, head_rows
